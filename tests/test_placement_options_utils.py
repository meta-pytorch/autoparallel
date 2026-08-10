# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch._functorch._aot_autograd.descriptors import PlainAOTInput, PlainAOTOutput
from torch.distributed._tensor.placement_types import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import OpSpec, OpStrategy
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel.collectives import flex_local_map
from autoparallel.optimize_sharding import ShardingOptimizer
from autoparallel.shardings.placement_options import (
    fill_missing_redistribute_cost,
    get_placement_options_for_node,
    keep_unique_configs,
    propagate_tensor_meta,
)
from autoparallel.shardings.propagation_rules import remove_invalid_configs


def _make_tensor_meta(shape, dtype=torch.float32):
    t = torch.empty(shape, dtype=dtype, device="meta")
    return TensorMeta(t.shape, t.stride(), t.dtype)


def call_local_map(x):
    return x


def _make_flex_local_map_graph(alternatives):
    return _make_local_map_graph_with_kwargs({"alternatives": alternatives})


def _make_local_map_graph_with_kwargs(local_map_kwargs):
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty(512, 128, device="meta")
    x.meta["desc"] = PlainAOTInput(0)

    local_map_node = graph.call_function(call_local_map, (x,))
    local_map_node.meta["val"] = torch.empty(512, 128, device="meta")
    local_map_node.meta["local_map_kwargs"] = local_map_kwargs

    output = graph.output((local_map_node,))
    output.meta["desc"] = (PlainAOTOutput(0),)
    return torch.fx.GraphModule(torch.nn.Module(), graph), local_map_node


# ===== remove_invalid_configs =====


class TestRemoveInvalidConfigs:
    def test_shard_divisible(self, device_mesh_1d):
        # mesh dim = 256, shape[0] = 256 → divisible
        tm = _make_tensor_meta((256, 64))
        spec = DTensorSpec(device_mesh_1d, (Shard(0),), tensor_meta=tm)
        strat = OpStrategy(
            [OpSpec(spec, input_specs=[spec], redistribute_cost=[[0.0]])]
        )
        result = remove_invalid_configs(strat, device_mesh_1d)
        assert len(result.strategies) == 1

    def test_shard_indivisible(self, device_mesh_1d):
        # mesh dim = 256, shape[0] = 100 → not divisible
        tm = _make_tensor_meta((100, 64))
        spec = DTensorSpec(device_mesh_1d, (Shard(0),), tensor_meta=tm)
        strat = OpStrategy(
            [OpSpec(spec, input_specs=[spec], redistribute_cost=[[0.0]])]
        )
        result = remove_invalid_configs(strat, device_mesh_1d)
        assert len(result.strategies) == 0

    def test_replicate_always_kept(self, device_mesh_1d):
        tm = _make_tensor_meta((7, 3))
        spec = DTensorSpec(device_mesh_1d, (Replicate(),), tensor_meta=tm)
        strat = OpStrategy(
            [OpSpec(spec, input_specs=[spec], redistribute_cost=[[0.0]])]
        )
        result = remove_invalid_configs(strat, device_mesh_1d)
        assert len(result.strategies) == 1

    def test_2d_mesh_second_shard_indivisible(self, device_mesh_2d):
        # 2D mesh: (32, 8). shape=(256, 10).
        # Shard(0) on mesh dim 0: 256/32=8, OK (even)
        # Shard(1) on mesh dim 1: 10/8, not divisible but 10>=8 → kept (uneven)
        tm = _make_tensor_meta((256, 10))
        spec = DTensorSpec(device_mesh_2d, (Shard(0), Shard(1)), tensor_meta=tm)
        strat = OpStrategy(
            [OpSpec(spec, input_specs=[spec], redistribute_cost=[[0.0]])]
        )
        result = remove_invalid_configs(strat, device_mesh_2d)
        assert len(result.strategies) == 1

    def test_none_output_spec_skipped(self, device_mesh_1d):
        tm = _make_tensor_meta((256,))
        in_spec = DTensorSpec(device_mesh_1d, (Replicate(),), tensor_meta=tm)
        strat = OpStrategy(
            [OpSpec(None, input_specs=[in_spec], redistribute_cost=[[0.0]])]
        )
        result = remove_invalid_configs(strat, device_mesh_1d)
        assert len(result.strategies) == 1

    def test_mix_valid_and_invalid(self, device_mesh_1d):
        tm_ok = _make_tensor_meta((256, 64))
        tm_bad = _make_tensor_meta((100, 64))
        spec_ok = DTensorSpec(device_mesh_1d, (Shard(0),), tensor_meta=tm_ok)
        spec_bad = DTensorSpec(device_mesh_1d, (Shard(0),), tensor_meta=tm_bad)
        strat = OpStrategy(
            [
                OpSpec(spec_ok, input_specs=[spec_ok], redistribute_cost=[[0.0]]),
                OpSpec(spec_bad, input_specs=[spec_bad], redistribute_cost=[[0.0]]),
            ]
        )
        result = remove_invalid_configs(strat, device_mesh_1d)
        assert len(result.strategies) == 1

    def test_uneven_shard_kept_when_nonempty(self, device_mesh_2d):
        # shape=10 on mesh dim 1 (size 8): 10 >= 8, uneven but valid
        tm = _make_tensor_meta((256, 10))
        spec = DTensorSpec(device_mesh_2d, (Shard(0), Shard(1)), tensor_meta=tm)
        strat = OpStrategy(
            [OpSpec(spec, input_specs=[spec], redistribute_cost=[[0.0]])]
        )
        result = remove_invalid_configs(strat, device_mesh_2d)
        assert len(result.strategies) == 1

    def test_uneven_shard_rejected_when_empty(self, device_mesh_1d):
        # shape=100 on mesh dim 0 (size 256): 100 < 256, would create empty shards
        tm = _make_tensor_meta((100, 64))
        spec = DTensorSpec(device_mesh_1d, (Shard(0),), tensor_meta=tm)
        strat = OpStrategy(
            [OpSpec(spec, input_specs=[spec], redistribute_cost=[[0.0]])]
        )
        result = remove_invalid_configs(strat, device_mesh_1d)
        assert len(result.strategies) == 0

    def test_uneven_shard_just_above_mesh(self, device_mesh_2d):
        # shape=257 on mesh dim 0 (size 32): 257 >= 32, valid
        tm = _make_tensor_meta((257, 64))
        spec = DTensorSpec(device_mesh_2d, (Shard(0), Replicate()), tensor_meta=tm)
        strat = OpStrategy(
            [OpSpec(spec, input_specs=[spec], redistribute_cost=[[0.0]])]
        )
        result = remove_invalid_configs(strat, device_mesh_2d)
        assert len(result.strategies) == 1


# ===== keep_unique_configs =====


class TestKeepUniqueConfigs:
    def test_duplicates_removed(self, device_mesh_1d):
        tm = _make_tensor_meta((4, 8))
        spec = DTensorSpec(device_mesh_1d, (Replicate(),), tensor_meta=tm)
        op_spec = OpSpec(spec, input_specs=[spec], redistribute_cost=[[0.0]])
        strat = OpStrategy([op_spec, op_spec])
        result = keep_unique_configs(strat)
        assert len(result.strategies) == 1

    def test_different_kept(self, device_mesh_1d):
        tm = _make_tensor_meta((256, 64))
        spec_r = DTensorSpec(device_mesh_1d, (Replicate(),), tensor_meta=tm)
        spec_s = DTensorSpec(device_mesh_1d, (Shard(0),), tensor_meta=tm)
        strat = OpStrategy(
            [
                OpSpec(spec_r, input_specs=[spec_r], redistribute_cost=[[0.0]]),
                OpSpec(spec_s, input_specs=[spec_s], redistribute_cost=[[0.0]]),
            ]
        )
        result = keep_unique_configs(strat)
        assert len(result.strategies) == 2


# ===== flex local_map placement options =====


class TestFlexLocalMapPlacementOptions:
    """Component-level tests: these hand-build the fx graph / local_map_kwargs and do
    NOT run Dynamo/AOTAutograd, so they do not prove the alternatives survive real
    tracing. End-to-end trace coverage lives in
    ``test_optimize_placement.py::test_flex_local_map_alternatives_visible_to_solver``.
    """

    @pytest.mark.parametrize(
        ("alternatives", "exception", "error"),
        (
            ([], ValueError, "requires at least one alternative"),
            ([None], TypeError, "must be a dict"),
            (
                [{"out_placements": ((Replicate(),),)}],
                ValueError,
                "requires in_placements",
            ),
            (
                [{"in_placements": ((Replicate(),),)}],
                ValueError,
                "requires out_placements",
            ),
            (
                [
                    {
                        "fn": None,
                        "in_placements": ((Replicate(),),),
                        "out_placements": ((Replicate(),),),
                    }
                ],
                TypeError,
                "fn must be callable",
            ),
            (
                [
                    {
                        "in_placements": ((Replicate(),),),
                        "out_placements": ((Replicate(),),),
                        "cost_hint": float("inf"),
                    }
                ],
                ValueError,
                "cost_hint must be finite and non-negative",
            ),
        ),
    )
    def test_rejects_invalid_alternatives(
        self, device_mesh_1d, alternatives, exception, error
    ):
        with pytest.raises(exception, match=error):
            flex_local_map(
                call_local_map,
                alternatives=alternatives,
                device_mesh=device_mesh_1d,
            )

    def test_rejects_mismatched_alternative_arity(self, device_mesh_1d):
        with pytest.raises(ValueError, match="different input arity"):
            flex_local_map(
                call_local_map,
                alternatives=[
                    {
                        "in_placements": ((Replicate(),),),
                        "out_placements": ((Replicate(),),),
                    },
                    {
                        "in_placements": (
                            (Replicate(),),
                            (Replicate(),),
                        ),
                        "out_placements": ((Replicate(),),),
                    },
                ],
                device_mesh=device_mesh_1d,
            )

    def test_rejects_in_grad_placements(self, device_mesh_1d):
        with pytest.raises(NotImplementedError, match="in_grad_placements"):
            flex_local_map(
                call_local_map,
                alternatives=[
                    {
                        "in_placements": ((Replicate(),),),
                        "out_placements": ((Replicate(),),),
                    }
                ],
                device_mesh=device_mesh_1d,
                in_grad_placements=((Replicate(),),),
            )

    def test_rejects_non_bool_auto_cost(self, device_mesh_1d):
        with pytest.raises(TypeError, match="auto_cost must be a bool"):
            flex_local_map(
                call_local_map,
                alternatives=[
                    {
                        "in_placements": ((Replicate(),),),
                        "out_placements": ((Replicate(),),),
                    }
                ],
                device_mesh=device_mesh_1d,
                auto_cost="yes",
            )

    def test_generates_one_strategy_per_alternative(self, device_mesh_1d):
        gm, local_map_node = _make_flex_local_map_graph(
            [
                {
                    "in_placements": ((Replicate(),),),
                    "out_placements": ((Replicate(),),),
                },
                {
                    "in_placements": ((Shard(0),),),
                    "out_placements": ((Shard(0),),),
                },
            ]
        )
        input_value = gm.graph.find_nodes(op="placeholder")[0].meta["val"]
        tensor_meta = _make_tensor_meta(input_value.shape)
        replicate_spec = DTensorSpec(
            device_mesh_1d, (Replicate(),), tensor_meta=tensor_meta
        )
        shard_spec = DTensorSpec(device_mesh_1d, (Shard(0),), tensor_meta=tensor_meta)
        input_strategy = OpStrategy(
            [
                OpSpec(
                    replicate_spec,
                    input_specs=[replicate_spec],
                    redistribute_cost=[[0.0]],
                ),
                OpSpec(shard_spec, input_specs=[shard_spec], redistribute_cost=[[0.0]]),
            ]
        )

        result = get_placement_options_for_node(
            device_mesh_1d, local_map_node, [input_strategy], [input_value], {}
        )

        assert len(result.strategies) == 2
        assert result.strategies[0].input_specs[0].placements == (Replicate(),)
        assert result.strategies[0].output_specs[0].placements == (Replicate(),)
        assert result.strategies[1].input_specs[0].placements == (Shard(0),)
        assert result.strategies[1].output_specs[0].placements == (Shard(0),)

    def test_wrapper_metadata_generates_one_strategy_per_alternative(
        self, device_mesh_1d
    ):
        # Checks only that flex_local_map's returned partial exposes recoverable
        # alternatives metadata off wrapped.args; it injects that into a hand-built
        # graph and does NOT trace, so it does not exercise the Dynamo path where the
        # carrier must survive (covered by the e2e trace test noted in the class
        # docstring).
        wrapped = flex_local_map(
            call_local_map,
            alternatives=[
                {
                    "in_placements": ((Replicate(),),),
                    "out_placements": ((Replicate(),),),
                },
                {
                    "in_placements": ((Shard(0),),),
                    "out_placements": ((Shard(0),),),
                },
            ],
            device_mesh=device_mesh_1d,
            redistribute_inputs=True,
        )
        local_map_kwargs = {
            "out_placements": wrapped.args[1],
            "in_placements": wrapped.args[2],
            "in_grad_placements": wrapped.args[3],
            "device_mesh": wrapped.args[4],
        }
        gm, local_map_node = _make_local_map_graph_with_kwargs(local_map_kwargs)
        input_value = gm.graph.find_nodes(op="placeholder")[0].meta["val"]
        tensor_meta = _make_tensor_meta(input_value.shape)
        input_spec = DTensorSpec(
            device_mesh_1d, (Replicate(),), tensor_meta=tensor_meta
        )
        input_strategy = OpStrategy(
            [
                OpSpec(
                    input_spec,
                    input_specs=[input_spec],
                    redistribute_cost=[[0.0]],
                )
            ]
        )

        result = get_placement_options_for_node(
            device_mesh_1d, local_map_node, [input_strategy], [input_value], {}
        )

        assert len(result.strategies) == 2
        assert result.strategies[0].output_specs[0].placements == (Replicate(),)
        assert result.strategies[1].output_specs[0].placements == (Shard(0),)

    def test_cost_hint_drives_solver_choice(self, device_mesh_1d):
        gm, local_map_node = _make_flex_local_map_graph(
            [
                {
                    "in_placements": ((Replicate(),),),
                    "out_placements": ((Shard(0),),),
                    "cost_hint": 10.0,
                },
                {
                    "in_placements": ((Replicate(),),),
                    "out_placements": ((Replicate(),),),
                    "cost_hint": 0.0,
                },
            ]
        )

        solution = ShardingOptimizer(gm, device_mesh_1d).get_solution()

        assert solution[local_map_node].output_specs[0].placements == (Replicate(),)

    def test_zero_cost_hints_leave_redistribution_to_drive_choice(self, device_mesh_1d):
        gm, local_map_node = _make_flex_local_map_graph(
            [
                {
                    "in_placements": ((Replicate(),),),
                    "out_placements": ((Replicate(),),),
                    "cost_hint": 0.0,
                },
                {
                    "in_placements": ((Shard(0),),),
                    "out_placements": ((Shard(0),),),
                    "cost_hint": 0.0,
                },
            ]
        )
        optimizer = ShardingOptimizer(gm, device_mesh_1d)
        optimizer.add_sharded_input_constraint([(Shard(0),)])

        solution = optimizer.get_solution()

        assert solution[local_map_node].output_specs[0].placements == (Shard(0),)


# ===== fill_missing_redistribute_cost =====


class TestFillMissingRedistributeCost:
    def test_already_set_untouched(self, device_mesh_1d):
        tm = _make_tensor_meta((256,))
        spec = DTensorSpec(device_mesh_1d, (Replicate(),), tensor_meta=tm)
        original_cost = [[42.0]]
        op_spec = OpSpec(spec, input_specs=[spec], redistribute_cost=original_cost)
        strat = OpStrategy([op_spec])
        input_strat = OpStrategy([OpSpec(spec, input_specs=[spec])])
        fill_missing_redistribute_cost(
            torch.ops.aten.ones_like.default, [input_strat], strat
        )
        assert strat.strategies[0].redistribute_cost == [[42.0]]

    def test_fills_for_handled_op(self, device_mesh_1d):
        tm = _make_tensor_meta((256,))
        spec = DTensorSpec(device_mesh_1d, (Replicate(),), tensor_meta=tm)
        op_spec = OpSpec(spec, input_specs=[spec])
        assert op_spec.redistribute_cost is None
        strat = OpStrategy([op_spec])
        input_strat = OpStrategy([OpSpec(spec, input_specs=[spec])])
        fill_missing_redistribute_cost(
            torch.ops.aten.ones_like.default, [input_strat], strat
        )
        assert strat.strategies[0].redistribute_cost is not None

    def test_raises_for_unhandled_op(self, device_mesh_1d):
        tm = _make_tensor_meta((256,))
        spec = DTensorSpec(device_mesh_1d, (Replicate(),), tensor_meta=tm)
        op_spec = OpSpec(spec, input_specs=[spec])
        strat = OpStrategy([op_spec])
        input_strat = OpStrategy([OpSpec(spec, input_specs=[spec])])
        with pytest.raises(AssertionError):
            fill_missing_redistribute_cost(
                torch.ops.aten.add.Tensor, [input_strat], strat
            )


# ===== propagate_tensor_meta =====


class TestPropagateTensorMeta:
    def test_overwrites_existing_meta(self, device_mesh_1d):
        tm = _make_tensor_meta((4, 8))
        spec = DTensorSpec(device_mesh_1d, (Replicate(),), tensor_meta=tm)
        strat = OpStrategy(
            [OpSpec(spec, input_specs=[spec], redistribute_cost=[[0.0]])]
        )
        propagate_tensor_meta(
            torch.ops.aten.neg.default,
            (torch.empty(4, 8, device="meta"),),
            {},
            strat,
        )
        # tensor_meta is overwritten (not the same object) but has the same shape
        result_tm = strat.strategies[0].output_spec.tensor_meta
        assert result_tm is not tm
        assert result_tm.shape == torch.Size([4, 8])

    def test_fills_when_meta_missing(self, device_mesh_1d):
        spec = DTensorSpec(device_mesh_1d, (Replicate(),))
        assert spec.tensor_meta is None
        op_spec = OpSpec(spec, input_specs=[spec], redistribute_cost=[[0.0]])
        strat = OpStrategy([op_spec])
        propagate_tensor_meta(
            torch.ops.aten.neg.default,
            (torch.empty(4, 8, device="meta"),),
            {},
            strat,
        )
        assert strat.strategies[0].output_spec.tensor_meta is not None
        assert strat.strategies[0].output_spec.tensor_meta.shape == torch.Size([4, 8])

    def test_skips_none_output_specs(self, device_mesh_1d):
        """Strategies with output_specs=None (non-tensor getitem) are skipped."""
        tm = _make_tensor_meta((4, 8))
        in_spec = DTensorSpec(device_mesh_1d, (Replicate(),), tensor_meta=tm)
        strat = OpStrategy(
            [OpSpec(None, input_specs=[in_spec], redistribute_cost=[[0.0]])]
        )
        # Should not crash even though output_specs is None
        propagate_tensor_meta(
            torch.ops.aten.neg.default,
            (torch.empty(4, 8, device="meta"),),
            {},
            strat,
        )
        assert strat.strategies[0].output_specs is None
