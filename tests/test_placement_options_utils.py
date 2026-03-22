# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch.distributed._tensor.placement_types import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import OpSpec, OpStrategy
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel.shardings.placement_options import (
    _has_tensor_meta,
    fill_missing_redistribute_cost,
    keep_unique_configs,
    propagate_tensor_meta,
)
from autoparallel.shardings.propagation_rules import remove_invalid_configs


@pytest.fixture(scope="module")
def device_mesh_2d():
    world_size = torch.distributed.get_world_size()
    return torch.distributed.device_mesh.init_device_mesh(
        "cuda", (world_size // 8, 8), mesh_dim_names=("dp", "tp")
    )


def _make_tensor_meta(shape, dtype=torch.float32):
    t = torch.empty(shape, dtype=dtype, device="meta")
    return TensorMeta(t.shape, t.stride(), t.dtype)


# ===== _has_tensor_meta =====


class TestHasTensorMeta:
    def test_both_input_and_output_have_meta(self, device_mesh_1d):
        tm = _make_tensor_meta((4, 8))
        spec = DTensorSpec(device_mesh_1d, (Replicate(),), tensor_meta=tm)
        strat = OpStrategy(
            [OpSpec(spec, input_specs=[spec], redistribute_cost=[[0.0]])]
        )
        assert _has_tensor_meta(strat) is True

    def test_output_has_meta_input_missing(self, device_mesh_1d):
        """This is the bug we hit: output has tensor_meta but input doesn't."""
        tm = _make_tensor_meta((4, 8))
        out_spec = DTensorSpec(device_mesh_1d, (Replicate(),), tensor_meta=tm)
        in_spec = DTensorSpec(device_mesh_1d, (Replicate(),))
        strat = OpStrategy(
            [OpSpec(out_spec, input_specs=[in_spec], redistribute_cost=[[0.0]])]
        )
        assert _has_tensor_meta(strat) is False

    def test_output_missing_meta(self, device_mesh_1d):
        spec = DTensorSpec(device_mesh_1d, (Replicate(),))
        strat = OpStrategy(
            [OpSpec(spec, input_specs=[spec], redistribute_cost=[[0.0]])]
        )
        assert _has_tensor_meta(strat) is False

    def test_input_specs_none(self, device_mesh_1d):
        tm = _make_tensor_meta((4, 8))
        spec = DTensorSpec(device_mesh_1d, (Replicate(),), tensor_meta=tm)
        strat = OpStrategy([OpSpec(spec)])
        assert _has_tensor_meta(strat) is False

    def test_output_specs_none(self, device_mesh_1d):
        """Non-tensor getitem output with input_specs also None → False
        because propagate_tensor_meta needs to fill input_specs."""
        strat = OpStrategy([OpSpec(None, input_specs=None)])
        assert _has_tensor_meta(strat) is False

    def test_output_specs_none_with_input(self, device_mesh_1d):
        """Non-tensor getitem output but input_specs set → True."""
        tm = _make_tensor_meta((4, 8))
        in_spec = DTensorSpec(device_mesh_1d, (Replicate(),), tensor_meta=tm)
        strat = OpStrategy([OpSpec(None, input_specs=[in_spec])])
        assert _has_tensor_meta(strat) is True

    def test_multi_output_all_have_meta(self, device_mesh_1d):
        tm = _make_tensor_meta((4, 8))
        spec = DTensorSpec(device_mesh_1d, (Replicate(),), tensor_meta=tm)
        out = (spec, spec)
        strat = OpStrategy([OpSpec(out, input_specs=[spec], redistribute_cost=[[0.0]])])
        assert _has_tensor_meta(strat) is True

    def test_multi_output_with_none_specs(self, device_mesh_1d):
        tm = _make_tensor_meta((4, 8))
        spec = DTensorSpec(device_mesh_1d, (Replicate(),), tensor_meta=tm)
        out = (spec, None, spec)
        strat = OpStrategy([OpSpec(out, input_specs=[spec], redistribute_cost=[[0.0]])])
        assert _has_tensor_meta(strat) is True

    def test_empty_strategy_list(self):
        strat = OpStrategy([])
        assert _has_tensor_meta(strat) is False


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
        # Shard(0) on mesh dim 0: 256/32=8, OK
        # Shard(1) on mesh dim 1: 10/8, not divisible → removed
        tm = _make_tensor_meta((256, 10))
        spec = DTensorSpec(device_mesh_2d, (Shard(0), Shard(1)), tensor_meta=tm)
        strat = OpStrategy(
            [OpSpec(spec, input_specs=[spec], redistribute_cost=[[0.0]])]
        )
        result = remove_invalid_configs(strat, device_mesh_2d)
        assert len(result.strategies) == 0

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
    def test_skips_when_meta_already_set(self, device_mesh_1d):
        tm = _make_tensor_meta((4, 8))
        spec = DTensorSpec(device_mesh_1d, (Replicate(),), tensor_meta=tm)
        strat = OpStrategy(
            [OpSpec(spec, input_specs=[spec], redistribute_cost=[[0.0]])]
        )
        # Should be a no-op — tensor_meta stays the same
        propagate_tensor_meta(
            torch.ops.aten.neg.default,
            (torch.empty(4, 8, device="meta"),),
            {},
            strat,
        )
        assert strat.strategies[0].output_spec.tensor_meta is tm

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
