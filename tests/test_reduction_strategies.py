# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.distributed._tensor.placement_types import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import OpSpec, OpStrategy
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard

from autoparallel.shardings.propagation_rules import _fixed_common_reduction_strategy


def _make_tensor_meta(shape, dtype=torch.float32):
    t = torch.empty(shape, dtype=dtype, device="meta")
    return TensorMeta(t.shape, t.stride(), t.dtype)


def _make_input_strategy(mesh, placements_list, shape):
    """Build an OpStrategy with one strategy per placement tuple."""
    tm = _make_tensor_meta(shape)
    strategies = []
    for placements in placements_list:
        spec = DTensorSpec(mesh, tuple(placements), tensor_meta=tm)
        strategies.append(OpSpec(spec, input_specs=(), redistribute_cost=[]))
    return OpStrategy(strategies)


def _output_placements(strategy):
    """Extract the set of output placement tuples from an OpStrategy."""
    return {s.output_specs.placements for s in strategy.strategies}


class TestReductionStrategyCompleteness:
    """Verify that the fixed reduction strategy generates Partial outputs
    for all input strategies where the reduced dims are sharded, not just
    the ones that appear before the first Partial(avg) strategy."""

    def test_sum_with_shard_on_reduced_dims_produces_partial(self, device_mesh_2d):
        mesh = device_mesh_2d
        shape = (32, 8192, 4096)
        # S(0)S(1): both sharded dims are in the reduction dims [0, 1].
        # The sum should produce P(sum)P(sum).
        input_strategy = _make_input_strategy(mesh, [(Shard(0), Shard(1))], shape)
        result = _fixed_common_reduction_strategy(
            input_strategy, reduce_dims=[0, 1], reduction_op="sum"
        )
        out = _output_placements(result)
        assert (Partial("sum"), Partial("sum")) in out

    def test_sum_preserves_shard_on_non_reduced_dim(self, device_mesh_2d):
        mesh = device_mesh_2d
        shape = (32, 8192, 4096)
        # S(2)S(2): hidden sharded on both mesh dims, reducing [0, 1].
        # Neither shard dim is in reduce_dims, so output should be S(0)S(0)
        # (dims shift after collapsing dims 0 and 1).
        input_strategy = _make_input_strategy(mesh, [(Shard(2), Shard(2))], shape)
        result = _fixed_common_reduction_strategy(
            input_strategy, reduce_dims=[0, 1], reduction_op="sum"
        )
        out = _output_placements(result)
        assert (Shard(0), Shard(0)) in out

    def test_partial_avg_does_not_suppress_subsequent_shard_strategies(
        self, device_mesh_2d
    ):
        """The upstream bug: a Partial(avg) strategy early in the list
        permanently flips reduction_linear to False, causing later
        Shard-on-reduced-dim strategies to collapse to Replicate."""
        mesh = device_mesh_2d
        shape = (32, 8192, 4096)
        input_strategy = _make_input_strategy(
            mesh,
            [
                (Replicate(), Replicate()),  # RR
                (Replicate(), Partial("avg")),  # RP(avg) — triggers the bug
                (Shard(0), Shard(1)),  # S(0)S(1) — must not be lost
                (Shard(0), Shard(2)),  # S(0)S(2)
            ],
            shape,
        )
        result = _fixed_common_reduction_strategy(
            input_strategy, reduce_dims=[0, 1], reduction_op="sum"
        )
        out = _output_placements(result)
        # S(0)S(1) reducing [0,1] → P(sum)P(sum): both sharded dims are reduced
        assert (
            Partial("sum"),
            Partial("sum"),
        ) in out, "S(0)S(1) strategy was suppressed — reduction_linear leaked across strategies"
        # S(0)S(2) reducing [0,1] → P(sum)S(0): dim 0 reduced, dim 2 shifts to 0
        assert (Partial("sum"), Shard(0)) in out

    def test_upstream_bug_reproducer(self, device_mesh_2d):
        """Minimal reproducer: if reduction_linear leaks, the second strategy
        (S(0)S(1)) gets forced through replicate_reduction_dims and its output
        becomes (RR) → RR, identical to the first strategy."""
        mesh = device_mesh_2d
        shape = (32, 8192, 4096)
        input_strategy = _make_input_strategy(
            mesh,
            [
                (Replicate(), Partial("avg")),  # flips reduction_linear in buggy code
                (Shard(0), Shard(1)),  # should produce P(sum)P(sum)
            ],
            shape,
        )
        result = _fixed_common_reduction_strategy(
            input_strategy, reduce_dims=[0, 1], reduction_op="sum"
        )
        # With the bug, both strategies collapse to the same output (RR),
        # so we'd only get 1 unique output. With the fix, we get 2.
        out = _output_placements(result)
        assert (
            len(out) == 2
        ), f"Expected 2 distinct output placements, got {len(out)}: {out}"
        assert (Replicate(), Replicate()) in out  # from RP(avg) input, non-linear
        assert (Partial("sum"), Partial("sum")) in out  # from S(0)S(1) input

    def test_mean_reduction_with_mixed_strategies(self, device_mesh_2d):
        mesh = device_mesh_2d
        shape = (32, 8192, 4096)
        input_strategy = _make_input_strategy(
            mesh,
            [
                (Replicate(), Replicate()),
                (Shard(0), Shard(1)),
            ],
            shape,
        )
        result = _fixed_common_reduction_strategy(
            input_strategy, reduce_dims=[0, 1], reduction_op="avg"
        )
        out = _output_placements(result)
        assert (Partial("avg"), Partial("avg")) in out
