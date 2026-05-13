# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.distributed.tensor._dtensor_spec import (
    DTensorSpec,
    ShardOrderEntry,
    TensorMeta,
)
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard
from torch.fx.experimental.proxy_tensor import make_fx

from autoparallel.apply_sharding import _compute_shard_order
from autoparallel.shardings.ordered_sharding import ordered_redistribute_local_tensor


class TestComputeShardOrder:
    def test_sorted_order(self):
        shard_order = (ShardOrderEntry(tensor_dim=0, mesh_dims=(2, 0, 1)),)
        result = _compute_shard_order(shard_order, reverse=False)
        assert result == (ShardOrderEntry(tensor_dim=0, mesh_dims=(0, 1, 2)),)

    def test_reversed_order(self):
        shard_order = (ShardOrderEntry(tensor_dim=0, mesh_dims=(2, 0, 1)),)
        result = _compute_shard_order(shard_order, reverse=True)
        assert result == (ShardOrderEntry(tensor_dim=0, mesh_dims=(2, 1, 0)),)

    def test_multiple_entries(self):
        shard_order = (
            ShardOrderEntry(tensor_dim=0, mesh_dims=(3, 1)),
            ShardOrderEntry(tensor_dim=1, mesh_dims=(2, 0)),
        )
        result = _compute_shard_order(shard_order, reverse=False)
        assert result == (
            ShardOrderEntry(tensor_dim=0, mesh_dims=(1, 3)),
            ShardOrderEntry(tensor_dim=1, mesh_dims=(0, 2)),
        )

    def test_already_sorted_is_idempotent(self):
        shard_order = (ShardOrderEntry(tensor_dim=0, mesh_dims=(0, 1, 2)),)
        result = _compute_shard_order(shard_order, reverse=False)
        assert result == shard_order

    def test_empty(self):
        assert _compute_shard_order((), reverse=False) == ()
        assert _compute_shard_order((), reverse=True) == ()


def _count_collectives(gm):
    """Count collective ops in a traced graph by type."""
    counts = {"all_gather": 0, "reduce_scatter": 0, "alltoall": 0}
    for n in gm.graph.nodes:
        if n.op != "call_function":
            continue
        name = getattr(n.target, "__name__", "")
        if "all_gather" in name:
            counts["all_gather"] += 1
        elif "reduce_scatter" in name:
            counts["reduce_scatter"] += 1
        elif "alltoall" in name:
            counts["alltoall"] += 1
    return counts


def _make_tensor_meta(shape):
    return TensorMeta(
        torch.Size(shape),
        torch.empty(shape, device="meta").stride(),
        torch.float32,
    )


class TestOrderedRedistributeFusion:
    """Test that ordered_redistribute_local_tensor fuses multi-dim collectives
    into single flat-mesh operations when possible."""

    def _make_specs(self, device_mesh_2d, src_plc, dst_plc, shape=(1024, 4096)):
        tm = _make_tensor_meta(shape)
        src = DTensorSpec(device_mesh_2d, src_plc, tensor_meta=tm)
        dst = DTensorSpec(device_mesh_2d, dst_plc, tensor_meta=tm)
        local_shape = list(shape)
        for mesh_size, p in zip(device_mesh_2d.shape, src_plc):
            if p.is_shard():
                local_shape[p.dim] //= mesh_size
        local = torch.randn(local_shape, device="meta")
        return src, dst, local

    def test_ss_to_rr_uses_single_allgather(self, device_mesh_2d):
        """S(0)S(0) -> RR with default order should fuse into one flat-mesh
        all-gather via _optimize_same_nd_sharding_as_1d."""
        src, dst, local = self._make_specs(
            device_mesh_2d, (Shard(0), Shard(0)), (Replicate(), Replicate())
        )

        def trace_fn(x):
            return ordered_redistribute_local_tensor(x, src, dst)

        gm = make_fx(trace_fn, tracing_mode="real")(local)
        counts = _count_collectives(gm)
        assert counts["all_gather"] == 1, (
            f"S(0)S(0)->RR should use 1 flat-mesh all-gather, "
            f"got {counts['all_gather']}"
        )

    def test_ss_to_rs_with_reversed_order_uses_single_allgather(self, device_mesh_2d):
        """S(0)S(0) -> RS(0) with reversed shard_order on source should
        produce a single all-gather.  This falls through to PyTorch's
        redistribute_local_tensor (not the flat-mesh path), which uses
        its graph-based planner to find a 1-step plan when shard_order
        is reversed."""
        src, dst, local = self._make_specs(
            device_mesh_2d, (Shard(0), Shard(0)), (Replicate(), Shard(0))
        )
        src.shard_order = (ShardOrderEntry(tensor_dim=0, mesh_dims=(1, 0)),)

        def trace_fn(x):
            return ordered_redistribute_local_tensor(x, src, dst)

        gm = make_fx(trace_fn, tracing_mode="real")(local)
        counts = _count_collectives(gm)
        assert counts["all_gather"] == 1, (
            f"S(0)S(0)->RS(0) with reversed order should use 1 all-gather, "
            f"got {counts['all_gather']}"
        )

    def test_ss_to_rr_default_order_does_not_produce_alltoall(self, device_mesh_2d):
        """S(0)S(0) -> RR with default shard_order should NOT produce any
        alltoall ops — it should go through the flat-mesh path."""
        src, dst, local = self._make_specs(
            device_mesh_2d, (Shard(0), Shard(0)), (Replicate(), Replicate())
        )

        def trace_fn(x):
            return ordered_redistribute_local_tensor(x, src, dst)

        gm = make_fx(trace_fn, tracing_mode="real")(local)
        counts = _count_collectives(gm)
        assert (
            counts["alltoall"] == 0
        ), f"S(0)S(0)->RR should not use alltoall, got {counts['alltoall']}"

    def test_pp_to_ss_uses_single_reduce_scatter(self, device_mesh_2d):
        """P(sum)P(sum) -> S(0)S(0) with default order should fuse into one
        flat-mesh reduce-scatter."""
        src, dst, local = self._make_specs(
            device_mesh_2d, (Partial(), Partial()), (Shard(0), Shard(0))
        )
        # Partial input is not sharded, so local is full size
        local = torch.randn(1024, 4096, device="meta")

        def trace_fn(x):
            return ordered_redistribute_local_tensor(x, src, dst)

        gm = make_fx(trace_fn, tracing_mode="real")(local)
        counts = _count_collectives(gm)
        assert counts["reduce_scatter"] == 1, (
            f"PP->SS should use 1 flat-mesh reduce-scatter, "
            f"got {counts['reduce_scatter']}"
        )

    def test_reversed_order_falls_through_to_redistribute(self, device_mesh_2d):
        """S(0)S(0) -> RR with reversed shard_order should NOT go through
        _optimize_same_nd_sharding_as_1d — it falls through to
        redistribute_local_tensor which may use multiple collectives."""
        src, dst, local = self._make_specs(
            device_mesh_2d, (Shard(0), Shard(0)), (Replicate(), Replicate())
        )
        src.shard_order = (ShardOrderEntry(tensor_dim=0, mesh_dims=(1, 0)),)

        def trace_fn(x):
            return ordered_redistribute_local_tensor(x, src, dst)

        gm = make_fx(trace_fn, tracing_mode="real")(local)
        counts = _count_collectives(gm)
        assert counts["all_gather"] >= 1


class TestShardOrderSpecIsolation:
    """Test that shard_order modifications don't leak between shared DTensorSpec
    objects (regression test for in-place mutation bug)."""

    def test_redistribute_does_not_mutate_input_specs(self, device_mesh_2d):
        """redistribute_tensor must not modify the shard_order on the specs
        it receives, since those specs may be shared across clustered nodes."""
        from autoparallel.apply_sharding import ApplyShardingInterpreter
        from autoparallel.shardings.ordered_sharding import OrderInfo

        tm = _make_tensor_meta([64, 64])

        # Two params sharing the same DTensorSpec objects (simulates clustering)
        shared_curr_spec = DTensorSpec(
            device_mesh_2d, (Shard(0), Shard(0)), tensor_meta=tm
        )
        shared_tgt_spec = DTensorSpec(
            device_mesh_2d, (Replicate(), Shard(0)), tensor_meta=tm
        )

        original_curr_order = shared_curr_spec.shard_order
        original_tgt_order = shared_tgt_spec.shard_order

        # Build a minimal graph and interpreter
        graph = torch.fx.Graph()
        p = graph.placeholder("p")
        t = graph.call_function(torch.ops.aten.t.default, (p,))
        graph.output(t)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        from torch.distributed.tensor._op_schema import OpSpec

        sharding_placement = {
            p: OpSpec(output_specs=shared_curr_spec, input_specs=[shared_curr_spec]),
            t: OpSpec(output_specs=shared_tgt_spec, input_specs=[shared_tgt_spec]),
        }

        interp = ApplyShardingInterpreter(
            gm, sharding_placement, enable_ordered_sharding_optimization=False
        )
        # Manually set param_placement_order for the consuming node
        interp.param_placement_order = {
            t: OrderInfo(is_target_reversed_order=False, need_reorder=True),
        }

        # Call redistribute_tensor — this should NOT mutate shared_curr_spec
        local = torch.randn(2, 64, device="meta")
        interp._curr_node = t

        def trace_fn(x):
            return interp.redistribute_tensor(x, shared_curr_spec, shared_tgt_spec, t)

        make_fx(trace_fn, tracing_mode="real")(local)

        # The original specs must be unmodified
        assert shared_curr_spec.shard_order == original_curr_order, (
            f"curr_spec.shard_order was mutated: "
            f"{shared_curr_spec.shard_order} != {original_curr_order}"
        )
        assert shared_tgt_spec.shard_order == original_tgt_order, (
            f"tgt_spec.shard_order was mutated: "
            f"{shared_tgt_spec.shard_order} != {original_tgt_order}"
        )
