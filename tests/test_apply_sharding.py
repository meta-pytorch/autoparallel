# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
from conftest import apply_cuda_patches
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import (
    DTensorSpec,
    ShardOrderEntry,
    TensorMeta,
)
from torch.distributed.tensor._op_schema import OpSpec
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard
from torch.fx.experimental.proxy_tensor import make_fx

from autoparallel.apply_sharding import (
    ApplyShardingInterpreter,
    _build_physical_placements,
    _project_shard_order,
)
from autoparallel.shardings.ordered_sharding import (
    OrderInfo,
    ordered_redistribute_local_tensor,
)


class TestProjectShardOrder:
    def test_projects_3d_storage_order(self):
        mesh = DeviceMesh(
            "cuda",
            torch.arange(8).reshape(2, 2, 2),
            mesh_dim_names=("dp", "cp", "tp"),
        )
        preferred = (ShardOrderEntry(tensor_dim=0, mesh_dims=(2, 1, 0)),)

        storage = DTensorSpec(mesh, (Shard(0), Shard(0), Shard(0)))
        compute = DTensorSpec(mesh, (Replicate(), Replicate(), Shard(0)))
        grad_compute = DTensorSpec(mesh, (Partial(), Partial(), Shard(0)))

        assert _project_shard_order(preferred, storage) == preferred
        assert _project_shard_order(preferred, compute) == (
            ShardOrderEntry(tensor_dim=0, mesh_dims=(2,)),
        )
        assert _project_shard_order(preferred, grad_compute) == (
            ShardOrderEntry(tensor_dim=0, mesh_dims=(2,)),
        )

    def test_uncovered_shard_falls_back_to_spec_order(self):
        mesh = DeviceMesh(
            "cuda",
            torch.arange(8).reshape(2, 2, 2),
            mesh_dim_names=("dp", "cp", "tp"),
        )
        preferred = (ShardOrderEntry(tensor_dim=0, mesh_dims=(2, 0)),)
        spec = DTensorSpec(mesh, (Shard(0), Shard(1), Shard(0)))

        assert _project_shard_order(preferred, spec) == spec.shard_order


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

    def _make_specs(self, mesh, src_plc, dst_plc, shape=(1024, 4096)):
        tm = _make_tensor_meta(shape)
        src = DTensorSpec(mesh, src_plc, tensor_meta=tm)
        dst = DTensorSpec(mesh, dst_plc, tensor_meta=tm)
        local_shape = list(shape)
        for mesh_size, p in zip(mesh.shape, src_plc):
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

    def test_3d_fsdplike_gather_has_no_alltoall(self):
        mesh = DeviceMesh(
            "cuda",
            torch.arange(8).reshape(2, 2, 2),
            mesh_dim_names=("dp", "cp", "tp"),
        )
        src, dst, local = self._make_specs(
            mesh,
            (Shard(0), Shard(0), Shard(0)),
            (Replicate(), Replicate(), Shard(0)),
            shape=(6144, 4096),
        )
        src.shard_order = (ShardOrderEntry(tensor_dim=0, mesh_dims=(2, 1, 0)),)

        def trace_fn(x):
            return ordered_redistribute_local_tensor(x, src, dst)

        gm = make_fx(trace_fn, tracing_mode="real")(local)
        counts = _count_collectives(gm)
        assert counts == {"all_gather": 2, "reduce_scatter": 0, "alltoall": 0}

    def test_3d_fsdplike_gradient_has_no_alltoall(self):
        mesh = DeviceMesh(
            "cuda",
            torch.arange(8).reshape(2, 2, 2),
            mesh_dim_names=("dp", "cp", "tp"),
        )
        src, dst, local = self._make_specs(
            mesh,
            (Partial(), Partial(), Shard(0)),
            (Shard(0), Shard(0), Shard(0)),
            shape=(6144, 4096),
        )
        dst.shard_order = (ShardOrderEntry(tensor_dim=0, mesh_dims=(2, 1, 0)),)

        def trace_fn(x):
            return ordered_redistribute_local_tensor(x, src, dst)

        gm = make_fx(trace_fn, tracing_mode="real")(local)
        counts = _count_collectives(gm)
        assert counts == {"all_gather": 0, "reduce_scatter": 2, "alltoall": 0}

    def test_4d_fsdplike_gather_has_no_alltoall(self):
        mesh = DeviceMesh(
            "cuda",
            torch.arange(16).reshape(2, 2, 2, 2),
            mesh_dim_names=("a", "b", "c", "d"),
        )
        src, dst, local = self._make_specs(
            mesh,
            (Shard(0), Shard(0), Shard(0), Shard(0)),
            (Replicate(), Shard(0), Replicate(), Shard(0)),
            shape=(4096, 1024),
        )
        src.shard_order = (ShardOrderEntry(tensor_dim=0, mesh_dims=(1, 3, 2, 0)),)

        def trace_fn(x):
            return ordered_redistribute_local_tensor(x, src, dst)

        gm = make_fx(trace_fn, tracing_mode="real")(local)
        counts = _count_collectives(gm)
        assert counts == {"all_gather": 2, "reduce_scatter": 0, "alltoall": 0}


class TestShardOrderSpecIsolation:
    """Test that shard_order modifications don't leak between shared DTensorSpec
    objects (regression test for in-place mutation bug)."""

    def test_redistribute_does_not_mutate_input_specs(self, device_mesh_2d):
        """redistribute_tensor must not modify the shard_order on the specs
        it receives, since those specs may be shared across clustered nodes."""
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
            gm,
            sharding_placement,
            param_placement_order={
                t: OrderInfo(
                    preferred_shard_order=(
                        ShardOrderEntry(tensor_dim=0, mesh_dims=(1, 0)),
                    )
                ),
            },
        )

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


def test_build_physical_placements_3d_uses_strided_shards():
    mesh = DeviceMesh(
        "cuda",
        torch.arange(8).reshape(2, 2, 2),
        mesh_dim_names=("dp", "cp", "tp"),
    )
    tm = _make_tensor_meta([6144, 4096])
    storage = DTensorSpec(
        mesh,
        (Shard(0), Shard(0), Shard(0)),
        tensor_meta=tm,
    )
    graph = torch.fx.Graph()
    param = graph.placeholder("param")
    sharding_placement = {
        param: OpSpec(
            output_specs=storage,
            input_specs=[storage],
        )
    }
    physical = _build_physical_placements(
        sharding_placement,
        {
            param: OrderInfo(
                preferred_shard_order=(
                    ShardOrderEntry(tensor_dim=0, mesh_dims=(2, 1, 0)),
                )
            )
        },
    )[param]

    from torch.distributed.tensor.placement_types import _StridedShard

    assert isinstance(physical.placements[0], _StridedShard)
    assert physical.placements[0].split_factor == 4
    assert isinstance(physical.placements[1], _StridedShard)
    assert physical.placements[1].split_factor == 2
    assert type(physical.placements[2]) is Shard
    _, decoded_order = DTensorSpec._normalize_placements_into_shard_order(
        physical.placements,
        physical.mesh,
    )
    assert decoded_order == (ShardOrderEntry(tensor_dim=0, mesh_dims=(2, 1, 0)),)


class TestProducerKeyedShardOrder:
    """A weight consumed by a node that is not on its chain -- a matmul, or the
    local_map region MoE expert weights are passed into -- must still be
    redistributed in the storage order its chain carries."""

    def _mesh(self):
        # (dp_shard_mod_ep=2, tp=8) -- the DeepSeek V3 folded MoE mesh
        return DeviceMesh(
            "cuda", torch.arange(16).reshape(2, 8), mesh_dim_names=("dp", "tp")
        )

    def _expert_weight_boundary(self, mesh):
        # DeepSeek V3 routed-expert weight on (dp_shard_mod_ep=2, tp=8), where tp
        # is also the EP axis: stored S(0)S(0), the region wants R S(0). Tensor
        # dim 0 is sharded by both axes, so in default (dp-major) order releasing
        # dp lowers to all-to-all -> all-gather -> all-to-all.
        tm = _make_tensor_meta([64, 1408, 2048])
        storage = DTensorSpec(mesh, (Shard(0), Shard(0)), tensor_meta=tm)
        region = DTensorSpec(mesh, (Replicate(), Shard(0)), tensor_meta=tm)
        return storage, region

    def _interpreter(self, mesh, storage, region, order):
        graph = torch.fx.Graph()
        param = graph.placeholder("param")
        cast = graph.call_function(
            torch.ops.prims.convert_element_type.default, (param, torch.bfloat16)
        )
        other = graph.placeholder("other")
        consumer = graph.call_function(torch.ops.aten.mul.Tensor, (cast, other))
        graph.output(consumer)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        sharding_placement = {
            param: OpSpec(output_specs=storage, input_specs=[storage]),
            cast: OpSpec(output_specs=storage, input_specs=[storage]),
            other: OpSpec(output_specs=region, input_specs=[region]),
            consumer: OpSpec(output_specs=region, input_specs=[region, region]),
        }
        interp = ApplyShardingInterpreter(
            gm,
            sharding_placement,
            # The order lives on the parameter chain. The consumer is deliberately
            # absent, which is the shape build_param_grad_linear_chains produces.
            param_placement_order={
                param: OrderInfo(preferred_shard_order=order),
                cast: OrderInfo(preferred_shard_order=order),
            },
        )
        interp._curr_node = consumer
        return interp, cast, consumer

    def test_producer_order_collapses_boundary_to_one_all_gather(self):
        mesh = self._mesh()
        storage, region = self._expert_weight_boundary(mesh)
        order = (ShardOrderEntry(tensor_dim=0, mesh_dims=(1, 0)),)
        interp, cast, consumer = self._interpreter(mesh, storage, region, order)
        local = torch.randn(4, 1408, 2048, device="meta")

        def trace_fn(x):
            return interp.redistribute_tensor(
                x, storage, region, consumer, producer=cast
            )

        counts = _count_collectives(make_fx(trace_fn, tracing_mode="real")(local))
        assert counts == {"all_gather": 1, "reduce_scatter": 0, "alltoall": 0}

    def test_order_is_resolved_from_the_producer_not_the_consumer(self):
        """The consumer carries no order of its own. Only the producer can supply
        it, and without one the specs keep their default order."""
        mesh = self._mesh()
        storage, region = self._expert_weight_boundary(mesh)
        order = (ShardOrderEntry(tensor_dim=0, mesh_dims=(1, 0)),)
        interp, cast, consumer = self._interpreter(mesh, storage, region, order)

        assert interp._compute_origin_and_target_shard_order(
            consumer, storage, region
        ) == (storage.shard_order, region.shard_order)

        curr_order, tgt_order = interp._compute_origin_and_target_shard_order(
            consumer, storage, region, producer=cast
        )
        assert curr_order == order
        assert curr_order != storage.shard_order


@apply_cuda_patches
def test_partial_subset_producer_establishes_order_without_collectives():
    mesh = DeviceMesh(
        "cuda",
        torch.arange(32).reshape(4, 2, 4),
        mesh_dim_names=("dp", "cp", "tp"),
    )
    graph = torch.fx.Graph()
    carrier = graph.placeholder("carrier")
    grad_producer = graph.call_function(torch.ops.aten.clone.default, (carrier,))
    grad_consumer = graph.call_function(torch.ops.aten.clone.default, (grad_producer,))
    param = graph.placeholder("param")
    forward_consumer = graph.call_function(torch.ops.aten.clone.default, (param,))
    graph.output((grad_consumer, forward_consumer))
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    weight_meta = _make_tensor_meta([14336, 4096])
    carrier_meta = _make_tensor_meta([8, 64, 14336])
    storage = DTensorSpec(mesh, (Shard(0), Shard(0), Shard(0)), tensor_meta=weight_meta)
    forward = DTensorSpec(
        mesh, (Replicate(), Replicate(), Shard(0)), tensor_meta=weight_meta
    )
    grad_source = DTensorSpec(
        mesh, (Partial(), Shard(0), Shard(0)), tensor_meta=weight_meta
    )
    carrier_source = DTensorSpec(
        mesh, (Shard(0), Replicate(), Shard(2)), tensor_meta=carrier_meta
    )
    carrier_target = DTensorSpec(
        mesh, (Shard(0), Shard(2), Shard(2)), tensor_meta=carrier_meta
    )
    preferred = (ShardOrderEntry(tensor_dim=0, mesh_dims=(2, 1, 0)),)
    regular = OrderInfo(preferred)
    remapped = OrderInfo(preferred, project_by_mesh_priority=True)
    interp = ApplyShardingInterpreter(
        gm,
        {},
        param_placement_order={
            param: regular,
            forward_consumer: regular,
            grad_producer: remapped,
            grad_consumer: remapped,
        },
    )

    def trace_redistribution(local, source, target, consumer, producer):
        def redistribute(value):
            return interp.redistribute_tensor(
                value, source, target, consumer, producer=producer
            )

        return _count_collectives(make_fx(redistribute, tracing_mode="real")(local))

    assert trace_redistribution(
        torch.randn(2, 64, 3584, device="meta"),
        carrier_source,
        carrier_target,
        grad_producer,
        carrier,
    ) == {"all_gather": 0, "reduce_scatter": 0, "alltoall": 0}
    lm_head_meta = _make_tensor_meta([8, 64, 128256])
    lm_head_source = DTensorSpec(
        mesh, (Shard(0), Shard(1), Shard(2)), tensor_meta=lm_head_meta
    )
    lm_head_target = DTensorSpec(
        mesh, (Shard(0), Shard(2), Shard(2)), tensor_meta=lm_head_meta
    )
    assert trace_redistribution(
        torch.randn(2, 32, 32064, device="meta"),
        lm_head_source,
        lm_head_target,
        grad_producer,
        carrier,
    ) == {"all_gather": 0, "reduce_scatter": 0, "alltoall": 1}
    assert trace_redistribution(
        torch.randn(448, 4096, device="meta"),
        storage,
        forward,
        forward_consumer,
        param,
    ) == {"all_gather": 2, "reduce_scatter": 0, "alltoall": 0}
    assert trace_redistribution(
        torch.randn(1792, 4096, device="meta"),
        grad_source,
        storage,
        grad_consumer,
        grad_producer,
    ) == {"all_gather": 0, "reduce_scatter": 1, "alltoall": 0}
