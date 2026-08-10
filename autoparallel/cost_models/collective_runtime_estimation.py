# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import cast

import torch
import torch.distributed.tensor._dtensor_spec as dtensor_spec
from torch._prims_common import check_contiguous_sizes_strides
from torch.distributed.tensor._collective_utils import (
    MeshTopoInfo,
    allgather_cost,
    allreduce_cost,
    reduce_scatter_cost,
    spec_to_bytes,
)
from torch.distributed.tensor.placement_types import Partial, Shard

from .compute_estimation import _concretize_unbacked_numel, compute_read_write_time
from .nccl_cost_model import (
    NCCLTopoConfig,
    derive_mesh_dim_topo,
    nccl_all_to_all_cost,
    nccl_allgather_cost,
    nccl_allreduce_cost,
    nccl_reduce_scatter_cost,
)

_nccl_topo_config: NCCLTopoConfig | None = None


def set_nccl_topo_config(config: NCCLTopoConfig | None) -> None:
    global _nccl_topo_config
    _nccl_topo_config = config
    _comms_cost_cache.clear()


def get_nccl_topo_config() -> NCCLTopoConfig | None:
    return _nccl_topo_config


def _node_value(value):
    if not isinstance(value, torch.fx.Node):
        return value
    for key in ("val", "example_value"):
        if key in value.meta:
            return value.meta[key]
    raise RuntimeError(f"FX node {value.name} has no example value")


def estimate_local_map_collective_cost(
    node, mesh, balanced_tokens=None
) -> float | None:
    if node.op != "call_function":
        return None

    if node.target == torch.ops._c10d_functional.wait_tensor.default:
        return 0.0
    elif node.target == torch.ops._c10d_functional.all_gather_into_tensor.default:
        collective = "allgather"
        group_index = 2
        group_size_index = 1
        tensor_value = node
    elif node.target == torch.ops._c10d_functional.reduce_scatter_tensor.default:
        collective = "reduce_scatter"
        group_index = 3
        group_size_index = 2
        tensor_value = node.args[0]
    elif node.target == torch.ops._c10d_functional.all_reduce.default:
        collective = "allreduce"
        group_index = 2
        group_size_index = None
        tensor_value = node.args[0]
    elif node.target == torch.ops._c10d_functional.all_to_all_single.default:
        collective = "all_to_all"
        group_index = 3
        group_size_index = None
        tensor_value = node.args[0]
    elif str(node.target).startswith("_c10d_functional."):
        raise RuntimeError(f"Unsupported local_map collective {node.target}")
    else:
        return None

    args = tuple(_node_value(arg) for arg in node.args)
    group_name = args[group_index]
    mesh_dims = [
        mesh_dim
        for mesh_dim in range(mesh.ndim)
        if mesh.get_group(mesh_dim).group_name == group_name
    ]
    if len(mesh_dims) != 1:
        raise RuntimeError(
            f"Collective group {group_name!r} maps to {len(mesh_dims)} mesh dimensions"
        )
    mesh_dim = mesh_dims[0]

    if group_size_index is not None:
        group_size = args[group_size_index]
        if group_size != mesh.size(mesh_dim):
            raise RuntimeError(
                f"Collective group size {group_size} does not match mesh dimension "
                f"size {mesh.size(mesh_dim)}"
            )

    tensor = _node_value(tensor_value)
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(f"Collective {node.target} has no tensor metadata")
    numel = (
        tensor.numel()
        if balanced_tokens is None
        else _concretize_unbacked_numel(tensor, balanced_tokens)
    )
    comm_bytes = numel * tensor.element_size()
    return collective_comm_cost(
        collective,
        int(comm_bytes),
        tuple(mesh.shape),
        mesh_dim,
        MeshTopoInfo.build_from_mesh(mesh),
    )


def _nccl_comm_cost(
    mesh_shape: tuple[int, ...],
    mesh_dim: int,
    comm_bytes: int,
    collective: str,
) -> float:
    """Compute communication cost using the NCCL model for a single mesh dim."""
    assert _nccl_topo_config is not None
    topo = derive_mesh_dim_topo(_nccl_topo_config, mesh_shape, mesh_dim)
    if collective == "allgather":
        return nccl_allgather_cost(comm_bytes, topo, _nccl_topo_config)
    elif collective == "allreduce":
        return nccl_allreduce_cost(comm_bytes, topo, _nccl_topo_config)
    elif collective == "reduce_scatter":
        return nccl_reduce_scatter_cost(comm_bytes, topo, _nccl_topo_config)
    elif collective == "all_to_all":
        return nccl_all_to_all_cost(comm_bytes, topo, _nccl_topo_config)
    else:
        return 0.0


def collective_comm_cost(
    collective: str,
    comm_bytes: int,
    mesh_shape: tuple[int, ...],
    mesh_dim: int,
    mesh_topo: MeshTopoInfo | None = None,
) -> float:
    """Single dispatch point for collective communication cost.

    Works with both the NCCL cost model (when configured via set_nccl_topo_config)
    and the default PyTorch cost model.

    Args:
        collective: one of "allgather", "allreduce", "reduce_scatter", "all_to_all"
        comm_bytes: total bytes communicated (post-collective size for allgather,
            pre-collective size for reduce_scatter)
        mesh_shape: shape of the device mesh
        mesh_dim: which mesh dimension the collective runs on
        mesh_topo: required when NCCL cost model is not configured
    """
    if _nccl_topo_config is not None:
        return _nccl_comm_cost(mesh_shape, mesh_dim, comm_bytes, collective)
    assert mesh_topo is not None
    comm_bytes_gb = comm_bytes / 1024**3
    if collective == "allgather":
        return allgather_cost(comm_bytes_gb, mesh_topo, mesh_dim)
    elif collective == "allreduce":
        return allreduce_cost(comm_bytes_gb, mesh_topo, mesh_dim)
    elif collective == "reduce_scatter":
        return reduce_scatter_cost(comm_bytes_gb, mesh_topo, mesh_dim)
    elif collective == "all_to_all":
        return all_to_all_cost(comm_bytes_gb, mesh_topo, mesh_dim)
    return 0.0


def all_to_all_cost(bytes_gb: float, mesh_topo: MeshTopoInfo, mesh_dim: int) -> float:
    num_devices_on_mesh_dim = mesh_topo.mesh_dim_devices[mesh_dim]
    mesh_dim_bandwidth = mesh_topo.mesh_dim_bandwidth[mesh_dim]
    num_hops = num_devices_on_mesh_dim - 1
    # base latency + comm latency
    latency = 6.6 + num_hops * mesh_topo.mesh_dim_latency[mesh_dim]  # us
    bw = (bytes_gb * num_hops / num_devices_on_mesh_dim) / mesh_dim_bandwidth  # s
    total_time = latency + bw * 1e6  # rescale to us
    # FIXME: this is a hack, we need to spend some more effort on the cost model
    total_time *= 5
    return total_time


# this is a copy-paste from https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/_collective_utils.py
# with iteration order introduced
def redistribute_cost(
    current_spec: "dtensor_spec.DTensorSpec",
    target_spec: "dtensor_spec.DTensorSpec",
    order: list[int],
) -> float:
    """
    This function returns the cost of redistribute from current to target DTensorSpec.

    NOTE:
    1. Only consider communication cost here, since computation costs for redistribute
       are quite trivial (i.e. we only need to narrow or simple division)
    2. Only consider redistribute cost on same mesh, cross mesh communication cost is
       not quite needed for operator strategy estimation/selection.
    """
    if current_spec.mesh != target_spec.mesh:
        # make infinite cost if meshes are not same
        # TODO: see if we want to support this once there's cross mesh communication
        return float("inf")

    if current_spec.is_replicated():
        # short-cut:
        # comm cost is 0 if current spec is already full replication
        # except if output is partial, which doesn't make sense for us
        if any(p.is_partial() for p in target_spec.placements):
            return float("inf")
        return 0.0

    mesh_topo = MeshTopoInfo.build_from_mesh(current_spec.mesh)
    cost = 0.0
    comm_bytes_gb = (
        spec_to_bytes(current_spec) / current_spec.num_shards / 1024 / 1024 / 1024
    )
    mesh_shape = tuple(current_spec.mesh.shape)
    # Transformation that considered for redistribute cost:
    # 1. allgather 2. alltoall
    # 3. allreduce 4. reduce_scatter
    curr_placements = [current_spec.placements[i] for i in order]
    tgt_placements = [target_spec.placements[i] for i in order]
    is_contiguous: bool = check_contiguous_sizes_strides(
        current_spec.shape, current_spec.stride
    )
    for i, current, target in zip(order, curr_placements, tgt_placements):
        if current == target:
            continue
        num_devices_on_mesh_dim = mesh_topo.mesh_dim_devices[i]
        if not is_contiguous:
            cost += compute_read_write_time(comm_bytes_gb * 2 * 1024**3)
        if current.is_shard() and target.is_replicate():
            current = cast(Shard, current)
            # allgather gives larger comm bytes
            comm_bytes_gb *= num_devices_on_mesh_dim
            cost += collective_comm_cost(
                "allgather", int(comm_bytes_gb * 1024**3), mesh_shape, i, mesh_topo
            )
            if current.dim != 0:
                # penalize cases like  S(1) -> R as there are additional compute cost
                # which corresponds to reshuffling the whole output tensor
                # we multiply the cost by 2 because we need to count input and output
                # reads for the reshuffle
                compute_cost = compute_read_write_time(comm_bytes_gb * 2 * 1024**3)
                cost += compute_cost
        elif current.is_shard() and target.is_shard():
            current = cast(Shard, current)
            target = cast(Shard, target)
            cost += collective_comm_cost(
                "all_to_all", int(comm_bytes_gb * 1024**3), mesh_shape, i, mesh_topo
            )

            num_copies = 0
            if current.dim != 0:
                num_copies += 1

            if target.dim != 0:
                num_copies += 1

            compute_cost = compute_read_write_time(comm_bytes_gb * 2 * 1024**3)
            cost += num_copies * compute_cost

        elif current.is_partial() and target.is_replicate():
            cost += collective_comm_cost(
                "allreduce", int(comm_bytes_gb * 1024**3), mesh_shape, i, mesh_topo
            )
        elif current.is_partial() and target.is_shard():
            target = cast(Shard, target)
            cost += collective_comm_cost(
                "reduce_scatter",
                int(comm_bytes_gb * 1024**3),
                mesh_shape,
                i,
                mesh_topo,
            )
            if target.dim != 0:
                # penalize cases like  P -> S(1) as there are additional compute cost
                # which corresponds to reshuffling the whole input tensor
                # we multiply the cost by 2 because we need to count input and output
                # reads for the reshuffle
                compute_cost = compute_read_write_time(comm_bytes_gb * 2 * 1024**3)
                cost += compute_cost
            # after reduce_scatter the comm bytes for further collectives halved.
            comm_bytes_gb /= num_devices_on_mesh_dim
        elif current.is_shard() and target.is_partial():
            # ban shard -> partial as it does not make sense to perform
            # this redistribute
            return float("inf")
        elif current.is_replicate() and target.is_partial():
            # ban replicate -> partial as it does not make sense to perform
            # this redistribute in our case
            return float("inf")

        # once we redistribute across one mesh dim, assume the output
        # is now contiguous. This is generally the case for most operations,
        # except when we fuse nd collectives into a 1d collective.
        is_contiguous = True

    return cost


_comms_cost_cache: dict[tuple, float] = {}


def reset_comms_cost_cache():
    _comms_cost_cache.clear()


def estimate_strategy_comms_cost(src_spec, tgt_spec):
    key = (
        src_spec.placements,
        src_spec.tensor_meta,
        tgt_spec.placements,
        tgt_spec.tensor_meta,
    )
    try:
        hash(key)  # fail fast if key contains unhashable types (e.g. SymInts)
    except TypeError:
        key = None
    if key is not None:
        cached = _comms_cost_cache.get(key)
        if cached is not None:
            return cached
    order = list(range(src_spec.mesh.ndim))
    if src_spec.placements == (Partial(), Partial()) and all(
        p.is_shard() for p in tgt_spec.placements
    ):
        order = [1, 0]
    cost = redistribute_cost(src_spec, tgt_spec, order)
    if key is not None:
        _comms_cost_cache[key] = cost
    return cost
