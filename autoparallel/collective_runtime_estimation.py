# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch.distributed.tensor._dtensor_spec as dtensor_spec
from torch.distributed.tensor._collective_utils import (
    MeshTopoInfo,
    allgather_cost,
    allreduce_cost,
    reduce_scatter_cost,
    spec_to_bytes,
)
from torch.distributed.tensor.placement_types import Partial, Shard


def all_to_all_cost(bytes_gb: float, mesh_topo: MeshTopoInfo, mesh_dim: int) -> float:
    num_devices_on_mesh_dim = mesh_topo.mesh_dim_devices[mesh_dim]
    mesh_dim_bandwidth = mesh_topo.mesh_dim_bandwidth[mesh_dim]
    num_hops = num_devices_on_mesh_dim**2
    # base latency + comm latency
    latency = 6.6 + num_hops * mesh_topo.mesh_dim_latency[mesh_dim]  # us
    bw = (bytes_gb * num_hops / num_devices_on_mesh_dim) / mesh_dim_bandwidth  # s
    return latency + bw * 1e6  # rescale to us


# this is a copy-paste from https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/_collective_utils.py
# with iteration order introduced
# TODO: this should be improved, as we just really use the non-canonical order for
# PP->S(0)S(0) for now
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
        return 0.0

    mesh_topo = MeshTopoInfo.build_from_mesh(current_spec.mesh)
    cost = 0.0
    comm_bytes_gb = (
        spec_to_bytes(current_spec) / current_spec.num_shards / 1024 / 1024 / 1024
    )
    # Transformation that considered for redistribute cost:
    # 1. allgather 2. alltoall
    # 3. allreduce 4. reduce_scatter
    curr_placements = [current_spec.placements[i] for i in order]
    tgt_placements = [target_spec.placements[i] for i in order]
    for i, current, target in zip(order, curr_placements, tgt_placements):
        if current == target:
            continue
        num_devices_on_mesh_dim = mesh_topo.mesh_dim_devices[i]
        if current.is_shard() and target.is_replicate():
            # allgather gives larger comm bytes
            comm_bytes_gb *= num_devices_on_mesh_dim
            # add up allgather comm cost
            cost += allgather_cost(comm_bytes_gb, mesh_topo, i)
        elif current.is_shard() and target.is_shard():
            # should be alltoall comm, since we haven't implement it yet, add penalty
            # to favor allgather instead
            # cost += all_to_all_cost(comm_bytes_gb, mesh_topo, i)
            cost += allgather_cost(comm_bytes_gb, mesh_topo, i) * 4.0
        elif current.is_partial() and target.is_replicate():
            # add up allreduce comm cost
            cost += allreduce_cost(comm_bytes_gb, mesh_topo, i)
        elif current.is_partial() and target.is_shard():
            # add up reduce_scatter comm cost
            cost += reduce_scatter_cost(comm_bytes_gb, mesh_topo, i)
            # after reduce_scatter the comm bytes for further collectives halved.
            comm_bytes_gb /= num_devices_on_mesh_dim
        elif current.is_shard() and target.is_partial():
            # ban shard -> partial as it does not make sense to perform
            # this redistribute
            return float("inf")

    return cost


def estimate_strategy_comms_cost(src_spec, tgt_spec):
    order = list(range(src_spec.mesh.ndim))
    if src_spec.placements == (Partial(), Partial()) and tgt_spec.placements == (
        Shard(0),
        Shard(0),
    ):
        order = [1, 0]
    comms_cost = redistribute_cost(src_spec, tgt_spec, order)
    return comms_cost
