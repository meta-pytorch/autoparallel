# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

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

from .compute_estimation import compute_read_write_time


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
            # add up allgather comm cost
            cost += allgather_cost(comm_bytes_gb, mesh_topo, i)
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
            # should be alltoall comm, since we haven't implement it yet, add penalty
            # to favor allgather instead
            cost += all_to_all_cost(comm_bytes_gb, mesh_topo, i)  # us

            num_copies = 0
            if current.dim != 0:
                num_copies += 1

            if target.dim != 0:
                num_copies += 1

            compute_cost = compute_read_write_time(comm_bytes_gb * 2 * 1024**3)
            cost += num_copies * compute_cost

        elif current.is_partial() and target.is_replicate():
            # add up allreduce comm cost
            cost += allreduce_cost(comm_bytes_gb, mesh_topo, i)
        elif current.is_partial() and target.is_shard():
            target = cast(Shard, target)
            # add up reduce_scatter comm cost
            cost += reduce_scatter_cost(comm_bytes_gb, mesh_topo, i)
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


def estimate_strategy_comms_cost(src_spec, tgt_spec):
    order = list(range(src_spec.mesh.ndim))
    if src_spec.placements == (Partial(), Partial()) and all(
        p.is_shard() for p in tgt_spec.placements
    ):
        order = [1, 0]
    comms_cost = redistribute_cost(src_spec, tgt_spec, order)
    return comms_cost
