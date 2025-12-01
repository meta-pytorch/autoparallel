# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
import torch.nn as nn
from autoparallel.graph_pp_runner import GraphPipelineStage
from torch._C._distributed_c10d import FakeWork, PythonCallbackWork
from torch.distributed import DeviceMesh
from torch.distributed._local_tensor import (
    local_tensor_mode,
    LocalIntNode,
    LocalRunnerMode,
    LocalTensor,
    LocalTensorMode,
    maybe_disable_local_tensor_mode,
)
from torch.distributed._local_tensor._c10d import local_p2p_op
from torch.distributed.pipelining.stage import InputInfo, PipelineStage
from torch.distributed.tensor import DTensor
from torch.export._unlift import _assign_attr
from torch.export.unflatten import _AttrKind


_pg_groups: list[list[int]] = []


def create_local_tensor_mode(dp_ep_mesh: DeviceMesh, pp_rank: int) -> LocalTensorMode:
    dp_ep_full_mesh = dp_ep_mesh._layout.remap_to_tensor(dp_ep_mesh._rank_map)
    dp_ep_ranks = dp_ep_full_mesh[pp_rank].flatten().tolist()
    print(f"Creating local tensor mode for ranks {dp_ep_ranks}")
    return LocalTensorMode(frozenset(dp_ep_ranks))


def cache_pp_groups(pp_mesh: DeviceMesh) -> list[list[int]]:
    pp_full_mesh = pp_mesh._layout.remap_to_tensor(pp_mesh._rank_map)
    pp_groups = []
    for i in range(pp_full_mesh.size(dim=0)):
        pp_group = pp_full_mesh[i].tolist()
        pp_groups.append(pp_group)
    global _pp_groups
    _pp_groups = pp_groups
    return pp_groups


def combine_works(works: list[dist.Work], ctx: str | None = None) -> dist.Work:
    def _wait_all(timeout) -> bool:
        for w in works:
            w.wait()
        return True

    return PythonCallbackWork(_wait_all)


def get_pp_peer(self: int, peer: int) -> torch.SymInt:
    pp_ret = {}
    global _pp_groups
    for pp_group in _pp_groups:
        global_rank = pp_group[self]
        global_peer = pp_group[peer]
        pp_ret[global_rank] = global_peer
    return torch.SymInt(LocalIntNode(pp_ret))


def expand_p2p_ops(
    ops: list[dist.P2POp], pp_rank: int, ctx: str | None = None
) -> list[dist.P2POp]:
    # Ops where generated from a perspective of pp group where rank 0 is present.

    def multi_isend(tensor, dst=None, group=None, tag=0, group_src=None):
        assert group_src is not None, "Expected group rank"
        peer = get_pp_peer(pp_rank, group_src)
        if not isinstance(tensor, LocalTensor):
            tensor = maybe_make_tensor_local(tensor)
        works = local_p2p_op(peer, tensor, dist.isend)
        return FakeWork()

    def multi_irecv(tensor, src=None, group=None, tag=0, group_src=None):
        assert group_src is not None, "Expected group rank"
        peer = get_pp_peer(pp_rank, group_src)
        assert isinstance(tensor, LocalTensor), "Expected LocalTensor"
        works = local_p2p_op(peer, tensor, dist.irecv)
        return combine_works(works)

    send_ops = []
    recv_ops = []
    for p2p_op in ops:
        op = p2p_op.op
        if op is dist.isend:
            p2p_op.op = multi_isend
            send_ops.append(p2p_op)
        elif op is dist.irecv:
            p2p_op.op = multi_irecv
            recv_ops.append(p2p_op)
        else:
            raise AssertionError("Unxpected op {op}")

    # Execute send ops first and then recv because the latter are blocking
    return send_ops + recv_ops


class LocalGraphPipelineStage(GraphPipelineStage):
    def log_name(self) -> str:
        return (
            f"PP rank {self.group_rank} Stage {self.stage_index} of {self.num_stages}"
        )

    def _get_recv_ops(self, recv_infos: tuple[InputInfo, ...]) -> list[dist.P2POp]:
        ops = super()._get_recv_ops(recv_infos)
        ops = expand_p2p_ops(ops, self.group_rank, self.log_name() + " _get_recv_ops")
        return ops

    def get_fwd_send_ops(self, fwd_chunk_id: int) -> list[dist.P2POp]:
        ops = super().get_fwd_send_ops(fwd_chunk_id)
        ops = expand_p2p_ops(
            ops, self.group_rank, self.log_name() + " get_fwd_send_ops"
        )
        return ops

    def get_bwd_send_ops(self, bwd_chunk_id: int) -> list[dist.P2POp]:
        ops = super().get_bwd_send_ops(bwd_chunk_id)
        ops = expand_p2p_ops(
            ops, self.group_rank, self.log_name() + " get_bwd_send_ops"
        )
        return ops

    def _get_init_p2p_neighbors_ops(self) -> list[dist.P2POp]:
        ops = super()._get_init_p2p_neighbors_ops()
        ops = expand_p2p_ops(
            ops, self.group_rank, self.log_name() + " _get_init_p2p_neighbors_ops"
        )
        return ops


def local_tensor_mode_if_enabled(
    ltm: LocalTensorMode | None = None,
) -> LocalTensorMode | None:

    for _ in range(2):
        if ltm is not None and not ltm._disable:
            return ltm
        ltm = local_tensor_mode()

    return None


def maybe_make_tensor_local(
    tensor: torch.Tensor,
    ltm: LocalTensorMode | None = None,
) -> torch.Tensor:
    ltm = local_tensor_mode_if_enabled(ltm)
    if ltm is None:
        return tensor

    if isinstance(tensor, LocalTensor):
        return tensor

    if isinstance(tensor, DTensor):
        tensor._local_tensor = maybe_make_tensor_local(tensor._local_tensor, ltm)
        return tensor

    local_tensor = ltm.rank_map(lambda r: tensor.clone().detach())
    local_tensor.requires_grad = tensor.requires_grad
    return local_tensor


def maybe_make_module_local(
    module: nn.Module,
    ltm: LocalTensorMode | None = None,
) -> None:
    ltm = local_tensor_mode_if_enabled(ltm)
    print(f"maybe_make_module_local {ltm.ranks}")
    if ltm is None:
        return

    for k, v in module.named_parameters():
        _assign_attr(
            nn.Parameter(
                data=maybe_make_tensor_local(v.data, ltm),
                requires_grad=v.requires_grad,
            ),
            module,
            k,
            attr_kind=_AttrKind.PARAMETER,
        )

    for k, v in module.named_buffers():
        _assign_attr(
            maybe_make_tensor_local(v, ltm), module, k, attr_kind=_AttrKind.BUFFER
        )
