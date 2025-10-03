# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
import torch.distributed.distributed_c10d as c10d
from torch.distributed._tensor.experimental import local_map as _local_map

_local_map_device_mesh = None


def local_map(*args, **kwargs):
    # TODO: ideally after we get out of the local map region we should
    # just reset the global device mesh to None. For now we just keep it
    # around.
    global _local_map_device_mesh
    _local_map_device_mesh = kwargs.get("device_mesh", None)
    return _local_map(*args, **kwargs)


def get_mesh_from_global():
    global _local_map_device_mesh
    if _local_map_device_mesh is None:
        raise RuntimeError(
            "No mesh found, make sure to call this collective in a local_map region"
        )
    return _local_map_device_mesh


def _get_group_name_from_axis_name(mesh_name):
    mesh = get_mesh_from_global()
    group = mesh.get_group(mesh_name)
    return group.group_name


def axis_size(axis_name):
    mesh = get_mesh_from_global()
    assert axis_name in mesh.mesh_dim_names
    axis_dim = mesh.mesh_dim_names.index(axis_name)
    return mesh.size(axis_dim)


def axis_index(axis_name):
    mesh = get_mesh_from_global()
    return mesh.get_local_rank(mesh_dim=axis_name)


def _all_gather_tensor(
    x: torch.Tensor,
    gather_dim: int,
    group_name: str,
) -> torch.Tensor:
    x = x.contiguous()
    group_size = c10d._get_group_size_by_name(group_name)
    tensor = torch.ops._c10d_functional.all_gather_into_tensor(
        x, group_size, group_name
    )
    res = torch.ops._c10d_functional.wait_tensor(tensor)
    if gather_dim != 0:
        # torch.cat access the data so we already need to wait here, first do wait
        # and then chunk + cat avoid us going through ACT dispatching logic again
        res = torch.cat(torch.chunk(res, group_size, dim=0), dim=gather_dim)
    return res


def _reduce_scatter_tensor(
    self: torch.Tensor, reduceOp: str, scatter_dim: int, group_name: str
):
    group_size = c10d._get_group_size_by_name(group_name)

    assert (
        self.size(scatter_dim) % group_size == 0
    ), f"input dimension 0 ({self.size(0)} must be a multiple of group_size {group_size})"
    if scatter_dim != 0:
        tensor_list = torch.chunk(self, group_size, dim=scatter_dim)
        self = torch.cat(tensor_list)

    tensor = torch.ops._c10d_functional.reduce_scatter_tensor(
        self,
        reduceOp.lower(),
        group_size,
        group_name,
    )
    res = torch.ops._c10d_functional.wait_tensor(tensor)
    return res


def _all_reduce(self: torch.Tensor, reduceOp: str, group_name: str):
    tensor = torch.ops._c10d_functional.all_reduce(self, reduceOp.lower(), group_name)
    res = torch.ops._c10d_functional.wait_tensor(tensor)
    return res


class _AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, gather_dim: int, axis_name: str):
        group_name = _get_group_name_from_axis_name(axis_name)
        ctx.group_name = group_name
        ctx.gather_dim = gather_dim
        return _all_gather_tensor(x, gather_dim, group_name)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):  # type: ignore[override]
        return (
            _reduce_scatter_tensor(grad_output, "sum", ctx.gather_dim, ctx.group_name),
            None,
            None,
        )


class _ReduceScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, scatter_dim: int, axis_name: str):
        group_name = _get_group_name_from_axis_name(axis_name)
        ctx.group_name = group_name
        ctx.scatter_dim = scatter_dim
        return _reduce_scatter_tensor(x, "sum", scatter_dim, group_name)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):  # type: ignore[override]
        return (
            _all_gather_tensor(grad_output, ctx.scatter_dim, ctx.group_name),
            None,
            None,
        )


class _AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, axis_name: str):
        group_name = _get_group_name_from_axis_name(axis_name)
        ctx.group_name = group_name
        return _all_reduce(x, "sum", group_name)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):  # type: ignore[override]
        # TODO: split this into a function that does all-reduce and one which is the identity
        return _all_reduce(grad_output, "sum", ctx.group_name), None


all_gather = _AllGather.apply
all_reduce = _AllReduce.apply
reduce_scatter = _ReduceScatter.apply
