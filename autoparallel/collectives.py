# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional, Tuple

import torch
import torch.distributed.distributed_c10d as c10d
from torch.distributed._tensor.experimental import local_map as _local_map
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Placement

# Import GroupName for type checking
GroupName = c10d.GroupName

_local_map_device_mesh = None


def with_sharding_constraint(
    x: torch.Tensor,
    shardings: Tuple[Placement, ...],
    device_mesh: Optional[DeviceMesh] = None,
) -> torch.Tensor:
    """Constrain the sharding of an intermediate tensor.

    Similar to JAX's with_sharding_constraint, this constrains the sharding
    of a tensor to a specific placement. This is useful for controlling
    intermediate tensor shardings within a computation.

    Args:
        x: The tensor to constrain.
        shardings: Tuple of placements specifying how the tensor should be
            sharded across each mesh dimension.
        device_mesh: The device mesh to use. If None, uses the mesh from
            the enclosing local_map region.

    Returns:
        The tensor with the specified sharding constraint applied.

    Example:
        >>> from torch.distributed.tensor.placement_types import Shard, Replicate
        >>> # Inside a local_map region or with explicit mesh:
        >>> x = with_sharding_constraint(x, (Shard(0), Replicate()))
    """
    if device_mesh is None:
        device_mesh = get_mesh_from_global()

    @_local_map(
        out_placements=(shardings,),
        in_placements=(shardings,),
        redistribute_inputs=True,
        device_mesh=device_mesh,
    )
    def identity(t):
        # clone() is required because local_map HOP doesn't support
        # input-to-output aliasing during dynamo tracing
        return t.clone()

    return identity(x)


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
    group_name: GroupName,
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
    self: torch.Tensor, reduceOp: str, scatter_dim: int, group_name: GroupName
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


def _all_reduce(self: torch.Tensor, reduceOp: str, group_name: GroupName):
    tensor = torch.ops._c10d_functional.all_reduce(self, reduceOp.lower(), group_name)
    res = torch.ops._c10d_functional.wait_tensor(tensor)
    return res


def _all_to_all(
    self: torch.Tensor,
    output_split_sizes: Optional[list[int]],
    input_split_sizes: Optional[list[int]],
    group_name: GroupName,
):
    group_size = c10d._get_group_size_by_name(group_name)
    if output_split_sizes is None or input_split_sizes is None:
        assert output_split_sizes is None and input_split_sizes is None, (
            "output_split_sizes and input_split_sizes must either be "
            "specified together or both set to None"
        )
        output_split_sizes = [self.shape[0] // group_size] * group_size
        input_split_sizes = output_split_sizes

    tensor = torch.ops._c10d_functional.all_to_all_single(
        self, output_split_sizes, input_split_sizes, group_name
    )
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


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        output_split_sizes: Optional[list[int]],
        input_split_sizes: Optional[list[int]],
        axis_name: str,
    ):
        group_name = _get_group_name_from_axis_name(axis_name)
        ctx.group_name = group_name
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        return _all_to_all(x, output_split_sizes, input_split_sizes, group_name)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):  # type: ignore[override]
        return _all_to_all(
            grad_output, ctx.input_split_sizes, ctx.output_split_sizes, ctx.group_name
        )


all_gather = _AllGather.apply
all_reduce = _AllReduce.apply
reduce_scatter = _ReduceScatter.apply
all_to_all = _AllToAll.apply
