# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.distributed.distributed_c10d as c10d


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
    def forward(ctx, x: torch.Tensor, gather_dim: int, group_name: str):
        ctx.group_name = group_name
        ctx.gather_dim = gather_dim
        return _all_gather_tensor(x, gather_dim, group_name)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return (
            _reduce_scatter_tensor(grad_output, "sum", ctx.gather_dim, ctx.group_name),
            None,
            None,
        )


class _ReduceScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scatter_dim: int, group_name: str):
        ctx.group_name = group_name
        ctx.scatter_dim = scatter_dim
        return _reduce_scatter_tensor(x, "sum", scatter_dim, group_name)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return (
            _all_gather_tensor(grad_output, ctx.scatter_dim, ctx.group_name),
            None,
            None,
        )


class _AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group_name: str):
        ctx.group_name = group_name
        return _all_reduce(x, "sum", group_name)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # TODO: split this into a function that does all-reduce and one which is the identity
        return _all_reduce(grad_output, "sum", ctx.group_name), None


all_gather = _AllGather.apply
all_reduce = _AllReduce.apply
reduce_scatter = _ReduceScatter.apply
