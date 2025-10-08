# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode


class TestMode(TorchDispatchMode):

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        kwargs = kwargs if kwargs else {}
        out = func(*args, **kwargs)
        print("Op:", func)
        print("Args:")
        pytree.tree_map_only(torch.Tensor, lambda x: print(x.shape), args)
        print("Out:")
        pytree.tree_map_only(torch.Tensor, lambda x: print(x.shape), out)
        print(
            "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
        )
        return out


@torch.library.custom_op("autoparallel::batched_mm", mutates_args=())
def batched_mm(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
) -> torch.Tensor:
    assert mat1.ndim == 3
    assert mat2.ndim == 2 or mat2.ndim == 3
    if mat2.ndim == 2:
        assert mat1.shape[2] == mat2.shape[0]
        mat2_expanded = mat2.expand(mat1.shape[0], -1, -1)
    else:
        assert mat1.shape[0] == mat2.shape[0]
        assert mat1.shape[2] == mat2.shape[1]
        mat2_expanded = mat2
    out = torch.bmm(mat1, mat2_expanded)
    return out


def setup_context_batched_mm(ctx, inputs, output):
    mat1, mat2 = inputs
    ctx.save_for_backward(mat1, mat2)


def backward_batched_mm(ctx, grad):
    assert grad.ndim == 3
    mat1, mat2 = ctx.saved_tensors
    grad1 = batched_mm(grad, mat2.transpose(-2, -1))
    grad2 = torch.sum(batched_mm(mat1.transpose(-2, -1), grad), dim=0)
    return grad1, grad2


torch.library.register_autograd(
    "autoparallel::batched_mm",
    backward_batched_mm,
    setup_context=setup_context_batched_mm,
)


if __name__ == "__main__":
    DEVICE = "cuda"

    mat1 = torch.rand(
        10, 32, 16, device=DEVICE, dtype=torch.float32, requires_grad=True
    )
    mat2 = torch.rand(48, 16, device=DEVICE, dtype=torch.float32, requires_grad=True)

    out = batched_mm(mat1, mat2.transpose(-2, -1))
    out = out.sum()
    with TestMode():
        out.backward()
    mat1grad = mat1.grad
    mat2grad = mat2.grad
    mat1.grad = None
    mat2.grad = None
    out3 = mat1 @ mat2.transpose(-2, -1)
    out3 = out3.sum()
    out3.backward()
    print(torch.allclose(out, out3))
    print(torch.allclose(mat1.grad, mat1grad))
    print(torch.allclose(mat2.grad, mat2grad))
