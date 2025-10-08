# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.autograd
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode


class TestMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        kwargs = kwargs if kwargs else {}
        out = func(*args, **kwargs)
        print(func)
        pytree.tree_map_only(torch.Tensor, lambda x: print(x.shape), args)
        print(
            "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
        )
        return out


def _validate_batched_args(b_args: list[int]) -> None:
    assert b_args in [
        [
            0,
        ],
        [
            1,
        ],
        [0, 1],
    ], f"Invalid batched_args: {b_args}"


def _transform_grouped_mm_args(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    batched_args: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if batched_args == [
        0,
    ]:
        mat1_exp = mat1
        mat2_exp = mat2.expand(mat1.shape[0], -1, -1, -1)
    elif batched_args == [
        1,
    ]:
        mat1_exp = mat1.expand(mat2.shape[0], -1, -1, -1)
        mat2_exp = mat2
    else:
        mat1_exp = mat1
        mat2_exp = mat2
    return mat1_exp, mat2_exp


def _validate_batched_grouped_mm_shapes(
    mat1_shape: torch.Size,
    mat2_shape: torch.Size,
    offs_shape: torch.Size,
) -> None:
    # Case 1: mat1 and mat2 are both 3D batched tensors
    # grouped_mm 2Dx2D case
    # Case 2: mat1 is 3D batched tensor and mat2 is 4D batched tensor
    # grouped_mm 2Dx3D case
    # Case 3: mat1 is 4D batched tensor and mat2 is 3D batched tensor
    # grouped_mm 3Dx2D case
    valid_shapes = (
        (len(mat1_shape) == 3 and len(mat2_shape) == 3)
        or (len(mat1_shape) == 4 and len(mat2_shape) == 3)
        or (len(mat1_shape) == 3 and len(mat2_shape) == 4)
    )
    assert valid_shapes, (
        f"Invalid tensor dimensions: expected mat1 and mat2 to be either both 3D, "
        f"or mat1 4D and mat2 3D, or mat1 3D and mat2 4D. "
        f"Got mat1={mat1_shape} and mat2={mat2_shape}."
    )
    assert len(offs_shape) == 2, f"offs needs to be 2D, got {offs_shape}"
    # OB = batch size
    # IB = sum(num_tokens_per_expert)
    # D = embedding dim
    # H = hidden dim
    # E = number of experts
    if len(mat1_shape) == 3 and len(mat2_shape) == 3:
        OB1, _, IB1 = mat1_shape
        OB2, IB2, _ = mat2_shape
        assert OB1 == OB2, f"Batch size mismatch: {OB1} vs {OB2}"
        assert IB1 == IB2, f"Total tokens mismatch: {IB1} vs {IB2}"
    elif len(mat1_shape) == 4 and len(mat2_shape) == 3:
        OB1, E1, _, H1 = mat1_shape
        OB2, H2, _ = mat2_shape
        OB3, E2 = offs_shape
        assert (
            OB1 == OB2 and OB2 == OB3
        ), f"Batch size mismatch: {OB1} vs {OB2} vs {OB3}"
        assert H1 == H2, f"Contracting dimension mismatch: {H1} vs {H2}"
        assert E1 == E2, f"Number of experts mismatch: {E1} vs {E2}"
    else:
        OB1, IB1, D1 = mat1_shape
        OB2, E1, D2, _ = mat2_shape
        OB3, E2 = offs_shape
        assert (
            OB1 == OB2 and OB2 == OB3
        ), f"Batch size mismatch: {OB1} vs {OB2} vs {OB3}"
        assert D1 == D2, f"Contracting dimension mismatch: {D1} vs {D2}"
        assert E1 == E2, f"Number of experts mismatch: {E1} vs {E2}"


@torch.library.custom_op("autoparallel::batched_grouped_mm", mutates_args=())
def batched_grouped_mm(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    offs: torch.Tensor,
    batched_args: list[int],
) -> torch.Tensor:
    _validate_batched_args(batched_args)
    mat1_exp, mat2_exp = _transform_grouped_mm_args(mat1, mat2, batched_args)
    _validate_batched_grouped_mm_shapes(mat1_exp.shape, mat2_exp.shape, offs.shape)
    res = []
    for m1, m2, off in zip(mat1_exp, mat2_exp, offs):
        res.append(torch._grouped_mm(m1, m2, off))
    return torch.stack(res, 0)


def setup_context_batched_grouped_mm(ctx, inputs, output):
    mat1, mat2, offs, batched_args = inputs
    ctx.save_for_backward(mat1, mat2, offs)
    ctx.batched_args = batched_args


def _backward_batched_grouped_mm_mat1(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    grad: torch.Tensor,
    offs: torch.Tensor,
    batched_args: list[int],
) -> torch.Tensor:
    if mat1.stride(-2) == 1 and mat1.stride(-1) == mat1.size(-2):
        if batched_args == [
            0,
        ]:
            b_args = [
                1,
            ]
        else:
            b_args = [0, 1]
        # if input was column-major, return grad_input as column-order for efficiency
        grad_mat1 = batched_grouped_mm(
            mat2, grad.transpose(-2, -1), offs, b_args
        ).transpose(-2, -1)
    else:
        if batched_args == [
            0,
        ]:
            b_args = [
                0,
            ]
        else:
            b_args = [0, 1]
        grad_mat1 = batched_grouped_mm(grad, mat2.transpose(-2, -1), offs, b_args)
    return grad_mat1


def _backward_batched_grouped_mm_mat2(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    grad: torch.Tensor,
    offs: torch.Tensor,
    batched_args: list[int],
) -> torch.Tensor:
    if mat2.stride(-2) == 1 and mat2.stride(-1) == mat2.size(-2):
        if batched_args == [
            1,
        ]:
            b_args = [
                0,
            ]
        else:
            b_args = [0, 1]
        # if experts were column-major, return experts_grad as column-order for efficiency
        grad_mat2 = batched_grouped_mm(
            grad.transpose(-2, -1),
            mat1,
            offs,
            b_args,
        ).transpose(-2, -1)
    else:
        if batched_args == [
            1,
        ]:
            b_args = [
                1,
            ]
        else:
            b_args = [0, 1]
        grad_mat2 = batched_grouped_mm(
            mat1.transpose(-2, -1),
            grad,
            offs,
            b_args,
        )
    return grad_mat2


def backward_batched_grouped_mm(
    ctx, grad: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, None, None]:
    mat1, mat2, offs = ctx.saved_tensors
    batched_args = ctx.batched_args
    grad_mat1 = _backward_batched_grouped_mm_mat1(mat1, mat2, grad, offs, batched_args)
    grad_mat2 = _backward_batched_grouped_mm_mat2(mat1, mat2, grad, offs, batched_args)
    if batched_args == [
        0,
    ]:
        grad_mat2 = torch.sum(grad_mat2, dim=0)
    elif batched_args == [
        1,
    ]:
        grad_mat1 = torch.sum(grad_mat1, dim=0)

    return grad_mat1, grad_mat2, None, None


torch.library.register_autograd(
    "autoparallel::batched_grouped_mm",
    backward_batched_grouped_mm,
    setup_context=setup_context_batched_grouped_mm,
)


def test_functional_correctness():
    """Test that our implementation matches reference torch._grouped_mm calls."""
    print("=== Functional correctness test ===")
    torch.manual_seed(42)
    device = "cuda"

    batch_size = 2
    seq_len = 64
    num_experts = 2
    dim = 128
    hidden_dim = 256

    mat1 = torch.rand(batch_size, seq_len, dim, requires_grad=True, device=device)
    mat2 = torch.rand(num_experts, dim, hidden_dim, requires_grad=True, device=device)
    offsets = torch.tensor([[32, 64], [16, 64]], dtype=torch.int32, device=device)

    # Test our implementation
    with TestMode():
        output_ours = torch.ops.autoparallel.batched_grouped_mm(
            mat1.bfloat16(),
            mat2.bfloat16(),
            offsets,
            [
                0,
            ],
        ).type_as(mat1)
        loss_ours = output_ours.sum()
        print("Backward")
        loss_ours.backward()
    grad1_ours = mat1.grad.clone()
    grad2_ours = mat2.grad.clone()

    # Reset and test reference
    mat1.grad = None
    mat2.grad = None

    # Reference implementation using direct torch._grouped_mm calls
    res_ref = []
    for m1, off in zip(mat1, offsets):
        res_ref.append(torch._grouped_mm(m1.bfloat16(), mat2.bfloat16(), off))
    output_ref = torch.stack(res_ref, 0).type_as(mat1)
    loss_ref = output_ref.sum()
    loss_ref.backward()

    # Compare results
    assert torch.allclose(
        output_ours, output_ref, rtol=1e-2, atol=1e-2
    ), "Forward outputs don't match"
    assert torch.allclose(
        grad1_ours, mat1.grad, rtol=1e-2, atol=1e-2
    ), "mat1 gradients don't match"
    assert torch.allclose(
        grad2_ours, mat2.grad, rtol=1e-2, atol=1e-2
    ), "mat2 gradients don't match"

    print("✓ Functional correctness test PASSED")


def test_gradient_properties():
    """Test gradient shapes, devices, and numerical properties."""
    print("=== Gradient properties test ===")
    torch.manual_seed(42)
    device = "cuda"

    batch_size = 3
    seq_len = 32
    num_experts = 4
    dim = 64
    hidden_dim = 96

    mat1 = torch.rand(batch_size, seq_len, dim, requires_grad=True, device=device)
    mat2 = torch.rand(num_experts, dim, hidden_dim, requires_grad=True, device=device)
    offsets = torch.tensor(
        [[8, 16, 24, 32], [4, 12, 20, 28], [6, 14, 22, 30]],
        dtype=torch.int32,
        device=device,
    )

    output = torch.ops.autoparallel.batched_grouped_mm(
        mat1.bfloat16(),
        mat2.bfloat16(),
        offsets,
        [
            0,
        ],
    ).type_as(mat1)

    loss = output.sum()
    loss.backward()

    # Check shapes
    assert (
        mat1.grad.shape == mat1.shape
    ), f"mat1 grad shape: {mat1.grad.shape} vs {mat1.shape}"
    assert (
        mat2.grad.shape == mat2.shape
    ), f"mat2 grad shape: {mat2.grad.shape} vs {mat2.shape}"

    # Check devices
    assert mat1.grad.device == mat1.device, f"mat1 grad device: {mat1.grad.device}"
    assert mat2.grad.device == mat2.device, f"mat2 grad device: {mat2.grad.device}"

    # Check numerical properties
    assert not torch.isnan(mat1.grad).any(), "mat1 grad has NaN"
    assert not torch.isnan(mat2.grad).any(), "mat2 grad has NaN"
    assert not torch.isinf(mat1.grad).any(), "mat1 grad has Inf"
    assert not torch.isinf(mat2.grad).any(), "mat2 grad has Inf"

    # Check gradients are not zero (would indicate broken backward)
    assert mat1.grad.abs().max() > 1e-6, "mat1 grad is suspiciously small"
    assert mat2.grad.abs().max() > 1e-6, "mat2 grad is suspiciously small"

    print("✓ Gradient properties test PASSED")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. Skipping tests.")
        exit(1)

    print(f"🚀 Running tests on CUDA device: {torch.cuda.get_device_name()}")

    test_functional_correctness()
    test_gradient_properties()

    print("\n🎉 All tests completed successfully!")
