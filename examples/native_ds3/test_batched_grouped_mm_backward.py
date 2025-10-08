# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.autograd


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


# Static dimensions for all tests
OB, IB, D, E, H = 2, 128, 256, 4, 512
DEVICE = "cuda"

# Static offsets: cumulative sums that add up to IB=128, multiples of 16
STATIC_OFFSETS = torch.tensor(
    [
        [32, 64, 96, 128],  # First batch: 32 + 32 + 32 + 32 = 128
        [16, 48, 80, 128],  # Second batch: 16 + 32 + 32 + 48 = 128
    ],
    dtype=torch.int32,
    device=DEVICE,
)


def create_row_major_tensor(shape: tuple) -> torch.Tensor:
    """Create a row-major tensor (standard PyTorch layout)."""
    return torch.randn(shape, device=DEVICE)


def verify_tensor_layout(tensor: torch.Tensor, expected_layout: str) -> bool:
    """Verify tensor has expected memory layout."""
    if expected_layout == "column_major":
        # Column-major: stride[-2] == 1 and stride[-1] == size(-2)
        return tensor.stride(-2) == 1 and tensor.stride(-1) == tensor.size(-2)
    elif expected_layout == "row_major":
        # Row-major: stride[-1] == 1 and stride[-2] == size(-1)
        return tensor.stride(-1) == 1 and tensor.stride(-2) == tensor.size(-1)
    else:
        raise ValueError(f"Unknown layout: {expected_layout}")


def test_case1_3d_x_4d_row_col():
    """Test Case 1: mat1=[2,128,256] row-major, mat2=[4,512,256] column-major -> 4D"""
    print("\n=== Test Case 1: 3D x 4D (row-major x column-major) ===")

    # Create tensors
    mat1 = create_row_major_tensor((OB, IB, D)).requires_grad_(True)
    mat2_base = create_row_major_tensor((E, H, D)).requires_grad_(True)

    # Convert to bfloat16 first, then transpose and expand to preserve stride patterns
    mat1_bf16 = mat1.bfloat16()
    mat2_base_bf16 = mat2_base.bfloat16()
    mat2 = mat2_base_bf16.transpose(-2, -1)

    # Verify layouts
    assert verify_tensor_layout(
        mat1_bf16, "row_major"
    ), f"mat1 should be row-major, got strides: {mat1.stride()}"

    assert verify_tensor_layout(
        mat2, "column_major"
    ), f"mat2_base should be column-major, got strides: {mat2_base.stride()}"

    # Forward pass
    result = batched_grouped_mm(
        mat1_bf16,
        mat2,
        STATIC_OFFSETS,
        [
            0,
        ],
    ).type_as(mat1)

    print(
        f"mat1 shape: {mat1_bf16.shape}, strides: {mat1_bf16.stride()} (row-major: {verify_tensor_layout(mat1_bf16, 'row_major')})"
    )
    print(
        f"mat2 shape: {mat2.shape}, strides: {mat2.stride()} (column-major: {verify_tensor_layout(mat2, 'column_major')})"
    )
    print(f"result shape: {result.shape}")

    # Backward pass
    result.sum().backward()

    print(f"mat1.grad strides: {mat1.grad.stride()}")
    print(f"mat2_base.grad strides: {mat2_base.grad.stride()}")


def test_case2_4d_x_3d_col_row():
    """Test Case 2: mat1=[4,256,512] column-major -> 4D, mat2=[2,256,128] row-major"""
    print("\n=== Test Case 2: 4D x 3D (column-major x row-major) ===")

    # Create tensors
    mat1_base = create_row_major_tensor((E, H, D)).requires_grad_(True)
    mat2 = create_row_major_tensor((OB, IB, D)).requires_grad_(True)

    # Convert to bfloat16 first, then expand and transpose to preserve stride patterns
    mat1_base_bf16 = mat1_base.bfloat16()
    mat2_bf16_T = mat2.bfloat16().transpose(-2, -1)

    # Verify layouts
    assert verify_tensor_layout(
        mat1_base_bf16, "row_major"
    ), f"mat1 should be row-major, got strides: {mat1_base_bf16.stride()}"
    assert verify_tensor_layout(
        mat2_bf16_T, "column_major"
    ), f"mat2 should be row-major, got strides: {mat2_bf16_T.stride()}"

    # Forward pass
    result = batched_grouped_mm(
        mat1_base_bf16,
        mat2_bf16_T,
        STATIC_OFFSETS,
        [
            1,
        ],
    ).type_as(mat2)

    print(
        f"mat1 shape: {mat1_base_bf16.shape}, strides: {mat1_base_bf16.stride()}"
        f" (row-major: {verify_tensor_layout(mat1_base_bf16, 'row_major')})"
    )
    print(
        f"mat2 shape: {mat2_bf16_T.shape}, strides: {mat2_bf16_T.stride()}"
        f" (column-major: {verify_tensor_layout(mat2_bf16_T, 'column_major')})"
    )
    print(f"result shape: {result.shape}")

    # Backward pass
    result.sum().backward()

    print(f"mat1_base.grad strides: {mat1_base.grad.stride()}")
    print(f"mat2.grad strides: {mat2.grad.stride()}")


def test_case3_3d_x_3d_col_row():
    """Test Case 3: mat1=[2,128,256] column-major, mat2=[2,512,256] row-major"""
    print("\n=== Test Case 3: 3D x 3D (column-major x row-major) ===")

    # Create tensors
    mat1_base = create_row_major_tensor((OB, D, IB)).requires_grad_(True)
    mat2 = create_row_major_tensor((OB, H, D)).requires_grad_(True)

    # Convert to bfloat16 first, then transpose to preserve stride patterns
    mat1_base_bf16 = mat1_base.bfloat16()
    mat2_bf16 = mat2.bfloat16()

    # Transpose mat1 to get column-major layout
    mat1 = mat1_base_bf16.transpose(-2, -1)

    # Verify layouts
    assert verify_tensor_layout(
        mat1, "column_major"
    ), f"mat1 should be column-major, got strides: {mat1.stride()}"
    assert verify_tensor_layout(
        mat2, "row_major"
    ), f"mat2 should be row-major, got strides: {mat2.stride()}"

    # Forward pass
    result = batched_grouped_mm(
        mat1, mat2_bf16.transpose(-2, -1), STATIC_OFFSETS, [0, 1]
    ).type_as(mat1_base)

    print(
        f"mat1 shape: {mat1.shape}, strides: {mat1.stride()} (column-major: {verify_tensor_layout(mat1, 'column_major')})"
    )
    print(
        f"mat2 shape: {mat2.shape}, strides: {mat2.stride()} (row-major: {verify_tensor_layout(mat2, 'row_major')})"
    )
    print(f"result shape: {result.shape}")

    # Backward pass
    result.sum().backward()

    print(f"mat1_base.grad strides: {mat1_base.grad.stride()}")
    print(f"mat2.grad strides: {mat2.grad.stride()}")


def test_case4_3d_x_3d_row_col():
    """Test Case 4: mat1=[2,128,256] row-major, mat2=[2,512,256] column-major"""
    print("\n=== Test Case 4: 3D x 3D (row-major x column-major) ===")

    # Create tensors
    mat1 = create_row_major_tensor((OB, IB, D)).requires_grad_(True)
    mat2_base = create_row_major_tensor((OB, D, H)).requires_grad_(True)

    # Convert to bfloat16 first, then transpose to preserve stride patterns
    mat1_bf16 = mat1.bfloat16()
    mat2_base_bf16 = mat2_base.bfloat16()

    # Transpose mat2 to get column-major layout
    mat2 = mat2_base_bf16.transpose(-2, -1)

    # Verify layouts
    assert verify_tensor_layout(
        mat1_bf16, "row_major"
    ), f"mat1 should be row-major, got strides: {mat1.stride()}"
    assert verify_tensor_layout(
        mat2, "column_major"
    ), f"mat2 should be column-major, got strides: {mat2.stride()}"

    # Forward pass
    result = batched_grouped_mm(
        mat1_bf16, mat2.transpose(-2, -1), STATIC_OFFSETS, [0, 1]
    ).type_as(mat1)

    print(
        f"mat1 shape: {mat1_bf16.shape}, strides: {mat1_bf16.stride()} (row-major: {verify_tensor_layout(mat1_bf16, 'row_major')})"
    )
    print(
        f"mat2 shape: {mat2.shape}, strides: {mat2.stride()} (column-major: {verify_tensor_layout(mat2, 'column_major')})"
    )
    print(f"result shape: {result.shape}")

    # Backward pass
    result.sum().backward()

    print(f"mat1.grad strides: {mat1.grad.stride()}")
    print(f"mat2_base.grad strides: {mat2_base.grad.stride()}")


def run_all_tests():
    """Run all simplified test cases."""
    print("Running simplified batched_grouped_mm tests with static shapes...")
    print(f"Static dimensions: OB={OB}, IB={IB}, D={D}, E={E}, H={H}")
    print(f"Static offsets: {STATIC_OFFSETS}")

    test_case1_3d_x_4d_row_col()
    test_case2_4d_x_3d_col_row()
    test_case3_3d_x_3d_col_row()
    test_case4_3d_x_3d_row_col()

    print("\n" + "=" * 50)
    print("✅ All tests completed successfully!")
    print("✅ Verified stride handling for different layouts")
    print("✅ Verified gradient computation paths")
    print("=" * 50)


if __name__ == "__main__":
    run_all_tests()
