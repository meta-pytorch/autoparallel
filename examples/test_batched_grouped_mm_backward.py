import torch
import torch.autograd


@torch.library.custom_op("autoparallel::batched_grouped_mm", mutates_args=())
def batched_grouped_mm(
    mat1: torch.Tensor, mat2: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    # TODO (@sanketpurandare): When mat1 is the experts tensor and mat2 is the batched output
    # that will support out = grouped_mm(mat2.T, mat1.T, off).T
    assert offs.ndim == 2
    assert mat1.ndim == 3
    assert mat2.ndim == 3, f"{mat2.shape}"
    ob1, num_experts1 = offs.shape
    ob2, _, dim1 = mat1.shape
    num_experts2, dim2, _ = mat2.shape
    assert ob1 == ob2, f"{mat1.shape} vs {offs.shape}"
    assert dim1 == dim2, f"{mat1.shape} vs {mat2.shape}"
    assert num_experts1 == num_experts2, f"{mat2.shape} vs {offs.shape}"
    res = []
    if (mat2.stride(-2) == 1 and mat2.stride(-1) == mat2.size(-2)) and (
        mat1.stride(-2) == 1 and mat1.stride(-1) == mat1.size(-2)
    ):
        print("Path 1 batched grouped mm")
        # if input was column-major, return output as column-order for efficiency
        for m1, off in zip(mat1.transpose(-2, -1), offs):
            res.append(
                torch._grouped_mm(mat2.transpose(-2, -1), m1, off).transpose(-2, -1)
            )
    else:
        print("Path 2 batched grouped mm")
        for m1, off in zip(mat1, offs):
            res.append(torch._grouped_mm(m1, mat2, off))
    return torch.stack(res, 0)


def setup_context_batched_grouped_mm(ctx, inputs, output):
    mat1, mat2, offs = inputs
    ctx.save_for_backward(mat1, mat2, offs)


def _backward_grouped_mm_mat1(
    mat1: torch.Tensor, mat2: torch.Tensor, grad: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    if mat1.stride(-2) == 1 and mat1.stride(-1) == mat1.size(-2):
        print("Path 1 mat1 grad")
        # if input was column-major, return grad as column-order for efficiency
        res = []
        for g, off in zip(grad.transpose(-2, -1), offs):
            res.append(torch._grouped_mm(mat2, g, off).transpose(-2, -1))
        grad_mat1 = torch.stack(res, 0)
    else:
        print("Path 2 mat1 grad")
        grad_mat1 = batched_grouped_mm(grad, mat2.transpose(-2, -1), offs)
    return grad_mat1


def _backward_grouped_mm_mat2(
    mat1: torch.Tensor, mat2: torch.Tensor, grad: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    partial_grad_mat2 = []
    if mat2.stride(-2) == 1 and mat2.stride(-1) == mat2.size(-2):
        print("Path 1 mat2 grad")
        # if input was column-major, return grad as column-order for efficiency
        for g, m1, off in zip(grad, mat1, offs):
            partial_grad_mat2.append(
                torch._grouped_mm(g.transpose(-2, -1), m1, off).transpose(-2, -1)
            )
    else:
        print("Path 2 mat2 grad")
        for g, m1, off in zip(grad, mat1, offs):
            partial_grad_mat2.append(torch._grouped_mm(m1.transpose(-2, -1), g, off))
    grad_mat2 = torch.stack(partial_grad_mat2, 0).sum(0)
    return grad_mat2


def backward_batched_grouped_mm(
    ctx, grad: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, None]:
    mat1, mat2, offs = ctx.saved_tensors
    grad_mat1 = _backward_grouped_mm_mat1(mat1, mat2, grad, offs)
    grad_mat2 = _backward_grouped_mm_mat2(mat1, mat2, grad, offs)
    return grad_mat1, grad_mat2, None


torch.library.register_autograd(
    "autoparallel::batched_grouped_mm",
    backward_batched_grouped_mm,
    setup_context=setup_context_batched_grouped_mm,
)


def test_simple_functional_correctness():
    """Test basic functional correctness without gradcheck - verify implementation works."""
    print("=== Simple functional correctness test ===")
    torch.manual_seed(42)
    device = "cuda"

    batch_size = 2
    seq_len = 64
    num_experts = 2
    dim = 128
    hidden_dim = 256

    print("Testing functional correctness")
    mat1 = torch.rand(batch_size, seq_len, dim, requires_grad=True, device=device)
    mat2 = torch.rand(num_experts, dim, hidden_dim, requires_grad=True, device=device)
    offsets = torch.tensor([[32, 64], [16, 64]], dtype=torch.int32, device=device)

    # Forward pass
    output1 = torch.ops.autoparallel.batched_grouped_mm(
        mat1.bfloat16(), mat2.bfloat16(), offsets
    ).type_as(mat1)

    # Simple backward pass
    loss1 = output1.sum()
    loss1.backward()
    grad1 = mat1.grad
    grad2 = mat2.grad

    assert mat1.grad is not None, "Case 1: mat1 gradient is None"
    assert mat2.grad is not None, "Case 1: mat2 gradient is None"
    assert not torch.isnan(mat1.grad).any(), "Case 1: mat1 gradient has NaN"
    assert not torch.isnan(mat2.grad).any(), "Case 1: mat2 gradient has NaN"

    mat1.grad = None
    mat2.grad = None
    res = []
    for m1, off in zip(mat1, offsets):
        res.append(torch._grouped_mm(m1.bfloat16(), mat2.bfloat16(), off))
    output2 = torch.stack(res, 0).type_as(mat1)
    loss2 = output2.sum()
    loss2.backward()
    grad1_2 = mat1.grad
    grad2_2 = mat2.grad

    assert torch.allclose(
        grad1, grad1_2, rtol=1e-2, atol=1e-2
    ), "Case 1: mat1 gradient mismatch"
    assert torch.allclose(
        grad2, grad2_2, rtol=1e-2, atol=1e-2
    ), "Case 1: mat2 gradient mismatch"
    mat1.grad = None
    mat2.grad = None
    print("✓ Simple functional correctness test PASSED")


def test_transpose_functional_correctness():
    """Test transpose functional correctness without gradcheck - verify implementation works."""
    print("=== Transpose functional correctness test ===")
    torch.manual_seed(42)
    device = "cuda"

    batch_size = 2
    seq_len = 64
    num_experts = 2
    dim = 128
    hidden_dim = 256

    print("Testing functional correctness")
    mat1 = torch.rand(batch_size, dim, seq_len, requires_grad=True, device=device)
    mat2 = torch.rand(num_experts, hidden_dim, dim, requires_grad=True, device=device)
    offsets = torch.tensor([[32, 64], [16, 64]], dtype=torch.int32, device=device)

    # Forward pass
    output1 = torch.ops.autoparallel.batched_grouped_mm(
        mat1.transpose(-2, -1).bfloat16(),
        mat2.transpose(-2, -1).bfloat16(),
        offsets,
    ).type_as(mat1)

    # Simple backward pass
    loss1 = output1.sum()
    loss1.backward()
    grad1 = mat1.grad
    grad2 = mat2.grad

    assert mat1.grad is not None, "Case 1: mat1 gradient is None"
    assert mat2.grad is not None, "Case 1: mat2 gradient is None"
    assert not torch.isnan(mat1.grad).any(), "Case 1: mat1 gradient has NaN"
    assert not torch.isnan(mat2.grad).any(), "Case 1: mat2 gradient has NaN"

    mat1.grad = None
    mat2.grad = None
    res = []
    for m1, off in zip(mat1, offsets):
        res.append(torch._grouped_mm(mat2.bfloat16(), m1.bfloat16(), off))
    output2 = torch.stack(res, 0).type_as(mat1)
    loss2 = output2.sum()
    loss2.backward()
    grad1_2 = mat1.grad
    grad2_2 = mat2.grad

    assert torch.allclose(
        grad1, grad1_2, rtol=1e-2, atol=1e-2
    ), "Case 1: mat1 gradient mismatch"
    assert torch.allclose(
        grad2, grad2_2, rtol=1e-2, atol=1e-2
    ), "Case 1: mat2 gradient mismatch"
    mat1.grad = None
    mat2.grad = None

    print("✓ Transpose functional correctness test PASSED")


if __name__ == "__main__":
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("⚠️  CUDA is not available. Skipping tests.")
        exit(1)

    print(f"🚀 Running tests on CUDA device: {torch.cuda.get_device_name()}")

    # Run all tests following the pattern from test case 1
    test_simple_functional_correctness()
    test_transpose_functional_correctness()

    print("\n🎉 All CUDA tests completed successfully!")
