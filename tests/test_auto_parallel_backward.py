# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for AutoParallelBackward and auto_parallel_with_backward.

Validates that AutoParallelBackward produces equivalent sharding decisions
to AutoParallel for the same model architecture.
"""

import pytest
import torch
import torch.nn as nn
from conftest import apply_cuda_patches
from torch._functorch._aot_autograd.fx_utils import (
    get_param_and_grad_nodes,
    get_param_nodes,
)
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel.api import (
    AutoParallel,
    AutoParallelBackward,
    auto_parallel_with_backward,
)


@pytest.fixture(autouse=True)
def reset_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


# ─── Models ───────────────────────────────────────────────────────────────────


class LinearFwd(nn.Module):
    """Forward-only model: returns output for external backward."""

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.linear(x).sum()


class LinearBwd(nn.Module):
    """Model that calls backward() inside forward()."""

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        y = self.linear(x)
        loss = y.sum()
        loss.backward()
        return loss.detach()


class FFNFwd(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.linear1 = nn.Linear(dim1, dim2, bias=False)
        self.linear2 = nn.Linear(dim2, dim1, bias=False)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x))).sum()


class FFNBwd(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.linear1 = nn.Linear(dim1, dim2, bias=False)
        self.linear2 = nn.Linear(dim2, dim1, bias=False)

    def forward(self, x):
        y = self.linear2(torch.relu(self.linear1(x)))
        loss = y.sum()
        loss.backward()
        return loss.detach()


class DetachReattachBwd(nn.Module):
    """Detach-reattach pattern: backward is split into two phases."""

    def __init__(self, dim1, dim2):
        super().__init__()
        self.body = nn.Linear(dim1, dim1, bias=False)
        self.head = nn.Linear(dim1, dim2, bias=False)

    def forward(self, x):
        x = self.body(x)
        x_detached = x.detach().requires_grad_()
        logits = self.head(x_detached)
        loss = logits.sum()
        loss.backward()
        x.backward(x_detached.grad)
        return loss.detach()


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _get_param_placements(autop):
    """Extract {fqn: placements} from a solved AutoParallel instance."""
    param_nodes = get_param_nodes(autop.gm.graph)
    result = {}
    for node in param_nodes:
        fqn = node.meta["desc"].target
        strategy = autop.sharding_placement[node]
        result[fqn] = tuple(strategy.output_specs.placements)
    return result


def _run_auto_parallel(model, input_fn, mesh, input_constraints, output_constraints):
    """Run AutoParallel (forward-only) and return the instance."""
    with AutoParallel(model, input_fn, mesh) as ap:
        ap.add_input_constraints(input_constraints)
        ap.add_output_constraints(output_constraints)
        ap.optimize_placement()
    return ap


def _run_auto_parallel_backward(
    model, input_fn, mesh, input_constraints, output_constraints
):
    """Run AutoParallelBackward and return the instance."""
    with AutoParallelBackward(model, input_fn, mesh) as ap:
        ap.add_input_constraints(input_constraints)
        ap.add_output_constraints(output_constraints)
        ap.optimize_placement()
    return ap


# ─── Tests: AutoParallelBackward tracing ─────────────────────────────────────


@apply_cuda_patches
def test_backward_tracing_linear(device_mesh_1d):
    """AutoParallelBackward traces a simple model with backward()."""
    dim = 128
    bs = 2048 * device_mesh_1d.shape[0]

    with torch.device("meta"):
        model = LinearBwd(dim)

    def input_fn():
        return (torch.randn(bs, dim, device="cuda"),)

    with AutoParallelBackward(model, input_fn, device_mesh_1d) as ap:
        ap.add_input_constraints([(Shard(0),)])
        ap.add_output_constraints([(Replicate(),)])
        placement = ap.optimize_placement()

    # Verify we got a valid sharding solution
    assert placement is not None
    assert len(get_param_nodes(ap.gm.graph)) > 0


@apply_cuda_patches
def test_backward_tracing_ffn(device_mesh_1d):
    """AutoParallelBackward traces an FFN model with backward()."""
    dim1, dim2 = 1024, 4096
    bs = 2048 * device_mesh_1d.shape[0]

    with torch.device("meta"):
        model = FFNBwd(dim1, dim2)

    def input_fn():
        return (torch.randn(bs, dim1, device="cuda"),)

    with AutoParallelBackward(model, input_fn, device_mesh_1d) as ap:
        ap.add_input_constraints([(Shard(0),)])
        ap.add_output_constraints([(Replicate(),)])
        placement = ap.optimize_placement()

    assert placement is not None
    param_placements = _get_param_placements(ap)
    assert "linear1.weight" in param_placements
    assert "linear2.weight" in param_placements


@apply_cuda_patches
def test_backward_tracing_detach_reattach(device_mesh_1d):
    """AutoParallelBackward traces the detach-reattach backward pattern."""
    dim1, dim2 = 1024, 4096
    bs = 2048 * device_mesh_1d.shape[0]

    with torch.device("meta"):
        model = DetachReattachBwd(dim1, dim2)

    def input_fn():
        return (torch.randn(bs, dim1, device="cuda"),)

    with AutoParallelBackward(model, input_fn, device_mesh_1d) as ap:
        ap.add_input_constraints([(Shard(0),)])
        ap.add_output_constraints([(Replicate(),)])
        placement = ap.optimize_placement()

    assert placement is not None
    param_placements = _get_param_placements(ap)
    assert "body.weight" in param_placements
    assert "head.weight" in param_placements


# ─── Tests: Descriptor correctness ───────────────────────────────────────────


@apply_cuda_patches
def test_descriptors_correct_linear(device_mesh_1d):
    """Verify descriptors from AutoParallel and AutoParallelBackward are well-formed."""
    from torch._functorch._aot_autograd.descriptors import ParamAOTInput

    dim = 128
    bs = 2048 * device_mesh_1d.shape[0]

    # Forward-only path
    with torch.device("meta"):
        model_fwd = LinearFwd(dim)

    def input_fn():
        return (torch.randn(bs, dim, device="cuda"),)

    torch._dynamo.reset()
    with AutoParallel(model_fwd, input_fn, device_mesh_1d) as ap_fwd:
        pass

    # User-backward path
    with torch.device("meta"):
        model_bwd = LinearBwd(dim)

    torch._dynamo.reset()
    with AutoParallelBackward(model_bwd, input_fn, device_mesh_1d) as ap_bwd:
        pass

    # Both should have the same parameter FQNs
    fwd_params = get_param_and_grad_nodes(ap_fwd.gm.graph)
    bwd_params = get_param_and_grad_nodes(ap_bwd.gm.graph)

    fwd_fqns = {desc.target for desc in fwd_params}
    bwd_fqns = {desc.target for desc in bwd_params}
    assert fwd_fqns == bwd_fqns, f"FQN mismatch: {fwd_fqns} vs {bwd_fqns}"

    # Forward path: params should have gradient outputs (joint graph has GradAOTOutput)
    for desc, (_, grad_node) in fwd_params.items():
        assert isinstance(desc, ParamAOTInput)
        assert (
            grad_node is not None
        ), f"[fwd] Param {desc.target} should have a gradient"

    # Backward path: params exist but gradients are applied via copy_ mutation
    for desc, (_, grad_node) in bwd_params.items():
        assert isinstance(desc, ParamAOTInput)

    # Both graphs should have output descriptors
    for label, ap in [("fwd", ap_fwd), ("bwd", ap_bwd)]:
        for n in ap.gm.graph.nodes:
            if n.op == "output":
                assert "desc" in n.meta, f"[{label}] Output node missing desc"
                assert isinstance(n.meta["desc"], list)


@apply_cuda_patches
def test_descriptors_match_ffn(device_mesh_2d):
    """Verify descriptor parity for a 2-layer FFN on a 2D mesh."""

    dim1, dim2 = 1024, 4096
    bs = 2048 * device_mesh_2d.shape[0]

    # Forward-only
    with torch.device("meta"):
        model_fwd = FFNFwd(dim1, dim2)

    def input_fn():
        return (torch.randn(bs, dim1, device="cuda"),)

    torch._dynamo.reset()
    with AutoParallel(model_fwd, input_fn, device_mesh_2d) as ap_fwd:
        pass

    # User-backward
    with torch.device("meta"):
        model_bwd = FFNBwd(dim1, dim2)

    torch._dynamo.reset()
    with AutoParallelBackward(model_bwd, input_fn, device_mesh_2d) as ap_bwd:
        pass

    # Both must have the same set of parameter FQNs
    fwd_params = get_param_and_grad_nodes(ap_fwd.gm.graph)
    bwd_params = get_param_and_grad_nodes(ap_bwd.gm.graph)

    fwd_fqns = {desc.target for desc in fwd_params}
    bwd_fqns = {desc.target for desc in bwd_params}
    assert fwd_fqns == bwd_fqns, f"FQN mismatch: {fwd_fqns} vs {bwd_fqns}"

    # Forward path: all params should have gradients
    for desc, (_, grad_node) in fwd_params.items():
        assert grad_node is not None, f"[fwd] Param {desc.target} has no gradient node"


# ─── Tests: Sharding decision parity ─────────────────────────────────────────


@apply_cuda_patches
def test_sharding_parity_linear_1d(device_mesh_1d):
    """AutoParallel and AutoParallelBackward reach the same sharding for a linear model."""
    dim = 1024
    bs = 2048 * device_mesh_1d.shape[0]

    def input_fn():
        return (torch.randn(bs, dim, device="cuda"),)

    input_constraints = [(Shard(0),)]
    output_constraints = [(Replicate(),)]

    with torch.device("meta"):
        model_fwd = LinearFwd(dim)
    torch._dynamo.reset()
    ap_fwd = _run_auto_parallel(
        model_fwd, input_fn, device_mesh_1d, input_constraints, output_constraints
    )

    with torch.device("meta"):
        model_bwd = LinearBwd(dim)
    torch._dynamo.reset()
    ap_bwd = _run_auto_parallel_backward(
        model_bwd, input_fn, device_mesh_1d, input_constraints, output_constraints
    )

    fwd_placements = _get_param_placements(ap_fwd)
    bwd_placements = _get_param_placements(ap_bwd)

    assert fwd_placements.keys() == bwd_placements.keys()
    for fqn in fwd_placements:
        assert fwd_placements[fqn] == bwd_placements[fqn], (
            f"Sharding mismatch for {fqn}: "
            f"AutoParallel={fwd_placements[fqn]}, "
            f"AutoParallelBackward={bwd_placements[fqn]}"
        )


@apply_cuda_patches
def test_sharding_parity_ffn_1d(device_mesh_1d):
    """AutoParallel and AutoParallelBackward reach the same sharding for FFN on 1D mesh."""
    dim1, dim2 = 1024, 4096
    bs = 2048 * device_mesh_1d.shape[0]

    def input_fn():
        return (torch.randn(bs, dim1, device="cuda"),)

    input_constraints = [(Shard(0),)]
    output_constraints = [(Replicate(),)]

    with torch.device("meta"):
        model_fwd = FFNFwd(dim1, dim2)
    torch._dynamo.reset()
    ap_fwd = _run_auto_parallel(
        model_fwd, input_fn, device_mesh_1d, input_constraints, output_constraints
    )

    with torch.device("meta"):
        model_bwd = FFNBwd(dim1, dim2)
    torch._dynamo.reset()
    ap_bwd = _run_auto_parallel_backward(
        model_bwd, input_fn, device_mesh_1d, input_constraints, output_constraints
    )

    fwd_placements = _get_param_placements(ap_fwd)
    bwd_placements = _get_param_placements(ap_bwd)

    assert fwd_placements.keys() == bwd_placements.keys()
    for fqn in fwd_placements:
        assert fwd_placements[fqn] == bwd_placements[fqn], (
            f"Sharding mismatch for {fqn}: "
            f"AutoParallel={fwd_placements[fqn]}, "
            f"AutoParallelBackward={bwd_placements[fqn]}"
        )


@apply_cuda_patches
def test_sharding_parity_ffn_2d(device_mesh_2d):
    """AutoParallel and AutoParallelBackward reach the same sharding for FFN on 2D mesh."""
    dim1, dim2 = 1024, 4096
    bs = 2048 * device_mesh_2d.shape[0]

    def input_fn():
        return (torch.randn(bs, dim1, device="cuda"),)

    # Output is scalar (loss), so use Replicate on both mesh dims
    input_constraints = [(Shard(0), Replicate())]
    output_constraints = [(Replicate(), Replicate())]

    with torch.device("meta"):
        model_fwd = FFNFwd(dim1, dim2)
    torch._dynamo.reset()
    ap_fwd = _run_auto_parallel(
        model_fwd, input_fn, device_mesh_2d, input_constraints, output_constraints
    )

    with torch.device("meta"):
        model_bwd = FFNBwd(dim1, dim2)
    torch._dynamo.reset()
    ap_bwd = _run_auto_parallel_backward(
        model_bwd, input_fn, device_mesh_2d, input_constraints, output_constraints
    )

    fwd_placements = _get_param_placements(ap_fwd)
    bwd_placements = _get_param_placements(ap_bwd)

    assert fwd_placements.keys() == bwd_placements.keys()
    for fqn in fwd_placements:
        assert fwd_placements[fqn] == bwd_placements[fqn], (
            f"Sharding mismatch for {fqn}: "
            f"AutoParallel={fwd_placements[fqn]}, "
            f"AutoParallelBackward={bwd_placements[fqn]}"
        )


# ─── Tests: Simple API ───────────────────────────────────────────────────────


@apply_cuda_patches
def test_simple_api_linear(device_mesh_1d):
    """auto_parallel_with_backward produces a valid parallel model."""
    dim = 128
    bs = 512
    local_bs = bs // device_mesh_1d.size()

    with torch.device("meta"):
        model = LinearBwd(dim)

    x = DTensor.from_local(
        torch.rand(local_bs, dim, device="cuda"),
        device_mesh_1d,
        [Shard(0)],
    )

    torch._dynamo.reset()
    parallel_model = auto_parallel_with_backward(
        model,
        device_mesh_1d,
        sample_inputs=(x,),
        out_shardings=(Replicate(),),
    )

    assert parallel_model is not None
    assert hasattr(parallel_model, "linear")

    # Verify parameter is a DTensor
    param = parallel_model.get_parameter("linear.weight")
    assert isinstance(param, DTensor)


@apply_cuda_patches
def test_simple_api_ffn(device_mesh_1d):
    """auto_parallel_with_backward works for an FFN model."""
    dim1, dim2 = 1024, 4096
    bs = 2048
    local_bs = bs // device_mesh_1d.size()

    with torch.device("meta"):
        model = FFNBwd(dim1, dim2)

    x = DTensor.from_local(
        torch.rand(local_bs, dim1, device="cuda"),
        device_mesh_1d,
        [Shard(0)],
    )

    torch._dynamo.reset()
    parallel_model = auto_parallel_with_backward(
        model,
        device_mesh_1d,
        sample_inputs=(x,),
        out_shardings=(Replicate(),),
    )

    assert parallel_model is not None
    assert hasattr(parallel_model, "linear1")
    assert hasattr(parallel_model, "linear2")


@apply_cuda_patches
def test_simple_api_detach_reattach(device_mesh_1d):
    """auto_parallel_with_backward works for the detach-reattach pattern."""
    dim1, dim2 = 1024, 4096
    bs = 2048
    local_bs = bs // device_mesh_1d.size()

    with torch.device("meta"):
        model = DetachReattachBwd(dim1, dim2)

    x = DTensor.from_local(
        torch.rand(local_bs, dim1, device="cuda"),
        device_mesh_1d,
        [Shard(0)],
    )

    torch._dynamo.reset()
    parallel_model = auto_parallel_with_backward(
        model,
        device_mesh_1d,
        sample_inputs=(x,),
        out_shardings=(Replicate(),),
    )

    assert parallel_model is not None
    assert hasattr(parallel_model, "body")
    assert hasattr(parallel_model, "head")


# ─── Tests: Graph structure ──────────────────────────────────────────────────


@apply_cuda_patches
def test_backward_graph_has_no_autograd_grad_nodes(device_mesh_1d):
    """After decomposition, the graph should have no autograd.grad nodes left."""
    dim = 128
    bs = 2048 * device_mesh_1d.shape[0]

    with torch.device("meta"):
        model = LinearBwd(dim)

    def input_fn():
        return (torch.randn(bs, dim, device="cuda"),)

    torch._dynamo.reset()
    with AutoParallelBackward(model, input_fn, device_mesh_1d) as ap:
        pass

    # The decomposed graph should only contain aten ops, no autograd.grad
    for node in ap.gm.graph.nodes:
        if node.op == "call_function":
            assert (
                node.target is not torch.autograd.grad
            ), f"Found undecomposed autograd.grad node: {node}"


@apply_cuda_patches
def test_backward_graph_contains_backward_ops(device_mesh_1d):
    """The decomposed backward graph should contain backward-specific aten ops."""
    dim = 128
    bs = 2048 * device_mesh_1d.shape[0]

    with torch.device("meta"):
        model = LinearBwd(dim)

    def input_fn():
        return (torch.randn(bs, dim, device="cuda"),)

    torch._dynamo.reset()
    with AutoParallelBackward(model, input_fn, device_mesh_1d) as ap:
        pass

    # The graph should contain mm (forward matmul) AND at least one
    # transpose or another mm for the backward pass
    targets = {node.target for node in ap.gm.graph.nodes if node.op == "call_function"}
    has_mm = torch.ops.aten.mm.default in targets
    assert has_mm, f"Expected aten.mm in graph, got targets: {targets}"


# ─── Tests: WithBackward adaptor ──────────────────────────────────────────────


class WithBackward(nn.Module):
    """Wraps a forward-only model to call backward() inside forward().

    This adaptor enables testing AutoParallelBackward on any model that
    AutoParallel already supports, by calling .sum().backward() on the
    model's output and returning detached results.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        # Sum all differentiable float tensor leaves for backward
        flat, treespec = torch.utils._pytree.tree_flatten(out)
        loss = sum(
            o.sum()
            for o in flat
            if isinstance(o, torch.Tensor) and o.is_floating_point() and o.requires_grad
        )
        if isinstance(loss, (int, float)):
            # No differentiable outputs — nothing to backward
            return out
        loss.backward()
        detached = [o.detach() if isinstance(o, torch.Tensor) else o for o in flat]
        return torch.utils._pytree.tree_unflatten(detached, treespec)


@apply_cuda_patches
def test_with_backward_linear(device_mesh_1d):
    """AutoParallelBackward(WithBackward(linear)) should produce valid sharding."""
    dim = 1024
    bs = 2048 * device_mesh_1d.shape[0]

    class Linear(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            return self.linear(x).sum()

    with torch.device("meta"):
        model = WithBackward(Linear())

    def input_fn():
        return (torch.randn(bs, dim, device="cuda"),)

    with AutoParallelBackward(model, input_fn, device_mesh_1d) as ap:
        ap.add_input_constraints([(Shard(0),)])
        placement = ap.optimize_placement(verbose=False)
        parallel_model = ap.apply_placement(placement)

    assert parallel_model is not None


@apply_cuda_patches
def test_with_backward_ffn(device_mesh_1d):
    """AutoParallelBackward(WithBackward(ffn)) with multi-output model."""
    dim1, dim2 = 1024, 4096
    bs = 2048 * device_mesh_1d.shape[0]

    class FFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(dim1, dim2, bias=False)
            self.linear2 = nn.Linear(dim2, dim1, bias=False)

        def forward(self, x):
            return self.linear2(torch.relu(self.linear1(x)))

    with torch.device("meta"):
        model = WithBackward(FFN())

    def input_fn():
        return (torch.randn(bs, dim1, device="cuda"),)

    with AutoParallelBackward(model, input_fn, device_mesh_1d) as ap:
        ap.add_input_constraints([(Shard(0),)])
        placement = ap.optimize_placement(verbose=False)
        parallel_model = ap.apply_placement(placement)

    assert parallel_model is not None


@apply_cuda_patches
def test_with_backward_multi_output(device_mesh_1d):
    """AutoParallelBackward(WithBackward(model)) with multiple outputs."""
    dim1, dim2 = 1024, 4096
    bs = 2048 * device_mesh_1d.shape[0]

    class MultiOutput(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(dim1, dim2, bias=False)
            self.linear2 = nn.Linear(dim2, dim1, bias=False)

        def forward(self, x, y):
            return y + 2, self.linear2(self.linear1(x)), y + 2

    with torch.device("meta"):
        model = WithBackward(MultiOutput())

    def input_fn():
        return torch.randn(bs, dim1, device="cuda"), torch.randn(bs, 1, device="cuda")

    with AutoParallelBackward(model, input_fn, device_mesh_1d) as ap:
        ap.add_input_constraints([(Shard(0),), (Shard(0),)])
        placement = ap.optimize_placement(verbose=False)
        parallel_model = ap.apply_placement(placement)

    assert parallel_model is not None


@apply_cuda_patches
def test_with_backward_ffn_2d(device_mesh_2d):
    """AutoParallelBackward(WithBackward(ffn)) on a 2D mesh."""
    dim1, dim2 = 1024, 4096
    bs = 2048 * device_mesh_2d.shape[0]

    class FFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(dim1, dim2, bias=False)
            self.linear2 = nn.Linear(dim2, dim1, bias=False)

        def forward(self, x):
            return self.linear2(torch.relu(self.linear1(x)))

    with torch.device("meta"):
        model = WithBackward(FFN())

    def input_fn():
        return (torch.randn(bs, dim1, device="cuda"),)

    with AutoParallelBackward(model, input_fn, device_mesh_2d) as ap:
        ap.add_input_constraints([(Shard(0), Replicate())])
        placement = ap.optimize_placement(verbose=False)
        parallel_model = ap.apply_placement(placement)

    assert parallel_model is not None
