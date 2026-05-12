# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
from conftest import apply_cuda_patches
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Shard

from autoparallel.api import AutoParallel


class SimpleLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.linear(x)


class StackedLinear(nn.Module):
    """Multiple identical linear layers for testing repeated_subgraphs."""

    def __init__(self, dim, n_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(dim, dim, bias=False) for _ in range(n_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _run_autop(mesh, model_fn, input_fn, mp_policy, repeated_subgraphs=False):
    """Run AutoParallel and return the solution and optimizer."""
    with torch.device("meta"):
        model = model_fn()

    with AutoParallel(
        model, input_fn, mesh, mp_policy, repeated_subgraphs=repeated_subgraphs
    ) as autop:
        autop.add_input_constraints([(Shard(0),) * mesh.ndim])
        autop.add_output_constraints([(Shard(0),) * mesh.ndim])
        sharding_placement = autop.optimize_placement(verbose=False)

    return sharding_placement, autop


def _assert_no_pre_cast_redistribution(
    sharding_placement, autop, require_linked_validation=False
):
    """Assert that no chosen pre-cast decision var has comm_cost > 0.

    This directly validates the solver constraint: no redistribution
    should occur on unary-chain edges before the dtype_cast node.
    """
    from torch._functorch._aot_autograd.fx_utils import get_param_and_grad_nodes

    opt = autop.sharding_optimizer
    validated_pre_cast_keys = 0
    validated_linked_keys = 0

    for param, grad in get_param_and_grad_nodes(opt.graph).values():
        if grad is None:
            continue

        # Build the pre-cast node set (same logic as the constraint)
        chain = [grad]
        n = grad
        while len(n.all_input_nodes) == 1:
            parent = n.all_input_nodes[0]
            if len(parent.all_input_nodes) != 1:
                break
            chain.append(parent)
            n = parent

        cast_idx = None
        for i, node in enumerate(chain):
            if node.target == torch.ops.autoparallel.dtype_cast.default:
                cast_idx = i
                break

        if cast_idx is None:
            continue

        pre_cast_node_idxs = set()
        for node in chain[cast_idx:]:
            if node in opt.node_map:
                pre_cast_node_idxs.add(opt.node_map[node])

        # Check that no chosen decision var on a pre-cast node has comm_cost > 0
        for key in opt.selected_keys:
            node_idx, argi, out_idx, inp_idx = key
            if node_idx not in pre_cast_node_idxs:
                continue
            dv = opt._resolve_decision_var(key)
            validated_pre_cast_keys += 1
            if key in opt.cluster_links:
                validated_linked_keys += 1
            assert dv.comm_cost == 0, (
                f"Pre-cast node {opt.nodes[node_idx].name} has chosen decision var "
                f"with comm_cost={dv.comm_cost} > 0 (strategy {dv.strategy}). "
                f"Redistribution should not happen before the dtype_cast."
            )

        # Also check dtype_cast output has Partial (indirect but readable check)
        dtype_cast_node = chain[cast_idx]
        spec = sharding_placement.get(dtype_cast_node)
        if spec is not None:
            assert any(p.is_partial() for p in spec.output_specs.placements), (
                f"dtype_cast {dtype_cast_node.name} should have Partial output "
                f"(no pre-cast reduction), but got {spec.output_specs.placements}"
            )

    assert validated_pre_cast_keys > 0, "Expected to validate at least one pre-cast key"
    if require_linked_validation:
        assert (
            validated_linked_keys > 0
        ), "Expected to validate at least one cluster-linked pre-cast key"


@apply_cuda_patches
def test_grad_reduce_dtype_f32_reduces_after_cast(device_mesh_1d):
    """With reduce_dtype=f32, gradient reductions should happen after dtype_cast."""
    dim = 1024
    mesh = device_mesh_1d

    def model_fn():
        return SimpleLinear(dim)

    def input_fn():
        return torch.randn(2048 * mesh.shape[0], dim, device="cuda")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    sharding_placement, autop = _run_autop(mesh, model_fn, input_fn, mp_policy)
    _assert_no_pre_cast_redistribution(sharding_placement, autop)


@apply_cuda_patches
def test_grad_reduce_dtype_f32_with_repeated_subgraphs(device_mesh_1d):
    """Same as above but with repeated_subgraphs=True (graph clustering).

    Verifies the constraint works correctly when cluster-linked nodes
    copy strategies from representative nodes.
    """
    dim = 1024
    mesh = device_mesh_1d

    def model_fn():
        return StackedLinear(dim, n_layers=4)

    def input_fn():
        return torch.randn(2048 * mesh.shape[0], dim, device="cuda")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    sharding_placement, autop = _run_autop(
        mesh, model_fn, input_fn, mp_policy, repeated_subgraphs=True
    )
    _assert_no_pre_cast_redistribution(
        sharding_placement, autop, require_linked_validation=True
    )


@apply_cuda_patches
def test_grad_reduce_dtype_bf16_allows_early_reduction(device_mesh_1d):
    """With reduce_dtype=bf16 (smaller than param_dtype=f32), the constraint
    should NOT fire, and the optimizer is free to reduce before the cast.
    """
    dim = 1024
    mesh = device_mesh_1d

    def model_fn():
        return SimpleLinear(dim)

    def input_fn():
        return torch.randn(2048 * mesh.shape[0], dim, device="cuda")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32, reduce_dtype=torch.bfloat16
    )
    sharding_placement, _ = _run_autop(mesh, model_fn, input_fn, mp_policy)
    assert sharding_placement is not None, "Optimizer should find a feasible solution"


@apply_cuda_patches
def test_grad_reduce_dtype_same_dtype_no_constraint(device_mesh_1d):
    """With reduce_dtype == param_dtype, no special constraint should fire."""
    dim = 1024
    mesh = device_mesh_1d

    def model_fn():
        return SimpleLinear(dim)

    def input_fn():
        return torch.randn(2048 * mesh.shape[0], dim, device="cuda")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32, reduce_dtype=torch.float32
    )
    sharding_placement, _ = _run_autop(mesh, model_fn, input_fn, mp_policy)
    assert sharding_placement is not None, "Optimizer should find a feasible solution"
