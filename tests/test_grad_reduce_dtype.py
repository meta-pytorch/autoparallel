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


def _run_and_get_grad_placements(mesh, model_fn, input_fn, mp_policy):
    """Run AutoParallel and return the placements of backward dtype_cast nodes."""
    with torch.device("meta"):
        model = model_fn()

    with AutoParallel(model, input_fn, mesh, mp_policy) as autop:
        autop.add_input_constraints([(Shard(0),) * mesh.ndim])
        autop.add_output_constraints([(Shard(0),) * mesh.ndim])
        sharding_placement = autop.optimize_placement(verbose=False)

    # Find backward dtype_cast nodes and their placements
    dtype_cast_placements = {}
    for node, spec in sharding_placement.items():
        if node.target == torch.ops.autoparallel.dtype_cast.default:
            # Check if this is a backward cast (bf16->f32 or f32->bf16)
            # by looking at output dtype vs input dtype
            if "f32" in str(node.meta.get("val", "")) or "f32" in str(node):
                dtype_cast_placements[node.name] = spec.output_specs.placements

    return sharding_placement, dtype_cast_placements, autop


@apply_cuda_patches
def test_grad_reduce_dtype_f32_reduces_after_cast(device_mesh_1d):
    """With reduce_dtype=f32, gradient reductions should happen after dtype_cast.

    The dtype_cast node should preserve P(sum) placement (no redistribution
    before the cast), so that the reduce-scatter happens in f32.
    """
    dim = 1024
    mesh = device_mesh_1d

    def model_fn():
        return SimpleLinear(dim)

    def input_fn():
        return torch.randn(2048 * mesh.shape[0], dim, device="cuda")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    sharding_placement, dtype_cast_placements, autop = _run_and_get_grad_placements(
        mesh, model_fn, input_fn, mp_policy
    )

    # Find backward dtype_cast nodes (those that cast to f32 in the grad chain)
    from torch._functorch._aot_autograd.fx_utils import get_param_and_grad_nodes

    opt = autop.sharding_optimizer
    for param, grad in get_param_and_grad_nodes(opt.graph).values():
        if grad is None:
            continue
        # Walk backward from grad to find dtype_cast
        n = grad
        while len(n.all_input_nodes) == 1:
            parent = n.all_input_nodes[0]
            if parent.target == torch.ops.autoparallel.dtype_cast.default:
                # The dtype_cast should have Partial in its output placement,
                # meaning no reduction happened before the cast
                spec = sharding_placement.get(parent)
                if spec is not None:
                    assert any(p.is_partial() for p in spec.output_specs.placements), (
                        f"dtype_cast {parent.name} should have Partial output "
                        f"(no pre-cast reduction), but got {spec.output_specs.placements}"
                    )
                break
            if len(parent.all_input_nodes) != 1:
                break
            n = parent


@apply_cuda_patches
def test_grad_reduce_dtype_bf16_allows_early_reduction(device_mesh_1d):
    """With reduce_dtype=bf16 (smaller than param_dtype=f32), the constraint
    should NOT fire, and the optimizer is free to reduce before the cast.

    We verify that the constraint doesn't interfere — the optimizer should
    find a valid solution without any infeasibility.
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
    # This should not raise — the constraint only fires for rescale > 1.0
    sharding_placement, _, _ = _run_and_get_grad_placements(
        mesh, model_fn, input_fn, mp_policy
    )
    assert sharding_placement is not None, "Optimizer should find a feasible solution"
