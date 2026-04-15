# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for _add_paired_output_constraint's disable logic.

The fix ensures that when a param node has a placement that its corresponding
grad node doesn't, the ILP decision variables for that placement are forced
to zero.  Without this, the solver could freely pick the unmatched placement,
producing a param/grad shape mismatch at runtime.

Standard DTensor propagation rules happen to produce matching param/grad
placements, so these tests synthetically remove a grad placement to create
the mismatch scenario that other backends (e.g. CuTe) can trigger naturally.
"""

from unittest.mock import patch

import pulp
import torch
from torch import nn
from torch._functorch._aot_autograd.fx_utils import get_param_and_grad_nodes
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel.api import AutoParallel


def _setup_with_grad_placement_removed(mesh, remove_placement_str):
    """Create a Linear model, trace it, then remove one placement from the
    grad node's strategies and rebuild the ILP.

    Returns (autop, opt, param_node, grad_node).
    The caller must eventually call autop.__exit__(None, None, None).
    """
    dim1, dim2 = 1024, 4096

    with torch.device("meta"):
        model = nn.Linear(dim1, dim2, bias=False)

    def input_fn():
        return torch.randn(512, dim1, device="cuda", requires_grad=True)

    autop = AutoParallel(model, input_fn, mesh)
    autop.__enter__()
    opt = autop.sharding_optimizer

    pairs = get_param_and_grad_nodes(opt.graph)
    param_node, grad_node = next((p, g) for p, g in pairs.values() if g is not None)

    # Remove one placement from grad to create a mismatch
    opt.strats[grad_node].strategies = [
        s
        for s in opt.strats[grad_node].strategies
        if str(s.output_specs.placements) != remove_placement_str
    ]

    # Rebuild the ILP with modified strategies
    opt._name_counters.clear()
    opt.decision_vars = opt._build_decision_vars()
    opt.prob = pulp.LpProblem("AutoParallel", pulp.LpMinimize)
    opt.add_default_constraints()

    return autop, opt, param_node, grad_node


# Placement to remove in all tests — must exist in both param and grad for
# nn.Linear(1024, 4096) on a (32, 8) mesh.
_REMOVED_PLACEMENT = "(Shard(dim=1), Shard(dim=1))"


@patch("torch.cuda.device_count", lambda: 8)
@patch("torch.cuda.get_device_name", lambda device: "H100")
def test_disable_constraints_added_for_unmatched_placements(device_mesh_2d):
    """The fix must add explicit disable constraints (sum == 0) for param
    placements that don't exist in the grad.

    Without the fix, no disable constraints are generated and the solver
    can freely choose the unmatched placement.
    """
    autop, opt, param_node, grad_node = _setup_with_grad_placement_removed(
        device_mesh_2d, _REMOVED_PLACEMENT
    )
    try:
        disable_constraints = [
            name
            for name in opt.prob.constraints
            if "grad_param_constraint_disable" in name
        ]
        assert (
            len(disable_constraints) > 0
        ), "Expected disable constraints for unmatched param placements"
    finally:
        autop.__exit__(None, None, None)


@patch("torch.cuda.device_count", lambda: 8)
@patch("torch.cuda.get_device_name", lambda device: "H100")
def test_unmatched_decision_vars_are_zero_in_solution(device_mesh_2d):
    """After solving, every decision variable for the unmatched param
    placement must be zero."""
    autop, opt, param_node, grad_node = _setup_with_grad_placement_removed(
        device_mesh_2d, _REMOVED_PLACEMENT
    )
    try:
        autop.add_input_constraints([(Shard(0), Replicate())])
        autop.add_output_constraints([(Shard(0), Replicate())])
        autop.add_parameter_memory_constraint(low=None, high=None)
        autop.optimize_placement()

        # Find the output index for the unmatched placement
        param_idx = opt.node_map[param_node]
        num_inp = len(opt.strats[param_node].strategies[0].redistribute_cost[0])
        unmatched_out_indices = [
            i
            for i, s in enumerate(opt.strats[param_node].strategies)
            if str(s.output_specs.placements) == _REMOVED_PLACEMENT
        ]
        assert len(unmatched_out_indices) == 1

        out_idx = unmatched_out_indices[0]
        for inp_idx in range(num_inp):
            var = opt.pulp_variables[(param_idx, 0, out_idx, inp_idx)]
            assert var.varValue == 0.0, (
                f"Decision var for unmatched placement {_REMOVED_PLACEMENT} "
                f"(inp_idx={inp_idx}) = {var.varValue}, expected 0.0"
            )
    finally:
        autop.__exit__(None, None, None)


@patch("torch.cuda.device_count", lambda: 8)
@patch("torch.cuda.get_device_name", lambda device: "H100")
def test_chosen_param_placement_matches_grad(device_mesh_2d):
    """The param's chosen placement must exist in the grad's strategy list."""
    autop, opt, param_node, grad_node = _setup_with_grad_placement_removed(
        device_mesh_2d, _REMOVED_PLACEMENT
    )
    try:
        autop.add_input_constraints([(Shard(0), Replicate())])
        autop.add_output_constraints([(Shard(0), Replicate())])
        autop.add_parameter_memory_constraint(low=None, high=None)
        solution = autop.optimize_placement()

        grad_placements = {
            str(s.output_specs.placements) for s in opt.strats[grad_node].strategies
        }
        chosen = str(solution[param_node].output_specs.placements)
        assert (
            chosen in grad_placements
        ), f"Param chose {chosen} which is not in grad placements {grad_placements}"
        assert chosen != _REMOVED_PLACEMENT
    finally:
        autop.__exit__(None, None, None)
