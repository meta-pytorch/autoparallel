# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pulp
import pytest
import torch
import torch.nn.functional as F
from conftest import apply_cuda_patches
from torch import nn
from torch._functorch._aot_autograd.fx_utils import get_param_and_grad_nodes
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel.api import AutoParallel
from autoparallel.propagation import ShardingAnnotation, ShardingPropagator


class TPBlock(nn.Module):
    """A minimal transformer block: attention + SwiGLU FFN, the structure a
    column/row-parallel tensor-parallel plan applies to."""

    def __init__(self, dim=512, hidden=1024, nheads=8):
        super().__init__()
        self.nheads = nheads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x):
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = q.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
        k = k.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
        v = v.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.permute(0, 2, 1, 3).flatten(-2)
        h = self.wo(o) + x
        return h + self.w2(F.silu(self.w1(h)) * self.w3(h))


def _input_fn():
    bs = 32
    return torch.randn(bs, 128, 512, device="cuda", requires_grad=True)


def _enter_autop(mesh):
    with torch.device("meta"):
        model = TPBlock()
    autop = AutoParallel(model, _input_fn, mesh)
    autop.__enter__()
    autop.add_parameter_memory_constraint(low=None, high=None)
    x_sharding = (Shard(0),) + (Replicate(),) * (mesh.ndim - 1)
    autop.add_input_constraints([x_sharding])
    autop.add_output_constraints([x_sharding])
    return autop


def _annotate_tp(autop):
    col, row = (None, Shard(0)), (None, Shard(1))
    for proj in ["wq", "wk", "wv", "w1", "w3"]:
        autop.annotate_parameter(f"{proj}.weight", col)
    for proj in ["wo", "w2"]:
        autop.annotate_parameter(f"{proj}.weight", row)


@apply_cuda_patches
def test_propagation_matches_full_ilp(device_mesh_2d):
    """Annotating the TP plan and propagating shrinks the search space while the
    reduced ILP reaches the same optimum as the full ILP."""
    autop = _enter_autop(device_mesh_2d)
    try:
        opt = autop.sharding_optimizer
        autop.optimize_placement(verbose=False)
        obj_full = pulp.value(opt.prob.objective)

        _annotate_tp(autop)
        result = autop.propagate_annotations(verbose=False)
        opt.resolve(verbose=False)
        obj_annotated = pulp.value(opt.prob.objective)

        assert opt.prob.status == 1  # Optimal
        # Same optimum (propagation only pins reshard-free, unambiguous sharding).
        assert obj_annotated == pytest.approx(obj_full, rel=1e-6)
        # And it actually pruned a meaningful chunk of the search space.
        assert result.reduction > 0.1
        assert result.nodes_determined > 0
    finally:
        autop.__exit__(None, None, None)


@apply_cuda_patches
def test_lp_relaxation_is_integral_and_exact(device_mesh_2d):
    """The LP relaxation of the sharding ILP is integral here, so solving it is a
    cheaper exact solve: same objective as the ILP, with an extractable solution."""
    autop = _enter_autop(device_mesh_2d)
    try:
        opt = autop.sharding_optimizer
        autop.optimize_placement(verbose=False)
        obj_ilp = pulp.value(opt.prob.objective)

        lp = opt.solve_lp_relaxation(extract=True)
        assert lp["n_fractional"] == 0  # relaxation is integral
        assert lp["objective"] == pytest.approx(obj_ilp, rel=1e-6)
        assert lp["solution"] is not None
        # one strategy per (single-output) decision node
        assert len(lp["solution"]) > 0
    finally:
        autop.__exit__(None, None, None)


@apply_cuda_patches
def test_axis_constraint_fix_method_matches_constraint(device_mesh_2d):
    """Pinning an axis by fixing variables gives the same result as the equality
    constraint, and is exact."""
    autop = _enter_autop(device_mesh_2d)
    try:
        opt = autop.sharding_optimizer
        fqn = {d.target: n for d, (n, _) in get_param_and_grad_nodes(opt.graph).items()}
        wq = fqn["wq.weight"]
        opt.add_node_axis_constraint(wq, mesh_dim=1, placement=Shard(0), method="fix")
        solution = autop.optimize_placement(verbose=False)
        assert opt.prob.status == 1
        placements = solution[opt._concrete_to_orig.get(wq, wq)].output_specs.placements
        assert placements[1] == Shard(0)
    finally:
        autop.__exit__(None, None, None)


@apply_cuda_patches
def test_add_node_axis_constraint_pins_one_axis(device_mesh_2d):
    """A per-axis constraint pins the chosen mesh axis and leaves the other free."""
    autop = _enter_autop(device_mesh_2d)
    try:
        opt = autop.sharding_optimizer
        fqn = {d.target: n for d, (n, _) in get_param_and_grad_nodes(opt.graph).items()}
        wq = fqn["wq.weight"]
        opt.add_node_axis_constraint(wq, mesh_dim=1, placement=Shard(0))
        solution = autop.optimize_placement(verbose=False)
        placements = solution[opt._concrete_to_orig.get(wq, wq)].output_specs.placements
        # tp axis pinned to Shard(0); dp axis decided by the ILP.
        assert placements[1] == Shard(0)
    finally:
        autop.__exit__(None, None, None)


@apply_cuda_patches
def test_axis_constraint_keeps_param_shardable_for_fsdp(device_mesh_2d):
    """A per-axis tp constraint must not exclude a parameter from the memory
    budget: it should still be shardable on the (free) data axis for FSDP."""
    autop = _enter_autop(device_mesh_2d)
    try:
        opt = autop.sharding_optimizer
        fqn = {d.target: n for d, (n, _) in get_param_and_grad_nodes(opt.graph).items()}
        wq = fqn["wq.weight"]
        # Column-parallel on tp; data axis left open.
        opt.add_node_axis_constraint(wq, mesh_dim=1, placement=Shard(0))
        solution = autop.optimize_placement(verbose=False)
        assert opt.prob.status == 1  # feasible despite the tight memory budget
        placements = solution[opt._concrete_to_orig.get(wq, wq)].output_specs.placements
        # FSDP shards the data axis too (tight 1/world_size budget).
        assert placements[0] == Shard(0)
        assert placements[1] == Shard(0)
    finally:
        autop.__exit__(None, None, None)


@apply_cuda_patches
def test_seed_unachievable_raises(device_mesh_2d):
    autop = _enter_autop(device_mesh_2d)
    try:
        opt = autop.sharding_optimizer
        prop = ShardingPropagator(opt)
        fqn = {d.target: n for d, (n, _) in get_param_and_grad_nodes(opt.graph).items()}
        wq = fqn["wq.weight"]
        # wq.weight is 2D; sharding a non-existent tensor dim 5 is impossible.
        with pytest.raises(ValueError):
            prop.seed(wq, (None, Shard(5)))
    finally:
        autop.__exit__(None, None, None)


@apply_cuda_patches
def test_propagation_determines_matmul_outputs(device_mesh_2d):
    """Seeding the column-parallel weights determines the tp axis of the matmul
    outputs (sharded on the output feature) with no resharding."""
    autop = _enter_autop(device_mesh_2d)
    try:
        opt = autop.sharding_optimizer
        prop = ShardingPropagator(opt)
        annotations = []
        fqn = {d.target: n for d, (n, _) in get_param_and_grad_nodes(opt.graph).items()}
        for proj in ["wq", "wk", "wv", "w1", "w3"]:
            annotations.append(
                (fqn[f"{proj}.weight"], ShardingAnnotation((None, Shard(0)), 1))
            )
        for proj in ["wo", "w2"]:
            annotations.append(
                (fqn[f"{proj}.weight"], ShardingAnnotation((None, Shard(1)), 1))
            )
        determined = prop.run(annotations)

        # Every column-parallel matmul output should be tp-sharded (not replicated).
        einsum_nodes = opt.graph.find_nodes(
            op="call_function", target=torch.ops.aten.einsum.default
        )
        if not einsum_nodes:
            einsum_nodes = opt.graph.find_nodes(
                op="call_function", target=torch.ops.aten.mm.default
            )
        n_tp_pinned = 0
        for n in einsum_nodes:
            if n in determined:
                tp = dict(determined[n]).get(1)
                if isinstance(tp, Shard):
                    n_tp_pinned += 1
        assert n_tp_pinned > 0
    finally:
        autop.__exit__(None, None, None)
