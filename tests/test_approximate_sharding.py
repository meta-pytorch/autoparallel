# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math

import pulp
import pytest
import torch
from conftest import apply_cuda_patches
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel._testing.models.llama3 import Transformer, TransformerModelArgs
from autoparallel.api import AutoParallel
from autoparallel.approximate_sharding import ApproximateShardingSolver


def _fake_2d_mesh():
    return torch.distributed.device_mesh.init_device_mesh(
        "cuda", (4, 2), mesh_dim_names=("dp", "tp")
    )


def _tiny_llama3_autop(mesh, solver="ilp"):
    vocab_size = 128
    seq_len = 16
    batch_size = 2 * mesh.shape[0]
    model_args = TransformerModelArgs(
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=vocab_size,
        multiple_of=32,
        rope_theta=500000,
        max_seq_len=seq_len,
    )
    with torch.device("meta"):
        model = Transformer(model_args)

    def input_fn():
        return torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    return AutoParallel(
        model, input_fn, mesh, mp_policy, repeated_subgraphs=True, solver=solver
    )


def _add_constraints(autop, mesh):
    autop.add_parameter_memory_constraint(low=None, high=None)
    autop.add_input_constraints([(Shard(0),) + (Replicate(),) * (mesh.ndim - 1)])
    autop.add_output_constraints([(Shard(0), Shard(2))])


@apply_cuda_patches
@pytest.mark.filterwarnings("ignore:Constructing LpVariable")
@pytest.mark.filterwarnings("ignore:Overwriting previously set objective")
def test_approx_objective_close_to_ilp():
    """The approximate solver should be much faster than the ILP while staying
    within a small objective gap on a tiny LLaMA3 block + 2D mesh."""
    mesh = _fake_2d_mesh()
    with _tiny_llama3_autop(mesh) as autop:
        _add_constraints(autop, mesh)
        opt = autop.sharding_optimizer

        autop.optimize_placement(verbose=False, solver="approx")
        approx_objective = pulp.value(opt.prob.objective)
        # The approx assignment must be ILP-feasible (flow consistency etc.);
        # an infeasible assignment can score artificially low and silently pass
        # the objective bound below.
        violated = [n for n, c in opt.prob.constraints.items() if not c.valid()]
        assert not violated, f"approx violated {len(violated)} constraints"

        autop.optimize_placement(verbose=False, solver="ilp")
        ilp_objective = pulp.value(opt.prob.objective)

        assert math.isfinite(approx_objective)
        assert ilp_objective > 0
        assert approx_objective >= ilp_objective - 1e-6  # ILP is optimal
        assert approx_objective <= ilp_objective * 1.20 + 1e-6, (
            f"approx={approx_objective} ilp={ilp_objective} "
            f"gap={(approx_objective / ilp_objective - 1) * 100:.1f}%"
        )


@apply_cuda_patches
@pytest.mark.filterwarnings("ignore:Constructing LpVariable")
@pytest.mark.filterwarnings("ignore:Overwriting previously set objective")
def test_lp_solver_matches_ilp():
    """The LP-relaxation solver returns an integral, ILP-feasible assignment whose
    objective equals the exact ILP optimum (the relaxation is integral here)."""
    mesh = _fake_2d_mesh()
    with _tiny_llama3_autop(mesh) as autop:
        _add_constraints(autop, mesh)
        opt = autop.sharding_optimizer

        autop.optimize_placement(verbose=False, solver="lp")
        lp_objective = pulp.value(opt.prob.objective)
        violated = [n for n, c in opt.prob.constraints.items() if not c.valid()]
        assert not violated, f"lp violated {len(violated)} constraints"

        autop.optimize_placement(verbose=False, solver="ilp")
        ilp_objective = pulp.value(opt.prob.objective)

        assert math.isfinite(lp_objective)
        assert lp_objective == pytest.approx(ilp_objective, rel=1e-6)


@apply_cuda_patches
@pytest.mark.filterwarnings("ignore:Constructing LpVariable")
@pytest.mark.filterwarnings("ignore:Overwriting previously set objective")
def test_optimality_check_logs_certified_gap(caplog):
    """optimality_check=True solves the LP lower bound and logs the certified gap."""
    mesh = _fake_2d_mesh()
    with _tiny_llama3_autop(mesh) as autop:
        _add_constraints(autop, mesh)
        with caplog.at_level(logging.INFO, logger="autoparallel.api"):
            autop.optimize_placement(
                verbose=False, solver="approx", optimality_check=True
            )
        assert any("optimality check" in r.message for r in caplog.records)


@apply_cuda_patches
@pytest.mark.filterwarnings("ignore:Constructing LpVariable")
def test_approx_objective_is_faithful():
    """The solver's internal energy must equal the exact ILP objective evaluated
    on its assignment (pulp.value), so comparisons against the ILP are valid."""
    mesh = _fake_2d_mesh()
    with _tiny_llama3_autop(mesh) as autop:
        _add_constraints(autop, mesh)
        opt = autop.sharding_optimizer

        solver = ApproximateShardingSolver(opt)
        solver.get_solution(verbose=False)

        pulp_objective = pulp.value(opt.prob.objective)
        internal_energy = solver.total_objective()
        assert math.isfinite(internal_energy)
        assert internal_energy == pytest.approx(pulp_objective, rel=1e-6)
        # No forbidden decision variable should be selected.
        assert all(key not in solver.forbidden for key in opt.selected_keys)
        # And every ILP constraint must hold (flow consistency, paired, memory).
        violated = [n for n, c in opt.prob.constraints.items() if not c.valid()]
        assert not violated, f"approx violated {len(violated)} constraints"


@apply_cuda_patches
@pytest.mark.filterwarnings("ignore:Constructing LpVariable")
def test_approx_respects_input_output_constraints():
    """User input/output placement constraints must be honored by the solution."""
    mesh = _fake_2d_mesh()
    x_sharding = (Shard(0),) + (Replicate(),) * (mesh.ndim - 1)
    out_sharding = (Shard(0), Shard(2))
    with _tiny_llama3_autop(mesh) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([out_sharding])

        solution = autop.optimize_placement(verbose=False, solver="approx")
        assert solution

        placements = {
            spec.placements
            for strat in solution.values()
            for spec in (
                strat.output_specs
                if isinstance(strat.output_specs, (list, tuple))
                else (strat.output_specs,)
            )
            if isinstance(spec, DTensorSpec)
        }
        assert x_sharding in placements
        assert out_sharding in placements


@apply_cuda_patches
@pytest.mark.filterwarnings("ignore:Constructing LpVariable")
def test_lite_build_matches_full():
    """Building with solver="approx" skips PuLP variables/constraints (faster
    setup); the resulting assignment must be byte-identical to running the
    approximate solver on a full PuLP build."""
    mesh = _fake_2d_mesh()

    with _tiny_llama3_autop(mesh, solver="ilp") as autop:
        _add_constraints(autop, mesh)
        assert autop.sharding_optimizer.prob is not None
        autop.optimize_placement(verbose=False, solver="approx")
        obj_full = autop.sharding_optimizer.profile["approximate"]["objective"]
        keys_full = set(autop.sharding_optimizer.selected_keys)

    with _tiny_llama3_autop(mesh, solver="approx") as autop:
        _add_constraints(autop, mesh)
        # Lite build: no PuLP problem or variables were constructed.
        assert autop.sharding_optimizer.prob is None
        assert not autop.sharding_optimizer.pulp_variables
        solution = autop.optimize_placement(verbose=False)
        obj_lite = autop.sharding_optimizer.profile["approximate"]["objective"]
        keys_lite = set(autop.sharding_optimizer.selected_keys)
        assert solution

    assert obj_lite == pytest.approx(obj_full, rel=1e-9)
    assert keys_lite == keys_full
