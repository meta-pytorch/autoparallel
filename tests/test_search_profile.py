# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import gc
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import pulp
import pytest
import torch
from search_profile import (
    ProfiledAutoParallel,
    finite,
    parse_args,
    solution_fingerprint,
    validate_args,
    validate_solution,
)
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.tensor.placement_types import Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from autoparallel._testing.models.dsv3 import (
    DeepSeekV3Model,
    build_moe_mesh,
    make_dsv3_config,
)


def _args(*values):
    return parse_args(
        [
            *values,
            "--solver",
            "approx",
            "--revision-label",
            "test",
            "--output",
            "unused.json",
        ]
    )


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (("--model", "llama1b", "--mesh", "8,8"), (64, (8, 8))),
        (("--model", "llama1b", "--mesh", "2,4,8"), (64, (2, 4, 8))),
        (("--model", "dsv3", "--moe-layout", "2d"), (64, None)),
        (("--model", "dsv3", "--moe-layout", "3d"), (64, None)),
        (("--model", "dsv3", "--moe-layout", "4d"), (64, None)),
    ],
)
def test_validate_search_profile_args(values, expected):
    assert validate_args(_args(*values)) == expected


@pytest.mark.parametrize(
    "values",
    [
        ("--model", "llama1b"),
        ("--model", "llama1b", "--mesh", "4,4"),
        ("--model", "llama1b", "--mesh", "1,1,1,64"),
        ("--model", "llama1b", "--mesh", "8,8", "--moe-layout", "2d"),
        ("--model", "dsv3", "--moe-layout", "2d", "--mesh", "8,8"),
    ],
)
def test_validate_search_profile_args_rejects_invalid_combinations(values):
    with pytest.raises(ValueError):
        validate_args(_args(*values))


def test_validate_search_profile_args_rejects_lazy_non_approx():
    args = parse_args(
        [
            "--model",
            "llama1b",
            "--mesh",
            "8,8",
            "--solver",
            "ilp",
            "--lazy-costs",
            "true",
            "--revision-label",
            "test",
            "--output",
            "unused.json",
        ]
    )
    with pytest.raises(ValueError, match="only with --solver approx"):
        validate_args(args)


def test_validate_search_profile_args_rejects_seeded_non_approx():
    args = parse_args(
        [
            "--model",
            "llama1b",
            "--mesh",
            "2,4,8",
            "--solver",
            "ilp",
            "--seeded",
            "--revision-label",
            "test",
            "--output",
            "unused.json",
        ]
    )
    with pytest.raises(ValueError, match="only with --solver approx"):
        validate_args(args)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"solver": "ilp", "pulp_status": "Not Solved"},
        {"solver": "approx", "solution_status": "No Solution Found"},
        {"objective": None},
        {"solution_nodes": 0},
        {"violations": ["constraint"]},
    ],
)
def test_validate_solution_rejects_invalid_results(kwargs):
    values = {
        "solver": "ilp",
        "objective": 1.0,
        "solution_nodes": 1,
        "violations": [],
        "pulp_status": "Optimal",
        "solution_status": "Optimal Solution Found",
    }
    values.update(kwargs)
    with pytest.raises(RuntimeError):
        validate_solution(**values)


@pytest.mark.parametrize(
    ("solver", "pulp_status", "solution_status"),
    [
        ("ilp", "Optimal", "Optimal Solution Found"),
        ("approx", "Not Solved", "Solution Found"),
    ],
)
def test_validate_solution_accepts_valid_results(solver, pulp_status, solution_status):
    validate_solution(
        solver=solver,
        objective=1.0,
        solution_nodes=1,
        violations=[],
        pulp_status=pulp_status,
        solution_status=solution_status,
    )


def test_llama1b_approx_search_e2e(tmp_path):
    output = tmp_path / "result.json"
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(root / "tests" / "search_profile.py"),
        "--model",
        "llama1b",
        "--mesh",
        "8,8",
        "--solver",
        "approx",
        "--revision-label",
        "pytest",
        "--output",
        str(output),
    ]
    env = os.environ | {
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": str(root),
    }
    completed = subprocess.run(
        command,
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=20 * 60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    result = json.loads(output.read_text())
    assert result["status"] == "success"
    assert result["request"]["model"] == "llama1b"
    assert result["request"]["solver"] == "approx"
    assert result["validation"]["solver_status"] == "Heuristic"
    assert result["validation"]["pulp_solution_status"] == "Solution Found"
    assert result["validation"]["constraint_violations"] == 0
    assert math.isfinite(result["objective"])
    assert result["solution_nodes"] > 0
    assert len(result["placement_sha256"]) == 64
    assert result["counts"]["decision_vars"] == 0
    assert result["counts"]["pulp_variables"] == 0
    assert result["counts"]["constraints"] == 0
    for name in (
        "graph_trace_s",
        "optimizer_init_s",
        "user_constraints_s",
        "solve_call_s",
        "factor_build_s",
        "solver_core_s",
        "search_total_s",
    ):
        assert result["timings"][name] >= 0


def _synchronize():
    torch.cuda.synchronize()


def _full_cpu(tensor):
    if isinstance(tensor, DTensor):
        tensor = tensor.full_tensor()
    return tensor.detach().cpu()


def _behavior_tolerances(dtype):
    if dtype == torch.bfloat16:
        tolerance = 2 * torch.finfo(dtype).eps
        return {"rtol": tolerance, "atol": tolerance}
    return {}


def _difference_stats(actual, expected):
    difference = (actual.float() - expected.float()).abs()
    return difference.max().item(), difference.sum().item(), difference.numel()


class TestRealDsv3SolverE2E(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    def _run_solver(self, solver, mesh, roles, tokens):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        config = make_dsv3_config(num_experts=8, max_seq_len=2048)
        with torch.device("meta"):
            model = DeepSeekV3Model(
                config,
                mesh=mesh,
                roles=roles,
                compute_dtype=torch.bfloat16,
            )

        global_batch_size = 8 * mesh.size()

        def input_fn():
            return torch.randint(
                0,
                config.vocab_size,
                (global_batch_size, 2048),
                device="cuda",
            )

        placement = (Shard(0),) * mesh.ndim
        model_setup_s = time.perf_counter() - started
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        autop = ProfiledAutoParallel(
            model,
            input_fn,
            mesh,
            mp_policy=mp_policy,
            dynamic=True,
            solver=solver,
        )
        enter_started = time.perf_counter()
        with autop:
            enter_s = time.perf_counter() - enter_started
            opt = autop.sharding_optimizer
            constraint_started = time.perf_counter()
            autop.add_parameter_memory_constraint(low=None, high=None)
            autop.add_input_constraints([placement])
            autop.add_output_constraints([placement])
            constraint_s = time.perf_counter() - constraint_started

            solve_started = time.perf_counter()
            solution = autop.optimize_placement(verbose=False)
            solve_s = time.perf_counter() - solve_started
            apply_started = time.perf_counter()
            parallel_mod = autop.apply_placement(solution)
            apply_s = time.perf_counter() - apply_started

        init_started = time.perf_counter()
        parallel_mod.to_empty(device="cuda")
        parallel_mod.init_weights(seed=0)
        _synchronize()
        init_s = time.perf_counter() - init_started

        forward_started = time.perf_counter()
        output = parallel_mod(tokens)
        _synchronize()
        forward_s = time.perf_counter() - forward_started
        output_materialize_started = time.perf_counter()
        full_output = DTensor.from_local(output, mesh, placement).full_tensor()
        loss = full_output.float().mean().abs() * 1e-3
        _synchronize()
        output_materialize_s = time.perf_counter() - output_materialize_started
        backward_started = time.perf_counter()
        loss.backward()
        _synchronize()
        backward_s = time.perf_counter() - backward_started

        gradients = {}
        for name, parameter in parallel_mod.named_parameters():
            gradients[name] = (
                None if parameter.grad is None else _full_cpu(parameter.grad)
            )

        fingerprint, solution_nodes = solution_fingerprint(solution)
        profile_key = "approximate" if solver == "approx" else solver
        solver_profile = opt.profile[profile_key]
        objective = finite(solver_profile["objective"])
        violations = []
        pulp_status = None
        solution_status = None
        if opt.prob is not None:
            objective = finite(pulp.value(opt.prob.objective))
            violations = [
                name
                for name, constraint in opt.prob.constraints.items()
                if not constraint.valid(1e-6)
            ]
            pulp_status = pulp.LpStatus.get(opt.prob.status, str(opt.prob.status))
            solution_status = pulp.LpSolution.get(
                getattr(opt.prob, "sol_status", None),
                str(getattr(opt.prob, "sol_status", None)),
            )
        elif solver == "approx":
            solution_status = (
                "Solution Found"
                if solver_profile["status"] == "Heuristic"
                else "No Solution Found"
            )
        validate_solution(
            solver,
            objective,
            solution_nodes,
            violations,
            pulp_status,
            solution_status,
        )
        result = {
            "solver": solver,
            "objective": objective,
            "placement_sha256": fingerprint,
            "solution_nodes": solution_nodes,
            "validation": {
                "solver_status": solver_profile["status"],
                "pulp_status": pulp_status,
                "pulp_solution_status": solution_status,
                "constraint_violations": len(violations),
            },
            "timings": {
                "model_setup_s": model_setup_s,
                "enter_total_s": enter_s,
                "graph_trace_s": autop.profile_graph_trace_s,
                "optimizer_init_s": opt.profile["timings"]["init_total_s"],
                "constraints_s": constraint_s,
                "solve_call_s": solve_s,
                "factor_build_s": solver_profile.get("build_s"),
                "solver_core_s": solver_profile.get(
                    "solve_s", solver_profile.get("solve_time")
                ),
                "apply_placement_s": apply_s,
                "materialize_init_s": init_s,
                "forward_s": forward_s,
                "output_materialize_s": output_materialize_s,
                "backward_s": backward_s,
                "total_s": time.perf_counter() - started,
            },
            "memory": {
                "process_max_rss_kib": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss,
                "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "max_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            },
        }
        output_cpu = full_output.detach().cpu()
        del output, full_output, loss, parallel_mod, autop, model
        gc.collect()
        torch.cuda.empty_cache()
        return result, output_cpu, gradients

    @pytest.mark.skipif(
        torch.cuda.device_count() < 4,
        reason="real DeepSeekV3 solver E2E requires four GPUs",
    )
    @with_comms
    def test_ilp_and_approx_match(self):
        mesh, roles = build_moe_mesh(
            dp_replicate=2,
            dp_shard=2,
            cp=1,
            tp=1,
            ep=2,
        )
        torch.manual_seed(0)
        placement = (Shard(0),) * mesh.ndim
        global_tokens = torch.randint(0, 2048, (32, 2048), device="cuda")
        tokens = distribute_tensor(global_tokens, mesh, placement).to_local()

        ilp, ilp_output, ilp_gradients = self._run_solver("ilp", mesh, roles, tokens)
        approx, approx_output, approx_gradients = self._run_solver(
            "approx", mesh, roles, tokens
        )

        assert approx["objective"] == pytest.approx(ilp["objective"], rel=1e-6)
        torch.testing.assert_close(
            approx_output,
            ilp_output,
            **_behavior_tolerances(approx_output.dtype),
        )
        output_max_abs, output_abs_sum, output_numel = _difference_stats(
            approx_output, ilp_output
        )
        assert approx_gradients.keys() == ilp_gradients.keys()
        gradient_max_abs = 0.0
        gradient_abs_sum = 0.0
        gradient_numel = 0
        for name in ilp_gradients:
            ilp_gradient = ilp_gradients[name]
            approx_gradient = approx_gradients[name]
            assert (ilp_gradient is None) == (approx_gradient is None), name
            if ilp_gradient is not None:
                torch.testing.assert_close(
                    approx_gradient,
                    ilp_gradient,
                    msg=lambda message: f"gradient mismatch for {name}: {message}",
                    **_behavior_tolerances(approx_gradient.dtype),
                )
                max_abs, abs_sum, numel = _difference_stats(
                    approx_gradient, ilp_gradient
                )
                gradient_max_abs = max(gradient_max_abs, max_abs)
                gradient_abs_sum += abs_sum
                gradient_numel += numel

        comparison = {
            "objective_match": True,
            "output_match": True,
            "output_max_abs": output_max_abs,
            "output_mean_abs": output_abs_sum / output_numel,
            "gradient_match": True,
            "gradient_max_abs": gradient_max_abs,
            "gradient_mean_abs": gradient_abs_sum / gradient_numel,
            "bfloat16_rtol": 2 * torch.finfo(torch.bfloat16).eps,
            "bfloat16_atol": 2 * torch.finfo(torch.bfloat16).eps,
        }

        local_report = {
            "rank": self.rank,
            "solvers": {"ilp": ilp, "approx": approx},
            "comparison": comparison,
        }
        per_rank = [None] * self.world_size
        torch.distributed.all_gather_object(per_rank, local_report)
        if self.rank == 0:
            print(
                "AUTOPARALLEL_E2E_BREAKDOWN "
                + json.dumps(
                    {
                        "environment": {
                            "python": sys.version.split()[0],
                            "torch": torch.__version__,
                            "cuda": torch.version.cuda,
                            "gpu": torch.cuda.get_device_name(),
                            "world_size": self.world_size,
                        },
                        "execution_order": ["ilp", "approx"],
                        "config": {
                            "model": "DeepSeekV3Model",
                            "dim": 256,
                            "n_layers": 6,
                            "n_dense_layers": 1,
                            "n_heads": 16,
                            "mesh_shape": list(mesh.shape),
                            "mesh_dim_names": list(mesh.mesh_dim_names),
                            "local_batch_size": 8,
                            "sequence_length": 2048,
                            "vocab_size": 2048,
                            "num_experts": 8,
                        },
                        "per_rank": per_rank,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
