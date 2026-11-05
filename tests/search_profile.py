# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import time
import traceback
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pulp
import torch
from torch._functorch._aot_autograd.fx_utils import (
    get_param_and_grad_nodes,
    get_plain_input_and_grad_nodes,
    get_plain_output_and_tangent_nodes,
)
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel.api import AutoParallel

LLAMA_CONFIGS = {
    "llama1b": {
        "dim": 2048,
        "n_layers": 16,
        "n_heads": 32,
        "n_kv_heads": 8,
        "ffn_dim_multiplier": 1.5,
        "multiple_of": 256,
    },
    "llama8b": {
        "dim": 4096,
        "n_layers": 32,
        "n_heads": 32,
        "n_kv_heads": 8,
        "ffn_dim_multiplier": 1.3,
        "multiple_of": 1024,
    },
}

DSV3_DEGREES = {
    "dp_replicate": 8,
    "dp_shard": 8,
    "cp": 1,
    "tp": 1,
    "ep": 8,
}


class ProfiledAutoParallel(AutoParallel):
    def build_model_graph(self):
        started = time.perf_counter()
        try:
            return super().build_model_graph()
        finally:
            self.profile_graph_trace_s = time.perf_counter() - started


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Profile AutoParallel placement search on fake H100 devices."
    )
    parser.add_argument("--model", choices=(*LLAMA_CONFIGS, "dsv3"), required=True)
    parser.add_argument("--mesh", help="Comma-separated LLaMA mesh dimensions")
    parser.add_argument("--moe-layout", choices=("2d",))
    parser.add_argument("--solver", choices=AutoParallel.SOLVER_CHOICES, required=True)
    parser.add_argument("--lazy-costs", choices=("true", "false"))
    parser.add_argument("--revision-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detailed-solution", action="store_true")
    return parser.parse_args(argv)


def fake_cuda_context(stack):
    properties = type(
        "Props",
        (),
        {
            "major": 9,
            "minor": 0,
            "name": "H100",
            "total_memory": 80 * 1024**3,
            "multi_processor_count": 132,
            "L2_cache_size": 50 * 1024**2,
        },
    )()
    for target, replacement in (
        ("torch.cuda.device_count", lambda: 8),
        ("torch.cuda.get_device_name", lambda *args, **kwargs: "H100"),
        ("torch.cuda.get_device_capability", lambda *args, **kwargs: (9, 0)),
        (
            "torch.cuda.get_device_properties",
            lambda *args, **kwargs: properties,
        ),
    ):
        stack.enter_context(patch(target, replacement))


def make_llama(model_name, mesh_shape):
    from autoparallel._testing.models.llama3 import Transformer, TransformerModelArgs

    names = {
        2: ("dp", "tp"),
        3: ("dp", "cp", "tp"),
    }.get(len(mesh_shape))
    if names is None:
        raise ValueError(f"LLaMA profiles require a 2D or 3D mesh, got {mesh_shape}")
    seq_len = 2048
    vocab_size = 128256
    config = {
        **LLAMA_CONFIGS[model_name],
        "rope_theta": 500000,
        "vocab_size": vocab_size,
        "max_seq_len": seq_len,
    }
    with torch.device("meta"):
        model = Transformer(TransformerModelArgs(**config))
    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda", mesh_shape, mesh_dim_names=names
    )
    batch_size = 16

    def input_fn():
        return torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")

    input_placement = (Shard(0),) + (Replicate(),) * (len(mesh_shape) - 1)
    output_placement = (Shard(0), Shard(2)) if len(mesh_shape) == 2 else input_placement
    expanded = {
        "family": "llama3",
        "config": config,
        "batch_size": batch_size,
        "sequence_length": seq_len,
        "mesh_shape": list(mesh_shape),
        "mesh_dim_names": list(names),
    }
    return model, input_fn, mesh, input_placement, output_placement, expanded


def make_dsv3():
    from autoparallel._testing.models.dsv3 import (
        DeepSeekV3Model,
        build_moe_mesh,
        make_dsv3_config,
    )

    mesh, roles = build_moe_mesh(**DSV3_DEGREES)
    config = make_dsv3_config(num_experts=64, max_seq_len=2048)
    with torch.device("meta"):
        model = DeepSeekV3Model(
            config, mesh=mesh, roles=roles, compute_dtype=torch.bfloat16
        )
    batch_size = 8 * mesh.size()

    def input_fn():
        return torch.randint(0, config.vocab_size, (batch_size, 2048), device="cuda")

    placement = (Shard(0),) * mesh.ndim
    expanded = {
        "family": "dsv3",
        "config": {
            "dim": config.dim,
            "vocab_size": config.vocab_size,
            "n_layers": len(config.layers),
            "n_dense_layers": sum(layer.moe is None for layer in config.layers),
            "num_experts": 64,
            "sequence_length": 2048,
            "batch_size": batch_size,
        },
        "degrees": DSV3_DEGREES,
        "mesh_shape": list(mesh.shape),
        "mesh_dim_names": list(mesh.mesh_dim_names),
        "ep_axis_names": list(roles.ep_axis_names),
        "ep_group_name": roles.ep_group_name,
    }
    return model, input_fn, mesh, placement, placement, expanded


def output_specs(value):
    specs = value if isinstance(value, (tuple, list)) else (value,)
    return [
        [repr(placement) for placement in spec.placements]
        for spec in specs
        if isinstance(spec, DTensorSpec)
    ]


def solution_fingerprint(solution):
    canonical = [
        (node.name, output_specs(strategy.output_specs))
        for node, strategy in sorted(solution.items(), key=lambda item: item[0].name)
    ]
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest(), len(canonical)


def spec_details(value):
    if isinstance(value, DTensorSpec):
        shape = None
        if value.tensor_meta is not None:
            shape = [str(dim) for dim in value.tensor_meta.shape]
        return {
            "placements": [repr(placement) for placement in value.placements],
            "shape": shape,
        }
    if isinstance(value, (tuple, list)):
        return [spec_details(item) for item in value]
    return None


def solution_details(solution, graph):
    roles = {}
    for desc, (node, grad) in get_param_and_grad_nodes(graph).items():
        roles[node.name] = f"parameter:{desc.target}"
        if grad is not None:
            roles[grad.name] = f"parameter_grad:{desc.target}"
    for desc, (node, grad) in get_plain_input_and_grad_nodes(graph).items():
        roles[node.name] = f"input:{desc.idx}"
        if grad is not None:
            roles[grad.name] = f"input_grad:{desc.idx}"
    for desc, (node, tangent) in get_plain_output_and_tangent_nodes(graph).items():
        roles[node.name] = f"output:{desc.idx}"
        if tangent is not None:
            roles[tangent.name] = f"output_tangent:{desc.idx}"

    rows = []
    for node, strategy in sorted(solution.items(), key=lambda item: item[0].name):
        stack = node.meta.get("stack_trace") or ""
        source = " | ".join(
            line.strip() for line in stack.splitlines()[-2:] if line.strip()
        )
        rows.append(
            {
                "node": node.name,
                "op": node.op,
                "target": getattr(node.target, "__name__", str(node.target)),
                "role": roles.get(node.name),
                "source": source,
                "output_specs": spec_details(strategy.output_specs),
                "input_specs": spec_details(strategy.input_specs),
            }
        )
    return rows


def cost_contributions(opt):
    selected = {}
    for key in opt.selected_keys:
        root_key = opt._cluster_root_key(key)
        if root_key in opt.decision_vars:
            selected[root_key] = opt.decision_vars[root_key]
    by_node = {}
    for key, decision in selected.items():
        node_idx, _arg_idx, out_idx, _inp_idx = key
        multiplier = 1 + len(opt._root_to_copies.get(node_idx, ()))
        row = by_node.setdefault(
            node_idx,
            {
                "node": opt.nodes[node_idx].name,
                "output_index": out_idx,
                "multiplier": multiplier,
                "compute": 0.0,
                "communication": 0.0,
                "transition": 0.0,
                "total": 0.0,
            },
        )
        row["compute"] += decision.compute_cost * multiplier
        row["communication"] += decision.comm_cost * multiplier
        row["transition"] += decision.sharding_transition_cost * multiplier
        row["total"] += decision.cost * multiplier
    return list(by_node.values())


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def run_git(*args):
    return subprocess.check_output(("git", *args), text=True).strip()


def git_metadata():
    status = run_git("status", "--short")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "worktree": run_git("rev-parse", "--show-toplevel"),
        "status": status,
    }


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def validate_args(args):
    if args.lazy_costs is not None and args.solver != "approx":
        raise ValueError("--lazy-costs is supported only with --solver approx")
    if args.model == "dsv3":
        if args.moe_layout != "2d" or args.mesh is not None:
            raise ValueError("dsv3 requires --moe-layout 2d and no --mesh")
        return 64, None
    if args.mesh is None or args.moe_layout is not None:
        raise ValueError("LLaMA requires --mesh and no --moe-layout")
    mesh_shape = tuple(int(part) for part in args.mesh.split(","))
    if len(mesh_shape) not in (2, 3):
        raise ValueError(f"LLaMA requires a 2D or 3D mesh, got {mesh_shape}")
    world_size = math.prod(mesh_shape)
    if world_size != 64:
        raise ValueError(f"world size must be 64, got {world_size}")
    return world_size, mesh_shape


def validate_solution(
    solver,
    objective,
    solution_nodes,
    violations,
    pulp_status,
    solution_status,
):
    if objective is None:
        raise RuntimeError("placement objective is not finite")
    if solution_nodes == 0:
        raise RuntimeError("placement solution is empty")
    if violations:
        raise RuntimeError(f"placement violates {len(violations)} constraints")
    if solver == "approx":
        if solution_status != "Solution Found":
            raise RuntimeError(
                f"approximate placement is not feasible: {solution_status}"
            )
    elif pulp_status != "Optimal":
        raise RuntimeError(f"{solver} placement is not optimal: {pulp_status}")


def main(argv=None):
    args = parse_args(argv)
    started = time.perf_counter()
    result = {
        "schema_version": 1,
        "status": "error",
        "request": vars(args) | {"output": str(args.output)},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "pulp": pulp.__version__,
            "platform": platform.platform(),
            "cpu_model": cpu_model(),
            "cpu_count": os.cpu_count(),
            "pid": os.getpid(),
        },
    }
    try:
        result["git"] = git_metadata()
        world_size, mesh_shape = validate_args(args)
        torch.manual_seed(0)
        with ExitStack() as stack:
            fake_cuda_context(stack)
            torch.distributed.init_process_group(
                "fake", store=FakeStore(), rank=0, world_size=world_size
            )

            model_started = time.perf_counter()
            built = (
                make_dsv3()
                if args.model == "dsv3"
                else make_llama(args.model, mesh_shape)
            )
            model, input_fn, mesh, input_placement, output_placement, expanded = built
            result["expanded_config"] = expanded
            result["timings"] = {
                "model_and_mesh_setup_s": time.perf_counter() - model_started
            }

            autop = ProfiledAutoParallel(
                model,
                input_fn,
                mesh,
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.float32
                ),
                repeated_subgraphs=True,
                dynamic=args.model == "dsv3",
                solver=args.solver,
                lazy_costs=(
                    None if args.lazy_costs is None else args.lazy_costs == "true"
                ),
            )
            enter_started = time.perf_counter()
            stack.enter_context(autop)
            enter_elapsed = time.perf_counter() - enter_started
            opt = autop.sharding_optimizer
            graph_trace_s = autop.profile_graph_trace_s
            optimizer_init_s = opt.profile["timings"]["init_total_s"]

            constraint_started = time.perf_counter()
            autop.add_parameter_memory_constraint(low=None, high=None)
            autop.add_input_constraints([input_placement])
            autop.add_output_constraints([output_placement])
            constraint_s = time.perf_counter() - constraint_started

            solve_started = time.perf_counter()
            solution = autop.optimize_placement(verbose=False)
            solve_call_s = time.perf_counter() - solve_started
            search_total_s = time.perf_counter() - enter_started
            unaccounted_s = max(
                search_total_s
                - graph_trace_s
                - optimizer_init_s
                - constraint_s
                - solve_call_s,
                0.0,
            )
            profile_key = {
                "approx": "approximate",
                "ilp": "ilp",
                "lp": "lp_relaxation",
            }[args.solver]
            solver_profile = opt.profile[profile_key]
            result["timings"].update(
                {
                    "enter_total_s": enter_elapsed,
                    "graph_trace_s": graph_trace_s,
                    "optimizer_init_s": optimizer_init_s,
                    "user_constraints_s": constraint_s,
                    "solve_call_s": solve_call_s,
                    "factor_build_s": solver_profile.get("build_s"),
                    "solver_core_s": solver_profile.get(
                        "solve_s", solver_profile.get("solve_time")
                    ),
                    "search_total_s": search_total_s,
                    "unaccounted_s": unaccounted_s,
                }
            )

            fingerprint, solution_nodes = solution_fingerprint(solution)
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
            elif args.solver == "approx":
                solution_status = (
                    "Solution Found"
                    if solver_profile["status"] == "Heuristic"
                    else "No Solution Found"
                )
            validation = {
                "solver_status": solver_profile["status"],
                "pulp_status": pulp_status,
                "pulp_solution_status": solution_status,
                "constraint_violations": len(violations),
                "violated_constraint_names": violations[:100],
            }
            result.update(
                {
                    "objective": objective,
                    "placement_sha256": fingerprint,
                    "solution_nodes": solution_nodes,
                    "validation": validation,
                    "optimizer_profile": opt.profile,
                    "counts": {
                        "graph_nodes": len(list(opt.graph.nodes)),
                        "strategy_nodes": len(opt.strats),
                        "decision_vars": len(opt.decision_vars),
                        "pulp_variables": len(opt.pulp_variables),
                        "constraints": (
                            len(opt.prob.constraints) if opt.prob is not None else 0
                        ),
                        "selected_keys": len(opt.selected_keys),
                    },
                }
            )
            validate_solution(
                args.solver,
                objective,
                solution_nodes,
                violations,
                pulp_status,
                solution_status,
            )

            if args.detailed_solution:
                result["solution_detail"] = solution_details(solution, autop.gm.graph)
                if opt.decision_vars:
                    contributions = cost_contributions(opt)
                    result["cost_contributions"] = contributions
                    result["cost_contribution_sum"] = sum(
                        row["total"] for row in contributions
                    )
            result["status"] = "success"
    except Exception as error:
        result["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        result["elapsed_s"] = time.perf_counter() - started
        result["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "output": str(args.output),
                    "objective": result.get("objective"),
                    "placement_sha256": result.get("placement_sha256"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
