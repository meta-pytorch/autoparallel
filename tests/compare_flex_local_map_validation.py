# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import math
import re
from pathlib import Path

import torch
from torch.testing._comparison import default_tolerances

CASES = (
    "plain-sharded",
    "plain-replicated-counts",
    "flex-default",
    "flex-replicated-counts",
    "flex-auto",
)
PYTEST_STAGES = (
    ("placement_options", "Public interface and placement"),
    ("optimize_placement", "Trace, solve, finalize, and apply"),
    ("canonical_graph", "Canonical graph comparison"),
    ("serialization", "Serialization round trip and apply"),
    ("pointwise_4gpu", "Real multi-GPU pointwise E2E"),
)
PYTEST_COUNT = re.compile(
    r"\b\d+ (?:passed|failed|errors?|skipped|xfailed|xpassed|deselected)\b"
)


def _output_paths(rank_dir):
    indexed = []
    for path in rank_dir.glob("mb*_output.pt"):
        match = re.fullmatch(r"mb(\d+)_output", path.stem)
        if match:
            indexed.append((int(match.group(1)), path))
    indexed.sort()
    indices = [index for index, _ in indexed]
    if indices != list(range(len(indices))):
        raise ValueError(f"non-contiguous microbatch outputs in {rank_dir}: {indices}")
    return [path for _, path in indexed]


def _load_case(root, case, rank):
    rank_dir = root / case / "0" / f"rank_{rank}"
    output_paths = _output_paths(rank_dir)
    if not output_paths:
        raise FileNotFoundError(f"no microbatch outputs in {rank_dir}")
    return {
        "model_state": torch.load(
            rank_dir / "model_state.pt", map_location="cpu", weights_only=True
        ),
        "global_model_state": torch.load(
            rank_dir / "global_model_state.pt",
            map_location="cpu",
            weights_only=True,
        ),
        "full_batch": torch.load(
            rank_dir / "full_batch.pt", map_location="cpu", weights_only=True
        ),
        "outputs": [
            torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
            for path in output_paths
        ],
        "gradients": torch.load(
            rank_dir / "gradients.pt", map_location="cpu", weights_only=True
        ),
        "global_gradients": torch.load(
            rank_dir / "global_gradients.pt", map_location="cpu", weights_only=True
        ),
        "final_buffers": torch.load(
            rank_dir / "final_buffers.pt", map_location="cpu", weights_only=True
        ),
        "global_final_buffers": torch.load(
            rank_dir / "global_final_buffers.pt",
            map_location="cpu",
            weights_only=True,
        ),
    }


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _flatten(value[key], f"{prefix}/{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _flatten(item, f"{prefix}/{index}")
    else:
        yield prefix, value


def _compare_tensors(actual, expected, exact):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return False, math.inf, math.inf, "shape or dtype mismatch"
    if torch.equal(actual, expected):
        return True, 0.0, 0.0, None

    rtol, atol = (0.0, 0.0) if exact else default_tolerances(actual, expected)
    actual = actual.flatten()
    expected = expected.flatten()
    chunk_size = 4 * 1024 * 1024
    mismatches = 0
    max_abs = 0.0
    max_rel = 0.0
    for start in range(0, actual.numel(), chunk_size):
        lhs = actual[start : start + chunk_size]
        rhs = expected[start : start + chunk_size]
        difference = (lhs - rhs).abs().float()
        mismatches += (~torch.isclose(lhs, rhs, rtol=rtol, atol=atol)).sum().item()
        if difference.numel():
            max_abs = max(max_abs, difference.max().item())
            denominator = rhs.abs().float()
            relative = torch.where(
                denominator == 0,
                torch.where(difference == 0, 0.0, torch.inf),
                difference / denominator,
            )
            max_rel = max(max_rel, relative.max().item())
    error = None
    if mismatches:
        error = f"{mismatches} values outside rtol={rtol}, atol={atol}"
    return not mismatches, max_abs, max_rel, error


def _compare_component(actual, expected, exact):
    actual_entries = dict(_flatten(actual))
    expected_entries = dict(_flatten(expected))
    names = sorted(set(actual_entries) | set(expected_entries))
    failures = []
    max_abs = 0.0
    max_rel = 0.0
    for name in names:
        if name not in actual_entries or name not in expected_entries:
            failures.append({"name": name, "error": "missing entry"})
            continue
        lhs = actual_entries[name]
        rhs = expected_entries[name]
        passed, entry_abs, entry_rel, error = _compare_tensors(lhs, rhs, exact)
        if not passed:
            failures.append({"name": name, "error": error})
        max_abs = max(max_abs, entry_abs)
        max_rel = max(max_rel, entry_rel)
    return {
        "entries": len(names),
        "failures": failures,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "passed": not failures,
    }


def _compare_pair(
    root, actual_case, expected_case, exact, world_size, component_names=None
):
    ranks = []
    for rank in range(world_size):
        actual = _load_case(root, actual_case, rank)
        expected = _load_case(root, expected_case, rank)
        names = component_names or tuple(actual)
        components = {
            name: _compare_component(actual[name], expected[name], exact)
            for name in names
        }
        ranks.append(
            {
                "rank": rank,
                "components": components,
                "passed": all(result["passed"] for result in components.values()),
            }
        )
    return {
        "actual": actual_case,
        "expected": expected_case,
        "comparison": "exact" if exact else "torch.testing.assert_close defaults",
        "ranks": ranks,
        "passed": all(rank["passed"] for rank in ranks),
    }


def _read_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"missing validation artifact: {path}")
    return json.loads(path.read_text())


def _read_exit_code(root, stage):
    path = root / f"{stage}.exit_code"
    if not path.is_file():
        raise FileNotFoundError(f"missing validation artifact: {path}")
    return int(path.read_text().strip())


def _pytest_result(root, stage, label):
    log_path = root / f"{stage}.log"
    if not log_path.is_file():
        raise FileNotFoundError(f"missing validation artifact: {log_path}")
    summary = None
    for line in log_path.read_text(errors="replace").splitlines():
        if PYTEST_COUNT.search(line):
            summary = line.strip(" =")
    if summary is None:
        raise ValueError(f"no pytest summary found in {log_path}")
    exit_code = _read_exit_code(root, stage)
    return {
        "stage": stage,
        "label": label,
        "summary": summary,
        "exit_code": exit_code,
        "passed": exit_code == 0,
    }


def _display_range(values):
    low = min(values)
    high = max(values)
    return str(low) if low == high else f"{low}-{high}"


def _moe_result(root, case, world_size):
    case_root = root / case / "0"
    rank_dirs = sorted(
        path
        for path in case_root.glob("rank_*")
        if path.is_dir() and path.name.removeprefix("rank_").isdigit()
    )
    rank_indices = [int(path.name.removeprefix("rank_")) for path in rank_dirs]
    output_counts = [len(_output_paths(path)) for path in rank_dirs]
    log_path = root / f"moe_{case}.log"
    if not log_path.is_file():
        raise FileNotFoundError(f"missing validation artifact: {log_path}")
    success_markers = log_path.read_text(errors="replace").count("All good!")
    exit_code = _read_exit_code(root, f"moe_{case}")
    passed = (
        exit_code == 0
        and rank_indices == list(range(world_size))
        and success_markers == world_size
        and bool(output_counts)
    )
    return {
        "case": case,
        "rank_indices": rank_indices,
        "success_markers": success_markers,
        "outputs_per_rank": output_counts,
        "exit_code": exit_code,
        "passed": passed,
    }


def _comparison_entry_range(comparison, names):
    values = [
        sum(
            result["entries"]
            for name, result in rank["components"].items()
            if name in names
        )
        for rank in comparison["ranks"]
    ]
    return _display_range(values)


def _format_error(value):
    return f"{value:.6g}"


def _write_summary(root, environment, pytest_results, moe_results, result):
    all_checks_passed = (
        all(stage["passed"] for stage in pytest_results)
        and all(case["passed"] for case in moe_results)
        and result["all_passed"]
        and _read_exit_code(root, "compile") == 0
    )
    gpu_descriptions = []
    for gpu in environment["gpus"]:
        memory_mib = gpu["total_memory_bytes"] / (1024 * 1024)
        gpu_descriptions.append(f'{gpu["index"]}: {gpu["name"]} ({memory_mib:.0f} MiB)')

    lines = [
        "# flex_local_map validation report",
        "",
        "All result counts and numerical values below were derived from this run's "
        "logs and saved tensors; none are expected-result constants.",
        "",
        "## Outcome",
        "",
        f'Overall result: **{"PASS" if all_checks_passed else "FAIL"}**',
        "",
        "## Measured environment",
        "",
        "| Field | Measured value |",
        "|---|---|",
        f'| Timestamp (UTC) | `{environment["timestamp_utc"]}` |',
        f'| Branch | `{environment["branch"]}` |',
        f'| HEAD | `{environment["head"]}` |',
        f'| Base | `{environment["base_commit"]}` (`{environment["base_ref"]}`) |',
        f'| Python | `{environment["python"]}` |',
        f'| PyTorch | `{environment["pytorch"]}` |',
        f'| CUDA / NCCL | `{environment["cuda"]}` / `{environment["nccl"]}` |',
        f'| CUDA devices | {environment["cuda_device_count"]} |',
        f'| GPUs | {"<br>".join(gpu_descriptions)} |',
        "",
        "The exact test inputs are recorded in `execution_config.json`; environment, "
        "git state, committed diff, source snapshots, and checksums are retained next "
        "to this report.",
        "",
        "## Measured correctness results",
        "",
        "| Stage | Pytest summary | Exit code | Result |",
        "|---|---|---:|---|",
    ]
    for stage in pytest_results:
        status = "pass" if stage["passed"] else "fail"
        lines.append(
            f'| {stage["label"]} | `{stage["summary"]}` | '
            f'{stage["exit_code"]} | {status} |'
        )

    lines.extend(
        [
            "",
            "## Measured distributed MoE results",
            "",
            "| Case | Rank artifacts | `All good!` markers | Outputs/rank | Exit code | Result |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for case in moe_results:
        status = "pass" if case["passed"] else "fail"
        rank_count = len(case["rank_indices"])
        outputs = _display_range(case["outputs_per_rank"])
        lines.append(
            f'| `{case["case"]}` | {rank_count} | {case["success_markers"]} | '
            f'{outputs} | {case["exit_code"]} | {status} |'
        )

    lines.extend(
        [
            "",
            "## Measured numerical comparisons",
            "",
            "| Actual vs expected | Mode | Ranks | State/rank | Inputs/rank | "
            "Outputs/rank | Gradients/rank | Buffers/rank | Max abs/rel | Result |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    groups = {
        "state": ("model_state", "global_model_state"),
        "inputs": ("full_batch",),
        "outputs": ("outputs",),
        "gradients": ("gradients", "global_gradients"),
        "buffers": ("final_buffers", "global_final_buffers"),
    }
    for comparison in result["comparisons"]:
        components = [
            component
            for rank in comparison["ranks"]
            for component in rank["components"].values()
        ]
        max_abs = max(component["max_abs_error"] for component in components)
        max_rel = max(component["max_rel_error"] for component in components)
        values = {
            name: _comparison_entry_range(comparison, component_names)
            for name, component_names in groups.items()
        }
        status = "pass" if comparison["passed"] else "fail"
        lines.append(
            f'| `{comparison["actual"]}` vs `{comparison["expected"]}` | '
            f'{comparison["comparison"]} | {len(comparison["ranks"])} | '
            f'{values["state"]} | {values["inputs"]} | {values["outputs"]} | '
            f'{values["gradients"]} | {values["buffers"]} | '
            f"`{_format_error(max_abs)} / {_format_error(max_rel)}` | {status} |"
        )

    lines.extend(
        [
            "",
            "Full per-rank/component comparison data is in "
            "`numerics_comparison.json`. Exact commands, logs, and exit codes are in "
            "the corresponding `*.command.txt`, `*.log`, and `*.exit_code` files.",
            "",
            "No latency, throughput, memory, speedup, or other performance result was measured.",
            "",
            f"Results directory: `{root.resolve()}`",
            "",
        ]
    )
    summary_path = root / "summary.md"
    summary_path.write_text("\n".join(lines))
    return summary_path, all_checks_passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--world-size", type=int)
    args = parser.parse_args()

    config = _read_json(args.results_dir / "execution_config.json")
    environment = _read_json(args.results_dir / "environment.json")
    world_size = args.world_size or config["world_size"]

    missing = [case for case in CASES if not (args.results_dir / case / "0").is_dir()]
    if missing:
        raise FileNotFoundError(f"missing case results: {missing}")

    comparisons = [
        _compare_pair(
            args.results_dir,
            "flex-default",
            "plain-sharded",
            exact=True,
            world_size=world_size,
        ),
        _compare_pair(
            args.results_dir,
            "flex-replicated-counts",
            "plain-replicated-counts",
            exact=True,
            world_size=world_size,
        ),
        _compare_pair(
            args.results_dir,
            "plain-replicated-counts",
            "plain-sharded",
            exact=False,
            world_size=world_size,
            component_names=(
                "global_model_state",
                "full_batch",
                "outputs",
                "global_gradients",
                "global_final_buffers",
            ),
        ),
        _compare_pair(
            args.results_dir,
            "flex-auto",
            "plain-sharded",
            exact=False,
            world_size=world_size,
            component_names=(
                "global_model_state",
                "full_batch",
                "outputs",
                "global_gradients",
                "global_final_buffers",
            ),
        ),
    ]
    result = {
        "comparisons": comparisons,
        "all_passed": all(comparison["passed"] for comparison in comparisons),
    }
    output_path = args.results_dir / "numerics_comparison.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")

    pytest_results = [
        _pytest_result(args.results_dir, stage, label) for stage, label in PYTEST_STAGES
    ]
    full_pytest_path = args.results_dir / "full_pytest.log"
    if config.get("full_pytest_requested") or full_pytest_path.exists():
        pytest_results.append(
            _pytest_result(args.results_dir, "full_pytest", "Full repository pytest")
        )
    moe_results = [_moe_result(args.results_dir, case, world_size) for case in CASES]
    summary_path, all_checks_passed = _write_summary(
        args.results_dir,
        environment,
        pytest_results,
        moe_results,
        result,
    )
    print(f"Numerical details: {output_path.resolve()}")
    print(f"Validation report: {summary_path.resolve()}")
    if not all_checks_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
