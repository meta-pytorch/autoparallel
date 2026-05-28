import argparse
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

sys.path.insert(0, "/home/wangkj/workspace/torchtitan")

from torchtitan.models.llama3 import llama3_configs  # noqa: E402

from autoparallel.api import AutoParallel
from autoparallel.cost_models.collective_runtime_estimation import set_nccl_topo_config
from autoparallel.cost_models.nccl_cost_model import detect_nccl_topo_config


WORLD_SIZE = 64
SEQ_LEN = 256
GLOBAL_BATCH = 64
MESHES = {
    1: ((64,), ("dp",)),
    2: ((8, 8), ("dp", "tp")),
    3: ((4, 4, 4), ("dp", "tp", "cp")),
    4: ((4, 4, 2, 2), ("dp", "tp", "cp", "ep")),
}


def init_dist():
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            "fake", store=FakeStore(), rank=0, world_size=WORLD_SIZE
        )


def flatten_profile(model_key, mesh_ndim, profile, total_wall_s, solve_ran):
    model = profile["model"]
    timings = profile["timings"]
    strategies = profile["strategies"]
    ilp = profile["ilp"]
    solve = profile.get("last_solve", {})
    return {
        "model_key": model_key,
        "mesh_ndim": mesh_ndim,
        "mesh_shape": "x".join(map(str, profile["mesh"]["shape"])),
        "mesh_size": profile["mesh"]["size"],
        "parameter_numel": model["parameter_numel"],
        "parameter_b": model["parameter_numel"] / 1_000_000_000,
        "parameter_gib": model["parameter_bytes"] / (1024**3),
        "graph_nodes": model["graph_nodes"],
        "tensor_nodes": model["tensor_nodes"],
        "parameter_nodes": model["parameter_nodes"],
        "strategy_options": strategies["strategy_options"],
        "option_tuples": strategies["option_tuples"],
        "max_strategies_per_node": strategies["max_strategies_per_node"],
        "unique_ilp_vars": ilp["unique_variables"],
        "logical_decision_vars": ilp["logical_decision_variables"],
        "cluster_copied_decision_vars": ilp["cluster_copied_decision_variables"],
        "constraints_init": ilp["constraints"],
        "constraints_presolve": profile.get("constraints_presolve", ilp["constraints"]),
        "constraints_solve": solve.get("constraints", ""),
        "strategy_enumeration_s": timings["strategy_enumeration_s"],
        "compute_cost_estimation_s": timings["compute_cost_estimation_s"],
        "edge_cost_estimation_s": timings["edge_cost_estimation_s"],
        "cost_estimation_s": timings["cost_estimation_s"],
        "decision_var_build_s": timings["decision_var_build_s"],
        "decision_var_overhead_s": timings["decision_var_overhead_s"],
        "ilp_construction_s": timings["ilp_construction_s"],
        "validation_s": timings["validation_s"],
        "objective_s": solve.get("objective_s", ""),
        "solve_s": solve.get("solve_s", ""),
        "extract_s": solve.get("extract_s", ""),
        "optimizer_pipeline_s": solve.get(
            "pipeline_total_s",
            timings["init_total_s"],
        ),
        "total_wall_s": total_wall_s,
        "objective": solve.get("objective", ""),
        "status": solve.get("status", "NotSolved"),
        "solve_ran": solve_ran,
    }


def run_one(model_key, mesh_ndim, skip_solve=False):
    init_dist()
    mesh_shape, mesh_dim_names = MESHES[mesh_ndim]
    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda", mesh_shape, mesh_dim_names=mesh_dim_names
    )
    set_nccl_topo_config(detect_nccl_topo_config(mesh))

    config = llama3_configs[model_key](attn_backend="sdpa")
    config.rope.max_seq_len = SEQ_LEN
    with torch.device("meta"):
        model = config.build()

    def input_fn():
        return torch.randint(
            0,
            config.vocab_size,
            (GLOBAL_BATCH, SEQ_LEN),
            device="cuda",
        )

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    t0 = time.perf_counter()
    with AutoParallel(
        model,
        input_fn,
        mesh,
        mp_policy,
        repeated_subgraphs=True,
    ) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)
        input_sharding = (Shard(0),) + (Replicate(),) * (mesh.ndim - 1)
        if mesh.ndim == 1:
            output_sharding = (Shard(0),)
        else:
            output_sharding = (Shard(0), Shard(2)) + (Replicate(),) * (
                mesh.ndim - 2
            )
        autop.add_input_constraints([input_sharding])
        autop.add_output_constraints([output_sharding])
        autop.sharding_optimizer.profile["constraints_presolve"] = len(
            autop.sharding_optimizer.prob.constraints
        )
        if not skip_solve:
            autop.optimize_placement(verbose=False)
        profile = autop.sharding_optimizer.profile
    return flatten_profile(
        model_key,
        mesh_ndim,
        profile,
        time.perf_counter() - t0,
        solve_ran=not skip_solve,
    )


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def load_rows(path):
    rows = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                row.setdefault(
                    "constraints_presolve",
                    row.get("constraints_solve") or row.get("constraints_init"),
                )
                row.setdefault("solve_ran", row.get("solve_s", "") != "")
                rows.append(row)
    rows.sort(key=lambda r: (r["mesh_ndim"], r["parameter_numel"]))
    return rows


def write_csv(rows, path):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def nice(v):
    if v >= 1_000_000_000:
        return f"{v / 1_000_000_000:.1f}B"
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v / 1_000:.1f}K"
    if v >= 10:
        return f"{v:.0f}"
    return f"{v:.2g}"


def write_svg(rows, path, x_key, series_key, title):
    metrics = [
        ("strategy_enumeration_s", "strategy enum (s)"),
        ("cost_estimation_s", "cost estimation (s)"),
        ("ilp_construction_s", "ILP construction (s)"),
        ("objective_s", "objective build (s)"),
        ("solve_s", "solve (s)"),
        ("optimizer_pipeline_s", "pipeline total (s)"),
        ("unique_ilp_vars", "unique ILP vars"),
        ("constraints_presolve", "constraints"),
    ]
    width = 1600
    height = 1000
    panel_w = 360
    panel_h = 180
    margin_l = 62
    margin_t = 120
    gap_x = 30
    gap_y = 50
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]

    def sx(x, xs, px):
        lo, hi = min(xs), max(xs)
        if lo == hi:
            return px + panel_w / 2
        return px + (x - lo) / (hi - lo) * panel_w

    def sy(y, ys, py):
        positives = [v for v in ys if v > 0]
        lo = min(positives)
        hi = max(positives)
        if lo == hi:
            return py + panel_h / 2
        return py + panel_h - (math.log10(max(y, lo)) - math.log10(lo)) / (
            math.log10(hi) - math.log10(lo)
        ) * panel_h

    series_values = sorted({r[series_key] for r in rows})
    x_values = sorted({float(r[x_key]) for r in rows})
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="32" y="34" font-family="sans-serif" font-size="22" font-weight="700">{title}</text>',
        '<text x="32" y="58" font-family="sans-serif" font-size="12" fill="#475569">Y axes are log scale. Missing series points timed out or were not run.</text>',
    ]
    for i, value in enumerate(series_values):
        x = 32 + (i % 8) * 180
        y = 84 + (i // 8) * 20
        svg.append(
            f'<line x1="{x}" y1="{y}" x2="{x + 28}" y2="{y}" stroke="{colors[i % len(colors)]}" stroke-width="3"/>'
        )
        svg.append(
            f'<text x="{x + 36}" y="{y + 4}" font-family="sans-serif" font-size="12" fill="#334155">{series_key}={value}</text>'
        )

    for idx, (metric, label) in enumerate(metrics):
        col = idx % 4
        row = idx // 4
        px = margin_l + col * (panel_w + gap_x)
        py = margin_t + row * (panel_h + gap_y)
        ys = [
            float(r[metric])
            for r in rows
            if r.get(metric) not in {"", None} and float(r[metric]) > 0
        ]
        if not ys:
            continue
        svg.extend(
            [
                f'<text x="{px}" y="{py - 14}" font-family="sans-serif" font-size="14" font-weight="700">{label}</text>',
                f'<rect x="{px}" y="{py}" width="{panel_w}" height="{panel_h}" fill="#f8fafc" stroke="#cbd5e1"/>',
                f'<line x1="{px}" y1="{py + panel_h}" x2="{px + panel_w}" y2="{py + panel_h}" stroke="#64748b"/>',
                f'<line x1="{px}" y1="{py}" x2="{px}" y2="{py + panel_h}" stroke="#64748b"/>',
                f'<text x="{px - 50}" y="{py + 12}" font-family="sans-serif" font-size="10" fill="#64748b">{nice(max(ys))}</text>',
                f'<text x="{px - 50}" y="{py + panel_h}" font-family="sans-serif" font-size="10" fill="#64748b">{nice(min(ys))}</text>',
            ]
        )
        for xv in x_values:
            svg.append(
                f'<text x="{sx(xv, x_values, px) - 16}" y="{py + panel_h + 18}" font-family="sans-serif" font-size="10" fill="#64748b">{nice(xv)}</text>'
            )
        for sidx, series in enumerate(series_values):
            pts = sorted(
                [r for r in rows if r[series_key] == series],
                key=lambda r: float(r[x_key]),
            )
            color = colors[sidx % len(colors)]
            coords = [
                (
                    sx(float(r[x_key]), x_values, px),
                    sy(float(r[metric]), ys, py),
                )
                for r in pts
                if r.get(metric) not in {"", None} and float(r[metric]) > 0
            ]
            if len(coords) >= 2:
                svg.append(
                    '<polyline points="'
                    + " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
                    + f'" fill="none" stroke="{color}" stroke-width="2.4"/>'
                )
            for x, y in coords:
                svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>')
    svg.append("</svg>")
    Path(path).write_text("\n".join(svg))


def plot(jsonl, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(jsonl)
    write_csv(rows, out_dir / "real_llama3_optimizer_sweep.csv")
    write_svg(
        rows,
        out_dir / "real_llama3_by_model_size.svg",
        "parameter_b",
        "mesh_ndim",
        "Real Llama3 optimizer profile vs model size",
    )
    write_svg(
        rows,
        out_dir / "real_llama3_by_mesh_dim.svg",
        "mesh_ndim",
        "model_key",
        "Real Llama3 optimizer profile vs mesh dimension",
    )


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run-one")
    run.add_argument("--model-key", choices=llama3_configs.keys(), required=True)
    run.add_argument("--mesh-ndim", type=int, choices=MESHES.keys(), required=True)
    run.add_argument("--out-jsonl", required=True)
    run.add_argument("--skip-solve", action="store_true")
    plot_cmd = sub.add_parser("plot")
    plot_cmd.add_argument("--jsonl", required=True)
    plot_cmd.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    )
    logging.getLogger("autoparallel.optimize_sharding").setLevel(logging.INFO)

    if args.cmd == "run-one":
        row = run_one(args.model_key, args.mesh_ndim, skip_solve=args.skip_solve)
        append_jsonl(args.out_jsonl, row)
        print(json.dumps(row, sort_keys=True))
    else:
        plot(args.jsonl, args.out_dir)


if __name__ == "__main__":
    main()
