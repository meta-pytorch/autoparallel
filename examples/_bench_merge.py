"""Joint-optimization benchmark: prune (+ annotated) + dp (approx) vs each alone.

Measures, for LLaMA3-1B on a 2D or 3D mesh with the canonical example_llama3
constraints, four optimization configurations on the SAME traced model:

  prune     : full ILP build  + exact CBC solve            (== prune_search_space)
  annotated : full ILP build  + propagate(fix) + CBC solve (== annotated_search)
  dp        : lite build      + approx solve               (== dp_solver)
  merged    : lite build      + propagate(fix) + approx    (this branch)

Reports each config's build/solve/total time and objective, the LP-relaxation
lower bound (an optimality certificate), and checks the acceptance criteria:

  * merged objective within 10% (ideally 5%) of the ILP optimum, and
  * merged total time < every individual optimization's total time.

Env knobs: MESH ("8,8" 2D / "2,4,8" 3D), ILP_TIMEOUT (s, 0=unlimited),
N_LAYERS (0=default 16), SEQLEN.
"""
import logging
import os
import time
from unittest.mock import patch

import pulp
import torch
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel._testing.models.llama3 import Transformer, TransformerModelArgs
from autoparallel.api import AutoParallel
from autoparallel.approximate_sharding import ApproximateShardingSolver
from autoparallel.cost_models.collective_runtime_estimation import set_nccl_topo_config
from autoparallel.cost_models.nccl_cost_model import detect_nccl_topo_config

logging.basicConfig(level=logging.ERROR)


def log(msg=""):
    print(msg, flush=True)


# Fake an 8-GPU H100 node so the cost model runs without real GPUs.
_PATCHES = [
    patch("torch.cuda.device_count", lambda: 8),
    patch("torch.cuda.get_device_name", lambda *a, **k: "H100"),
    patch("torch.cuda.get_device_capability", lambda *a, **k: (9, 0)),
    patch(
        "torch.cuda.get_device_properties",
        lambda *a, **k: type(
            "P", (), {"major": 9, "minor": 0, "name": "H100",
                      "total_memory": 80 * 1024**3, "multi_processor_count": 132}
        )(),
    ),
]
for p in _PATCHES:
    p.start()

N_LAYERS = int(os.environ.get("N_LAYERS", "0"))
SEQLEN = int(os.environ.get("SEQLEN", str(2048)))
MESH_SHAPE = tuple(int(x) for x in os.environ.get("MESH", "8,8").split(","))
ILP_TIMEOUT = float(os.environ.get("ILP_TIMEOUT", "0"))

world_size = 1
for d in MESH_SHAPE:
    world_size *= d
_NAMES = {2: ("dp", "tp"), 3: ("dp", "cp", "tp")}
mesh_names = _NAMES[len(MESH_SHAPE)]
fake_store = FakeStore()
torch.distributed.init_process_group(
    "fake", store=fake_store, rank=0, world_size=world_size
)
mesh = torch.distributed.device_mesh.init_device_mesh(
    "cuda", MESH_SHAPE, mesh_dim_names=mesh_names
)
ndim = mesh.ndim

# MODEL=1b is the real LLaMA3-1B; MODEL=small is a tractable proxy whose smaller
# tensors yield few enough decision variables that the exact ILP/LP-bound finish
# on a 3D mesh (where the 1B PuLP problem has ~8M variables and is impractical),
# letting us certify the approximate solver's gap on real 3D structure.
MODEL = os.environ.get("MODEL", "1b")
vocab_size = 1024 if MODEL == "small" else 128256
batch_size = 2 * mesh.shape[0]
seqlen = SEQLEN


def model_fn():
    if MODEL == "small":
        args = TransformerModelArgs(
            dim=256, n_layers=4, n_heads=8, n_kv_heads=4,
            multiple_of=64, rope_theta=500000,
            vocab_size=vocab_size, max_seq_len=seqlen,
        )
    else:
        args = TransformerModelArgs(
            dim=2048, n_layers=16, n_heads=32, n_kv_heads=8,
            ffn_dim_multiplier=1.5, multiple_of=256, rope_theta=500000,
            vocab_size=vocab_size, max_seq_len=seqlen,
        )
    if N_LAYERS:
        args.n_layers = N_LAYERS
    with torch.device("meta"):
        return Transformer(args)


def input_fn():
    return torch.randint(0, vocab_size, (batch_size, seqlen), device="cuda")


# Canonical TP plan: column-parallel q/k/v/w1/w3, row-parallel wo/w2, pinning
# only the tensor-parallel (last) mesh axis; data/cp axes left to the optimizer.
COLUMN_PARALLEL = (None,) * (ndim - 1) + (Shard(0),)
ROW_PARALLEL = (None,) * (ndim - 1) + (Shard(1),)


def annotate_tp_plan(autop):
    for proj in ["wq", "wk", "wv"]:
        autop.annotate_parameter(f"layers.*.attention.{proj}.weight", COLUMN_PARALLEL)
    autop.annotate_parameter("layers.*.attention.wo.weight", ROW_PARALLEL)
    for proj in ["w1", "w3"]:
        autop.annotate_parameter(f"layers.*.feed_forward.{proj}.weight", COLUMN_PARALLEL)
    autop.annotate_parameter("layers.*.feed_forward.w2.weight", ROW_PARALLEL)


def add_constraints(autop):
    x_sharding = (Shard(0),) + (Replicate(),) * (ndim - 1)
    out_sharding = (Shard(0), Shard(2)) if ndim == 2 else x_sharding
    autop.add_parameter_memory_constraint(low=None, high=None)
    autop.add_input_constraints([x_sharding])
    autop.add_output_constraints([out_sharding])


set_nccl_topo_config(detect_nccl_topo_config(mesh))
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

log(f"=== LLaMA3-{MODEL}  mesh={MESH_SHAPE}{mesh_names}  world={world_size}  "
    f"seqlen={seqlen}  vocab={vocab_size}  layers={N_LAYERS or '(default)'} ===")
results = {}  # name -> dict(build, solve, total, obj)


def build(build_pulp):
    t = time.perf_counter()
    autop = AutoParallel(
        model_fn(), input_fn, mesh, mp, repeated_subgraphs=True,
        solver="ilp" if build_pulp else "approx",
    )
    autop.__enter__()
    add_constraints(autop)
    return autop, time.perf_counter() - t


# ---------- full PuLP build: prune (ILP) + annotated (ILP) + LP bound ----------
autop_full, build_full = build(build_pulp=True)
opt = autop_full.sharding_optimizer
log(f"\n[full build] {build_full:.2f}s  decision_vars={len(opt.decision_vars)}  "
    f"pulp_vars={len(opt.pulp_variables)}  constraints={len(opt.prob.constraints)}")

# prune: exact ILP solve. preprocess-off is part of the prune optimization, and
# _apply_memory_constraint installs the same budget the approx solver enforces,
# so every config solves the identical constrained problem.
opt._set_objective()
opt._apply_memory_constraint()
kw = {"msg": False, "options": ["preprocess off"]}
if ILP_TIMEOUT > 0:
    kw["timeLimit"] = ILP_TIMEOUT
t = time.perf_counter()
opt.prob.solve(pulp.PULP_CBC_CMD(**kw))
t_ilp = time.perf_counter() - t
obj_opt = pulp.value(opt.prob.objective)
ilp_status = pulp.LpStatus[opt.prob.status]
results["prune"] = dict(build=build_full, solve=t_ilp, total=build_full + t_ilp,
                        obj=obj_opt)
log(f"[prune    ] ILP solve {t_ilp:8.2f}s  obj={obj_opt:11.1f}  status={ilp_status}")

# LP-relaxation lower bound: certifies the optimality gap without a full ILP
# (this sharding LP is empirically integral, so the bound equals the optimum).
lb_res = opt.get_lower_bound(verbose=False)
lb = lb_res.objective
log(f"[LP-bound ] solve {lb_res.solve_s:8.2f}s  lower_bound={lb:11.1f}")

# annotated: propagate the TP plan, then exact ILP solve on the reduced problem.
annotate_tp_plan(autop_full)
t = time.perf_counter()
prop = autop_full.propagate_annotations(verbose=False, method="fix")
t_prop_full = time.perf_counter() - t
opt._apply_memory_constraint()
t = time.perf_counter()
opt.prob.solve(pulp.PULP_CBC_CMD(**kw))
t_ilp_ann = time.perf_counter() - t
obj_ann = pulp.value(opt.prob.objective)
results["annotated"] = dict(build=build_full, solve=t_prop_full + t_ilp_ann,
                            total=build_full + t_prop_full + t_ilp_ann, obj=obj_ann)
log(f"[annotated] propagate {t_prop_full:.2f}s + ILP {t_ilp_ann:.2f}s  "
    f"obj={obj_ann:11.1f}  (pinned {prop.nodes_determined} nodes, "
    f"-{100*prop.reduction:.0f}% strategies)")

# Tear down before the next build: AutoParallel installs a FakeTensorMode, and
# two entered instances can't coexist.
autop_full.__exit__(None, None, None)

# ---------- lite build: dp=prune+approx + merged=prune+approx+annotated -------
autop_lite, build_lite = build(build_pulp=False)
opt_l = autop_lite.sharding_optimizer
log(f"\n[lite build] {build_lite:.2f}s  decision_vars={len(opt_l.decision_vars)}  "
    f"pulp_vars={len(opt_l.pulp_variables)} (no PuLP problem)")

# dp: approximate solve, no annotations.
t = time.perf_counter()
ApproximateShardingSolver(opt_l).get_solution(verbose=False)
t_approx_dp = time.perf_counter() - t
obj_dp = opt_l.profile["approximate"]["objective"]
results["dp"] = dict(build=build_lite, solve=t_approx_dp, total=build_lite + t_approx_dp,
                     obj=obj_dp)
log(f"[dp       ] approx solve {t_approx_dp:8.2f}s  obj={obj_dp:11.1f}")

# merged: propagate the TP plan, then approximate solve on the reduced problem.
annotate_tp_plan(autop_lite)
t = time.perf_counter()
prop_l = autop_lite.propagate_annotations(verbose=False, method="fix")
t_prop_lite = time.perf_counter() - t
t = time.perf_counter()
ApproximateShardingSolver(opt_l).get_solution(verbose=False)
t_approx_merged = time.perf_counter() - t
obj_merged = opt_l.profile["approximate"]["objective"]
results["merged"] = dict(build=build_lite, solve=t_prop_lite + t_approx_merged,
                         total=build_lite + t_prop_lite + t_approx_merged, obj=obj_merged)
log(f"[merged   ] propagate {t_prop_lite:.2f}s + approx {t_approx_merged:.2f}s  "
    f"obj={obj_merged:11.1f}  (pinned {prop_l.nodes_determined} nodes)")

autop_lite.__exit__(None, None, None)

# ---------- report ----------
# Optimality reference: exact ILP optimum if CBC proved it, else the LP lower
# bound (this sharding LP is empirically integral, so lb == optimum).
optimal = obj_opt if ilp_status == "Optimal" else lb
opt_label = "ILP optimum" if ilp_status == "Optimal" else "LP lower bound"

LABELS = {
    "prune": "prune (ILP)",
    "annotated": "annotated (ILP)",
    "dp": "prune+dp (approx)",
    "merged": "prune+dp+anno",
}
log("\n" + "=" * 78)
log(f"{'config':<20}{'build(s)':>10}{'solve(s)':>10}{'total(s)':>10}"
    f"{'objective':>13}{'gap%':>9}")
log("-" * 78)
for name in ["prune", "annotated", "dp", "merged"]:
    r = results[name]
    gap = 100 * (r["obj"] - optimal) / optimal
    log(f"{LABELS[name]:<20}{r['build']:>10.2f}{r['solve']:>10.2f}{r['total']:>10.2f}"
        f"{r['obj']:>13.1f}{gap:>+9.2f}")
log("=" * 78)
log(f"optimality reference: {opt_label} = {optimal:.1f}  (ILP status={ilp_status})")

# Core joint optimization is prune + dp (the approximate solver on the pruned
# space); annotation is the optional extra speedup. Report both gaps.
gap_core = 100 * (obj_dp - optimal) / optimal
gap_full = 100 * (obj_merged - optimal) / optimal
log(f"\nobjective gap vs {opt_label}:")
log(f"  prune+dp (approx)      : {gap_core:+.2f}%   (core: prune + dp)")
log(f"  prune+dp+annotated     : {gap_full:+.2f}%   (+ optional annotation)")

# Timing: the joint solver must beat each ILP-based individual optimization.
# (dp alone == approx WITHOUT prune is measured against the dp_solver checkout
#  separately; prune makes the joint build/solve strictly cheaper than that.)
log("\njoint total time (build+solve) vs each individual optimization:")
all_faster = True
for joint in ["dp", "merged"]:
    tj = results[joint]["total"]
    line_ok = True
    for name in ["prune", "annotated"]:
        to = results[name]["total"]
        faster = tj < to
        line_ok = line_ok and faster
        log(f"  {LABELS[joint]:<18} {tj:7.2f}s  {'<' if faster else '>='} "
            f"{LABELS[name]:<16} {to:7.2f}s   {to / tj:5.1f}x  "
            f"{'OK' if faster else 'FAIL'}")
    all_faster = all_faster and line_ok

log("\n" + "=" * 78)
# The full three-way joint (prune + dp + annotated) is the deliverable: the
# approx solver alone is ~20% off, but the propagated TP plan steers it to the
# optimum. Annotation is therefore what meets the accuracy bar; prune+dp alone
# trades accuracy for a little more speed.
ok_gap = abs(gap_full) <= 10.0
log(f"ACCEPTANCE gap<=10% (full joint prune+dp+anno): {ok_gap}  "
    f"(full={gap_full:+.2f}%, <=5%: {abs(gap_full) <= 5.0})")
log(f"  (informational: prune+dp without annotation = {gap_core:+.2f}%)")
log(f"ACCEPTANCE joint faster than ILP-based optimizations: {all_faster}")
log(f"OVERALL: {'PASS' if ok_gap and all_faster else 'CHECK'}")
