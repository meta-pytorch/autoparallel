"""Compare the Lagrangian memory-constrained approximate solve against the LP
(relaxation) optimum across a sweep of parameter-memory budgets.

The optimizer (the expensive build) is constructed ONCE; each budget only
re-runs the cheap solves. For every budget factor `high` (with low=0):
  - LP: set the memory constraint and solve the (integral) relaxation -> the
    exact constrained optimum (gold standard).
  - Lagrangian approx: fold lambda * ratio into the unaries and bisect lambda
    until the achieved memory lands in the same [low, high] budget.
The two solvers are pinned to the SAME numeric budget (read back from the LP's
constraint rows) so the comparison is apples-to-apples.

Env: MODEL_TYPE (1b|8b), MESH ("8,8"), N_LAYERS (0=default), SEQLEN,
HIGH_FACTORS (comma list, default sweep), BP_ITERS.
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


def log(msg):
    print(msg, flush=True)


_PATCHES = [
    patch("torch.cuda.device_count", lambda: 8),
    patch("torch.cuda.get_device_name", lambda *a, **k: "H100"),
    patch("torch.cuda.get_device_capability", lambda *a, **k: (9, 0)),
    patch(
        "torch.cuda.get_device_properties",
        lambda *a, **k: type(
            "P",
            (),
            {
                "major": 9,
                "minor": 0,
                "name": "H100",
                "total_memory": 80 * 1024**3,
                "multi_processor_count": 132,
            },
        )(),
    ),
]
for p in _PATCHES:
    p.start()

MODEL_TYPE = os.environ.get("MODEL_TYPE", "1b")
N_LAYERS = int(os.environ.get("N_LAYERS", "0"))
SEQLEN = int(os.environ.get("SEQLEN", str(2048 * 4)))
MESH_SHAPE = tuple(int(x) for x in os.environ.get("MESH", "8,8").split(","))
BP_ITERS = int(os.environ.get("BP_ITERS", "120"))
HIGH_FACTORS = [
    float(x)
    for x in os.environ.get(
        "HIGH_FACTORS", "0.0156,0.03125,0.0625,0.125,0.25,0.5,1.0"
    ).split(",")
]
# On budgets where the LP relaxation is fractional (its optimum is an
# unachievable lower bound) also solve the true ILP to report the achievable gap.
RUN_ILP = os.environ.get("RUN_ILP", "0") == "1"
ILP_TIMEOUT = float(os.environ.get("ILP_TIMEOUT", "300"))

world_size = 1
for d in MESH_SHAPE:
    world_size *= d

_NAMES = {1: ("dp",), 2: ("dp", "tp"), 3: ("dp", "cp", "tp")}
mesh_names = _NAMES[len(MESH_SHAPE)]
fake_store = FakeStore()
torch.distributed.init_process_group(
    "fake", store=fake_store, rank=0, world_size=world_size
)
mesh = torch.distributed.device_mesh.init_device_mesh(
    "cuda", MESH_SHAPE, mesh_dim_names=mesh_names
)

vocab_size = 128256
batch_size = int(os.environ.get("BATCH", str(2 * mesh.shape[0])))
seqlen = SEQLEN


def model_fn():
    args = TransformerModelArgs(
        dim=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.5,
        multiple_of=256,
        rope_theta=500000,
        vocab_size=vocab_size,
        max_seq_len=seqlen,
    )
    if MODEL_TYPE == "8b":
        args = TransformerModelArgs(
            dim=4096,
            n_layers=32,
            n_heads=32,
            n_kv_heads=8,
            ffn_dim_multiplier=1.3,
            multiple_of=1024,
            rope_theta=500000,
            vocab_size=vocab_size,
            max_seq_len=seqlen,
        )
    if N_LAYERS:
        args.n_layers = N_LAYERS
    with torch.device("meta"):
        return Transformer(args)


def input_fn():
    return torch.randint(0, vocab_size, (batch_size, seqlen), device="cuda")


set_nccl_topo_config(detect_nccl_topo_config(mesh))
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

log(
    f"model={MODEL_TYPE} layers={N_LAYERS or 'default'} mesh={MESH_SHAPE} "
    f"world={world_size} seqlen={seqlen} bp_iters={BP_ITERS}"
)

# ---- build once ----
t = time.perf_counter()
autop = AutoParallel(model_fn(), input_fn, mesh, mp, repeated_subgraphs=True)
autop.__enter__()
ndim = mesh.ndim
x_sharding = (Shard(0),) + (Replicate(),) * (ndim - 1)
out_sharding = (Shard(0), Shard(2)) if ndim == 2 else x_sharding
# Build with a LOOSE budget so the approx build does not pin params to the
# min-ratio (fully-sharded) choices; the per-budget sweep overrides the budget
# numerically afterward. (A tight default would prune param strategies at build
# time and freeze the achievable memory.)
autop.add_parameter_memory_constraint(low=0.0, high=1.0)
autop.add_input_constraints([x_sharding])
autop.add_output_constraints([out_sharding])
opt = autop.sharding_optimizer
log(
    f"[build] optimizer ready in {time.perf_counter() - t:.2f}s "
    f"vars={len(opt.pulp_variables)} nodes={len(opt.nodes)}"
)

# build the approximate solver once (ratios / factor graph / mem unary cached)
t = time.perf_counter()
approx = ApproximateShardingSolver(opt, bp_iters=BP_ITERS)
approx._build_problem()
approx._build_factors()
approx._build_mem_unary()
log(
    f"[build] approx solver ready in {time.perf_counter() - t:.2f}s "
    f"groups={len(approx.groups)} "
    f"params={len(approx._memory['param_idxs']) if approx._memory else 0}"
)
opt._set_objective()


def lp_budget():
    """Read back the exact [low, high] the LP applied, so approx uses the same."""
    ch = opt.prob.constraints["memory_constraint_high"]
    cl = opt.prob.constraints["memory_constraint_low"]
    return -cl.constant, -ch.constant


log("\n" + "=" * 110)
log(
    f"{'high_f':>8} | {'budget':>16} | {'LP obj':>12} {'frac':>7} {'LP s':>6} | "
    f"{'approx obj':>12} {'mem':>7} {'lam':>9} {'feas':>5} {'s':>5} | "
    f"{'gap/LP':>7} {'ILP obj':>12} {'gap/ILP':>8}"
)
log("-" * 110)

rows = []
for hf in HIGH_FACTORS:
    opt._memory_constraint = (0.0, hf)
    t = time.perf_counter()
    lp = opt.solve_lp_relaxation(verbose=False, extract=False)
    lp_s = time.perf_counter() - t
    lp_obj = lp["objective"]
    frac = f"{lp['n_fractional']}/{lp['n_vars']}"
    blow, bhigh = lp_budget()

    approx._memory["budget_low"] = blow
    approx._memory["budget_high"] = bhigh
    approx._memory["tight"] = abs(bhigh - blow) < 1e-9
    t = time.perf_counter()
    res = approx.solve_lagrangian(blow, bhigh, max_iter=24)
    ap_s = time.perf_counter() - t
    ap_obj = res["objective"]
    gap = (ap_obj - lp_obj) / lp_obj * 100 if lp_obj else float("nan")

    ilp_obj, gap_ilp = None, None
    if RUN_ILP and lp["n_fractional"] > 0:
        opt._set_objective()
        opt._apply_memory_constraint()
        opt.prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=ILP_TIMEOUT))
        ilp_obj = pulp.value(opt.prob.objective)
        gap_ilp = (ap_obj - ilp_obj) / ilp_obj * 100 if ilp_obj else float("nan")

    rows.append((hf, lp_obj, ap_obj, gap, res["feasible"], ilp_obj, gap_ilp))
    log(
        f"{hf:>8.4g} | [{blow:>6.3f},{bhigh:>7.3f}] | {lp_obj:>12.1f} {frac:>7} "
        f"{lp_s:>5.1f}s | {ap_obj:>12.1f} {res['memory']:>7.3f} {res['lam']:>9.4g} "
        f"{str(res['feasible']):>5} {ap_s:>4.1f}s | {gap:>+6.2f}% "
        f"{('%.1f' % ilp_obj) if ilp_obj else '-':>12} "
        f"{('%+.2f%%' % gap_ilp) if gap_ilp is not None else '-':>8}"
    )

log("=" * 110)
gaps = [r[3] for r in rows if r[1]]
feas = [r[4] for r in rows]
if gaps:
    log(
        f"gap vs LP: mean={sum(gaps)/len(gaps):+.2f}% max={max(gaps):+.2f}% "
        f"min={min(gaps):+.2f}%  feasible={sum(feas)}/{len(feas)}"
    )
gi = [r[6] for r in rows if r[6] is not None]
if gi:
    log(
        f"gap vs ILP (fractional-LP budgets): mean={sum(gi)/len(gi):+.2f}% "
        f"max={max(gi):+.2f}%"
    )
