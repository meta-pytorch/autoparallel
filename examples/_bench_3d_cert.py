"""3D optimality certificate for the merged solver on full LLaMA3-1B.

The 3D ILP has ~8M binary variables; the exact CBC solve (and even CBC's LP
relaxation) is impractical (a 2.6 GB MPS file; CBC simplex runs for hours). The
LP relaxation is empirically integral for this problem (verified on 2D, where it
equals the exact optimum), so its objective is a tight lower bound on the ILP
optimum. We solve that LP with HiGHS (scipy.optimize.linprog), which handles the
8M-variable sparse LP in minutes, then compare to the approximate solvers.

One full PuLP build feeds: the HiGHS LP lower bound (optimality reference), and
the prune+dp / merged approximate objectives. Reports the certified gaps. Env:
MESH, SEQLEN.
"""
import logging
import os
import time
from unittest.mock import patch

import numpy as np
import pulp
import scipy.sparse as sp
import torch
from scipy.optimize import linprog
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel._testing.models.llama3 import Transformer, TransformerModelArgs
from autoparallel.api import AutoParallel
from autoparallel.approximate_sharding import ApproximateShardingSolver
from autoparallel.cost_models.collective_runtime_estimation import set_nccl_topo_config
from autoparallel.cost_models.nccl_cost_model import detect_nccl_topo_config

logging.basicConfig(level=logging.ERROR)
for fn, val in [("device_count", lambda: 8), ("get_device_name", lambda *a, **k: "H100"),
                ("get_device_capability", lambda *a, **k: (9, 0))]:
    patch(f"torch.cuda.{fn}", val).start()
patch("torch.cuda.get_device_properties", lambda *a, **k: type(
    "P", (), {"major": 9, "minor": 0, "name": "H100",
              "total_memory": 80 * 1024**3, "multi_processor_count": 132})()).start()


def log(m=""):
    print(m, flush=True)


SEQLEN = int(os.environ.get("SEQLEN", "2048"))
MESH_SHAPE = tuple(int(x) for x in os.environ.get("MESH", "2,4,8").split(","))
ws = 1
for d in MESH_SHAPE:
    ws *= d
names = {2: ("dp", "tp"), 3: ("dp", "cp", "tp")}[len(MESH_SHAPE)]
torch.distributed.init_process_group("fake", store=FakeStore(), rank=0, world_size=ws)
mesh = torch.distributed.device_mesh.init_device_mesh("cuda", MESH_SHAPE, mesh_dim_names=names)
ndim = mesh.ndim
vocab_size = 128256
batch_size = 2 * mesh.shape[0]


def model_fn():
    args = TransformerModelArgs(
        dim=2048, n_layers=16, n_heads=32, n_kv_heads=8, ffn_dim_multiplier=1.5,
        multiple_of=256, rope_theta=500000, vocab_size=vocab_size, max_seq_len=SEQLEN)
    with torch.device("meta"):
        return Transformer(args)


def input_fn():
    return torch.randint(0, vocab_size, (batch_size, SEQLEN), device="cuda")


def lp_lower_bound_highs(opt):
    """Solve the LP relaxation (binaries -> [0,1]) of opt.prob with HiGHS and
    return its objective: a tight lower bound on the ILP optimum."""
    variables = opt.prob.variables()
    idx = {v.name: i for i, v in enumerate(variables)}
    n = len(variables)
    c = np.zeros(n)
    for v, coeff in opt.prob.objective.items():
        c[idx[v.name]] += coeff
    rows_eq, cols_eq, data_eq, b_eq = [], [], [], []
    rows_ub, cols_ub, data_ub, b_ub = [], [], [], []
    r_eq = r_ub = 0
    for con in opt.prob.constraints.values():
        rhs = -con.constant
        items = list(con.items())
        if con.sense == pulp.LpConstraintEQ:
            for v, coeff in items:
                rows_eq.append(r_eq); cols_eq.append(idx[v.name]); data_eq.append(coeff)
            b_eq.append(rhs); r_eq += 1
        else:  # LE: a<=b ; GE: a>=b -> -a<=-b
            sign = 1.0 if con.sense == pulp.LpConstraintLE else -1.0
            for v, coeff in items:
                rows_ub.append(r_ub); cols_ub.append(idx[v.name]); data_ub.append(sign * coeff)
            b_ub.append(sign * rhs); r_ub += 1
    A_eq = sp.csr_matrix((data_eq, (rows_eq, cols_eq)), shape=(r_eq, n)) if r_eq else None
    A_ub = sp.csr_matrix((data_ub, (rows_ub, cols_ub)), shape=(r_ub, n)) if r_ub else None
    res = linprog(c, A_ub=A_ub, b_ub=(b_ub or None), A_eq=A_eq, b_eq=(b_eq or None),
                  bounds=(0, 1), method="highs")
    if not res.success:
        raise RuntimeError(f"HiGHS LP failed: {res.message}")
    return res.fun, n, r_eq + r_ub


set_nccl_topo_config(detect_nccl_topo_config(mesh))
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
x = (Shard(0),) + (Replicate(),) * (ndim - 1)
out = (Shard(0), Shard(2)) if ndim == 2 else x

log(f"=== 3D cert (HiGHS): LLaMA3-1B mesh={MESH_SHAPE}{names} seqlen={SEQLEN} ===")
t = time.perf_counter()
autop = AutoParallel(model_fn(), input_fn, mesh, mp, repeated_subgraphs=True, solver="ilp")
autop.__enter__()
autop.add_parameter_memory_constraint(low=None, high=None)
autop.add_input_constraints([x])
autop.add_output_constraints([out])
opt = autop.sharding_optimizer
opt._set_objective()
opt._apply_memory_constraint()
log(f"[build] {time.perf_counter()-t:.1f}s  decision_vars={len(opt.decision_vars)}  "
    f"pulp_vars={len(opt.pulp_variables)}  constraints={len(opt.prob.constraints)}")

# prune+dp (approx, no annotation) on the same problem.
t = time.perf_counter()
ApproximateShardingSolver(opt).get_solution(verbose=False)
prune_dp = opt.profile["approximate"]["objective"]
log(f"[prune+dp]  approx {time.perf_counter()-t:.1f}s  objective={prune_dp:.1f}")

# merged (prune+dp+annotated): propagate the TP plan, then approx-solve.
cp = (None,) * (ndim - 1) + (Shard(0),)
rp = (None,) * (ndim - 1) + (Shard(1),)
for proj in ["wq", "wk", "wv"]:
    autop.annotate_parameter(f"layers.*.attention.{proj}.weight", cp)
autop.annotate_parameter("layers.*.attention.wo.weight", rp)
for proj in ["w1", "w3"]:
    autop.annotate_parameter(f"layers.*.feed_forward.{proj}.weight", cp)
autop.annotate_parameter("layers.*.feed_forward.w2.weight", rp)
autop.propagate_annotations(verbose=False, method="fix")
t = time.perf_counter()
ApproximateShardingSolver(opt).get_solution(verbose=False)
merged = opt.profile["approximate"]["objective"]
log(f"[merged]    approx {time.perf_counter()-t:.1f}s  objective={merged:.1f}")

# LP relaxation lower bound via HiGHS = optimality reference.
t = time.perf_counter()
lb, nvar, ncon = lp_lower_bound_highs(opt)
log(f"[LP-bound]  HiGHS {time.perf_counter()-t:.1f}s  lower_bound={lb:.1f}  "
    f"(vars={nvar} cons={ncon})")

log("")
for name, obj in [("prune+dp", prune_dp), ("merged", merged)]:
    gap = 100 * (obj - lb) / lb
    log(f"=== 3D {name:<9} gap = {gap:+.2f}%  (obj {obj:.1f} vs LP lower bound "
        f"{lb:.1f})  <=10%: {abs(gap)<=10}  <=5%: {abs(gap)<=5} ===")
