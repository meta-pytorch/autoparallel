"""e2e prune+dp (approx) search across LLaMA3 sizes: latency + accuracy.

For one MODEL on one MESH:
  * latency: lite build (build_pulp=False) + ApproximateShardingSolver -> the
    production prune+dp path (build_s, approx_s, total, objective).
  * accuracy: a separate full PuLP build -> HiGHS LP-relaxation lower bound
    (this sharding LP is integral, so the bound equals the exact ILP optimum);
    gap = (approx_obj - lb) / lb.

Env: MODEL (1b|3b|8b|70b), MESH (e.g. 2,4,8), SEQLEN. One model per process.
"""
import gc
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

MODEL = os.environ.get("MODEL", "1b")
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

_CFG = {
    "1b": dict(dim=2048, n_layers=16, n_heads=32, n_kv_heads=8, ffn_dim_multiplier=1.5, multiple_of=256),
    "3b": dict(dim=3072, n_layers=28, n_heads=24, n_kv_heads=8, ffn_dim_multiplier=1.0, multiple_of=256),
    "8b": dict(dim=4096, n_layers=32, n_heads=32, n_kv_heads=8, ffn_dim_multiplier=1.3, multiple_of=1024),
    "70b": dict(dim=8192, n_layers=80, n_heads=64, n_kv_heads=8, ffn_dim_multiplier=1.3, multiple_of=4096),
}


def model_fn():
    args = TransformerModelArgs(
        rope_theta=500000, vocab_size=vocab_size, max_seq_len=SEQLEN, **_CFG[MODEL]
    )
    with torch.device("meta"):
        return Transformer(args)


def input_fn():
    return torch.randint(0, vocab_size, (batch_size, SEQLEN), device="cuda")


def constrain(autop):
    x = (Shard(0),) + (Replicate(),) * (ndim - 1)
    out = (Shard(0), Shard(2)) if ndim == 2 else x
    autop.add_parameter_memory_constraint(low=None, high=None)
    autop.add_input_constraints([x])
    autop.add_output_constraints([out])


def lp_lower_bound_highs(opt):
    """LP relaxation (binaries -> [0,1]) of the built problem, solved with HiGHS.
    Objective is read from decision_vars and constraints from prob.constraints
    using id()-keyed indexing (avoids hashing the long PuLP var names)."""
    opt._set_objective()
    opt._apply_memory_constraint()
    variables = opt.prob.variables()
    vidx = {id(v): i for i, v in enumerate(variables)}
    n = len(variables)
    c = np.zeros(n)
    for key, dv in opt.decision_vars.items():
        mult = 1 + len(opt._root_to_copies.get(key[0], ()))
        c[vidx[id(dv.var)]] += dv.cost * mult
    re = ru = 0
    reqr, reqc, reqd, beq = [], [], [], []
    rubr, rubc, rubd, bub = [], [], [], []
    for con in opt.prob.constraints.values():
        rhs = -con.constant
        if con.sense == pulp.LpConstraintEQ:
            for v, co in con.items():
                reqr.append(re); reqc.append(vidx[id(v)]); reqd.append(co)
            beq.append(rhs); re += 1
        else:
            s = 1.0 if con.sense == pulp.LpConstraintLE else -1.0
            for v, co in con.items():
                rubr.append(ru); rubc.append(vidx[id(v)]); rubd.append(s * co)
            bub.append(s * rhs); ru += 1
    A_eq = sp.csr_matrix((reqd, (reqr, reqc)), shape=(re, n)) if re else None
    A_ub = sp.csr_matrix((rubd, (rubr, rubc)), shape=(ru, n)) if ru else None
    # Dual simplex: far faster than the barrier (IPM) on this near-integral,
    # network-flow-like LP. We only need the optimal objective as the bound.
    method = os.environ.get("LP_METHOD", "highs-ds")
    res = linprog(c, A_ub=A_ub, b_ub=(bub or None), A_eq=A_eq, b_eq=(beq or None),
                  bounds=(0, 1), method=method, options={"disp": True})
    if not res.success:
        raise RuntimeError(f"HiGHS failed: {res.message}")
    return res.fun, n, re + ru


set_nccl_topo_config(detect_nccl_topo_config(mesh))
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
print(f"### MODEL={MODEL} mesh={MESH_SHAPE}{names} seqlen={SEQLEN} ###", flush=True)

# ---- latency: lite build + prune+dp approx (production path) ----
t = time.perf_counter()
autop = AutoParallel(model_fn(), input_fn, mesh, mp, repeated_subgraphs=True, solver="approx")
autop.__enter__()
constrain(autop)
build_lite = time.perf_counter() - t
opt = autop.sharding_optimizer
n_dv = len(opt.decision_vars)
params = opt.profile["model"]["parameter_numel"]
t = time.perf_counter()
ApproximateShardingSolver(opt).get_solution(verbose=False)
approx_s = time.perf_counter() - t
obj = opt.profile["approximate"]["objective"]
print(f"[latency] params={params/1e9:.2f}B  lite_build={build_lite:.1f}s  "
      f"approx={approx_s:.1f}s  total={build_lite + approx_s:.1f}s  "
      f"decision_vars={n_dv}  obj={obj:.1f}", flush=True)
autop.__exit__(None, None, None)
del autop, opt
gc.collect()

if os.environ.get("ACCURACY", "1") != "1":
    print(f"[RESULT] MODEL={MODEL} params={params/1e9:.2f}B  "
          f"prune+dp: build={build_lite:.1f}s approx={approx_s:.1f}s "
          f"total={build_lite+approx_s:.1f}s  obj={obj:.1f}  (LP skipped)", flush=True)
    raise SystemExit(0)

# ---- accuracy: full build + HiGHS LP lower bound ----
t = time.perf_counter()
autop = AutoParallel(model_fn(), input_fn, mesh, mp, repeated_subgraphs=True, solver="ilp")
autop.__enter__()
constrain(autop)
full_build = time.perf_counter() - t
opt = autop.sharding_optimizer
t = time.perf_counter()
lb, nvar, ncon = lp_lower_bound_highs(opt)
lp_s = time.perf_counter() - t
gap = 100 * (obj - lb) / lb
print(f"[accuracy] full_build={full_build:.1f}s  lp_solve={lp_s:.1f}s  "
      f"lower_bound={lb:.1f}  vars={nvar} cons={ncon}", flush=True)
print(f"[RESULT] MODEL={MODEL} params={params/1e9:.2f}B  "
      f"prune+dp: build={build_lite:.1f}s approx={approx_s:.1f}s total={build_lite+approx_s:.1f}s  "
      f"obj={obj:.1f}  LP_lb={lb:.1f}  gap={gap:+.2f}%", flush=True)
