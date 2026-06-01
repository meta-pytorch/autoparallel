"""Re-solve the 70B LP relaxation and report how integral the optimum is: count
fractional variables in the HiGHS solution. If ~all variables are 0/1, the LP
optimum is reachable by integers (so an approx gap is a real solver failure); if
many are fractional, the LP bound is loose (and the approx may be near-optimal).
Also reports the objective with the memory constraint dropped, to test whether
the memory budget is the fractionality source. Env: MODEL, MESH, SEQLEN."""
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
from autoparallel.cost_models.collective_runtime_estimation import set_nccl_topo_config
from autoparallel.cost_models.nccl_cost_model import detect_nccl_topo_config

logging.basicConfig(level=logging.ERROR)
for fn, val in [("device_count", lambda: 8), ("get_device_name", lambda *a, **k: "H100"),
                ("get_device_capability", lambda *a, **k: (9, 0))]:
    patch(f"torch.cuda.{fn}", val).start()
patch("torch.cuda.get_device_properties", lambda *a, **k: type(
    "P", (), {"major": 9, "minor": 0, "name": "H100",
              "total_memory": 80 * 1024**3, "multi_processor_count": 132})()).start()

MODEL = os.environ.get("MODEL", "70b")
SEQLEN = int(os.environ.get("SEQLEN", "2048"))
MESH_SHAPE = tuple(int(x) for x in os.environ.get("MESH", "2,4,8").split(","))
DROP_MEM = os.environ.get("DROP_MEM", "0") == "1"
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
    "8b": dict(dim=4096, n_layers=32, n_heads=32, n_kv_heads=8, ffn_dim_multiplier=1.3, multiple_of=1024),
    "70b": dict(dim=8192, n_layers=80, n_heads=64, n_kv_heads=8, ffn_dim_multiplier=1.3, multiple_of=4096),
}


def model_fn():
    args = TransformerModelArgs(rope_theta=500000, vocab_size=vocab_size,
                                max_seq_len=SEQLEN, **_CFG[MODEL])
    with torch.device("meta"):
        return Transformer(args)


def input_fn():
    return torch.randint(0, vocab_size, (batch_size, SEQLEN), device="cuda")


set_nccl_topo_config(detect_nccl_topo_config(mesh))
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
print(f"### LP integrality MODEL={MODEL} mesh={MESH_SHAPE}{names} drop_mem={DROP_MEM} ###", flush=True)

t = time.perf_counter()
autop = AutoParallel(model_fn(), input_fn, mesh, mp, repeated_subgraphs=True, solver="ilp")
autop.__enter__()
x = (Shard(0),) + (Replicate(),) * (ndim - 1)
out = (Shard(0), Shard(2)) if ndim == 2 else x
autop.add_parameter_memory_constraint(low=None, high=None)
autop.add_input_constraints([x])
autop.add_output_constraints([out])
opt = autop.sharding_optimizer
print(f"[build] full_build={time.perf_counter()-t:.1f}s", flush=True)

opt._set_objective()
if not DROP_MEM:
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
        sgn = 1.0 if con.sense == pulp.LpConstraintLE else -1.0
        for v, co in con.items():
            rubr.append(ru); rubc.append(vidx[id(v)]); rubd.append(sgn * co)
        bub.append(sgn * rhs); ru += 1
A_eq = sp.csr_matrix((reqd, (reqr, reqc)), shape=(re, n)) if re else None
A_ub = sp.csr_matrix((rubd, (rubr, rubc)), shape=(ru, n)) if ru else None
t = time.perf_counter()
res = linprog(c, A_ub=A_ub, b_ub=(bub or None), A_eq=A_eq, b_eq=(beq or None),
              bounds=(0, 1), method="highs-ds", options={"disp": True})
print(f"[lp] solve={time.perf_counter()-t:.1f}s status={res.message}", flush=True)
xv = res.x
freq = np.abs(xv - np.round(xv))
nfrac = int((freq > 1e-6).sum())
nfrac4 = int((freq > 1e-4).sum())
# weight fractionality by objective contribution to see if it matters
frac_obj = float(np.abs(c * freq).sum())
print(f"[RESULT] MODEL={MODEL} drop_mem={DROP_MEM} obj={res.fun:.1f} "
      f"vars={n} fractional(>1e-6)={nfrac} ({100*nfrac/n:.4f}%) "
      f"fractional(>1e-4)={nfrac4} frac_obj_weight={frac_obj:.1f}", flush=True)
