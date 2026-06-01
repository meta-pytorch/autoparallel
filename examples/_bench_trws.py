"""Prototype TRW-S (tree-reweighted sequential message passing) on the approx
solver's faithful factor graph, validated against the CBC-exact optimum. If TRW-S
(optionally + the existing local search) reaches the optimum where plain min-sum
BP does not, it is the fix. Env: MODEL, MESH, SEQLEN, ITERS."""
import logging
import os
import time
from unittest.mock import patch

import numpy as np
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
for fn, val in [("device_count", lambda: 8), ("get_device_name", lambda *a, **k: "H100"),
                ("get_device_capability", lambda *a, **k: (9, 0))]:
    patch(f"torch.cuda.{fn}", val).start()
patch("torch.cuda.get_device_properties", lambda *a, **k: type(
    "P", (), {"major": 9, "minor": 0, "name": "H100",
              "total_memory": 80 * 1024**3, "multi_processor_count": 132})()).start()

MODEL = os.environ.get("MODEL", "1b")
SEQLEN = int(os.environ.get("SEQLEN", "2048"))
MESH_SHAPE = tuple(int(x) for x in os.environ.get("MESH", "8,8").split(","))
ITERS = int(os.environ.get("ITERS", "1000"))
USE_CBC = os.environ.get("CBC", "1") == "1"
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
    args = TransformerModelArgs(rope_theta=500000, vocab_size=vocab_size,
                                max_seq_len=SEQLEN, **_CFG[MODEL])
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


set_nccl_topo_config(detect_nccl_topo_config(mesh))
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
print(f"### TRW-S MODEL={MODEL} mesh={MESH_SHAPE}{names} iters={ITERS} ###", flush=True)

backend = "ilp" if USE_CBC else "approx"
t = time.perf_counter()
autop = AutoParallel(model_fn(), input_fn, mesh, mp, repeated_subgraphs=True, solver=backend)
autop.__enter__()
constrain(autop)
opt = autop.sharding_optimizer
print(f"[build] {time.perf_counter()-t:.1f}s decision_vars={len(opt.decision_vars)}", flush=True)

obj_cbc = None
if USE_CBC:
    opt._set_objective()
    opt._apply_memory_constraint()
    t = time.perf_counter()
    opt.prob.solve(pulp.PULP_CBC_CMD(msg=False, options=["preprocess off"]))
    obj_cbc = pulp.value(opt.prob.objective)
    print(f"[cbc] obj={obj_cbc:.1f} status={pulp.LpStatus[opt.prob.status]} "
          f"({time.perf_counter()-t:.1f}s)", flush=True)


_REF = obj_cbc if obj_cbc else float(os.environ.get("LP_LB", "0")) or None


def gap(o):
    return 100 * (o - _REF) / _REF if _REF else float("nan")


# Stock approx (BP + local search) for comparison.
a0 = ApproximateShardingSolver(opt)
t = time.perf_counter()
a0.get_solution(verbose=False)
print(f"[stock approx] obj={opt.profile['approximate']['objective']:.1f} "
      f"gap={gap(opt.profile['approximate']['objective']):+.2f}% ({time.perf_counter()-t:.1f}s)", flush=True)

# Build a fresh factor graph for TRW-S.
A = ApproximateShardingSolver(opt)
A._build_problem()
A._build_factors()
G = len(A.groups)
nbrs = A.nbrs
unary = A.g_unary
order = sorted(range(G), key=lambda g: min(A.groups[g].members))
pos = [0] * G
for i, g in enumerate(order):
    pos[g] = i
gamma = []
for g in range(G):
    indeg = sum(1 for h in nbrs[g] if pos[h] < pos[g])
    outdeg = sum(1 for h in nbrs[g] if pos[h] > pos[g])
    gamma.append(1.0 / max(1, max(indeg, outdeg)))

msg = {}
for g in range(G):
    for h in nbrs[g]:
        msg[(g, h)] = np.zeros(len(unary[h]))

t = time.perf_counter()
best = float("inf")
best_snap = None
for it in range(ITERS):
    for forward in (True, False):
        seq = order if forward else order[::-1]
        for p in seq:
            if not nbrs[p]:
                continue
            agg = unary[p].copy()
            for r in nbrs[p]:
                agg += msg[(r, p)]
            wp = gamma[p] * agg
            for q in nbrs[p]:
                if (pos[q] > pos[p]) != forward:
                    continue
                P = A._pair_matrix(p, q)  # (D_p, D_q)
                mm = (wp - msg[(q, p)])[:, None] + P
                mq = mm.min(axis=0)
                mq -= mq.min()
                msg[(p, q)] = mq
    A._decode(msg)
    e = A._fast_total_energy()
    if e < best - 1e-6:
        best = e
        best_snap = [g.current for g in A.groups]
    if it < 5 or it % 50 == 0:
        print(f"  [trws it={it}] decode_energy={e:.1f} best={best:.1f} gap={gap(best):+.2f}%", flush=True)
trws_s = time.perf_counter() - t
for gid, ci in enumerate(best_snap):
    A._set_group(gid, ci)
print(f"[TRW-S] best={best:.1f} gap={gap(best):+.2f}% ({trws_s:.1f}s, {ITERS} iters)", flush=True)

# Polish TRW-S result with the existing local search.
deadline = time.perf_counter() + 60
A._memory_repair()
A._coordinate_descent(deadline)
A._star_block_search(deadline)
polished = A._fast_total_energy()
print(f"[TRW-S + local search] obj={polished:.1f} gap={gap(polished):+.2f}%", flush=True)
print(f"[RESULT] MODEL={MODEL} mesh={MESH_SHAPE} cbc={obj_cbc} "
      f"stock_gap={gap(opt.profile['approximate']['objective']):+.2f}% "
      f"trws_gap={gap(best):+.2f}% trws_ls_gap={gap(polished):+.2f}%", flush=True)
