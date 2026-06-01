"""Diagnose whether the approx solver's objective is stuck in a local-optimum
basin that a stronger search escapes. Build once, run the stock BP+localsearch,
then run iterated local search (perturb a random set of groups, re-optimize,
keep best) for a time budget. If ILS beats the stock objective meaningfully, the
gap is a move-set/init weakness (and the LP bound is ~reachable); if not, 607260
is robust. Env: MODEL, MESH, SEQLEN, LP_LB, ILS_S."""
import logging
import os
import random
import time
from unittest.mock import patch

import numpy as np
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

MODEL = os.environ.get("MODEL", "70b")
SEQLEN = int(os.environ.get("SEQLEN", "2048"))
MESH_SHAPE = tuple(int(x) for x in os.environ.get("MESH", "2,4,8").split(","))
LP_LB = float(os.environ.get("LP_LB", "0"))
ILS_S = float(os.environ.get("ILS_S", "180"))
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
print(f"### ILS MODEL={MODEL} mesh={MESH_SHAPE}{names} LP_lb={LP_LB} ils_s={ILS_S} ###", flush=True)

t = time.perf_counter()
autop = AutoParallel(model_fn(), input_fn, mesh, mp, repeated_subgraphs=True, solver="approx")
autop.__enter__()
x = (Shard(0),) + (Replicate(),) * (ndim - 1)
out = (Shard(0), Shard(2)) if ndim == 2 else x
autop.add_parameter_memory_constraint(low=None, high=None)
autop.add_input_constraints([x])
autop.add_output_constraints([out])
opt = autop.sharding_optimizer
print(f"[build] lite_build={time.perf_counter()-t:.1f}s decision_vars={len(opt.decision_vars)}", flush=True)


def gap(o):
    return 100 * (o - LP_LB) / LP_LB if LP_LB else float("nan")


s = ApproximateShardingSolver(opt)
s._build_problem()
s._build_factors()
G = len(s.groups)
domains = [g.domain for g in s.groups]
multi = [d for d in domains if d > 1]
edges = len(s.C)
print(f"[graph] groups={G} multi_choice_groups={len(multi)} "
      f"max_domain={max(domains)} sum_domain={sum(domains)} pair_edges={edges}", flush=True)

# Stock solve (BP + local search), mirrors _solve's BP candidate.
deadline = time.perf_counter() + 1e9
s._belief_propagation()
s._memory_repair()
s._coordinate_descent(deadline)
s._star_block_search(deadline)
stock = s._fast_total_energy()
best = stock
best_snap = [g.current for g in s.groups]
print(f"[stock] bp+cd+star energy={stock:.1f} gap={gap(stock):+.2f}%", flush=True)

# Iterated local search: perturb k random multi-choice groups, re-optimize, keep best.
rng = random.Random(0)
multi_gids = [g for g in range(G) if s.groups[g].domain > 1]
t0 = time.perf_counter()
iters = 0
accepts = 0
while time.perf_counter() - t0 < ILS_S:
    iters += 1
    # restore best, then kick
    for gid, ci in enumerate(best_snap):
        s._set_group(gid, ci)
    k = rng.randint(1, max(2, len(multi_gids) // 10))
    for gid in rng.sample(multi_gids, min(k, len(multi_gids))):
        s._set_group(gid, rng.randrange(s.groups[gid].domain))
    s._memory_repair()
    s._coordinate_descent(deadline)
    s._star_block_search(deadline)
    e = s._fast_total_energy()
    if e < best - 1e-6:
        best = e
        best_snap = [g.current for g in s.groups]
        accepts += 1
        print(f"[ils] iter={iters} NEW BEST energy={best:.1f} gap={gap(best):+.2f}% "
              f"(k={k})", flush=True)

for gid, ci in enumerate(best_snap):
    s._set_group(gid, ci)
exact = s._write_back()
print(f"[ILS done] iters={iters} accepts={accepts} stock={stock:.1f} "
      f"best={best:.1f} exact_obj={exact:.1f} gap={gap(exact):+.2f}% "
      f"(improvement vs stock = {100*(stock-best)/stock:.2f}%)", flush=True)
