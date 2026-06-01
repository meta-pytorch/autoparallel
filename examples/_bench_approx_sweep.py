"""Build one model (lite) once, then run ApproximateShardingSolver under several
hyperparameter configs to see whether the objective gap (vs a known LP lower
bound) is closable by tuning (candidate pruning / BP iters / time / local search)
or is structural. Env: MODEL, MESH, SEQLEN, LP_LB (reference lower bound)."""
import logging
import os
import time
from unittest.mock import patch

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
        rope_theta=500000, vocab_size=vocab_size, max_seq_len=SEQLEN, **_CFG[MODEL])
    with torch.device("meta"):
        return Transformer(args)


def input_fn():
    return torch.randint(0, vocab_size, (batch_size, SEQLEN), device="cuda")


set_nccl_topo_config(detect_nccl_topo_config(mesh))
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
print(f"### approx sweep MODEL={MODEL} mesh={MESH_SHAPE}{names} LP_lb={LP_LB} ###", flush=True)

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

CONFIGS = [
    ("default", dict()),
    ("cand=256", dict(candidate_limit=256)),
    ("cand=None", dict(candidate_limit=None)),
    ("bp=100", dict(bp_iters=100)),
    ("sweeps=200,star=20,t=600", dict(max_sweeps=200, star_passes=20, max_time_s=600)),
    ("star_children=64,domain=4096", dict(max_star_children=64, group_domain_limit=4096)),
    ("ALL generous", dict(candidate_limit=None, bp_iters=100, max_sweeps=200,
                          star_passes=20, max_time_s=900, max_star_children=64,
                          group_domain_limit=4096)),
]

best = None
for name, cfg in CONFIGS:
    t = time.perf_counter()
    solver = ApproximateShardingSolver(opt, **cfg)
    solver.get_solution(verbose=False)
    dt = time.perf_counter() - t
    ap = opt.profile["approximate"]
    obj = ap["objective"]
    gap = 100 * (obj - LP_LB) / LP_LB if LP_LB else float("nan")
    winner = "bp" if ap["bp_energy"] <= ap["greedy_energy"] else "greedy"
    print(f"[cfg] {name:30s} obj={obj:.1f} gap={gap:+.2f}% "
          f"bp={ap['bp_energy']:.1f} greedy={ap['greedy_energy']:.1f} win={winner} "
          f"t={dt:.1f}s", flush=True)
    if best is None or obj < best[1]:
        best = (name, obj)

print(f"[BEST] {best[0]} obj={best[1]:.1f} "
      f"gap={100*(best[1]-LP_LB)/LP_LB:+.2f}%" if LP_LB else f"[BEST] {best[0]} obj={best[1]:.1f}",
      flush=True)
