"""Minimal approx-solver timing, for the 'dp alone' (approx WITHOUT prune)
baseline. Run it with PYTHONPATH pointing at the dp_solver checkout to get the
unpruned numbers, and at the merge checkout to cross-check prune+dp.

Reports lite-build time, approx solve time, decision-var count and objective for
LLaMA3-1B with the canonical constraints. Env: MESH, SEQLEN, N_LAYERS.
"""
import logging
import os
import time
from unittest.mock import patch

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

import autoparallel
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

N_LAYERS = int(os.environ.get("N_LAYERS", "0"))
SEQLEN = int(os.environ.get("SEQLEN", "2048"))
MESH_SHAPE = tuple(int(x) for x in os.environ.get("MESH", "8,8").split(","))
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
    if N_LAYERS:
        args.n_layers = N_LAYERS
    with torch.device("meta"):
        return Transformer(args)


def input_fn():
    return torch.randint(0, vocab_size, (batch_size, SEQLEN), device="cuda")


set_nccl_topo_config(detect_nccl_topo_config(mesh))
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
x = (Shard(0),) + (Replicate(),) * (ndim - 1)
out = (Shard(0), Shard(2)) if ndim == 2 else x

print(f"autoparallel = {autoparallel.__file__}", flush=True)
print(f"=== dp-alone (approx) LLaMA3-1B mesh={MESH_SHAPE}{names} seqlen={SEQLEN} "
      f"layers={N_LAYERS or 16} ===", flush=True)

t = time.perf_counter()
autop = AutoParallel(model_fn(), input_fn, mesh, mp, repeated_subgraphs=True, solver="approx")
autop.__enter__()
autop.add_parameter_memory_constraint(low=None, high=None)
autop.add_input_constraints([x])
autop.add_output_constraints([out])
t_build = time.perf_counter() - t
opt = autop.sharding_optimizer

t = time.perf_counter()
ApproximateShardingSolver(opt).get_solution(verbose=False)
t_solve = time.perf_counter() - t
obj = opt.profile["approximate"]["objective"]

print(f"[dp-alone] build={t_build:.2f}s  approx_solve={t_solve:.2f}s  "
      f"total={t_build + t_solve:.2f}s  obj={obj:.1f}  "
      f"decision_vars={len(opt.decision_vars)}", flush=True)
