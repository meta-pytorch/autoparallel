"""Dump the lite-build phase breakdown (tracing vs strategy enumeration vs
decision-var cost estimation) for LLaMA3-1B on a 3D mesh, to see where the
~615s build time goes. Env: MESH, SEQLEN."""
import json
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
from autoparallel.cost_models.collective_runtime_estimation import set_nccl_topo_config
from autoparallel.cost_models.nccl_cost_model import detect_nccl_topo_config

logging.basicConfig(level=logging.ERROR)
for fn, val in [("device_count", lambda: 8), ("get_device_name", lambda *a, **k: "H100"),
                ("get_device_capability", lambda *a, **k: (9, 0))]:
    patch(f"torch.cuda.{fn}", val).start()
patch("torch.cuda.get_device_properties", lambda *a, **k: type(
    "P", (), {"major": 9, "minor": 0, "name": "H100",
              "total_memory": 80 * 1024**3, "multi_processor_count": 132})()).start()

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


set_nccl_topo_config(detect_nccl_topo_config(mesh))
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
print(f"=== build profile: mesh={MESH_SHAPE}{names} seqlen={SEQLEN} ===", flush=True)

t = time.perf_counter()
autop = AutoParallel(model_fn(), input_fn, mesh, mp, repeated_subgraphs=True, solver="approx")
autop.__enter__()
enter_s = time.perf_counter() - t
opt = autop.sharding_optimizer
tm = opt.profile["timings"]
init = tm.get("init_total_s", 0.0)
tracing = enter_s - init  # __enter__ = tracing + ShardingOptimizer construction

print(json.dumps({
    "enter_total_s": round(enter_s, 1),
    "tracing_s (enter - optimizer_init)": round(tracing, 1),
    "optimizer_init_total_s": round(init, 1),
    "  strategy_enumeration_s": round(tm.get("strategy_enumeration_s", 0), 1),
    "  decision_var_build_s": round(tm.get("decision_var_build_s", 0), 1),
    "    compute_cost_estimation_s": round(tm.get("compute_cost_estimation_s", 0), 1),
    "    edge_cost_estimation_s": round(tm.get("edge_cost_estimation_s", 0), 1),
    "    pulp_var_creation_s (0 in lite)": round(tm.get("pulp_var_creation_s", 0), 1),
    "  validation_s": round(tm.get("validation_s", 0), 1),
    "decision_vars": len(opt.decision_vars),
    "graph_nodes": opt.profile["model"]["graph_nodes"],
    "strategy_options": opt.profile["strategies"]["strategy_options"],
    "option_tuples (edges)": opt.profile["strategies"]["option_tuples"],
}, indent=2), flush=True)
