"""A/B verify that the fast build (AP_FAST_BUILD=1) produces byte-identical
decision_vars + approx objective as the baseline (AP_FAST_BUILD=0), and report
build time. Run the same MESH/MODEL with both env values and diff the dv_hash.
Env: MESH, SEQLEN, MODEL (tiny|1b)."""
import hashlib
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

SEQLEN = int(os.environ.get("SEQLEN", "2048"))
MODEL = os.environ.get("MODEL", "tiny")
MESH_SHAPE = tuple(int(x) for x in os.environ.get("MESH", "4,2").split(","))
ws = 1
for d in MESH_SHAPE:
    ws *= d
names = {2: ("dp", "tp"), 3: ("dp", "cp", "tp")}[len(MESH_SHAPE)]
torch.distributed.init_process_group("fake", store=FakeStore(), rank=0, world_size=ws)
mesh = torch.distributed.device_mesh.init_device_mesh("cuda", MESH_SHAPE, mesh_dim_names=names)
ndim = mesh.ndim
vocab_size = 128 if MODEL == "tiny" else 128256
batch_size = 2 * mesh.shape[0]


def model_fn():
    if MODEL == "tiny":
        args = TransformerModelArgs(dim=64, n_layers=2, n_heads=4, n_kv_heads=2,
                                    vocab_size=vocab_size, multiple_of=32,
                                    rope_theta=500000, max_seq_len=SEQLEN)
    else:
        args = TransformerModelArgs(dim=2048, n_layers=16, n_heads=32, n_kv_heads=8,
                                    ffn_dim_multiplier=1.5, multiple_of=256,
                                    rope_theta=500000, vocab_size=vocab_size,
                                    max_seq_len=SEQLEN)
    with torch.device("meta"):
        return Transformer(args)


def input_fn():
    return torch.randint(0, vocab_size, (batch_size, SEQLEN), device="cuda")


set_nccl_topo_config(detect_nccl_topo_config(mesh))
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
x = (Shard(0),) + (Replicate(),) * (ndim - 1)
out = (Shard(0), Shard(2)) if ndim == 2 else x

t = time.perf_counter()
autop = AutoParallel(model_fn(), input_fn, mesh, mp, repeated_subgraphs=True, solver="ilp")
autop.__enter__()
autop.add_parameter_memory_constraint(low=None, high=None)
autop.add_input_constraints([x])
autop.add_output_constraints([out])
build_s = time.perf_counter() - t
opt = autop.sharding_optimizer

# Canonical, exact dump of every decision var's costs.
items = []
for key in sorted(opt.decision_vars.keys()):
    dv = opt.decision_vars[key]
    items.append((key, repr(dv.cost), repr(dv.comm_cost), repr(dv.compute_cost),
                  repr(dv.sharding_transition_cost)))
dv_hash = hashlib.sha256(repr(items).encode()).hexdigest()

t = time.perf_counter()
ApproximateShardingSolver(opt).get_solution(verbose=False)
approx_s = time.perf_counter() - t
obj = opt.profile["approximate"]["objective"]

print(f"AP_FAST_BUILD={os.environ.get('AP_FAST_BUILD', '1')}  MODEL={MODEL} "
      f"MESH={MESH_SHAPE}  build={build_s:.2f}s  approx={approx_s:.2f}s  "
      f"n_dv={len(opt.decision_vars)}  dv_hash={dv_hash[:32]}  "
      f"approx_obj={obj!r}", flush=True)
