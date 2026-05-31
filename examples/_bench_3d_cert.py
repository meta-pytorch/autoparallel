"""3D optimality certificate for the merged solver on full LLaMA3-1B.

The 3D ILP has ~8M binary variables; the exact CBC solve is impractical (a 2.6 GB
MPS file). The LP relaxation, however, is empirically integral for this problem
(verified on 2D, where it equals the exact optimum), so its objective is a tight
lower bound on the ILP optimum. This script does ONE full PuLP build, then:

  1. get_lower_bound()  -> LP lower bound (the optimality reference)
  2. annotate + propagate + ApproximateShardingSolver  -> merged objective

and reports the certified gap = (merged - lb) / lb. Slow (one ~13min build + a
multi-minute LP solve) but a one-shot 3D certificate. Env: MESH, SEQLEN.
"""
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


set_nccl_topo_config(detect_nccl_topo_config(mesh))
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
x = (Shard(0),) + (Replicate(),) * (ndim - 1)
out = (Shard(0), Shard(2)) if ndim == 2 else x

log(f"=== 3D cert: LLaMA3-1B mesh={MESH_SHAPE}{names} seqlen={SEQLEN} ===")
t = time.perf_counter()
autop = AutoParallel(model_fn(), input_fn, mesh, mp, repeated_subgraphs=True, solver="ilp")
autop.__enter__()
autop.add_parameter_memory_constraint(low=None, high=None)
autop.add_input_constraints([x])
autop.add_output_constraints([out])
opt = autop.sharding_optimizer
log(f"[build] {time.perf_counter()-t:.1f}s  decision_vars={len(opt.decision_vars)}  "
    f"pulp_vars={len(opt.pulp_variables)}")

# LP-relaxation lower bound = optimality reference (the exact ILP is intractable).
opt._set_objective()
t = time.perf_counter()
lb = opt.get_lower_bound(verbose=False).objective
log(f"[LP-bound] {time.perf_counter()-t:.1f}s  lower_bound={lb:.1f}")

# Merged solver on the same build: propagate the TP plan, then approx-solve.
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
log(f"[merged] approx {time.perf_counter()-t:.1f}s  objective={merged:.1f}")

gap = 100 * (merged - lb) / lb
log(f"\n=== 3D certified gap = {gap:+.2f}%  (merged {merged:.1f} vs LP lower bound "
    f"{lb:.1f}) ===")
log(f"acceptance gap<=10%: {abs(gap)<=10}  (<=5%: {abs(gap)<=5})")
