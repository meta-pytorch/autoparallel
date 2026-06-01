"""prune+dp+annotation (the full joint config) vs prune+dp alone, compared to a
known optimum/LP lower bound. Lite build + optional TP-plan annotation + approx.
Env: MODEL, MESH, SEQLEN, LP_LB."""
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
    args = TransformerModelArgs(rope_theta=500000, vocab_size=vocab_size,
                                max_seq_len=SEQLEN, **_CFG[MODEL])
    with torch.device("meta"):
        return Transformer(args)


def input_fn():
    return torch.randint(0, vocab_size, (batch_size, SEQLEN), device="cuda")


COLUMN_PARALLEL = (None,) * (ndim - 1) + (Shard(0),)
ROW_PARALLEL = (None,) * (ndim - 1) + (Shard(1),)


def annotate_tp_plan(autop):
    for proj in ["wq", "wk", "wv"]:
        autop.annotate_parameter(f"layers.*.attention.{proj}.weight", COLUMN_PARALLEL)
    autop.annotate_parameter("layers.*.attention.wo.weight", ROW_PARALLEL)
    for proj in ["w1", "w3"]:
        autop.annotate_parameter(f"layers.*.feed_forward.{proj}.weight", COLUMN_PARALLEL)
    autop.annotate_parameter("layers.*.feed_forward.w2.weight", ROW_PARALLEL)


def constrain(autop):
    x = (Shard(0),) + (Replicate(),) * (ndim - 1)
    out = (Shard(0), Shard(2)) if ndim == 2 else x
    autop.add_parameter_memory_constraint(low=None, high=None)
    autop.add_input_constraints([x])
    autop.add_output_constraints([out])


set_nccl_topo_config(detect_nccl_topo_config(mesh))
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
print(f"### anno MODEL={MODEL} mesh={MESH_SHAPE}{names} LP_lb={LP_LB} ###", flush=True)


def gap(o):
    return 100 * (o - LP_LB) / LP_LB if LP_LB else float("nan")


# prune+dp (no annotation)
t = time.perf_counter()
autop = AutoParallel(model_fn(), input_fn, mesh, mp, repeated_subgraphs=True, solver="approx")
autop.__enter__()
constrain(autop)
build_s = time.perf_counter() - t
opt = autop.sharding_optimizer
t = time.perf_counter()
ApproximateShardingSolver(opt).get_solution(verbose=False)
dp_s = time.perf_counter() - t
obj_dp = opt.profile["approximate"]["objective"]
print(f"[dp]     build={build_s:.1f}s approx={dp_s:.1f}s obj={obj_dp:.1f} gap={gap(obj_dp):+.2f}%", flush=True)

# + annotation
t = time.perf_counter()
annotate_tp_plan(autop)
prop = autop.propagate_annotations(verbose=False, method="fix")
prop_s = time.perf_counter() - t
t = time.perf_counter()
ApproximateShardingSolver(opt).get_solution(verbose=False)
ann_s = time.perf_counter() - t
obj_ann = opt.profile["approximate"]["objective"]
print(f"[dp+anno] build={build_s:.1f}s propagate={prop_s:.1f}s approx={ann_s:.1f}s "
      f"total={build_s+prop_s+ann_s:.1f}s obj={obj_ann:.1f} gap={gap(obj_ann):+.2f}% "
      f"(pinned {prop.nodes_determined} nodes)", flush=True)
print(f"[RESULT] MODEL={MODEL} mesh={MESH_SHAPE} dp_gap={gap(obj_dp):+.2f}% "
      f"dp+anno_gap={gap(obj_ann):+.2f}% dp+anno_total={build_s+prop_s+ann_s:.1f}s", flush=True)
