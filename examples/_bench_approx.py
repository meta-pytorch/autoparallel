"""Benchmark approximate solver vs ILP: objective + solve time.

Setting: LLaMA3 (1b default) on a 2D (dp, tp) mesh with vocab parallelism and
the canonical example_llama3 constraints. Both solvers run on the SAME built
optimizer: approx first (it only fills varValues/objective via an idempotent
_set_objective), then a fresh CBC solve for the ILP. This avoids building the
(expensive) strategy graph twice.

Env knobs: MODEL_TYPE (1b|8b), MESH ("8,8"), N_LAYERS (0=default), SEQLEN,
REPEATED (1|0), RUN_ILP (1|0), ILP_TIMEOUT (seconds, 0=unlimited).
"""
import logging
import os
import time
from unittest.mock import patch

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
_alog = logging.getLogger("autoparallel.approximate_sharding")
_alog.setLevel(logging.INFO)
_alog.addHandler(logging.StreamHandler())


def log(msg):
    print(msg, flush=True)


_PATCHES = [
    patch("torch.cuda.device_count", lambda: 8),
    patch("torch.cuda.get_device_name", lambda *a, **k: "H100"),
    patch("torch.cuda.get_device_capability", lambda *a, **k: (9, 0)),
    patch(
        "torch.cuda.get_device_properties",
        lambda *a, **k: type(
            "P", (), {"major": 9, "minor": 0, "name": "H100",
                      "total_memory": 80 * 1024**3, "multi_processor_count": 132}
        )(),
    ),
]
for p in _PATCHES:
    p.start()

MODEL_TYPE = os.environ.get("MODEL_TYPE", "1b")
N_LAYERS = int(os.environ.get("N_LAYERS", "0"))
SEQLEN = int(os.environ.get("SEQLEN", str(2048 * 4)))
MESH_SHAPE = tuple(int(x) for x in os.environ.get("MESH", "8,8").split(","))
REPEATED = os.environ.get("REPEATED", "1") == "1"
RUN_ILP = os.environ.get("RUN_ILP", "1") == "1"
LP_BOUND = os.environ.get("LP_BOUND", "1") == "1"
ILP_TIMEOUT = float(os.environ.get("ILP_TIMEOUT", "1200"))

world_size = 1
for d in MESH_SHAPE:
    world_size *= d

_NAMES = {1: ("dp",), 2: ("dp", "tp"), 3: ("dp", "cp", "tp"),
          4: ("dp", "cp", "tp", "ep")}
mesh_names = _NAMES[len(MESH_SHAPE)]
fake_store = FakeStore()
torch.distributed.init_process_group("fake", store=fake_store, rank=0, world_size=world_size)
mesh = torch.distributed.device_mesh.init_device_mesh(
    "cuda", MESH_SHAPE, mesh_dim_names=mesh_names
)

vocab_size = 128256
batch_size = 2 * mesh.shape[0]
seqlen = SEQLEN


def model_fn():
    if MODEL_TYPE == "1b":
        args = TransformerModelArgs(
            dim=2048, n_layers=16, n_heads=32, n_kv_heads=8,
            ffn_dim_multiplier=1.5, multiple_of=256, rope_theta=500000,
            vocab_size=vocab_size, max_seq_len=seqlen,
        )
    elif MODEL_TYPE == "8b":
        args = TransformerModelArgs(
            dim=4096, n_layers=32, n_heads=32, n_kv_heads=8,
            ffn_dim_multiplier=1.3, multiple_of=1024, rope_theta=500000,
            vocab_size=vocab_size, max_seq_len=seqlen,
        )
    else:
        raise ValueError(MODEL_TYPE)
    if N_LAYERS:
        args.n_layers = N_LAYERS
    with torch.device("meta"):
        return Transformer(args)


def input_fn():
    return torch.randint(0, vocab_size, (batch_size, seqlen), device="cuda")


set_nccl_topo_config(detect_nccl_topo_config(mesh))
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

log(f"model={MODEL_TYPE} layers={N_LAYERS or 'default'} mesh={MESH_SHAPE} "
    f"world={world_size} seqlen={seqlen} repeated_subgraphs={REPEATED} "
    f"ilp_timeout={ILP_TIMEOUT}")

t = time.perf_counter()
autop = AutoParallel(model_fn(), input_fn, mesh, mp, repeated_subgraphs=REPEATED)
autop.__enter__()
ndim = mesh.ndim
x_sharding = (Shard(0),) + (Replicate(),) * (ndim - 1)
# vocab-parallel output only defined for 2D (matches example_llama3); otherwise
# constrain the output like the input.
out_sharding = (Shard(0), Shard(2)) if ndim == 2 else x_sharding
autop.add_parameter_memory_constraint(low=None, high=None)
autop.add_input_constraints([x_sharding])
autop.add_output_constraints([out_sharding])
opt = autop.sharding_optimizer
log(f"[build] optimizer ready in {time.perf_counter() - t:.2f}s  "
    f"vars={len(opt.pulp_variables)} constraints={len(opt.prob.constraints)} "
    f"nodes={len(opt.nodes)}")

# ---- APPROX ----
t = time.perf_counter()
approx = ApproximateShardingSolver(opt)
approx.get_solution(verbose=True)
ap_t = time.perf_counter() - t
ap_obj = pulp.value(opt.prob.objective)
prof = opt.profile.get("approximate", {})
log(f"\n[APPROX] objective={ap_obj:.2f}  solve_time={ap_t:.3f}s")
log(f"         groups={prof.get('groups')} sweeps={prof.get('sweeps')} "
    f"build={prof.get('build_s'):.3f}s search={prof.get('solve_s'):.3f}s "
    f"writeback={ap_t - prof.get('build_s', 0) - prof.get('solve_s', 0):.3f}s")

# ---- LP relaxation lower bound (certified suboptimality upper bound) ----
if LP_BOUND:
    lb_res = opt.get_lower_bound(verbose=False)
    lb = lb_res.objective
    if lb and lb > 0:
        cert = (ap_obj - lb) / lb
        log(f"\n[LP-bound] lower_bound={lb:.2f}  solve={lb_res.solve_s:.2f}s  "
            f"=> approx within {cert*100:.2f}% of optimum (certified upper bound)")

# ---- ILP (fresh CBC solve on the same problem) ----
if RUN_ILP:
    opt._set_objective()  # idempotent: objective already populated by approx
    kw = {"msg": True}
    if ILP_TIMEOUT > 0:
        kw["timeLimit"] = ILP_TIMEOUT
    log(f"\n[ILP] solving with CBC (timeLimit={ILP_TIMEOUT or 'none'})...")
    t = time.perf_counter()
    opt.prob.solve(pulp.PULP_CBC_CMD(**kw))
    ilp_t = time.perf_counter() - t
    ilp_obj = pulp.value(opt.prob.objective)
    status = pulp.LpStatus[opt.prob.status]
    log(f"[ILP]    objective={ilp_obj:.2f}  solve_time={ilp_t:.3f}s  status={status}")

    gap = (ap_obj - ilp_obj) / ilp_obj
    log(f"\n=== objective gap = {gap*100:+.2f}%   solve speedup = {ilp_t/ap_t:.1f}x ===")
    log(f"=== within 20% ? {abs(gap) <= 0.20}   (ILP status: {status}) ===")
