"""Benchmark LP-relaxation solve time for LLaMA3 on a 3D mesh."""
import logging
import os
import time

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel._testing.models.llama3 import Transformer, TransformerModelArgs
from autoparallel.api import AutoParallel
from autoparallel.cost_models.collective_runtime_estimation import set_nccl_topo_config
from autoparallel.cost_models.nccl_cost_model import detect_nccl_topo_config

logging.basicConfig(level=logging.WARNING)

MODEL_TYPE = os.environ.get("MODEL_TYPE", "8b")
N_LAYERS = int(os.environ.get("N_LAYERS", "0"))  # 0 => use default for model
SEQLEN = int(os.environ.get("SEQLEN", str(2048 * 4)))
MESH_SHAPE = tuple(int(x) for x in os.environ.get("MESH", "2,4,8").split(","))
MESH_NAMES = ("dp", "cp", "tp")

world_size = 1
for d in MESH_SHAPE:
    world_size *= d

fake_store = FakeStore()
torch.distributed.init_process_group("fake", store=fake_store, rank=0, world_size=world_size)

mesh = torch.distributed.device_mesh.init_device_mesh("cuda", MESH_SHAPE, mesh_dim_names=MESH_NAMES)

batch_size = 2 * mesh.shape[0]
seqlen = SEQLEN
vocab_size = 128256
device = torch.device("cuda")


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
    elif MODEL_TYPE == "70b":
        args = TransformerModelArgs(
            dim=8192, n_layers=80, n_heads=64, n_kv_heads=8,
            ffn_dim_multiplier=1.3, multiple_of=4096, rope_theta=500000,
            vocab_size=vocab_size, max_seq_len=seqlen,
        )
    else:
        raise ValueError(MODEL_TYPE)
    if N_LAYERS:
        args.n_layers = N_LAYERS
    return Transformer(args)


def input_fn():
    return torch.randint(0, vocab_size, (batch_size, seqlen), device=device)


set_nccl_topo_config(detect_nccl_topo_config(mesh))

with torch.device("meta"):
    model = model_fn()

mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

print(f"=== model={MODEL_TYPE} n_layers={model.model_args.n_layers} "
      f"mesh={MESH_SHAPE}{MESH_NAMES} world_size={world_size} ===")

print("[build] entering AutoParallel (graph export + strategy enumeration)...", flush=True)
t_build = time.perf_counter()
with AutoParallel(model, input_fn, mesh, mp_policy, repeated_subgraphs=True) as autop:
    print(f"[build] AutoParallel ready in {time.perf_counter() - t_build:.2f} s", flush=True)
    autop.add_parameter_memory_constraint(low=None, high=None)
    x_sharding = (Shard(0),) + (Replicate(),) * (mesh.ndim - 1)
    autop.add_input_constraints([x_sharding])
    autop.add_output_constraints([x_sharding])
    print(f"[build+constraints] {time.perf_counter() - t_build:.2f} s")

    opt = autop.sharding_optimizer
    print(f"[problem] unique_vars={len(opt.pulp_variables)} "
          f"constraints={len(opt.prob.constraints)}", flush=True)

    mode = os.environ.get("SOLVE_MODE", "lp")  # lp | ilp | both

    if mode in ("lp", "both"):
        res = opt.get_lower_bound(verbose=False)
        print(f"[LP relaxation] status={res.status} objective={res.objective:.4f}")
        print(f"[LP relaxation] solve_s={res.solve_s:.3f}  total_s={res.total_s:.3f}", flush=True)

    if mode in ("ilp", "both"):
        print("[ILP] solving (this may take a long time)...", flush=True)
        t_ilp = time.perf_counter()
        opt.get_solution(verbose=True)
        import pulp
        obj = pulp.value(opt.prob.objective)
        print(f"[ILP] status={pulp.LpStatus[opt.prob.status]} objective={obj}")
        print(f"[ILP] solve+extract_s={time.perf_counter() - t_ilp:.3f}", flush=True)
