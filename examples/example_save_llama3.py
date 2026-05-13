# Example: Trace LLaMA-3 8B, optimize, and save the optimizer state.
#
# This script runs the expensive tracing + ILP construction once and saves
# the result.  The saved file can then be loaded in a notebook for
# interactive exploration without needing the model code or a process group.

import logging
import time

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel._testing.models.llama3 import Transformer, TransformerModelArgs
from autoparallel.api import AutoParallel

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("autoparallel")
logger.setLevel(logging.DEBUG)

world_size = 64
fake_store = FakeStore()
torch.distributed.init_process_group(
    "fake", store=fake_store, rank=0, world_size=world_size
)

mesh = torch.distributed.device_mesh.init_device_mesh(
    "cuda",
    (world_size // 8, 8),
    mesh_dim_names=("dp", "tp"),
)

batch_size = 2 * mesh.shape[0]
seqlen = 2048 * 4
vocab_size = 128256


def model_fn():
    return Transformer(
        TransformerModelArgs(
            dim=4096,
            n_layers=32,
            n_heads=32,
            n_kv_heads=8,
            ffn_dim_multiplier=1.3,
            multiple_of=1024,
            rope_theta=500000,
            vocab_size=vocab_size,
            max_seq_len=seqlen,
        )
    )


def input_fn():
    return torch.randint(0, vocab_size, (batch_size, seqlen), device="cuda")


with torch.device("meta"):
    model = model_fn()

mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

t0 = time.time()
with AutoParallel(model, input_fn, mesh, mp_policy) as autop:
    autop.add_parameter_memory_constraint(low=None, high=None)
    x_sharding = (Shard(0),) + (Replicate(),) * (mesh.ndim - 1)
    autop.add_input_constraints([x_sharding])
    autop.add_output_constraints([(Shard(0), Shard(2))])

    sharding_placement = autop.optimize_placement(verbose=False)
    print(f"Tracing + optimization took {time.time() - t0:.1f}s")

    # Save the full optimizer state
    t_save = time.time()
    autop.sharding_optimizer.save("llama3_8b.ap")
    print(f"Saved optimizer state to llama3_8b.ap ({time.time() - t_save:.1f}s)")

    # Also save a lightweight placements file
    autop.sharding_optimizer.save_placements("llama3_8b_solution.json")
    print("Saved placements to llama3_8b_solution.json")
    print(f"Total time: {time.time() - t0:.1f}s")
