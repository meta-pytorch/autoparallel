# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""Automatic mesh discovery on LLaMA-3 1B.

Instead of hand-picking a DeviceMesh shape, pass ``mesh="auto"`` and let
AutoParallel enumerate candidate factorizations of the GPU count, score each
with the sharding ILP (reusing a single traced graph), and select the
lowest-cost feasible mesh.

Run (single node, 8 GPUs):
    python examples/example_auto_mesh.py
"""

import logging
import os
import sys

# Ensure this checkout's autoparallel is imported, not an unrelated editable
# install that may shadow it when the script is run from the examples/ dir.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from torch.distributed.fsdp import MixedPrecisionPolicy  # noqa: E402
from torch.distributed.tensor.placement_types import Replicate, Shard  # noqa: E402
from torch.testing._internal.distributed.fake_pg import FakeStore  # noqa: E402

from autoparallel._testing.models.llama3 import (  # noqa: E402
    Transformer,
    TransformerModelArgs,
)
from autoparallel.api import AutoParallel  # noqa: E402
from autoparallel.cost_models.nccl_cost_model import (  # noqa: E402
    enumerate_candidate_meshes,
    h100_topo_config,
)

logging.basicConfig(level=logging.INFO)

world_size = 8
gpus_per_node = 8

fake_store = FakeStore()
torch.distributed.init_process_group(
    "fake", store=fake_store, rank=0, world_size=world_size
)

# LLaMA-3.2 1B architecture.
vocab_size = 128256
seqlen = 4096
batch_size = 16


def model_fn():
    args = TransformerModelArgs(
        dim=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.5,
        multiple_of=256,
        rope_theta=500000,
        vocab_size=vocab_size,
        max_seq_len=seqlen,
    )
    return Transformer(args)


def input_fn():
    return torch.randint(0, vocab_size, (batch_size, seqlen), device="cuda")


# Use an explicit H100 topology so the example does not depend on local CUDA
# hardware detection.
topo = h100_topo_config(
    num_nodes=world_size // gpus_per_node, gpus_per_node=gpus_per_node
)

mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)


def add_constraints(optimizer, mesh):
    """Constrain each candidate's ILP so discovery is constraint-aware."""
    ndim = mesh.ndim
    x_sharding = (Shard(0),) + (Replicate(),) * (ndim - 1)
    optimizer.add_parameter_memory_constraint(0.0, 1.0 / mesh.size())
    optimizer.add_sharded_input_constraint([x_sharding])
    optimizer.add_sharded_output_constraint([x_sharding])


def candidate_fn(n_gpus, topo_config, max_dims):
    """Restrict the search to realistic 1D/2D meshes for a single node.

    On 8 GPUs, 3D+ meshes are impractical and explode the ILP, so we keep the
    pure-FSDP 1D mesh and the FSDP×TP 2D meshes.
    """
    return [
        c
        for c in enumerate_candidate_meshes(n_gpus, topo_config, max_dims)
        if c.ndim <= 2
    ]


with torch.device("meta"):
    model = model_fn()

with AutoParallel(
    model,
    input_fn,
    "auto",
    mp_policy=mp_policy,
    cost_model=topo,
    world_size=world_size,
    mesh_candidate_fn=candidate_fn,
    mesh_constraint_fn=add_constraints,
    mesh_prune=True,
    repeated_subgraphs=True,
) as autop:
    result = autop.mesh_discovery_result

    print("\n=== Mesh discovery results ===")
    print(f"{'shape':<18}{'dim_names':<26}{'cost':>14}  status")
    for e in sorted(
        result.evaluations,
        key=lambda e: (e.cost if e.feasible else float("inf")),
    ):
        if e.pruned:
            status = f"pruned (lb={e.lower_bound:.1f})"
        elif not e.feasible:
            status = "infeasible"
        else:
            status = "evaluated"
        cost = f"{e.cost:.2f}" if e.feasible else "-"
        print(
            f"{str(e.candidate.shape):<18}"
            f"{str(e.candidate.dim_names):<26}"
            f"{cost:>14}  {status}"
        )

    print(
        f"\nSelected mesh: shape={result.best.shape} "
        f"names={result.best.dim_names} cost={result.best_cost:.2f}"
    )
    print(
        f"Candidates: {len(result.evaluations)} total, "
        f"{result.n_evaluated} fully evaluated, {result.n_pruned} pruned\n"
    )

    # The selected mesh's optimizer already has the discovery-time constraints;
    # re-solve to obtain the final placement on the chosen mesh.
    sharding_placement = autop.optimize_placement(verbose=False)
    print(f"Optimized placement on chosen mesh {tuple(autop.mesh.shape)}: OK")

print("All good!")
