# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding annotations + Shardy-like propagation on LLaMA3-1B (2D mesh).

By default AutoParallel hands the whole sharding decision to the ILP.  At scale
a user usually already knows the tensor-parallel plan ("these projections are
column-parallel, those are row-parallel").  This example shows how to express
that plan as a few *annotations*, propagate it through the graph the way Shardy
does, and turn the unambiguous part of the graph into ILP constraints.

The annotations pin only the **tensor-parallel (tp) axis** of the transformer
body weights.  Everything else -- the data/FSDP axis, the residual stream
(replicate vs sequence-parallel), the vocab/embedding sharding, and where the
collectives go -- is left to the ILP.  Propagation then determines the sharding
of the activations that *follow* from the plan with no resharding and constrains
them, which shrinks the search space and the solve time while leaving the
genuine cost tradeoffs to the solver.

Run it (no GPUs needed -- uses a fake process group):

    python examples/example_llama3_annotated.py
"""

import logging
import time

import pulp
import torch
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel._testing.models.llama3 import (
    Transformer,
    TransformerModelArgs,
    apply_ac,
)
from autoparallel.api import AutoParallel

logging.basicConfig(level=logging.WARNING)

world_size = 64
fake_store = FakeStore()
torch.distributed.init_process_group(
    "fake", store=fake_store, rank=0, world_size=world_size
)

# 2D mesh: data/FSDP on dp, tensor-parallel on tp.
dp, tp = world_size // 8, 8
mesh = torch.distributed.device_mesh.init_device_mesh(
    "cuda", (dp, tp), mesh_dim_names=("dp", "tp")
)

# Small-batch / long-sequence regime, where tensor parallelism is worthwhile.
vocab_size = 128256
seqlen = 2048
batch_size = 2 * dp


def model_fn():
    # LLaMA-3.2-1B-ish config.
    return Transformer(
        TransformerModelArgs(
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
    )


def input_fn():
    return torch.randint(0, vocab_size, (batch_size, seqlen), device="cuda")


def annotate_tp_plan(autop):
    """The 'conscious' tensor-parallel plan, as a handful of annotations.

    Only the tp axis is pinned (the data axis is left ``None`` = open).  A glob
    pattern annotates the matching weight in every layer at once.
    """
    column_parallel = (None, Shard(0))  # shard the output dim (dim 0 of [out, in])
    row_parallel = (None, Shard(1))  # shard the input dim (dim 1 of [out, in])
    for proj in ["wq", "wk", "wv"]:
        autop.annotate_parameter(f"layers.*.attention.{proj}.weight", column_parallel)
    autop.annotate_parameter("layers.*.attention.wo.weight", row_parallel)
    for proj in ["w1", "w3"]:
        autop.annotate_parameter(
            f"layers.*.feed_forward.{proj}.weight", column_parallel
        )
    autop.annotate_parameter("layers.*.feed_forward.w2.weight", row_parallel)


with torch.device("meta"):
    model = model_fn()
apply_ac(model, mode="full")

with AutoParallel(model, input_fn, mesh, repeated_subgraphs=True) as autop:
    autop.add_parameter_memory_constraint(low=None, high=None)
    autop.add_input_constraints([(Shard(0), Replicate())])
    autop.add_output_constraints([(Shard(0), Shard(2))])  # vocab-parallel logits
    opt = autop.sharding_optimizer
    print(
        f"ILP: {len(opt.strats)} nodes, {len(opt.decision_vars)} decision variables "
        f"on a ({dp}, {tp}) mesh"
    )

    # --- Baseline: full ILP, no annotations ---
    t = time.perf_counter()
    autop.optimize_placement(verbose=False)
    t_baseline = time.perf_counter() - t
    obj_baseline = pulp.value(opt.prob.objective)
    print(
        f"baseline full ILP : objective {obj_baseline:11.1f}   solve {t_baseline:6.1f}s"
    )

    # --- Annotated: propagate the TP plan, then solve the reduced problem ---
    annotate_tp_plan(autop)
    result = autop.propagate_annotations(verbose=False)
    t = time.perf_counter()
    opt.resolve(verbose=False)
    t_annotated = time.perf_counter() - t
    obj_annotated = pulp.value(opt.prob.objective)
    print(
        f"annotated + propag: objective {obj_annotated:11.1f}   solve {t_annotated:6.1f}s"
    )

    gap = 100 * (obj_annotated - obj_baseline) / obj_baseline
    print(
        f"\npropagation pinned {result.nodes_determined} nodes "
        f"({result.axis_constraints} per-axis constraints), shrinking the "
        f"output-strategy search space by {100 * result.reduction:.1f}% "
        f"({result.strategies_before} -> {result.strategies_after})"
    )
    print(
        f"objective gap vs full ILP: {gap:+.2f}%   "
        f"solve speedup: {t_baseline / max(t_annotated, 1e-9):.1f}x"
    )
