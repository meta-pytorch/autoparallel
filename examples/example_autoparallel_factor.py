# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Comparison of the original enumeration-based ILP vs the factor-based ILP.

Uses the same transformer Block model as example_autoparallel.py, but instead
of running the full AutoParallel pipeline, it:
  1. Traces the model to obtain the FX graph.
  2. Runs the *original* ShardingOptimizer (enumeration-based).
  3. Runs the *factor-based* FactorShardingOptimizer on the same graph.
  4. Prints a side-by-side comparison of ILP sizes and solutions.

Usage:
    python examples/example_autoparallel_factor.py
"""

import time

import torch
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel.api import AutoParallel
from autoparallel.optimize_sharding import ShardingOptimizer
from autoparallel.optimize_sharding_new import FactorShardingOptimizer

# ---------------------------------------------------------------------------
# Model (same as example_autoparallel.py, minus activation checkpointing for
# simplicity)
# ---------------------------------------------------------------------------


class Block(nn.Module):
    def __init__(self, nheads, dim1, dim2):
        super().__init__()
        self.nheads = nheads
        bias = False
        self.wq = nn.Linear(dim1, dim1, bias=bias)
        self.wk = nn.Linear(dim1, dim1, bias=bias)
        self.wv = nn.Linear(dim1, dim1, bias=bias)
        self.wo = nn.Linear(dim1, dim1, bias=bias)
        self.w1 = nn.Linear(dim1, dim2, bias=bias)
        self.w2 = nn.Linear(dim2, dim1, bias=bias)

    def forward(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
        k = k.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
        v = v.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)

        o = nn.functional.scaled_dot_product_attention(q, k, v)
        o = o.permute(0, 2, 1, 3).flatten(-2)

        o = self.wo(o)
        o0 = o + x

        o = self.w1(o0)
        o = torch.nn.functional.relu(o)
        o = self.w2(o)

        o = o0 + o
        return o


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

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

bs = 8 * mesh.shape[0]
seq_len = 256
nheads = 48
dim1 = 6144
dim2 = dim1 * 4


def input_fn():
    return torch.rand(bs, seq_len, dim1, device="cuda")


x_sharding = (Shard(0),) + (Replicate(),) * (mesh.ndim - 1)

mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
mp_policy = None

# ---------------------------------------------------------------------------
# Trace the model (reuse AutoParallel for graph capture only)
# ---------------------------------------------------------------------------

print("=" * 70)
print("Tracing model...")
print("=" * 70)

with torch.device("meta"):
    model = Block(nheads, dim1, dim2)

with AutoParallel(model, input_fn, mesh, mp_policy) as autop:
    gm = autop.gm  # the traced FX graph (joint fwd + bwd)

    # ------------------------------------------------------------------
    # 1. Original (enumeration-based) optimizer
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Running ORIGINAL (enumeration-based) ShardingOptimizer")
    print("=" * 70)

    t0 = time.perf_counter()
    orig_opt = ShardingOptimizer(gm, mesh)
    orig_opt.add_grad_param_constraints()
    orig_opt.add_sharded_input_constraint([x_sharding])
    orig_opt.add_sharded_output_constraint([x_sharding])
    orig_solution = orig_opt.get_solution(verbose=False)
    t_orig = time.perf_counter() - t0

    print(f"  Solve time: {t_orig:.2f}s")
    print(f"  ILP variables:   {len(orig_opt.ds):,}")
    print(f"  ILP constraints: {len(orig_opt.prob.constraints):,}")
    print(f"  ILP status:      {orig_opt.prob.status}")

    # ------------------------------------------------------------------
    # 2. Factor-based optimizer
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Running FACTOR-BASED FactorShardingOptimizer")
    print("=" * 70)

    t0 = time.perf_counter()
    factor_opt = FactorShardingOptimizer(gm, mesh)
    factor_opt.add_input_constraints([x_sharding])
    factor_opt.add_output_constraints([x_sharding])
    factor_solution = factor_opt.get_solution(verbose=False)
    t_factor = time.perf_counter() - t0

    print(f"  Solve time: {t_factor:.2f}s")
    print(factor_opt.get_log(verbose=True))

    # ------------------------------------------------------------------
    # 3. Comparison
    # ------------------------------------------------------------------
    stats = factor_opt.get_stats()

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"  Mesh shape:              {tuple(mesh.shape)}")
    print(f"  Graph nodes:             {stats['num_graph_nodes']}")
    print()
    print(f"  Original ILP variables:  {len(orig_opt.ds):,}")
    print(f"  Factor ILP variables:    {stats['num_factor_ilp_variables']:,}")
    print(f"  Variable reduction:      {stats['variable_reduction_ratio']:.1f}x")
    print()
    print(f"  Original ILP constraints:{len(orig_opt.prob.constraints):,}")
    print(f"  Factor ILP constraints:  {stats['num_factor_ilp_constraints']:,}")
    print()
    print(f"  Unique factors:          {stats['num_unique_factors']}")

    # ------------------------------------------------------------------
    # 4. Show per-node placement comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PER-NODE PLACEMENT COMPARISON (first 30 call_function nodes)")
    print("=" * 70)

    call_fn_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
    for node in call_fn_nodes[:30]:
        orig_spec = orig_solution.get(node)
        factor_spec = factor_solution.get(node)

        if orig_spec is not None and hasattr(orig_spec, "output_specs"):
            os = orig_spec.output_specs
            if isinstance(os, DTensorSpec):
                orig_plc = tuple(os.placements)
            elif isinstance(os, (list, tuple)) and os:
                orig_plc = tuple(os[0].placements)
            else:
                orig_plc = "?"
        else:
            orig_plc = "?"
        factor_plc = tuple(factor_spec.placements) if factor_spec is not None else "?"
        match = "OK" if str(orig_plc) == str(factor_plc) else "DIFF"
        op_name = str(node)
        # Truncate long op names
        if len(op_name) > 40:
            op_name = op_name[:37] + "..."
        print(f"  [{match:4s}] {op_name:42s}  orig={orig_plc}  factor={factor_plc}")

print("\nDone.")
