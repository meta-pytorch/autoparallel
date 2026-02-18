# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Comparison of three sharding optimizers:
  1. Original enumeration-based ILP (ShardingOptimizer)
  2. Factor-based ILP (FactorShardingOptimizer)
  3. Independent per-mesh-dim ILP (IndependentShardingOptimizer)

Uses the same transformer Block model as example_autoparallel.py, but instead
of running the full AutoParallel pipeline, it traces the model and runs all
three optimizers on the same FX graph for comparison.

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
from autoparallel.optimize_sharding_independent import IndependentShardingOptimizer

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

world_size = 128 # 64

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
    orig_opt.add_input_constraints([x_sharding])
    orig_opt.add_output_constraints([x_sharding])
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
    factor_opt.add_grad_param_constraints()
    factor_opt.add_input_constraints([x_sharding])
    factor_opt.add_output_constraints([x_sharding])
    factor_solution = factor_opt.get_solution(verbose=False)
    t_factor = time.perf_counter() - t0

    print(f"  Solve time: {t_factor:.2f}s")
    print(factor_opt.get_log(verbose=True))

    # parallel_mod = autop.apply_placement(factor_solution)

    # ------------------------------------------------------------------
    # 3. Independent per-mesh-dim optimizer
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Running INDEPENDENT per-mesh-dim IndependentShardingOptimizer")
    print("=" * 70)

    t0 = time.perf_counter()
    indep_opt = IndependentShardingOptimizer(gm, mesh)
    indep_opt.add_grad_param_constraints()
    indep_opt.add_input_constraints([x_sharding])
    indep_opt.add_output_constraints([x_sharding])
    indep_solution = indep_opt.get_solution(verbose=False)
    t_indep = time.perf_counter() - t0

    print(f"  Solve time: {t_indep:.2f}s")
    print(indep_opt.get_log(verbose=True))

    # ------------------------------------------------------------------
    # 4. Comparison
    # ------------------------------------------------------------------
    stats = factor_opt.get_stats()
    orig_stats = orig_opt.get_stats()
    indep_stats = indep_opt.get_stats()

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"  Mesh shape:              {tuple(mesh.shape)}")
    print(f"  Graph nodes:             {stats['num_graph_nodes']}")
    print()
    print(f"  Original ILP variables:     {orig_stats['num_ilp_variables']:,}")
    print(f"  Factor ILP variables:       {stats['num_ilp_variables']:,}")
    print(f"  Independent ILP variables:  {indep_stats['num_ilp_variables']:,}")
    print()
    print(f"  Original ILP constraints:   {orig_stats['num_ilp_constraints']:,}")
    print(f"  Factor ILP constraints:     {stats['num_ilp_constraints']:,}")
    print(f"  Independent ILP constraints:{indep_stats['num_ilp_constraints']:,}")
    print()
    print(f"  Unique factors:          {stats['num_unique_factors']}")
    print()
    print(f"  Original solve time:     {t_orig:.2f}s")
    print(f"  Factor solve time:       {t_factor:.2f}s")
    print(f"  Independent solve time:  {t_indep:.2f}s")

    # ------------------------------------------------------------------
    # 5. Show per-node placement comparison
    # ------------------------------------------------------------------
    n_show = 100
    print("\n" + "=" * 70)
    print(f"PER-NODE PLACEMENT COMPARISON (first {n_show} call_function nodes)")
    print("=" * 70)

    def _get_placements(spec):
        if spec is not None and hasattr(spec, "output_specs"):
            os = spec.output_specs
            if isinstance(os, DTensorSpec):
                return tuple(os.placements)
            elif isinstance(os, (list, tuple)) and os:
                return tuple(os[0].placements)
        return "?"

    call_fn_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
    for node in call_fn_nodes[:n_show]:
        orig_plc = _get_placements(orig_solution.get(node))
        factor_plc = _get_placements(factor_solution.get(node))
        indep_plc = _get_placements(indep_solution.get(node))

        match_f = "OK" if str(orig_plc) == str(factor_plc) else "DIFF"
        match_i = "OK" if str(orig_plc) == str(indep_plc) else "DIFF"
        op_name = str(node)
        if len(op_name) > 35:
            op_name = op_name[:32] + "..."
        print(
            f"  {op_name:37s}  orig={str(orig_plc):30s}  "
            f"factor[{match_f:4s}]={str(factor_plc):30s}  "
            f"indep[{match_i:4s}]={str(indep_plc)}"
        )

print("\nDone.")
