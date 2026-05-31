# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from math import comb, prod

import pytest
import torch
from torch import nn
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel.api import AutoParallel
from autoparallel.cost_models.nccl_cost_model import (
    MeshCandidate,
    count_intra_node_dims,
    enumerate_candidate_meshes,
    exact_factorizations,
    h100_topo_config,
    is_node_aligned,
    ordered_factorizations,
)
from autoparallel.mesh_discovery import (
    DiscoveryResult,
    binary_mesh_partition,
    build_device_mesh,
    discover_mesh,
    topology_tiered_factorization,
)

# ---------------------------------------------------------------------------
# Candidate enumeration combinatorics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_gpus,m",
    [(256, 8), (2048, 11), (8192, 13), (65536, 16)],
)
def test_ordered_factorization_counts_match_table(n_gpus, m):
    # Problem-statement table: ordered factorizations of 2^M into exactly k
    # parts number C(M-1, k-1).
    for n_parts in (1, 2, 3, 4):
        assert len(exact_factorizations(n_gpus, n_parts)) == comb(m - 1, n_parts - 1)
    total = len(ordered_factorizations(n_gpus, 4))
    assert total == sum(comb(m - 1, k - 1) for k in range(1, 5))


def test_ordered_factorizations_are_valid_and_ordered():
    facts = ordered_factorizations(64, 4)
    # All factor products equal n, all factors >= 2, capped at 4 parts.
    for f in facts:
        assert prod(f) == 64
        assert all(x >= 2 for x in f)
        assert 1 <= len(f) <= 4
    # Ordered: (16, 4) and (4, 16) are both present and distinct.
    assert (16, 4) in facts
    assert (4, 16) in facts
    # No duplicates.
    assert len(facts) == len(set(facts))


def test_ordered_factorizations_non_power_of_two():
    # 72 = GB200 node. Factors need not be powers of two.
    facts = ordered_factorizations(72, 4)
    for f in facts:
        assert prod(f) == 72
        assert all(x >= 2 for x in f)
    assert (72,) in facts
    assert (8, 9) in facts
    assert (9, 8) in facts
    assert (2, 2, 2, 9) in facts


def test_exact_factorizations_zero_parts():
    assert exact_factorizations(1, 0) == ((),)
    assert exact_factorizations(2, 0) == ()


# ---------------------------------------------------------------------------
# Topology classification
# ---------------------------------------------------------------------------


def test_count_intra_node_dims():
    topo = h100_topo_config(num_nodes=32, gpus_per_node=8)
    # (32, 8): inner dim of 8 fills a node -> 1 intra dim; 32 spans nodes.
    assert count_intra_node_dims((32, 8), topo) == 1
    # (32, 2, 2, 2): trailing 2*2*2 == 8 == node -> 3 intra dims.
    assert count_intra_node_dims((32, 2, 2, 2), topo) == 3
    # (256,): a single dim spanning all nodes -> 0 intra dims.
    assert count_intra_node_dims((256,), topo) == 0
    # No topology -> 0.
    assert count_intra_node_dims((32, 8), None) == 0


def test_is_node_aligned():
    assert is_node_aligned((32, 8), 8)
    assert is_node_aligned((4, 2, 2, 2), 8)
    assert is_node_aligned((4,), 8)  # fits within a node
    assert not is_node_aligned((64, 4), 8)  # suffix product 4 != 8, 256 > 8
    assert not is_node_aligned((16, 16), 8)


def test_mesh_candidate_dim_names():
    cands = {c.ndim: c for c in enumerate_candidate_meshes(256, None, 4)}
    assert cands[1].dim_names == ("dp",)
    assert cands[2].dim_names == ("dp", "tp")
    assert cands[3].dim_names == ("dp", "cp", "tp")
    assert cands[4].dim_names == ("dp", "cp", "ep", "tp")


def test_enumerate_orders_node_aligned_first():
    topo = h100_topo_config(num_nodes=32, gpus_per_node=8)
    cands = enumerate_candidate_meshes(256, topo, 4)
    aligned = [is_node_aligned(c.shape, 8) for c in cands]
    # Once we hit the first non-aligned candidate, no aligned one follows.
    first_unaligned = aligned.index(False) if False in aligned else len(aligned)
    assert all(aligned[:first_unaligned])
    assert not any(aligned[first_unaligned:])


# ---------------------------------------------------------------------------
# Generators agree / specialize
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_gpus", [256, 1024, 4096])
def test_binary_partition_equals_ordered_factorizations(n_gpus):
    binary = {c.shape for c in binary_mesh_partition(n_gpus, None, 4)}
    ordered = set(ordered_factorizations(n_gpus, 4))
    assert binary == ordered


def test_binary_partition_prune_within_node():
    topo = h100_topo_config(num_nodes=32, gpus_per_node=8)
    pruned = {
        c.shape for c in binary_mesh_partition(256, topo, 4, prune_within_node=True)
    }
    full = {
        c.shape for c in binary_mesh_partition(256, topo, 4, prune_within_node=False)
    }
    assert pruned < full
    # Pruning stops splitting groups <= gpus_per_node, so no candidate splits a
    # trailing in-node group into multiple dims.
    assert (32, 2, 2, 2) not in pruned
    assert (32, 8) in pruned


def test_topology_tiered_is_node_aligned_subset():
    topo = h100_topo_config(num_nodes=32, gpus_per_node=8)
    tiered = topology_tiered_factorization(256, topo, 4)
    full = {c.shape for c in enumerate_candidate_meshes(256, topo, 4)}
    for c in tiered:
        assert prod(c.shape) == 256
        assert is_node_aligned(c.shape, 8)
        assert c.shape in full
    # Every tiered candidate has its inner tier exactly fill nodes.
    for c in tiered:
        assert c.n_intra_dims >= 1


def test_topology_tiered_single_node():
    topo = h100_topo_config(num_nodes=1, gpus_per_node=8)
    tiered = topology_tiered_factorization(8, topo, 4)
    shapes = {c.shape for c in tiered}
    # All 8 GPUs are intra-node: candidates are factorizations of 8.
    assert shapes == set(ordered_factorizations(8, 4))
    for c in tiered:
        assert c.n_intra_dims == c.ndim


def test_topology_tiered_non_divisible_falls_back():
    topo = h100_topo_config(num_nodes=1, gpus_per_node=8)
    # 12 > 8 and not divisible by 8 -> no clean tier boundary, fall back to
    # full enumeration.
    tiered = {c.shape for c in topology_tiered_factorization(12, topo, 4)}
    full = {c.shape for c in enumerate_candidate_meshes(12, topo, 4)}
    assert tiered == full


# ---------------------------------------------------------------------------
# discover_mesh search
# ---------------------------------------------------------------------------


def _candidates(shapes):
    return [MeshCandidate(s, tuple(f"d{i}" for i in range(len(s))), 0) for s in shapes]


def test_discover_mesh_picks_minimum_cost():
    cands = _candidates([(8,), (2, 4), (4, 2), (2, 2, 2)])
    costs = {(8,): 10.0, (2, 4): 3.0, (4, 2): 7.0, (2, 2, 2): 5.0}

    def evaluate(c):
        return costs[c.shape], True

    result = discover_mesh(cands, evaluate)
    assert result.best.shape == (2, 4)
    assert result.best_cost == 3.0
    assert result.n_evaluated == 4
    assert result.n_pruned == 0


def test_discover_mesh_skips_infeasible():
    cands = _candidates([(8,), (2, 4)])

    def evaluate(c):
        if c.shape == (2, 4):
            return float("inf"), False
        return 10.0, True

    result = discover_mesh(cands, evaluate)
    assert result.best.shape == (8,)


def test_discover_mesh_all_infeasible_raises():
    cands = _candidates([(8,), (2, 4)])
    with pytest.raises(RuntimeError, match="no feasible mesh"):
        discover_mesh(cands, lambda c: (float("inf"), False))


def test_discover_mesh_empty_raises():
    with pytest.raises(ValueError, match="No candidate"):
        discover_mesh([], lambda c: (0.0, True))


def test_discover_mesh_lower_bound_pruning():
    cands = _candidates([(8,), (2, 4), (4, 2), (2, 2, 2)])
    # True costs; lower bounds are admissible (<= true cost).
    costs = {(8,): 10.0, (2, 4): 3.0, (4, 2): 7.0, (2, 2, 2): 5.0}
    bounds = {(8,): 9.0, (2, 4): 1.0, (4, 2): 6.0, (2, 2, 2): 4.0}
    evaluated = []

    def evaluate(c):
        evaluated.append(c.shape)
        return costs[c.shape], True

    def lower_bound(c):
        return bounds[c.shape]

    result = discover_mesh(cands, evaluate, lower_bound=lower_bound)
    assert result.best.shape == (2, 4)
    assert result.best_cost == 3.0
    # Order by bound: (2,4)=1 evaluated (cost 3, incumbent=3). Then (2,2,2)
    # bound=4 >= 3 -> pruned; (4,2) bound=6 >= 3 -> pruned; (8,) bound=9 -> pruned.
    assert evaluated == [(2, 4)]
    assert result.n_evaluated == 1
    assert result.n_pruned == 3
    pruned = {e.candidate.shape for e in result.evaluations if e.pruned}
    assert pruned == {(8,), (4, 2), (2, 2, 2)}


# ---------------------------------------------------------------------------
# ShardingOptimizer LP relaxation lower bound (real graph)
# ---------------------------------------------------------------------------


class _MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim, bias=False)
        self.fc2 = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def test_relaxed_cost_is_lower_bound(device_mesh_2d):
    dim = 256
    with torch.device("meta"):
        model = _MLP(dim)

    def input_fn():
        return torch.rand(64, dim, device="cuda")

    with AutoParallel(model, input_fn, device_mesh_2d) as autop:
        autop.add_input_constraints([(Shard(0), Replicate())])
        autop.add_output_constraints([(Shard(0), Replicate())])
        opt = autop.sharding_optimizer
        lb = opt.relaxed_cost()
        solution = autop.optimize_placement(verbose=False)
        assert solution is not None
        import pulp

        ilp_cost = pulp.value(opt.prob.objective)
        # LP relaxation is an admissible lower bound on the ILP optimum.
        assert lb <= ilp_cost + 1e-6
        # Variable categories were restored to binary.
        assert all(dv.var.cat == pulp.LpBinary for dv in opt.decision_vars.values())


# ---------------------------------------------------------------------------
# AutoParallel mesh="auto" integration
# ---------------------------------------------------------------------------


def test_auto_mesh_discovers_and_optimizes():
    world_size = torch.distributed.get_world_size()
    dim = 256
    with torch.device("meta"):
        model = _MLP(dim)

    def input_fn():
        return torch.rand(64, dim, device="cuda")

    topo = h100_topo_config(num_nodes=world_size // 8, gpus_per_node=8)

    # Limit the candidate set so the test builds only a few optimizers.
    def candidate_fn(n_gpus, topo_config, max_dims):
        return [
            c
            for c in enumerate_candidate_meshes(n_gpus, topo_config, max_dims)
            if c.shape in {(world_size,), (world_size // 8, 8)}
        ]

    with AutoParallel(
        model,
        input_fn,
        "auto",
        cost_model=topo,
        mesh_candidate_fn=candidate_fn,
    ) as autop:
        # A concrete mesh was selected.
        assert autop.mesh is not None
        assert autop.mesh.size() == world_size
        assert autop.mesh_discovery_result is not None
        result = autop.mesh_discovery_result
        assert isinstance(result, DiscoveryResult)
        assert result.best.size == world_size
        # optimize_placement works on the chosen optimizer.
        ndim = autop.mesh.ndim
        x_sharding = (Shard(0),) + (Replicate(),) * (ndim - 1)
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([x_sharding])
        solution = autop.optimize_placement(verbose=False)
        assert solution is not None


def test_auto_mesh_reuses_traced_graph():
    # The traced graph is built once and reused across all candidate meshes.
    world_size = torch.distributed.get_world_size()
    dim = 128
    with torch.device("meta"):
        model = _MLP(dim)

    def input_fn():
        return torch.rand(32, dim, device="cuda")

    topo = h100_topo_config(num_nodes=world_size // 8, gpus_per_node=8)

    def candidate_fn(n_gpus, topo_config, max_dims):
        return [
            c
            for c in enumerate_candidate_meshes(n_gpus, topo_config, max_dims)
            if c.shape in {(world_size,), (world_size // 8, 8)}
        ]

    with AutoParallel(
        model, input_fn, "auto", cost_model=topo, mesh_candidate_fn=candidate_fn
    ) as autop:
        # Both candidate optimizers were built from the same traced graph module.
        evals = autop.mesh_discovery_result.evaluations
        assert len({e.candidate.shape for e in evals}) == 2
        assert autop.gm is autop.sharding_optimizer.orig_gm


def test_auto_mesh_pruning_runs():
    world_size = torch.distributed.get_world_size()
    dim = 128
    with torch.device("meta"):
        model = _MLP(dim)

    def input_fn():
        return torch.rand(32, dim, device="cuda")

    topo = h100_topo_config(num_nodes=world_size // 8, gpus_per_node=8)

    def candidate_fn(n_gpus, topo_config, max_dims):
        return [
            c
            for c in enumerate_candidate_meshes(n_gpus, topo_config, max_dims)
            if c.shape in {(world_size,), (world_size // 8, 8), (world_size // 2, 2)}
        ]

    with AutoParallel(
        model,
        input_fn,
        "auto",
        cost_model=topo,
        mesh_candidate_fn=candidate_fn,
        mesh_prune=True,
    ) as autop:
        result = autop.mesh_discovery_result
        # Every candidate has a recorded lower bound.
        assert all(e.lower_bound is not None for e in result.evaluations)
        # Selected mesh is feasible and lowest-cost among the evaluated set.
        evaluated = [e for e in result.evaluations if not e.pruned and e.feasible]
        assert result.best_cost == min(e.cost for e in evaluated)


def test_auto_mesh_rejects_bad_string():
    with torch.device("meta"):
        model = _MLP(64)
    with pytest.raises(ValueError, match="must be a DeviceMesh or 'auto'"):
        AutoParallel(model, lambda: torch.rand(8, 64), "automatic")


def test_build_device_mesh():
    world_size = torch.distributed.get_world_size()
    cand = MeshCandidate((world_size // 8, 8), ("dp", "tp"), 1)
    mesh = build_device_mesh(cand, "cuda")
    assert tuple(mesh.shape) == (world_size // 8, 8)
    assert mesh.mesh_dim_names == ("dp", "tp")
