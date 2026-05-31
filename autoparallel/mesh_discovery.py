# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""Automatic device-mesh discovery for AutoParallel.

Users normally must hand-pick a DeviceMesh shape (e.g. ``(dp, tp) = (32, 8)``)
before running AutoParallel, which requires expert knowledge of how TP/CP/EP
map onto the hardware topology. This module removes that requirement: given
``n_gpus`` and an optional topology config, it enumerates candidate mesh shapes,
scores each with the sharding ILP (reusing a single traced graph), and returns
the lowest-cost feasible mesh.

Two candidate generators are provided, both capped at ``MAX_MESH_DIMS`` dims:

  * :func:`topology_tiered_factorization` — factorizes the per-node GPU count
    and the node count into separate tiers, emitting only node-aligned shapes
    where the innermost dims fill whole nodes. This is the small, high-quality
    default for standard topologies.

  * :func:`binary_mesh_partition` — recursively bisects ``n_gpus`` into an outer
    factor and an inner group. Without pruning it enumerates the same space as
    :func:`enumerate_candidate_meshes`; with ``prune_within_node`` it stops
    splitting once a group fits inside a node.

:func:`discover_mesh` runs a branch-and-bound search over candidates: an
optional cheap ``lower_bound`` (e.g. the ILP's LP relaxation) prunes candidates
whose lower bound already exceeds the best feasible cost found so far.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from .cost_models.nccl_cost_model import (
    MAX_MESH_DIMS,
    MeshCandidate,
    NCCLTopoConfig,
    _name_mesh_dims,
    count_intra_node_dims,
    enumerate_candidate_meshes,
    exact_factorizations,
    is_node_aligned,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


def topology_tiered_factorization(
    n_gpus: int,
    topo_config: NCCLTopoConfig,
    max_dims: int = MAX_MESH_DIMS,
) -> list[MeshCandidate]:
    """Generate node-aligned candidate meshes by tiering the factorization.

    The per-node GPU count and the node count are factored independently into
    an inner (intra-node) tier and an outer (inter-node) tier, then concatenated
    so the innermost dims fill whole nodes. This emits only topology-friendly
    shapes, a strict subset of :func:`enumerate_candidate_meshes`.

    Falls back to the full enumeration when ``n_gpus`` does not divide evenly
    into nodes (no clean tier boundary exists).
    """
    gpn = topo_config.gpus_per_node
    if n_gpus <= gpn:
        intra_size, n_nodes = n_gpus, 1
    elif n_gpus % gpn == 0:
        intra_size, n_nodes = gpn, n_gpus // gpn
    else:
        logger.warning(
            "n_gpus=%d is not divisible by gpus_per_node=%d; topology tiering "
            "cannot align to node boundaries, falling back to full enumeration.",
            n_gpus,
            gpn,
        )
        return enumerate_candidate_meshes(n_gpus, topo_config, max_dims)

    seen: set[tuple[int, ...]] = set()
    candidates: list[MeshCandidate] = []
    for n_intra in range(1, max_dims + 1):
        for n_inter in range(0, max_dims - n_intra + 1):
            for inter in exact_factorizations(n_nodes, n_inter):
                for intra in exact_factorizations(intra_size, n_intra):
                    shape = inter + intra
                    if not shape or shape in seen:
                        continue
                    seen.add(shape)
                    candidates.append(
                        MeshCandidate(
                            shape=shape,
                            dim_names=_name_mesh_dims(len(shape)),
                            n_intra_dims=count_intra_node_dims(shape, topo_config),
                        )
                    )
    candidates.sort(key=lambda c: (-c.n_intra_dims, c.ndim, c.shape))
    return candidates


def _binary_partition(
    g: int, max_dims: int, gpus_per_node: Optional[int], prune_within_node: bool
) -> list[tuple[int, ...]]:
    shapes: list[tuple[int, ...]] = [(g,)]
    # Once a group fits inside a node, optionally keep it as a single intra-node
    # dim instead of splitting it further.
    if prune_within_node and gpus_per_node is not None and g <= gpus_per_node:
        return shapes
    if max_dims < 2:
        return shapes
    for outer in range(2, g // 2 + 1):
        if g % outer != 0:
            continue
        for rest in _binary_partition(
            g // outer, max_dims - 1, gpus_per_node, prune_within_node
        ):
            shapes.append((outer,) + rest)
    return shapes


def binary_mesh_partition(
    n_gpus: int,
    topo_config: NCCLTopoConfig | None = None,
    max_dims: int = MAX_MESH_DIMS,
    prune_within_node: bool = False,
) -> list[MeshCandidate]:
    """Generate candidate meshes by recursive binary partitioning.

    At each step the remaining group of ``g`` GPUs is either left as one mesh
    dimension or split into an outer factor and an inner group, recursively. With
    ``prune_within_node=False`` and no topology this enumerates the same set as
    :func:`enumerate_candidate_meshes`; with ``prune_within_node=True`` it stops
    splitting groups that already fit inside a node.
    """
    gpn = topo_config.gpus_per_node if topo_config is not None else None
    shapes = _binary_partition(n_gpus, max_dims, gpn, prune_within_node)
    candidates = [
        MeshCandidate(
            shape=shape,
            dim_names=_name_mesh_dims(len(shape)),
            n_intra_dims=count_intra_node_dims(shape, topo_config),
        )
        for shape in shapes
    ]
    if topo_config is not None:
        aligned_gpn = topo_config.gpus_per_node
        candidates.sort(
            key=lambda c: (
                0 if is_node_aligned(c.shape, aligned_gpn) else 1,
                -c.n_intra_dims,
                c.ndim,
                c.shape,
            )
        )
    return candidates


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


@dataclass
class MeshEvaluation:
    """The outcome of scoring a single candidate mesh."""

    candidate: MeshCandidate
    cost: float
    feasible: bool
    pruned: bool = False
    lower_bound: Optional[float] = None


@dataclass
class DiscoveryResult:
    best: MeshCandidate
    best_cost: float
    evaluations: list[MeshEvaluation] = field(default_factory=list)

    @property
    def n_evaluated(self) -> int:
        return sum(1 for e in self.evaluations if not e.pruned)

    @property
    def n_pruned(self) -> int:
        return sum(1 for e in self.evaluations if e.pruned)


def discover_mesh(
    candidates: list[MeshCandidate],
    evaluate: Callable[[MeshCandidate], tuple[float, bool]],
    lower_bound: Optional[Callable[[MeshCandidate], float]] = None,
    verbose: bool = False,
) -> DiscoveryResult:
    """Branch-and-bound search for the lowest-cost feasible mesh.

    Args:
        candidates: meshes to consider (evaluated in the given order, except
            that a ``lower_bound`` re-sorts them cheapest-bound-first).
        evaluate: maps a candidate to ``(cost, feasible)``. Expensive (full ILP
            solve); only called for candidates not pruned by ``lower_bound``.
        lower_bound: optional cheap, admissible lower bound on a candidate's
            cost (e.g. the ILP's LP relaxation). When a candidate's bound is
            ``>=`` the best feasible cost found so far, ``evaluate`` is skipped.
        verbose: log per-candidate progress.

    Returns:
        A :class:`DiscoveryResult` with the winning candidate, its cost, and a
        per-candidate evaluation log.
    """
    if not candidates:
        raise ValueError("No candidate meshes to evaluate")

    order = candidates
    bounds: dict[tuple[int, ...], float] = {}
    if lower_bound is not None:
        for c in candidates:
            bounds[c.shape] = lower_bound(c)
        order = sorted(candidates, key=lambda c: bounds[c.shape])

    best_cost = math.inf
    best: Optional[MeshCandidate] = None
    evaluations: list[MeshEvaluation] = []

    for cand in order:
        lb = bounds.get(cand.shape)
        if lb is not None and lb >= best_cost:
            if verbose:
                logger.info(
                    "mesh %s pruned (lower bound %.4f >= incumbent %.4f)",
                    cand.shape,
                    lb,
                    best_cost,
                )
            evaluations.append(
                MeshEvaluation(
                    cand, cost=lb, feasible=False, pruned=True, lower_bound=lb
                )
            )
            continue

        cost, feasible = evaluate(cand)
        evaluations.append(
            MeshEvaluation(cand, cost=cost, feasible=feasible, lower_bound=lb)
        )
        if verbose:
            logger.info("mesh %s -> cost=%.4f feasible=%s", cand.shape, cost, feasible)
        if feasible and cost < best_cost:
            best_cost = cost
            best = cand

    if best is None:
        raise RuntimeError(
            "Mesh discovery found no feasible mesh among "
            f"{len(candidates)} candidates. Constraints may be too strict."
        )

    if verbose:
        logger.info(
            "selected mesh %s (cost=%.4f); evaluated %d, pruned %d",
            best.shape,
            best_cost,
            sum(1 for e in evaluations if not e.pruned),
            sum(1 for e in evaluations if e.pruned),
        )
    return DiscoveryResult(best=best, best_cost=best_cost, evaluations=evaluations)


# ---------------------------------------------------------------------------
# DeviceMesh construction
# ---------------------------------------------------------------------------


def build_device_mesh(candidate: MeshCandidate, device_type: str = "cuda"):
    """Materialize a :class:`DeviceMesh` from a candidate.

    DeviceMesh construction reads concrete rank tensors, so it must run outside
    any active FakeTensorMode (e.g. the one pushed during graph tracing).
    """
    from torch._subclasses.fake_tensor import unset_fake_temporarily
    from torch.distributed.device_mesh import init_device_mesh

    with unset_fake_temporarily():
        return init_device_mesh(
            device_type,
            candidate.shape,
            mesh_dim_names=candidate.dim_names,
        )
