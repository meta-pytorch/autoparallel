# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Factor-based sharding optimization using Integer Linear Programming (ILP).

This module reformulates the sharding optimization problem using "factors" —
logical dimensions of computation inspired by Shardy's factor-based propagation
(see shardy/dialect/sdy/ir/attrs.td OpShardingRuleAttr).

Key idea
--------
Instead of enumerating all placement combinations per op — which is O((d+1)^k)
per tensor where d = tensor dims and k = mesh dims — each op is decomposed into
factors, and the ILP decides which mesh dimension (if any) shards each factor.

    Original ILP variables per op:  O(A × (d+1)^(2k))
    Factor ILP variables per op:    O(F × k)

    where A = args, F = factors, d = tensor dims, k = mesh dims

For a matmul on a 4D mesh: ~13,000 → ~12 variables per op.

Factor extraction
-----------------
Factors are extracted *generically* from existing DTensor OpStrategy objects by
inspecting placement patterns on a single mesh dimension.  Because most
OpStrategies are Cartesian products of per-mesh-dim "atoms" (via
``expand_to_full_mesh_op_strategy``), each unique non-trivial atom corresponds
to exactly one factor.  This means we reuse all existing DTensor op rules
without writing per-op factor definitions.

Example: ``mm(A[M,K], B[K,N]) -> C[M,N]``

    1D atoms (from mesh dim 0):
        (C=R,    A=R,    B=R   )  → all-replicate, skip
        (C=S(0), A=S(0), B=R   )  → Factor "M": {A.dim0, C.dim0}
        (C=S(1), A=R,    B=S(1))  → Factor "N": {B.dim1, C.dim1}
        (C=P,    A=S(1), B=S(0))  → Factor "K": {A.dim1, B.dim0}, reduction

    ≡ Shardy's  ([i,k],[k,j])->([i,j])  {i=M, j=N, k=K}  reduction={k}
"""

from __future__ import annotations

import operator
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import pulp
import torch
import torch.fx
from torch._functorch._aot_autograd.descriptors import PlainAOTInput, PlainAOTOutput
from torch._functorch._aot_autograd.fx_utils import (
    get_plain_input_and_grad_nodes,
    get_plain_output_and_tangent_nodes,
)
from torch.distributed._tensor.placement_types import TensorMeta
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import OpSpec, OpStrategy
from torch.distributed.tensor.placement_types import Partial, Placement, Replicate, Shard
from torch.utils._pytree import tree_flatten, tree_map_only

from .cost_models.compute_estimation import estimate_strategy_runtime_cost
from .shardings.placement_options import get_placement_options
from .shardings.propagation_rules import _create_all_options

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


def _is_partial(p: Placement) -> bool:
    """Check if a placement is Partial (reduction output)."""
    return isinstance(p, Partial)


@dataclass
class Factor:
    """A computation factor — a logical dimension of the computation.

    Analogous to Shardy's factor concept (from ``OpShardingRuleAttr``).
    For ``C[M,N] = A[M,K] @ B[K,N]``, the factors are M, N, K.

    Attributes
    ----------
    id : int
        Unique id within the parent ``FactorRule``.
    size : int
        Size along this factor (e.g. M=1024).
    is_reduction : bool
        True if this factor is a contraction/reduction dimension (the output
        is ``Partial`` when it is sharded).
    operand_dims : list[int | None]
        For each operand, the tensor dim mapped to this factor (None = not mapped).
    result_dims : list[int | None]
        For each result, the tensor dim mapped to this factor (None = not mapped).
    """

    id: int
    size: int
    is_reduction: bool = False
    operand_dims: list = field(default_factory=list)
    result_dims: list = field(default_factory=list)


@dataclass
class FactorRule:
    """Factor decomposition for one operation.

    Analogous to Shardy's ``OpShardingRuleAttr``.

    Example — ``mm(A[M,K], B[K,N]) -> C[M,N]``::

        factors = [
            Factor(0, M, operand_dims=[0, None], result_dims=[0]),             # M
            Factor(1, N, operand_dims=[None, 1],  result_dims=[1]),            # N
            Factor(2, K, operand_dims=[1, 0],     result_dims=[], is_reduction=True),  # K
        ]
    """

    factors: list[Factor]
    num_operands: int
    num_results: int


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------


class UnionFind:
    """Disjoint-set (union-find) for merging factors across dataflow edges."""

    def __init__(self) -> None:
        self.parent: dict[int, int] = {}
        self.rank: dict[int, int] = {}

    def make_set(self, x: int) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> int:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return rx
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return rx


# ---------------------------------------------------------------------------
# Factor extraction from OpStrategy
# ---------------------------------------------------------------------------


def _get_primary_tensor(val: Any) -> torch.Tensor | None:
    """Extract the primary tensor from a node's meta 'val'.

    For single-tensor outputs, returns the tensor directly.
    For tuple outputs (multi-output ops like SDPA), returns the first tensor.
    """
    if isinstance(val, torch.Tensor):
        return val
    if isinstance(val, (tuple, list)):
        for v in val:
            if isinstance(v, torch.Tensor):
                return v
    return None


def _infer_factor_size(
    node: torch.fx.Node,
    operand_dims: list[int | None],
    result_dims: list[int | None],
) -> int:
    """Infer the size of a factor from the node's tensor metadata."""
    # Try the node's own output first.
    val = node.meta.get("val")
    if val is not None:
        if isinstance(val, torch.Tensor):
            for d in result_dims:
                if d is not None and d < len(val.shape):
                    return val.shape[d]
        elif isinstance(val, (tuple, list)):
            # Multi-output: result_dims[ri] corresponds to output tensor ri.
            for ri, d in enumerate(result_dims):
                if d is not None and ri < len(val):
                    v = val[ri]
                    if isinstance(v, torch.Tensor) and d < len(v.shape):
                        return v.shape[d]
    # Fall back to operand shapes.
    for arg_idx, d in enumerate(operand_dims):
        if d is None or arg_idx >= len(node.args):
            continue
        arg = node.args[arg_idx]
        if isinstance(arg, torch.fx.Node):
            arg_out = _get_primary_tensor(arg.meta.get("val"))
            if arg_out is not None and d < len(arg_out.shape):
                return arg_out.shape[d]
    return 1  # fallback


def extract_factors_from_strategy(
    op_strategy: OpStrategy,
    node: torch.fx.Node,
) -> FactorRule:
    """Convert an ``OpStrategy`` into a ``FactorRule``.

    Each *unique* per-mesh-dimension placement pattern (excluding all-replicate)
    in the strategy set corresponds to one factor.  We inspect mesh dim 0,
    which is valid because ``expand_to_full_mesh_op_strategy`` replicates the
    same ``single_mesh_dim_strategies`` across all mesh dims.

    Parameters
    ----------
    op_strategy : OpStrategy
        Multi-dim strategy (may have been expanded via Cartesian product).
    node : torch.fx.Node
        The FX node, used for shape metadata.

    Returns
    -------
    FactorRule
    """
    if not op_strategy.strategies:
        return FactorRule(factors=[], num_operands=0, num_results=1)

    first_spec = op_strategy.strategies[0]
    num_operands = len(first_spec.input_specs) if first_spec.input_specs else 0

    # Determine number of result tensors.
    first_out = first_spec.output_specs
    if isinstance(first_out, DTensorSpec):
        num_results = 1
    elif isinstance(first_out, (list, tuple)):
        num_results = len(first_out)
    else:
        num_results = 1

    # Collect unique 1-D "atoms" by looking at mesh dim 0.
    seen: dict[str, tuple] = {}
    for spec in op_strategy.strategies:
        out_specs = spec.output_specs
        if isinstance(out_specs, DTensorSpec):
            out_ps = (out_specs.placements[0],)
        elif isinstance(out_specs, (list, tuple)):
            out_ps = tuple(
                s.placements[0] if isinstance(s, DTensorSpec) else None
                for s in out_specs
            )
        else:
            out_ps = (Replicate(),)
        in_ps = tuple(
            s.placements[0]
            for s in (spec.input_specs or [])
            if isinstance(s, DTensorSpec)
        )
        all_ps = out_ps + in_ps
        if all(p is None or isinstance(p, Replicate) for p in all_ps):
            continue  # skip the all-replicate atom
        key = str(all_ps)
        if key not in seen:
            seen[key] = (out_ps, in_ps)

    # Each atom → one Factor.
    factors: list[Factor] = []
    for factor_id, (out_ps, in_ps) in enumerate(seen.values()):
        is_reduction = any(_is_partial(p) for p in out_ps if p is not None)
        result_dims = [
            p.dim if p is not None and isinstance(p, Shard) else None
            for p in out_ps
        ]
        operand_dims = [p.dim if isinstance(p, Shard) else None for p in in_ps]
        size = _infer_factor_size(node, operand_dims, result_dims)
        factors.append(
            Factor(
                id=factor_id,
                size=size,
                is_reduction=is_reduction,
                operand_dims=operand_dims,
                result_dims=result_dims,
            )
        )

    return FactorRule(factors=factors, num_operands=num_operands, num_results=num_results)


def _placeholder_factor_rule(node: torch.fx.Node) -> FactorRule:
    """Create a ``FactorRule`` for a placeholder / get_attr node.

    Each tensor dimension becomes an independent spatial factor.
    """
    out = _get_primary_tensor(node.meta.get("val"))
    if out is None:
        return FactorRule(factors=[], num_operands=0, num_results=1)
    shape = out.shape
    factors = [
        Factor(id=d, size=shape[d], operand_dims=[], result_dims=[d])
        for d in range(len(shape))
    ]
    return FactorRule(factors=factors, num_operands=0, num_results=1)


# ---------------------------------------------------------------------------
# Factor-based sharding optimizer
# ---------------------------------------------------------------------------


class FactorShardingOptimizer:
    """Sharding optimizer using factor-based ILP variables.

    Public API mirrors :class:`ShardingOptimizer` where possible so
    that it can be used as a drop-in replacement (modulo output format).

    Parameters
    ----------
    gm : torch.fx.GraphModule
        Traced FX graph (joint forward + backward).
    mesh : DeviceMesh
        Target device mesh.
    rescale_grad_comm_cost_for_mp : float
        Scaling factor for gradient communication costs (mixed precision).
    """

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        mesh: Any,
        rescale_grad_comm_cost_for_mp: float = 1.0,
    ) -> None:
        self.gm = gm
        self.graph = gm.graph
        self.nodes = list(self.graph.nodes)
        self.mesh = mesh
        self.node_map: dict[torch.fx.Node, int] = {
            n: i for i, n in enumerate(self.nodes)
        }

        # -- Step 1: build multi-dim strategies (reuses existing DTensor rules)
        # NOTE: in a production implementation you would build strategies for a
        # *1-D* mesh only (O(d) per node instead of O((d+1)^k)), e.g. via
        #   flat_mesh = mesh._flatten("flat")
        # For this POC we reuse the real mesh so that all existing op rules work
        # unchanged.  The key savings come from the ILP reformulation below.
        self.strats = self._build_sharding_metadata()

        # -- Step 2: extract factor rules from strategies.
        self.factor_rules: dict[torch.fx.Node, FactorRule] = {}
        self._extract_all_factor_rules()

        # -- Step 3: merge factors across dataflow edges (union-find).
        self.uf = UnionFind()
        self.factor_keys: dict[tuple[int, int], int] = {}  # (node_idx, local) → gid
        self._next_gid = 0
        self._build_factor_graph()

        # -- Step 4: collect per-root metadata for cost model.
        # root → [(node, factor, local_idx)]
        self.factor_ops: dict[
            int, list[tuple[torch.fx.Node, Factor, int]]
        ] = defaultdict(list)
        self._collect_factor_metadata()

        # -- Step 5: build ILP.
        self._cost_cache: dict[torch.fx.Node, float] = {}
        self.prob = pulp.LpProblem("AutoParallel_Factor", pulp.LpMinimize)
        self.y_vars: dict[tuple[int, int], pulp.LpVariable] = {}
        self._build_ilp()

    # -----------------------------------------------------------------
    # Step 1 — strategy building (mirrors ShardingOptimizer)
    # -----------------------------------------------------------------

    def _build_sharding_metadata(self) -> dict[torch.fx.Node, OpStrategy]:
        strats: dict[torch.fx.Node, OpStrategy] = {}
        for node in self.graph.nodes:
            if node.op == "placeholder":
                strats[node] = _create_all_options(
                    self.mesh, node.meta["val"].shape, tensor=node.meta["val"]
                )
            elif node.op == "call_function":
                user_strats = tree_map_only(
                    torch.fx.Node, lambda x: strats[x], node.args
                )
                user_args = tree_map_only(
                    torch.fx.Node, lambda x: x.meta["val"], node.args
                )
                user_kwargs = tree_map_only(
                    torch.fx.Node, lambda x: x.meta["val"], node.kwargs
                )
                strats[node] = get_placement_options(
                    self.mesh,
                    node.target,
                    user_strats,
                    user_args,
                    user_kwargs,
                )
            elif node.op == "get_attr":
                strats[node] = _create_all_options(
                    self.mesh, node.meta["val"].shape, tensor=node.meta["val"]
                )
        return strats

    # -----------------------------------------------------------------
    # Step 2 — factor extraction
    # -----------------------------------------------------------------

    def _extract_all_factor_rules(self) -> None:
        for node in self.graph.nodes:
            if node.op in ("placeholder", "get_attr"):
                self.factor_rules[node] = _placeholder_factor_rule(node)
            elif node.op == "call_function" and node in self.strats:
                self.factor_rules[node] = extract_factors_from_strategy(
                    self.strats[node], node
                )
            elif node.op == "output":
                self.factor_rules[node] = FactorRule(
                    factors=[], num_operands=0, num_results=0
                )

    # -----------------------------------------------------------------
    # Step 3 — factor graph (union-find merging across edges)
    # -----------------------------------------------------------------

    def _alloc_gid(self) -> int:
        gid = self._next_gid
        self._next_gid += 1
        return gid

    def _build_factor_graph(self) -> None:
        # Register every factor.
        for node in self.graph.nodes:
            rule = self.factor_rules.get(node)
            if rule is None:
                continue
            nidx = self.node_map[node]
            for li, _ in enumerate(rule.factors):
                gid = self._alloc_gid()
                self.factor_keys[(nidx, li)] = gid
                self.uf.make_set(gid)

        # Merge spatial factors across producer → consumer edges.
        for node in self.graph.nodes:
            if node.op != "call_function":
                continue
            consumer_rule = self.factor_rules.get(node)
            if consumer_rule is None:
                continue
            cidx = self.node_map[node]

            for arg_pos, arg in enumerate(node.args):
                if not isinstance(arg, torch.fx.Node):
                    continue
                producer_rule = self.factor_rules.get(arg)
                if producer_rule is None:
                    continue
                pidx = self.node_map[arg]

                # For getitem nodes, the consumer's operand 0 corresponds
                # to a specific result index of the multi-output producer.
                result_idx = 0
                if (
                    node.target is operator.getitem
                    and arg_pos == 0
                    and len(node.args) > 1
                    and isinstance(node.args[1], int)
                ):
                    result_idx = node.args[1]

                # Match: consumer operand dim == producer result dim on the
                # same positional dimension → same logical factor.
                for c_li, c_fac in enumerate(consumer_rule.factors):
                    if arg_pos >= len(c_fac.operand_dims):
                        continue
                    c_dim = c_fac.operand_dims[arg_pos]
                    if c_dim is not None:
                        # Spatial factor: match by dimension index.
                        for p_li, p_fac in enumerate(producer_rule.factors):
                            if not p_fac.result_dims:
                                continue
                            if result_idx >= len(p_fac.result_dims):
                                continue
                            p_dim = p_fac.result_dims[result_idx]
                            if p_dim is not None and p_dim == c_dim:
                                pk = self.factor_keys.get((pidx, p_li))
                                ck = self.factor_keys.get((cidx, c_li))
                                if pk is not None and ck is not None:
                                    self.uf.union(pk, ck)

        # Merge reduction factors (Partial → Partial pass-through) in a
        # separate pass, after spatial merging is complete.
        self._merge_reduction_factors()

    def _merge_reduction_factors(self) -> None:
        """Merge reduction factors across edges only when ALL operands agree.

        For data-preserving ops (view, permute, alias, …), a Partial can
        propagate through the op because the strategy has an atom
        ``(out=Partial, in=Partial)`` which produces a reduction factor with
        ``operand_dims=[None]`` and ``result_dims=[None]``.

        For multi-operand ops like add/mul, the reduction factor has
        ``operand_dims=[None, None]`` — ALL operands must be Partial.
        We must only merge when every operand that maps to ``None`` in the
        factor can provide a Partial from its producer.  Otherwise, the
        resulting placement (e.g. ``add(Partial, Shard)``) is invalid.
        """
        for node in self.graph.nodes:
            if node.op != "call_function":
                continue
            consumer_rule = self.factor_rules.get(node)
            if consumer_rule is None:
                continue
            cidx = self.node_map[node]

            for c_li, c_fac in enumerate(consumer_rule.factors):
                if not c_fac.is_reduction:
                    continue
                ck = self.factor_keys.get((cidx, c_li))
                if ck is None:
                    continue

                # Collect producer reduction keys for each operand that
                # needs Partial, and validate that ALL can provide it.
                merge_pairs: list[tuple[int, int]] = []
                all_valid = True

                for arg_pos, c_od in enumerate(c_fac.operand_dims):
                    if c_od is not None:
                        continue  # Spatial dim on this operand, not Partial

                    # This operand must be Partial for the factor to propagate.
                    if arg_pos >= len(node.args):
                        all_valid = False
                        break
                    arg = node.args[arg_pos]
                    if not isinstance(arg, torch.fx.Node):
                        all_valid = False
                        break
                    producer_rule = self.factor_rules.get(arg)
                    if producer_rule is None:
                        all_valid = False
                        break
                    pidx = self.node_map[arg]

                    # Find a reduction factor at the producer.
                    found = False
                    for p_li, p_fac in enumerate(producer_rule.factors):
                        if p_fac.is_reduction:
                            pk = self.factor_keys.get((pidx, p_li))
                            if pk is not None:
                                merge_pairs.append((pk, ck))
                                found = True
                            break
                    if not found:
                        all_valid = False
                        break

                if all_valid and merge_pairs:
                    for pk, ck_val in merge_pairs:
                        self.uf.union(pk, ck_val)

    # -----------------------------------------------------------------
    # Step 4 — metadata collection
    # -----------------------------------------------------------------

    def _collect_factor_metadata(self) -> None:
        for node in self.graph.nodes:
            rule = self.factor_rules.get(node)
            if rule is None:
                continue
            nidx = self.node_map[node]
            for li, fac in enumerate(rule.factors):
                gid = self.factor_keys.get((nidx, li))
                if gid is None:
                    continue
                root = self.uf.find(gid)
                self.factor_ops[root].append((node, fac, li))

    def _unique_roots(self) -> set[int]:
        return {self.uf.find(gid) for gid in self.factor_keys.values()}

    # -----------------------------------------------------------------
    # Step 5 — ILP construction
    # -----------------------------------------------------------------

    def _build_ilp(self) -> None:
        roots = self._unique_roots()

        # --- Variables: y[root, mesh_dim] ∈ {0, 1} ---
        for r in roots:
            for m in range(self.mesh.ndim):
                self.y_vars[(r, m)] = pulp.LpVariable(f"y_{r}_m{m}", cat="Binary")

        # --- Constraints ---
        # NOTE: we intentionally omit a factor-uniqueness constraint here.
        # A factor MAY be assigned to multiple mesh dims simultaneously,
        # which corresponds to placements like (Shard(0), Shard(0)) where a
        # single tensor dim is sharded across several mesh dims.  The tensor
        # exclusion constraint already prevents invalid combos (two different
        # factors claiming the same tensor dim on the same mesh dim).
        self._add_tensor_exclusion()

        # --- Objective ---
        self._add_objective(roots)

    # ---- constraints ------------------------------------------------

    def _add_factor_uniqueness(self, roots: set[int]) -> None:
        """Each factor is assigned to *at most one* mesh dimension."""
        for r in roots:
            self.prob += (
                pulp.lpSum(self.y_vars[(r, m)] for m in range(self.mesh.ndim)) <= 1,
                f"fac_uniq_{r}",
            )

    def _add_tensor_exclusion(self) -> None:
        """Per tensor per mesh dim, at most one factor can be sharded.

        This encodes the DTensor invariant: a tensor dimension can only appear
        as ``Shard(d)`` for a single ``d`` on each mesh dimension.

        Important: multiple factors at the same node may share a root (after
        union-find merging, e.g. nheads and head_dim from unflatten both map to
        the hidden input dimension).  We must deduplicate by root to avoid
        counting the same ILP variable twice, which would turn ``sum <= 1``
        into ``2*y <= 1`` and incorrectly force that variable to 0.
        """
        cid = 0
        for node in self.graph.nodes:
            rule = self.factor_rules.get(node)
            if rule is None or not rule.factors:
                continue
            nidx = self.node_map[node]

            # — result tensors (one exclusion set per result) —
            for ri in range(rule.num_results):
                for m in range(self.mesh.ndim):
                    vs = []
                    seen_roots: set[int] = set()
                    for li, fac in enumerate(rule.factors):
                        # Include both spatial and reduction factors: a
                        # tensor can only be Shard(d) OR Partial on each
                        # mesh dim, never both simultaneously.
                        has_spatial = (
                            ri < len(fac.result_dims)
                            and fac.result_dims[ri] is not None
                        )
                        if has_spatial or fac.is_reduction:
                            gid = self.factor_keys.get((nidx, li))
                            if gid is not None:
                                root = self.uf.find(gid)
                                if root not in seen_roots:
                                    seen_roots.add(root)
                                    vs.append(self.y_vars[(root, m)])
                    if len(vs) > 1:
                        self.prob += pulp.lpSum(vs) <= 1, f"tex_r_{cid}"
                        cid += 1

            # — operand tensors —
            for oi in range(rule.num_operands):
                for m in range(self.mesh.ndim):
                    vs = []
                    seen_roots: set[int] = set()
                    for li, fac in enumerate(rule.factors):
                        if (
                            oi < len(fac.operand_dims)
                            and fac.operand_dims[oi] is not None
                        ):
                            gid = self.factor_keys.get((nidx, li))
                            if gid is not None:
                                root = self.uf.find(gid)
                                if root not in seen_roots:
                                    seen_roots.add(root)
                                    vs.append(self.y_vars[(root, m)])
                    if len(vs) > 1:
                        self.prob += pulp.lpSum(vs) <= 1, f"tex_o_{cid}"
                        cid += 1

    # ---- objective --------------------------------------------------

    def _add_objective(self, roots: set[int]) -> None:
        """Build the cost function.

        For each factor *f* assigned to mesh dim *m* the cost coefficient
        includes three components:

        1. **Compute benefit** (all factors): sharding any dimension divides
           work by ``mesh.shape[m]``.
        2. **Redistribution penalty** (reduction factors at "exit" edges):
           when a ``Partial`` output reaches a consumer that doesn't share
           the reduction root, redistribution is needed.  The exact cost
           depends on the consumer's placement on that mesh dim:

           - **Partial → Shard** (reduce-scatter): B·(n-1)/n
           - **Partial → Replicate** (all-reduce): 2B·(n-1)/n

           This is captured exactly via auxiliary continuous variables that
           linearize the product ``y[r,m] · (1 - any_consumer_spatial_on_m)``.
        3. **All-gather penalty** (spatial factors at "exit" edges): when a
           producer is ``Shard(d)`` on mesh dim *m* but a consumer doesn't
           share that factor (via union-find), an all-gather is needed.
           Cost ≈ B·(n-1)/n.

        All three are linear in the ``y`` and ``z`` variables, keeping the
        ILP linear.
        """
        ag_bytes, _rs_bytes = self._compute_redistribution_bytes()
        exit_info = self._compute_reduction_exit_info()
        terms: list[Any] = []

        for r in roots:
            refs = self.factor_ops.get(r, [])
            for m in range(self.mesh.ndim):
                mesh_size = self.mesh.shape[m]
                var = self.y_vars[(r, m)]
                cost = 0.0

                for node, fac, _ in refs:
                    if node.op != "call_function":
                        continue

                    # Compute benefit: work is divided by mesh_size
                    # regardless of whether the factor is a reduction
                    # or spatial dimension.
                    compute = self._compute_cost(node)
                    benefit = compute * (1.0 - 1.0 / mesh_size)
                    cost -= benefit

                # All-gather penalty at spatial exit edges.
                if r in ag_bytes:
                    ag_comm = ag_bytes[r] * (mesh_size - 1) / mesh_size
                    cost += ag_comm / self._BW * 1e6

                if cost != 0.0:
                    terms.append(cost * var)

        # Reduction exit edges: linearized Partial → {Shard, Replicate} cost.
        #
        # For each (reduction_root r, consumer u) exit edge, on mesh dim m:
        #
        #   base cost  = B·(n-1)/n · y[r,m]          (reduce-scatter)
        #   extra cost = B·(n-1)/n · z                (upgrade to all-reduce)
        #
        # where z is a continuous auxiliary variable satisfying:
        #   z ≥ y[r,m] − Σ_s y[s,m]    for consumer's spatial roots s
        #   z ≥ 0                       (implicit from lowBound=0)
        #
        # Since z has a positive coefficient and we minimize, the solver
        # sets z = max(0, y[r,m] − Σ y[s,m]).
        #
        # • Consumer has spatial factor on m (Σ≥1) → z=0, total = B   (reduce-scatter)
        # • Consumer fully replicated on m  (Σ=0)  → z=y,  total = 2B (all-reduce)
        z_id = 0
        for (r, uidx), bytes_val in exit_info.items():
            consumer_spatial = self._get_spatial_roots_at_node(uidx)
            for m in range(self.mesh.ndim):
                mesh_size = self.mesh.shape[m]
                y_r_m = self.y_vars[(r, m)]

                # Base reduce-scatter cost (always incurred when y[r,m]=1).
                comm_unit = bytes_val * (mesh_size - 1) / mesh_size / self._BW * 1e6
                terms.append(comm_unit * y_r_m)

                # Extra cost for Partial → Replicate (linearized).
                valid_roots = [s for s in consumer_spatial if (s, m) in self.y_vars]
                if valid_roots:
                    z = pulp.LpVariable(f"z_pr_{z_id}", lowBound=0)
                    spatial_sum = pulp.lpSum(
                        self.y_vars[(s, m)] for s in valid_roots
                    )
                    self.prob += z >= y_r_m - spatial_sum, f"z_pr_lb_{z_id}"
                    terms.append(comm_unit * z)
                    z_id += 1
                else:
                    # No spatial factors at consumer → always all-reduce.
                    # Extra cost = B·(n-1)/n · y[r,m] (doubling the base).
                    terms.append(comm_unit * y_r_m)

        self._num_z_vars = z_id

        if terms:
            self.prob += pulp.lpSum(terms)

    # ---- cost helpers -----------------------------------------------

    # Rough inter-node bandwidth (bytes/s).  50 GB/s is a reasonable
    # default for NVLink / high-end InfiniBand.
    _BW: float = 50e9

    @staticmethod
    def _output_bytes(node: torch.fx.Node) -> float:
        val = _get_primary_tensor(node.meta.get("val"))
        if val is not None:
            return float(val.numel() * val.element_size())
        return 0.0

    def _compute_cost(self, node: torch.fx.Node) -> float:
        """Estimate unsharded compute cost for a node in microseconds.

        Uses ``estimate_strategy_runtime_cost`` from ``compute_estimation.py``
        which accounts for both FLOP-bound ops (matmul, bmm, SDPA) and
        memory-bound ops (pointwise add, relu, etc.).  View ops return 0.
        Results are cached per node.
        """
        if node in self._cost_cache:
            return self._cost_cache[node]
        try:
            cost = estimate_strategy_runtime_cost(node, None)
        except Exception:
            cost = 0.0
        self._cost_cache[node] = cost
        return cost

    def _compute_redistribution_bytes(
        self,
    ) -> tuple[dict[int, float], dict[int, float]]:
        """For each factor root, total output bytes at "exit" edges.

        Returns ``(ag_bytes, rs_bytes)``:

        * **ag_bytes** — for *spatial* factors: bytes needing an all-gather
          at edges where the consumer doesn't share the root.
        * **rs_bytes** — for *reduction* factors: bytes needing a
          reduce-scatter at edges where the ``Partial`` doesn't propagate
          to the consumer.
        """
        # node_idx → set of factor roots at that node
        node_roots: dict[int, set[int]] = defaultdict(set)
        for (nidx, _li), gid in self.factor_keys.items():
            node_roots[nidx].add(self.uf.find(gid))

        ag_bytes: dict[int, float] = defaultdict(float)
        rs_bytes: dict[int, float] = defaultdict(float)
        for root, refs in self.factor_ops.items():
            for node, fac, _li in refs:
                for user in node.users:
                    if user.op != "call_function":
                        continue
                    uidx = self.node_map.get(user)
                    if uidx is None:
                        continue
                    if root in node_roots.get(uidx, set()):
                        continue  # factor propagates — no redistribution

                    if fac.is_reduction:
                        # Partial exits here → reduce-scatter
                        rs_bytes[root] += self._output_bytes(node)
                    elif any(d is not None for d in fac.result_dims):
                        # Shard exits here → all-gather
                        ag_bytes[root] += self._output_bytes(node)

        return dict(ag_bytes), dict(rs_bytes)

    def _compute_reduction_exit_info(self) -> dict[tuple[int, int], float]:
        """For each (reduction_root, consumer_nidx) pair, total bytes at exits.

        Used by the linearized Partial → Replicate cost model to distinguish
        reduce-scatter (consumer is Shard) from all-reduce (consumer is
        Replicate) on each mesh dimension.
        """
        node_roots: dict[int, set[int]] = defaultdict(set)
        for (nidx, _li), gid in self.factor_keys.items():
            node_roots[nidx].add(self.uf.find(gid))

        exit_info: dict[tuple[int, int], float] = defaultdict(float)
        for root, refs in self.factor_ops.items():
            for node, fac, _li in refs:
                if not fac.is_reduction:
                    continue
                for user in node.users:
                    if user.op != "call_function":
                        continue
                    uidx = self.node_map.get(user)
                    if uidx is None:
                        continue
                    if root in node_roots.get(uidx, set()):
                        continue  # factor propagates — no redistribution
                    exit_info[(root, uidx)] += self._output_bytes(node)

        return dict(exit_info)

    def _get_spatial_roots_at_node(self, nidx: int) -> set[int]:
        """Get unique roots for spatial (non-reduction) result factors at a node."""
        node = self.nodes[nidx]
        rule = self.factor_rules.get(node)
        if rule is None:
            return set()
        roots: set[int] = set()
        for li, fac in enumerate(rule.factors):
            if not fac.is_reduction and any(d is not None for d in fac.result_dims):
                gid = self.factor_keys.get((nidx, li))
                if gid is not None:
                    roots.add(self.uf.find(gid))
        return roots

    # -----------------------------------------------------------------
    # User constraints
    # -----------------------------------------------------------------

    def add_node_constraint(
        self,
        node: torch.fx.Node,
        placement: tuple[Placement, ...],
    ) -> None:
        """Pin a node's output to a specific placement."""
        rule = self.factor_rules.get(node)
        if rule is None:
            return
        nidx = self.node_map[node]

        for m, p in enumerate(placement):
            if isinstance(p, Shard):
                for li, fac in enumerate(rule.factors):
                    if any(d == p.dim for d in fac.result_dims):
                        gid = self.factor_keys.get((nidx, li))
                        if gid is not None:
                            root = self.uf.find(gid)
                            self.prob += (
                                self.y_vars[(root, m)] == 1,
                                f"pin_{nidx}_f{li}_m{m}",
                            )
                        break
            elif isinstance(p, Replicate):
                seen_roots: set[int] = set()
                for li, fac in enumerate(rule.factors):
                    if any(d is not None for d in fac.result_dims):
                        gid = self.factor_keys.get((nidx, li))
                        if gid is not None:
                            root = self.uf.find(gid)
                            if root not in seen_roots:
                                seen_roots.add(root)
                                self.prob += (
                                    self.y_vars[(root, m)] == 0,
                                    f"rep_{nidx}_r{root}_m{m}",
                                )

    def add_input_constraints(
        self, input_placements: list[tuple[Placement, ...] | None] | None = None
    ) -> None:
        """Constrain input placements (and their corresponding gradients).

        Uses ``get_plain_input_and_grad_nodes`` to correctly map inputs to
        their gradient nodes in the joint fwd+bwd graph, matching the
        original :class:`ShardingOptimizer` behaviour.
        """
        mut_ips = None
        if input_placements is not None:
            mut_ips = {i: p for i, p in enumerate(input_placements)}

        for desc, (node, grad_node) in get_plain_input_and_grad_nodes(
            self.graph
        ).items():
            if input_placements is None:
                placement = None
            else:
                assert isinstance(desc, PlainAOTInput)
                assert mut_ips is not None
                placement = mut_ips.pop(desc.idx, None)

            if placement is not None:
                self.add_node_constraint(node, tuple(placement))
                if grad_node is not None:
                    self.add_node_constraint(grad_node, tuple(placement))

    def add_output_constraints(
        self, output_placements: list[tuple[Placement, ...] | None] | None = None
    ) -> None:
        """Constrain output placements (and their corresponding tangents).

        Uses ``get_plain_output_and_tangent_nodes`` to correctly map outputs to
        their tangent nodes in the joint fwd+bwd graph, matching the
        original :class:`ShardingOptimizer` behaviour.
        """
        mut_ops = None
        if output_placements is not None:
            mut_ops = {i: p for i, p in enumerate(output_placements)}

        for desc, (node, tangent_node) in get_plain_output_and_tangent_nodes(
            self.graph
        ).items():
            if output_placements is None:
                placement = None
            else:
                assert isinstance(desc, PlainAOTOutput)
                assert mut_ops is not None
                placement = mut_ops.pop(desc.idx, None)

            if placement is not None:
                self.add_node_constraint(node, tuple(placement))
                if tangent_node is not None:
                    self.add_node_constraint(tangent_node, tuple(placement))

    # -----------------------------------------------------------------
    # Solve
    # -----------------------------------------------------------------

    def get_solution(self, verbose: bool = False) -> dict[torch.fx.Node, OpSpec]:
        """Solve the factor ILP and reconstruct per-node OpSpecs."""
        solver = pulp.PULP_CBC_CMD(msg=verbose)
        self.prob.solve(solver)

        if self.prob.status == -1:
            diag = self._infeasibility_diagnostics()
            raise RuntimeError(
                "Factor-based ILP is infeasible.  "
                "Check that input / output constraints are satisfiable.\n" + diag
            )

        # Extract factor → mesh-dim assignments.
        assignment: dict[int, int] = {}  # root → mesh_dim
        for (root, m), var in self.y_vars.items():
            if var.varValue is not None and var.varValue > 0.5:
                assignment[root] = m

        # Reconstruct per-node placements.
        result: dict[torch.fx.Node, OpSpec] = {}
        for node in self.graph.nodes:
            if node.op == "output":
                continue
            rule = self.factor_rules.get(node)
            if rule is None:
                continue
            nidx = self.node_map[node]

            # --- Build output_specs ---
            output_specs = None
            val = node.meta.get("val")

            if val is not None and isinstance(val, torch.Tensor):
                placements: list[Placement] = [Replicate()] * self.mesh.ndim
                for li, fac in enumerate(rule.factors):
                    gid = self.factor_keys.get((nidx, li))
                    if gid is None:
                        continue
                    root = self.uf.find(gid)
                    m = assignment.get(root)
                    if m is None:
                        continue
                    if fac.is_reduction:
                        placements[m] = Partial()
                    else:
                        if fac.result_dims and fac.result_dims[0] is not None:
                            placements[m] = Shard(fac.result_dims[0])
                tensor_meta = TensorMeta(val.shape, val.stride(), val.dtype)
                output_specs = DTensorSpec(
                    self.mesh, tuple(placements), tensor_meta=tensor_meta
                )
            elif val is not None and isinstance(val, (tuple, list)):
                # Multi-output op (e.g. SDPA).  Build per-output placements
                # using the corresponding result_dims index for each output.
                specs = []
                for ri, v in enumerate(val):
                    if isinstance(v, torch.Tensor):
                        plc_list = [Replicate()] * self.mesh.ndim
                        for li, fac in enumerate(rule.factors):
                            gid = self.factor_keys.get((nidx, li))
                            if gid is None:
                                continue
                            root = self.uf.find(gid)
                            m_assigned = assignment.get(root)
                            if m_assigned is None:
                                continue
                            if fac.is_reduction:
                                plc_list[m_assigned] = Partial()
                            elif (
                                ri < len(fac.result_dims)
                                and fac.result_dims[ri] is not None
                            ):
                                plc_list[m_assigned] = Shard(fac.result_dims[ri])
                        tm = TensorMeta(v.shape, v.stride(), v.dtype)
                        specs.append(
                            DTensorSpec(self.mesh, tuple(plc_list), tensor_meta=tm)
                        )
                    else:
                        specs.append(None)
                output_specs = tuple(specs)

            if output_specs is None:
                continue

            # --- Build input_specs ---
            if node.op in ("placeholder", "get_attr"):
                # Convention: placeholders use output_specs as input_specs.
                if isinstance(output_specs, DTensorSpec):
                    input_specs = [output_specs]
                else:
                    input_specs = None
            else:
                input_specs = self._build_input_specs(
                    node, rule, nidx, assignment
                ) or None

            result[node] = OpSpec(
                output_specs=output_specs, input_specs=input_specs
            )

        return result

    def _build_input_specs(
        self,
        node: torch.fx.Node,
        rule: FactorRule,
        nidx: int,
        assignment: dict[int, int],
    ) -> list[DTensorSpec]:
        """Reconstruct input DTensorSpecs from factor assignments.

        For each operand, the placement on each mesh dim is derived from the
        factor assigned to that mesh dim:

        - ``operand_dims[oi]`` is not None → ``Shard(dim)``
        - ``operand_dims[oi]`` is None and ``is_reduction`` → ``Partial``
        - otherwise → ``Replicate``
        """
        if rule.num_operands == 0:
            return []

        # Tensor input nodes in tree_flatten order (matches operand indexing).
        flat_args, _ = tree_flatten(node.args)
        tensor_args = [a for a in flat_args if isinstance(a, torch.fx.Node)]

        input_specs: list[DTensorSpec] = []
        for oi in range(rule.num_operands):
            inp_placements: list[Placement] = [Replicate()] * self.mesh.ndim
            for li, fac in enumerate(rule.factors):
                gid = self.factor_keys.get((nidx, li))
                if gid is None:
                    continue
                root = self.uf.find(gid)
                m_assigned = assignment.get(root)
                if m_assigned is None:
                    continue
                if oi < len(fac.operand_dims):
                    od = fac.operand_dims[oi]
                    if od is not None:
                        inp_placements[m_assigned] = Shard(od)
                    elif fac.is_reduction:
                        inp_placements[m_assigned] = Partial()

            # Get TensorMeta from the corresponding input node.
            inp_tm = None
            if oi < len(tensor_args):
                arg_val = tensor_args[oi].meta.get("val")
                if isinstance(arg_val, torch.Tensor):
                    inp_tm = TensorMeta(
                        arg_val.shape, arg_val.stride(), arg_val.dtype
                    )
                elif isinstance(arg_val, (tuple, list)):
                    # Multi-output producer (e.g. getitem consuming SDPA).
                    # Use the getitem index to find the correct tensor.
                    if (
                        node.target is operator.getitem
                        and oi == 0
                        and len(node.args) > 1
                        and isinstance(node.args[1], int)
                    ):
                        idx = node.args[1]
                        if idx < len(arg_val) and isinstance(
                            arg_val[idx], torch.Tensor
                        ):
                            v = arg_val[idx]
                            inp_tm = TensorMeta(v.shape, v.stride(), v.dtype)

            input_specs.append(
                DTensorSpec(
                    self.mesh, tuple(inp_placements), tensor_meta=inp_tm
                )
            )

        return input_specs

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    def _infeasibility_diagnostics(self) -> str:
        """Build a diagnostic string to help debug infeasible ILPs.

        Scans all equality constraints to detect variables pinned to both 0 and
        1 (the most common cause of infeasibility in the factor ILP).
        """
        # Collect per-variable equality constraints.
        pinned_to_1: dict[str, list[str]] = defaultdict(list)
        pinned_to_0: dict[str, list[str]] = defaultdict(list)
        for name, c in self.prob.constraints.items():
            # An equality constraint y == v has sense EQ (0) and constant = -v
            if c.sense == 0 and len(c) == 1:
                # single-variable equality
                for var, coeff in c.items():
                    val = -c.constant / coeff
                    if abs(val - 1.0) < 1e-9:
                        pinned_to_1[var.name].append(name)
                    elif abs(val) < 1e-9:
                        pinned_to_0[var.name].append(name)

        conflicts = []
        for var_name in set(pinned_to_1) & set(pinned_to_0):
            conflicts.append(
                f"  Variable {var_name}:\n"
                f"    pinned to 1 by: {pinned_to_1[var_name]}\n"
                f"    pinned to 0 by: {pinned_to_0[var_name]}"
            )

        if conflicts:
            return "Conflicting constraints found:\n" + "\n".join(conflicts)
        return (
            "No direct 0-vs-1 conflicts found; infeasibility may be caused "
            "by interacting inequality constraints (tensor exclusion, factor "
            "uniqueness)."
        )

    def get_stats(self) -> dict[str, Any]:
        """Return ILP size statistics (useful for comparing with original)."""
        roots = self._unique_roots()

        # Estimate original variable count.
        orig_vars = 0
        for node, strat in self.strats.items():
            if not strat.strategies:
                continue
            n_out = len(strat.strategies)
            first = strat.strategies[0]
            n_args = len(first.input_specs) if first.input_specs else 0
            orig_vars += max(n_args, 1) * n_out * n_out

        n_factor_vars = len(self.y_vars)
        n_aux_vars = getattr(self, "_num_z_vars", 0)
        return {
            "num_graph_nodes": len(self.nodes),
            "num_unique_factors": len(roots),
            "num_factor_ilp_variables": n_factor_vars + n_aux_vars,
            "num_factor_y_variables": n_factor_vars,
            "num_factor_z_variables": n_aux_vars,
            "num_factor_ilp_constraints": len(self.prob.constraints),
            "mesh_shape": tuple(self.mesh.shape),
            "estimated_original_ilp_variables": orig_vars,
            "variable_reduction_ratio": orig_vars / max(n_factor_vars, 1),
        }

    def get_log(self, verbose: bool = False) -> str:
        """Human-readable summary."""
        lines: list[str] = []
        lines.append(f"Factor ILP status: {pulp.LpStatus[self.prob.status]}")
        s = self.get_stats()
        lines.append(f"Unique factors:           {s['num_unique_factors']}")
        lines.append(f"Factor ILP variables:     {s['num_factor_ilp_variables']} ({s['num_factor_y_variables']} y + {s['num_factor_z_variables']} z)")
        lines.append(f"Factor ILP constraints:   {s['num_factor_ilp_constraints']}")
        lines.append(
            f"Est. original ILP vars:   {s['estimated_original_ilp_variables']}"
        )
        lines.append(f"Variable reduction:       {s['variable_reduction_ratio']:.1f}x")

        if verbose and self.prob.status == 1:
            lines.append("")
            lines.append("Factor assignments:")
            for (root, m), var in sorted(self.y_vars.items()):
                if var.varValue is not None and var.varValue > 0.5:
                    refs = self.factor_ops.get(root, [])
                    desc = ""
                    if refs:
                        _, fac, _ = refs[0]
                        kind = "reduction" if fac.is_reduction else "spatial"
                        desc = f" ({kind}, size={fac.size})"
                    lines.append(f"  Factor {root} → mesh dim {m}{desc}")

        return "\n".join(lines)
