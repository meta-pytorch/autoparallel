# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Approximate sharding solver.

The ILP in :mod:`optimize_sharding` selects, for every operation, an output
placement and (per argument) the input placement of its producer. The flow
constraint forces a consumer's input placement to equal its producer's chosen
output placement, so the only genuinely free variables are the per-node output
strategy indices ``x_v``. The problem therefore reduces to a pairwise discrete
energy minimization over a DAG::

    E(x) = Σ_v U_v(x_v) + Σ_{(u,v)} B_{uv}(x_u, x_v)

where ``U_v`` is the compute cost and ``B_{uv}`` is the communication +
sharding-transition cost on the edge from producer ``u`` to consumer ``v``.

This is a pairwise MRF. The autograd DAG has small in-degree (<3) but large
out-degree (tens) and a wide topological frontier (hundreds), so exact
frontier/junction-tree DP blows up. We instead solve it with **min-sum belief
propagation** (max-product in min-sum form) on the graph of *coupled groups*,
which propagates coordinated decisions globally, then polish with group-level
coordinate descent and a star-block local search.

Nodes that must be chosen jointly are merged into groups: repeated-subgraph
cluster copies share a strategy index, and forward/backward pairs share an
output placement. The solver reuses the strategies, decision variables and
constraints already built by ``ShardingOptimizer`` (it replaces only the
CBC/ILP *solve*, not problem construction) and writes its assignment back into
the PuLP variables, so the result is scored with the exact same objective as the
ILP (``pulp.value(prob.objective)``).
"""

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pulp
import torch
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor.placement_types import Replicate, Shard

from .cost_models.compute_estimation import _get_sharded_shape_stride

logger = logging.getLogger(__name__)

INF = float("inf")
BIG = 1e12  # finite stand-in for forbidden combinations (avoids NaN in min-sum)

# Paired forward/backward constraints couple two nodes to the *same output
# placement* (the strategy index may differ between the two strategy lists).
_PAIRED_PREFIXES = (
    "grad_param_constraint",
    "grad_input_constraint",
    "grad_output_constraint",
)


@dataclass
class ApproximateSolveResult:
    objective: float
    status: str
    build_s: float
    solve_s: float
    total_s: float
    num_groups: int
    num_nodes: int


@dataclass
class _Group:
    """A set of node indices chosen jointly (cluster copies share a strategy
    index; forward/backward pairs share an output placement)."""

    members: list[int]
    cost_bearing: list[int] = field(default_factory=list)
    choices: list[dict[int, int]] = field(default_factory=list)  # member -> out_idx
    current: int = 0

    @property
    def domain(self) -> int:
        return len(self.choices)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


class ApproximateShardingSolver:
    """Approximate solver for the sharding placement problem on an already-built
    :class:`ShardingOptimizer`.

    Call :meth:`get_solution` for a ``{node: OpSpec}`` dict (same format as
    ``ShardingOptimizer.get_solution``); it also fills the PuLP variables and
    ``optimizer.selected_keys`` so the assignment can be scored/inspected exactly
    like an ILP solution.
    """

    def __init__(
        self,
        optimizer,
        candidate_limit: Optional[int] = 64,
        bp_iters: int = 400,
        bp_tol: float = 1e-3,
        max_sweeps: int = 12,
        max_time_s: float = 60.0,
        star_passes: int = 2,
        max_star_children: int = 32,
        group_domain_limit: int = 512,
    ):
        self.opt = optimizer
        self.candidate_limit = candidate_limit
        self.bp_iters = bp_iters
        self.bp_tol = bp_tol
        self.max_sweeps = max_sweeps
        self.max_time_s = max_time_s
        self.star_passes = star_passes
        self.max_star_children = max_star_children
        self.group_domain_limit = group_domain_limit

        # Populated by _build_problem().
        self.cost_bearing: list[int] = []
        self.node_mult: dict[int, int] = {}
        self.forbidden: set[tuple] = set()
        self.allowed_out: dict[int, list[int]] = {}
        self.groups: list[_Group] = []
        self.node_to_group: dict[int, int] = {}
        self.input_edges: dict[int, list[tuple[int, int]]] = {}
        self._arg_prod: dict[int, dict[int, int]] = {}
        self.consumers: dict[int, list[tuple[int, int]]] = defaultdict(list)
        self.cur_out: dict[int, int] = {}
        self._memory: Optional[dict[str, Any]] = None

        # Populated by _build_factors().
        self.g_unary: list[np.ndarray] = []
        self.C: dict[tuple, np.ndarray] = {}
        self.nbrs: list[list[int]] = []

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def get_solution(self, verbose: bool = False):
        result, solution = self._solve(verbose=verbose)
        self.result = result
        return solution

    def _solve(self, verbose: bool = False):
        opt = self.opt
        if getattr(opt, "solver_backend", "ilp") != "ilp":
            raise RuntimeError(
                "ApproximateShardingSolver requires an ILP-built optimizer "
                "(decision_vars / pulp_variables / constraints)."
            )
        t0 = time.perf_counter()
        self._build_problem()
        t_bp = time.perf_counter()
        self._build_factors()
        t_bf = time.perf_counter()
        t_build = t_bf - t0
        if verbose:
            logger.info(
                "approx build: problem=%.2fs %s factors=%.2fs groups=%d "
                "cost_bearing=%d edges=%d max_domain=%d",
                t_bp - t0, getattr(self, "_build_times", {}), t_bf - t_bp,
                len(self.groups), len(self.cost_bearing),
                sum(len(v) for v in self.input_edges.values()),
                max((g.domain for g in self.groups), default=0),
            )

        deadline = t0 + self.max_time_s
        # TRW-S init, then local-search polish. TRW-S reaches the exact MAP on the
        # (integral) sharding problem, so the old greedy second candidate it used
        # to be compared against is strictly dominated and has been dropped; the
        # polish remains for the memory budget and as a local-search safety net.
        t_bp0 = time.perf_counter()
        self._belief_propagation(deadline)
        if verbose:
            logger.info("approx phase: trws iter=%s delta=%.4g in %.2fs; "
                        "decode energy=%.1f",
                        getattr(self, "_bp_last_iter", None),
                        getattr(self, "_bp_last_delta", float("nan")),
                        time.perf_counter() - t_bp0,
                        self._fast_total_energy())
        self._memory_repair()
        self._coordinate_descent(deadline)
        if verbose:
            logger.info("approx phase: trws+cd energy=%.1f", self._fast_total_energy())
        self._star_block_search(deadline)
        bp_energy = self._fast_total_energy()
        if verbose:
            logger.info("approx phase: trws+cd+star energy=%.1f", bp_energy)
        t_solve = time.perf_counter() - t0 - t_build

        objective = self._write_back()
        total_s = time.perf_counter() - t0
        infeasible = not math.isfinite(objective)
        status = "Infeasible" if infeasible else "Heuristic"
        result = ApproximateSolveResult(
            objective=objective,
            status=status,
            build_s=t_build,
            solve_s=t_solve,
            total_s=total_s,
            num_groups=len(self.groups),
            num_nodes=len(self.cost_bearing),
        )
        logger.info(
            "ApproximateShardingSolver: status=%s objective=%.4f "
            "(trws+polish=%.1f) groups=%d nodes=%d "
            "timings={build=%.3fs,solve=%.3fs,total=%.3fs}",
            status,
            objective,
            bp_energy,
            len(self.groups),
            len(self.cost_bearing),
            t_build,
            t_solve,
            total_s,
        )
        opt.profile["approximate"] = {
            "objective": objective,
            "status": status,
            "build_s": t_build,
            "solve_s": t_solve,
            "total_s": total_s,
            "groups": len(self.groups),
            "bp_energy": bp_energy,
        }
        if infeasible:
            raise RuntimeError(
                "ApproximateShardingSolver could not find a feasible assignment. "
                "User constraints may be contradictory or the mesh too small."
            )
        solution = opt._to_orig_solution(opt._extract_and_validate_solution())
        return result, solution

    # ------------------------------------------------------------------ #
    # Problem construction
    # ------------------------------------------------------------------ #
    def _build_problem(self):
        opt = self.opt
        # cluster_links is node-level: copy node idx -> root node idx.
        cluster_linked = set(opt.cluster_links)
        self.cost_bearing = [
            opt.node_map[node]
            for node in opt.strats
            if node.op != "output" and opt.node_map[node] not in cluster_linked
        ]

        root_to_copies: dict[int, set] = defaultdict(set)
        for copy_idx, root_idx in opt.cluster_links.items():
            root_to_copies[root_idx].add(copy_idx)
        self.node_mult = {
            v: 1 + len(root_to_copies.get(v, ())) for v in self.cost_bearing
        }

        self.allowed_out = {}
        for node, strat in opt.strats.items():
            if node.op == "output":
                continue
            self.allowed_out[opt.node_map[node]] = list(range(len(strat.strategies)))

        t = time.perf_counter()
        if opt.prob is None:
            # Lite build: no PuLP problem was constructed, derive topology directly.
            paired_edges, authoritative = self._topology_direct()
        else:
            paired_edges, authoritative = self._parse_constraints()
        # Flow edges are taken from the ILP's output_input_consistent constraints
        # (the authoritative producer per consumer-arg), NOT from _all_input_nodes:
        # the two disagree for some ops (einsum list-args, alias/backward nodes),
        # and trusting _all_input_nodes yields flow-infeasible assignments. The
        # producer here is the (possibly cluster-resolved) node carrying the
        # producer's pulp variable; the ILP guarantees its out_idx range matches
        # the consumer's inp_idx range for that arg.
        self._arg_prod: dict[int, dict[int, int]] = defaultdict(dict)
        flow_couplings = []  # producer sets forced to share an out_idx
        for (c_idx, argi), producers in authoritative.items():
            rep = min(producers)  # all coupled -> same out, any representative
            self._arg_prod[c_idx][argi] = rep
            if len(producers) > 1:
                flow_couplings.append(producers)
        self.input_edges = {}
        self.consumers = defaultdict(list)
        for v in self.cost_bearing:
            edges = sorted(self._arg_prod.get(v, {}).items())
            self.input_edges[v] = edges
            for argi, p in edges:
                self.consumers[p].append((v, argi))
        t_parse = time.perf_counter()

        # Remove fully-forbidden out_idx for cost-bearing nodes.
        for v in self.cost_bearing:
            node = opt.nodes[v]
            self.allowed_out[v] = [
                o for o in self.allowed_out[v] if not self._out_fully_forbidden(v, node, o)
            ]
        t_forbid = time.perf_counter()

        self._build_memory_info()  # also pins params when the budget is tight
        t_mem = time.perf_counter()
        self._build_groups(paired_edges, flow_couplings)
        t_groups = time.perf_counter()
        self._prune_candidates()
        self._build_times = {
            "parse": t_parse - t,
            "forbid": t_forbid - t_parse,
            "memory": t_mem - t_forbid,
            "groups": t_groups - t_mem,
            "prune": time.perf_counter() - t_groups,
        }

    # Constraint families that never restrict the per-node out_idx domain and
    # are handled structurally (flow/uniqueness) or via the cost sentinel below.
    # Skipping them by name avoids materializing items() for the ~majority of the
    # (often >100k) constraints.
    _SKIP_PREFIXES = (
        "unique_decision",
        "same_across_args",
        "inf_cases",
        "memory_constraint",
    )

    def _parse_constraints(self):
        opt = self.opt
        # inf-cost keys are forced to 0 by add_inf_cost_constraint, which also
        # stamps dv.cost = 10000.0. Detect them directly instead of parsing the
        # (very numerous) inf_cases constraints.
        for key, dv in opt.decision_vars.items():
            if dv.cost == 10000.0:
                self.forbidden.add(key)

        var_to_key = {var: key for key, var in opt.pulp_variables.items()}
        restrict: dict[int, set] = {}
        paired_edges: list[tuple[int, int, frozenset]] = []
        # (consumer_idx, argi) -> set of producer_idx, from flow constraints. A
        # clustered consumer's single inp variable is shared across all its
        # copies, so the ILP couples one producer per copy (resolved to its root)
        # to that inp, forcing them all equal; we collect the whole set.
        authoritative: dict[tuple[int, int], set] = {}
        for name, c in opt.prob.constraints.items():
            if name.startswith("output_input_consistent"):
                # +side = producer (grouped by out), -side = consumer (grouped by
                # inp at a fixed arg). One +var and one -var pin down the edge.
                pos_key = neg_key = None
                for var, coeff in c.items():
                    k = var_to_key.get(var)
                    if k is None:
                        continue
                    if coeff > 0:
                        pos_key = pos_key or k
                    else:
                        neg_key = neg_key or k
                    if pos_key is not None and neg_key is not None:
                        break
                if pos_key is not None and neg_key is not None:
                    authoritative.setdefault(
                        (neg_key[0], neg_key[1]), set()
                    ).add(pos_key[0])
                continue
            if name.startswith(self._SKIP_PREFIXES):
                continue
            items = list(c.items())
            if not items:
                continue
            rhs = -c.constant
            coeffs = [coeff for _, coeff in items]
            keys = [var_to_key.get(var) for var, _ in items]
            if any(k is None for k in keys):
                continue
            all_pos = all(coeff > 0 for coeff in coeffs)
            if c.sense == pulp.LpConstraintEQ and rhs == 0 and all_pos:
                self.forbidden.update(keys)  # Σ vars == 0  (inf / dtype / disable)
            elif c.sense == pulp.LpConstraintEQ and rhs == 1 and all_pos:
                nodes = {k[0] for k in keys}
                if len(nodes) == 1:
                    n = next(iter(nodes))
                    out_set = {k[2] for k in keys}
                    restrict[n] = restrict.get(n, out_set) & out_set
            elif (
                c.sense == pulp.LpConstraintEQ
                and rhs == 0
                and any(name.startswith(p) for p in _PAIRED_PREFIXES)
                and "disable" not in name
            ):
                pos = {k for k, coeff in zip(keys, coeffs) if coeff > 0}
                neg = {k for k, coeff in zip(keys, coeffs) if coeff < 0}
                na, nb = {k[0] for k in neg}, {k[0] for k in pos}
                oa, ob = {k[2] for k in neg}, {k[2] for k in pos}
                if len(na) == 1 and len(nb) == 1 and len(oa) == 1 and len(ob) == 1:
                    paired_edges.append(
                        (next(iter(na)), next(iter(nb)),
                         frozenset({(next(iter(oa)), next(iter(ob)))}))
                    )
        # method="fix" axis pins leave no PuLP row to parse above, so replay the
        # log to recover them (constraint-method pins are also picked up here,
        # idempotently with their == 1 rows).
        for n, out_set in self._axis_restrict_from_log().items():
            restrict[n] = restrict.get(n, out_set) & out_set
        for n, out_set in restrict.items():
            if n in self.allowed_out:
                self.allowed_out[n] = [o for o in self.allowed_out[n] if o in out_set]
        return paired_edges, authoritative

    def _topology_direct(self):
        """Compute the same topology (forbidden / out_idx restrictions / paired
        edges / flow producers) that _parse_constraints extracts, but directly
        from the graph + cluster_links + _constraint_log, WITHOUT a PuLP problem.
        This lets the optimizer skip building millions of PuLP variables and
        constraints when only the approximate solver is used.

        Mirrors ShardingOptimizer.add_inf_cost_constraint /
        add_grad_reduce_dtype_constraints / add_forward_backward_consistency_constraints /
        _add_paired_output_constraint / add_node_constraint /
        add_output_input_consistent_constraint. Verified byte-identical to
        _parse_constraints on a full build (see tests)."""
        from torch._functorch._aot_autograd.fx_utils import (
            get_param_and_grad_nodes,
            get_plain_input_and_grad_nodes,
            get_plain_output_and_tangent_nodes,
        )

        opt = self.opt
        cl = opt.cluster_links  # node-level: copy node idx -> root node idx

        def rootkey(k):
            return opt._cluster_root_key(k)

        cluster_linked = set(cl)
        node_root = dict(cl)

        def nroot(idx):
            return node_root.get(idx, idx)

        # 1. inf-cost forbidden (== add_inf_cost_constraint).
        for key, dv in opt.decision_vars.items():
            if not math.isfinite(dv.cost) or dv.cost == 10000.0:
                self.forbidden.add(key)

        # 2a. forward param-dtype forbidden (== add_grad_reduce_dtype_constraints
        #     forward part, unconditional). Force the FSDP allgather to run after
        #     a downcasting param dtype_cast (in the smaller param_dtype) by
        #     forbidding any pre-cast redistribution.
        cast_op = torch.ops.autoparallel.dtype_cast.default
        fwd_pre_cast: set[int] = set()
        for param, _grad in get_param_and_grad_nodes(opt.graph).values():
            n = param
            while True:
                if n.target == cast_op:
                    break
                users = list(n.users.keys())
                if len(users) != 1:
                    break
                child = users[0]
                if len(child.all_input_nodes) != 1:
                    break
                n = child
            if n.target != cast_op:
                continue
            if n.meta["val"].dtype.itemsize >= param.meta["val"].dtype.itemsize:
                continue  # only constrain downcasts
            node = n
            while node != param:
                if node in opt.node_map:
                    fwd_pre_cast.add(opt.node_map[node])
                node = node.all_input_nodes[0]
        for key, dv in opt.decision_vars.items():
            if key[0] in fwd_pre_cast and dv.comm_cost > 0:
                self.forbidden.add(key)

        # 2. grad-reduce-dtype (backward) forbidden
        #    (== add_grad_reduce_dtype_constraints backward part).
        if getattr(opt, "force_grad_reduce_in_higher_precision", False):
            cast_op = torch.ops.autoparallel.dtype_cast.default
            pre_cast: set[int] = set()
            for param, grad in get_param_and_grad_nodes(opt.graph).values():
                if grad is None:
                    continue
                chain = [grad]
                n = grad
                while len(n.all_input_nodes) == 1:
                    parent = n.all_input_nodes[0]
                    if len(parent.all_input_nodes) != 1:
                        break
                    chain.append(parent)
                    n = parent
                cast_idx = next(
                    (i for i, nd in enumerate(chain) if nd.target == cast_op), None
                )
                if cast_idx is None:
                    continue
                for nd in chain[cast_idx:]:
                    if nd in opt.node_map:
                        pre_cast.add(opt.node_map[nd])
            for key, dv in opt.decision_vars.items():
                if key[0] in pre_cast and dv.comm_cost > 0:
                    self.forbidden.add(key)

        # 3. forward/backward paired output constraints + disables
        #    (== add_forward_backward_consistency_constraints / _add_paired_output_constraint).
        paired_edges: list[tuple[int, int, frozenset]] = []

        def add_paired(node_a, node_b):
            idx_a, idx_b = opt.node_map[node_a], opt.node_map[node_b]
            strat_a = [str(s.output_specs) for s in opt.strats[node_a].strategies]
            strat_b = [str(s.output_specs) for s in opt.strats[node_b].strategies]
            num_inp_a = len(opt.strats[node_a].strategies[0].redistribute_cost[0])
            for out_idx, sp in enumerate(strat_a):
                if sp not in strat_b:
                    for inp in range(num_inp_a):
                        self.forbidden.add(rootkey((idx_a, 0, out_idx, inp)))
                    continue
                out_idx_b = strat_b.index(sp)
                ra = rootkey((idx_a, 0, out_idx, 0))[0]
                rb = rootkey((idx_b, 0, out_idx_b, 0))[0]
                paired_edges.append((ra, rb, frozenset({(out_idx, out_idx_b)})))

        for param, grad in get_param_and_grad_nodes(opt.graph).values():
            if grad is not None:
                add_paired(param, grad)
        for node, gnode in get_plain_input_and_grad_nodes(opt.graph).values():
            if gnode is not None:
                add_paired(node, gnode)
        for node, tnode in get_plain_output_and_tangent_nodes(opt.graph).values():
            if tnode is not None:
                add_paired(node, tnode)

        # 4. user node/input/output placement restrictions (== add_node_constraint),
        #    replayed from _constraint_log.
        restrict: dict[int, set] = {}
        for fname, kwargs in getattr(opt, "_constraint_log", []):
            if fname != "add_node_constraint":
                continue
            node = next(
                (nd for nd in opt.nodes if nd.name == kwargs["node_name"]), None
            )
            if node is None or node not in opt.strats:
                continue
            placement = kwargs["placement"]
            if placement is None:
                placement = (Shard(0),) + (Replicate(),) * (opt.mesh.ndim - 1)
            out_set = set()
            for i, s in enumerate(opt.strats[node].strategies):
                specs = s.output_specs
                if isinstance(specs, DTensorSpec):
                    if specs.placements == placement:
                        out_set.add(i)
                elif isinstance(specs, (list, tuple)):
                    for spec in specs:
                        if isinstance(spec, DTensorSpec):
                            if spec.placements == placement:
                                out_set.add(i)
                            break
            r = nroot(opt.node_map[node])
            restrict[r] = restrict.get(r, out_set) & out_set
        # 4b. per-axis placement restrictions (== add_node_axis_constraint), what
        #     sharding propagation emits. With method="fix" these leave no PuLP
        #     row to parse, so replaying the log is the only way the approx solver
        #     sees the pin.
        for r, out_set in self._axis_restrict_from_log().items():
            restrict[r] = restrict.get(r, out_set) & out_set
        for n_idx, out_set in restrict.items():
            if n_idx in self.allowed_out:
                self.allowed_out[n_idx] = [
                    o for o in self.allowed_out[n_idx] if o in out_set
                ]

        # 5. flow producers (== add_output_input_consistent_constraint): for each
        #    consumer-arg, the set of (cluster-resolved) producers feeding it.
        authoritative: dict[tuple[int, int], set] = {}
        for node in opt.graph.nodes:
            if node.op == "output" or node not in opt.node_map:
                continue
            p_idx = opt.node_map[node]
            p_linked = p_idx in cluster_linked
            p_root = nroot(p_idx)
            for user in node.users:
                if user.op == "output" or user not in opt.node_map:
                    continue
                u_idx = opt.node_map[user]
                if p_linked and u_idx in cluster_linked:
                    continue
                ain = opt._all_input_nodes(user)
                argi = next((i for i, x in enumerate(ain) if x is node), None)
                if argi is None:
                    continue
                ispecs = opt.strats[user].strategies[0].input_specs
                if argi < len(ispecs) and ispecs[argi] is None:
                    continue
                authoritative.setdefault((nroot(u_idx), argi), set()).add(p_root)

        return paired_edges, authoritative

    def _axis_restrict_from_log(self):
        """out_idx restrictions implied by add_node_axis_constraint calls,
        replayed from _constraint_log → {root_node_idx: set(out_idx)}.

        This is how the approximate solver honors propagated per-axis pins: keep
        only the strategies whose output placement matches the pinned axis,
        exactly like ShardingOptimizer.add_node_axis_constraint. It works whether
        the pin was applied as a PuLP row ("constraint") or as variable bounds
        ("fix", which leaves no row to parse) and in the lite (no-PuLP) build."""
        opt = self.opt
        node_root = dict(opt.cluster_links)  # node-level: copy idx -> root idx
        restrict: dict[int, set] = {}
        for fname, kwargs in getattr(opt, "_constraint_log", []):
            if fname != "add_node_axis_constraint":
                continue
            node = next(
                (nd for nd in opt.nodes if nd.name == kwargs["node_name"]), None
            )
            if node is None or node not in opt.strats:
                continue
            mesh_dim, placement = kwargs["mesh_dim"], kwargs["placement"]
            out_set = set()
            for i, s in enumerate(opt.strats[node].strategies):
                specs = s.output_specs
                if isinstance(specs, DTensorSpec):
                    spec = specs
                elif isinstance(specs, (list, tuple)):
                    spec = next((x for x in specs if isinstance(x, DTensorSpec)), None)
                else:
                    spec = None
                if spec is not None and spec.placements[mesh_dim] == placement:
                    out_set.add(i)
            r = node_root.get(opt.node_map[node], opt.node_map[node])
            restrict[r] = restrict.get(r, out_set) & out_set
        return restrict

    def _is_forbidden(self, key) -> bool:
        """A strategy edge is forbidden if a constraint ruled it out OR it was
        pruned for infinite cost. Pruning removes such keys from decision_vars
        entirely (see ShardingOptimizer._build_decision_vars), so a key missing
        from decision_vars is just as forbidden as one in ``self.forbidden``."""
        return key in self.forbidden or key not in self.opt.decision_vars

    def _surviving_dv(self, v, argi, o):
        """A DecisionVar for (v, argi, o, *) using any inp_idx that survived
        pruning, or None if every edge for that (arg, out) was pruned.
        compute_cost / input_spec are identical across inp_idx for a fixed out."""
        strat = self.opt.strats[self.opt.nodes[v]].strategies[o]
        n_inp = (
            len(strat.redistribute_cost[argi])
            if argi < len(strat.redistribute_cost)
            else 1
        )
        for inp in range(n_inp):
            dv = self.opt.decision_vars.get((v, argi, o, inp))
            if dv is not None:
                return dv
        return None

    def _out_fully_forbidden(self, v, node, o):
        strat = self.opt.strats[node].strategies[o]
        for argi, costs in enumerate(strat.redistribute_cost):
            if all(self._is_forbidden((v, argi, o, inp)) for inp in range(len(costs))):
                return True
        return False

    def _build_groups(self, paired_edges, flow_couplings):
        opt = self.opt
        n = len(opt.nodes)
        uf = _UnionFind(n)
        # cluster_links is node-level: (copy node idx, root node idx) pairs.
        cluster_pairs = set(opt.cluster_links.items())
        for li, ri in cluster_pairs:
            uf.union(li, ri)
        for a, b, _ in paired_edges:
            uf.union(a, b)

        allow: dict[tuple, dict[int, set]] = defaultdict(lambda: defaultdict(set))
        adj: dict[int, set] = defaultdict(set)
        for li, ri in cluster_pairs:
            for o in self.allowed_out.get(ri, []):
                allow[(ri, li)][o].add(o)
            for o in self.allowed_out.get(li, []):
                allow[(li, ri)][o].add(o)
            adj[li].add(ri)
            adj[ri].add(li)
        for a, b, pairs in paired_edges:
            for oa, ob in pairs:
                allow[(a, b)][oa].add(ob)
                allow[(b, a)][ob].add(oa)
            adj[a].add(b)
            adj[b].add(a)
        # Flow couplings: producers feeding a clustered consumer's shared inp are
        # forced to the same out_idx (same-index coupling, star to the rep).
        for producers in flow_couplings:
            ps = sorted(producers)
            rep = ps[0]
            for q in ps[1:]:
                uf.union(rep, q)
                for o in self.allowed_out.get(rep, []):
                    allow[(rep, q)][o].add(o)
                for o in self.allowed_out.get(q, []):
                    allow[(q, rep)][o].add(o)
                adj[rep].add(q)
                adj[q].add(rep)

        comps: dict[int, list[int]] = defaultdict(list)
        for node in opt.strats:
            if node.op == "output":
                continue
            v = opt.node_map[node]
            comps[uf.find(v)].append(v)

        cost_bearing_set = set(self.cost_bearing)
        self.groups = []
        self.node_to_group = {}
        for members in comps.values():
            members.sort()
            group = _Group(members=members)
            group.cost_bearing = [m for m in members if m in cost_bearing_set]
            group.choices = self._enumerate_choices(members, allow, adj)
            if not group.choices:
                raise RuntimeError(
                    f"No feasible joint choice for group {members}; "
                    "constraints are contradictory."
                )
            gid = len(self.groups)
            self.groups.append(group)
            for m in members:
                self.node_to_group[m] = gid

    def _enumerate_choices(self, members, allow, adj):
        if len(members) == 1:
            v = members[0]
            return [{v: o} for o in self.allowed_out.get(v, [])]
        member_set = set(members)
        # BFS order from a representative so every member after the first is
        # adjacent to an already-assigned one; coupling then propagates
        # deterministically (no spurious K-way branching that explodes the
        # domain for large cluster+paired groups).
        order = []
        seen = set()
        for start in members:
            if start in seen:
                continue
            queue = [start]
            seen.add(start)
            while queue:
                m = queue.pop(0)
                order.append(m)
                for nb in adj[m]:
                    if nb in member_set and nb not in seen:
                        seen.add(nb)
                        queue.append(nb)
        results: list[dict[int, int]] = []
        limit = self.group_domain_limit

        def candidates(m, assign):
            cand = None
            for nb in adj[m]:
                if nb in assign and nb in member_set:
                    allowed = allow[(nb, m)].get(assign[nb], set())
                    cand = allowed if cand is None else (cand & allowed)
            cand = set(self.allowed_out.get(m, [])) if cand is None else (
                cand & set(self.allowed_out.get(m, [])))
            return cand

        def dfs(i, assign):
            if len(results) >= limit:
                return
            if i == len(order):
                results.append(dict(assign))
                return
            m = order[i]
            for val in sorted(candidates(m, assign)):
                assign[m] = val
                dfs(i + 1, assign)
                del assign[m]
                if len(results) >= limit:
                    return

        dfs(0, {})
        if len(results) >= limit:
            logger.warning(
                "Approximate solver: group of %d nodes hit group_domain_limit=%d.",
                len(members), limit,
            )
        return results

    def _prune_candidates(self):
        if self.candidate_limit is None:
            return
        for group in self.groups:
            if len(group.members) != 1 or len(group.choices) <= self.candidate_limit:
                continue
            v = group.members[0]
            node = self.opt.nodes[v]
            lbs = sorted(
                (self._choice_lower_bound(v, node, c[v]), ci)
                for ci, c in enumerate(group.choices)
            )
            keep = {ci for _, ci in lbs[: self.candidate_limit]}
            group.choices = [group.choices[ci] for ci in sorted(keep)]

    def _choice_lower_bound(self, v, node, o):
        opt = self.opt
        strat = opt.strats[node].strategies[o]
        mult = self.node_mult[v]
        dv0 = self._surviving_dv(v, 0, o)
        if dv0 is None:
            return INF  # every edge for this output strategy was pruned
        lb = dv0.compute_cost * len(strat.redistribute_cost)
        lb *= mult
        for argi, _p in self.input_edges.get(v, []):
            best = INF
            for inp in range(len(strat.redistribute_cost[argi])):
                key = (v, argi, o, inp)
                if self._is_forbidden(key):
                    continue
                dv = opt.decision_vars[key]
                best = min(best, dv.comm_cost + dv.sharding_transition_cost)
            if math.isfinite(best):
                lb += mult * best
        return lb

    # ------------------------------------------------------------------ #
    # Memory constraint (ratios, budget, tight-budget param pinning)
    # ------------------------------------------------------------------ #
    def _build_memory_info(self):
        opt = self.opt
        factors = None
        for fname, kwargs in getattr(opt, "_constraint_log", []):
            if fname == "add_parameter_memory_constraint":
                factors = kwargs
        if factors is None:
            return
        try:
            from torch._functorch._aot_autograd.fx_utils import get_param_nodes

            param_nodes = get_param_nodes(opt.graph)
        except Exception:
            return

        low_f, high_f = factors["memory_factor_low"], factors["memory_factor_high"]
        budget_low = budget_high = 0.0
        param_idxs, ratios = [], {}
        for node in param_nodes:
            v = opt.node_map[node]
            param_idxs.append(v)
            r = {o: self._param_ratio(v, node, o) for o in self.allowed_out.get(v, [])}
            ratios[v] = r
            best = min(r.values())
            budget_low += max(best, low_f)
            budget_high += max(best, high_f)

        tight = abs(budget_high - budget_low) < 1e-9
        if tight:
            # Σ ratio == Σ min(ratio) forces every param to a min-ratio choice.
            for v in param_idxs:
                r = ratios[v]
                mn = min(r.values())
                self.allowed_out[v] = [o for o in self.allowed_out[v]
                                       if r[o] <= mn + 1e-12]
        self._memory = {
            "param_idxs": param_idxs,
            "ratios": ratios,
            "budget_low": budget_low,
            "budget_high": budget_high,
            "tight": tight,
        }

    def _param_ratio(self, v, node, o):
        spec = self._surviving_dv(v, 0, o).input_spec
        new_shape, _ = _get_sharded_shape_stride(spec)
        return math.prod(new_shape) / math.prod(spec.tensor_meta.shape)

    # ------------------------------------------------------------------ #
    # Factor graph (numpy unary + pairwise matrices over groups)
    # ------------------------------------------------------------------ #
    def _build_factors(self):
        G = len(self.groups)
        # per member, its out_idx across its group's choices
        member_vals = []
        for group in self.groups:
            mv = {}
            for m in group.cost_bearing:
                mv[m] = np.array([c[m] for c in group.choices], dtype=np.int64)
            # also predecessors that are non-cost-bearing but in this group
            for m in group.members:
                if m not in mv:
                    mv[m] = np.array([c[m] for c in group.choices], dtype=np.int64)
            member_vals.append(mv)

        self.g_unary = [np.zeros(g.domain) for g in self.groups]
        for gid, group in enumerate(self.groups):
            for m in group.cost_bearing:
                vals = member_vals[gid][m]
                self.g_unary[gid] += self.node_mult[m] * self._self_cost_vec(m, vals)

        C: dict[tuple, np.ndarray] = {}
        nbr_set: list[set] = [set() for _ in range(G)]
        for v in self.cost_bearing:
            gv = self.node_to_group[v]
            mult = self.node_mult[v]
            for argi, p in self.input_edges[v]:
                gp = self.node_to_group[p]
                R = self._edge_matrix(v, argi, p)  # (Kv, Kp) raw, BIG if forbidden
                av = member_vals[gv][v]
                bp = member_vals[gp][p]
                contrib = mult * R[np.ix_(av, bp)]  # (D_gv, D_gp)
                if gv == gp:
                    self.g_unary[gv] += np.diagonal(contrib)
                else:
                    a, b = (gv, gp) if gv < gp else (gp, gv)
                    mat = contrib if gv < gp else contrib.T
                    if (a, b) in C:
                        C[(a, b)] += mat
                    else:
                        C[(a, b)] = mat.copy()
                    nbr_set[a].add(b)
                    nbr_set[b].add(a)
        self.C = C
        self.nbrs = [sorted(s) for s in nbr_set]

    def _self_cost_vec(self, m, out_indices):
        """Vectorized self-cost (compute + producer-less arg costs) for node m
        over an array of out_idx."""
        opt = self.opt
        node = opt.nodes[m]
        prod = self._arg_prod.get(m, {})
        out = np.empty(len(out_indices))
        for i, o in enumerate(out_indices):
            strat = opt.strats[node].strategies[o]
            n_args = len(strat.redistribute_cost)
            dv0 = self._surviving_dv(m, 0, o)
            if dv0 is None:  # whole output strategy pruned
                out[i] = BIG
                continue
            c = dv0.compute_cost * n_args
            # Args with no flow edge (constructors / None-spec) are scored at
            # inp=0 here; args with a producer are charged via the pairwise edges.
            for argi in range(n_args):
                if argi in prod:
                    continue
                key = (m, argi, o, 0)
                if self._is_forbidden(key):
                    c = BIG
                    break
                dv = opt.decision_vars[key]
                c += dv.comm_cost + dv.sharding_transition_cost
            out[i] = c
        return out

    def _edge_matrix(self, v, argi, p):
        """Raw (Kv, Kp) edge cost matrix R[o_v][o_p] = comm + transition, BIG when
        the (o_v, o_p) combination is forbidden. Only entries that can actually be
        indexed by the group choices are filled; the rest are BIG."""
        opt = self.opt
        Kv = len(opt.strats[opt.nodes[v]].strategies)
        Kp = len(opt.strats[opt.nodes[p]].strategies)
        R = np.full((Kv, Kp), BIG)
        gv = self.node_to_group[v]
        gp = self.node_to_group[p]
        ov_vals = sorted({c[v] for c in self.groups[gv].choices})
        op_vals = sorted({c[p] for c in self.groups[gp].choices})
        for ov in ov_vals:
            for op in op_vals:
                key = (v, argi, ov, op)
                if self._is_forbidden(key):
                    continue
                dv = opt.decision_vars[key]
                R[ov, op] = dv.comm_cost + dv.sharding_transition_cost
        return R

    def _pair_matrix(self, g, h):
        """Pairwise cost oriented as (x_g, x_h)."""
        if g < h:
            return self.C[(g, h)]
        return self.C[(h, g)].T

    # ------------------------------------------------------------------ #
    # Energy (fast, numpy)
    # ------------------------------------------------------------------ #
    def _fast_group_energy(self, gid, ci):
        e = self.g_unary[gid][ci]
        for h in self.nbrs[gid]:
            ch = self.groups[h].current
            e += self.C[(gid, h)][ci, ch] if gid < h else self.C[(h, gid)][ch, ci]
        return e

    def _fast_total_energy(self):
        total = 0.0
        for gid, g in enumerate(self.groups):
            total += self.g_unary[gid][g.current]
        for (a, b), mat in self.C.items():
            total += mat[self.groups[a].current, self.groups[b].current]
        return total

    # ------------------------------------------------------------------ #
    # Belief propagation (min-sum) + decode
    # ------------------------------------------------------------------ #
    def _belief_propagation(self, deadline=None):
        """Sequential tree-reweighted message passing (TRW-S).

        Plain loopy min-sum BP settles into globally-inconsistent fixed points on
        this MRF (empirically 5-16% above the optimum). TRW-S optimizes a convex
        upper bound over a tree decomposition (here: monotonic chains induced by a
        node ordering), so on the integral sharding problem it converges to the
        exact MAP. Node g is reweighted by 1/(chains through g) = 1/max(in,out)deg
        under the ordering; forward and backward half-sweeps send only along edges
        oriented with the pass. We decode each sweep and keep the best assignment."""
        G = len(self.groups)
        if G == 0:
            return
        unary = self.g_unary
        nbrs = self.nbrs

        order = sorted(range(G), key=lambda g: min(self.groups[g].members))
        pos = [0] * G
        for i, g in enumerate(order):
            pos[g] = i
        gamma = np.ones(G)
        for g in range(G):
            indeg = sum(1 for h in nbrs[g] if pos[h] < pos[g])
            outdeg = sum(1 for h in nbrs[g] if pos[h] > pos[g])
            gamma[g] = 1.0 / max(indeg, outdeg, 1)

        msg: dict[tuple, np.ndarray] = {}
        for g in range(G):
            for h in nbrs[g]:
                msg[(g, h)] = np.zeros(len(unary[h]))

        # We decode every sweep and keep the best assignment. The decoded energy
        # converges in long, irregular plateaus (it can sit at a high value for
        # ~100 sweeps, drop, plateau again, then drop to the optimum), so neither
        # an energy-plateau counter nor a message-delta threshold detects true
        # convergence without stopping on a false plateau. We therefore run a
        # fixed sweep budget (bounded by the time deadline), which is enough for
        # the slowest converger observed, and an exact fixed point ends early.
        best_e = INF
        best_snap = None
        for sweep in range(self.bp_iters):
            max_delta = 0.0
            for forward in (True, False):
                for g in order if forward else order[::-1]:
                    if not nbrs[g]:
                        continue
                    wp = unary[g].copy()
                    for r in nbrs[g]:
                        wp += msg[(r, g)]
                    wp *= gamma[g]
                    for h in nbrs[g]:
                        if (pos[h] > pos[g]) != forward:
                            continue
                        P = self._pair_matrix(g, h)  # (D_g, D_h)
                        m = ((wp - msg[(h, g)])[:, None] + P).min(axis=0)
                        m -= m.min()
                        d = np.abs(m - msg[(g, h)]).max()
                        if d > max_delta:
                            max_delta = d
                        msg[(g, h)] = m
            self._decode(msg)
            e = self._fast_total_energy()
            if e < best_e:
                best_e, best_snap = e, [grp.current for grp in self.groups]
            self._bp_last_iter = sweep + 1
            self._bp_last_delta = max_delta
            if max_delta == 0.0 or (
                deadline is not None and time.perf_counter() > deadline
            ):
                break

        if best_snap is not None:
            for gid, ci in enumerate(best_snap):
                self._set_group(gid, ci)

    def _decode(self, msg):
        """Sequential topological decode: fix each group to the argmin of its
        belief conditioned on already-decoded neighbors (exact pairwise cost) and
        BP messages for the rest. Produces a consistent, forbidden-avoiding
        assignment, unlike independent argmin on a loopy graph."""
        G = len(self.groups)
        order = sorted(range(G), key=lambda g: min(self.groups[g].members))
        decided: dict[int, int] = {}
        for g in order:
            b = self.g_unary[g].copy()
            for h in self.nbrs[g]:
                if h in decided:
                    b = b + self._pair_matrix(g, h)[:, decided[h]]
                else:
                    b = b + msg[(h, g)]
            ci = int(np.argmin(b))
            decided[g] = ci
            self._set_group(g, ci)

    # ------------------------------------------------------------------ #
    # Local search
    # ------------------------------------------------------------------ #
    def _set_group(self, gid, ci):
        group = self.groups[gid]
        group.current = ci
        for m, o in group.choices[ci].items():
            self.cur_out[m] = o

    def _coordinate_descent(self, deadline):
        for _ in range(self.max_sweeps):
            if time.perf_counter() > deadline:
                break
            improved = False
            for gid in range(len(self.groups)):
                if self.groups[gid].domain <= 1:
                    continue
                cur = self.groups[gid].current
                best_i, best_e = cur, self._fast_group_energy(gid, cur)
                for ci in range(self.groups[gid].domain):
                    if ci == cur:
                        continue
                    e = self._fast_group_energy(gid, ci)
                    if e < best_e - 1e-6 and self._memory_ok_after(gid, ci):
                        best_i, best_e = ci, e
                if best_i != cur:
                    self._set_group(gid, best_i)
                    improved = True
            if not improved:
                break

    def _star_block_search(self, deadline):
        ranked = sorted(
            ((len(self.nbrs[g]), g) for g in range(len(self.groups))
             if len(self.nbrs[g]) >= 2 and self.groups[g].domain > 1),
            reverse=True,
        )
        for _ in range(self.star_passes):
            if time.perf_counter() > deadline:
                break
            improved = False
            for _deg, gid in ranked:
                if time.perf_counter() > deadline:
                    break
                if self._optimize_star(gid):
                    improved = True
            if not improved:
                break

    def _optimize_star(self, gid):
        children = [h for h in self.nbrs[gid] if self.groups[h].domain > 1]
        child_costs = sorted(
            ((self._fast_group_energy(h, self.groups[h].current), h) for h in children),
            reverse=True,
        )
        child_ids = [h for _e, h in child_costs[: self.max_star_children]]
        if not child_ids:
            return False
        block = [gid, *child_ids]
        base = self._block_energy(block)
        best_energy = base
        best_center = self.groups[gid].current
        best_children = {h: self.groups[h].current for h in child_ids}
        for ci in range(self.groups[gid].domain):
            self._set_group(gid, ci)
            if not self._memory_ok_after(gid, ci):
                continue
            chosen = {}
            for h in child_ids:
                b_i, b_e = self.groups[h].current, INF
                for hi in range(self.groups[h].domain):
                    e = self._fast_group_energy(h, hi)
                    if e < b_e:
                        b_i, b_e = hi, e
                self._set_group(h, b_i)
                chosen[h] = b_i
            energy = self._block_energy(block)
            if energy < best_energy - 1e-6 and self._block_memory_ok():
                best_energy = energy
                best_center = ci
                best_children = dict(chosen)
        self._set_group(gid, best_center)
        for h, hi in best_children.items():
            self._set_group(h, hi)
        return best_energy < base - 1e-6

    def _block_energy(self, gids):
        total = 0.0
        seen_edges = set()
        for g in gids:
            total += self.g_unary[g][self.groups[g].current]
            for h in self.nbrs[g]:
                key = (g, h) if g < h else (h, g)
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                a, b = key
                total += self.C[key][self.groups[a].current, self.groups[b].current]
        return total

    # ------------------------------------------------------------------ #
    # Memory repair
    # ------------------------------------------------------------------ #
    def _current_memory(self):
        if self._memory is None:
            return 0.0
        return sum(self._memory["ratios"][v][self.cur_out[v]]
                   for v in self._memory["param_idxs"])

    def _memory_ok_after(self, gid, ci):
        if self._memory is None or self._memory.get("tight"):
            return True
        ratios = self._memory["ratios"]
        choice = self.groups[gid].choices[ci]
        delta = sum(ratios[m][o] - ratios[m][self.cur_out[m]]
                    for m, o in choice.items() if m in ratios)
        mem = self._current_memory() + delta
        return (self._memory["budget_low"] - 1e-6 <= mem
                <= self._memory["budget_high"] + 1e-6)

    def _block_memory_ok(self):
        if self._memory is None or self._memory.get("tight"):
            return True
        mem = self._current_memory()
        return (self._memory["budget_low"] - 1e-6 <= mem
                <= self._memory["budget_high"] + 1e-6)

    def _memory_repair(self):
        if self._memory is None or self._memory.get("tight"):
            return
        low, high = self._memory["budget_low"], self._memory["budget_high"]
        ratios = self._memory["ratios"]
        param_groups = {self.node_to_group[v] for v in self._memory["param_idxs"]
                        if v in self.node_to_group}
        for _ in range(2 * max(1, len(param_groups))):
            mem = self._current_memory()
            if low - 1e-6 <= mem <= high + 1e-6:
                return
            over = mem > high
            best = None
            for gid in param_groups:
                group = self.groups[gid]
                cur_e = self._fast_group_energy(gid, group.current)
                for ci in range(group.domain):
                    if ci == group.current:
                        continue
                    choice = group.choices[ci]
                    dmem = sum(ratios[m][choice[m]] - ratios[m][self.cur_out[m]]
                               for m in choice if m in ratios)
                    if (dmem < -1e-9) != over and abs(dmem) > 1e-9:
                        continue
                    if abs(dmem) <= 1e-9:
                        continue
                    score = (self._fast_group_energy(gid, ci) - cur_e) / abs(dmem)
                    if best is None or score < best[0]:
                        best = (score, gid, ci)
            if best is None:
                logger.warning("Approximate solver: memory repair stuck at %.4f "
                               "(budget=[%.4f,%.4f]).", mem, low, high)
                return
            self._set_group(best[1], best[2])

    # ------------------------------------------------------------------ #
    # Write-back
    # ------------------------------------------------------------------ #
    def total_objective(self):
        """Exact objective of the current assignment via decision_vars (for
        verification); equals pulp.value(prob.objective) after write-back."""
        total = 0.0
        for v in self.cost_bearing:
            node = self.opt.nodes[v]
            o = self.cur_out[v]
            strat = self.opt.strats[node].strategies[o]
            prod = self._arg_prod.get(v, {})
            n_args = len(strat.redistribute_cost)
            c = 0.0
            for argi in range(n_args):
                p = prod.get(argi)
                inp = self.cur_out[p] if p is not None else 0
                key = (v, argi, o, inp)
                if self._is_forbidden(key):
                    return INF
                c += self.opt.decision_vars[key].cost
            total += self.node_mult[v] * c
        return total

    def _write_back(self):
        opt = self.opt
        has_pulp = bool(opt.pulp_variables)
        if has_pulp:
            for var in opt.pulp_variables.values():
                var.varValue = 0
        selected = []
        feasible = True
        for v in self.cost_bearing:
            node = opt.nodes[v]
            o = self.cur_out[v]
            strat = opt.strats[node].strategies[o]
            prod = self._arg_prod.get(v, {})
            for argi in range(len(strat.redistribute_cost)):
                p = prod.get(argi)
                inp = self.cur_out[p] if p is not None else 0
                key = (v, argi, o, inp)
                if self._is_forbidden(key):
                    feasible = False
                # A pruned key has no PuLP variable; the infeasible flag above
                # already records it (and raises in _solve).
                if has_pulp and key in opt.pulp_variables:
                    opt.pulp_variables[key].varValue = 1
                selected.append(key)
        opt.selected_keys = list(selected)
        for rk in selected:
            opt.selected_keys.extend(opt._linked_option_keys(rk))
        # Populate prob.objective (when a PuLP problem exists) so callers can also
        # score via pulp.value(prob.objective); the returned value uses the
        # equivalent but cheaper total_objective(). In the lite (no-PuLP) build,
        # there is no problem to populate.
        if opt.prob is not None:
            opt.prob.status = pulp.LpStatusOptimal
            opt.prob.sol_status = pulp.LpSolutionOptimal
            opt._set_objective()
        return INF if not feasible else self.total_objective()
