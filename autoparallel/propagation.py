# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Shardy-like sharding propagation to seed and shrink the ILP search space.

The ILP in :mod:`optimize_sharding` enumerates, for every node, every valid
combination of input/output placements and lets the solver pick the global
optimum.  For large models this search space is enormous even though, in
practice, a handful of user decisions ("these weights are tensor-parallel",
"the batch is data-parallel") already pin down the strategy for the vast
majority of the graph.

This module lets the user attach a small number of *sharding annotations* and
then propagates them through the graph the way `Shardy
<https://github.com/openxla/shardy>`_ does: it pushes each known sharding along
edges that require no resharding, narrowing every node's set of candidate
strategies until the unambiguous nodes are fully determined.  Determined nodes
are turned into ILP constraints, which collapses the search space and the solve
time while leaving the genuinely ambiguous decisions (and where to place the
necessary collectives) to the ILP.

Key design points that mirror Shardy:

* **Per-mesh-axis propagation.**  A placement is propagated one mesh axis at a
  time.  This is what lets, e.g., the tensor-parallel sharding of a weight flow
  through a matmul on the ``tp`` axis while the ``dp`` axis is independently
  resolved (data-parallel batch, with FSDP all-gathers left to the ILP).  It is
  the analogue of Shardy projecting tensor shardings onto per-factor axes.
* **Conservative, reshard-free propagation.**  Along an edge we only narrow a
  consumer to the placements it can take *without* a reshard from the producer
  (zero ``redistribute_cost``).  At a genuine reshard boundary (a necessary
  collective, e.g. an all-reduce or all-gather) no zero-cost option exists, so
  propagation stops there and the ILP decides the collective.  This never
  empties a domain.
* **Priority rounds.**  Annotations carry a priority (lower = applied first,
  matching Shardy).  Data/activation annotations propagate before weight
  annotations so that, where they compete for the same mesh axis (the ``dp``
  axis of a matmul), the data-parallel sharding wins and the weight is
  all-gathered rather than the activation being resharded.
"""

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor.placement_types import Placement

logger = logging.getLogger(__name__)

# A per-axis placement value; ``None`` means "open" (unconstrained on that axis).
AxisPlacement = Optional[Placement]


@dataclass(frozen=True)
class ShardingAnnotation:
    """A user-provided sharding hint for one tensor (graph node).

    Args:
        placements: one entry per mesh dimension.  Each entry is a
            :class:`Placement` (e.g. ``Shard(0)``, ``Replicate()``) or ``None``
            to leave that mesh axis open for propagation / the ILP to decide.
            Leaving an axis open is the common case for weights: the user pins
            the tensor-parallel axis and lets FSDP on the data axis be chosen by
            the optimizer.
        priority: lower numbers are propagated first.  Activation/IO hints
            should have a smaller priority than weight hints so the
            data-parallel axis wins shared-axis conflicts.
    """

    placements: tuple[AxisPlacement, ...]
    priority: int = 0


# Micro-strategy: a single strategy projected onto one mesh axis.
# ``in_reqs`` is the per-axis input placement required for each tensor argument
# (``None`` for non-tensor / undefined args); ``out`` is the per-axis output
# placement produced.
@dataclass(frozen=True)
class _Micro:
    in_reqs: tuple[AxisPlacement, ...]
    out: AxisPlacement


@dataclass
class PropagationResult:
    """Summary of a propagation run, for logging and tests."""

    determined: dict = field(default_factory=dict)  # node -> [(mesh_dim, placement)]
    strategies_before: int = 0
    strategies_after: int = 0
    nodes_touched: int = 0
    nodes_determined: int = 0
    axis_constraints: int = 0

    @property
    def reduction(self) -> float:
        if self.strategies_before == 0:
            return 0.0
        return 1.0 - self.strategies_after / self.strategies_before


class ShardingPropagator:
    """Propagates sharding annotations over an optimizer's strategy graph.

    The propagator works on the optimizer's concrete graph and reuses its
    per-node ``OpStrategy`` list (``optimizer.strats``) as the per-op sharding
    rules.  It maintains, for every single-output node and every mesh axis, the
    set of still-feasible per-axis (input-requirement, output) micro-strategies
    and shrinks them to a fixed point.
    """

    def __init__(self, optimizer):
        self.opt = optimizer
        self.mesh = optimizer.mesh
        self.ndim = optimizer.mesh.ndim

        # node -> list (indexed by mesh dim) of list[_Micro]
        self.micros: dict = {}
        # node -> list (indexed by mesh dim) of set[int] (feasible micro indices)
        self.dom: dict = {}
        # nodes whose domain has been narrowed below the initial full set
        self.touched: set = set()
        self._initial_strategy_count: dict = {}

        self._build_micros()

    # ---- construction ----

    def _build_micros(self):
        for node, op_strat in self.opt.strats.items():
            if node.op == "output":
                continue
            strategies = op_strat.strategies
            if not strategies:
                continue
            # Multi-output nodes (tuple output_specs, e.g. SDPA) are propagation
            # barriers: there is no single output placement to project, so we
            # neither narrow them nor propagate across them.  Their getitem
            # users are single-output and handled normally.
            if not isinstance(strategies[0].output_specs, DTensorSpec):
                continue

            args = self.opt._all_input_nodes(node)
            n_args = len(args)
            self._initial_strategy_count[node] = len(strategies)

            per_axis_index: list = [dict() for _ in range(self.ndim)]
            per_axis_micros: list = [[] for _ in range(self.ndim)]
            for s in strategies:
                out_pl = s.output_specs.placements
                in_pls = []
                for a in range(n_args):
                    isp = s.input_specs[a] if a < len(s.input_specs) else None
                    in_pls.append(
                        isp.placements if isinstance(isp, DTensorSpec) else None
                    )
                for m in range(self.ndim):
                    in_reqs = tuple(None if pl is None else pl[m] for pl in in_pls)
                    micro = _Micro(in_reqs=in_reqs, out=out_pl[m])
                    idx = per_axis_index[m]
                    if micro not in idx:
                        idx[micro] = len(per_axis_micros[m])
                        per_axis_micros[m].append(micro)
            self.micros[node] = per_axis_micros
            self.dom[node] = [
                set(range(len(per_axis_micros[m]))) for m in range(self.ndim)
            ]

    # ---- accessors ----

    def _out_set(self, node, m) -> set:
        micros = self.micros[node][m]
        return {micros[i].out for i in self.dom[node][m]}

    def _in_req_set(self, node, m, a) -> set:
        micros = self.micros[node][m]
        return {micros[i].in_reqs[a] for i in self.dom[node][m]}

    def _consumer_edges(self, node):
        """Yield (consumer, arg_index) for each tensor edge out of ``node``."""
        for user in node.users:
            if user not in self.dom:
                continue
            in_nodes = self.opt._all_input_nodes(user)
            for a, src in enumerate(in_nodes):
                if src is node:
                    yield user, a

    # ---- seeding ----

    def seed(self, node, placements: tuple) -> bool:
        node = self.opt._normalize_node(node)
        if node not in self.dom:
            logger.debug("seed: %s is not a single-output node, ignoring", node)
            return False
        changed = False
        for m in range(self.ndim):
            want = placements[m] if m < len(placements) else None
            if want is None:
                continue
            micros = self.micros[node][m]
            # Seeding is authoritative: recompute from the full strategy set so a
            # user annotation overrides any earlier (lower-priority) propagation
            # that may have narrowed this axis away from the annotated value.
            keep = {i for i in range(len(micros)) if micros[i].out == want}
            if not keep:
                available = {micros[i].out for i in range(len(micros))}
                raise ValueError(
                    f"Annotation {placements} is not achievable for node "
                    f"{node} on mesh dim {m}: this op only supports "
                    f"{available} on that axis"
                )
            if keep != self.dom[node][m]:
                self.dom[node][m] = keep
                changed = True
        if changed:
            self.touched.add(node)
        return changed

    # ---- narrowing ----

    def _narrow_from_producers(self, node) -> bool:
        """Narrow ``node`` (as a consumer) toward reshard-free inputs."""
        changed = False
        args = self.opt._all_input_nodes(node)
        for a, producer in enumerate(args):
            if producer not in self.dom:
                continue  # barrier or non-tensor producer
            for m in range(self.ndim):
                prod_outs = self._out_set(producer, m)
                cur = self.dom[node][m]
                micros = self.micros[node][m]
                keep = {i for i in cur if micros[i].in_reqs[a] in prod_outs}
                # Only tighten when a zero-reshard option exists; an empty keep
                # means this edge is a genuine reshard boundary -> leave it to
                # the ILP.
                if keep and keep != cur:
                    self.dom[node][m] = keep
                    changed = True
        return changed

    def _narrow_from_consumer(self, node) -> bool:
        """Narrow ``node`` (as a producer) toward what its single consumer wants.

        Restricted to single-consumer producers: a multi-consumer value (e.g. a
        residual stream) may legitimately be resharded for some consumers, so we
        do not let one consumer dictate it.

        Placeholders (parameters, buffers, graph inputs) are never narrowed this
        way: their placement is the *stored* sharding, which legitimately differs
        from the *compute* sharding the consumer needs by a reshard (e.g. an FSDP
        all-gather on the data axis).  Inferring the storage sharding from the
        consumer would wrongly pin, e.g., a weight to Replicate on the data axis
        and defeat FSDP.  A placeholder's sharding comes only from its own
        annotation; everything else about it is left to the ILP.
        """
        if node.op in ("placeholder", "get_attr"):
            return False
        edges = list(self._consumer_edges(node))
        if len(edges) != 1:
            return False
        consumer, a = edges[0]
        changed = False
        for m in range(self.ndim):
            cons_reqs = self._in_req_set(consumer, m, a)
            cur = self.dom[node][m]
            micros = self.micros[node][m]
            keep = {i for i in cur if micros[i].out in cons_reqs}
            if keep and keep != cur:
                self.dom[node][m] = keep
                changed = True
        return changed

    def _narrow_node(self, node) -> bool:
        c1 = self._narrow_from_producers(node)
        c2 = self._narrow_from_consumer(node)
        changed = c1 or c2
        if changed:
            self.touched.add(node)
        return changed

    def propagate(self):
        """Run the worklist narrowing to a fixed point."""
        wl = deque(self.dom.keys())
        inq = set(self.dom.keys())
        steps = 0
        while wl:
            node = wl.popleft()
            inq.discard(node)
            steps += 1
            if not self._narrow_node(node):
                continue
            # Re-enqueue neighbors whose domains may now narrow further.
            neighbors = list(self.opt._all_input_nodes(node))
            neighbors += [u for u in node.users]
            for nb in neighbors:
                if nb in self.dom and nb not in inq:
                    wl.append(nb)
                    inq.add(nb)
        logger.debug("propagation fixpoint reached in %d worklist steps", steps)

    # ---- results ----

    def determined(self) -> dict:
        """node -> list[(mesh_dim, placement)] for every determined axis of a
        node that propagation actually touched."""
        res = {}
        for node in self.dom:
            if node not in self.touched:
                continue
            axes = []
            for m in range(self.ndim):
                outs = self._out_set(node, m)
                if len(outs) == 1:
                    axes.append((m, next(iter(outs))))
            if axes:
                res[node] = axes
        return res

    def _feasible_strategy_count(self, node, determined_axes) -> int:
        """How many of ``node``'s strategies satisfy all determined axes."""
        strategies = self.opt.strats[node].strategies
        count = 0
        for s in strategies:
            spec = s.output_specs
            if not isinstance(spec, DTensorSpec):
                count += 1
                continue
            if all(spec.placements[m] == p for m, p in determined_axes):
                count += 1
        return count

    def run(self, annotations) -> dict:
        """Seed ``annotations`` in priority order and propagate to a fixed point.

        ``annotations`` is a list of ``(node, ShardingAnnotation)``.  Returns the
        ``determined()`` mapping.
        """
        by_priority: dict = defaultdict(list)
        for node, ann in annotations:
            by_priority[ann.priority].append((node, ann))
        for priority in sorted(by_priority):
            for node, ann in by_priority[priority]:
                self.seed(node, ann.placements)
            self.propagate()
        return self.determined()

    def _paired_boundary_nodes(self) -> set:
        """Backward nodes tied to a forward node by a forward/backward
        consistency constraint: parameter gradients, input gradients, and output
        tangents.  These must be left to the pairing (which mirrors the forward
        decision onto them); constraining them independently can contradict it.
        """
        from torch._functorch._aot_autograd.fx_utils import (
            get_param_and_grad_nodes,
            get_plain_input_and_grad_nodes,
            get_plain_output_and_tangent_nodes,
        )

        graph = self.opt.graph
        nodes = set()
        for _p, grad in get_param_and_grad_nodes(graph).values():
            if grad is not None:
                nodes.add(grad)
        for _i, grad in get_plain_input_and_grad_nodes(graph).values():
            if grad is not None:
                nodes.add(grad)
        for _o, tangent in get_plain_output_and_tangent_nodes(graph).values():
            if tangent is not None:
                nodes.add(tangent)
        return nodes

    def _backward_node_set(self) -> set:
        """Nodes belonging to the backward pass: everything reachable from a
        tangent (incoming-gradient) placeholder.

        Propagation does not constrain these.  Their sharding is tied to the
        forward pass by the optimizer's forward/backward consistency constraints
        (param<->grad, input<->grad, output<->tangent), so constraining them
        independently risks contradicting that pairing (e.g. forcing a weight's
        gradient to a placement its parameter cannot take).  Leaving them to the
        ILP keeps the problem feasible while the forward constraints already
        collapse most of the backward search space through the pairing.
        """
        seeds = [
            n
            for n in self.opt.graph.nodes
            if n.op == "placeholder" and n.name.startswith("tangents")
        ]
        backward = set()
        stack = list(seeds)
        while stack:
            n = stack.pop()
            for u in n.users:
                if u not in backward:
                    backward.add(u)
                    stack.append(u)
        return backward

    def _total_strategy_count(self) -> int:
        total = 0
        for node, op_strat in self.opt.strats.items():
            if node.op == "output":
                continue
            total += len(op_strat.strategies)
        return total

    def apply_to_optimizer(
        self, forward_only=False, aggressive=False, method="fix"
    ) -> PropagationResult:
        """Emit per-axis constraints for every determined axis of every touched
        node and return a summary of the search-space reduction.

        Nodes the user already constrained explicitly are skipped, as are the
        forward/backward *paired boundary* nodes (parameter/input gradients and
        output tangents), whose sharding is decided by the pairing rather than
        propagation.  When ``forward_only`` is set, all backward-pass nodes are
        skipped (more conservative; only the forward graph is constrained).  A
        node is also skipped if its determined axes do not co-occur in any single
        strategy (a safety net, not expected in practice).

        By default (``aggressive=False``) an axis is only pinned when it is a
        genuine ``Shard``.  A Shard encodes the tensor-parallel structure the
        annotations describe and is invariant in the optimum.  ``Replicate`` and
        ``Partial`` are deliberately *not* pinned:

        * Pinning ``Replicate`` would forbid the ILP from instead sharding that
          axis (e.g. choosing sequence parallelism on the residual stream).
        * ``Partial`` is a pending reduction whose collective (all-reduce /
          reduce-scatter) the ILP places; pinning it fixes where the reduction
          happens and can even be infeasible (a Partial value cannot be added to
          a Replicate residual without first reducing it).

        Both are genuine cost tradeoffs, so leaving them open keeps the optimum
        reachable while costing little search-space reduction.

        ``method`` is forwarded to :meth:`ShardingOptimizer.add_node_axis_constraint`:
        ``"fix"`` (default) removes the ruled-out decision variables so the
        problem actually shrinks, ``"constraint"`` adds equality rows instead.
        """
        determined = self.determined()
        already = set(self.opt._node_constraint_names.values())
        excluded = self._paired_boundary_nodes()
        if forward_only:
            excluded |= self._backward_node_set()

        result = PropagationResult(determined=determined)
        result.strategies_before = self._total_strategy_count()
        result.nodes_touched = len(self.touched)

        strategies_saved = 0
        for node, axes in determined.items():
            if node.name in already or node in excluded:
                continue
            pin_axes = [(m, p) for m, p in axes if aggressive or p.is_shard()]
            if not pin_axes:
                continue
            full = len(self.opt.strats[node].strategies)
            feasible = self._feasible_strategy_count(node, pin_axes)
            if feasible == 0 or feasible == full:
                continue
            for m, p in pin_axes:
                self.opt.add_node_axis_constraint(
                    node, m, p, constraint_name="propagated", method=method
                )
                result.axis_constraints += 1
            result.nodes_determined += 1
            strategies_saved += full - feasible

        result.strategies_after = result.strategies_before - strategies_saved
        logger.info(
            "propagation: touched %d nodes, constrained %d nodes with %d "
            "per-axis constraints; output-strategy choices %d -> %d (%.1f%% "
            "reduction)",
            result.nodes_touched,
            result.nodes_determined,
            result.axis_constraints,
            result.strategies_before,
            result.strategies_after,
            100.0 * result.reduction,
        )
        return result
