# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Sharding optimization using Integer Linear Programming (ILP).

This module solves the optimal sharding strategy problem by formulating it as an ILP
where each binary variable x_{i,a,o,j} ∈ {0,1} represents a choice of input placement j
and output placement o for operation i and argument a. The objective minimizes total cost:

    minimize: Σ_{i,a,o,j} c_{i,a,o,j} * x_{i,a,o,j}

where:
- x_{i,a,o,j}: binary decision variable (1 if strategy selected, 0 otherwise)
- c_{i,a,o,j}: total cost (communication + computation) for this strategy choice

subject to the following constraint categories:

1. UNIQUENESS CONSTRAINTS: Each operation-argument pair must select exactly one
   input-output placement combination.

   ∀i,a: Σ_{o,j} x_{i,a,o,j} = 1

   → Implemented in: add_unique_decision_constraint()

2. CONSISTENCY CONSTRAINTS: For multi-argument operations, all arguments must agree
   on the same output placement to ensure the operation can execute correctly.

   ∀i,o: Σ_j x_{i,0,o,j} = Σ_j x_{i,1,o,j} = ... = Σ_j x_{i,A_i-1,o,j}
   where A_i is the number of arguments for operation i.

   → Implemented in: add_same_output_across_args_constraint()

3. FLOW CONSTRAINTS: The output placement of producer operations must match the
   input placement of consumer operations (dataflow consistency).

   ∀(i→k): Σ_j x_{i,0,o,j} = Σ_j x_{k,a,j,o}
   where operation i feeds into operation k at argument position a.

   → Implemented in: add_output_input_consistent_constraint()

4. COST CONSTRAINTS: Variables with infinite cost (invalid configurations) are
   forced to zero. Efficiency penalties for inefficient collective operations
   (e.g., non-batch-dim shard-to-replicate) are embedded in the cost coefficients
   computed by the cost model (see cost_models/collective_runtime_estimation.py).

   ∀i,a,o,j: c_{i,a,o,j} = ∞ ⟹ x_{i,a,o,j} = 0

   → Implemented in: add_inf_cost_constraint()

5. USER CONSTRAINTS (optional): Force specific placements for inputs, outputs,
   parameters, or memory usage bounds.

   5a. Input/Output constraints: x_{i,a,o*,j*} = 1 for specified (o*,j*)
       → Implemented in: add_sharded_input_constraint(), add_sharded_output_constraint()

   5b. Memory constraints: Σ_{params} (size_ratio * x_{param}) ≤ memory_limit
       → Implemented in: add_parameter_memory_constraint()

   5c. Forward-backward consistency: x_{fwd} = x_{bwd} for paired nodes
       → Implemented in: add_forward_backward_consistency_constraints()

   5d. General node constraints: Force specific placement for any node
       → Implemented in: add_node_constraint()

The solver finds the globally optimal sharding strategy that minimizes total
runtime cost while satisfying all constraints.
"""

import logging
import math
import operator
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import pulp
import torch
from torch._functorch._aot_autograd.descriptors import PlainAOTInput, PlainAOTOutput
from torch._functorch._aot_autograd.fx_utils import (
    get_param_and_grad_nodes,
    get_param_nodes,
    get_plain_input_and_grad_nodes,
    get_plain_output_and_tangent_nodes,
)
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor.placement_types import Placement, Replicate, Shard
from torch.utils._pytree import tree_map_only

from .cost_models.collective_runtime_estimation import estimate_strategy_comms_cost
from .cost_models.compute_estimation import (
    _get_sharded_shape_stride,
    estimate_strategy_runtime_cost,
)
from .graph_passes.graph_clustering import get_identical_regions
from .graph_passes.graph_utils import (
    all_input_nodes,
    build_param_derived_set,
    build_terminal_derived_set,
)
from .shardings.placement_options import get_placement_options_for_node
from .shardings.propagation_rules import _create_all_options

logger = logging.getLogger(__name__)


def concretize_symint(val):
    """Concretize a SymInt to a plain int, pass through other values.

    For unbacked SymInts (hint=None), returns the SymInt unchanged.
    """
    if isinstance(val, torch.SymInt):
        hint = val.node.hint
        return hint if hint is not None else val
    return val


def concretize_args(args):
    """Concretize all SymInts and symbolic FakeTensors in an args tree.

    Returns an args tree where:
    - SymInts are replaced with their concrete hint values (via concretize_symint)
    - FakeTensors with symbolic shapes are replaced with concrete FakeTensors
    - Other values are left unchanged
    """
    from torch._subclasses.fake_tensor import FakeTensor

    def concretize(x):
        if isinstance(x, torch.SymInt):
            return concretize_symint(x)
        if isinstance(x, FakeTensor) and any(
            isinstance(s, torch.SymInt) for s in x.shape
        ):
            concrete_shape = [concretize_symint(s) for s in x.shape]
            concrete_stride = [concretize_symint(s) for s in x.stride()]
            if any(not isinstance(s, int) for s in concrete_shape):
                return x
            with x.fake_mode:
                return torch.empty_strided(
                    concrete_shape, concrete_stride, dtype=x.dtype, device=x.device
                )
        return x

    return tree_map_only((torch.SymInt, FakeTensor), concretize, args)


def _produces_tensor(val):
    """Check if a node's meta value represents tensor output(s)."""
    if isinstance(val, torch.Tensor):
        return True
    if isinstance(val, (tuple, list)):
        return any(_produces_tensor(v) for v in val)
    return False


@dataclass
class DecisionVar:
    """A decision variable in the ILP, representing one (node, arg, output_placement,
    input_placement) choice with its associated costs and strategy metadata."""

    var: Any  # pulp.LpVariable
    cost: float
    compute_cost: float
    comm_cost: float
    sharding_transition_cost: float
    strategy: Any  # OpSpec
    output_spec: Any  # DTensorSpec or tuple[DTensorSpec | None, ...]
    input_spec: Any  # DTensorSpec


def _assert_has_tensor_meta(spec_or_specs, node, label):
    """Assert that all DTensorSpecs in a spec (possibly a tuple) have tensor_meta."""
    if isinstance(spec_or_specs, (list, tuple)):
        for spec in spec_or_specs:
            if spec:
                assert (
                    spec.tensor_meta is not None
                ), f"{node} {label} doesn't have a tensor_meta"
    elif spec_or_specs is not None:
        assert (
            spec_or_specs.tensor_meta is not None
        ), f"{node} {label} doesn't have a tensor_meta"


class ShardingOptimizer:
    def __init__(
        self, gm, mesh, rescale_grad_comm_cost_for_mp=1.0, repeated_subgraphs=False
    ):
        self.gm = gm
        self.graph = gm.graph
        self.mesh = mesh
        self.rescale_grad_comm_cost_for_mp = rescale_grad_comm_cost_for_mp
        self._name_counters: dict[str, int] = {}
        t0 = time.perf_counter()
        self.strats = self.build_sharding_metadata()
        # nodes/node_map are derived from strats (not graph.nodes) so that
        # shape-computation nodes skipped by build_sharding_metadata don't
        # appear and indices stay consistent.
        self.nodes = list(self.strats.keys())
        self.node_map = {node: i for i, node in enumerate(self.nodes)}
        logger.debug("Placement options took %.3fs", time.perf_counter() - t0)
        from autoparallel.shardings.placement_options import get_placement_options_timer

        get_placement_options_timer().report()

        self.cluster_links: dict[tuple, tuple] = {}
        if repeated_subgraphs:
            t = time.time()
            clusters = get_identical_regions(self.gm.graph, self.strats)
            logger.debug(f"Found {len(clusters)} clusters in {time.time() - t:.2f}s")
            self.create_cluster_links(clusters)

        t0 = time.perf_counter()
        self.decision_vars = self._build_decision_vars()
        t1 = time.perf_counter()
        self.validate()
        t2 = time.perf_counter()
        self.prob = pulp.LpProblem("AutoParallel", pulp.LpMinimize)
        self.add_default_constraints()
        t3 = time.perf_counter()
        n_unique_vars = len(set(id(v) for v in self.pulp_variables.values()))
        n_constraints = len(self.prob.constraints)
        logger.debug(
            "ILP construction took %.3fs "
            "(decision_vars=%.3fs, validate=%.3fs, constraints=%.3fs)",
            t3 - t0,
            t1 - t0,
            t2 - t1,
            t3 - t2,
        )
        logger.debug(
            "ILP problem size: %d unique vars, %d decision vars, %d constraints",
            n_unique_vars,
            len(self.decision_vars),
            n_constraints,
        )

    def _get_next_name(self, prefix):
        idx = self._name_counters.setdefault(prefix, 0)
        self._name_counters[prefix] += 1
        return prefix + f"_{idx:03}"

    def build_sharding_metadata(self):
        strats = {}
        for node in self.graph.nodes:
            if node.op in ("placeholder", "get_attr"):
                val = node.meta.get("val")
                if isinstance(val, torch.Tensor):
                    val = concretize_args(val)
                    strats[node] = _create_all_options(self.mesh, val.shape, tensor=val)
                else:
                    # GraphModule submodules used by HOPs — not added to
                    # strats, invisible to the ILP. _all_input_nodes filters
                    # them. Guard: every skipped node must be consumed by a HOP.
                    assert any(
                        isinstance(u.target, torch._ops.HigherOrderOperator)
                        or "local_map" in u.name
                        for u in node.users
                    ), f"Non-tensor get_attr {node} is not used by a HOP"
            elif node.op == "call_function":
                if not _produces_tensor(node.meta.get("val")):
                    # Shape-computation nodes (sym_size, operator.mul, etc.)
                    # produce scalars, not tensors — skip sharding.
                    continue
                user_strats = tree_map_only(
                    torch.fx.Node,
                    lambda x: strats.get(x, concretize_symint(x.meta.get("val"))),
                    node.args,
                )
                user_args = concretize_args(
                    tree_map_only(torch.fx.Node, lambda x: x.meta.get("val"), node.args)
                )
                user_kwargs = concretize_args(
                    tree_map_only(
                        torch.fx.Node, lambda x: x.meta.get("val"), node.kwargs
                    )
                )
                strats[node] = get_placement_options_for_node(
                    self.mesh, node, user_strats, user_args, user_kwargs
                )
            elif node.op == "output":
                user_strats = tree_map_only(
                    torch.fx.Node, lambda x: strats[x], node.args
                )
                strats[node] = user_strats
            else:
                raise ValueError(f"Unexpected node op: {node.op}")
        return strats

    def create_cluster_links(self, clusters):
        """Create a mapping between identical optimization nodes to reduce the
        optimization space. If cluster_links[key1] == key2, the optimization
        problem uses key2's variable in place of key1."""
        for cluster_group in clusters:
            cluster0 = cluster_group[0]
            for cluster_i in cluster_group[1:]:
                for n0, ni in zip(cluster0, cluster_i):
                    idx0 = self.node_map[n0]
                    idx1 = self.node_map[ni]
                    options_n0 = list(self.walk_over_options(n0))
                    options_ni = list(self.walk_over_options(ni))
                    assert options_n0 == options_ni, (
                        f"Problem with graph clustering: {n0} and {ni} don't have the same number "
                        "of input/output placements. Please report a bug"
                    )
                    for argi, out_idx, inp_idx in options_n0:
                        self.cluster_links[(idx1, argi, out_idx, inp_idx)] = (
                            idx0,
                            argi,
                            out_idx,
                            inp_idx,
                        )

    def _all_input_nodes(self, node):
        """Variant of node.all_input_nodes that preserves duplicate nodes.

        Filters out nodes not in self.strats (e.g., shape-computation nodes
        like sym_size / operator.mul, and HOP submodule get_attr nodes).
        """
        return [n for n in all_input_nodes(node) if n in self.strats]

    def walk_over_options(self, node, constrain_arg=None):
        """Yield (argi, out_idx, inp_idx) for all valid strategy combinations."""
        if node not in self.strats:
            return
        op_strategy = self.strats[node]
        for argi in range(len(op_strategy.strategies[0].input_specs)):
            if constrain_arg is not None and argi != constrain_arg:
                continue
            for out_idx, strategy in enumerate(op_strategy.strategies):
                for inp_idx in range(len(strategy.redistribute_cost[argi])):
                    yield argi, out_idx, inp_idx

    def _create_pulp_variables(self):
        """Create PuLP binary variables for all decision points, resolving
        cluster links so that identical nodes share the same variable.

        Returns a dict mapping every (node_idx, argi, out_idx, inp_idx) key
        to its PuLP variable.
        """
        # Map each key to its canonical root key
        root_for_key = {}
        for node, _ in self.strats.items():
            if node.op == "output":
                continue
            node_idx = self.node_map[node]
            for argi, out_idx, inp_idx in self.walk_over_options(node):
                key = (node_idx, argi, out_idx, inp_idx)
                root_for_key[key] = self.cluster_links.get(key, key)

        # Create one PuLP variable per unique root
        root_variables: dict[tuple, pulp.LpVariable] = {}
        for root_key in set(root_for_key.values()):
            root_node_idx, argi, out_idx, inp_idx = root_key
            root_node = self.nodes[root_node_idx]
            root_variables[root_key] = pulp.LpVariable(
                f"n={root_node},s={root_node_idx},arg={argi},"
                f"output_p={out_idx},input_p={inp_idx}",
                cat=pulp.LpBinary,
            )

        return {key: root_variables[root] for key, root in root_for_key.items()}

    def _compute_edge_costs(
        self,
        node,
        output_strategy,
        argi,
        inp_idx,
        default_comm_cost,
        producer_strategy,
        grad_param_nodes,
    ):
        """Compute communication and sharding transition costs for transitioning
        from input placement inp_idx to the output_strategy's expected input at argi.

        The transition cost is a tie-breaker that penalizes unnecessary redistributions:
        when multiple redistribution chains have the same communication cost, we prefer
        fewer individual redistributions so they can be fused into single collectives.

        Returns (comm_cost, sharding_transition_cost).
        """
        comm_cost = default_comm_cost
        sharding_transition_cost = 0

        if producer_strategy is not None:
            src_spec = producer_strategy.strategies[inp_idx].output_specs
            # TODO: operator.getitem being special is something
            # we might want to change in the future
            if node.target == operator.getitem:
                src_spec = src_spec[node.args[1]]
            tgt_spec = output_strategy.input_specs[argi]
            if isinstance(src_spec, DTensorSpec) and isinstance(tgt_spec, DTensorSpec):
                comm_cost = estimate_strategy_comms_cost(src_spec, tgt_spec)
                if src_spec.placements != tgt_spec.placements:
                    sharding_transition_cost = 1

        if node in grad_param_nodes:
            comm_cost = comm_cost / self.rescale_grad_comm_cost_for_mp

        return comm_cost, sharding_transition_cost

    def _build_decision_vars(self):
        """Build DecisionVar entries for every (node_idx, argi, out_idx, inp_idx)
        combination in the strategy space."""
        t_pulp_start = time.perf_counter()
        self.pulp_variables = self._create_pulp_variables()
        t_pulp_end = time.perf_counter()
        grad_param_nodes = set(
            x[1] for x in get_param_and_grad_nodes(self.graph).values()
        )

        # Precompute which node indices are cluster-linked so we can
        # copy costs from the root instead of recomputing them.
        self._cluster_linked_node_idxs = {key[0] for key in self.cluster_links}

        t_compute = 0.0
        t_edge = 0.0
        n_vars = 0
        n_cluster_copied = 0

        decision_vars = {}
        strats_items = [
            (self.node_map[node], node, strat) for node, strat in self.strats.items()
        ]

        # Two passes: root nodes first (so their entries exist), then linked nodes.
        for is_linked_pass in (False, True):
            for node_idx, node, op_strategy in strats_items:
                if node.op == "output":
                    continue
                is_linked = node_idx in self._cluster_linked_node_idxs
                if is_linked != is_linked_pass:
                    continue

                num_args = len(op_strategy.strategies[0].input_specs)

                for out_idx, output_strategy in enumerate(op_strategy.strategies):
                    if is_linked:
                        root_key = self.cluster_links[(node_idx, 0, out_idx, 0)]
                        per_arg_compute = decision_vars[root_key].compute_cost
                    else:
                        tc0 = time.perf_counter()
                        compute_cost = estimate_strategy_runtime_cost(
                            node, output_strategy
                        )
                        tc1 = time.perf_counter()
                        t_compute += tc1 - tc0
                        per_arg_compute = compute_cost / num_args

                    for argi, redist_costs in enumerate(
                        output_strategy.redistribute_cost
                    ):
                        for inp_idx, default_comm_cost in enumerate(redist_costs):
                            key = (node_idx, argi, out_idx, inp_idx)

                            if is_linked:
                                root_key = self.cluster_links[key]
                                root_dv = decision_vars[root_key]
                                comm_cost = root_dv.comm_cost
                                transition_cost = root_dv.sharding_transition_cost
                                n_cluster_copied += 1
                            else:
                                all_input_nodes = self._all_input_nodes(node)
                                producer_strategy = (
                                    self.strats[all_input_nodes[argi]]
                                    if all_input_nodes
                                    else None
                                )
                                te0 = time.perf_counter()
                                comm_cost, transition_cost = self._compute_edge_costs(
                                    node,
                                    output_strategy,
                                    argi,
                                    inp_idx,
                                    default_comm_cost,
                                    producer_strategy,
                                    grad_param_nodes,
                                )
                                te1 = time.perf_counter()
                                t_edge += te1 - te0

                            redist_costs[inp_idx] = comm_cost

                            if not is_linked:
                                decision_vars[key] = DecisionVar(
                                    var=self.pulp_variables[key],
                                    cost=comm_cost + per_arg_compute + transition_cost,
                                    compute_cost=per_arg_compute,
                                    comm_cost=comm_cost,
                                    sharding_transition_cost=transition_cost,
                                    strategy=output_strategy,
                                    output_spec=output_strategy.output_specs,
                                    input_spec=output_strategy.input_specs[argi],
                                )
                            n_vars += 1

        self._root_to_linked: dict[tuple, list[tuple]] = defaultdict(list)
        for linked_key, root_key in self.cluster_links.items():
            self._root_to_linked[root_key].append(linked_key)

        logger.debug(
            "_build_decision_vars breakdown (%d vars, %d cluster-copied): "
            "pulp_vars=%.3fs, compute_cost=%.3fs, edge_cost=%.3fs",
            n_vars,
            n_cluster_copied,
            t_pulp_end - t_pulp_start,
            t_compute,
            t_edge,
        )
        return decision_vars

    def _resolve_decision_var(self, key):
        """Return a DecisionVar for key, reconstructing on the fly for linked keys."""
        dv = self.decision_vars.get(key)
        if dv is not None:
            return dv
        root_key = self.cluster_links[key]
        root_dv = self.decision_vars[root_key]
        node_idx, argi, out_idx, _ = key
        strategy = self.strats[self.nodes[node_idx]].strategies[out_idx]
        return DecisionVar(
            var=self.pulp_variables[key],
            cost=root_dv.cost,
            compute_cost=root_dv.compute_cost,
            comm_cost=root_dv.comm_cost,
            sharding_transition_cost=root_dv.sharding_transition_cost,
            strategy=strategy,
            output_spec=strategy.output_specs,
            input_spec=strategy.input_specs[argi],
        )

    def _collect_vars(self, node, node_idx, argi, group_by, resolve_clusters=False):
        """Collect PuLP variables for a node's options, grouped by strategy index.

        Args:
            group_by: "out_idx" to group by output strategy index,
                      "inp_idx" to group by input strategy index.
            resolve_clusters: If False, skip cluster-linked keys. If True,
                              resolve through cluster_links to root variables.
        """
        result = {}
        for _, out_idx, inp_idx in self.walk_over_options(node, argi):
            key = (node_idx, argi, out_idx, inp_idx)
            if key in self.cluster_links:
                if not resolve_clusters:
                    continue
                var = self.pulp_variables[self.cluster_links[key]]
            else:
                var = self.pulp_variables[key]
            group_key = out_idx if group_by == "out_idx" else inp_idx
            result.setdefault(group_key, []).append(var)
        return result

    def validate(self):
        for node in self.graph.nodes:
            if node.op != "call_function":
                continue
            if node not in self.strats:
                continue
            strat = self.strats[node]
            strat0 = strat.strategies[0]
            all_input_nodes = self._all_input_nodes(node)
            num_input_nodes = len(all_input_nodes)
            if len(strat0.redistribute_cost) != num_input_nodes:
                # only constructor functions allowed here
                assert num_input_nodes == 0, f"{num_input_nodes}"
                assert (
                    len(strat0.redistribute_cost) == 1
                ), f"{len(strat0.redistribute_cost)}"
            assert (len(strat0.redistribute_cost) == num_input_nodes) or (
                num_input_nodes == 0 and len(strat0.redistribute_cost) == 1
            ), f"{node}, {len(strat0.redistribute_cost)}, {num_input_nodes}"

            _assert_has_tensor_meta(strat0.output_specs, node, "output_specs")
            for input_spec in strat0.input_specs:
                _assert_has_tensor_meta(input_spec, node, "input_spec")

            for i, arg in enumerate(all_input_nodes):
                strat_arg = self.strats[arg]
                num_arg_strats = len(strat_arg.strategies)
                assert (
                    len(strat0.redistribute_cost[i]) == num_arg_strats
                ), f"{node}, {len(strat0.redistribute_cost[i])}, {num_arg_strats}"

    # ---- Default constraints ----

    def add_unique_decision_constraint(self):
        """UNIQUENESS (Category 1): Each (operation, argument) pair selects exactly
        one input-output placement combination.

        ∀i,a: Σ_{o,j} x_{i,a,o,j} = 1
        """
        for node in self.graph.nodes:
            if node.op not in {"placeholder", "call_function", "get_attr"}:
                continue
            if node not in self.node_map:
                continue
            node_idx = self.node_map[node]
            if node_idx in self._cluster_linked_node_idxs:
                continue
            arg_vars = {}
            for argi, out_idx, inp_idx in self.walk_over_options(node):
                key = (node_idx, argi, out_idx, inp_idx)
                var = self.pulp_variables[key]
                arg_vars.setdefault(argi, []).append(var)
            for eqs in arg_vars.values():
                self.prob += (
                    pulp.lpSum(eqs) == 1,
                    self._get_next_name("unique_decision"),
                )

    def add_same_output_across_args_constraint(self):
        """CONSISTENCY (Category 2): For multi-argument operations, all arguments
        agree on the same output placement.

        ∀i,o: Σ_j x_{i,0,o,j} = Σ_j x_{i,1,o,j} = ... = Σ_j x_{i,A_i-1,o,j}
        """
        for node in self.graph.nodes:
            if node.op != "call_function":
                continue
            if node not in self.node_map:
                continue
            node_idx = self.node_map[node]
            if node_idx in self._cluster_linked_node_idxs:
                continue
            if len(self._all_input_nodes(node)) <= 1:
                continue
            vars_per_output = {}
            for argi, out_idx, inp_idx in self.walk_over_options(node):
                key = (node_idx, argi, out_idx, inp_idx)
                var = self.pulp_variables[key]
                vars_per_output.setdefault((argi, out_idx), []).append(var)
            eqs_per_arg = [[] for _ in self._all_input_nodes(node)]
            for (argi, out_idx), value in vars_per_output.items():
                eqs_per_arg[argi].append(pulp.lpSum(value))
            arg0 = eqs_per_arg[0]
            for arg_eqs in eqs_per_arg[1:]:
                assert len(arg0) == len(arg_eqs)
                for i in range(len(arg0)):
                    self.prob += (
                        arg0[i] == arg_eqs[i],
                        self._get_next_name("same_across_args"),
                    )

    def add_output_input_consistent_constraint(self):
        """FLOW (Category 3): The output placement of producer operations matches
        the input placement of consumer operations.

        ∀(i→k): Σ_j x_{i,0,o,j} = Σ_j x_{k,a,j,o}
        """
        for node in self.graph.nodes:
            if node.op == "output":
                continue
            if node not in self.node_map:
                continue
            node_idx = self.node_map[node]
            producer_is_linked = node_idx in self._cluster_linked_node_idxs
            # All args agree on the same output (ensured by consistency constraint),
            # so we use arg 0 for the producer side.
            for user in node.users:
                if user.op == "output":
                    continue
                if user not in self.node_map:
                    continue
                user_idx = self.node_map[user]
                # Skip edges where both endpoints are cluster-linked;
                # the root-to-root edge already covers this.
                if producer_is_linked and user_idx in self._cluster_linked_node_idxs:
                    continue
                user_argi = [
                    i for i, n in enumerate(self._all_input_nodes(user)) if n == node
                ]
                assert len(user_argi) >= 1
                # Use the first matching arg; the same-output-across-args
                # constraint already ensures all args agree.
                user_argi = user_argi[0]

                vars_producer = self._collect_vars(
                    node, node_idx, argi=0, group_by="out_idx"
                )
                if not vars_producer:
                    vars_producer = self._collect_vars(
                        node,
                        node_idx,
                        argi=0,
                        group_by="out_idx",
                        resolve_clusters=True,
                    )

                vars_consumer = self._collect_vars(
                    user, user_idx, argi=user_argi, group_by="inp_idx"
                )
                if not vars_consumer:
                    vars_consumer = self._collect_vars(
                        user,
                        user_idx,
                        argi=user_argi,
                        group_by="inp_idx",
                        resolve_clusters=True,
                    )

                # Skip edges where the consumer arg has no sharding decision
                # (e.g. None input_specs for HOP SymInt args).
                if not vars_consumer or not vars_producer:
                    user_strat = self.strats[user]
                    assert (
                        user_argi < len(user_strat.strategies[0].input_specs)
                        and user_strat.strategies[0].input_specs[user_argi] is None
                    ), (
                        f"Missing variables for non-None input_spec at "
                        f"{user}[{user_argi}]"
                    )
                    continue

                assert (
                    vars_producer.keys() == vars_consumer.keys()
                ), f"{vars_producer}, {vars_consumer}"

                for k in vars_producer:
                    self.prob += (
                        pulp.lpSum(vars_producer[k]) == pulp.lpSum(vars_consumer[k]),
                        self._get_next_name("output_input_consistent"),
                    )

    def add_inf_cost_constraint(self):
        """COST (Category 4): Variables with infinite cost (invalid configurations)
        are forced to zero.

        ∀i,a,o,j: c_{i,a,o,j} = ∞ ⟹ x_{i,a,o,j} = 0
        """
        for key, dv in self.decision_vars.items():
            if not math.isfinite(dv.cost):
                dv.cost = 10000.0
                self.prob += (dv.var == 0, self._get_next_name("inf_cases"))

    def add_default_constraints(self):
        self.add_unique_decision_constraint()
        self.add_same_output_across_args_constraint()
        self.add_output_input_consistent_constraint()
        self.add_inf_cost_constraint()
        self.add_forward_backward_consistency_constraints()

    # ---- Prefetch overlap ----

    def apply_prefetch_discount(self, scale=0.0):
        """Discount communication costs for prefetchable edges.

        Scales down comm costs for:
        - Forward: edges where the producer is parameter-derived (FSDP
          all-gathers that can be prefetched ahead of compute)
        - Backward: edges into terminal-derived nodes (gradient
          reduce-scatters that can overlap with earlier backward compute)

        Must be called before get_solution(). A scale of 0.0 means fully
        overlapped (free); 1.0 means no discount.

        Returns the number of decision vars modified.
        """
        param_derived = build_param_derived_set(self.graph)
        terminal_derived = build_terminal_derived_set(self.graph)

        logger.debug(
            "terminal_derived: %d param nodes, %d grad nodes",
            len(get_param_nodes(self.graph)),
            sum(
                1
                for _, grad in get_param_and_grad_nodes(self.graph).values()
                if grad is not None
            ),
        )

        n_modified = 0
        for key, dv in self.decision_vars.items():
            node_idx, argi, out_idx, inp_idx = key
            node = self.nodes[node_idx]

            input_nodes = self._all_input_nodes(node)
            if not input_nodes:
                continue
            producer = input_nodes[argi]

            is_prefetchable = producer in param_derived or node in terminal_derived
            if is_prefetchable and dv.comm_cost > 0 and math.isfinite(dv.comm_cost):
                dv.comm_cost *= scale
                dv.cost = dv.comm_cost + dv.compute_cost + dv.sharding_transition_cost
                n_modified += 1

        logger.debug(
            "apply_prefetch_discount(scale=%.2f): modified %d decision vars "
            "(%d param-derived nodes, %d terminal-derived nodes)",
            scale,
            n_modified,
            len(param_derived),
            len(terminal_derived),
        )
        return n_modified

    # ---- Solution ----

    def _set_objective(self):
        """Add the cost minimization objective to the ILP."""
        terms = []
        for key, dv in self.decision_vars.items():
            multiplier = 1 + len(self._root_to_linked.get(key, []))
            terms.append(dv.var * dv.cost * multiplier)
        self.prob += pulp.lpSum(terms)

    def _solve(self, verbose=False):
        solver = pulp.PULP_CBC_CMD(msg=verbose)
        # Use a dedicated temp directory for PuLP's intermediate files (.mps,
        # .sol, etc.) so they are always cleaned up, even if the process is
        # killed.  Without this, leftover files can fill up /tmp (tmpfs).
        with tempfile.TemporaryDirectory() as tmpdir:
            solver.tmpDir = tmpdir
            self.prob.solve(solver)

        self.selected_keys = [
            key for key, dv in self.decision_vars.items() if dv.var.value() == 1
        ]
        for root_key in list(self.selected_keys):
            self.selected_keys.extend(self._root_to_linked.get(root_key, []))

        if self.prob.status == -1:
            logger.warning(self.get_violated_constraints_log())
            raise RuntimeError(
                "The sharding optimizer could not find a feasible solution. "
                "This typically means the user-specified constraints are "
                "contradictory or the device mesh is too small for the requested "
                "sharding. Check the WARNING log above for the list of violated "
                "constraints, and consider relaxing input/output constraints or "
                "using a larger mesh."
            )

    def _extract_and_validate_solution(self):
        """Validate the ILP solution and return the optimal strategy per node."""
        selected_by_node = {}
        for key in self.selected_keys:
            node = self.nodes[key[0]]
            selected_by_node.setdefault(node, []).append(
                self._resolve_decision_var(key)
            )

        # Validate: each (node, arg) pair has exactly one selection
        seen = set()
        for key in self.selected_keys:
            node_arg = (key[0], key[1])
            if node_arg in seen:
                node = self.nodes[key[0]]
                raise RuntimeError(
                    f"Multiple solutions for {node}, key={node_arg}, "
                    f"solutions: {[str(dv.strategy) for dv in selected_by_node[node]]}"
                )
            seen.add(node_arg)

        # Validate: all args of a node agree on the same strategy
        for node, dvs in selected_by_node.items():
            assert all(
                dvs[0].strategy == dv.strategy for dv in dvs
            ), f"{[dv.var for dv in dvs]}: {[str(dv.strategy) for dv in dvs]}"

        return {node: dvs[0].strategy for node, dvs in selected_by_node.items()}

    def get_solution(self, verbose=False):
        t0 = time.perf_counter()
        self._set_objective()
        self._solve(verbose)
        obj_value = pulp.value(self.prob.objective)
        logger.debug(
            "ILP solve took %.3fs (objective=%.4f)", time.perf_counter() - t0, obj_value
        )
        return self._extract_and_validate_solution()

    def resolve(self, verbose=False):
        """Re-solve the ILP after adding or removing constraints.

        Unlike get_solution(), this does not re-set the objective, so it can
        be called multiple times after modifying constraints.
        """
        t0 = time.perf_counter()
        self._solve(verbose)
        obj_value = pulp.value(self.prob.objective)
        logger.debug(
            "ILP re-solve took %.3fs (objective=%.4f)",
            time.perf_counter() - t0,
            obj_value,
        )
        return self._extract_and_validate_solution()

    def remove_constraints(self, names):
        """Remove constraints by name, allowing re-solve to revert to the
        unconstrained optimum."""
        for name in names:
            del self.prob.constraints[name]

    def diff_solutions(self, solution_a, solution_b):
        """Compare two solutions and report placement changes and cost diffs.

        Args:
            solution_a: baseline solution dict (node → OpSpec), e.g. from get_solution()
            solution_b: counterfactual solution dict, e.g. from resolve()
        """
        from torch.distributed.tensor._op_schema import _pretty_print_spec

        changes = defaultdict(list)
        unchanged = 0

        for node in solution_a:
            if node not in solution_b:
                continue
            spec_a = solution_a[node].output_specs
            spec_b = solution_b[node].output_specs
            if not isinstance(spec_a, DTensorSpec) or not isinstance(
                spec_b, DTensorSpec
            ):
                continue
            if spec_a.placements == spec_b.placements:
                unchanged += 1
            else:
                key = (
                    _pretty_print_spec(spec_a),
                    _pretty_print_spec(spec_b),
                )
                changes[key].append(node)

        # Compute objective values from selected_keys for current solution
        # (solution_b is the current state after last solve)
        cost_a = self._compute_solution_cost(solution_a)
        cost_b = self._compute_solution_cost(solution_b)

        lines = []
        lines.append(
            f"Objective: {cost_a['total']:.1f} -> {cost_b['total']:.1f} "
            f"({cost_b['total'] - cost_a['total']:+.1f})"
        )
        lines.append(
            f"  compute: {cost_a['compute']:.1f} -> {cost_b['compute']:.1f} "
            f"({cost_b['compute'] - cost_a['compute']:+.1f})"
        )
        lines.append(
            f"  comm:    {cost_a['comm']:.1f} -> {cost_b['comm']:.1f} "
            f"({cost_b['comm'] - cost_a['comm']:+.1f})"
        )
        lines.append(
            f"  trans:   {cost_a['transition']:.1f} -> {cost_b['transition']:.1f} "
            f"({cost_b['transition'] - cost_a['transition']:+.1f})"
        )
        lines.append("")
        lines.append(
            f"Placement changes ({sum(len(v) for v in changes.values())} nodes "
            f"changed, {unchanged} unchanged):"
        )
        for (old, new), nodes in sorted(changes.items(), key=lambda x: -len(x[1])):
            lines.append(f"  {old} -> {new}: {len(nodes)} nodes")

        result = "\n".join(lines)
        print(result)
        return result

    def _compute_solution_cost(self, solution):
        """Compute the total cost breakdown for a given solution."""
        total_compute = 0.0
        total_comm = 0.0
        total_transition = 0.0

        for node, strategy in solution.items():
            out_idx = None
            for i, s in enumerate(self.strats[node].strategies):
                if s is strategy:
                    out_idx = i
                    break
            if out_idx is None:
                continue

            compute_cost = estimate_strategy_runtime_cost(node, strategy)
            total_compute += compute_cost

            input_nodes = self._all_input_nodes(node)
            for argi, pred in enumerate(input_nodes):
                pred_strategy = solution.get(pred)
                if pred_strategy is None:
                    continue
                pred_out_idx = None
                for i, s in enumerate(self.strats[pred].strategies):
                    if s is pred_strategy:
                        pred_out_idx = i
                        break
                if pred_out_idx is None:
                    continue
                comm = strategy.redistribute_cost[argi][pred_out_idx]
                if math.isfinite(comm):
                    total_comm += comm

                # Transition cost
                pred_spec = pred_strategy.output_specs
                input_spec = strategy.input_specs[argi]
                if (
                    isinstance(pred_spec, DTensorSpec)
                    and isinstance(input_spec, DTensorSpec)
                    and pred_spec.placements != input_spec.placements
                ):
                    total_transition += 1

        return {
            "total": total_compute + total_comm + total_transition,
            "compute": total_compute,
            "comm": total_comm,
            "transition": total_transition,
        }

    # ---- Logging ----

    def get_violated_constraints_log(self):
        violated_constraints = [
            (k, c) for k, c in self.prob.constraints.items() if not c.valid()
        ]
        log_str = f"Violated constraints: {[x[0] for x in violated_constraints]}"
        for cname, c in violated_constraints:
            log_str += f"\n========= {cname} ============="
            for cc, v in c.items():
                log_str += f"\n{cc}, coeff={v}, value={cc.value()}"
        return log_str

    def get_log(self, colored=False, verbose=False):
        from autoparallel.log_formatting import format_sharding_log

        selected_by_node = {}
        for key in self.selected_keys:
            node = self.nodes[key[0]]
            selected_by_node.setdefault(node, []).append(
                self._resolve_decision_var(key)
            )

        return format_sharding_log(
            graph=self.graph,
            opt=selected_by_node,
            colored=colored,
            verbose=verbose,
            violated_constraints_log=self.get_violated_constraints_log(),
        )

    def print_costs_for_node(self, node, arg=0, **kwargs):
        from tabulate import tabulate  # type: ignore
        from torch.distributed.tensor._op_schema import _pretty_print_spec

        tgt_strat = self.strats[node]
        src_strat = self.strats[self._all_input_nodes(node)[arg]]
        src_placements = [""] + [
            _pretty_print_spec(x.output_specs) for x in src_strat.strategies
        ]
        costs = [[str(x)] + x.redistribute_cost[arg] for x in tgt_strat.strategies]
        print(tabulate(costs, headers=src_placements, **kwargs))

    def explain_placement(self, target_placement, start_node=None, max_nodes=None):
        """Diagnose why the optimizer didn't choose a specific placement.

        Walks the graph in topological order and, for each node, reports whether
        target_placement is available, what its compute cost would be compared
        to the chosen placement, and flags the first node where the target
        becomes unavailable.

        Shows both compute and communication costs. Communication cost for the
        target assumes all predecessors also use target_placement (i.e., "what
        if the entire graph used this placement?").

        Must be called after get_solution().

        Args:
            target_placement: tuple of Placement objects, e.g. (Shard(0), Replicate())
            start_node: optional node to begin from (default: first graph node)
            max_nodes: optional limit on number of nodes to report
        """
        from tabulate import tabulate  # type: ignore
        from torch.distributed.tensor._op_schema import _pretty_print_spec

        # Build chosen out_idx per node from the solution
        chosen_out_idx = {}
        for key in self.selected_keys:
            node_idx, argi, out_idx, inp_idx = key
            if argi == 0:
                chosen_out_idx[node_idx] = out_idx

        target_placement = tuple(target_placement)
        target_str = _pretty_print_spec(DTensorSpec(self.mesh, target_placement))

        # Build target_out_idx per node: which strategy index produces
        # target_placement, if any.
        target_out_idx_map = {}
        for ni, node in enumerate(self.nodes):
            if node.op == "output":
                continue
            op_strategy = self.strats.get(node)
            if op_strategy is None:
                continue
            for i, strategy in enumerate(op_strategy.strategies):
                spec = strategy.output_specs
                if (
                    isinstance(spec, DTensorSpec)
                    and spec.placements == target_placement
                ):
                    target_out_idx_map[ni] = i
                    break

        started = start_node is None
        rows = []
        count = 0

        for node_idx, node in enumerate(self.nodes):
            if node.op == "output":
                continue

            if not started:
                if node is start_node:
                    started = True
                else:
                    continue

            if max_nodes is not None and count >= max_nodes:
                break

            op_strategy = self.strats[node]
            out_idx = chosen_out_idx.get(node_idx)
            if out_idx is None:
                continue

            chosen_strategy = op_strategy.strategies[out_idx]
            chosen_spec = chosen_strategy.output_specs

            # Skip multi-output nodes (tuples)
            if not isinstance(chosen_spec, DTensorSpec):
                continue

            chosen_placement_str = _pretty_print_spec(chosen_spec)
            chosen_compute = estimate_strategy_runtime_cost(node, chosen_strategy)

            # Compute communication cost for the chosen strategy: sum
            # redistribute_cost across all args using predecessors' chosen
            # out_idx.
            input_nodes = self._all_input_nodes(node)
            chosen_comm = 0.0
            for argi, pred in enumerate(input_nodes):
                pred_idx = self.node_map[pred]
                pred_out = chosen_out_idx.get(pred_idx, 0)
                chosen_comm += chosen_strategy.redistribute_cost[argi][pred_out]

            # Get output shape for display
            shape = chosen_spec.tensor_meta.shape if chosen_spec.tensor_meta else "?"
            shape_str = str(list(shape)) if shape != "?" else "?"

            # Search for target placement among available strategies
            target_out_idx = target_out_idx_map.get(node_idx)

            if target_out_idx is not None:
                target_strategy = op_strategy.strategies[target_out_idx]
                target_compute = estimate_strategy_runtime_cost(node, target_strategy)

                # Compute communication cost for the target strategy:
                # assume each predecessor also uses target_placement. If a
                # predecessor doesn't have it, fall back to its chosen placement.
                target_comm = 0.0
                for argi, pred in enumerate(input_nodes):
                    pred_idx = self.node_map[pred]
                    pred_tgt_out = target_out_idx_map.get(
                        pred_idx, chosen_out_idx.get(pred_idx, 0)
                    )
                    target_comm += target_strategy.redistribute_cost[argi][pred_tgt_out]

                chosen_total = chosen_compute + chosen_comm
                target_total = target_compute + target_comm

                # Determine status based on total cost
                if chosen_spec.placements == target_placement:
                    status = "CHOSEN"
                elif abs(chosen_total - target_total) < 0.01:
                    status = "TIE"
                elif target_total < chosen_total:
                    status = "TARGET CHEAPER"
                else:
                    status = "CHOSEN CHEAPER"

                # Flag kernel launch floor on compute
                if (
                    abs(target_compute - 7.0) < 0.01
                    and abs(chosen_compute - 7.0) < 0.01
                ):
                    status += " [floor]"

                target_compute_str = f"{target_compute:.1f}"
                target_comm_str = f"{target_comm:.1f}"
            else:
                target_compute_str = "N/A"
                target_comm_str = "N/A"
                status = "UNAVAILABLE"

            rows.append(
                [
                    str(node),
                    shape_str,
                    chosen_placement_str,
                    f"{chosen_compute:.1f}",
                    f"{chosen_comm:.1f}",
                    target_compute_str,
                    target_comm_str,
                    status,
                ]
            )
            count += 1

        headers = [
            "Node",
            "Shape",
            "Chosen",
            "Ch.Comp",
            "Ch.Comm",
            "Tgt.Comp",
            "Tgt.Comm",
            "Status",
        ]
        result = f"explain_placement: target={target_str}\n\n"
        result += tabulate(rows, headers=headers)
        print(result)
        return result

    # ---- User constraints ----

    def _add_node_constraint(
        self, node, output_constraint_indices, constraint_name=None
    ):
        if constraint_name is None:
            constraint_name = "user_constraint"
        node_idx = self.node_map[node]
        vars_per_arg = {}
        for argi, out_idx, inp_idx in self.walk_over_options(node):
            if out_idx in output_constraint_indices:
                var = self.pulp_variables[(node_idx, argi, out_idx, inp_idx)]
                vars_per_arg.setdefault(argi, []).append(var)
        names = []
        for eqs in vars_per_arg.values():
            name = self._get_next_name(constraint_name)
            self.prob += (pulp.lpSum(eqs) == 1, name)
            names.append(name)
        return names

    def _add_paired_output_constraint(self, node_a, node_b, constraint_name):
        """Constrains two nodes to have matching output placements.

        For each output strategy of node_a that also exists in node_b, adds:
            Σ_j x_{a, 0, oi_a, j} == Σ_j x_{b, 0, oi_b, j}

        where oi_a and oi_b are output strategy indices with the same placement.
        """
        idx_a = self.node_map[node_a]
        idx_b = self.node_map[node_b]
        strat_a = [str(s.output_specs) for s in self.strats[node_a].strategies]
        strat_b = [str(s.output_specs) for s in self.strats[node_b].strategies]
        num_inp_a = len(self.strats[node_a].strategies[0].redistribute_cost[0])
        num_inp_b = len(self.strats[node_b].strategies[0].redistribute_cost[0])
        for out_idx, sp in enumerate(strat_a):
            if sp not in strat_b:
                # This placement exists in node_a but not in node_b.
                # Disable it: force sum of its decision variables to 0.
                v_a = [
                    self.pulp_variables[(idx_a, 0, out_idx, inp_idx)]
                    for inp_idx in range(num_inp_a)
                ]
                self.prob += (
                    pulp.lpSum(v_a) == 0,
                    self._get_next_name(constraint_name + "_disable"),
                )
                continue
            out_idx_b = strat_b.index(sp)
            v_a = [
                self.pulp_variables[(idx_a, 0, out_idx, inp_idx)]
                for inp_idx in range(num_inp_a)
            ]
            v_b = [
                self.pulp_variables[(idx_b, 0, out_idx_b, inp_idx)]
                for inp_idx in range(num_inp_b)
            ]
            self.prob += (
                pulp.lpSum(v_b) == pulp.lpSum(v_a),
                self._get_next_name(constraint_name),
            )

    def add_forward_backward_consistency_constraints(self):
        """USER (Category 5c): Forward-backward consistency constraints.
        Ensures that paired forward/backward nodes have matching output placements
        for parameters, plain inputs, and plain outputs.

        Σ_j x_{fwd, 0, oi, j} = Σ_j x_{bwd, 0, oi', j}
        where oi and oi' have matching placements.
        """
        for param, grad in get_param_and_grad_nodes(self.graph).values():
            if grad is None:
                continue
            self._add_paired_output_constraint(param, grad, "grad_param_constraint")

        for node, grad_node in get_plain_input_and_grad_nodes(self.graph).values():
            if grad_node is None:
                continue
            self._add_paired_output_constraint(node, grad_node, "grad_input_constraint")

        for node, tangent_node in get_plain_output_and_tangent_nodes(
            self.graph
        ).values():
            if tangent_node is None:
                continue
            self._add_paired_output_constraint(
                node, tangent_node, "grad_output_constraint"
            )

    def add_parameter_memory_constraint(
        self, memory_factor_low: float, memory_factor_high: float
    ):
        """USER (Category 5b): Constrain total parameter memory usage.

        Σ_{params} (size_ratio * x_{param}) ≤ memory_limit
        """
        param_nodes: list[torch.fx.Node] = get_param_nodes(self.graph)
        elms: list[pulp.LpAffineExpression] = []
        budget_low: float = 0.0
        budget_high: float = 0.0
        for node in param_nodes:
            node_idx = self.node_map[node]
            num_out_strat = len(self.strats[node].strategies)
            ratios: list[float] = []
            for out_idx in range(num_out_strat):
                dv = self._resolve_decision_var((node_idx, 0, out_idx, 0))
                spec: DTensorSpec = dv.input_spec
                assert spec.tensor_meta is not None
                tensor_shape: torch.Size = spec.tensor_meta.shape
                new_tensor_shape, _ = _get_sharded_shape_stride(spec)
                new_size: int = math.prod(new_tensor_shape)
                old_size: int = math.prod(tensor_shape)
                ratio = new_size / old_size
                ratios.append(ratio)
                elms.append(dv.var * ratio)
            best_ratio: float = min(ratios)
            budget_low += max(best_ratio, memory_factor_low)
            budget_high += max(best_ratio, memory_factor_high)

        self.prob += (pulp.lpSum(elms) <= budget_high, "memory_constraint_high")
        self.prob += (pulp.lpSum(elms) >= budget_low, "memory_constraint_low")

    def add_node_constraint(self, node, placement=None, constraint_name=None):
        """USER (Category 5d): Force a specific placement for a node.

        For nodes with tuple output_specs (e.g. SDPA), the placement is matched
        against the first DTensorSpec element in the tuple.
        """
        assert node in self.strats, (node, self.strats.keys())
        strat = self.strats[node]
        if placement is None:
            # default is Shard(0) to parallelize on the batch
            placement = (Shard(0),) + (Replicate(),) * (self.mesh.ndim - 1)
        output_constraint_indices = []
        for i, s in enumerate(strat.strategies):
            specs = s.output_specs
            if isinstance(specs, DTensorSpec):
                if specs.placements == placement:
                    output_constraint_indices.append(i)
            elif isinstance(specs, (list, tuple)):
                for spec in specs:
                    if isinstance(spec, DTensorSpec):
                        if spec.placements == placement:
                            output_constraint_indices.append(i)
                        break
        if len(output_constraint_indices) == 0:
            raise RuntimeError(
                f"Couldn't find appropriate constraint {node} {constraint_name} {placement}"
            )
        return self._add_node_constraint(
            node,
            output_constraint_indices=output_constraint_indices,
            constraint_name=constraint_name,
        )

    def _add_io_placement_constraints(
        self,
        nodes_dict,
        placements,
        desc_type,
        constraint_name,
        error_message,
    ):
        """Shared implementation for add_sharded_input_constraint and
        add_sharded_output_constraint. Only constrains the forward-side node;
        the backward side is handled by add_forward_backward_consistency_constraints."""
        remaining = None
        if placements is not None:
            remaining = {i: p for i, p in enumerate(placements)}

        for desc, (node, _companion_node) in nodes_dict.items():
            if placements is None:
                placement = None
            else:
                assert isinstance(desc, desc_type)
                assert remaining is not None
                placement = remaining.pop(desc.idx)

            self.add_node_constraint(node, placement, constraint_name=constraint_name)

        ignored = []
        if remaining is not None:
            for i, p in remaining.items():
                if p is not None:
                    ignored.append(i)

        if ignored:
            raise RuntimeError(error_message.format(indices=ignored))

    def add_sharded_input_constraint(
        self, input_placements: Optional[list[Optional[tuple[Placement, ...]]]] = None
    ):
        """USER (Category 5a): Force specific placements for input nodes.
        The corresponding gradient inputs are automatically constrained via
        add_forward_backward_consistency_constraints()."""
        self._add_io_placement_constraints(
            nodes_dict=get_plain_input_and_grad_nodes(self.graph),
            placements=input_placements,
            desc_type=PlainAOTInput,
            constraint_name="input_constraint",
            error_message=(
                "We were unable to respect placements for inputs at indices {indices}.  "
                "This is because the traced joint graph did not actually have a dedicated "
                "placeholder node for these inputs.  "
                "This typically occurs because some inputs aliased each other; inspect the "
                "joint graph from tlparse for more details.  "
                "You can either remove an explicit placement for this input (replace it with "
                "None) or clone the inputs before tracing to remove aliasing."
            ),
        )

    def add_sharded_output_constraint(self, output_placements=None):
        """USER (Category 5a): Force specific placements for output nodes.
        The corresponding tangent/gradient nodes are automatically constrained via
        add_forward_backward_consistency_constraints()."""
        self._add_io_placement_constraints(
            nodes_dict=get_plain_output_and_tangent_nodes(self.graph),
            placements=output_placements,
            desc_type=PlainAOTOutput,
            constraint_name="output_constraint",
            error_message=(
                "We were unable to respect placements for outputs at indices {indices}.  "
                "This is because the traced joint graph did not actually have a dedicated "
                "output node for these inputs.  "
                "This typically occurs because some outputs aliased each other; inspect the "
                "joint graph from tlparse for more details.  "
                "You can either remove an explicit placement for this output (replace it with "
                "None), stop the model from returning aliases of the tensor or clone the "
                "outputs before returning them from the graph to avoid aliasing."
            ),
        )
