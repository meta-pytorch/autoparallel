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

   5c. Parameter-gradient consistency: x_{param} = x_{grad_param}
       → Implemented in: add_grad_param_constraints()

   5d. General node constraints: Force specific placement for any node
       → Implemented in: add_node_constraint()

The solver finds the globally optimal sharding strategy that minimizes total
runtime cost while satisfying all constraints.
"""

import math
import operator
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
from torch.utils._pytree import tree_flatten, tree_map_only

from .cost_models.collective_runtime_estimation import estimate_strategy_comms_cost
from .cost_models.compute_estimation import (
    _get_sharded_shape_stride,
    estimate_strategy_runtime_cost,
)
from .graph_passes.graph_clustering import get_identical_regions
from .shardings.placement_options import (
    get_local_map_placement_option,
    get_placement_options,
)
from .shardings.propagation_rules import _create_all_options


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
        self.nodes = list(self.graph.nodes)
        self.mesh = mesh
        self.rescale_grad_comm_cost_for_mp = rescale_grad_comm_cost_for_mp
        self.node_map = {node: i for i, node in enumerate(self.graph.nodes)}
        self._name_counters: dict[str, int] = {}
        self.strats = self.build_sharding_metadata()

        self.cluster_links: dict[tuple, tuple] = {}
        if repeated_subgraphs:
            t = time.time()
            clusters = get_identical_regions(self.gm.graph, self.strats)
            print(f"Found {len(clusters)} clusters in {time.time() - t:.2f}s")
            self.create_cluster_links(clusters)

        self.decision_vars = self._build_decision_vars()
        self.validate()
        self.prob = pulp.LpProblem("AutoParallel", pulp.LpMinimize)
        self.add_default_constraints()

    def _get_next_name(self, prefix):
        idx = self._name_counters.setdefault(prefix, 0)
        self._name_counters[prefix] += 1
        return prefix + f"_{idx:03}"

    def build_sharding_metadata(self):
        strats = {}
        for node in self.graph.nodes:
            if node.op == "placeholder":
                strats[node] = _create_all_options(
                    self.mesh, node.meta["val"].shape, tensor=node.meta["val"]
                )
            elif node.op == "call_function":
                # TODO: kwargs?
                user_strats = tree_map_only(
                    torch.fx.Node, lambda x: strats[x], node.args
                )
                user_args = tree_map_only(
                    torch.fx.Node, lambda x: x.meta["val"], node.args
                )
                user_kwargs = tree_map_only(
                    torch.fx.Node, lambda x: x.meta["val"], node.kwargs
                )
                if local_map_kwargs := node.meta.get("local_map_kwargs", {}):
                    assert local_map_kwargs["in_placements"] is not None
                    assert local_map_kwargs["out_placements"] is not None
                    assert (
                        local_map_kwargs.get("in_grad_placements", None) is None
                    ), "Not yet implemented"
                    assert local_map_kwargs.get("device_mesh", None) in (
                        self.mesh,
                        None,
                    ), "Not yet implemented"
                    assert not user_kwargs
                    # TODO: get rid of this when HOP can install as a subgraph
                    assert "call_local_map" in str(
                        node.target
                    ) or "call_local_map_backward" in str(node.target)
                    strat = get_local_map_placement_option(
                        self.mesh,
                        user_strats,
                        user_args,
                        node,
                        local_map_kwargs["in_placements"],
                        local_map_kwargs["out_placements"],
                    )
                else:
                    strat = get_placement_options(
                        self.mesh, node.target, user_strats, user_args, user_kwargs
                    )
                strats[node] = strat
            elif node.op == "get_attr":
                strats[node] = _create_all_options(
                    self.mesh, node.meta["val"].shape, tensor=node.meta["val"]
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
        """Variant of node.all_input_nodes that preserves duplicate nodes."""
        # TODO: add kwargs?
        return [x for x in tree_flatten(node.args)[0] if isinstance(x, torch.fx.Node)]

    def walk_over_options(self, node, constrain_arg=None):
        """Yield (argi, out_idx, inp_idx) for all valid strategy combinations."""
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
        for node_idx, (node, _) in enumerate(self.strats.items()):
            if node.op == "output":
                continue
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
        pulp_variables = self._create_pulp_variables()
        grad_param_nodes = set(
            x[1] for x in get_param_and_grad_nodes(self.graph).values()
        )

        decision_vars = {}
        for node_idx, (node, op_strategy) in enumerate(self.strats.items()):
            if node.op == "output":
                continue

            all_input_nodes = self._all_input_nodes(node)
            num_args = len(op_strategy.strategies[0].input_specs)

            for out_idx, output_strategy in enumerate(op_strategy.strategies):
                compute_cost = estimate_strategy_runtime_cost(node, output_strategy)
                per_arg_compute = compute_cost / num_args

                for argi, redist_costs in enumerate(output_strategy.redistribute_cost):
                    producer_strategy = (
                        self.strats[all_input_nodes[argi]] if all_input_nodes else None
                    )
                    for inp_idx, default_comm_cost in enumerate(redist_costs):
                        comm_cost, transition_cost = self._compute_edge_costs(
                            node,
                            output_strategy,
                            argi,
                            inp_idx,
                            default_comm_cost,
                            producer_strategy,
                            grad_param_nodes,
                        )
                        # Update OpSpec redistribute_cost so print_costs_for_node
                        # reflects the recomputed costs
                        redist_costs[inp_idx] = comm_cost

                        key = (node_idx, argi, out_idx, inp_idx)
                        decision_vars[key] = DecisionVar(
                            var=pulp_variables[key],
                            cost=comm_cost + per_arg_compute + transition_cost,
                            compute_cost=per_arg_compute,
                            comm_cost=comm_cost,
                            sharding_transition_cost=transition_cost,
                            strategy=output_strategy,
                            output_spec=output_strategy.output_specs,
                            input_spec=output_strategy.input_specs[argi],
                        )

        return decision_vars

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
                var = self.decision_vars[self.cluster_links[key]].var
            else:
                var = self.decision_vars[key].var
            group_key = out_idx if group_by == "out_idx" else inp_idx
            result.setdefault(group_key, []).append(var)
        return result

    def validate(self):
        for node in self.graph.nodes:
            if node.op != "call_function":
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
        for node_idx, node in enumerate(self.graph.nodes):
            if node.op not in {"placeholder", "call_function", "get_attr"}:
                continue
            arg_vars = {}
            for argi, out_idx, inp_idx in self.walk_over_options(node):
                key = (node_idx, argi, out_idx, inp_idx)
                if key in self.cluster_links:
                    continue
                var = self.decision_vars[key].var
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
        for node_idx, node in enumerate(self.graph.nodes):
            if node.op != "call_function":
                continue
            if len(self._all_input_nodes(node)) <= 1:
                continue
            vars_per_output = {}
            for argi, out_idx, inp_idx in self.walk_over_options(node):
                key = (node_idx, argi, out_idx, inp_idx)
                if key in self.cluster_links:
                    continue
                var = self.decision_vars[key].var
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
        for node_idx, node in enumerate(self.graph.nodes):
            if node.op == "output":
                continue
            # All args agree on the same output (ensured by consistency constraint),
            # so we use arg 0 for the producer side.
            for user in node.users:
                if user.op == "output":
                    continue
                user_idx = self.node_map[user]
                user_argi = [i for i, n in enumerate(user.all_input_nodes) if n == node]
                assert len(user_argi) == 1
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
                if key in self.cluster_links:
                    continue
                self.prob += (dv.var == 0, self._get_next_name("inf_cases"))

    def add_default_constraints(self):
        self.add_unique_decision_constraint()
        self.add_same_output_across_args_constraint()
        self.add_output_input_consistent_constraint()
        self.add_inf_cost_constraint()

    # ---- Solution ----

    def _set_objective(self):
        """Add the cost minimization objective to the ILP."""
        # Deduplicate variables that appear multiple times (from cluster links)
        cost_per_var = defaultdict(int)
        for dv in self.decision_vars.values():
            cost_per_var[dv.var] += dv.cost
        self.prob += pulp.lpSum([var * cost for var, cost in cost_per_var.items()])

    def _solve(self, verbose=False):
        solver = pulp.PULP_CBC_CMD(msg=verbose)
        self.prob.solve(solver)

        self.selected_keys = [
            key for key, dv in self.decision_vars.items() if dv.var.value() == 1
        ]

        if self.prob.status == -1:
            print(self.get_violated_constraints_log())
            raise RuntimeError("Unsolvable problem")

    def _extract_and_validate_solution(self):
        """Validate the ILP solution and return the optimal strategy per node."""
        selected_by_node = {}
        for key in self.selected_keys:
            node = self.nodes[key[0]]
            selected_by_node.setdefault(node, []).append(self.decision_vars[key])

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
        self._set_objective()
        self._solve(verbose)
        return self._extract_and_validate_solution()

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
            selected_by_node.setdefault(node, []).append(self.decision_vars[key])

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
                var = self.decision_vars[(node_idx, argi, out_idx, inp_idx)].var
                vars_per_arg.setdefault(argi, []).append(var)
        for eqs in vars_per_arg.values():
            self.prob += (
                pulp.lpSum(eqs) == 1,
                self._get_next_name(constraint_name),
            )

    def add_grad_param_constraints(self):
        """USER (Category 5c): Ensure parameters and their gradients have matching
        sharding strategies.

        x_{param} = x_{grad_param}
        """
        for param, grad in get_param_and_grad_nodes(self.graph).values():
            if grad is None:
                continue
            param_idx = self.node_map[param]
            grad_idx = self.node_map[grad]
            param_strats = self.strats[param]
            grad_strats = self.strats[grad]
            num_out_strat = len(param_strats.strategies)
            num_inp_g_strat = len(grad_strats.strategies[0].redistribute_cost[0])

            strat_p = [str(strat.output_specs) for strat in param_strats.strategies]
            strat_gp = [str(strat.output_specs) for strat in grad_strats.strategies]
            for out_idx in range(num_out_strat):
                v_p = self.decision_vars[(param_idx, 0, out_idx, 0)].var
                sp = strat_p[out_idx]
                # TODO: fix this case
                if sp not in strat_gp:
                    continue
                grad_out_idx = strat_gp.index(sp)
                v_gp = []
                for inp_idx in range(num_inp_g_strat):
                    v_gp.append(
                        self.decision_vars[(grad_idx, 0, grad_out_idx, inp_idx)].var
                    )
                self.prob += (
                    pulp.lpSum(v_gp) == v_p,
                    self._get_next_name("grad_param_constraint"),
                )

    def add_parameter_memory_constraint(
        self, memory_factor_low: float, memory_factor_high: float
    ):
        """USER (Category 5b): Constrain total parameter memory usage.

        Σ_{params} (size_ratio * x_{param}) ≤ memory_limit
        """
        param_nodes: list[torch.fx.Node] = get_param_nodes(self.graph)
        elms: list[pulp.LpAffineExpression] = []
        num_params_to_consider: int = 0
        world_size: int = math.prod(self.mesh.shape)
        for node in param_nodes:
            node_idx = self.node_map[node]
            can_be_fully_sharded: bool = node.meta["val"].numel() >= world_size
            num_params_to_consider += int(can_be_fully_sharded)
            if not can_be_fully_sharded:
                continue
            num_out_strat = len(self.strats[node].strategies)
            for out_idx in range(num_out_strat):
                dv = self.decision_vars[(node_idx, 0, out_idx, 0)]
                spec: DTensorSpec = dv.input_spec
                assert spec.tensor_meta is not None
                tensor_shape: torch.Size = spec.tensor_meta.shape
                new_tensor_shape, _ = _get_sharded_shape_stride(spec)
                new_size: int = math.prod(new_tensor_shape)
                old_size: int = math.prod(tensor_shape)
                elms.append(dv.var * new_size / old_size)

        memory_factor_low *= num_params_to_consider
        memory_factor_high *= num_params_to_consider
        self.prob += (pulp.lpSum(elms) <= memory_factor_high, "memory_constraint_high")
        self.prob += (pulp.lpSum(elms) >= memory_factor_low, "memory_constraint_low")

    def add_node_constraint(self, node, placement=None, constraint_name=None):
        """USER (Category 5d): Force a specific placement for a node."""
        assert node in self.strats, (node, self.strats.keys())
        strat = self.strats[node]
        if placement is None:
            # default is Shard(0) to parallelize on the batch
            placement = (Shard(0),) + (Replicate(),) * (self.mesh.ndim - 1)
        output_constraint_indices = [
            i
            for i, s in enumerate(strat.strategies)
            if s.output_specs.placements == placement
        ]
        if len(output_constraint_indices) == 0:
            raise RuntimeError(
                f"Couldn't find appropriate constraint {node} {constraint_name} {placement}"
            )
        self._add_node_constraint(
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
        grad_constraint_name,
        error_message,
    ):
        """Shared implementation for add_sharded_input_constraint and
        add_sharded_output_constraint."""
        remaining = None
        if placements is not None:
            remaining = {i: p for i, p in enumerate(placements)}

        for desc, (node, companion_node) in nodes_dict.items():
            if placements is None:
                placement = None
            else:
                assert isinstance(desc, desc_type)
                assert remaining is not None
                placement = remaining.pop(desc.idx)

            self.add_node_constraint(node, placement, constraint_name=constraint_name)
            if companion_node is not None:
                self.add_node_constraint(
                    companion_node, placement, constraint_name=grad_constraint_name
                )

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
        """USER (Category 5a): Force specific placements for input nodes and
        their corresponding gradient inputs."""
        self._add_io_placement_constraints(
            nodes_dict=get_plain_input_and_grad_nodes(self.graph),
            placements=input_placements,
            desc_type=PlainAOTInput,
            constraint_name="input_constraint",
            grad_constraint_name="grad_input_constraint",
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
        """USER (Category 5a): Force specific placements for output nodes and
        their corresponding gradient outputs."""
        self._add_io_placement_constraints(
            nodes_dict=get_plain_output_and_tangent_nodes(self.graph),
            placements=output_placements,
            desc_type=PlainAOTOutput,
            constraint_name="output_constraint",
            grad_constraint_name="grad_output_constraint",
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
