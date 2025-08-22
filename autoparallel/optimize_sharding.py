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
   forced to zero.

   ∀i,a,o,j: c_{i,a,o,j} = ∞ ⟹ x_{i,a,o,j} = 0

   → Implemented in: add_inf_cost_constraint()

5. EFFICIENCY CONSTRAINTS: Penalize inefficient collective operations like
   non-batch dimension shard-to-replicate conversions and forbid invalid
   transitions like replicate-to-partial.

   - Shard(dim≠0) → Replicate: multiply cost by 4
   - Replicate → Partial: x_{i,a,o,j} = 0 (forbidden)
   - Partial → Shard(dim≠0): multiply cost by 4

   → Implemented in: penalize_inefficient_collectives()

6. USER CONSTRAINTS (optional): Force specific placements for inputs, outputs,
   parameters, or memory usage bounds.

   6a. Input/Output constraints: x_{i,a,o*,j*} = 1 for specified (o*,j*)
       → Implemented in: add_sharded_input_constraint(), add_sharded_output_constraint()

   6b. Memory constraints: Σ_{params} (size_ratio * x_{param}) ≤ memory_limit
       → Implemented in: add_parameter_memory_constraint()

   6c. Parameter-gradient consistency: x_{param} = x_{grad_param}
       → Implemented in: add_grad_param_constraints()

   6d. General node constraints: Force specific placement for any node
       → Implemented in: add_node_constraint()

The solver finds the globally optimal sharding strategy that minimizes total
runtime cost while satisfying all constraints.
"""

import math
import operator
import time
from collections import defaultdict
from typing import Optional

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

from .compute_estimation import (
    _get_sharded_shape_stride,
    estimate_strategy_runtime_cost,
)
from .graph_clustering import get_identical_regions
from .propagation_rules import _create_all_options
from .utils import get_local_map_placement_option, get_placement_options


def _debug_node(node):
    def my_print(x):
        x = x.meta["val"]
        if isinstance(x, torch.Tensor):
            return (x.shape, x.stride())
        return x

    print(tree_map_only(torch.fx.Node, my_print, node.args))


_GLOBAL_NAMES: dict[str, int] = {}


def _get_next_name(name):
    global _GLOBAL_NAMES  # noqa: F824
    idx = _GLOBAL_NAMES.setdefault(name, 0)
    _GLOBAL_NAMES[name] += 1
    return name + f"_{idx:03}"


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
        self.strats = self.build_sharding_metadata()

        self.cluster_links = {}
        if repeated_subgraphs:
            t = time.time()
            clusters = get_identical_regions(self.gm.graph, self.strats)
            print(f"Found {len(clusters)} clusters in {time.time() - t:.2f}s")
            self.create_cluster_links(clusters)

        # ds: Decision variables dictionary mapping (s_i, argi, ss, ii) -> ILP variable data
        # Each key represents a choice of input placement ii and output placement ss
        # for operation s_i and argument argi (corresponds to x_{i,a,o,j} in math notation)
        self.ds, self.num_inp_out, self.num_args = self.build_ds()
        self.validate()
        self.prob = pulp.LpProblem("AutoParallel", pulp.LpMinimize)
        self.add_default_constraints()

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
                if local_map_kwargs := node.meta.get("custom", {}).get(
                    "dtensor_local_map_kwargs"
                ):
                    assert "call_local_map" in str(node.target)
                    assert not user_kwargs
                    strat = get_local_map_placement_option(
                        self.mesh,
                        user_strats,
                        user_args,
                        node.meta["val"],
                        local_map_kwargs["in_placements"],
                        local_map_kwargs["out_placements"],
                    )

                    assert not node.kwargs
                    node.kwargs = {
                        "_inline": True
                    }  # notify the HOP to desugar in the next trace

                    strats[node] = strat
                else:
                    strat = get_placement_options(
                        self.mesh, node.target, user_strats, user_args, user_kwargs
                    )
                    strats[node] = strat
            elif node.op == "output":
                user_strats = tree_map_only(
                    torch.fx.Node, lambda x: strats[x], node.args
                )
                strats[node] = user_strats
            else:
                raise ValueError(f"Oups {node.op}")
        return strats

    def create_cluster_links(self, clusters):
        """
        This function creates a mapping between optimization nodes that are
        identical. We use this to reduce the optimization space.
        If cluster_links[key1] == key2, this means that everywhere in the
        optimization problem we will be using key2 instead.
        """
        for cluster_group in clusters:
            cluster0 = cluster_group[0]
            for cluster_i in cluster_group[1:]:
                for n0, ni in zip(cluster0, cluster_i):
                    s0 = self.node_map[n0]
                    s1 = self.node_map[ni]
                    for argi, oi, ii in self.walk_over_options(n0):
                        self.cluster_links[(s1, argi, oi, ii)] = (s0, argi, oi, ii)

    def _build_pulp_variable(self, key, ds):
        """
        This function creates the PuLP optimization variable, taking into
        consideration that if there are identical nodes (defined by cluster_links)
        then we will reuse the optimization variable.
        """
        s_i, argi, ss, ii = key
        node = self.nodes[s_i]
        if key in self.cluster_links and self.cluster_links[key] in ds:
            # if we already have a root variable created for a similar node, use it
            va = ds[self.cluster_links[key]]["va"]
        elif key in self.cluster_links and self.cluster_links[key] not in ds:
            # if we have a similar node for which the root variable has not been created yet,
            # create the new variable early and populate the ds with "va" field
            # the remaining fields will be populated when the root variable is created
            new_key = self.cluster_links[key]
            new_s_i = new_key[0]
            new_node = self.nodes[new_s_i]
            va = pulp.LpVariable(
                f"n={new_node},s={new_s_i},arg={argi},output_p={ss},input_p={ii}",
                cat=pulp.LpBinary,
            )
            ds[new_key] = {"va": va}
        elif key in ds:
            # if we are the root variable, make sure that we haven't yet
            # been initialized. This happens if we have been created before by
            # a similar node which isn't a root
            va = ds[key]["va"]
            assert "cost" not in ds[key]
        else:
            # we are a root variable which came first in the iteration order
            va = pulp.LpVariable(
                f"n={node},s={s_i},arg={argi},output_p={ss},input_p={ii}",
                cat=pulp.LpBinary,
            )
        return va

    def build_ds(self):
        """
        Build decision variables (ds) for the ILP optimization.

        Creates binary variables x_{i,a,o,j} for each valid combination of:
        - s_i: operation index
        - argi: argument index
        - ss: output placement strategy index (o in math notation)
        - ii: input placement strategy index (j in math notation)

        Returns:
            ds: Dictionary mapping (s_i, argi, ss, ii) -> {
                "va": PuLP binary variable,
                "cost": communication + computation cost,
                "full_strat": complete strategy object,
                "out_strat": output placement specification,
                "inp_strat": input placement specification
            }
            num_inp_out: Metadata about strategy counts per operation-argument
            num_args: Number of arguments per operation
        """
        strats = self.strats
        ds = {}
        num_inp_out = {}
        num_args = {}
        grad_param_nodes = set(
            x[1] for x in get_param_and_grad_nodes(self.graph).values()
        )
        sharding_transition_scale = 1
        for s_i, (node, s) in enumerate(strats.items()):
            if node.op == "output":
                continue
            num_args[s_i] = len(s.strategies[0].input_specs)
            for argi, argv in enumerate(s.strategies[0].redistribute_cost):
                num_inp_out[(s_i, argi)] = {
                    "num_input_strat": len(argv),
                    "num_output_strat": len(s.strategies),
                }
            for ss, ssi in enumerate(s.strategies):
                compute_cost = estimate_strategy_runtime_cost(node, ssi)
                for argi, xxi in enumerate(ssi.redistribute_cost):
                    if node.op != "placeholder":
                        argi_strat = self.strats[self._all_input_nodes(node)[argi]]
                    for ii, comm_cost in enumerate(xxi):
                        if node in grad_param_nodes:
                            comm_cost = comm_cost / self.rescale_grad_comm_cost_for_mp
                        # Imagine we start node_i from S(0)S(0) and we want to reach node_{i+2} at
                        # RR, and that node_{i+1} is an op with zero cost (like alias).
                        # In this case, all of the following chains yield the same cost:
                        # - S(0)S(0) -> S(0)R -> RR
                        # - S(0)S(0) -> RS(0) -> RR
                        # - S(0)S(0) -> RR -> RR
                        # - S(0)S(0) -> S(0)S(0) -> RR
                        # all of those cases are in principle equivalent. But if we want to
                        # optimize S(0)S(0) -> RR to dispatch into a single collective instead of two
                        # then we need to vafor the last two cases where redistribution happens
                        # in a single go. To do this, we add a tie-break cost that is 1 if a redistribution
                        # happens prior to getting to this configuration, and 0 otherwise. This way,
                        # we will favor having fewer redistributions happening in the graph.
                        if node.op != "placeholder" and node.target != operator.getitem:
                            original_placement = argi_strat.strategies[
                                ii
                            ].output_specs.placements
                            current_placement = ssi.input_specs[argi].placements
                            redistribution_happened = (
                                current_placement != original_placement
                            )
                            sharding_transition_cost = (
                                int(redistribution_happened) * sharding_transition_scale
                            )
                        else:
                            sharding_transition_cost = 0
                        key = (s_i, argi, ss, ii)
                        # NOTE: this modifies ds in-place sometimes
                        # we might want to refactor this in the future
                        va = self._build_pulp_variable(key, ds)
                        ds[key] = {
                            "va": va,
                            "cost": comm_cost
                            + compute_cost / num_args[s_i]
                            + sharding_transition_cost,
                            "compute_cost": compute_cost / num_args[s_i],
                            "comm_cost": comm_cost,
                            "sharding_transition_cost": sharding_transition_cost,
                            "full_strat": ssi,
                            "out_strat": ssi.output_specs,
                            "inp_strat": ssi.input_specs[argi],
                        }
        return ds, num_inp_out, num_args

    def _all_input_nodes(self, node):
        """
        Variant of node.all_input_nodes which preserve duplicate nodes
        """
        # TODO: add kwargs?
        return [x for x in tree_flatten(node.args)[0] if isinstance(x, torch.fx.Node)]

    def walk_over_options(self, node, constrain_arg=None):
        # TODO: use an iterator over node inputs, as done in penalize_inefficient_collectives
        # the problem is when nodes don't have input_nodes, like placeholders or
        # constructor functions. Doing it this way for now
        tgt_op_strat = self.strats[node]
        for argi in range(len(tgt_op_strat.strategies[0].input_specs)):
            if constrain_arg is not None:
                if argi != constrain_arg:
                    continue
            for oi, tgt_strat in enumerate(tgt_op_strat.strategies):
                for ii in range(len(tgt_strat.redistribute_cost[argi])):
                    yield argi, oi, ii

    def add_unique_decision_constraint(self):
        """
        UNIQUENESS CONSTRAINTS (Category 1): Each operation-argument pair must select exactly one
        input-output placement combination.

        Mathematical form: ∀i,a: Σ_{o,j} x_{i,a,o,j} = 1
        """
        # a single pair of input-output policy is chosen
        for s_i, node in enumerate(self.graph.nodes):
            if node.op not in {"placeholder", "call_function"}:
                continue
            arg_vars = {}
            for arg, oi, ii in self.walk_over_options(node):
                key = (s_i, arg, oi, ii)
                if key in self.cluster_links:
                    continue
                va = self.ds[key]["va"]
                arg_vars.setdefault(arg, []).append(va)
            for eqs in arg_vars.values():
                self.prob += (pulp.lpSum(eqs) == 1, _get_next_name("unique_decision"))

    def add_same_output_across_args_constraint(self):
        """
        CONSISTENCY CONSTRAINTS (Category 2): For multi-argument operations, all arguments must agree
        on the same output placement to ensure the operation can execute correctly.

        Mathematical form: ∀i,o: Σ_j x_{i,0,o,j} = Σ_j x_{i,1,o,j} = ... = Σ_j x_{i,A_i-1,o,j}
        """
        # enforce that the same output policy is chosen
        # across arguments
        for s_i, node in enumerate(self.graph.nodes):
            if node.op != "call_function":
                continue
            if len(self._all_input_nodes(node)) <= 1:
                continue
            vars_per_output = {}
            for arg, oi, ii in self.walk_over_options(node):
                key = (s_i, arg, oi, ii)
                if key in self.cluster_links:
                    continue
                va = self.ds[key]["va"]
                vars_per_output.setdefault((arg, oi), []).append(va)
            eqs_per_arg = [[] for _ in self._all_input_nodes(node)]
            for (arg, oi), value in vars_per_output.items():
                eqs_per_arg[arg].append(pulp.lpSum(value))
            arg0 = eqs_per_arg[0]
            for argi in eqs_per_arg[1:]:
                assert len(arg0) == len(argi)
                for i in range(len(arg0)):
                    self.prob += (
                        arg0[i] == argi[i],
                        _get_next_name("same_across_args"),
                    )

    def add_output_input_consistent_constraint(self):
        """
        FLOW CONSTRAINTS (Category 3): The output placement of producer operations must match the
        input placement of consumer operations (dataflow consistency).

        Mathematical form: ∀(i→k): Σ_j x_{i,0,o,j} = Σ_j x_{k,a,j,o}
        """
        # enforce that the input of strat_{i+1} == output of strat_{i}
        for s_i, node in enumerate(self.graph.nodes):
            if node.op == "output":
                continue
            argi = 0
            for user in node.users:
                # TODO: check this
                if user.op == "output":
                    continue
                s_j = self.node_map[user]
                argj = [i for i, n in enumerate(user.all_input_nodes) if n == node]
                assert len(argj) == 1
                argj = argj[0]
                # argj in s_j corresponds to argi in s_i

                vars_s_i = {}
                for _, s_i_oi, s_i_ii in self.walk_over_options(node, argi):
                    key = (s_i, argi, s_i_oi, s_i_ii)
                    if key in self.cluster_links:
                        continue
                    va = self.ds[key]["va"]
                    vars_s_i.setdefault(s_i_oi, []).append(va)

                vars_s_j = {}
                for _, s_j_oi, s_j_ii in self.walk_over_options(user, argj):
                    key = (s_j, argj, s_j_oi, s_j_ii)
                    if key in self.cluster_links:
                        continue
                    va = self.ds[key]["va"]
                    vars_s_j.setdefault(s_j_ii, []).append(va)

                if vars_s_i.keys() != vars_s_j.keys():
                    vars_s_j = {}
                    for _, s_j_oi, s_j_ii in self.walk_over_options(user, argj):
                        key = (s_j, argj, s_j_oi, s_j_ii)
                        if key in self.cluster_links:
                            va = self.ds[self.cluster_links[key]]["va"]
                        else:
                            va = self.ds[key]["va"]
                        vars_s_j.setdefault(s_j_ii, []).append(va)

                if vars_s_i.keys() != vars_s_j.keys():
                    vars_s_i = {}
                    for _, s_i_oi, s_i_ii in self.walk_over_options(node, argi):
                        key = (s_i, argi, s_i_oi, s_i_ii)
                        if key in self.cluster_links:
                            va = self.ds[self.cluster_links[key]]["va"]
                        else:
                            va = self.ds[key]["va"]
                        vars_s_i.setdefault(s_i_oi, []).append(va)

                assert vars_s_i.keys() == vars_s_j.keys(), f"{vars_s_i}, {vars_s_j}"

                for k in vars_s_i:
                    self.prob += (
                        pulp.lpSum(vars_s_i[k]) == pulp.lpSum(vars_s_j[k]),
                        _get_next_name("output_input_consistent"),
                    )

    def add_inf_cost_constraint(self):
        """
        COST CONSTRAINTS (Category 4): Variables with infinite cost (invalid configurations) are
        forced to zero.

        Mathematical form: ∀i,a,o,j: c_{i,a,o,j} = ∞ ⟹ x_{i,a,o,j} = 0
        """
        # force inf cost values to be 0, as the solver doesn't accept inf
        for key, x in self.ds.items():
            if not math.isfinite(x["cost"]):
                # set the cost to an arbitrary number
                x["cost"] = 10000.0
                if key in self.cluster_links:
                    continue
                self.prob += (x["va"] == 0, _get_next_name("inf_cases"))

    def add_default_constraints(self):
        self.add_unique_decision_constraint()
        self.add_same_output_across_args_constraint()
        self.add_output_input_consistent_constraint()
        self.add_inf_cost_constraint()

        self.penalize_inefficient_collectives()

    def penalize_inefficient_collectives(self):
        """
        EFFICIENCY CONSTRAINTS (Category 5): Penalize inefficient collective operations like
        non-batch dimension shard-to-replicate conversions and forbid invalid transitions.

        - Shard(dim≠0) → Replicate: multiply cost by 4
        - Replicate → Partial: x_{i,a,o,j} = 0 (forbidden)
        - Partial → Shard(dim≠0): multiply cost by 4

        When performing shard_{n} -> replicate (for n != 0), there is additional
        computation cost associated. Let's penalize it here while we don't add
        the computation cost together in the comm cost
        """
        # return
        for s_i, node in enumerate(self.graph.nodes):
            if node.op != "call_function":
                continue
            tgt_op_strat = self.strats[node]
            for counter, parent in enumerate(node.all_input_nodes):
                curr_op_strat = self.strats[parent]

                for oi, tgt_strat in enumerate(tgt_op_strat.strategies):
                    spec = tgt_strat.input_specs[counter]
                    if not isinstance(spec, DTensorSpec):
                        # TODO: check if this is correct
                        continue

                    for ii, curr_strat in enumerate(curr_op_strat.strategies):
                        curr_spec = curr_strat.output_specs
                        if not isinstance(curr_spec, DTensorSpec):
                            continue
                        for tgt_plc, curr_plc in zip(
                            spec.placements, curr_spec.placements
                        ):
                            if (
                                tgt_plc.is_replicate()
                                and curr_plc.is_shard()
                                and curr_plc.dim != 0
                            ):
                                # penalize case S(1) -> R as there are additional compute cost
                                # TODO: add proper compute cost in the optimization objective
                                self.ds[(s_i, counter, oi, ii)]["cost"] *= 4
                            elif tgt_plc.is_partial() and curr_plc.is_replicate():
                                # forbit  R -> P case as this doesn't make sense for us
                                self.prob += self.ds[(s_i, counter, oi, ii)]["va"] == 0
                            elif (
                                tgt_plc.is_shard()
                                and tgt_plc.dim != 0
                                and curr_plc.is_partial()
                            ):
                                # penalize case P -> S(1) as there are additional compute cost
                                self.ds[(s_i, counter, oi, ii)]["cost"] *= 4

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

    def print_old(self):
        ds = self.ds
        res = self.res

        import pprint

        nodes = list(self.graph.nodes)
        pprint.pprint(
            [(nodes[x[0]], str(ds[x]["full_strat"]), ds[x]["cost"]) for x in res]
        )
        total_cost = sum(ds[x]["cost"] for x in res)
        print(f"total_cost: {total_cost:.2f}")
        print(self.get_violated_constraints_log())

    def get_log(self, colored=False):

        from torch.fx.graph import _color_fns, _identity

        opt = {}
        nodes = list(self.graph.nodes)
        for x in self.res:
            opt.setdefault(nodes[x[0]], []).append(self.ds[x])

        # TODO: use python_code to have a shorter representation
        # of the graph. We could use node.format_node() instead
        # but it is a bit more verbose
        # also, this is a bit hacky as it makes a bunch of assumptions regarding lines
        # so would be better to make it more robust
        code = self.graph.python_code("self", colored=colored).src.strip().split("\n")
        if colored:
            txt_color = _color_fns["blue"]
            attr_color = _color_fns["red"]
        else:
            txt_color = _identity
            attr_color = _identity
        l_id = 1
        plc_txt = txt_color("# placement=")
        cost_txt = txt_color(", cost=")
        for node in nodes:
            if node.op == "output":
                continue
            d = opt[node]
            strat = str(d[0]["full_strat"])
            costs = [
                (x["comm_cost"], x["compute_cost"], x["sharding_transition_cost"])
                for x in d
            ]
            line = f"  {plc_txt}{attr_color(strat)}{cost_txt}{attr_color(str(costs))}"
            if node.op == "placeholder":
                line = f"    # {node.name}: {line}"
                code.insert(l_id, line)
                l_id += 1
                continue
            # LOL
            while not code[l_id].lstrip().startswith(repr(node)):
                l_id += 1
            code[l_id] += line
            l_id += 1
        code = "\n".join(code)
        total_cost = sum(self.ds[x]["cost"] for x in self.res)
        code += f"\ntotal_cost: {total_cost:.2f}"
        code += "\n" + self.get_violated_constraints_log()
        return code

    def print_costs_for_node(self, node, arg=0, **kwargs):
        from tabulate import tabulate  # type: ignore
        from torch.distributed.tensor._op_schema import _pretty_print_spec

        tgt_strat = self.strats[node]
        src_strat = self.strats[self._all_input_nodes(node)[arg]]
        src_placements = [""] + [
            _pretty_print_spec(x.output_specs) for x in src_strat.strategies
        ]
        costs = [[str(x)] + x.redistribute_cost[arg] for x in tgt_strat.strategies]
        # dst_placements = [str(x) for x in tgt_strat.strategies]
        # costs = [x.redistribute_cost[arg] for x in tgt_strat.strategies]
        # print(dst_placements)
        print(tabulate(costs, headers=src_placements, **kwargs))

    def get_solution(self, verbose=False):
        # add cost
        opt_target = defaultdict(int)
        # let's remove potentially duplicate variables in the program and just
        # add their costs
        for x in self.ds.values():
            opt_target[x["va"]] += x["cost"]
        self.prob += pulp.lpSum([va * cost for va, cost in opt_target.items()])

        # solver = pulp.HiGHS(msg=verbose)
        solver = pulp.PULP_CBC_CMD(msg=verbose)
        self.prob.solve(solver)

        sol = {k: v["va"].value() for k, v in self.ds.items()}

        self.res = [k for k, v in sol.items() if v == 1]

        if self.prob.status == -1:
            print(self.get_log())
            raise RuntimeError("Unsolvable problem")

        opt = {}
        nodes = list(self.graph.nodes)
        for x in self.res:
            opt.setdefault(nodes[x[0]], []).append(self.ds[x])

        # validate that a single solution is chosen
        # per node
        seen = set()
        for r in self.res:
            key = (r[0], r[1])
            if key in seen:
                raise RuntimeError(
                    f"Multiple solutions for {nodes[key[0]]}, key={key}, "
                    f"solutions: {[str(x['full_strat']) for x in opt[nodes[key[0]]]]}"
                )
            seen.add(key)

        # Let's simplify the output representation
        # and just return a single OpSpec
        # for each node
        for k in opt.values():
            assert all(
                k[0]["full_strat"] == kk["full_strat"] for kk in k
            ), f"{[kk['va'] for kk in k]}: {[str(kk['full_strat']) for kk in k]}"

        new_opt = {}
        for k, v in opt.items():
            new_opt[k] = v[0]["full_strat"]

        opt = new_opt

        # TODO: assert all nodes have a placement?
        return opt

    def _add_node_constraint(self, node, oi, constraint_name=None):
        if constraint_name is None:
            constraint_name = "user_constraint"
        s_i = self.node_map[node]
        vars_per_arg = {}
        for argi, oi_, ii in self.walk_over_options(node):
            if oi_ == oi:
                va = self.ds[(s_i, argi, oi, ii)]["va"]
                vars_per_arg.setdefault(argi, []).append(va)
        for eqs in vars_per_arg.values():
            self.prob += (pulp.lpSum(eqs) == 1, _get_next_name(constraint_name))

    def add_grad_param_constraints(self):
        """
        USER CONSTRAINTS (Category 6c): Parameter-gradient consistency constraints.
        Ensures parameters and their gradients have matching sharding strategies.

        Mathematical form: x_{param} = x_{grad_param}
        """
        for param, grad in get_param_and_grad_nodes(self.graph).values():
            if grad is None:
                continue
            s_i = self.node_map[param]
            s_j = self.node_map[grad]
            # parameters have a single input strat, so remove one loop
            # i.e., self.num_args[s_i] == 1 and num_inp_strat == 1
            num_out_strat = self.num_inp_out[(s_i, 0)]["num_output_strat"]
            num_inp_g_strat = self.num_inp_out[(s_j, 0)]["num_input_strat"]
            strat_p = [
                str(strat.output_specs) for strat in self.strats[param].strategies
            ]
            assert num_out_strat == len(strat_p)
            strat_gp = [
                str(strat.output_specs) for strat in self.strats[grad].strategies
            ]
            for oi in range(num_out_strat):
                v_p = self.ds[(s_i, 0, oi, 0)]["va"]
                sp = strat_p[oi]
                # TODO: fix this case
                if sp not in strat_gp:
                    continue
                v_gp = []
                ooi = strat_gp.index(sp)
                for ii in range(num_inp_g_strat):
                    v_gp.append(self.ds[(s_j, 0, ooi, ii)]["va"])
                self.prob += (
                    pulp.lpSum(v_gp) == v_p,
                    _get_next_name("grad_param_constraint"),
                )

    def add_parameter_memory_constraint(self, memory_factor_low, memory_factor_high):
        """
        USER CONSTRAINTS (Category 6b): Memory constraints for parameters.
        Ensures total parameter memory usage stays within specified bounds.

        Mathematical form: Σ_{params} (size_ratio * x_{param}) ≤ memory_limit
        """
        # get all parameters
        param_nodes = get_param_nodes(self.graph)
        elms = []
        for node in param_nodes:
            s_i = self.node_map[node]
            vv = self.num_inp_out[(s_i, 0)]
            for ii in range(vv["num_output_strat"]):
                data = self.ds[(s_i, 0, ii, 0)]
                spec = data["inp_strat"]
                tensor_shape = spec.tensor_meta.shape
                new_tensor_shape, _ = _get_sharded_shape_stride(spec)
                new_size = math.prod(new_tensor_shape)
                old_size = math.prod(tensor_shape)
                elms.append(data["va"] * new_size / old_size)

        memory_factor_low *= len(param_nodes)
        memory_factor_high *= len(param_nodes)
        self.prob += (pulp.lpSum(elms) <= memory_factor_high, "memory_constraint_high")
        self.prob += (pulp.lpSum(elms) >= memory_factor_low, "memory_constraint_low")

    def add_node_constraint(self, node, placement=None, constraint_name=None):
        """
        USER CONSTRAINTS (Category 6d): General node constraints.
        Force specific placement for any node.

        Mathematical form: x_{i,a,o*,j*} = 1 for specified (o*,j*)
        """
        assert node in self.strats, (node, self.strats.keys())
        strat = self.strats[node]
        if placement is None:
            # default is Shard(0) to parallelize on the batch
            placement = (Shard(0),) + (Replicate(),) * (self.mesh.ndim - 1)
        for oi, s in enumerate(strat.strategies):
            spec = s.output_specs
            if spec.placements == placement:
                break
        else:
            raise RuntimeError(
                f"Couldn't find appropriate constraint {node} {constraint_name} {placement}"
            )
        self._add_node_constraint(node, oi=oi, constraint_name=constraint_name)

    def add_sharded_input_constraint(
        self, input_placements: Optional[list[Optional[tuple[Placement, ...]]]] = None
    ):
        """
        USER CONSTRAINTS (Category 6a): Input placement constraints.
        Force specific placements for input nodes and their corresponding gradient inputs.

        Mathematical form: x_{i,a,o*,j*} = 1 for specified input placements (o*,j*)
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
                placement = mut_ips.pop(desc.idx)

            self.add_node_constraint(
                node, placement, constraint_name="input_constraint"
            )
            if grad_node is not None:
                self.add_node_constraint(
                    grad_node, placement, constraint_name="grad_input_constraint"
                )

        ignored_placements = []
        if mut_ips is not None:
            for i, p in mut_ips.items():
                if p is not None:
                    ignored_placements.append(i)

        if ignored_placements:
            raise RuntimeError(
                f"We were unable to respect placements for inputs at indices {ignored_placements}.  "
                f"This is because the traced joint graph did not actually have a dedicated placeholder node for these inputs.  "
                f"This typically occurs because some inputs aliased each other; inspect the joint graph from tlparse for more details.  "
                f"You can either remove an explicit placement for this input (replace it with None) or clone "
                "the inputs before tracing to remove aliasing."
            )

    def add_sharded_output_constraint(self, output_placements=None):
        """
        USER CONSTRAINTS (Category 6a): Output placement constraints.
        Force specific placements for output nodes and their corresponding gradient outputs.

        Mathematical form: x_{i,a,o*,j*} = 1 for specified output placements (o*,j*)
        """
        mut_ops = None
        if output_placements is not None:
            mut_ops = {i: p for i, p in enumerate(output_placements)}

        output_and_tangent_nodes_index = get_plain_output_and_tangent_nodes(self.graph)
        for desc, (node, tangent_node) in output_and_tangent_nodes_index.items():
            if output_placements is None:
                placement = None
            else:
                assert isinstance(desc, PlainAOTOutput)
                assert mut_ops is not None
                placement = mut_ops.pop(desc.idx)

            self.add_node_constraint(
                node, placement, constraint_name="output_constraint"
            )
            if tangent_node is not None:
                self.add_node_constraint(
                    tangent_node, placement, constraint_name="grad_output_constraint"
                )

        ignored_placements = []
        if mut_ops is not None:
            for i, p in mut_ops.items():
                if p is not None:
                    ignored_placements.append(i)

        if ignored_placements:
            raise RuntimeError(
                f"We were unable to respect placements for outputs at indices {ignored_placements}.  "
                f"This is because the traced joint graph did not actually have a dedicated output node for these inputs.  "
                f"This typically occurs because some outputs aliased each other; inspect the joint graph from tlparse for more details.  "
                f"You can either remove an explicit placement for this output (replace it with None),"
                "stop the model from returning aliases of the tensor or clone the outputs before returning "
                "them from the graph to avoid aliasing."
            )

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
            ospec = strat0.output_specs
            if isinstance(ospec, (list, tuple)):
                for spec in ospec:
                    if spec:
                        assert (
                            spec.tensor_meta is not None
                        ), f"{node} doesn't have a tensor_meta"
            else:
                assert (
                    ospec.tensor_meta is not None
                ), f"{node} doesn't have a tensor_meta"

            for ospec in strat0.input_specs:
                if isinstance(ospec, (list, tuple)):
                    for spec in ospec:
                        assert (
                            spec.tensor_meta is not None
                        ), f"{node} input_spec doesn't have a tensor_meta"
                else:
                    assert (
                        ospec.tensor_meta is not None
                    ), f"{node} input_spec doesn't have a tensor_meta"
            for i, arg in enumerate(all_input_nodes):
                strat_arg = self.strats[arg]
                num_arg_strats = len(strat_arg.strategies)
                assert (
                    len(strat0.redistribute_cost[i]) == num_arg_strats
                ), f"{node}, {len(strat0.redistribute_cost[i])}, {num_arg_strats}"
