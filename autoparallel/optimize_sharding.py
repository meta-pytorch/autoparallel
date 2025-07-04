# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math

import pulp
import torch
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.utils._pytree import tree_flatten, tree_map_only

from .compute_estimation import _get_sharded_shape, estimate_strategy_runtime_cost
from .propagation_rules import _create_all_options
from .utils import get_placement_options


def _debug_node(node):
    def my_print(x):
        x = x.meta["val"]
        if isinstance(x, torch.Tensor):
            return (x.shape, x.stride())
        return x

    print(tree_map_only(torch.fx.Node, my_print, node.args))


_GLOBAL_NAMES: dict[str, int] = {}


def _get_next_name(name):
    global _GLOBAL_NAMEs
    idx = _GLOBAL_NAMES.setdefault(name, 0)
    _GLOBAL_NAMES[name] += 1
    return name + f"_{idx:03}"


class ShardingOptimizer:
    def __init__(self, gm, mesh):
        self.gm = gm
        self.graph = gm.graph
        self.mesh = mesh
        self.node_map = {node: i for i, node in enumerate(self.graph.nodes)}
        self.strats = self.build_sharding_metadata()
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

    def build_ds(self):
        strats = self.strats
        ds = {}
        num_inp_out = {}
        num_args = {}
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
                    for ii, comm_cost in enumerate(xxi):
                        va = pulp.LpVariable(
                            f"n={node},s={s_i},arg={argi},output_p={ss},input_p={ii}",
                            cat=pulp.LpBinary,
                        )
                        ds[(s_i, argi, ss, ii)] = {
                            "va": va,
                            "cost": comm_cost + compute_cost / num_args[s_i],
                            "full_strat": ssi,
                            "out_strat": ssi.output_specs,
                            "inp_strat": ssi.input_specs[argi],
                        }
        return ds, num_inp_out, num_args

    def _all_input_nodes(self, node):
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
        # a single pair of input-output policy is chosen
        for s_i, node in enumerate(self.graph.nodes):
            if node.op not in {"placeholder", "call_function"}:
                continue
            arg_vars = {}
            for arg, oi, ii in self.walk_over_options(node):
                va = self.ds[(s_i, arg, oi, ii)]["va"]
                arg_vars.setdefault(arg, []).append(va)
            for eqs in arg_vars.values():
                self.prob += (pulp.lpSum(eqs) == 1, _get_next_name("unique_decision"))

    def add_same_output_across_args_constraint(self):
        # enforce that the same output policy is chosen
        # across arguments
        for s_i, node in enumerate(self.graph.nodes):
            if node.op != "call_function":
                continue
            if len(self._all_input_nodes(node)) <= 1:
                continue
            vars_per_output = {}
            for arg, oi, ii in self.walk_over_options(node):
                va = self.ds[(s_i, arg, oi, ii)]["va"]
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
                    va = self.ds[(s_i, argi, s_i_oi, s_i_ii)]["va"]
                    vars_s_i.setdefault(s_i_oi, []).append(va)

                vars_s_j = {}
                for _, s_j_oi, s_j_ii in self.walk_over_options(user, argj):
                    va = self.ds[(s_j, argj, s_j_oi, s_j_ii)]["va"]
                    vars_s_j.setdefault(s_j_ii, []).append(va)

                assert vars_s_i.keys() == vars_s_j.keys()
                for k in vars_s_i:
                    self.prob += (
                        pulp.lpSum(vars_s_i[k]) == pulp.lpSum(vars_s_j[k]),
                        _get_next_name("output_input_consistent"),
                    )

    def add_inf_cost_constraint(self):
        # force inf cost values to be 0, as the solver doesn't accept inf
        for x in self.ds.values():
            if not math.isfinite(x["cost"]):
                self.prob += (x["va"] == 0, _get_next_name("inf_cases"))
                # set the cost to an arbitrary number
                x["cost"] = 10000.0

    def add_default_constraints(self):
        self.add_unique_decision_constraint()
        self.add_same_output_across_args_constraint()
        self.add_output_input_consistent_constraint()
        self.add_inf_cost_constraint()

        self.penalize_inefficient_collectives()

    def penalize_inefficient_collectives(self):
        """
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
            costs = [x["cost"] for x in d]
            line = f"  {plc_txt}{attr_color(strat)}{cost_txt}{attr_color(str(costs))}"
            if node.op == "placeholder":
                line = f"    # {node.name}: {line}"
                code.insert(l_id, line)
                l_id += 1
                continue
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
        src_strat = self.strats[node.args[arg]]
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
        self.prob += pulp.lpSum([x["va"] * x["cost"] for x in self.ds.values()])

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

    def get_param_nodes(self):
        # NOTE: this relies my customized export_module
        param_nodes = [
            x for x in self.graph.find_nodes(op="placeholder") if "param" in x.target
        ]
        return param_nodes

    def get_input_nodes(self):
        # NOTE: this relies my customized export_module
        input_nodes = [
            x for x in self.graph.find_nodes(op="placeholder") if "input" in x.target
        ]
        return input_nodes

    def get_tangent_nodes(self):
        # NOTE: this relies my customized export_module
        tangent_nodes = [
            x for x in self.graph.find_nodes(op="placeholder") if "tangent" in x.target
        ]
        return tangent_nodes

    def get_fn_output_nodes(self):
        # NOTE: this relies my customized export_module
        output_nodes = [
            x
            for x in self.graph.find_nodes(op="output")[0].all_input_nodes
            if "output" in x.name
        ]
        return output_nodes

    def get_grad_input_nodes(self):
        # NOTE: this relies my customized export_module
        grad_input_nodes = [
            x
            for x in self.graph.find_nodes(op="output")[0].all_input_nodes
            if "grad_input" in x.name
        ]
        return grad_input_nodes

    def get_grad_param_nodes(self):
        # NOTE: this relies my customized export_module
        grad_param_nodes = [
            x
            for x in self.graph.find_nodes(op="output")[0].all_input_nodes
            if "grad_param" in x.name
        ]
        return grad_param_nodes

    def add_grad_param_constraints(self):
        # TODO: need to make sure that the params and grads are aligned, which are not always the case
        # and we might have fewer gradients than parameters

        # suppose joint graph case
        param_nodes = self.get_param_nodes()
        grad_nodes = self.get_grad_param_nodes()
        # assert len(param_nodes) == len(grad_nodes)

        for param, grad in zip(param_nodes, grad_nodes):
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
        # get all parameters
        param_nodes = self.get_param_nodes()
        elms = []
        for node in param_nodes:
            s_i = self.node_map[node]
            vv = self.num_inp_out[(s_i, 0)]
            for ii in range(vv["num_output_strat"]):
                data = self.ds[(s_i, 0, ii, 0)]
                spec = data["inp_strat"]
                tensor_shape = spec.tensor_meta.shape
                new_tensor_shape = _get_sharded_shape(spec)
                new_size = math.prod(new_tensor_shape)
                old_size = math.prod(tensor_shape)
                elms.append(data["va"] * new_size / old_size)

        memory_factor_low *= len(param_nodes)
        memory_factor_high *= len(param_nodes)
        self.prob += (pulp.lpSum(elms) <= memory_factor_high, "memory_constraint_high")
        self.prob += (pulp.lpSum(elms) >= memory_factor_low, "memory_constraint_low")

    def add_node_constraint(self, node, placement=None, constraint_name=None):
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

    def add_sharded_input_constraint(self, input_placements=None):
        input_nodes = self.get_input_nodes()
        if input_placements is None:
            input_placements = [None] * len(input_nodes)

        assert len(input_placements) == len(
            input_nodes
        ), f"number of input placements {len(input_placements)} doesn't match number of input nodes {len(input_nodes)}"
        for node, placement in zip(input_nodes, input_placements):
            self.add_node_constraint(
                node, placement, constraint_name="input_constraint"
            )

        # ensure gradients of inputs have same sharding as input
        grad_input_nodes = self.get_grad_input_nodes()
        for node in grad_input_nodes:
            # grad_input nodes are numbered according to input, so
            # get the index corresponding to the output
            input_idx = int(node.name.split("_")[-1])
            placement = input_placements[input_idx]
            self.add_node_constraint(
                node, placement, constraint_name="grad_input_constraint"
            )

    def add_sharded_output_constraint(self, output_placements=None):
        # add final constraint on the output strategy
        output_nodes = self.get_fn_output_nodes()

        if output_placements is None:
            output_placements = [None] * len(output_nodes)

        assert len(output_placements) == len(
            output_nodes
        ), f"number of output placements {len(output_placements)} doesn't match number of output nodes {len(output_nodes)}"
        for node, placement in zip(output_nodes, output_placements):
            self.add_node_constraint(
                node, placement, constraint_name="output_constraint"
            )

        # ensure gradients of outputs have same sharding as output
        tangent_nodes = self.get_tangent_nodes()
        for node in tangent_nodes:
            # tangent nodes are numbered according to output, so
            # get the index corresponding to the output
            output_idx = int(node.name.split("_")[-1])
            placement = output_placements[output_idx]
            self.add_node_constraint(
                node, placement, constraint_name="grad_output_constraint"
            )

    def validate(self):
        for node in self.graph.nodes:
            if node.op != "call_function":
                continue
            strat = self.strats[node]
            strat0 = strat.strategies[0]
            # use the following instead of all_input_nodes as all_input_nodes remove duplicate nodes
            # TODO: add kwargs?
            all_input_nodes = [
                x for x in tree_flatten(node.args)[0] if isinstance(x, torch.fx.Node)
            ]
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
