# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for build_param_derived_set and build_terminal_derived_set."""

from types import SimpleNamespace

import pulp
import pytest
import torch
import torch.fx
from conftest import apply_cuda_patches
from torch._functorch._aot_autograd.descriptors import (
    GradAOTOutput,
    ParamAOTInput,
    PlainAOTInput,
    PlainAOTOutput,
)

from autoparallel.graph_passes.graph_utils import (
    build_param_derived_set,
    build_terminal_derived_set,
)
from autoparallel.optimize_sharding import DecisionVar, ShardingOptimizer

# ---------------------------------------------------------------------------
# Helpers for building synthetic joint FX graphs
# ---------------------------------------------------------------------------

_dummy_op = torch.ops.aten.abs.default


def _make_placeholder(graph, name, desc):
    node = graph.placeholder(name)
    node.meta["desc"] = desc
    return node


def _make_call(graph, *args):
    return graph.call_function(_dummy_op, args=args)


def _make_output(graph, outputs, descs):
    """Create the output node with AOTAutograd-style desc metadata."""
    out = graph.output(tuple(outputs))
    out.meta["desc"] = descs
    return out


# ---------------------------------------------------------------------------
# build_param_derived_set
# ---------------------------------------------------------------------------


class TestBuildParamDerivedSet:
    def test_chain_propagation(self):
        """param -> cast -> alias is all param-derived."""
        graph = torch.fx.Graph()
        param_w = _make_placeholder(graph, "param_w", ParamAOTInput("w"))
        input_x = _make_placeholder(graph, "input_x", PlainAOTInput(0))
        cast_w = _make_call(graph, param_w)
        alias_w = _make_call(graph, cast_w)
        mm_fwd = _make_call(graph, input_x, alias_w)  # has non-param input
        _make_output(graph, [mm_fwd], [PlainAOTOutput(0)])

        param_derived = build_param_derived_set(graph)

        assert param_w in param_derived
        assert cast_w in param_derived
        assert alias_w in param_derived
        assert input_x not in param_derived
        assert mm_fwd not in param_derived

    def test_fan_in_both_param(self):
        """Node with two param-derived inputs IS param-derived."""
        graph = torch.fx.Graph()
        p1 = _make_placeholder(graph, "p1", ParamAOTInput("a"))
        p2 = _make_placeholder(graph, "p2", ParamAOTInput("b"))
        combined = _make_call(graph, p1, p2)
        _make_output(graph, [combined], [PlainAOTOutput(0)])

        param_derived = build_param_derived_set(graph)

        assert combined in param_derived

    def test_fan_in_mixed(self):
        """Node with one param + one non-param input is NOT param-derived."""
        graph = torch.fx.Graph()
        param = _make_placeholder(graph, "param", ParamAOTInput("w"))
        inp = _make_placeholder(graph, "inp", PlainAOTInput(0))
        mixed = _make_call(graph, param, inp)
        _make_output(graph, [mixed], [PlainAOTOutput(0)])

        param_derived = build_param_derived_set(graph)

        assert param in param_derived
        assert inp not in param_derived
        assert mixed not in param_derived

    def test_no_params(self):
        """Graph with no param placeholders → only empty set."""
        graph = torch.fx.Graph()
        x = _make_placeholder(graph, "x", PlainAOTInput(0))
        y = _make_call(graph, x)
        _make_output(graph, [y], [PlainAOTOutput(0)])

        param_derived = build_param_derived_set(graph)

        assert len(param_derived) == 0

    def test_long_chain(self):
        """Param-derived propagates through an arbitrarily long chain."""
        graph = torch.fx.Graph()
        param = _make_placeholder(graph, "param", ParamAOTInput("w"))
        node = param
        chain = [param]
        for _ in range(10):
            node = _make_call(graph, node)
            chain.append(node)
        _make_output(graph, [node], [PlainAOTOutput(0)])

        param_derived = build_param_derived_set(graph)

        for n in chain:
            assert n in param_derived

    def test_param_derived_stops_at_non_param_input(self):
        """Even if one branch is param-derived, mixing with a non-param stops propagation."""
        graph = torch.fx.Graph()
        param = _make_placeholder(graph, "param", ParamAOTInput("w"))
        inp = _make_placeholder(graph, "inp", PlainAOTInput(0))
        cast_p = _make_call(graph, param)  # param-derived
        mm = _make_call(graph, inp, cast_p)  # NOT param-derived
        post_mm = _make_call(graph, mm)  # NOT param-derived (input isn't)
        _make_output(graph, [post_mm], [PlainAOTOutput(0)])

        param_derived = build_param_derived_set(graph)

        assert cast_p in param_derived
        assert mm not in param_derived
        assert post_mm not in param_derived


# ---------------------------------------------------------------------------
# build_terminal_derived_set
# ---------------------------------------------------------------------------


class TestBuildTerminalDerivedSet:
    def test_chain_to_grad_output(self):
        """Nodes whose only users flow into a grad output are terminal-derived."""
        graph = torch.fx.Graph()
        param_desc = ParamAOTInput("w")
        param = _make_placeholder(graph, "param", param_desc)
        inp = _make_placeholder(graph, "inp", PlainAOTInput(0))
        fwd_out = _make_call(graph, inp, param)
        # Backward: a chain leading to the gradient
        bwd_mm = _make_call(graph, fwd_out)
        view_g = _make_call(graph, bwd_mm)
        alias_g = _make_call(graph, view_g)
        _make_output(
            graph,
            [fwd_out, alias_g],
            [PlainAOTOutput(0), GradAOTOutput(grad_of=param_desc)],
        )

        terminal = build_terminal_derived_set(graph)

        assert alias_g in terminal
        assert view_g in terminal
        assert bwd_mm in terminal
        # fwd_out has users in both forward output and backward chain
        assert fwd_out not in terminal

    def test_partial_users_not_terminal(self):
        """Node with one grad user + one compute user is NOT terminal-derived."""
        graph = torch.fx.Graph()
        param_desc = ParamAOTInput("w")
        param = _make_placeholder(graph, "param", param_desc)
        inp = _make_placeholder(graph, "inp", PlainAOTInput(0))
        fwd = _make_call(graph, inp, param)
        # bwd_node feeds into grad output AND another compute node
        bwd_node = _make_call(graph, fwd)
        grad_node = _make_call(graph, bwd_node)  # → grad output
        compute = _make_call(graph, bwd_node)  # → plain output
        _make_output(
            graph,
            [fwd, grad_node, compute],
            [
                PlainAOTOutput(0),
                GradAOTOutput(grad_of=param_desc),
                PlainAOTOutput(1),
            ],
        )

        terminal = build_terminal_derived_set(graph)

        # grad_node's only user is the output (as a grad) → terminal
        assert grad_node in terminal
        # bwd_node has two users: grad_node (terminal) and compute (not terminal)
        assert bwd_node not in terminal

    def test_no_grads(self):
        """Graph with no gradient outputs → empty terminal-derived set."""
        graph = torch.fx.Graph()
        inp = _make_placeholder(graph, "inp", PlainAOTInput(0))
        out = _make_call(graph, inp)
        _make_output(graph, [out], [PlainAOTOutput(0)])

        terminal = build_terminal_derived_set(graph)

        assert len(terminal) == 0

    def test_multiple_params(self):
        """Multiple param gradients each get their own terminal chain."""
        graph = torch.fx.Graph()
        desc_a = ParamAOTInput("a")
        desc_b = ParamAOTInput("b")
        pa = _make_placeholder(graph, "pa", desc_a)
        pb = _make_placeholder(graph, "pb", desc_b)
        inp = _make_placeholder(graph, "inp", PlainAOTInput(0))
        fwd = _make_call(graph, inp, pa, pb)
        # Two independent gradient chains
        grad_chain_a = _make_call(graph, fwd)
        grad_a = _make_call(graph, grad_chain_a)
        grad_chain_b = _make_call(graph, fwd)
        grad_b = _make_call(graph, grad_chain_b)
        _make_output(
            graph,
            [fwd, grad_a, grad_b],
            [
                PlainAOTOutput(0),
                GradAOTOutput(grad_of=desc_a),
                GradAOTOutput(grad_of=desc_b),
            ],
        )

        terminal = build_terminal_derived_set(graph)

        assert grad_a in terminal
        assert grad_chain_a in terminal
        assert grad_b in terminal
        assert grad_chain_b in terminal
        # fwd feeds into both grad chains AND plain output
        assert fwd not in terminal


# ---------------------------------------------------------------------------
# Integration test: apply_prefetch_discount on a real traced graph
# ---------------------------------------------------------------------------


class FFN(torch.nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.linear1 = torch.nn.Linear(dim1, dim2, bias=False)
        self.linear2 = torch.nn.Linear(dim2, dim1, bias=False)

    def forward(self, x):
        return self.linear2(self.linear1(x))


def test_prefetch_discount_replaces_scale_without_mutating_costs():
    optimizer = ShardingOptimizer.__new__(ShardingOptimizer)
    key = (0, 0, 0, 0)
    illegal_key = (1, 0, 0, 0)
    variable = pulp.LpVariable("prefetch_test", cat=pulp.LpBinary)
    illegal_variable = pulp.LpVariable("prefetch_illegal", cat=pulp.LpBinary)
    decision = DecisionVar(
        var=variable,
        cost=16.0,
        compute_cost=5.0,
        comm_cost=10.0,
        sharding_transition_cost=1.0,
        strategy=None,
        output_spec=None,
        input_spec=None,
    )
    illegal_decision = DecisionVar(
        var=illegal_variable,
        cost=10000.0,
        compute_cost=0.0,
        comm_cost=float("inf"),
        sharding_transition_cost=0.0,
        strategy=None,
        output_spec=None,
        input_spec=None,
    )
    optimizer.decision_vars = {key: decision, illegal_key: illegal_decision}
    optimizer.cluster_links = {}
    optimizer._root_to_linked = {}
    optimizer._prefetch_discount = 1.0
    optimizer._prefetchable_keys = {key, illegal_key}
    optimizer.prob = pulp.LpProblem("prefetch_test", pulp.LpMinimize)

    assert optimizer.apply_prefetch_discount(0.5) == 2
    optimizer._set_objective()
    assert optimizer.prob.objective[variable] == pytest.approx(11.0)
    assert optimizer.prob.objective[illegal_variable] == 10000.0
    assert decision.cost == 16.0
    assert decision.comm_cost == 10.0

    optimizer.apply_prefetch_discount(0.0)
    assert optimizer.prob.objective[variable] == pytest.approx(6.0)
    assert decision.cost == 16.0
    assert decision.comm_cost == 10.0


def test_default_prefetch_scale_skips_key_classification(monkeypatch):
    optimizer = ShardingOptimizer.__new__(ShardingOptimizer)
    optimizer._prefetch_discount = 1.0

    def fail():
        pytest.fail("default scale should not classify prefetchable keys")

    monkeypatch.setattr(optimizer, "_get_prefetchable_keys", fail)

    assert optimizer._get_comm_scale((0, 0, 0, 0)) == 1.0


def test_prefetchable_key_cache_is_cost_independent():
    graph = torch.fx.Graph()
    parameter = _make_placeholder(graph, "parameter", ParamAOTInput("weight"))
    consumer = _make_call(graph, parameter)
    _make_output(graph, [consumer], [PlainAOTOutput(0)])

    zero_key = (1, 0, 0, 0)
    infinite_key = (1, 0, 1, 0)
    optimizer = ShardingOptimizer.__new__(ShardingOptimizer)
    optimizer.graph = graph
    optimizer.nodes = [parameter, consumer]
    optimizer.decision_vars = {
        zero_key: DecisionVar(
            var=pulp.LpVariable("prefetch_zero", cat=pulp.LpBinary),
            cost=1.0,
            compute_cost=1.0,
            comm_cost=0.0,
            sharding_transition_cost=0.0,
            strategy=None,
            output_spec=None,
            input_spec=None,
        ),
        infinite_key: DecisionVar(
            var=pulp.LpVariable("prefetch_infinite", cat=pulp.LpBinary),
            cost=10000.0,
            compute_cost=0.0,
            comm_cost=float("inf"),
            sharding_transition_cost=0.0,
            strategy=None,
            output_spec=None,
            input_spec=None,
        ),
    }
    optimizer._prefetchable_keys = None
    optimizer._all_input_nodes = lambda node: list(node.all_input_nodes)

    assert optimizer._get_prefetchable_keys() == {zero_key, infinite_key}


def test_solution_cost_breakdown_remains_undiscounted(monkeypatch):
    producer = object()
    consumer = object()
    producer_strategy = SimpleNamespace(
        input_specs=[], output_specs=None, redistribute_cost=[]
    )
    consumer_strategy = SimpleNamespace(
        input_specs=[None], output_specs=None, redistribute_cost=[[10.0]]
    )
    decisions = {
        0: DecisionVar(
            var=pulp.LpVariable("producer", cat=pulp.LpBinary),
            cost=2.0,
            compute_cost=2.0,
            comm_cost=0.0,
            sharding_transition_cost=0.0,
            strategy=producer_strategy,
            output_spec=None,
            input_spec=None,
        ),
        1: DecisionVar(
            var=pulp.LpVariable("consumer", cat=pulp.LpBinary),
            cost=15.0,
            compute_cost=5.0,
            comm_cost=10.0,
            sharding_transition_cost=0.0,
            strategy=consumer_strategy,
            output_spec=None,
            input_spec=None,
        ),
    }
    optimizer = ShardingOptimizer.__new__(ShardingOptimizer)
    optimizer.strats = {
        producer: SimpleNamespace(strategies=[producer_strategy]),
        consumer: SimpleNamespace(strategies=[consumer_strategy]),
    }
    optimizer.node_map = {producer: 0, consumer: 1}
    optimizer._all_input_nodes = lambda node: [] if node is producer else [producer]
    optimizer._resolve_decision_var = lambda key: decisions[key[0]]
    optimizer._prefetch_discount = 0.0

    def fail(_key):
        pytest.fail("base cost breakdown should not apply the prefetch scale")

    monkeypatch.setattr(optimizer, "_get_comm_scale", fail)

    costs = optimizer._compute_solution_cost(
        {producer: producer_strategy, consumer: consumer_strategy}
    )

    assert costs == {
        "total": 17.0,
        "compute": 7.0,
        "comm": 10.0,
        "transition": 0.0,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@apply_cuda_patches
def test_apply_prefetch_discount(device_mesh_2d):
    from autoparallel.api import AutoParallel

    dim1, dim2 = 1024, 4096
    bs = 2048 * device_mesh_2d.shape[0]

    def input_fn():
        return torch.randn(bs, dim1, device="cuda")

    with torch.device("meta"):
        model = FFN(dim1, dim2)

    with AutoParallel(model, input_fn, device_mesh_2d) as autop:
        optimizer = autop.sharding_optimizer

        # Snapshot original comm costs for decision vars with nonzero comm
        original_costs = {}
        for key, dv in optimizer.decision_vars.items():
            if dv.comm_cost > 0 and dv.comm_cost < float("inf"):
                original_costs[key] = dv.comm_cost

        assert len(original_costs) > 0, "Expected some edges with nonzero comm cost"

        n_affected = optimizer.apply_prefetch_discount(scale=0.5)
        assert n_affected > 0, "Expected some decision vars to be discounted"
        assert all(
            optimizer.decision_vars[key].comm_cost == cost
            for key, cost in original_costs.items()
        )

        optimizer._set_objective()
        key = next(iter(optimizer._get_prefetchable_keys()))
        dv = optimizer.decision_vars[key]
        multiplier = 1 + len(optimizer._root_to_linked.get(key, []))
        expected = (dv.cost - 0.5 * dv.comm_cost) * multiplier
        assert optimizer.prob.objective[dv.var] == pytest.approx(expected)

        optimizer.apply_prefetch_discount(scale=0.0)
        expected = (dv.cost - dv.comm_cost) * multiplier
        assert optimizer.prob.objective[dv.var] == pytest.approx(expected)
        assert dv.comm_cost == original_costs[key]
