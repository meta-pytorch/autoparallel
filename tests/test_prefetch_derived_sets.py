# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for build_param_derived_set and build_terminal_derived_set."""

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

        n_modified = optimizer.apply_prefetch_discount(scale=0.0)
        assert n_modified > 0, "Expected some decision vars to be modified"

        # Verify modified vars have comm_cost == 0 and cost is recomputed
        n_zeroed = 0
        for key, dv in optimizer.decision_vars.items():
            if key in original_costs and dv.comm_cost == 0.0:
                assert dv.cost == dv.compute_cost + dv.sharding_transition_cost
                n_zeroed += 1

        assert n_zeroed > 0, "Expected some edges to be zeroed out"
