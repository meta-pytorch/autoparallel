# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math
import operator

import pytest
import torch
import torch.nn.functional as F

from autoparallel.graph_passes.graph_utils import all_input_nodes
from autoparallel.optimize_sharding import DPBasedShardingSolver


class _FakeOptimizer:
    def __init__(self, graph):
        self.graph = graph
        self.strats = {node: object() for node in graph.nodes}
        self.nodes = list(self.strats.keys())

    def _all_input_nodes(self, node):
        return [
            input_node
            for input_node in all_input_nodes(node)
            if input_node in self.strats
        ]


def _assert_predecessors_match_graph_indegrees(topology):
    topology_nodes = set(topology.nodes)
    assert set(topology.predecessors) == topology_nodes
    assert set(topology.node_to_index) == topology_nodes

    for node in topology.nodes:
        expected_predecessors = [
            input_node
            for input_node in all_input_nodes(node)
            if input_node in topology_nodes
        ]
        predecessors = topology.predecessors[node]
        assert len(predecessors) == len(expected_predecessors)
        assert predecessors == expected_predecessors


def test_dp_solver_builds_topological_order_for_merge_graph():
    class MergeModule(torch.nn.Module):
        def forward(self, x, y):
            a = x + y
            b = x * 2
            return a + b

    graph = torch.fx.symbolic_trace(MergeModule()).graph
    solver = DPBasedShardingSolver(_FakeOptimizer(graph))

    topology = solver.build_topological_order()

    assert all(node.op != "output" for node in topology.nodes)
    assert topology.nodes == [node for node in graph.nodes if node.op != "output"]
    _assert_predecessors_match_graph_indegrees(topology)

    for node, predecessors in topology.predecessors.items():
        node_index = topology.node_to_index[node]
        for pred in predecessors:
            assert topology.node_to_index[pred] < node_index

    merge = topology.nodes[-1]
    assert [pred.name for pred in topology.predecessors[merge]] == ["add", "mul"]


def test_dp_solver_preserves_duplicate_predecessors():
    class DuplicateInputModule(torch.nn.Module):
        def forward(self, x):
            return x + x

    graph = torch.fx.symbolic_trace(DuplicateInputModule()).graph
    solver = DPBasedShardingSolver(_FakeOptimizer(graph))

    topology = solver.build_topological_order()
    _assert_predecessors_match_graph_indegrees(topology)

    add_node = next(node for node in topology.nodes if node.op == "call_function")
    predecessors = topology.predecessors[add_node]
    assert len(predecessors) == 2
    assert predecessors[0] is predecessors[1]
    assert predecessors[0].name == "x"


def test_dp_solver_topology_for_tiny_transformer_forward():
    class TinyTransformerBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q = torch.nn.Linear(8, 8)
            self.k = torch.nn.Linear(8, 8)
            self.v = torch.nn.Linear(8, 8)
            self.o = torch.nn.Linear(8, 8)
            self.ff1 = torch.nn.Linear(8, 16)
            self.ff2 = torch.nn.Linear(16, 8)

        def forward(self, x):
            q = self.q(x)
            k = self.k(x)
            v = self.v(x)
            scores = q @ k.transpose(-2, -1) / math.sqrt(8)
            attn = F.softmax(scores, dim=-1)
            attn_out = attn @ v
            x = x + self.o(attn_out)
            hidden = F.relu(self.ff1(x))
            return x + self.ff2(hidden)

    block = TinyTransformerBlock()
    assert block(torch.randn(2, 4, 8)).shape == (2, 4, 8)

    graph = torch.fx.symbolic_trace(block).graph
    solver = DPBasedShardingSolver(_FakeOptimizer(graph))

    topology = solver.build_topological_order()
    _assert_predecessors_match_graph_indegrees(topology)
    node_names = [node.name for node in topology.nodes]

    assert node_names == [
        "x",
        "q",
        "k",
        "v",
        "transpose",
        "matmul",
        "truediv",
        "softmax",
        "matmul_1",
        "o",
        "add",
        "ff1",
        "relu",
        "ff2",
        "add_1",
    ]

    add_nodes = [node for node in topology.nodes if node.target is operator.add]
    assert [node.name for node in add_nodes] == ["add", "add_1"]
    assert [pred.name for pred in topology.predecessors[add_nodes[0]]] == ["x", "o"]
    assert [pred.name for pred in topology.predecessors[add_nodes[1]]] == [
        "add",
        "ff2",
    ]


def test_dp_solver_solution_is_not_implemented():
    class SimpleModule(torch.nn.Module):
        def forward(self, x):
            return x + 1

    graph = torch.fx.symbolic_trace(SimpleModule()).graph
    solver = DPBasedShardingSolver(_FakeOptimizer(graph))

    with pytest.raises(NotImplementedError, match="only builds topological order"):
        solver.get_solution()
