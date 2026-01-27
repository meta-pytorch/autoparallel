# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for log_formatting module."""

import unittest

import torch
import torch.fx

from autoparallel.log_formatting import format_sharding_log


class MockStrategy:
    """Mock strategy object for testing."""

    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return self.name


def make_opt_entry(
    strat_name: str,
    comm_cost: float = 0.0,
    compute_cost: float = 0.0,
    transition_cost: float = 0.0,
) -> dict:
    """Helper to create an optimization entry."""
    return {
        "full_strat": MockStrategy(strat_name),
        "comm_cost": comm_cost,
        "compute_cost": compute_cost,
        "sharding_transition_cost": transition_cost,
        "cost": comm_cost + compute_cost + transition_cost,
    }


class TestFormatShardingLog(unittest.TestCase):
    def test_simple_graph_with_annotations(self):
        """Test that annotations appear on correct lines for a simple graph."""

        # Create a simple graph: y = x + 1
        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        traced = torch.fx.symbolic_trace(SimpleModule())
        graph = traced.graph

        nodes = list(graph.nodes)
        # nodes[0] is placeholder 'x', nodes[1] is add, nodes[2] is output
        x_node = nodes[0]
        add_node = nodes[1]

        opt = {
            x_node: [make_opt_entry("Replicate()", comm_cost=0.0)],
            add_node: [make_opt_entry("Shard(0)", comm_cost=1.0, compute_cost=2.0)],
        }

        result = format_sharding_log(graph, opt)

        # Check that the result contains expected annotations
        self.assertIn("# placement=Replicate()", result)
        self.assertIn("# placement=Shard(0)", result)
        self.assertIn("cost=[(1.0, 2.0, 0.0)]", result)

        # Check cost summary
        self.assertIn("total_cost: 3.00", result)
        self.assertIn("comm_cost: 1.00", result)
        self.assertIn("compute_cost: 2.00", result)
        self.assertIn("transition_cost: 0.00", result)

    def test_multiple_operations(self):
        """Test graph with multiple operations."""

        class MultiOpModule(torch.nn.Module):
            def forward(self, x, y):
                a = x + y
                b = a * 2
                return b

        traced = torch.fx.symbolic_trace(MultiOpModule())
        graph = traced.graph

        nodes = list(graph.nodes)
        # nodes: x, y, add, mul, output
        x_node, y_node, add_node, mul_node, _ = nodes

        opt = {
            x_node: [make_opt_entry("Replicate()")],
            y_node: [make_opt_entry("Shard(0)")],
            add_node: [make_opt_entry("Shard(0)", compute_cost=1.0)],
            mul_node: [make_opt_entry("Shard(0)", compute_cost=0.5)],
        }

        result = format_sharding_log(graph, opt)

        # All nodes should have annotations
        self.assertIn("# placement=Replicate()", result)
        # Should have Shard(0) annotations for y, add, and mul
        self.assertEqual(result.count("# placement=Shard(0)"), 3)

        # Check total costs
        self.assertIn("total_cost: 1.50", result)
        self.assertIn("compute_cost: 1.50", result)

    def test_cost_accumulation(self):
        """Test that costs are correctly accumulated across all nodes."""

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        traced = torch.fx.symbolic_trace(SimpleModule())
        graph = traced.graph

        nodes = list(graph.nodes)
        x_node = nodes[0]
        add_node = nodes[1]

        opt = {
            x_node: [
                make_opt_entry(
                    "R", comm_cost=1.0, compute_cost=2.0, transition_cost=0.5
                )
            ],
            add_node: [
                make_opt_entry(
                    "S", comm_cost=3.0, compute_cost=4.0, transition_cost=1.5
                )
            ],
        }

        result = format_sharding_log(graph, opt)

        # Total: 1+2+0.5 + 3+4+1.5 = 12.0
        self.assertIn("total_cost: 12.00", result)
        self.assertIn("comm_cost: 4.00", result)  # 1 + 3
        self.assertIn("compute_cost: 6.00", result)  # 2 + 4
        self.assertIn("transition_cost: 2.00", result)  # 0.5 + 1.5

    def test_multiple_outputs_per_node(self):
        """Test nodes with multiple optimization entries (e.g., getitem)."""

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        traced = torch.fx.symbolic_trace(SimpleModule())
        graph = traced.graph

        nodes = list(graph.nodes)
        add_node = nodes[1]

        # Node with multiple optimization entries
        opt = {
            add_node: [
                make_opt_entry("Shard(0)", comm_cost=1.0),
                make_opt_entry("Shard(1)", comm_cost=2.0),
            ],
        }

        result = format_sharding_log(graph, opt)

        # First strategy should be shown in placement
        self.assertIn("# placement=Shard(0)", result)
        # Cost should show both entries
        self.assertIn("[(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]", result)
        # Total should include both
        self.assertIn("total_cost: 3.00", result)

    def test_violated_constraints_log(self):
        """Test that violated constraints log is appended."""

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        traced = torch.fx.symbolic_trace(SimpleModule())
        graph = traced.graph

        opt = {}
        violated = (
            "WARNING: Constraint X was violated\nWARNING: Constraint Y was violated"
        )

        result = format_sharding_log(graph, opt, violated_constraints_log=violated)

        self.assertIn("WARNING: Constraint X was violated", result)
        self.assertIn("WARNING: Constraint Y was violated", result)

    def test_empty_opt_dict(self):
        """Test with empty optimization dict - should still produce valid output."""

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        traced = torch.fx.symbolic_trace(SimpleModule())
        graph = traced.graph

        opt = {}

        result = format_sharding_log(graph, opt)

        # Should have the function definition
        self.assertIn("def forward", result)
        # Should have zero costs
        self.assertIn("total_cost: 0.00", result)

    def test_colored_output(self):
        """Test that colored output includes ANSI codes."""

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        traced = torch.fx.symbolic_trace(SimpleModule())
        graph = traced.graph

        nodes = list(graph.nodes)
        add_node = nodes[1]

        opt = {
            add_node: [make_opt_entry("Shard(0)")],
        }

        result = format_sharding_log(graph, opt, colored=True)

        # ANSI escape codes start with \x1b[
        self.assertIn("\x1b[", result)

    def test_verbose_output(self):
        """Test that verbose output includes additional information."""

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        traced = torch.fx.symbolic_trace(SimpleModule())
        graph = traced.graph

        nodes = list(graph.nodes)
        add_node = nodes[1]

        opt = {
            add_node: [make_opt_entry("Shard(0)")],
        }

        result_normal = format_sharding_log(graph, opt, verbose=False)
        result_verbose = format_sharding_log(graph, opt, verbose=True)

        # Verbose output should be longer (contains type annotations, etc.)
        self.assertGreater(len(result_verbose), len(result_normal))

    def test_shard_order_attribute(self):
        """Test that shard_order attribute is included when present."""

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        traced = torch.fx.symbolic_trace(SimpleModule())
        graph = traced.graph

        nodes = list(graph.nodes)
        add_node = nodes[1]

        # Set shard_order on the node's meta
        add_node.meta["shard_order"] = (1, 0)

        opt = {
            add_node: [make_opt_entry("Shard(0)")],
        }

        result = format_sharding_log(graph, opt)

        self.assertIn("shard_order=(1, 0)", result)

    def test_annotation_on_correct_line(self):
        """Test that annotations appear on the line where node is assigned."""

        class MultiLineModule(torch.nn.Module):
            def forward(self, x):
                a = x + 1
                b = a * 2
                c = b - 3
                return c

        traced = torch.fx.symbolic_trace(MultiLineModule())
        graph = traced.graph

        nodes = list(graph.nodes)
        # x, add, mul, sub, output
        add_node = nodes[1]
        mul_node = nodes[2]
        sub_node = nodes[3]

        opt = {
            add_node: [make_opt_entry("Strategy_A")],
            mul_node: [make_opt_entry("Strategy_B")],
            sub_node: [make_opt_entry("Strategy_C")],
        }

        result = format_sharding_log(graph, opt)
        lines = result.split("\n")

        # Find lines with each strategy and verify they're on the right operation
        for line in lines:
            if "Strategy_A" in line:
                # Should be on a line with add operation
                self.assertIn("add", line.lower())
            if "Strategy_B" in line:
                # Should be on a line with mul operation
                self.assertIn("mul", line.lower())
            if "Strategy_C" in line:
                # Should be on a line with sub operation
                self.assertIn("sub", line.lower())

    def test_placeholder_annotations(self):
        """Test that placeholder nodes get annotated correctly."""

        class TwoInputModule(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        traced = torch.fx.symbolic_trace(TwoInputModule())
        graph = traced.graph

        nodes = list(graph.nodes)
        x_node = nodes[0]
        y_node = nodes[1]

        opt = {
            x_node: [make_opt_entry("Replicate()")],
            y_node: [make_opt_entry("Shard(0)")],
        }

        result = format_sharding_log(graph, opt)

        # Both placeholders should have annotations
        self.assertIn("Replicate()", result)
        self.assertIn("Shard(0)", result)

    def test_no_output_node_annotation(self):
        """Test that output nodes are not annotated even if in opt dict."""

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        traced = torch.fx.symbolic_trace(SimpleModule())
        graph = traced.graph

        nodes = list(graph.nodes)
        output_node = nodes[-1]

        # Try to add output node to opt (should be ignored)
        opt = {
            output_node: [make_opt_entry("ShouldNotAppear")],
        }

        result = format_sharding_log(graph, opt)

        self.assertNotIn("ShouldNotAppear", result)


class TestFormatShardingLogEdgeCases(unittest.TestCase):
    def test_node_not_in_opt(self):
        """Test that nodes not in opt dict are skipped gracefully."""

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                a = x + 1
                b = a * 2
                return b

        traced = torch.fx.symbolic_trace(SimpleModule())
        graph = traced.graph

        nodes = list(graph.nodes)
        # Only annotate one of the operations
        add_node = nodes[1]

        opt = {
            add_node: [make_opt_entry("OnlyThis")],
        }

        result = format_sharding_log(graph, opt)

        # Should have exactly one placement annotation
        self.assertEqual(result.count("# placement="), 1)
        self.assertIn("OnlyThis", result)

    def test_complex_strategy_string(self):
        """Test that complex strategy strings are handled correctly."""

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        traced = torch.fx.symbolic_trace(SimpleModule())
        graph = traced.graph

        nodes = list(graph.nodes)
        add_node = nodes[1]

        complex_strat = "(Shard(0), Replicate(), Partial(sum))"
        opt = {
            add_node: [make_opt_entry(complex_strat)],
        }

        result = format_sharding_log(graph, opt)

        self.assertIn(complex_strat, result)

    def test_zero_costs(self):
        """Test handling of zero costs."""

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        traced = torch.fx.symbolic_trace(SimpleModule())
        graph = traced.graph

        nodes = list(graph.nodes)
        add_node = nodes[1]

        opt = {
            add_node: [make_opt_entry("S", comm_cost=0.0, compute_cost=0.0)],
        }

        result = format_sharding_log(graph, opt)

        self.assertIn("total_cost: 0.00", result)
        self.assertIn("[(0.0, 0.0, 0.0)]", result)

    def test_large_costs(self):
        """Test handling of large cost values."""

        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        traced = torch.fx.symbolic_trace(SimpleModule())
        graph = traced.graph

        nodes = list(graph.nodes)
        add_node = nodes[1]

        opt = {
            add_node: [
                make_opt_entry(
                    "S", comm_cost=1e10, compute_cost=2e10, transition_cost=3e10
                )
            ],
        }

        result = format_sharding_log(graph, opt)

        self.assertIn("total_cost: 60000000000.00", result)


if __name__ == "__main__":
    unittest.main()
