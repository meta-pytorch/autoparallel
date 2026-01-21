# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Log formatting utilities for sharding optimization results.

This module provides functions to format and display the sharding optimization
results in a human-readable format, annotating the FX graph with placement
and cost information.
"""

from typing import Any

import torch.fx


def format_sharding_log(
    graph: torch.fx.Graph,
    opt: dict[torch.fx.Node, list[dict[str, Any]]],
    colored: bool = False,
    verbose: bool = False,
    violated_constraints_log: str = "",
) -> str:
    """
    Format the sharding optimization results as annotated Python code.

    This function takes an FX graph and the optimization results, and produces
    a string representation of the graph with placement and cost annotations
    for each node.

    Args:
        graph: The FX graph to format.
        opt: Dictionary mapping nodes to their optimization results. Each value
            is a list of dicts containing:
            - "full_strat": The strategy object
            - "comm_cost": Communication cost
            - "compute_cost": Computation cost
            - "sharding_transition_cost": Cost of sharding transitions
            - "cost": Total cost
        colored: Whether to use ANSI color codes in the output.
        verbose: Whether to include verbose information (shapes, stack traces).
        violated_constraints_log: Optional string with violated constraints info.

    Returns:
        A string containing the annotated Python code representation of the graph.
    """
    from torch.fx.graph import _color_fns, _identity

    nodes = list(graph.nodes)

    # Get PythonCode object which includes the lineno_map for robust line mapping
    python_code = graph.python_code("self", colored=colored, verbose=verbose)
    # Don't strip - we need to preserve line numbers that match lineno_map
    code_lines = python_code.src.split("\n")
    lineno_map = python_code._lineno_map

    # The lineno_map keys are line numbers relative to prologue start (1-indexed).
    # _prologue_start tells us which line in fn_code the prologue begins.
    # So lineno_map[k] -> node_idx means:
    #   - k=1 is the prologue line (def forward...)
    #   - k=2,3,... are body lines
    # The actual line index in code_lines (0-indexed) is: _prologue_start + k - 2
    # (since _prologue_start is 1-indexed and k starts at 1 for prologue)
    prologue_start = python_code._prologue_start

    # Build reverse mapping: node_index -> list of line numbers (0-indexed in code_lines)
    node_to_lines: dict[int, list[int]] = {}
    if lineno_map is not None:
        for lineno, node_idx in lineno_map.items():
            if node_idx is not None:
                # Convert from lineno_map's numbering to 0-indexed code_lines index
                # lineno is 1-indexed from prologue, prologue_start is 1-indexed in fn_code
                line_idx = prologue_start + lineno - 2
                node_to_lines.setdefault(node_idx, []).append(line_idx)

    if colored:
        txt_color = _color_fns["blue"]
        attr_color = _color_fns["red"]
    else:
        txt_color = _identity
        attr_color = _identity

    plc_txt = txt_color("# placement=")
    cost_txt = txt_color(", cost=")

    def is_node_assignment_line(line: str, node_repr: str) -> bool:
        """Check if this line is where the node is assigned (not just mentioned)."""
        stripped = line.lstrip()
        # Check for patterns like "node_name: " or "node_name = " or "node_name,"
        # These indicate the node is being defined/assigned on this line
        if stripped.startswith(node_repr + ":"):
            return True
        if stripped.startswith(node_repr + " ="):
            return True
        if stripped.startswith(node_repr + ","):
            return True
        if stripped.startswith(node_repr + " "):
            return True
        if stripped == node_repr:
            return True
        return False

    # Track lines we've already annotated to avoid duplicates
    annotated_lines: set[int] = set()

    # For placeholders, we'll collect annotations to add as comments
    placeholder_annotations: list[str] = []

    # Track totals for summary
    total_cost = 0.0
    total_comm_cost = 0.0
    total_compute_cost = 0.0
    total_transition_cost = 0.0

    for node_idx, node in enumerate(nodes):
        if node.op == "output":
            continue
        if node not in opt:
            continue

        d = opt[node]

        # Accumulate costs
        for entry in d:
            total_cost += entry.get("cost", 0.0)
            total_comm_cost += entry.get("comm_cost", 0.0)
            total_compute_cost += entry.get("compute_cost", 0.0)
            total_transition_cost += entry.get("sharding_transition_cost", 0.0)

        strat = str(d[0]["full_strat"])
        costs = [
            (x["comm_cost"], x["compute_cost"], x["sharding_transition_cost"])
            for x in d
        ]
        shard_order = node.meta.get("shard_order")
        if shard_order:
            annotation = f"  {plc_txt}{attr_color(strat)} {shard_order=} {cost_txt}{attr_color(str(costs))}"
        else:
            annotation = (
                f"  {plc_txt}{attr_color(strat)} {cost_txt}{attr_color(str(costs))}"
            )

        node_repr = repr(node)

        if node.op == "placeholder":
            # Placeholders may be on a shared line (pytree unpacking)
            # or have their own line. Try to find the right line first.
            found = False
            if lineno_map is not None and node_idx in node_to_lines:
                for line_idx in sorted(node_to_lines[node_idx]):
                    if 0 <= line_idx < len(code_lines):
                        line = code_lines[line_idx]
                        # Skip comment lines
                        if line.lstrip().startswith("#"):
                            continue
                        # Check if this is the assignment line for this node
                        if is_node_assignment_line(line, node_repr):
                            if line_idx not in annotated_lines:
                                code_lines[line_idx] += annotation
                                annotated_lines.add(line_idx)
                            found = True
                            break
            if not found:
                # Fallback: collect as a comment to insert
                placeholder_annotations.append(f"    # {node.name}:{annotation}")
            continue

        # For call_function nodes, find the line where the node is assigned
        found = False
        if lineno_map is not None and node_idx in node_to_lines:
            for line_idx in sorted(node_to_lines[node_idx]):
                if 0 <= line_idx < len(code_lines):
                    line = code_lines[line_idx]
                    # Skip comment lines
                    if line.lstrip().startswith("#"):
                        continue
                    # Check if this is the assignment line for this node
                    if is_node_assignment_line(line, node_repr):
                        if line_idx not in annotated_lines:
                            code_lines[line_idx] += annotation
                            annotated_lines.add(line_idx)
                        found = True
                        break

        if not found:
            # Fallback: search all lines for the node assignment
            for line_idx, line in enumerate(code_lines):
                if line.lstrip().startswith("#"):
                    continue
                if is_node_assignment_line(line, node_repr):
                    if line_idx not in annotated_lines:
                        code_lines[line_idx] += annotation
                        annotated_lines.add(line_idx)
                    break

    # Insert placeholder annotations after the function definition line
    if placeholder_annotations:
        # Find the first line after "def forward" that's not empty
        insert_idx = 1
        for i, line in enumerate(code_lines):
            if line.lstrip().startswith("def "):
                insert_idx = i + 1
                break
        # Insert in reverse order so they appear in the correct order
        for ann in reversed(placeholder_annotations):
            code_lines.insert(insert_idx, ann)

    code = "\n".join(code_lines)
    code += f"\ntotal_cost: {total_cost:.2f}"
    code += f"\n  comm_cost: {total_comm_cost:.2f}"
    code += f"\n  compute_cost: {total_compute_cost:.2f}"
    code += f"\n  transition_cost: {total_transition_cost:.2f}"
    if violated_constraints_log:
        code += "\n" + violated_constraints_log
    return code
