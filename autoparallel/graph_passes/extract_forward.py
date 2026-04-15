# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch.fx
import torch.utils._pytree as pytree


def _reachable_from_output(graph: torch.fx.Graph) -> set[torch.fx.Node]:
    """Return the set of nodes transitively reachable from the output node."""
    output_node = next(n for n in graph.nodes if n.op == "output")
    reachable: set[torch.fx.Node] = set()
    worklist = [output_node]
    while worklist:
        node = worklist.pop()
        if node in reachable:
            continue
        reachable.add(node)
        for inp in node.all_input_nodes:
            worklist.append(inp)
    return reachable


def extract_forward_graph(
    joint_gm: torch.fx.GraphModule, num_fwd_outputs: int
) -> torch.fx.GraphModule:
    """Extract a forward-only graph from a joint forward+backward graph.

    Trims the output to forward-only entries, then removes all nodes
    not reachable from the trimmed output (backward nodes, tangent
    placeholders, backward-only collectives, saved activations).
    """
    gm = copy.deepcopy(joint_gm)
    graph = gm.graph

    # The joint graph output is nested: ((fwd_outs...), (bwd_outs...)).
    # Flatten to individual leaf nodes, keep only the first
    # num_fwd_outputs leaves, and set them as a flat output tuple.
    output_node = next(n for n in graph.nodes if n.op == "output")
    flat_outputs = pytree.arg_tree_leaves(*output_node.args)
    fwd_outputs = flat_outputs[:num_fwd_outputs]
    output_node.args = (tuple(fwd_outputs),)

    # Compute reachability from the trimmed output and remove
    # everything else (backward nodes, tangent placeholders,
    # backward-only collectives, etc.).
    reachable = _reachable_from_output(graph)
    for node in reversed(list(graph.nodes)):
        if node not in reachable:
            node.replace_all_uses_with(None)
            graph.erase_node(node)

    gm.recompile()
    return gm
