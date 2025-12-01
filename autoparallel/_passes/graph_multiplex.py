# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch
import torch.fx as fx


def multiplex_fw_bw_graph(
    fw_gm: fx.GraphModule, bw_gm: fx.GraphModule
) -> fx.GraphModule:
    """
    Multiplexes forward and backward graphs into a single unified graph module.

    This function combines a forward graph and a backward graph into one multiplexed
    graph by merging their nodes and outputs. The resulting graph has:
    - All placeholders from both forward and backward graphs (backward followed by forward)
    - All computation nodes from both graphs (backward followed by forward)
    - Combined outputs (backward outputs followed by forward outputs)

    Args:
        fw_gm: The forward graph module containing the forward computation
        bw_gm: The backward graph module containing the backward computation

    Returns:
        A multiplexed fx.GraphModule containing both forward and backward computations
        with backward outputs appearing before forward outputs

    Note:
        The function preserves node metadata during the merging process.
    """
    # Mapping to track correspondence between backward graph nodes and new nodes
    old_node_to_new_node: dict[torch.fx.Node, torch.fx.Node] = {}

    # Start with a deep copy of the forward graph as the base
    multiplexed_gm = copy.deepcopy(fw_gm)

    # Collect all placeholder nodes from the backward graph
    bw_placeholders = bw_gm.graph.find_nodes(op="placeholder")
    fw_placeholders = fw_gm.graph.find_nodes(op="placeholder")

    # Insert backward placeholders at the beginning of the multiplexed graph
    # Reversed order ensures correct execution sequence
    with multiplexed_gm.graph.inserting_before():
        for n in reversed(bw_placeholders):
            new_placeholder = multiplexed_gm.graph.placeholder(n.name)
            new_placeholder.meta = n.meta
            new_placeholder.target = new_placeholder.name
            old_node_to_new_node[n] = new_placeholder

    # Find the last placeholder and the output node in the multiplexed graph
    multiplxed_gm_placeholders = multiplexed_gm.graph.find_nodes(op="placeholder")
    assert len(multiplxed_gm_placeholders) == (
        len(fw_placeholders) + len(bw_placeholders)
    )
    insert_point = multiplxed_gm_placeholders[-1]

    # Copy all computation nodes from backward graph into multiplexed graph
    fw_outputs = fw_gm.graph.find_nodes(op="output")
    bw_outputs = bw_gm.graph.find_nodes(op="output")
    assert len(bw_outputs) == 1 and len(fw_outputs) == 1
    bw_graph_op_node = bw_outputs[0]
    for n in bw_gm.graph.nodes:
        if n.op == "placeholder":
            continue
        if n.op == "output":
            continue
        with multiplexed_gm.graph.inserting_after(insert_point):
            # Copy node and remap its arguments using the node mapping
            new_node = multiplexed_gm.graph.node_copy(
                n, lambda x: old_node_to_new_node[x]
            )
            new_node.meta = n.meta
            old_node_to_new_node[n] = new_node
            insert_point = new_node

    # Collect output arguments from backward graph, remapping to new nodes
    bw_op_node_args = [
        old_node_to_new_node[n] if n is not None else None
        for n in bw_graph_op_node.args[0]
    ]

    # Collect output arguments from multiplexed graph (will contain only fwd_outs)
    multiplexed_graph_outputs = multiplexed_gm.graph.find_nodes(op="output")
    assert len(multiplexed_graph_outputs) == 1
    multiplexed_graph_op_node = multiplexed_graph_outputs[0]
    fw_op_node_args = list(multiplexed_graph_op_node.args[0])

    # Update output node args to prepend backward outputs before forward outputs
    multiplexed_graph_op_node.args = (tuple(bw_op_node_args + fw_op_node_args),)

    multiplexed_gm.graph.eliminate_dead_code()
    multiplexed_gm.graph.lint()
    multiplexed_gm.recompile()
    return multiplexed_gm
