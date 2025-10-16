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
    bw_placeholders = []
    for n in bw_gm.graph.nodes:
        if n.op == "placeholder":
            bw_placeholders.append(n)

    # Insert backward placeholders at the beginning of the multiplexed graph
    # Reversed order ensures correct execution sequence
    with multiplexed_gm.graph.inserting_before():
        for n in reversed(bw_placeholders):
            new_placeholder = multiplexed_gm.graph.placeholder(n.name)
            new_placeholder.meta = n.meta
            new_placeholder.target = new_placeholder.name
            old_node_to_new_node[n] = new_placeholder

    # Find the last placeholder and the output node in the multiplexed graph
    insert_point = None
    multiplexed_graph_op_node = None
    for n in multiplexed_gm.graph.nodes:
        if n.op == "placeholder":
            insert_point = n
        if n.op == "output":
            multiplexed_graph_op_node = n

    # Copy all computation nodes from backward graph into multiplexed graph
    bw_graph_op_node = None
    for n in bw_gm.graph.nodes:
        if n.op == "placeholder":
            continue
        if n.op == "output":
            bw_graph_op_node = n
            continue
        with multiplexed_gm.graph.inserting_after(insert_point):
            # Copy node and remap its arguments using the node mapping
            new_node = multiplexed_gm.graph.node_copy(
                n, lambda x: old_node_to_new_node[x]
            )
            new_node.meta = n.meta
            old_node_to_new_node[n] = new_node
            insert_point = new_node

    assert bw_graph_op_node is not None
    assert multiplexed_graph_op_node is not None

    # Collect output arguments from backward graph, remapping to new nodes
    bw_op_node_args = [
        old_node_to_new_node[n] if n is not None else None
        for n in bw_graph_op_node.args[0]
    ]

    # Collect output arguments from forward graph
    fw_op_node_args = list(multiplexed_graph_op_node.args[0])

    # Remove the old output node and create new combined output
    insert_point = multiplexed_graph_op_node.prev
    multiplexed_gm.graph.erase_node(multiplexed_graph_op_node)

    # Create combined output with backward outputs first, then forward outputs
    with multiplexed_gm.graph.inserting_after(insert_point):
        multiplexed_gm.graph.output(bw_op_node_args + fw_op_node_args)

    multiplexed_gm.graph.eliminate_dead_code()
    multiplexed_gm.graph.lint()
    multiplexed_gm.recompile()
    return multiplexed_gm
