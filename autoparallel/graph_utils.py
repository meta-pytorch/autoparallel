# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

import torch
from torch._functorch.aot_autograd import JointWithDescriptors
from torch._inductor.fx_passes.joint_graph import patterns
from torch._inductor.fx_passes.post_grad import remove_assert_ops, remove_noop_ops
from torch._inductor.pattern_matcher import stable_topological_sort
from torch.fx import GraphModule
from torch.fx.experimental._backward_state import BackwardState


def cleanup_graph(gm: torch.fx.GraphModule, aggressive: bool = False) -> None:
    # TODO: we can switch the default "aggresive" to True and things should
    # be even better as we can remove more redundant nodes early on
    # I'm keeping compatibility with previous behavior for now, and will
    # switch the flag in the future

    # TODO: Make the DCE match exactly the AOTAutograd logic, I don't
    # think I trust the default FX DCE logic
    gm.graph.eliminate_dead_code()
    gm.recompile()
    remove_noop_ops(gm.graph)
    # TODO: We shouldn't actually remove these
    remove_assert_ops(gm.graph)
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()

    if aggressive:
        maybe_count = patterns.apply(gm)
        if maybe_count is not None:
            stable_topological_sort(gm.graph)
            gm.graph.lint()
            gm.recompile()


def update_joint_with_descriptors(
    joint_with_descriptors: JointWithDescriptors,
    updated_gm: GraphModule,
) -> None:
    """
    Assuming we have transformed updated_gm since the time it was captured,
    (e.g. by parallelizing it),
    this util updates the joint_with_descriptors struct to reference the new gm, and
    updates any copies of tensor meta/shape stored in joint_with_descriptors relating to input arguments,
    which may have changed shape since the initial trace.
    """
    # TODO: should we upstream a util like this?
    placeholders = [n for n in updated_gm.graph.nodes if n.op == "placeholder"]
    new_local_args = [n.meta["val"] for n in placeholders]
    joint_with_descriptors.graph_module = updated_gm
    joint_with_descriptors._aot_graph_capture.graph_module = updated_gm

    new_flat_args: list[Union[torch.Tensor, int, torch.SymInt, BackwardState]] = []
    for orig, new in zip(joint_with_descriptors._aot_state.flat_args, new_local_args):
        if isinstance(orig, torch.nn.Parameter):
            new_flat_args.append(torch.nn.Parameter(new))
        else:
            new_flat_args.append(new)

    tangent_idx = len(joint_with_descriptors._aot_state.flat_args)
    new_local_tangents = new_local_args[tangent_idx:]
    joint_with_descriptors._aot_graph_capture.updated_flat_args = (
        new_flat_args,
        new_local_tangents,
    )
    joint_with_descriptors._aot_state.flat_args = new_flat_args
    joint_with_descriptors._aot_state.fw_metadata.traced_tangents = new_local_tangents


def _add_alias(gm):
    """
    Helper function to add alias nodes to every node in the graph
    this gives more configuration opportunities
    """
    graph = gm.graph

    nodes = [n for n in graph.nodes if n.op == "call_function"]
    node_map = {node: idx for idx, node in enumerate(nodes)}
    inputs = graph.find_nodes(op="placeholder")
    for node in inputs:
        if len(node.users) == 0:
            # node is not used, don't add alias for it
            continue
        first_user = nodes[min(node_map[n] for n in node.users)]
        with graph.inserting_before(first_user):
            alias_node = graph.call_function(torch.ops.aten.alias.default, args=(node,))
            alias_node.meta.update(node.meta)

            def delete_user_cb(n):
                return n != alias_node

            node.replace_all_uses_with(alias_node, delete_user_cb=delete_user_cb)

    """
    for node in nodes:
        # skip ops which return tuple
        if not isinstance(node.meta["val"], torch.Tensor):
            continue
        with graph.inserting_after(node):
            alias_node = graph.call_function(torch.ops.aten.alias.default, args=(node,))
            alias_node.meta.update(node.meta)

            def delete_user_cb(n):
                return n != alias_node

            node.replace_all_uses_with(alias_node, delete_user_cb=delete_user_cb)

    """

    for node in graph.find_nodes(op="output")[0].all_input_nodes:
        with graph.inserting_after(node):
            alias_node = graph.call_function(torch.ops.aten.alias.default, args=(node,))
            alias_node.meta.update(node.meta)

            def delete_user_cb(n):
                return n != alias_node

            node.replace_all_uses_with(alias_node, delete_user_cb=delete_user_cb)

    gm.recompile()
    return gm
