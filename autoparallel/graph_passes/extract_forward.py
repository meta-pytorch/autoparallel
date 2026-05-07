# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch.fx
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import unset_fake_temporarily


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
    joint_gm: torch.fx.GraphModule,
    num_fwd_outputs: int,
    num_primals: int | None = None,
) -> torch.fx.GraphModule:
    """Extract a forward-only graph from a joint forward+backward graph.

    Trims the output to forward-only entries, then removes all nodes
    not reachable from the trimmed output (backward nodes, tangent
    placeholders, backward-only collectives, saved activations).
    """
    # FakeTensorMode may still be active (pushed onto the ExitStack by
    # aot_export_joint_with_descriptors during __enter__).  deepcopy
    # clones tensor storage, and FakeTensorMode intercepts the
    # aten.set_.source_Storage call, which fails when the cloned
    # (real) storage device doesn't match the original (meta) device.
    with unset_fake_temporarily():
        gm = copy.deepcopy(joint_gm)
    graph = gm.graph

    # The joint graph output is nested: ((fwd_outs...), (bwd_outs...)).
    # Flatten to individual leaf nodes, keep only the first
    # num_fwd_outputs leaves, and set them as a flat output tuple.
    output_node = next(n for n in graph.nodes if n.op == "output")
    flat_outputs = pytree.arg_tree_leaves(*output_node.args)
    fwd_outputs = flat_outputs[:num_fwd_outputs]
    output_node.args = (tuple(fwd_outputs),)

    # Identify the first num_primals placeholders — these are function
    # inputs (params, buffers, user inputs) that must stay to match the
    # calling convention, even when unused (e.g. unused parameters).
    # The remaining placeholders are tangents (backward-only inputs).
    if num_primals is not None:
        placeholders = [n for n in graph.nodes if n.op == "placeholder"]
        primal_nodes = set(placeholders[:num_primals])
    else:
        primal_nodes = set()

    # Compute reachability from the trimmed output and remove
    # everything else, except primal placeholders.
    reachable = _reachable_from_output(graph)
    for node in reversed(list(graph.nodes)):
        if node not in reachable and node not in primal_nodes:
            node.replace_all_uses_with(None)
            graph.erase_node(node)

    gm.recompile()
    return gm
