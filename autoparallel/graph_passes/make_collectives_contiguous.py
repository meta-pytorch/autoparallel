# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch

_COLLECTIVES_REQUIRING_CONTIGUOUS = {
    torch.ops._c10d_functional.all_gather_into_tensor.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
}


def make_collectives_contiguous(gm: torch.fx.GraphModule) -> None:
    """Insert clone(memory_format=contiguous) before collectives that require it.

    NCCL collectives like all_gather_into_tensor and reduce_scatter_tensor
    require contiguous input tensors. When AP inserts these collectives, the
    input may be non-contiguous (e.g. after a transpose or view). This pass
    walks the graph and inserts a contiguous clone on any such input.
    """
    graph = gm.graph
    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or node.target not in _COLLECTIVES_REQUIRING_CONTIGUOUS
        ):
            continue
        tensor_arg = node.args[0]
        if not isinstance(tensor_arg, torch.fx.Node):
            continue
        # Skip if the input is already a contiguous clone
        if (
            tensor_arg.op == "call_function"
            and tensor_arg.target == torch.ops.aten.clone.default
            and len(tensor_arg.kwargs) > 0
            and tensor_arg.kwargs.get("memory_format") == torch.contiguous_format
        ):
            continue
        with graph.inserting_before(node):
            clone_node = graph.call_function(
                torch.ops.aten.clone.default,
                args=(tensor_arg,),
                kwargs={"memory_format": torch.contiguous_format},
            )
            clone_node.meta.update(tensor_arg.meta)
            node.replace_input_with(tensor_arg, clone_node)
    gm.recompile()
