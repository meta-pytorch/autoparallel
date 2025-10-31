# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch.fx as fx
import torch.utils._pytree as pytree
from torch._functorch._aot_autograd.descriptors import AOTOutput

# TODO(ivankobzarev): Remove parititoner function fork once https://github.com/pytorch/pytorch/pull/166725 is landed


class InvalidNodeBase:
    def __repr__(self):
        return "Invalid Node"


InvalidNode = InvalidNodeBase()


def _extract_graph_with_inputs_outputs(
    joint_graph: fx.Graph,
    inputs: list[fx.Node],
    outputs: list[fx.Node],
    outputs_descs: list[AOTOutput],
    subgraph: Optional[str] = None,
    ignore_must_be_in_fw_bw: bool = False,
) -> fx.Graph:
    """
    Given a graph, extracts out a subgraph that takes the specified nodes as
    inputs and returns the specified outputs.

    This includes specifying non-placeholder nodes as inputs.

    The general strategy is to initialize all inputs with proxies as we
    encounter them, and trace through the graph, only keeping values which take
    in valid proxies. Then, all dead code is eliminated.
    """
    new_graph = fx.Graph()
    env = {}

    # Add new placeholder nodes in the order specified by the inputs
    for node in inputs:
        new_node = new_graph.placeholder(node.name)
        # Can't use node_copy here as we may be turning previous call_function into placeholders
        new_node.meta = node.meta
        # pyrefly: ignore [unsupported-operation]
        env[node] = new_node

    for node in joint_graph.nodes:
        if node in env:
            # Node must be one of our inputs. (Any member of env which wasn't an
            # input to start must have been created by this loop and won't be in
            # joint_graph.nodes).
            continue
        elif node.op == "placeholder":
            env[node] = InvalidNode  # type: ignore[assignment]
        elif node.op == "call_function":
            all_args = pytree.arg_tree_leaves(*node.args, **node.kwargs)
            all_args = [
                isinstance(env[x], InvalidNodeBase)
                for x in all_args
                if isinstance(x, fx.Node)
            ]
            if any(all_args):
                env[node] = InvalidNode  # type: ignore[assignment]
                continue
            # pyrefly: ignore [unsupported-operation, bad-argument-type]
            env[node] = new_graph.node_copy(node, lambda x: env[x])
        elif node.op == "get_attr":
            # pyrefly: ignore [unsupported-operation, bad-argument-type]
            env[node] = new_graph.node_copy(node, lambda x: env[x])
        elif node.op == "output":
            pass
    output_values = []
    for x in outputs:
        if isinstance(x, fx.Node):
            if x not in env:
                raise RuntimeError(f"Node {x} couldn't be found in env")
            assert not isinstance(
                env[x], InvalidNodeBase
            ), f"Node {x} was invalid, but is output"
            output_values.append(env[x])
        else:
            output_values.append(x)
    out = new_graph.output(tuple(output_values))
    out.meta["desc"] = outputs_descs

    new_graph.eliminate_dead_code()
    new_graph.lint()
    return new_graph
