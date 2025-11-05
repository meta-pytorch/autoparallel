# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch.fx as fx
from functorch.compile import default_partition

# we are running the default partitioner on the bw graph, which requires AC tags being removed.
# At this stage we have already finished running AC anyway, since we have a bw graph


def remove_recompute_tags(bw_gm):
    for n in bw_gm.graph.nodes:
        if "recompute" in n.meta:
            del n.meta["recompute"]


# We are using the default partitioner to split our backward into dI and dW subgraphs.
# We want to generate the dI subgraph *first*, because:
# - in pipelining we generally want to schedule dI compute before dW
# - the dI compute will potentially compute more activations that we need to plumb into dW compute
# Today, the default partitioner requires that your split on the first K outputs of your combined graph.
# So here, we reorder the outputs of the backward so grad_inputs are first.


def reorder_output_grads(bw_gm, num_weight_gradients):
    outputs = bw_gm.graph.find_nodes(op="output")
    assert len(outputs) == 1
    output = outputs[0]
    assert isinstance(output.args[0], tuple)
    grad_weights, grad_inputs = (
        output.args[0][:num_weight_gradients],
        output.args[0][num_weight_gradients:],
    )
    new_out_tuple = grad_inputs + grad_weights
    with bw_gm.graph.inserting_after(output):
        # TODO: also set the new node's meta properly
        new_out = bw_gm.graph.output(new_out_tuple)
    output.replace_all_uses_with(new_out)
    bw_gm.graph.erase_node(output)
    return len(grad_inputs)


# TODO: in theory we can infer num_weight_gradients from the graph metadata directly


def split_di_dw_graph(
    bw_gm: fx.GraphModule, *, num_weight_gradients
) -> tuple[fx.GraphModule, fx.GraphModule]:
    # we could consider doing this is a non-mutating way
    bw_gm = copy.deepcopy(bw_gm)
    remove_recompute_tags(bw_gm)
    num_input_gradients = reorder_output_grads(bw_gm, num_weight_gradients)
    bw_gm.recompile()

    args = [x.meta["val"] for x in bw_gm.graph.find_nodes(op="placeholder")]

    bw_inputs, bw_weights = default_partition(
        bw_gm, args, num_fwd_outputs=num_input_gradients
    )
    return bw_inputs, bw_weights
