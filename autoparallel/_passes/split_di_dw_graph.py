# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import copy
import itertools
import operator

import sympy
import torch
import torch.fx as fx
from torch._functorch.partitioners import (
    SavedForBackwardsAOTOutput,
    _extract_fwd_bwd_outputs,
    _extract_graph_with_inputs_outputs,
    _is_backward_state,
    _is_bwd_seed_offset,
    _is_fwd_seed_offset,
    _is_primal,
    _remove_by_name,
    find_symbol_binding_fx_nodes,
    free_symbols,
    is_sym_node,
    is_symbol_binding_fx_node,
)
from torch.utils._ordered_set import OrderedSet

from autoparallel.apply_sharding import rename_placeholder_node

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


# This is a copy of the function used by the default partitioner,
# which does *not* reorder symint activations.
# This is reordering is needed by the custom autograd.Function in AOTDispatcher,
# but isn't needed in our dI/dW splitting since there is no autograd in the loop.
# TODO: provide a way to gt this behavior automatically out of the default partitioner
def _extract_fwd_bwd_modules(
    joint_module: fx.GraphModule,
    saved_values: list[fx.Node],
    saved_sym_nodes: list[fx.Node],
    *,
    num_fwd_outputs: int,
) -> tuple[fx.GraphModule, fx.GraphModule]:
    (
        fwd_outputs,
        bwd_outputs,
        fwd_outputs_descs,
        bwd_outputs_descs,
    ) = _extract_fwd_bwd_outputs(joint_module, num_fwd_outputs=num_fwd_outputs)
    placeholders = joint_module.graph.find_nodes(op="placeholder")
    primal_inputs = [*filter(_is_primal, placeholders)]
    fwd_seed_offset_inputs = [*filter(_is_fwd_seed_offset, placeholders)]
    bwd_seed_offset_inputs = [*filter(_is_bwd_seed_offset, placeholders)]
    backward_state_inputs = [*filter(_is_backward_state, placeholders)]

    bwd_graph = _extract_graph_with_inputs_outputs(
        joint_module.graph,
        saved_values + saved_sym_nodes + bwd_seed_offset_inputs,
        bwd_outputs,
        bwd_outputs_descs,
        "backward",
        ignore_must_be_in_fw_bw=True,
    )

    distributed_enabled = torch.distributed.is_available()

    for node in bwd_graph.find_nodes(op="placeholder"):
        # This is to filter out saved values that don't actually end up being used by the backwards pass
        if not node.users:
            _remove_by_name(saved_values, node.name)
            _remove_by_name(saved_sym_nodes, node.name)
        # wait_tensor is a bit special: if we have a "dead activation" that is not used in the bw,
        # but this dead activation is actually a collective,
        # then the collective will generally by followed by a wait_tensor() call.
        # we need to peak one node further to see if this wait_tensor is dead as well.
        elif distributed_enabled and all(
            n.target is torch.ops._c10d_functional.wait_tensor.default
            and len(n.users) == 0
            for n in node.users
        ):
            _remove_by_name(saved_values, node.name)
            _remove_by_name(saved_sym_nodes, node.name)
        elif _is_backward_state(node):
            # BackwardState is saved directly
            _remove_by_name(saved_values, node.name)
            assert backward_state_inputs

    # Now that we have the finalized list of saved values, we need to ensure
    # we propagate all symbols which are referenced by backwards inputs.
    # These are not directly used in the graph but are required for downstream
    # sizevar assignment
    saved_symbols: OrderedSet[sympy.Symbol] = OrderedSet()
    saved_sym_nodes_binding = []
    saved_sym_nodes_derived = []

    # Some symbols may already be bound in the directly saved_sym_nodes,
    # keep track of them so we don't re-bind them
    for node in saved_sym_nodes:
        symbol = is_symbol_binding_fx_node(node)
        if symbol:
            saved_symbols.add(symbol)
            saved_sym_nodes_binding.append(node)
        else:
            saved_sym_nodes_derived.append(node)

    # Now go through all of the prospective backward inputs and track any
    # other symbols we need to bind
    symbol_bindings = find_symbol_binding_fx_nodes(joint_module.graph)
    for node in itertools.chain(saved_sym_nodes_derived, saved_values):
        if "val" not in node.meta:
            continue
        new_symbols = free_symbols(node.meta["val"]) - saved_symbols
        # NB: Deterministic order please!
        for s in sorted(new_symbols, key=lambda s: s.name):
            # NB: For well formed graphs, the symbol should always be present,
            # but we also have ways to produce ill-formed graphs, e.g., direct
            # make_fx usages, so don't choke in this case
            if s not in symbol_bindings:
                continue
            saved_sym_nodes_binding.append(symbol_bindings[s])
        saved_symbols |= new_symbols

    # Update saved_sym_nodes that are now reordered to have all bindings at
    # front. This can also be used later on to figure out the position of saved
    # sym nodes in the output of fwd graph.
    saved_sym_nodes.clear()
    saved_sym_nodes.extend(saved_sym_nodes_binding + saved_sym_nodes_derived)

    # Now, we re-generate the fwd/bwd graphs.
    # NB: This might increase compilation time, but I doubt it matters
    fwd_graph = _extract_graph_with_inputs_outputs(
        joint_module.graph,
        primal_inputs + fwd_seed_offset_inputs,
        fwd_outputs + saved_values + saved_sym_nodes,
        fwd_outputs_descs
        + [
            SavedForBackwardsAOTOutput(i)
            for i in range(len(saved_values) + len(saved_sym_nodes))
        ],
        "forward",
        ignore_must_be_in_fw_bw=True,
    )
    bwd_graph = _extract_graph_with_inputs_outputs(
        joint_module.graph,
        saved_values + saved_sym_nodes + bwd_seed_offset_inputs + backward_state_inputs,
        bwd_outputs,
        bwd_outputs_descs,
        "backward",
        ignore_must_be_in_fw_bw=True,
    )

    fwd_module = fx._lazy_graph_module._make_graph_module(joint_module, fwd_graph)
    bwd_module = fx._lazy_graph_module._make_graph_module(joint_module, bwd_graph)
    return fwd_module, bwd_module


# TODO: in theory we can infer num_weight_gradients from the graph metadata directly
def split_di_dw_graph(
    bw_gm_old: fx.GraphModule, *, num_weight_gradients
) -> tuple[fx.GraphModule, fx.GraphModule, int]:
    # we could consider doing this is a non-mutating way
    bw_gm = copy.deepcopy(bw_gm_old)
    placeholders = bw_gm.graph.find_nodes(op="placeholder")
    for p in placeholders:
        if p.name.startswith("tangent"):
            name_suffix = p.name[8:]
            rename_placeholder_node(bw_gm, p, f"not_tngnt{name_suffix}")

    remove_recompute_tags(bw_gm)
    num_input_gradients = reorder_output_grads(bw_gm, num_weight_gradients)
    bw_gm.recompile()

    args = list(bw_gm.graph.find_nodes(op="placeholder"))

    #    bw_inputs, bw_weights = default_partition(bw_gm, args, num_fwd_outputs=num_input_gradients)
    #    return bw_inputs, bw_weights, num_input_gradients

    (
        grad_inps,
        grad_weights,
        grad_inp_descs,
        grad_weight_descs,
    ) = _extract_fwd_bwd_outputs(bw_gm, num_fwd_outputs=num_input_gradients)
    bw_inputs_gm = _extract_graph_with_inputs_outputs(
        bw_gm.graph,
        args,
        grad_inps,
        grad_inp_descs,
        "forward",
        ignore_must_be_in_fw_bw=True,
    )
    bw_inputs_gm_node_names = OrderedSet(
        node.name for node in bw_inputs_gm.nodes if node.op != "output"
    )
    saved_values = []
    saved_sym_nodes = []

    for node in bw_gm.graph.nodes:
        if node.name not in bw_inputs_gm_node_names:
            # Not handling mutations for now,
            # we can try to re-use more of and/or consolidate with default partitioner
            continue
        if is_sym_node(node):
            saved_sym_nodes.append(node)
        elif (
            "tensor_meta" not in node.meta
            and node.op == "call_function"
            and not isinstance(node.meta.get("val"), torch._subclasses.FakeTensor)
        ):
            users = node.users
            assert all(user.target == operator.getitem for user in users)
            saved_values.extend(users)
        else:
            backward_usages = [
                n for n in node.users if n.name not in bw_inputs_gm_node_names
            ]
            if "tensor_meta" in node.meta and all(
                is_sym_node(n) for n in backward_usages
            ):
                saved_sym_nodes.extend(backward_usages)
            else:
                saved_values.append(node)
    saved_values = list(dict.fromkeys(saved_values).keys())
    saved_sym_nodes = list(dict.fromkeys(saved_sym_nodes).keys())
    bw_inputs, bw_weights = _extract_fwd_bwd_modules(
        bw_gm,
        saved_values,
        saved_sym_nodes=saved_sym_nodes,
        num_fwd_outputs=num_input_gradients,
    )
    return bw_inputs, bw_weights, num_input_gradients
