# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Copied from
# https://github.com/pytorch/pytorch/blob/1cc5f6b623907579a0e3e172b061391b171b9fa5/torch/_functorch/aot_autograd.py#L1211
# with modifications to support passing the gradients

import torch
from torch import nn
from torch._functorch.aot_autograd import (  # create_graph_signature,
    Optional,
    ViewAndMutationMeta,
    _aot_export_function,
    create_functional_call,
    make_fx,
    nullcontext,
    pytree,
    wraps,
)


def aot_export_module(
    mod: nn.Module,
    args,
    *,
    decompositions: Optional[dict] = None,
    # If true, we'll return a joint forward-backward graph,
    # As well as metadata on the loss + gradients in the backward.
    trace_joint: bool,
    pre_dispatch: bool = False,
    # If None, will be infered from inputs and mod.graph.nodes if mod is a graph module,
    # but the inferred result might be wrong.
    dynamic_shapes: Optional[bool] = None,
    kwargs=None,
) -> tuple[torch.fx.GraphModule, pytree.TreeSpec, int, int, ViewAndMutationMeta]:
    """
    This function takes in a module, and returns:
    (1) an FX graph that can be exported
    (2) some metadata about the graph

    If `trace_joint=True` we will return a joint graph of the forward + backward.

    The traced FX graph will have the following properties compared to the original module:
    (1) Inputs and outputs to the module will be pytree-flattened
    (2) Parameters and buffers on the module will be lifted into graph inputs,
        graph_inputs = (*parameters, *buffers, *user_inputs)
    (3) The graph will be fully functionalized
    (4) Any input mutations will be converted into additional outputs in the graph,
        meaning whoever calls this graph is responsible for applying the mutations
        back to the original inputs.
    (5) If is_joint is provided the graph will return parameter gradients in addition to user outputs.
        The graph output will look like:
        graph_outputs = (*updated_inputs, *user_outputs, *param_gradients)

    There are also several restrictions on what modules can use this API. In particular:
    (1) If trace_joint is specified, we expect the loss function to be **fused**
        into the module forward. One of the outputs to the forward must be a scalar loss,
        which is specified with `output_loss_index`.
        All other outputs to the forward are presumed to not require gradients.
    (2) This API cannot capture optimizers (although in theory we could build an API for this).
    (3) Metadata mutations on params/buffers/inputs are banned.
    (4) Data mutations on anything that requires gradients are banned (parameters)
    (5) If an input is mutated, it is not allowed to alias any other inputs.
    (6) Parameters must not be duplicated.
    """
    if pre_dispatch and trace_joint:
        raise RuntimeError("pre_dispatch is not supported when trace_joint is True.")

    class MyMod(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            out = self.module(*args, **kwargs)
            flat_out, self.spec = pytree.tree_flatten(out)
            return flat_out

    mod = MyMod(mod)

    named_parameters = dict(mod.named_parameters(remove_duplicate=False))
    named_buffers = dict(mod.named_buffers(remove_duplicate=False))

    params_and_buffers = {
        **dict(named_parameters),
        **dict(named_buffers),
    }
    params_and_buffers_flat_l, params_spec = pytree.tree_flatten(params_and_buffers)
    params_and_buffers_flat = tuple(params_and_buffers_flat_l)
    params_len = len(params_and_buffers_flat)
    buffer_len = sum([not isinstance(x, nn.Parameter) for x in params_and_buffers_flat])

    kwargs = kwargs or {}

    functional_call = create_functional_call(
        mod, params_spec, params_len, store_orig_mod=True
    )

    ctx = nullcontext
    fn_to_trace = functional_call

    full_args0: list[torch.Tensor] = []
    # First, the params
    # NB: It is REQUIRED that parameters come first, Inductor infers "fixed"
    # parameters by looking at the difference in parameter count outside
    # and inside AOTAutograd, and assumes the prefix of arguments are fixed
    # arguments
    full_args0.extend(params_and_buffers_flat)
    # Next, the input args
    full_args0.extend(args)

    with ctx():
        fx_g, metadata, in_spec, out_spec = _aot_export_function(
            fn_to_trace,
            full_args0,
            decompositions=decompositions,
            num_params_buffers=params_len,
            no_tangents=False,
            pre_dispatch=pre_dispatch,
            dynamic_shapes=dynamic_shapes,
            kwargs=kwargs,
        )

    if trace_joint:

        @wraps(functional_call)
        def flattened_joint(*args):
            fake_tangents = args[-len(metadata.traced_tangents) :]
            args = args[: -len(metadata.traced_tangents)]
            fw_outs, gradients = fx_g(args, fake_tangents)
            assert len(gradients) == len(args)
            output_gradients = []
            for a, grad in zip(args, gradients):
                if isinstance(a, torch.Tensor) and a.requires_grad:
                    if grad is None:
                        grad = torch.zeros_like(a)
                    output_gradients.append(grad)
                else:
                    assert grad is None
            return *fw_outs, *output_gradients

        full_args: list[torch.Tensor] = []
        full_args.extend(params_and_buffers_flat)
        full_args.extend(pytree.tree_flatten(args)[0])
        full_args.extend(metadata.traced_tangents)
        fx_g = make_fx(flattened_joint, record_module_stack=True)(*full_args)

    # user_args_flat = pytree.arg_tree_leaves(*args, **kwargs)

    # TODO: use create_graph_signature
    # return fx_g, create_graph_signature(
    #     fx_g,
    #     metadata,
    #     in_spec,
    #     out_spec,
    #     user_args_flat=user_args_flat,
    #     params_and_buffers_flat=params_and_buffers_flat,
    #     param_names=list(named_parameters.keys()),
    #     buffer_names=list(named_buffers.keys()),
    #     trace_joint=trace_joint,
    #     num_user_fw_outs=num_fw_outs,
    #     loss_index=0,
    # )

    return fx_g, mod.spec, params_len, buffer_len, metadata


def apply_node_renaming(fx_g, params_len, buffer_len, metadata):
    # originally implemented to make tangent args explicit for the partitioner
    # but then I extended it to rename input, parameter, output, grad_param and grad_input
    # does as well for convenience

    # TODO: this is confusing as params_len contains buffers as well
    # we should improve this
    params_len -= buffer_len

    def rename_nodes(fx_g, nodes, new_name, idxs=None):
        if idxs is None:
            idxs = list(range(len(nodes)))

        assert len(idxs) == len(nodes), f"{len(idxs)}, {len(nodes)}"
        for i, old_node in zip(idxs, nodes):
            with fx_g.graph.inserting_before(old_node):
                # new_node = fx_g.graph.placeholder(f"{new_name}_{i}")
                if old_node.op == "placeholder":
                    new_node = fx_g.graph.placeholder(f"{new_name}_{i}")
                else:
                    new_node = fx_g.graph.create_node(
                        old_node.op,
                        old_node.target,
                        old_node.args,
                        old_node.kwargs,
                        f"{new_name}_{i}",
                        old_node.type,
                    )
                new_node.meta.update(old_node.meta)
                old_node.replace_all_uses_with(new_node)
                fx_g.graph.erase_node(old_node)
        fx_g.recompile()

    # TODO: align number of grad names with inputs everywhere?
    all_output_nodes = fx_g.graph.find_nodes(op="output")[0].all_input_nodes
    output_nodes = all_output_nodes[: metadata.num_outputs]
    rename_nodes(fx_g, output_nodes, "output")
    param_grad = all_output_nodes[
        metadata.num_outputs : metadata.num_outputs + params_len
    ]
    rename_nodes(fx_g, param_grad, "grad_param")
    grad_inputs = all_output_nodes[metadata.num_outputs + params_len :]
    inputs_that_require_grad = [
        i for i, n in enumerate(metadata.input_info[params_len:]) if n.requires_grad
    ]
    rename_nodes(fx_g, grad_inputs, "grad_input", inputs_that_require_grad)

    tangent_nodes = fx_g.graph.find_nodes(op="placeholder")[
        -len(metadata.traced_tangents) :
    ]
    outputs_that_require_grad = [
        i for i, n in enumerate(metadata.output_info) if n.requires_grad
    ]
    rename_nodes(fx_g, tangent_nodes, "tangents", outputs_that_require_grad)
    input_nodes = fx_g.graph.find_nodes(op="placeholder")[
        params_len + buffer_len : -len(metadata.traced_tangents)
    ]
    rename_nodes(fx_g, input_nodes, "input")
    param_nodes = fx_g.graph.find_nodes(op="placeholder")[:params_len]
    rename_nodes(fx_g, param_nodes, "param")

    buffer_nodes = fx_g.graph.find_nodes(op="placeholder")[
        params_len : params_len + buffer_len
    ]
    rename_nodes(fx_g, buffer_nodes, "buffer")
