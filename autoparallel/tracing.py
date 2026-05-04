# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import copy
import operator
from contextlib import contextmanager

import torch
from torch._functorch._aot_autograd.descriptors import (
    BufferAOTInput,
    GradAOTOutput,
    ParamAOTInput,
    PlainAOTInput,
    PlainAOTOutput,
)
from torch._inductor.decomposition import select_decomp_table
from torch._subclasses import FakeTensorMode


def _get_decomp_table():
    decomp_table = copy.copy(select_decomp_table())
    # TODO: removing those as they cause missing DTensor propagation rules
    decomp_table.pop(torch.ops.aten.full_like.default)
    decomp_table.pop(torch.ops.aten.empty_like.default)
    decomp_table.pop(torch.ops.aten.threshold_backward.default)
    decomp_table.pop(torch.ops.aten.native_layer_norm.default)
    decomp_table.pop(torch.ops.aten.embedding_dense_backward.default)
    decomp_table.pop(torch.ops.aten.native_layer_norm_backward.default)
    decomp_table.pop(torch.ops.aten._softmax_backward_data.default)
    decomp_table.pop(torch.ops.aten._softmax.default)
    decomp_table.pop(torch.ops.aten.stack.default)

    # decompose addmm to allow for TP on mm
    decomp_table.pop(torch.ops.aten.addmm.default)

    def addmm_decomp(self, mat1, mat2, beta=1, alpha=1):
        return self + mat1 @ mat2

    decomp_table[torch.ops.aten.addmm.default] = addmm_decomp
    # decomp_table = None

    return decomp_table


def move_to_fake(model: torch.nn.Module, mode: FakeTensorMode, device: torch.device):
    """
    Move the model to the fake mode and move the weights to the fake device
    """

    def assert_is_meta_tensor(name, t):
        assert isinstance(t, torch.Tensor) and t.device == torch.device(
            "meta"
        ), f"tensor {name} must be on meta device, not {t.device}"

    # Use remove_duplicate=False so aliased params/buffers (e.g. rope.cache
    # and freqs_cis pointing to the same tensor) get the same fake tensor,
    # preserving aliasing through tracing.
    fake_memo: dict[int, torch.Tensor] = {}

    def _move_to_fake(module, k, device, parameter=True):
        submod = module
        while len(k.split(".")) > 1:
            submod_name, k = k.split(".", 1)
            submod = getattr(submod, submod_name)

        orig = getattr(submod, k)
        if id(orig) in fake_memo:
            fake_tensor = fake_memo[id(orig)]
        else:
            fake_tensor = mode.from_tensor(orig).to(device)
            if parameter:
                fake_tensor = torch.nn.Parameter(
                    fake_tensor, requires_grad=fake_tensor.requires_grad
                )
            fake_memo[id(orig)] = fake_tensor
            # Also map the fake tensor's id so aliased submodules
            # (where setattr already replaced the original) are recognized.
            fake_memo[id(fake_tensor)] = fake_tensor
        setattr(submod, k, fake_tensor)

    with mode:
        for k, p in model.named_parameters(remove_duplicate=False):
            if id(p) not in fake_memo:
                assert_is_meta_tensor(k, p)
            _move_to_fake(model, k, device, parameter=True)
        for k, b in model.named_buffers(remove_duplicate=False):
            if id(b) not in fake_memo:
                assert_is_meta_tensor(k, b)
            _move_to_fake(model, k, device, parameter=False)

    return model


@contextmanager
def enable_local_map_wrapping():
    from torch._dynamo.variables.higher_order_ops import (
        LocalMapWrappedHigherOrderVariable as vt_cls,
    )
    from torch._higher_order_ops import local_map as local_map_module

    with vt_cls.enable(), local_map_module.defer_inlining():
        yield


def _add_unused_params_and_buffers(model, graph_module):
    """Register parameters/buffers from model that are missing from graph_module.

    Dynamo only captures parameters/buffers actually used in forward(). This
    adds unused ones as get_attr nodes so aot_export_joint_with_descriptors
    lifts them into the joint graph and they appear in params_spec/buffers_spec.
    """
    from torch.fx.graph_module import _assign_attr

    existing_params = set(dict(graph_module.named_parameters()))
    existing_buffers = set(dict(graph_module.named_buffers()))

    graph = graph_module.graph
    # Insert after all existing placeholder/get_attr nodes, before computation.
    insert_before = None
    for node in graph.nodes:
        if node.op not in ("placeholder", "get_attr"):
            insert_before = node
            break

    added = False
    for fqn, param in model.named_parameters():
        if fqn not in existing_params:
            _assign_attr(param, graph_module, fqn)
            with graph.inserting_before(insert_before):
                n = graph.create_node("get_attr", target=fqn)
                n.meta["val"] = param
            added = True

    for fqn, buf in model.named_buffers():
        if fqn not in existing_buffers:
            _assign_attr(buf, graph_module, fqn)
            with graph.inserting_before(insert_before):
                n = graph.create_node("get_attr", target=fqn)
                n.meta["val"] = buf
            added = True

    if added:
        graph_module.recompile()


def infer_grad_param_mapping(gm):
    """Analyze autograd.grad nodes to map gradient outputs to parameter placeholders.

    In a dynamo-captured graph with trace_autograd_ops=True, backward() is lowered
    to torch.autograd.grad() calls. This function walks those nodes to determine
    which outputs of the graph are gradients of which parameters.

    Returns: dict mapping output_position (int) -> placeholder_node
    """
    # Find the output node and its args
    output_node = None
    for n in gm.graph.nodes:
        if n.op == "output":
            output_node = n
            break
    assert output_node is not None

    # output_node.args[0] is the tuple of output values
    output_args = output_node.args[0]

    # Build a map from node -> output position
    node_to_output_pos = {}
    for i, arg in enumerate(output_args):
        if isinstance(arg, torch.fx.Node):
            node_to_output_pos[arg] = i

    # Find all autograd.grad call nodes
    grad_nodes = []
    for n in gm.graph.nodes:
        if n.op == "call_function" and n.target is torch.autograd.grad:
            grad_nodes.append(n)

    # For each grad node, trace its getitem users to output positions
    # autograd.grad(outputs, inputs, ...) returns a tuple of grads
    # args[0] = outputs (loss), args[1] = inputs (params to diff w.r.t.)
    mapping = {}
    for grad_node in grad_nodes:
        # args[1] is the sequence of tensors to differentiate with respect to
        wrt_tensors = grad_node.args[1]
        if isinstance(wrt_tensors, torch.fx.Node):
            # Single tensor, not a list — wrap for uniform handling
            wrt_tensors = [wrt_tensors]

        for user in grad_node.users:
            if user.op == "call_function" and user.target is operator.getitem:
                idx = user.args[1]
                if idx < len(wrt_tensors):
                    param_node = wrt_tensors[idx]
                    # Trace this getitem through to the output
                    # The grad value might go through empty_like → copy_ pattern
                    # or directly to output
                    out_pos = _trace_to_output(user, node_to_output_pos)
                    if out_pos is not None and param_node.op == "placeholder":
                        mapping[out_pos] = param_node

    return mapping


def _trace_to_output(node, node_to_output_pos):
    """Trace a node through copy_ chains to find its output position."""
    # Direct case: node itself is in the output
    if node in node_to_output_pos:
        return node_to_output_pos[node]

    # Follow through empty_like → copy_ pattern:
    # grad_getitem → used by copy_(empty_like_result, grad_getitem) → empty_like_result in output
    for user in node.users:
        if user.op == "call_function" and user.target is torch.ops.aten.copy_.default:
            # copy_(dst, src) — if our node is the src (args[1]), the dst might be in output
            if user.args[1] is node:
                dst = user.args[0]
                if isinstance(dst, torch.fx.Node) and dst in node_to_output_pos:
                    return node_to_output_pos[dst]

    # Also check if any user is directly in output
    for user in node.users:
        if user in node_to_output_pos:
            return node_to_output_pos[user]

    return None


def annotate_user_backward_descriptors(gm, model):
    """Set meta["desc"] on all placeholder and output nodes for a user-backward graph.

    This produces the same descriptor structure that aot_export_joint_with_descriptors
    would produce, enabling reuse of ShardingOptimizer including
    forward_backward_consistency_constraints.
    """
    # Build FQN maps from the model
    param_fqns = set(dict(model.named_parameters()))
    buffer_fqns = set(dict(model.named_buffers()))

    # Get the grad→param mapping
    grad_mapping = infer_grad_param_mapping(gm)

    # Annotate placeholders
    plain_input_idx = 0
    for n in gm.graph.nodes:
        if n.op != "placeholder":
            continue

        target = n.target
        if target in param_fqns:
            n.meta["desc"] = ParamAOTInput(target=target)
        elif target in buffer_fqns:
            n.meta["desc"] = BufferAOTInput(target=target)
        else:
            n.meta["desc"] = PlainAOTInput(idx=plain_input_idx)
            plain_input_idx += 1

    # Annotate outputs
    output_node = None
    for n in gm.graph.nodes:
        if n.op == "output":
            output_node = n
            break
    assert output_node is not None

    output_args = output_node.args[0]
    plain_output_idx = 0
    for i, arg in enumerate(output_args):
        if i in grad_mapping:
            param_node = grad_mapping[i]
            # Get the descriptor we just set on the param placeholder
            param_desc = param_node.meta["desc"]
            desc = GradAOTOutput(grad_of=param_desc)
        else:
            desc = PlainAOTOutput(idx=plain_output_idx)
            plain_output_idx += 1

        # Set desc on the output arg node if it's a node
        if isinstance(arg, torch.fx.Node):
            arg.meta["desc"] = desc
