# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import copy
from contextlib import contextmanager

import torch
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
