# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import nn
from torch._subclasses import FakeTensorMode

from autoparallel.export_module import aot_export_module, apply_node_renaming


@pytest.mark.parametrize("requires_grad_1", [False, True])
@pytest.mark.parametrize("requires_grad_2", [False, True])
def test_graph_export(requires_grad_1, requires_grad_2):
    class Model(nn.Module):
        def __init__(self, dim1):
            super().__init__()
            self.linear = nn.Linear(dim1, dim1)
            self.register_buffer("buf", torch.rand(1))

        def forward(self, x, y):
            return y + 2, self.linear(x) * self.buf, x + y + self.buf

    # needed because of https://github.com/pytorch/pytorch/issues/148977
    torch.__future__.set_swap_module_params_on_conversion(True)
    fake_mode = FakeTensorMode()
    with fake_mode:
        b = 32
        dim = 128
        inputs = (
            torch.rand(b, dim, requires_grad=requires_grad_1),
            torch.rand(b, 1, requires_grad=requires_grad_2),
        )
        model = Model(dim)
        gm, _, params_len, buffer_len, metadata = aot_export_module(
            model, inputs, trace_joint=True
        )
        apply_node_renaming(gm, params_len, buffer_len, metadata)

    input_nodes = [
        n for n in gm.graph.find_nodes(op="placeholder") if "input" in n.name
    ]
    tangent_nodes = [
        n for n in gm.graph.find_nodes(op="placeholder") if "tangents" in n.name
    ]
    grad_inputs = [
        n
        for n in gm.graph.find_nodes(op="output")[0].all_input_nodes
        if "grad_input" in n.name
    ]
    output_nodes = [
        n
        for n in gm.graph.find_nodes(op="output")[0].all_input_nodes
        if "output" in n.name
    ]

    buffer_nodes = [
        n for n in gm.graph.find_nodes(op="placeholder") if "buffer" in n.name
    ]

    o = model(*inputs)
    expected_tangents = [i for i, x in enumerate(o) if x.requires_grad]
    assert len(expected_tangents) == len(tangent_nodes)
    for i, tn in zip(expected_tangents, tangent_nodes):
        assert tn.name == f"tangents_{i}", f"{tn.name}, {i}"

    assert len(output_nodes) == len(o)
    assert len(buffer_nodes) == 1
    assert len(input_nodes) == 2
    assert len(grad_inputs) == int(requires_grad_1) + int(requires_grad_2)
