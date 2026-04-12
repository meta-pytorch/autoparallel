# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch._functorch._aot_autograd.fx_utils import get_param_nodes
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel.api import AutoParallel


class FFN(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.linear1 = nn.Linear(dim1, dim2, bias=False)
        self.linear2 = nn.Linear(dim2, dim1, bias=False)

    def forward(self, x):
        return self.linear2(self.linear1(x))


class TransformerBlock(nn.Module):
    def __init__(self, nheads, dim1, dim2):
        super().__init__()
        self.nheads = nheads
        self.wq = nn.Linear(dim1, dim1, bias=False)
        self.wk = nn.Linear(dim1, dim1, bias=False)
        self.wv = nn.Linear(dim1, dim1, bias=False)
        self.wo = nn.Linear(dim1, dim1, bias=False)
        self.w1 = nn.Linear(dim1, dim2, bias=False)
        self.w2 = nn.Linear(dim2, dim1, bias=False)

    def forward(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
        k = k.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
        v = v.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)

        o = nn.functional.scaled_dot_product_attention(q, k, v)
        o = o.permute(0, 2, 1, 3).flatten(-2)
        o = self.wo(o)

        o0 = o + x
        o = self.w1(o0)
        o = torch.nn.functional.relu(o)
        o = self.w2(o)
        return o0 + o


@patch("torch.cuda.device_count", lambda: 8)
@patch("torch.cuda.get_device_name", lambda device: "H100")
def test_dynamic_produces_same_placement_as_static_1d(device_mesh_1d):
    """ILP solution should be identical with dynamic=True vs dynamic=False."""
    dim1, dim2 = 1024, 4096
    bs = 2048 * device_mesh_1d.shape[0]

    def input_fn():
        return torch.randn(bs, dim1, device="cuda")

    with torch.device("meta"):
        model = FFN(dim1, dim2)

    placement = (Shard(0),)
    with AutoParallel(model, input_fn, device_mesh_1d) as autop_static:
        autop_static.add_input_constraints([placement])
        autop_static.add_output_constraints([placement])
        static_placement = autop_static.optimize_placement(verbose=False)

    with torch.device("meta"):
        model = FFN(dim1, dim2)

    with AutoParallel(model, input_fn, device_mesh_1d, dynamic=True) as autop_dynamic:
        autop_dynamic.add_input_constraints([placement])
        autop_dynamic.add_output_constraints([placement])
        dynamic_placement = autop_dynamic.optimize_placement(verbose=False)

    # Compare placements for all nodes
    for node_s, node_d in zip(
        autop_static.gm.graph.nodes, autop_dynamic.gm.graph.nodes
    ):
        if node_s not in static_placement:
            continue
        if node_d not in dynamic_placement:
            continue
        sp = static_placement[node_s]
        dp = dynamic_placement[node_d]
        assert sp.output_specs.placements == dp.output_specs.placements, (
            f"Placement mismatch for {node_s.name}: "
            f"static={sp.output_specs.placements} vs dynamic={dp.output_specs.placements}"
        )


@patch("torch.cuda.device_count", lambda: 8)
@patch("torch.cuda.get_device_name", lambda device: "H100")
def test_dynamic_produces_same_placement_as_static_2d(device_mesh_2d):
    """ILP solution for transformer block should be identical with dynamic=True."""
    dim1 = 6144
    dim2 = dim1 * 4
    nheads = 48
    bs = 8 * device_mesh_2d.shape[0]

    def input_fn():
        return torch.randn(bs, 256, dim1, device="cuda", requires_grad=True)

    with torch.device("meta"):
        model = TransformerBlock(nheads, dim1, dim2)

    placement = (Shard(0), Replicate())
    with AutoParallel(model, input_fn, device_mesh_2d) as autop_static:
        autop_static.add_input_constraints([placement])
        autop_static.add_output_constraints([placement])
        static_placement = autop_static.optimize_placement(verbose=False)

    with torch.device("meta"):
        model = TransformerBlock(nheads, dim1, dim2)

    with AutoParallel(model, input_fn, device_mesh_2d, dynamic=True) as autop_dynamic:
        autop_dynamic.add_input_constraints([placement])
        autop_dynamic.add_output_constraints([placement])
        dynamic_placement = autop_dynamic.optimize_placement(verbose=False)

    param_nodes_s = get_param_nodes(autop_static.gm.graph)
    param_nodes_d = get_param_nodes(autop_dynamic.gm.graph)
    for node_s, node_d in zip(param_nodes_s, param_nodes_d):
        sp = static_placement[node_s].output_specs.placements
        dp = dynamic_placement[node_d].output_specs.placements
        assert (
            sp == dp
        ), f"Param placement mismatch for {node_s.name}: static={sp} vs dynamic={dp}"


@patch("torch.cuda.device_count", lambda: 8)
@patch("torch.cuda.get_device_name", lambda device: "H100")
def test_dynamic_apply_placement_ffn(device_mesh_1d):
    """apply_placement should succeed with dynamic=True for a simple FFN."""
    dim1, dim2 = 1024, 4096
    bs = 2048 * device_mesh_1d.shape[0]

    def input_fn():
        return torch.randn(bs, dim1, device="cuda")

    with torch.device("meta"):
        model = FFN(dim1, dim2)

    placement = (Shard(0),)
    with AutoParallel(model, input_fn, device_mesh_1d, dynamic=True) as autop:
        autop.add_input_constraints([placement])
        autop.add_output_constraints([placement])
        autop.optimize_placement(verbose=False)
        parallel_model = autop.apply_placement()

    assert parallel_model is not None


@patch("torch.cuda.device_count", lambda: 8)
@patch("torch.cuda.get_device_name", lambda device: "H100")
def test_dynamic_apply_placement_transformer(device_mesh_2d):
    """apply_placement should succeed with dynamic=True for transformer block."""
    dim1 = 6144
    dim2 = dim1 * 4
    nheads = 48
    bs = 8 * device_mesh_2d.shape[0]

    def input_fn():
        return torch.randn(bs, 256, dim1, device="cuda", requires_grad=True)

    with torch.device("meta"):
        model = TransformerBlock(nheads, dim1, dim2)

    placement = (Shard(0), Replicate())
    with AutoParallel(
        model, input_fn, device_mesh_2d, dynamic=True, compile=False
    ) as autop:
        autop.add_input_constraints([placement])
        autop.add_output_constraints([placement])
        autop.optimize_placement(verbose=False)
        parallel_model = autop.apply_placement()

    assert parallel_model is not None


@patch("torch.cuda.device_count", lambda: 8)
@patch("torch.cuda.get_device_name", lambda device: "H100")
def test_dynamic_joint_graph_has_symbolic_shapes(device_mesh_2d):
    """The joint graph from dynamic=True should have symbolic shapes on inputs."""
    dim1 = 6144
    dim2 = dim1 * 4
    nheads = 48
    bs = 8 * device_mesh_2d.shape[0]

    def input_fn():
        return torch.randn(bs, 256, dim1, device="cuda", requires_grad=True)

    with torch.device("meta"):
        model = TransformerBlock(nheads, dim1, dim2)

    placement = (Shard(0), Replicate())
    with AutoParallel(
        model, input_fn, device_mesh_2d, dynamic=True, compile=False
    ) as autop:
        autop.add_input_constraints([placement])
        autop.add_output_constraints([placement])

        # Check the joint graph has symbolic shapes on input placeholders
        has_symint = False
        for node in autop.gm.graph.nodes:
            if node.op == "placeholder" and isinstance(
                node.meta.get("val"), torch.Tensor
            ):
                for s in node.meta["val"].shape:
                    if isinstance(s, torch.SymInt):
                        has_symint = True
                        break
        assert has_symint, "joint graph should have symbolic shapes on inputs"


@patch("torch.cuda.device_count", lambda: 8)
@patch("torch.cuda.get_device_name", lambda device: "H100")
def test_dynamic_check_forward_args_accepts_different_batch(device_mesh_1d):
    """_check_forward_args should accept different batch sizes with dynamic shapes."""
    from autoparallel.input_validation import (
        _check_forward_args,
        _compute_expected_inputs,
    )

    dim1, dim2 = 1024, 4096
    bs = 2048 * device_mesh_1d.shape[0]

    def input_fn():
        return torch.randn(bs, dim1, device="cuda")

    with torch.device("meta"):
        model = FFN(dim1, dim2)

    placement = (Shard(0),)
    with AutoParallel(model, input_fn, device_mesh_1d, dynamic=True) as autop:
        autop.add_input_constraints([placement])
        autop.add_output_constraints([placement])
        autop.optimize_placement(verbose=False)

        expected = _compute_expected_inputs(
            autop._traced_inputs, autop.input_constraints, device_mesh_1d
        )

    # The expected shapes should have SymInt for the batch dim
    assert isinstance(expected[0].shape[0], torch.SymInt), "batch dim should be SymInt"

    # Different batch sizes should be accepted
    local_bs = bs // device_mesh_1d.size()
    for test_bs in [local_bs, local_bs * 2, 1]:
        _check_forward_args(
            [torch.randn(test_bs, dim1)],
            expected,
        )

    # Wrong non-batch dim should be rejected
    with pytest.raises(ValueError, match="has shape"):
        _check_forward_args(
            [torch.randn(local_bs, dim1 + 1)],
            expected,
        )
