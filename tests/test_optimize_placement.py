# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import nn
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel.api import AutoParallel


@pytest.fixture(scope="module", autouse=True)
def init_pg():
    world_size = 256
    fake_store = FakeStore()
    torch.distributed.init_process_group(
        "fake", store=fake_store, rank=0, world_size=world_size
    )


@pytest.fixture(scope="module")
def device_mesh_1d():
    world_size = torch.distributed.get_world_size()
    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda", (world_size,), mesh_dim_names=("dp",)
    )
    return mesh


@pytest.fixture(scope="module")
def device_mesh_2d():
    world_size = torch.distributed.get_world_size()
    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda",
        (world_size // 8, 8),
        mesh_dim_names=(
            "dp",
            "tp",
        ),
    )
    return mesh


class FFN(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        bias = False
        self.linear1 = nn.Linear(dim1, dim2, bias=bias)
        self.linear2 = nn.Linear(dim2, dim1, bias=bias)

    def forward(self, x, y):
        return y + 2, self.linear2(self.linear1(x)), y + 2


class TransformerBlock(nn.Module):
    def __init__(self, nheads, dim1, dim2):
        super().__init__()
        self.nheads = nheads
        bias = False
        self.wq = nn.Linear(dim1, dim1, bias=bias)
        self.wk = nn.Linear(dim1, dim1, bias=bias)
        self.wv = nn.Linear(dim1, dim1, bias=bias)
        self.wo = nn.Linear(dim1, dim1, bias=bias)
        self.w1 = nn.Linear(dim1, dim2, bias=bias)
        self.w2 = nn.Linear(dim2, dim1, bias=bias)

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

        o = o0 + o
        return o


def _make_model_and_input_fn(mesh, model_type="ffn_with_multiple_input_output"):
    if model_type == "ffn_with_multiple_input_output":
        bs = 2048 * mesh.shape[0]
        dim1 = 1024
        dim2 = 4096

        def model_fn():
            return FFN(dim1, dim2).cuda()

        def input_fn():
            return torch.randn(bs, dim1).cuda(), torch.randn(bs, 1).cuda()

    elif model_type == "transformer_block":
        bs = 8 * mesh.shape[0]
        dim1 = 6144
        dim2 = dim1 * 4
        nheads = 48

        def model_fn():
            return TransformerBlock(nheads, dim1, dim2).cuda()

        def input_fn():
            return torch.randn(bs, 256, dim1, device="cuda", requires_grad=True)

    return model_fn, input_fn


@pytest.mark.parametrize(
    "model_type", ["ffn_with_multiple_input_output", "transformer_block"]
)
@pytest.mark.parametrize("high_mem", [None, 1.0])
def test_optimization_finds_fsdp_and_ddp_1d(device_mesh_1d, high_mem, model_type):
    low_mem = 0
    model_fn, input_fn = _make_model_and_input_fn(device_mesh_1d, model_type)

    autop = AutoParallel(model_fn, input_fn, device_mesh_1d)
    autop.add_parameter_memory_constraint(low=low_mem, high=high_mem)

    sharding_placement = autop.optimize_placement()

    # check parameters are sharded as expected, i.e., either replicated or sharded
    param_nodes = [
        n for n in autop.gm.graph.find_nodes(op="placeholder") if "param" in n.target
    ]
    placement = {None: (Shard(0),), 1.0: (Replicate(),)}[high_mem]
    for node in param_nodes:
        assert sharding_placement[node].output_specs.placements == placement

    mm_nodes = autop.gm.graph.find_nodes(
        op="call_function", target=torch.ops.aten.mm.default
    )
    len_mm_nodes = {"ffn_with_multiple_input_output": 5, "transformer_block": 18}[
        model_type
    ]
    len_fwd_mm_nodes = {"ffn_with_multiple_input_output": 2, "transformer_block": 6}[
        model_type
    ]
    assert len(mm_nodes) == len_mm_nodes
    fwd_mm_nodes = mm_nodes[0:len_fwd_mm_nodes]
    bwd_mm_grad_weight_nodes = mm_nodes[len_fwd_mm_nodes::2]
    bwd_mm_grad_input_nodes = mm_nodes[(len_fwd_mm_nodes + 1) :: 2]

    # and check that matmuls have full replication on weights during fwd,
    # which maps to DDP / FSDP

    # fwd
    for node in fwd_mm_nodes:
        p = sharding_placement[node]
        # input and output are sharded on batch
        assert p.input_specs[0].placements == (Shard(0),)
        assert p.output_specs.placements == (Shard(0),)
        # weight is replicated, mimicing DDP
        assert p.input_specs[1].placements == (Replicate(),)

    # bwd grad weight
    for node in bwd_mm_grad_weight_nodes:
        p = sharding_placement[node]
        assert p.input_specs[0].placements == (Shard(1),)
        assert p.output_specs.placements == (Partial("sum"),)
        assert p.input_specs[1].placements == (Shard(0),)

    # bwd grad inputs
    for node in bwd_mm_grad_input_nodes:
        p = sharding_placement[node]
        assert p.input_specs[0].placements == (Shard(0),)
        assert p.output_specs.placements == (Shard(0),)
        assert p.input_specs[1].placements == (Replicate(),)


_expected_param_placements_ffn = [(Shard(0), Shard(0)), (Shard(0), Shard(1))]


# some characteristic 2d placements for matmul for input1, input2, output
_mm1 = [(Shard(0), Replicate()), (Replicate(), Shard(1)), (Shard(0), Shard(1))]
_mm2 = [(Shard(0), Shard(1)), (Replicate(), Shard(0)), (Shard(0), Partial("sum"))]
_mm3 = [(Shard(1), Replicate()), (Shard(0), Shard(1)), (Partial("sum"), Shard(1))]
_mm4 = [(Shard(1), Shard(0)), (Shard(0), Replicate()), (Partial("sum"), Shard(0))]


_expected_node_placements_ffn = [
    _mm1,
    _mm2,
    _mm3,
    _mm1,
    _mm4,
]


_expected_param_placements_transformer_block = [
    (Shard(0), Shard(0)),
    (Shard(0), Shard(0)),
    (Shard(0), Shard(0)),
    (Shard(0), Shard(1)),
    (Shard(0), Shard(0)),
    (Shard(0), Shard(1)),
]

_expected_node_placements_transformer_block = [
    _mm1,
    _mm1,
    _mm1,
    _mm2,
    _mm1,
    _mm2,
    _mm3,
    _mm1,
    _mm4,
    _mm2,
    _mm3,
    _mm1,
    _mm4,
    _mm2,
    _mm4,
    _mm2,
    _mm4,
    _mm2,
]


@pytest.mark.parametrize(
    "model_type,expected_param_placements,expected_node_placements",
    [
        (
            "ffn_with_multiple_input_output",
            _expected_param_placements_ffn,
            _expected_node_placements_ffn,
        ),
        (
            "transformer_block",
            _expected_param_placements_transformer_block,
            _expected_node_placements_transformer_block,
        ),
    ],
)
def test_optimization_finds_fsdp_tp_2d(
    device_mesh_2d, model_type, expected_param_placements, expected_node_placements
):
    low_mem = 0
    high_mem = None
    model_fn, input_fn = _make_model_and_input_fn(device_mesh_2d, model_type)

    autop = AutoParallel(model_fn, input_fn, device_mesh_2d)
    autop.add_parameter_memory_constraint(low=low_mem, high=high_mem)

    sharding_placement = autop.optimize_placement()

    # check parameters are sharded as expected
    param_nodes = [
        n for n in autop.gm.graph.find_nodes(op="placeholder") if "param" in n.target
    ]
    for node, expected_placement in zip(param_nodes, expected_param_placements):
        assert sharding_placement[node].output_specs.placements == expected_placement

    # chekc that matmul nodes are sharded following FSDP + TP
    mm_nodes = autop.gm.graph.find_nodes(
        op="call_function", target=torch.ops.aten.mm.default
    )
    for node, expected_placements in zip(mm_nodes, expected_node_placements):
        p = sharding_placement[node]
        assert p.input_specs[0].placements == expected_placements[0]
        assert p.input_specs[1].placements == expected_placements[1]
        assert p.output_specs.placements == expected_placements[2]

    # chekc that sdpa nodes (if present) are sharded following FSDP + TP
    sdpa_nodes = autop.gm.graph.find_nodes(
        op="call_function",
        target=torch.ops.aten._scaled_dot_product_efficient_attention.default,
    )
    for node in sdpa_nodes:
        p = sharding_placement[node]
        placement = (Shard(0), Shard(1))
        assert p.input_specs[0].placements == placement
        assert p.input_specs[1].placements == placement
        assert p.input_specs[2].placements == placement

        assert p.output_specs[0].placements == placement
