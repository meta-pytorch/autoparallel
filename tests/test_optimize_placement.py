# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter

import pytest
import torch
import torch.nn.functional as F
from conftest import apply_cuda_patches
from torch import nn
from torch._functorch._aot_autograd.fx_utils import get_param_nodes
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)

from autoparallel._flex_local_map import _body
from autoparallel.api import AutoParallel, auto_parallel
from autoparallel.collectives import (
    flex_local_map,
    get_flex_local_map_alternatives,
    local_map,
)


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


def _make_model_and_input_fn(
    mesh, model_type="ffn_with_multiple_input_output", device="cuda"
):
    if model_type == "ffn_with_multiple_input_output":
        bs = 2048 * mesh.shape[0]
        dim1 = 1024
        dim2 = 4096

        def model_fn():
            return FFN(dim1, dim2)

        def input_fn():
            return torch.randn(bs, dim1).to(device), torch.randn(bs, 1).to(device)

    elif model_type == "transformer_block":
        bs = 8 * mesh.shape[0]
        dim1 = 6144
        dim2 = dim1 * 4
        nheads = 48

        def model_fn():
            return TransformerBlock(nheads, dim1, dim2)

        def input_fn():
            return torch.randn(bs, 256, dim1, device=device, requires_grad=True)

    return model_fn, input_fn


@apply_cuda_patches
@pytest.mark.parametrize(
    "model_type", ["ffn_with_multiple_input_output", "transformer_block"]
)
@pytest.mark.parametrize("high_mem", [None, 1.0])
def test_optimization_finds_fsdp_and_ddp_1d(device_mesh_1d, high_mem, model_type):
    low_mem = 0
    device = "cuda"
    model_fn, input_fn = _make_model_and_input_fn(device_mesh_1d, model_type, device)
    with torch.device("meta"):
        model = model_fn()

    with AutoParallel(model, input_fn, device_mesh_1d) as autop:
        placement = (Shard(0),)
        n_inputs = 2 if model_type == "ffn_with_multiple_input_output" else 1
        n_outputs = 3 if model_type == "ffn_with_multiple_input_output" else 1
        autop.add_input_constraints([placement] * n_inputs)
        autop.add_output_constraints([placement] * n_outputs)
        autop.add_parameter_memory_constraint(low=low_mem, high=high_mem)

        sharding_placement = autop.optimize_placement()

    # check parameters are sharded as expected, i.e., either replicated or sharded
    param_nodes = get_param_nodes(autop.gm.graph)
    placement = {None: (Shard(0),), 1.0: (Replicate(),)}[high_mem]
    for node in param_nodes:
        assert sharding_placement[node].output_specs.placements == placement

    mm_nodes = autop.gm.graph.find_nodes(
        op="call_function", target=torch.ops.aten.mm.default
    )
    einsum_nodes = autop.gm.graph.find_nodes(
        op="call_function", target=torch.ops.aten.einsum.default
    )
    linear_nodes = mm_nodes + einsum_nodes
    is_einsum = len(einsum_nodes) > 0

    if is_einsum:
        len_linear_nodes = {
            "ffn_with_multiple_input_output": 5,
            "transformer_block": 18,
        }[model_type]
        len_fwd_linear_nodes = {
            "ffn_with_multiple_input_output": 2,
            "transformer_block": 6,
        }[model_type]
    else:
        len_linear_nodes = {
            "ffn_with_multiple_input_output": 5,
            "transformer_block": 18,
        }[model_type]
        len_fwd_linear_nodes = {
            "ffn_with_multiple_input_output": 2,
            "transformer_block": 6,
        }[model_type]

    assert len(linear_nodes) == len_linear_nodes
    fwd_linear_nodes = linear_nodes[0:len_fwd_linear_nodes]
    bwd_linear_grad_weight_nodes = linear_nodes[len_fwd_linear_nodes::2]
    bwd_linear_grad_input_nodes = linear_nodes[(len_fwd_linear_nodes + 1) :: 2]

    # and check that matmuls have full replication on weights during fwd,
    # which maps to DDP / FSDP

    # fwd
    for node in fwd_linear_nodes:
        p = sharding_placement[node]
        # input and output are sharded on batch
        assert p.input_specs[0].placements == (Shard(0),)
        assert p.output_specs.placements == (Shard(0),)
        # weight is replicated, mimicing DDP
        assert p.input_specs[1].placements == (Replicate(),)

    # bwd grad weight
    # For mm: [N, B*S] @ [B*S, K] → batch dim is at position 1 for input 0
    # For einsum: bsn,bsk->nk → batch dim is at position 0 for both inputs
    bwd_grad_weight_shard = (Shard(0),) if is_einsum else (Shard(1),)
    for node in bwd_linear_grad_weight_nodes:
        p = sharding_placement[node]
        assert p.input_specs[0].placements == bwd_grad_weight_shard
        assert p.output_specs.placements == (Partial("sum"),)
        assert p.input_specs[1].placements == (Shard(0),)

    # bwd grad inputs
    for node in bwd_linear_grad_input_nodes:
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


@apply_cuda_patches
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
    device = "cuda"
    model_fn, input_fn = _make_model_and_input_fn(device_mesh_2d, model_type, device)
    with torch.device("meta"):
        model = model_fn()

    with AutoParallel(model, input_fn, device_mesh_2d) as autop:
        placement = (Shard(0), Replicate())
        n_inputs = 2 if model_type == "ffn_with_multiple_input_output" else 1
        n_outputs = 3 if model_type == "ffn_with_multiple_input_output" else 1
        autop.add_input_constraints([placement] * n_inputs)
        autop.add_output_constraints([placement] * n_outputs)
        autop.add_parameter_memory_constraint(low=low_mem, high=high_mem)

        sharding_placement = autop.optimize_placement()

    # check parameters are sharded as expected
    param_nodes = get_param_nodes(autop.gm.graph)
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


def test_in_graph_tensor_ctor(device_mesh_1d):
    dim = 128

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            self.register_buffer("buf", torch.empty(dim))

        def forward(self, x):
            # dumb stuff to have in-graph tensor creation
            x += torch.full([256, 256, 6144], 0, dtype=torch.bfloat16).sum()
            return self.linear(x) + self.buf

        def init_weights(self):
            self.linear.weight = torch.nn.Parameter(torch.ones(dim, dim) * 9.0)
            with torch.no_grad():
                self.linear.bias.fill_(98.6)
            self.buf = torch.arange(dim)

    with torch.device("meta"):
        model = Model(dim)

    batch_size = 512
    local_batch_size = batch_size // device_mesh_1d.size()
    x = DTensor.from_local(
        torch.rand(local_batch_size, dim, device="cuda"),
        device_mesh_1d,
        [Shard(0)],
    )
    parallel_mod = auto_parallel(
        model,
        device_mesh_1d,
        sample_inputs=(x,),
        out_shardings=(Shard(0),),
    )
    parallel_mod.to_empty(device="cuda")
    parallel_mod.init_weights()
    assert torch.equal(
        parallel_mod.get_parameter("linear.weight").full_tensor(),
        torch.full((dim, dim), 9.0, device="cuda"),
    )
    assert torch.equal(
        parallel_mod.get_parameter("linear.bias").full_tensor(),
        torch.full((dim,), 98.6, device="cuda"),
    )
    assert torch.equal(
        parallel_mod.get_buffer("buf").full_tensor(), torch.arange(dim, device="cuda")
    )


class LocalMapTransformerBlock(nn.Module):
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
        @local_map(
            out_placements=((Shard(0), Shard(2)),),
            in_placements=(
                (Shard(0), Shard(2)),  # query
                (Shard(0), Replicate()),  # key
                (Shard(0), Replicate()),  # value
            ),
            redistribute_inputs=True,
            in_grad_placements=None,
        )
        def _context_parallel_attention(query, key, value):
            out = F.scaled_dot_product_attention(
                query=query, key=key, value=value, is_causal=False
            )
            return (out,)

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
        k = k.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
        v = v.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)

        o = _context_parallel_attention(q, k, v)[0]
        o = o.permute(0, 2, 1, 3).flatten(-2)

        o = self.wo(o)

        o0 = o + x

        o = self.w1(o0)
        o = torch.nn.functional.relu(o)
        o = self.w2(o)

        o = o0 + o
        return o


@apply_cuda_patches
def test_local_map_placement_respected(device_mesh_2d, device="cuda"):
    bs = 8 * device_mesh_2d.shape[0]
    dim1 = 6144
    dim2 = dim1 * 4
    nheads = 48
    seq_len = 256

    def model_fn():
        return LocalMapTransformerBlock(nheads, dim1, dim2)

    def input_fn():
        return torch.randn(bs, seq_len, dim1, device=device, requires_grad=True)

    with torch.device("meta"):
        model = model_fn()

    with AutoParallel(model, input_fn, device_mesh_2d) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)

        x_sharding = (Shard(0), Shard(1))
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([x_sharding])

        sharding_placement = autop.optimize_placement()

    local_map_nodes = []
    for node in autop.gm.graph.nodes:
        if "local_map_kwargs" in node.meta:
            local_map_nodes.append(node)

    assert len(local_map_nodes) == 2, "Expected a fw and bw node"
    fw_node = local_map_nodes[0]
    bw_node = local_map_nodes[1]

    fw_spec = sharding_placement[fw_node]
    bw_spec = sharding_placement[bw_node]

    # Check fw inputs
    assert len(fw_spec.input_specs) == 3  # query, key, value
    q_spec, k_spec, v_spec = fw_spec.input_specs
    assert q_spec.placements == (Shard(dim=0), Shard(dim=2))
    assert k_spec.placements == v_spec.placements == (Shard(0), Replicate())

    # Check fw outputs incl saved activations
    assert len(fw_spec.output_specs) == 8
    fw_out_spec, *act_specs = fw_spec.output_specs
    assert fw_out_spec.placements == (Shard(0), Shard(2))
    for act_spec in act_specs:
        assert act_spec.placements == (Replicate(), Replicate())

    # Check bw inputs incl saved activations
    assert len(bw_spec.input_specs) == 8
    *act_specs, bw_in_spec = bw_spec.input_specs
    assert bw_in_spec.placements == (Shard(0), Shard(2))
    for act_spec in act_specs:
        assert act_spec.placements == (Replicate(), Replicate())

    # Check bw outputs
    assert len(bw_spec.output_specs) == 3  # query, key, value
    grad_q_spec, grad_k_spec, grad_v_spec = bw_spec.output_specs
    assert grad_q_spec.placements == (Shard(dim=0), Shard(dim=2))
    assert grad_k_spec.placements == grad_v_spec.placements == (Shard(0), Replicate())


def _flex_attention_body(query, key, value):
    out = F.scaled_dot_product_attention(
        query=query, key=key, value=value, is_causal=False
    )
    return (out,)


def _flex_clone_body(x):
    return (x.clone(),)


def _flex_where_identity_body(x):
    return (torch.where(x == x, x, x).clone(),)


def _flex_sin_body(x):
    return (torch.sin(x),)


def _flex_shifted_cos_body(x):
    return (torch.cos(x - torch.pi / 2),)


def _flex_sin_with_detached_body(x):
    return torch.sin(x), x.detach().clone()


def _flex_shifted_cos_with_detached_body(x):
    return torch.cos(x - torch.pi / 2), x.detach().clone()


def _make_flex_pointwise(
    mesh,
    default_fn,
    alternative_fn,
    out_placements,
    *,
    names=("default", "alternative"),
    cost_hints=(100.0, 0.0),
):
    in_placements = ((Replicate(),),)
    return flex_local_map(
        default_fn,
        alternatives=[
            {
                "name": names[0],
                "in_placements": in_placements,
                "out_placements": out_placements,
                "cost_hint": cost_hints[0],
            },
            {
                "name": names[1],
                "fn": alternative_fn,
                "in_placements": in_placements,
                "out_placements": out_placements,
                "cost_hint": cost_hints[1],
            },
        ],
        device_mesh=mesh,
        redistribute_inputs=True,
    )


class FlexLocalMapDifferentActivations(nn.Module):
    def __init__(self, mesh, cost_hints=(100.0, 0.0)):
        super().__init__()
        self.scale = nn.Parameter(torch.empty(()))
        placements = ((Replicate(),),)
        self.pointwise = _make_flex_pointwise(
            mesh,
            _flex_clone_body,
            _flex_where_identity_body,
            placements,
            cost_hints=cost_hints,
        )

    def forward(self, x):
        return self.pointwise(x * self.scale)[0]


def _flex_two_differentiable_outputs(x):
    return x.clone(), torch.sin(x)


def _flex_one_differentiable_output(x):
    return x.clone(), x.detach().clone()


class FlexLocalMapChangedDifferentiableOutputs(nn.Module):
    def __init__(self, mesh):
        super().__init__()
        placements = ((Replicate(),), (Replicate(),))
        self.pointwise = _make_flex_pointwise(
            mesh,
            _flex_two_differentiable_outputs,
            _flex_one_differentiable_output,
            placements,
        )

    def forward(self, x):
        return self.pointwise(x)


class FlexLocalMapMultiOutput(nn.Module):
    def __init__(self, mesh):
        super().__init__()
        out_placements = ((Replicate(),), (Replicate(),))
        self.pointwise = _make_flex_pointwise(
            mesh,
            _flex_sin_with_detached_body,
            _flex_shifted_cos_with_detached_body,
            out_placements,
        )

    def forward(self, x):
        return self.pointwise(x)


class RepeatedFlexLocalMap(nn.Module):
    def __init__(self, mesh):
        super().__init__()
        placements = ((Replicate(),),)
        self.pointwise = _make_flex_pointwise(
            mesh,
            _flex_sin_body,
            _flex_shifted_cos_body,
            placements,
        )

    def forward(self, x):
        return self.pointwise(self.pointwise(x)[0])[0]


class FlexLocalMapPointwiseBlock(nn.Module):
    def __init__(self, mesh, **kwargs):
        super().__init__()
        self.linear = nn.Linear(128, 128, bias=False)
        placements = ((Replicate(),),)
        self.pointwise = _make_flex_pointwise(
            mesh,
            _flex_sin_body,
            _flex_shifted_cos_body,
            placements,
            **kwargs,
        )

    def forward(self, x):
        return self.pointwise(self.linear(x))[0]


class RepeatedFlexLocalMapBlocks(nn.Module):
    def __init__(self, mesh, mismatch=None):
        super().__init__()
        kwargs = {}
        if mismatch == "name":
            kwargs["names"] = ("default", "different_alternative")
        elif mismatch == "cost_hint":
            kwargs["cost_hints"] = (100.0, 1.0)
        self.layers = nn.ModuleList(
            [
                FlexLocalMapPointwiseBlock(mesh),
                FlexLocalMapPointwiseBlock(mesh, **kwargs),
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FlexLocalMapDifferentPlacements(nn.Module):
    def __init__(self, mesh):
        super().__init__()
        self.pointwise = flex_local_map(
            _flex_sin_body,
            alternatives=[
                {
                    "name": "replicated",
                    "in_placements": ((Replicate(),),),
                    "out_placements": ((Replicate(),),),
                    "cost_hint": 0.0,
                },
                {
                    "name": "sharded",
                    "fn": _flex_shifted_cos_body,
                    "in_placements": ((Shard(0),),),
                    "out_placements": ((Shard(0),),),
                    "cost_hint": 1_000_000.0,
                },
            ],
            device_mesh=mesh,
            redistribute_inputs=True,
        )

    def forward(self, x):
        return self.pointwise(x)[0]


class PlainLocalMapReference(nn.Module):
    def __init__(
        self,
        mesh,
        fn,
        input_placement,
        output_placements,
        *,
        use_scale=False,
        unwrap_single=False,
    ):
        super().__init__()
        self.scale = nn.Parameter(torch.empty(())) if use_scale else None
        self.unwrap_single = unwrap_single
        self.pointwise = local_map(
            fn,
            in_placements=(input_placement,),
            out_placements=output_placements,
            device_mesh=mesh,
            redistribute_inputs=True,
        )

    def forward(self, x):
        if self.scale is not None:
            x = x * self.scale
        output = self.pointwise(x)
        return output[0] if self.unwrap_single else output


def _canonical_target(target):
    if isinstance(target, str):
        return target
    name = getattr(target, "name", None)
    if callable(name):
        try:
            return name()
        except TypeError:
            pass
    module = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None)
    if module is not None and qualname is not None:
        return f"{module}.{qualname}"
    return str(target)


def _canonical_tensor(value):
    return (
        "tensor",
        tuple(int(dim) if isinstance(dim, int) else str(dim) for dim in value.shape),
        str(value.dtype),
        tuple(int(dim) for dim in value.stride()),
        value.requires_grad,
        value.device.type,
    )


def _canonical_local_map_kwargs(kwargs):
    keys = (
        "in_placements",
        "out_placements",
        "in_grad_placements",
        "redistribute_inputs",
        "device_mesh",
    )
    return tuple(
        (key, _canonical_value(kwargs.get(key))) for key in keys if key in kwargs
    )


def _canonical_value(value):
    if isinstance(value, torch.Tensor):
        return _canonical_tensor(value)
    if isinstance(value, torch.fx.GraphModule):
        return _canonical_graph_signature(value)
    if isinstance(value, DeviceMesh):
        return (
            "device_mesh",
            value.device_type,
            tuple(value.shape),
            tuple(value.mesh_dim_names) if value.mesh_dim_names is not None else None,
        )
    if isinstance(value, Placement):
        return (type(value).__name__, repr(value))
    if isinstance(value, dict):
        return tuple(
            (str(key), _canonical_value(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, tuple):
        return ("tuple", tuple(_canonical_value(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_canonical_value(item) for item in value))
    if isinstance(value, slice):
        return (
            "slice",
            _canonical_value(value.start),
            _canonical_value(value.stop),
            _canonical_value(value.step),
        )
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (torch.dtype, torch.device, torch.layout)):
        return str(value)
    if callable(value):
        return ("callable", _canonical_target(value))
    return (type(value).__name__, repr(value))


def _canonical_arg(value, node_signatures):
    if isinstance(value, torch.fx.Node):
        return ("node", node_signatures[value])
    if isinstance(value, tuple):
        return ("tuple", tuple(_canonical_arg(item, node_signatures) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_canonical_arg(item, node_signatures) for item in value))
    if isinstance(value, dict):
        return tuple(
            (str(key), _canonical_arg(item, node_signatures))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, slice):
        return (
            "slice",
            _canonical_arg(value.start, node_signatures),
            _canonical_arg(value.stop, node_signatures),
            _canonical_arg(value.step, node_signatures),
        )
    return _canonical_value(value)


def _fetch_attr(module, target):
    value = module
    for atom in target.split("."):
        value = getattr(value, atom)
    return value


def _canonical_meta(meta):
    result = []
    if "local_map_kwargs" in meta:
        result.append(
            (
                "local_map_kwargs",
                _canonical_local_map_kwargs(meta["local_map_kwargs"]),
            )
        )
    for key in ("num_activations", "is_backward", "partitioner_tag"):
        if key in meta:
            result.append((key, _canonical_value(meta[key])))
    for key in ("example_value", "val"):
        if key in meta:
            result.append(("example_value", _canonical_value(meta[key])))
            break
    return tuple(result)


def _canonical_graph_signature(gm):
    node_signatures = {}
    placeholder_index = 0
    output_signature = None
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            signature = (
                "placeholder",
                placeholder_index,
                _canonical_meta(node.meta),
            )
            placeholder_index += 1
        elif node.op == "get_attr":
            signature = (
                "get_attr",
                _canonical_value(_fetch_attr(gm, node.target)),
                _canonical_meta(node.meta),
            )
        else:
            signature = (
                node.op,
                _canonical_target(node.target),
                _canonical_arg(node.args, node_signatures),
                _canonical_arg(node.kwargs, node_signatures),
                _canonical_meta(node.meta),
            )
        node_signatures[node] = signature
        if node.op == "output":
            output_signature = signature

    counts = Counter(node_signatures.values())
    return (
        "graph",
        _canonical_meta(gm.meta),
        tuple(sorted(counts.items(), key=lambda item: repr(item[0]))),
        output_signature,
    )


def _local_map_graph_signatures(gm):
    nodes = [node for node in gm.graph.nodes if "local_map_kwargs" in node.meta]
    forward = [
        node for node in nodes if node.meta.get("partitioner_tag") != "is_backward"
    ]
    backward = [
        node for node in nodes if node.meta.get("partitioner_tag") == "is_backward"
    ]
    assert len(forward) == 1
    assert len(backward) == 1
    return {
        "forward": _canonical_graph_signature(_body(gm, forward[0])),
        "backward": _canonical_graph_signature(_body(gm, backward[0])),
    }


def _capture_graph_signatures(
    model,
    input_fn,
    mesh,
    input_constraints,
    output_constraints,
    *,
    force_flex_alternative=None,
):
    with AutoParallel(model, input_fn, mesh) as autop:
        autop.add_input_constraints(input_constraints)
        autop.add_output_constraints(output_constraints)
        initial_bodies = _local_map_graph_signatures(autop.gm)
        solution = autop.optimize_placement()
        if force_flex_alternative is not None:
            forward = next(
                node
                for node in autop.gm.graph.nodes
                if get_flex_local_map_alternatives(
                    node.meta.get("local_map_kwargs", {})
                )
                is not None
                and node.meta.get("partitioner_tag") != "is_backward"
            )
            autop.sharding_optimizer.add_node_constraint(
                forward, placement=input_constraints[0]
            )
            solution = autop.sharding_optimizer.resolve()
        if force_flex_alternative is not None:
            selected = {
                spec.flex_local_map_alternative_index
                for node, spec in solution.items()
                if get_flex_local_map_alternatives(
                    node.meta.get("local_map_kwargs", {})
                )
                is not None
            }
            assert selected == {force_flex_alternative}
        autop.apply_placement(solution)
        bodies = _local_map_graph_signatures(autop.gm)
        return {
            **bodies,
            "initial_forward": initial_bodies["forward"],
            "joint": _canonical_graph_signature(autop.gm),
            "parallel": _canonical_graph_signature(autop.parallel_gm),
        }


@apply_cuda_patches
@pytest.mark.parametrize(
    ("cost_hints", "expected_index"),
    (((0.0, 100.0), 0), ((100.0, 0.0), 1)),
)
def test_flex_local_map_selected_body_forward_backward(
    device_mesh_1d, cost_hints, expected_index, device="cuda"
):
    shape = (512, 128)

    def input_fn():
        return torch.randn(*shape, device=device, requires_grad=True)

    with torch.device("meta"):
        model = FlexLocalMapDifferentActivations(device_mesh_1d, cost_hints)

    with AutoParallel(model, input_fn, device_mesh_1d) as autop:
        replicated = (Replicate(),)
        autop.add_input_constraints([replicated])
        autop.add_output_constraints([replicated])
        solution = autop.optimize_placement()
        selected = {
            spec.flex_local_map_alternative_index
            for node, spec in solution.items()
            if get_flex_local_map_alternatives(node.meta.get("local_map_kwargs", {}))
            is not None
        }
        assert selected == {expected_index}
        parallel_model = autop.apply_placement(solution)

    parallel_model.to_empty(device=device)
    with torch.no_grad():
        parallel_model.scale.fill_(1.25)
    actual_input = torch.randn(*shape, device=device, requires_grad=True)
    expected_input = actual_input.detach().clone().requires_grad_(True)
    expected_scale = torch.tensor(1.25, device=device, requires_grad=True)
    actual = parallel_model(actual_input)
    expected = _flex_where_identity_body(expected_input * expected_scale)[0]
    actual.sum().backward()
    expected.sum().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_input.grad, expected_input.grad)
    torch.testing.assert_close(
        parallel_model.scale.grad.to_local(), expected_scale.grad
    )


@apply_cuda_patches
def test_flex_local_map_apply_requires_selected_alternative(
    device_mesh_1d, device="cuda"
):
    shape = (512, 128)

    def input_fn():
        return torch.randn(*shape, device=device, requires_grad=True)

    with torch.device("meta"):
        model = FlexLocalMapDifferentActivations(device_mesh_1d)

    with AutoParallel(model, input_fn, device_mesh_1d) as autop:
        replicated = (Replicate(),)
        autop.add_input_constraints([replicated])
        autop.add_output_constraints([replicated])
        solution = autop.optimize_placement()
        forward = next(
            node
            for node in solution
            if get_flex_local_map_alternatives(node.meta.get("local_map_kwargs", {}))
            is not None
            and node.meta.get("partitioner_tag") != "is_backward"
        )
        del solution[forward].flex_local_map_alternative_index
        with pytest.raises(RuntimeError, match="has no selected alternative"):
            autop.apply_placement(solution)


@apply_cuda_patches
def test_flex_local_map_rejects_changed_differentiable_outputs(
    device_mesh_1d, device="cuda"
):
    shape = (512, 128)

    def input_fn():
        return torch.randn(*shape, device=device, requires_grad=True)

    with torch.device("meta"):
        model = FlexLocalMapChangedDifferentiableOutputs(device_mesh_1d)

    with AutoParallel(model, input_fn, device_mesh_1d) as autop:
        replicated = (Replicate(),)
        autop.add_input_constraints([replicated])
        autop.add_output_constraints([replicated, replicated])
        solution = autop.optimize_placement()
        with pytest.raises(RuntimeError, match="changed its differentiable outputs"):
            autop.apply_placement(solution)


@apply_cuda_patches
def test_flex_local_map_selected_body_with_nondifferentiable_output(
    device_mesh_1d, device="cuda"
):
    shape = (512, 128)

    def input_fn():
        return torch.randn(*shape, device=device, requires_grad=True)

    with torch.device("meta"):
        model = FlexLocalMapMultiOutput(device_mesh_1d)

    with AutoParallel(model, input_fn, device_mesh_1d) as autop:
        replicated = (Replicate(),)
        autop.add_input_constraints([replicated])
        autop.add_output_constraints([replicated, replicated])
        parallel_model = autop.apply_placement(autop.optimize_placement())

    actual_input = torch.randn(*shape, device=device, requires_grad=True)
    expected_input = actual_input.detach().clone().requires_grad_(True)
    actual = parallel_model(actual_input)
    expected = _flex_shifted_cos_with_detached_body(expected_input)
    actual[0].sum().backward()
    expected[0].sum().backward()

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    torch.testing.assert_close(actual_input.grad, expected_input.grad)


@apply_cuda_patches
def test_repeated_flex_local_map_selected_body_dynamic_shape(
    device_mesh_1d, device="cuda"
):
    traced_shape = (512, 128)
    runtime_shape = (256, 128)

    def input_fn():
        return torch.randn(*traced_shape, device=device, requires_grad=True)

    with torch.device("meta"):
        model = RepeatedFlexLocalMap(device_mesh_1d)

    with AutoParallel(model, input_fn, device_mesh_1d, dynamic=True) as autop:
        replicated = (Replicate(),)
        autop.add_input_constraints([replicated])
        autop.add_output_constraints([replicated])
        parallel_model = autop.apply_placement(autop.optimize_placement())

    actual_input = torch.randn(*runtime_shape, device=device, requires_grad=True)
    expected_input = actual_input.detach().clone().requires_grad_(True)
    actual = parallel_model(actual_input)
    expected = _flex_shifted_cos_body(_flex_shifted_cos_body(expected_input)[0])[0]
    actual.sum().backward()
    expected.sum().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_input.grad, expected_input.grad)


@apply_cuda_patches
def test_repeated_flex_local_map_shares_cluster_choice(device_mesh_1d, device="cuda"):
    shape = (512, 128)

    def input_fn():
        return torch.randn(*shape, device=device, requires_grad=True)

    with torch.device("meta"):
        model = RepeatedFlexLocalMapBlocks(device_mesh_1d)

    with AutoParallel(model, input_fn, device_mesh_1d) as autop:
        replicated = (Replicate(),)
        autop.add_input_constraints([replicated])
        autop.add_output_constraints([replicated])
        opt = autop.sharding_optimizer
        flex_nodes = [
            node
            for node in opt.strats
            if get_flex_local_map_alternatives(node.meta.get("local_map_kwargs", {}))
            is not None
        ]
        roots_by_phase = {}
        for node in flex_nodes:
            node_idx = opt.node_map[node]
            roots = {
                root_key[0]
                for linked_key, root_key in opt.cluster_links.items()
                if linked_key[0] == node_idx
            }
            root_idx = next(iter(roots)) if roots else node_idx
            roots_by_phase.setdefault(node.meta.get("partitioner_tag"), set()).add(
                root_idx
            )
        assert set(roots_by_phase) == {"is_forward", "is_backward"}
        assert len(roots_by_phase["is_forward"]) == 1

        solution = autop.optimize_placement()
        selected = {
            solution[
                opt._concrete_to_orig.get(node, node)
            ].flex_local_map_alternative_index
            for node in flex_nodes
        }
        assert selected == {1}


@apply_cuda_patches
@pytest.mark.parametrize("mismatch", ("name", "cost_hint"))
def test_repeated_flex_local_map_rejects_mismatched_contracts(
    device_mesh_1d, mismatch, device="cuda"
):
    shape = (512, 128)

    def input_fn():
        return torch.randn(*shape, device=device, requires_grad=True)

    with torch.device("meta"):
        model = RepeatedFlexLocalMapBlocks(device_mesh_1d, mismatch)

    with pytest.raises(RuntimeError, match="must declare the same flex alternatives"):
        with AutoParallel(model, input_fn, device_mesh_1d):
            pass


@apply_cuda_patches
def test_flex_local_map_resolve_then_apply_different_placements(
    device_mesh_1d, device="cuda"
):
    traced_shape = (512, 128)

    def input_fn():
        return torch.randn(*traced_shape, device=device, requires_grad=True)

    with torch.device("meta"):
        model = FlexLocalMapDifferentPlacements(device_mesh_1d)

    with AutoParallel(model, input_fn, device_mesh_1d) as autop:
        sharded = (Shard(0),)
        autop.add_input_constraints([sharded])
        autop.add_output_constraints([sharded])
        flex_nodes = [
            node
            for node in autop.gm.graph.nodes
            if get_flex_local_map_alternatives(node.meta.get("local_map_kwargs", {}))
            is not None
        ]
        forward = next(
            node
            for node in flex_nodes
            if node.meta.get("partitioner_tag") != "is_backward"
        )
        initial = autop.optimize_placement()
        assert initial[forward].flex_local_map_alternative_index == 0

        autop.sharding_optimizer.add_node_constraint(forward, placement=sharded)
        resolved = autop.sharding_optimizer.resolve()
        assert resolved[forward].flex_local_map_alternative_index == 1
        parallel_model = autop.apply_placement(resolved)

    local_shape = (traced_shape[0] // device_mesh_1d.size(), traced_shape[1])
    actual_input = torch.randn(*local_shape, device=device, requires_grad=True)
    expected_input = actual_input.detach().clone().requires_grad_(True)
    actual = parallel_model(actual_input)
    expected = _flex_shifted_cos_body(expected_input)[0]
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_input.grad, expected_input.grad)


@apply_cuda_patches
@pytest.mark.parametrize(
    "case", ("same_boundary", "different_boundary", "nondifferentiable_output")
)
def test_flex_local_map_grafted_graph_matches_direct_trace(
    device_mesh_1d, case, device="cuda"
):
    shape = (512, 128)

    def input_fn():
        return torch.randn(*shape, device=device, requires_grad=True)

    replicated = (Replicate(),)
    sharded = (Shard(0),)
    with torch.device("meta"):
        if case == "same_boundary":
            flex_model = FlexLocalMapDifferentActivations(device_mesh_1d)
            direct_model = PlainLocalMapReference(
                device_mesh_1d,
                _flex_where_identity_body,
                replicated,
                (replicated,),
                use_scale=True,
                unwrap_single=True,
            )
            input_constraints = [replicated]
            output_constraints = [replicated]
        elif case == "different_boundary":
            flex_model = FlexLocalMapDifferentPlacements(device_mesh_1d)
            direct_model = PlainLocalMapReference(
                device_mesh_1d,
                _flex_shifted_cos_body,
                sharded,
                (sharded,),
                unwrap_single=True,
            )
            input_constraints = [sharded]
            output_constraints = [sharded]
        else:
            flex_model = FlexLocalMapMultiOutput(device_mesh_1d)
            direct_model = PlainLocalMapReference(
                device_mesh_1d,
                _flex_shifted_cos_with_detached_body,
                replicated,
                (replicated, replicated),
            )
            input_constraints = [replicated]
            output_constraints = [replicated, replicated]

    flex_signatures = _capture_graph_signatures(
        flex_model,
        input_fn,
        device_mesh_1d,
        input_constraints,
        output_constraints,
        force_flex_alternative=1,
    )
    direct_signatures = _capture_graph_signatures(
        direct_model,
        input_fn,
        device_mesh_1d,
        input_constraints,
        output_constraints,
    )

    assert flex_signatures["initial_forward"] != direct_signatures["forward"]
    for graph in ("forward", "backward", "joint", "parallel"):
        assert flex_signatures[graph] == direct_signatures[graph], graph


class FlexLocalMapTransformerBlock(nn.Module):
    def __init__(self, nheads, dim1, dim2, mesh):
        super().__init__()
        self.nheads = nheads
        bias = False
        self.wq = nn.Linear(dim1, dim1, bias=bias)
        self.wk = nn.Linear(dim1, dim1, bias=bias)
        self.wv = nn.Linear(dim1, dim1, bias=bias)
        self.wo = nn.Linear(dim1, dim1, bias=bias)
        self.w1 = nn.Linear(dim1, dim2, bias=bias)
        self.w2 = nn.Linear(dim2, dim1, bias=bias)
        # flex_local_map must be built OUTSIDE forward so its out_placements carrier
        # is a Dynamo Source at trace time (an in-forward-constructed tuple subclass
        # traces as an empty value); this needs an explicit device_mesh.
        self.attention = flex_local_map(
            _flex_attention_body,
            alternatives=[
                {
                    "name": "sequence_parallel",
                    "in_placements": (
                        (Shard(0), Shard(2)),
                        (Shard(0), Replicate()),
                        (Shard(0), Replicate()),
                    ),
                    "out_placements": ((Shard(0), Shard(2)),),
                    "cost_hint": 0.0,
                },
                {
                    "name": "replicated_boundary",
                    "in_placements": (
                        (Shard(0), Replicate()),
                        (Shard(0), Replicate()),
                        (Shard(0), Replicate()),
                    ),
                    "out_placements": ((Shard(0), Replicate()),),
                    "cost_hint": 0.0,
                },
            ],
            redistribute_inputs=True,
            device_mesh=mesh,
        )

    def forward(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
        k = k.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
        v = v.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)

        o = self.attention(q, k, v)[0]
        o = o.permute(0, 2, 1, 3).flatten(-2)

        o = self.wo(o)
        o0 = o + x
        o = self.w2(torch.nn.functional.relu(self.w1(o0)))
        return o0 + o


@apply_cuda_patches
def test_flex_local_map_alternatives_visible_to_solver(device_mesh_2d, device="cuda"):
    # Validates the graph AutoParallel hands to the solver, not e2e execution:
    # flex_local_map with 2 alternatives must trace (surviving the model deepcopy
    # in AutoParallel.__init__), expose both alternatives on the forward local_map
    # node, produce one solver strategy per alternative, and let the solver pick one.
    bs = 8 * device_mesh_2d.shape[0]
    dim1 = 6144
    dim2 = dim1 * 4
    nheads = 48
    seq_len = 256

    def input_fn():
        return torch.randn(bs, seq_len, dim1, device=device, requires_grad=True)

    with torch.device("meta"):
        model = FlexLocalMapTransformerBlock(nheads, dim1, dim2, mesh=device_mesh_2d)

    # Entering the context deep-copies and traces the model; before the
    # _FlexLocalMapOutPlacements.__getnewargs__ fix this raised in copy.deepcopy.
    with AutoParallel(model, input_fn, device_mesh_2d) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)
        x_sharding = (Shard(0), Shard(1))
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([x_sharding])

        local_map_nodes = [
            n for n in autop.gm.graph.nodes if "local_map_kwargs" in n.meta
        ]
        assert len(local_map_nodes) == 2, "Expected a fw and bw local_map node"

        # The alternatives survive Dynamo + AOTAutograd on the forward node.
        fw_node = next(n for n in local_map_nodes if "backward" not in str(n.target))
        alternatives = get_flex_local_map_alternatives(fw_node.meta["local_map_kwargs"])
        assert alternatives is not None
        assert len(alternatives) == 2

        # Both the forward and (after normalize_flex_local_map_backward) the backward
        # local_map node are alternative-aware; the forward has a single declared output.
        opt = autop.sharding_optimizer
        flex_nodes = [
            n
            for n in opt.strats
            if get_flex_local_map_alternatives(n.meta.get("local_map_kwargs", {}))
            is not None
        ]
        assert len(flex_nodes) == 2
        fw_flex = next(
            n
            for n in flex_nodes
            if len(n.meta["local_map_kwargs"]["out_placements"]) == 1
        )
        strategies = opt.strats[fw_flex].strategies
        assert len(strategies) == 2
        assert {s.output_specs[0].placements for s in strategies} == {
            (Shard(0), Shard(2)),
            (Shard(0), Replicate()),
        }

        sharding_placement = autop.optimize_placement()

    # The solver produced a feasible solution that selected one of the alternatives.
    chosen = sharding_placement[fw_node].output_specs[0].placements
    assert chosen in ((Shard(0), Shard(2)), (Shard(0), Replicate()))


def _flex_fw_bw_nodes(nodes):
    """Return (forward, backward) flex local_map nodes. The forward node has a single
    declared output; the backward node's outputs are the (mirrored) input grads."""
    flex = [
        n
        for n in nodes
        if get_flex_local_map_alternatives(n.meta.get("local_map_kwargs", {}))
        is not None
    ]
    assert len(flex) == 2, f"expected fw+bw flex local_map nodes, got {len(flex)}"
    fw = next(n for n in flex if len(n.meta["local_map_kwargs"]["out_placements"]) == 1)
    bw = next(n for n in flex if n is not fw)
    return fw, bw


@apply_cuda_patches
def test_flex_local_map_backward_is_alternative_aware(device_mesh_2d, device="cuda"):
    # normalize_flex_local_map_backward makes the backward local_map node
    # alternative-aware: one solver strategy per alternative, mirrored (its grad outputs
    # take the forward's input placements), and the solver places the forward and
    # backward on the same alternative.
    bs = 8 * device_mesh_2d.shape[0]
    dim1 = 6144
    dim2 = dim1 * 4
    nheads = 48
    seq_len = 256

    def input_fn():
        return torch.randn(bs, seq_len, dim1, device=device, requires_grad=True)

    with torch.device("meta"):
        model = FlexLocalMapTransformerBlock(nheads, dim1, dim2, mesh=device_mesh_2d)

    with AutoParallel(model, input_fn, device_mesh_2d) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)
        x_sharding = (Shard(0), Shard(1))
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([x_sharding])

        # Backward node has one strategy per alternative, mirrored: its first grad output
        # takes the forward's first input placement per alternative.
        opt = autop.sharding_optimizer
        _, bw_concrete = _flex_fw_bw_nodes(list(opt.strats))
        bw_strats = opt.strats[bw_concrete].strategies
        assert len(bw_strats) == 2
        assert {s.output_specs[0].placements for s in bw_strats} == {
            (Shard(0), Shard(2)),
            (Shard(0), Replicate()),
        }

        fw_node, bw_node = _flex_fw_bw_nodes(list(autop.gm.graph.nodes))
        sharding_placement = autop.optimize_placement()

    # Forward and backward are placed on the same alternative.
    fw_idx = getattr(
        sharding_placement[fw_node], "flex_local_map_alternative_index", None
    )
    bw_idx = getattr(
        sharding_placement[bw_node], "flex_local_map_alternative_index", None
    )
    assert fw_idx is not None
    assert fw_idx == bw_idx


@apply_cuda_patches
def test_get_attr_nodes(device_mesh_1d):
    """Test that get_attr nodes (module attributes like constant tensors) are handled correctly."""
    dim1 = 256
    dim2 = dim1 * 4
    bs = 8 * device_mesh_1d.shape[0]

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(dim1, dim2)

        def forward(self, x):
            # new_tensor creates a constant tensor that becomes a get_attr node in the FX graph
            y = x.new_tensor([0, 1, 2, 3, 4, 5, 6, 7])[None, :, None]
            return self.linear(x) + y

    def input_fn():
        return torch.rand(bs, 8, dim1, device="cuda", requires_grad=True)

    with torch.device("meta"):
        model = Model()

    with AutoParallel(model, input_fn, device_mesh_1d) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)

        x_sharding = (Shard(0),)
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([x_sharding])

        sharding_placement = autop.optimize_placement()

    # Find get_attr nodes in the graph
    get_attr_nodes = [node for node in autop.gm.graph.nodes if node.op == "get_attr"]
    assert len(get_attr_nodes) > 0, "Expected at least one get_attr node"

    # The get_attr node should have a valid sharding placement
    for node in get_attr_nodes:
        assert (
            node in sharding_placement
        ), f"get_attr node {node} missing from sharding_placement"
        spec = sharding_placement[node]
        # The constant tensor is small and used in broadcasting, so it should be replicated
        assert spec.output_specs.placements == (
            Replicate(),
        ), f"Expected get_attr node to be Replicate(), got {spec.output_specs.placements}"


@apply_cuda_patches
def test_parameter_memory_constraint_indivisible_param(device_mesh_2d):
    """Parameter whose size is >= world_size but not divisible by it
    should not make the memory constraint infeasible."""
    # world_size = 32*8 = 256. A bias of size 280 is >= 256 but 280 % 256 != 0,
    # so it can't be fully sharded across all devices.
    dim1 = 1024
    dim2 = 280

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(dim1, dim2, bias=True)
            self.linear2 = nn.Linear(dim2, dim1, bias=True)

        def forward(self, x):
            return self.linear2(F.relu(self.linear1(x)))

    bs = 8 * device_mesh_2d.shape[0]

    def input_fn():
        return torch.rand(bs, dim1, device="cuda", requires_grad=True)

    with torch.device("meta"):
        model = Model()

    with AutoParallel(model, input_fn, device_mesh_2d) as autop:
        x_sharding = (Shard(0), Replicate())
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([x_sharding])
        autop.add_parameter_memory_constraint(low=None, high=None)

        sharding_placement = autop.optimize_placement()

    # Should solve without error. Verify params got some placement.
    param_nodes = get_param_nodes(autop.gm.graph)
    assert len(param_nodes) > 0
    for node in param_nodes:
        assert node in sharding_placement


@apply_cuda_patches
def test_world_size_larger_than_parameter(device_mesh_1d):
    # make a parameter which is smaller than the world size
    dim: int = device_mesh_1d.shape[0] // 2

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.param = nn.Parameter(torch.rand(dim))

        def forward(self, x):
            return x + self.param

        def init_weights(self):
            self.param.uniform_()

    def input_fn():
        b = 512
        inputs = (torch.rand(b, dim, device="cuda"),)
        return inputs

    with torch.device("meta"):
        model = Model(dim)
    with AutoParallel(
        model,
        input_fn,
        device_mesh_1d,
    ) as autop:
        x_sharding = (Shard(0),)
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([x_sharding])
        autop.add_parameter_memory_constraint()
        sharding_placement = autop.optimize_placement()

    # check parameters are sharded as expected
    param_nodes = get_param_nodes(autop.gm.graph)
    for node in param_nodes:
        assert sharding_placement[node].output_specs.placements == (Replicate(),)


def _setup_memory_and_node_constraint(mesh, memory_first):
    """Set up a model where one param is forced to Replicate via add_node_constraint
    while add_parameter_memory_constraint wants to shard everything.

    Without lazy application, the memory constraint would count the
    Replicate-constrained param and become infeasible.

    Args:
        memory_first: if True, add_parameter_memory_constraint is called before
            add_node_constraint (the previously-broken order).
    """
    dim1 = 1024
    dim2 = 4096

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(dim1, dim2, bias=False)
            self.linear2 = nn.Linear(dim2, dim1, bias=False)

        def forward(self, x):
            return self.linear2(F.relu(self.linear1(x)))

    bs = 2048 * mesh.shape[0]

    def input_fn():
        return torch.rand(bs, dim1, device="cuda", requires_grad=True)

    with torch.device("meta"):
        model = Model()

    autop = AutoParallel(model, input_fn, mesh)
    autop.__enter__()

    opt = autop.sharding_optimizer
    # Pick the first param node and force it to Replicate, which conflicts with
    # a tight memory budget that expects all params to be sharded.
    param_nodes = get_param_nodes(opt.graph)
    constrained_node = param_nodes[0]
    orig_constrained_node = opt._concrete_to_orig.get(
        constrained_node, constrained_node
    )
    replicate_placement = (Replicate(),) * mesh.ndim

    x_sharding = (Shard(0),) + (Replicate(),) * (mesh.ndim - 1)
    autop.add_input_constraints([x_sharding])
    autop.add_output_constraints([x_sharding])

    if memory_first:
        autop.add_parameter_memory_constraint(low=None, high=None)
        opt.add_node_constraint(constrained_node, placement=replicate_placement)
    else:
        opt.add_node_constraint(constrained_node, placement=replicate_placement)
        autop.add_parameter_memory_constraint(low=None, high=None)

    return autop, opt, orig_constrained_node, replicate_placement


@apply_cuda_patches
@pytest.mark.parametrize("memory_first", [True, False])
def test_node_constraint_excludes_from_memory_budget_get_solution(
    device_mesh_1d, memory_first
):
    """add_node_constraint + add_parameter_memory_constraint should not conflict,
    regardless of call order.  Verified via get_solution (the primary solve path)."""
    (
        autop,
        opt,
        constrained_node,
        replicate_placement,
    ) = _setup_memory_and_node_constraint(device_mesh_1d, memory_first)
    try:
        solution = autop.optimize_placement()
        assert solution[constrained_node].output_specs.placements == replicate_placement
    finally:
        autop.__exit__(None, None, None)


@apply_cuda_patches
@pytest.mark.parametrize("memory_first", [True, False])
def test_node_constraint_excludes_from_memory_budget_resolve(
    device_mesh_1d, memory_first
):
    """Same as the get_solution test but exercises the resolve() path."""
    (
        autop,
        opt,
        constrained_node,
        replicate_placement,
    ) = _setup_memory_and_node_constraint(device_mesh_1d, memory_first)
    try:
        # First solve to set the objective
        autop.optimize_placement()
        # Re-solve via resolve()
        solution = opt.resolve()
        assert solution[constrained_node].output_specs.placements == replicate_placement
    finally:
        autop.__exit__(None, None, None)


@apply_cuda_patches
def test_node_constraint_after_solve_resolve(device_mesh_1d):
    """Solve once with memory constraint, then add a node constraint and resolve().

    The memory constraint must be rebuilt to exclude the newly constrained
    param, otherwise the re-solve becomes infeasible.
    """
    dim1 = 1024
    dim2 = 4096
    mesh = device_mesh_1d

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(dim1, dim2, bias=False)
            self.linear2 = nn.Linear(dim2, dim1, bias=False)

        def forward(self, x):
            return self.linear2(F.relu(self.linear1(x)))

    bs = 2048 * mesh.shape[0]

    def input_fn():
        return torch.rand(bs, dim1, device="cuda", requires_grad=True)

    with torch.device("meta"):
        model = Model()

    with AutoParallel(model, input_fn, mesh) as autop:
        x_sharding = (Shard(0),)
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([x_sharding])
        autop.add_parameter_memory_constraint(low=None, high=None)

        # First solve — all params can be sharded
        autop.optimize_placement()

        # Now force one param to Replicate and re-solve
        opt = autop.sharding_optimizer
        param_nodes = get_param_nodes(opt.graph)
        constrained_node = param_nodes[0]
        orig_node = opt._concrete_to_orig.get(constrained_node, constrained_node)
        replicate = (Replicate(),)
        opt.add_node_constraint(constrained_node, placement=replicate)

        solution = opt.resolve()
        assert solution[orig_node].output_specs.placements == replicate


@apply_cuda_patches
def test_remove_memory_constraint_then_resolve(device_mesh_1d):
    """Removing the memory constraint by name should prevent it from being
    rebuilt on the next resolve()."""
    dim1 = 1024
    dim2 = 4096
    mesh = device_mesh_1d

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(dim1, dim2, bias=False)
            self.linear2 = nn.Linear(dim2, dim1, bias=False)

        def forward(self, x):
            return self.linear2(F.relu(self.linear1(x)))

    bs = 2048 * mesh.shape[0]

    def input_fn():
        return torch.rand(bs, dim1, device="cuda", requires_grad=True)

    with torch.device("meta"):
        model = Model()

    with AutoParallel(model, input_fn, mesh) as autop:
        x_sharding = (Shard(0),)
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([x_sharding])
        # Memory constraint forces sharding
        autop.add_parameter_memory_constraint(low=None, high=None)
        solution = autop.optimize_placement()

        opt = autop.sharding_optimizer
        param_nodes = get_param_nodes(opt.graph)
        for node in param_nodes:
            orig = opt._concrete_to_orig.get(node, node)
            assert solution[orig].output_specs.placements == (Shard(0),)

        # Remove memory constraint and re-solve — optimizer is free to replicate
        opt.remove_constraints(["memory_constraint_high", "memory_constraint_low"])
        solution = opt.resolve()
        assert "memory_constraint_high" not in opt.prob.constraints
        assert "memory_constraint_low" not in opt.prob.constraints


@apply_cuda_patches
def test_remove_node_constraint_restores_memory_budget(device_mesh_1d):
    """After removing a node constraint, that param should be included in the
    memory budget again on the next resolve()."""
    dim1 = 1024
    dim2 = 4096
    mesh = device_mesh_1d

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(dim1, dim2, bias=False)
            self.linear2 = nn.Linear(dim2, dim1, bias=False)

        def forward(self, x):
            return self.linear2(F.relu(self.linear1(x)))

    bs = 2048 * mesh.shape[0]

    def input_fn():
        return torch.rand(bs, dim1, device="cuda", requires_grad=True)

    with torch.device("meta"):
        model = Model()

    with AutoParallel(model, input_fn, mesh) as autop:
        x_sharding = (Shard(0),)
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([x_sharding])
        autop.add_parameter_memory_constraint(low=None, high=None)

        opt = autop.sharding_optimizer
        param_nodes = get_param_nodes(opt.graph)
        constrained_node = param_nodes[0]
        orig_node = opt._concrete_to_orig.get(constrained_node, constrained_node)
        replicate = (Replicate(),)

        # Force one param to Replicate (excluded from memory budget)
        constraint_names = opt.add_node_constraint(
            constrained_node, placement=replicate
        )
        solution = autop.optimize_placement()
        assert solution[orig_node].output_specs.placements == replicate

        # Remove the node constraint — param should be back in the memory budget
        opt.remove_constraints(constraint_names)
        solution = opt.resolve()
        # With memory budget enforced and no node constraint, the optimizer
        # should shard this param again
        assert solution[orig_node].output_specs.placements == (Shard(0),)
