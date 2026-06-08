# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn.functional as F
from conftest import apply_cuda_patches
from torch import nn
from torch._functorch._aot_autograd.fx_utils import get_param_nodes
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard

from autoparallel.api import AutoParallel, auto_parallel
from autoparallel.collectives import local_map


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


@apply_cuda_patches
def test_invalid_strategies_are_pruned(device_mesh_2d):
    """Infinite-cost (invalid) strategy edges must not be materialized as
    variables or constraints, and pruning them must not change the optimum."""
    import math

    mesh = device_mesh_2d
    model_fn, input_fn = _make_model_and_input_fn(mesh, "transformer_block")
    with torch.device("meta"):
        model = model_fn()

    with AutoParallel(model, input_fn, mesh) as autop:
        autop.add_input_constraints([(Shard(0), Replicate())])
        autop.add_output_constraints([(Shard(0), Replicate())])
        autop.add_parameter_memory_constraint(low=None, high=None)
        opt = autop.sharding_optimizer

        # Invariant: every materialized decision var is finite-cost, and the
        # PuLP variable set is exactly the set of valid (finite) keys.
        assert all(math.isfinite(dv.cost) for dv in opt.decision_vars.values())
        assert set(opt.pulp_variables) == opt._valid_keys
        assert all(k in opt._valid_keys for k in opt.decision_vars)

        # No inf-cost (== 0) constraints should be emitted any more.
        assert not any(name.startswith("inf_cases") for name in opt.prob.constraints)

        # The pruned problem must still solve to a valid solution.
        solution = autop.optimize_placement()
        param_nodes = get_param_nodes(autop.gm.graph)
        for node in param_nodes:
            assert node in solution
