# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import operator

import pytest
import torch
from conftest import apply_cuda_patches
from torch import nn
from torch._functorch._aot_autograd.fx_utils import get_param_and_grad_nodes
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Shard

from autoparallel.api import AutoParallel
from autoparallel.optimize_sharding import ShardingOptimizer


class SimpleLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.linear(x)


class StackedLinear(nn.Module):
    """Multiple identical linear layers for testing repeated_subgraphs."""

    def __init__(self, dim, n_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(dim, dim, bias=False) for _ in range(n_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@apply_cuda_patches
@pytest.mark.parametrize(
    "storage_dtype, reduce_dtype, layer_norm, reuse",
    [
        (torch.float32, torch.float32, False, False),
        (torch.bfloat16, torch.float32, False, False),
        (torch.float32, torch.bfloat16, False, False),
        (torch.float32, torch.bfloat16, True, False),
        (torch.float32, torch.float32, False, True),
    ],
)
def test_autocast_reduction_dtype_without_param_dtype(
    device_mesh_1d, storage_dtype, reduce_dtype, layer_norm, reuse
):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1024, 1024)
            self.norm = nn.LayerNorm(1024) if layer_norm else nn.Identity()

        def forward(self, x):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = self.linear(x)
                if reuse:
                    output = output + self.linear(x)
                return self.norm(output).float()

    mesh = torch.distributed.device_mesh.DeviceMesh(
        "cuda", list(range(8)), mesh_dim_names=("dp",)
    )
    with torch.device("meta"):
        model = Model().to(storage_dtype)

    def inputs():
        return torch.randn(2048 * mesh.size(), 1024, device="cuda")

    policy = MixedPrecisionPolicy(reduce_dtype=reduce_dtype)
    with AutoParallel(model, inputs, mesh, policy, repeated_subgraphs=False) as autop:
        autop.add_input_constraints([(Shard(0),)])
        autop.add_output_constraints([(Shard(0),)])
        placement = autop.optimize_placement()
        opt = autop.sharding_optimizer
        producers = [
            opt.nodes[key[0]].all_input_nodes[key[1]]
            for key in opt.selected_keys
            if opt._resolve_decision_var(key).comm_cost > 0
        ]
        assert producers
        assert all(node.meta["val"].dtype == reduce_dtype for node in producers)

        autop.apply_placement(placement)
        reductions = [
            node
            for target in (
                torch.ops._c10d_functional.all_reduce,
                torch.ops._c10d_functional.all_reduce.default,
                torch.ops._c10d_functional.reduce_scatter_tensor,
                torch.ops._c10d_functional.reduce_scatter_tensor.default,
            )
            for node in autop.parallel_gm.graph.find_nodes(
                op="call_function", target=target
            )
        ]
        assert reductions
        assert all(
            node.args[0].meta["val"].dtype == reduce_dtype for node in reductions
        )


@apply_cuda_patches
def test_grad_reduce_dtype_rejects_unavailable_dtype(device_mesh_1d):
    with torch.device("meta"):
        model = SimpleLinear(64)

    def inputs():
        return torch.randn(512, 64, device="cuda")

    with AutoParallel(model, inputs, device_mesh_1d, repeated_subgraphs=False) as autop:
        with pytest.raises(ValueError, match="linear.weight.*torch.bfloat16"):
            ShardingOptimizer(
                autop.gm, device_mesh_1d, grad_reduce_dtype=torch.bfloat16
            )


@apply_cuda_patches
@pytest.mark.parametrize("tie_weights", [False, True])
def test_backward_dtype_casts_follow_gradient_producers(device_mesh_1d, tie_weights):
    with torch.device("meta"):
        model = StackedLinear(64, n_layers=4)
        if tie_weights:
            model.layers[-1].weight = model.layers[0].weight

    def inputs():
        return torch.randn(512, 64, device="cuda")

    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    with AutoParallel(
        model, inputs, device_mesh_1d, policy, repeated_subgraphs=False
    ) as autop:
        graph = autop.gm.graph
        order = {node: index for index, node in enumerate(graph.nodes)}
        casts = [
            node
            for target in (
                torch.ops.autoparallel.dtype_cast.default,
                torch.ops.prims.convert_element_type.default,
            )
            for node in graph.find_nodes(op="call_function", target=target)
            if node.meta.get("partitioner_tag") == "is_backward"
            and node.meta["val"].dtype == torch.float32
            and node.args[0].meta["val"].dtype != torch.float32
        ]
        assert len(casts) >= len(list(model.parameters()))
        for node in casts:
            assert node.meta["partitioner_tag"] == "is_backward"
            assert 1 <= order[node] - order[node.args[0]] <= 2


@apply_cuda_patches
def test_parameter_casts_preserve_distinct_call_sites(device_mesh_1d):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(512, 512, bias=False)
            self.last = nn.Linear(512, 512, bias=False)

        def forward(self, x):
            hidden = self.first(x)
            return self.last(hidden)

    with torch.device("meta"):
        model = Model()

    def inputs():
        return torch.randn(512, 512, device="cuda")

    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    with AutoParallel(model, inputs, device_mesh_1d, policy) as autop:
        for desc, (param, _) in get_param_and_grad_nodes(autop.gm.graph).items():
            placement = Shard(0) if desc.target == "first.weight" else Shard(1)
            autop.sharding_optimizer.add_node_constraint(param, (placement,))
        autop.optimize_placement()


@apply_cuda_patches
@pytest.mark.parametrize(
    "storage_dtype, param_dtype, reduce_dtype",
    [
        (torch.float32, torch.bfloat16, torch.float32),
        (torch.bfloat16, torch.bfloat16, torch.float32),
        (torch.float32, torch.float32, torch.bfloat16),
        (torch.float32, torch.float32, torch.float32),
    ],
)
def test_reused_parameter_reduction_dtype(
    device_mesh_1d, storage_dtype, param_dtype, reduce_dtype
):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(512, dtype=storage_dtype))

        def forward(self, x, y):
            return x * self.weight + y * self.weight

    with torch.device("meta"):
        model = Model()

    def inputs():
        return (
            torch.randn(512, 512, device="cuda"),
            torch.randn(512, 512, device="cuda"),
        )

    policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    with AutoParallel(
        model, inputs, device_mesh_1d, policy, repeated_subgraphs=False
    ) as autop:
        autop.add_input_constraints([(Shard(0),), (Shard(0),)])
        autop.add_output_constraints([(Shard(0),)])
        for param, _ in get_param_and_grad_nodes(autop.gm.graph).values():
            autop.sharding_optimizer.add_node_constraint(param, (Shard(0),))
        for node in autop.gm.graph.find_nodes(
            op="call_function", target=torch.ops.aten.mul.Tensor
        ):
            if node.meta.get("partitioner_tag") == "is_forward":
                autop.sharding_optimizer.add_node_constraint(node, (Shard(0),))
        autop.apply_placement(autop.optimize_placement())
        reductions = [
            node
            for target in (
                torch.ops._c10d_functional.reduce_scatter_tensor,
                torch.ops._c10d_functional.reduce_scatter_tensor.default,
                torch.ops._c10d_functional.all_reduce,
                torch.ops._c10d_functional.all_reduce.default,
            )
            for node in autop.parallel_gm.graph.find_nodes(
                op="call_function", target=target
            )
        ]
        assert reductions, [
            node.target
            for node in autop.parallel_gm.graph.nodes
            if "c10d" in str(node.target)
        ]
        assert all(
            node.args[0].meta["val"].dtype == reduce_dtype for node in reductions
        )


@apply_cuda_patches
@pytest.mark.parametrize("layer", ["layer_norm", "conv"])
@pytest.mark.parametrize("reduce_dtype", [None, torch.float32])
def test_multi_output_backward_gradient_dtype(device_mesh_1d, layer, reduce_dtype):
    mesh = torch.distributed.device_mesh.DeviceMesh(
        "cuda", list(range(8)), mesh_dim_names=("dp",)
    )
    with torch.device("meta"):
        if layer == "layer_norm":
            model = nn.Sequential(nn.Linear(16, 16), nn.LayerNorm(16))
            shape = (16, 16)
        else:
            model = nn.Conv2d(3, 8, 3, padding=1)
            shape = (16, 3, 8, 8)

    def inputs():
        return torch.randn(shape, device="cuda")

    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=reduce_dtype)
    with AutoParallel(model, inputs, mesh, policy, repeated_subgraphs=False) as autop:
        projections = [
            node
            for node in autop.gm.graph.find_nodes(
                op="call_function", target=operator.getitem
            )
            if node.meta.get("partitioner_tag") == "is_backward"
            and isinstance(node.args[0].meta.get("val"), tuple)
        ]
        assert projections
        autop.optimize_placement()


@apply_cuda_patches
def test_positional_parameter_does_not_constrain_residual_accumulations(device_mesh_1d):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.pos_embed = nn.Parameter(torch.ones(1, 16, 32))
            self.layers = nn.ModuleList([nn.Linear(32, 32) for _ in range(4)])

        def forward(self, x):
            x = x + self.pos_embed
            for layer in self.layers:
                x = x + torch.relu(layer(x))
            return x

    with torch.device("meta"):
        model = Model()

    def inputs():
        return torch.randn(256, 16, 32, device="cuda")

    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    with AutoParallel(
        model, inputs, device_mesh_1d, policy, repeated_subgraphs=False
    ) as autop:
        opt = autop.sharding_optimizer
        accumulations = [
            node
            for node in opt.graph.nodes
            if node.meta.get("is_gradient_acc")
            and node.meta["val"].dtype == torch.bfloat16
        ]
        assert accumulations
        blocked_variables = {
            variable
            for name, constraint in opt.prob.constraints.items()
            if name.startswith("grad_reduce_dtype")
            for variable in constraint
        }
        assert blocked_variables
        residual_variables = {
            opt._get_pulp_variable((opt.node_map[node], argi, out_idx, inp_idx))
            for node in accumulations
            for argi, out_idx, inp_idx in opt.walk_over_options(node)
        }
        assert blocked_variables.isdisjoint(residual_variables)


def test_grad_reduce_dtype_accepts_serialized_tensor_metadata(monkeypatch):
    graph = torch.fx.Graph()
    param = graph.placeholder("param")
    cast = graph.call_function(torch.ops.autoparallel.dtype_cast.default, (param,))
    graph.output(cast)
    param.meta["tensor_meta"] = torch.empty(1, dtype=torch.float32).to("meta")
    cast.meta["tensor_meta"] = torch.empty(1, dtype=torch.bfloat16).to("meta")

    optimizer = ShardingOptimizer.__new__(ShardingOptimizer)
    optimizer.graph = graph
    optimizer.nodes = [cast]
    optimizer.node_map = {cast: 0}
    optimizer.decision_vars = {}
    optimizer.force_grad_reduce_in_higher_precision = False

    monkeypatch.setattr(
        "autoparallel.optimize_sharding.get_param_and_grad_nodes",
        lambda _graph: {"param": (param, None)},
    )

    optimizer.add_grad_reduce_dtype_constraints()


def _run_autop(mesh, model_fn, input_fn, mp_policy, repeated_subgraphs=False):
    """Run AutoParallel and return the solution and optimizer."""
    with torch.device("meta"):
        model = model_fn()

    with AutoParallel(
        model, input_fn, mesh, mp_policy, repeated_subgraphs=repeated_subgraphs
    ) as autop:
        autop.add_input_constraints([(Shard(0),) * mesh.ndim])
        autop.add_output_constraints([(Shard(0),) * mesh.ndim])
        sharding_placement = autop.optimize_placement(verbose=False)

    return sharding_placement, autop


def _assert_no_pre_cast_redistribution(
    sharding_placement, autop, require_linked_validation=False
):
    """Assert that no chosen pre-cast decision var has comm_cost > 0.

    This directly validates the solver constraint: no redistribution
    should occur on unary-chain edges before the dtype_cast node.
    """
    opt = autop.sharding_optimizer
    validated_pre_cast_keys = 0
    validated_linked_keys = 0

    casts = {
        opt.node_map[node]: node
        for target in (
            torch.ops.autoparallel.dtype_cast.default,
            torch.ops.prims.convert_element_type.default,
        )
        for node in opt.graph.find_nodes(op="call_function", target=target)
        if node.meta.get("partitioner_tag") == "is_backward"
        and node.meta["val"].dtype == torch.float32
        and node.args[0].meta["val"].dtype != torch.float32
    }
    for key in opt.selected_keys:
        node_idx, argi, out_idx, inp_idx = key
        if node_idx not in casts or argi != 0:
            continue
        dv = opt._resolve_decision_var(key)
        validated_pre_cast_keys += 1
        if key in opt.cluster_links:
            validated_linked_keys += 1
        assert dv.comm_cost == 0, (
            f"Gradient redistribution before FP32 conversion at {casts[node_idx]}: "
            f"{dv.comm_cost}"
        )
        assert any(p.is_partial() for p in dv.output_spec.placements)

    assert validated_pre_cast_keys > 0, "Expected to validate at least one pre-cast key"
    if require_linked_validation:
        assert (
            validated_linked_keys > 0
        ), "Expected to validate at least one cluster-linked pre-cast key"


@apply_cuda_patches
def test_grad_reduce_dtype_f32_reduces_after_cast(device_mesh_1d):
    """With reduce_dtype=f32, gradient reductions should happen after dtype_cast."""
    dim = 1024
    mesh = device_mesh_1d

    def model_fn():
        return SimpleLinear(dim)

    def input_fn():
        return torch.randn(2048 * mesh.shape[0], dim, device="cuda")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    sharding_placement, autop = _run_autop(mesh, model_fn, input_fn, mp_policy)
    _assert_no_pre_cast_redistribution(sharding_placement, autop)


@apply_cuda_patches
def test_grad_reduce_dtype_f32_with_repeated_subgraphs(device_mesh_1d):
    """Same as above but with repeated_subgraphs=True (graph clustering).

    Verifies the constraint works correctly when cluster-linked nodes
    copy strategies from representative nodes.
    """
    dim = 1024
    mesh = device_mesh_1d

    def model_fn():
        return StackedLinear(dim, n_layers=4)

    def input_fn():
        return torch.randn(2048 * mesh.shape[0], dim, device="cuda")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    sharding_placement, autop = _run_autop(
        mesh, model_fn, input_fn, mp_policy, repeated_subgraphs=True
    )
    _assert_no_pre_cast_redistribution(
        sharding_placement, autop, require_linked_validation=True
    )


@apply_cuda_patches
def test_grad_reduce_dtype_bf16_allows_early_reduction(device_mesh_1d):
    """With reduce_dtype=bf16 (smaller than param_dtype=f32), the constraint
    should NOT fire, and the optimizer is free to reduce before the cast.
    """
    dim = 1024
    mesh = device_mesh_1d

    def model_fn():
        return SimpleLinear(dim)

    def input_fn():
        return torch.randn(2048 * mesh.shape[0], dim, device="cuda")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32, reduce_dtype=torch.bfloat16
    )
    sharding_placement, _ = _run_autop(mesh, model_fn, input_fn, mp_policy)
    assert sharding_placement is not None, "Optimizer should find a feasible solution"


@apply_cuda_patches
def test_grad_reduce_dtype_same_dtype_no_constraint(device_mesh_1d):
    """With reduce_dtype == param_dtype, no special constraint should fire."""
    dim = 1024
    mesh = device_mesh_1d

    def model_fn():
        return SimpleLinear(dim)

    def input_fn():
        return torch.randn(2048 * mesh.shape[0], dim, device="cuda")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32, reduce_dtype=torch.float32
    )
    sharding_placement, _ = _run_autop(mesh, model_fn, input_fn, mp_policy)
    assert sharding_placement is not None, "Optimizer should find a feasible solution"


# ---- Forward dtype_cast constraint tests ----


def _assert_no_fwd_pre_cast_redistribution(
    sharding_placement, autop, require_linked_validation=False
):
    """Assert that no forward dtype_cast has redistribution on its input edge.

    Validates that every forward dtype_cast node matches its producer's
    placement, so allgather happens on the cast output (smaller dtype)
    rather than on the raw parameter (larger storage dtype).
    """
    from torch._functorch._aot_autograd.fx_utils import get_param_and_grad_nodes

    opt = autop.sharding_optimizer
    validated_keys = 0
    validated_linked_keys = 0

    for param, _grad in get_param_and_grad_nodes(opt.graph).values():
        # Walk forward from param through single-user unary nodes
        n = param
        while True:
            if n.target == torch.ops.autoparallel.dtype_cast.default:
                break
            users = list(n.users.keys())
            if len(users) != 1:
                break
            child = users[0]
            if len(child.all_input_nodes) != 1:
                break
            n = child

        if n.target != torch.ops.autoparallel.dtype_cast.default:
            continue

        # Only check downcasts (storage > param_dtype)
        storage_dtype = param.meta["val"].dtype
        cast_dtype = n.meta["val"].dtype
        if cast_dtype.itemsize >= storage_dtype.itemsize:
            continue

        # Collect pre-cast node indices
        fwd_pre_cast_node_idxs = set()
        node = n
        while node != param:
            if node in opt.node_map:
                fwd_pre_cast_node_idxs.add(opt.node_map[node])
            node = node.all_input_nodes[0]

        for key in opt.selected_keys:
            node_idx, argi, out_idx, inp_idx = key
            if node_idx not in fwd_pre_cast_node_idxs:
                continue
            dv = opt._resolve_decision_var(key)
            validated_keys += 1
            if key in opt.cluster_links:
                validated_linked_keys += 1
            assert dv.comm_cost == 0, (
                f"Forward pre-cast node {opt.nodes[node_idx].name} has chosen "
                f"decision var with comm_cost={dv.comm_cost} > 0. "
                f"Redistribution should not happen before the dtype_cast."
            )

    assert validated_keys > 0, "Expected to validate at least one forward pre-cast key"
    if require_linked_validation:
        assert (
            validated_linked_keys > 0
        ), "Expected to validate at least one cluster-linked forward pre-cast key"


@apply_cuda_patches
def test_fwd_allgather_in_param_dtype(device_mesh_1d):
    """With param_dtype=bf16 and f32 storage, forward allgather should happen
    after the dtype_cast (in bf16), not before it (in f32)."""
    dim = 1024
    mesh = device_mesh_1d

    def model_fn():
        return SimpleLinear(dim)

    def input_fn():
        return torch.randn(2048 * mesh.shape[0], dim, device="cuda")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    sharding_placement, autop = _run_autop(mesh, model_fn, input_fn, mp_policy)
    _assert_no_fwd_pre_cast_redistribution(sharding_placement, autop)


@apply_cuda_patches
def test_fwd_allgather_in_param_dtype_reduce_bf16(device_mesh_1d):
    """With param_dtype=bf16, reduce_dtype=bf16, and f32 storage, the forward
    constraint should still fire (it depends on storage > param_dtype, not on
    reduce_dtype)."""
    dim = 1024
    mesh = device_mesh_1d

    def model_fn():
        return SimpleLinear(dim)

    def input_fn():
        return torch.randn(2048 * mesh.shape[0], dim, device="cuda")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
    )
    sharding_placement, autop = _run_autop(mesh, model_fn, input_fn, mp_policy)
    _assert_no_fwd_pre_cast_redistribution(sharding_placement, autop)


@apply_cuda_patches
def test_fwd_allgather_with_repeated_subgraphs(device_mesh_1d):
    """Forward constraint with repeated_subgraphs=True (cluster-linked nodes)."""
    dim = 1024
    mesh = device_mesh_1d

    def model_fn():
        return StackedLinear(dim, n_layers=4)

    def input_fn():
        return torch.randn(2048 * mesh.shape[0], dim, device="cuda")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    sharding_placement, autop = _run_autop(
        mesh, model_fn, input_fn, mp_policy, repeated_subgraphs=True
    )
    _assert_no_fwd_pre_cast_redistribution(
        sharding_placement, autop, require_linked_validation=True
    )


@apply_cuda_patches
def test_fwd_no_constraint_when_upcasting(device_mesh_1d):
    """With param_dtype=f32 and bf16 storage (upcast), the forward constraint
    should NOT fire since the storage dtype is already smaller."""
    dim = 1024
    mesh = device_mesh_1d

    def model_fn():
        m = SimpleLinear(dim)
        # Force bf16 storage so dtype_cast goes bf16 -> f32
        m.linear.weight = nn.Parameter(m.linear.weight.to(torch.bfloat16))
        return m

    def input_fn():
        return torch.randn(2048 * mesh.shape[0], dim, device="cuda")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32, reduce_dtype=torch.bfloat16
    )
    sharding_placement, autop = _run_autop(mesh, model_fn, input_fn, mp_policy)

    # Verify no fwd_param_dtype constraints were added to the ILP
    opt = autop.sharding_optimizer
    fwd_constraint_names = [
        name for name in opt.prob.constraints if name.startswith("fwd_param_dtype")
    ]
    assert len(fwd_constraint_names) == 0, (
        f"Forward dtype constraint should not fire for upcasting, "
        f"but found {len(fwd_constraint_names)} constraints: {fwd_constraint_names}"
    )
