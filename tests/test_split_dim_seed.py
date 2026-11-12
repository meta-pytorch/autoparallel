# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math

import pulp
import pytest
import torch
from conftest import apply_cuda_patches
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor._dtensor_spec import TensorMeta
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard

from autoparallel._testing.models.dsv3 import (
    DeepSeekV3Model,
    MoEMeshRoles,
    build_moe_local_map_placements,
    build_moe_mesh,
    make_dsv3_config,
)
from autoparallel.api import AutoParallel
from autoparallel.approximate_sharding import ApproximateShardingSolver
from autoparallel.cost_models.collective_runtime_estimation import (
    estimate_strategy_comms_cost,
)
from autoparallel.cost_models.nccl_cost_model import h100_topo_config
from autoparallel.mesh_search import (
    _build_split_dim_seed,
    _combine_op_spec_seeds,
    _project_local_map_to_dim,
    _split_dim_seed_cache_key,
)
from autoparallel.optimize_sharding import (
    ShardingOptimizer,
    _seed_spec_is_well_formed,
    _strategy_matches_seed,
)
from autoparallel.shardings.propagation_rules import (
    DTensorSpecSeed,
    OpSpecSeed,
    StrategySeed,
)

pytestmark = [
    pytest.mark.filterwarnings("ignore:Constructing LpVariable.*:DeprecationWarning"),
    pytest.mark.filterwarnings(
        "ignore:Using LpProblem.constraints.*:DeprecationWarning"
    ),
]


class TinyMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = torch.nn.Linear(16, 32)
        self.out_proj = torch.nn.Linear(32, 16)

    def forward(self, x):
        return self.out_proj(torch.relu(self.in_proj(x)))


def _input_fn():
    return torch.randn(8, 16, device="cuda", requires_grad=True)


def _tensor_meta(shape, dtype=torch.float32):
    tensor = torch.empty(shape, dtype=dtype, device="meta")
    return TensorMeta(tensor.shape, tensor.stride(), tensor.dtype)


def _spec_seed(placements, tensor_meta):
    return DTensorSpecSeed(tuple(placements), tensor_meta)


def test_combine_split_dim_op_specs_preserves_rmsnorm_seeded_edge():
    input_meta = _tensor_meta((8, 2048, 256))
    output_meta = _tensor_meta((256,))
    sum_per_dim = [
        OpSpecSeed(
            output_specs=_spec_seed((Partial("sum"),), output_meta),
            input_specs=(_spec_seed((Shard(0),), input_meta),),
        )
        for _ in range(3)
    ]
    dtype_per_dim = [
        OpSpecSeed(
            output_specs=_spec_seed((Shard(0),), output_meta),
            input_specs=(_spec_seed((Shard(0),), output_meta),),
        )
        for _ in range(3)
    ]

    sum_seed = _combine_op_spec_seeds(sum_per_dim)
    dtype_seed = _combine_op_spec_seeds(dtype_per_dim)

    assert sum_seed is not None
    assert isinstance(sum_seed.output_specs, DTensorSpecSeed)
    assert sum_seed.output_specs.placements == (Partial("sum"),) * 3
    assert sum_seed.input_specs is not None
    assert sum_seed.input_specs[0].placements == (Shard(0),) * 3
    assert dtype_seed is not None
    assert isinstance(dtype_seed.output_specs, DTensorSpecSeed)
    assert dtype_seed.output_specs.placements == (Shard(0),) * 3
    assert dtype_seed.input_specs is not None
    assert dtype_seed.input_specs[0].placements == (Shard(0),) * 3


def test_combine_split_dim_op_specs_preserves_optional_outputs():
    tensor_meta = _tensor_meta((8, 2048, 256))
    per_dim = [
        OpSpecSeed(
            output_specs=(_spec_seed((Shard(0),), tensor_meta), None),
            input_specs=(_spec_seed((Replicate(),), tensor_meta),),
        )
        for _ in range(3)
    ]

    combined = _combine_op_spec_seeds(per_dim)

    assert combined is not None
    assert isinstance(combined.output_specs, tuple)
    assert combined.output_specs[0].placements == (Shard(0),) * 3
    assert combined.output_specs[1] is None


def test_combine_split_dim_op_specs_rejects_structural_mismatch():
    tensor_meta = _tensor_meta((256,))
    single = OpSpecSeed(
        output_specs=_spec_seed((Shard(0),), tensor_meta),
        input_specs=(_spec_seed((Shard(0),), tensor_meta),),
    )
    multiple = OpSpecSeed(
        output_specs=(_spec_seed((Shard(0),), tensor_meta), None),
        input_specs=(_spec_seed((Shard(0),), tensor_meta),),
    )

    assert _combine_op_spec_seeds([single, single, multiple]) is None


def test_seed_spec_validation_rejects_invalid_target_mesh_specs():
    tensor_meta = _tensor_meta((256,))

    assert not _seed_spec_is_well_formed(
        _spec_seed((Shard(0), Shard(0)), tensor_meta), 3
    )
    assert not _seed_spec_is_well_formed(_spec_seed((Shard(1),) * 3, tensor_meta), 3)
    assert not _seed_spec_is_well_formed(DTensorSpecSeed((Replicate(),) * 3, None), 3)


@apply_cuda_patches
def test_split_dim_preserves_complete_rmsnorm_seeded_edge():
    with unset_fake_temporarily():
        mesh = init_device_mesh(
            "cuda", (2, 2, 2), mesh_dim_names=("outer", "middle", "inner")
        )

    input_value = torch.empty((8, 2048, 256), device="meta")
    output_value = torch.empty((256,), device="meta")
    graph = torch.fx.Graph()
    input_node = graph.placeholder("input")
    sum_node = graph.call_function(torch.ops.aten.sum.dim_IntList, (input_node, [0, 1]))
    dtype_node = graph.call_function(
        torch.ops.autoparallel.dtype_cast.default, (sum_node, torch.float32)
    )
    alias_node = graph.call_function(torch.ops.aten.alias.default, (dtype_node,))
    output_node = graph.output(alias_node)
    input_node.meta["val"] = input_value
    sum_node.meta["val"] = output_value
    dtype_node.meta["val"] = output_value
    alias_node.meta["val"] = output_value
    output_node.meta["val"] = output_value
    gm = torch.fx.GraphModule({}, graph)

    sss = (Shard(0),) * 3
    ppp = (Partial("sum"),) * 3
    input_meta = _tensor_meta(input_value.shape)
    output_meta = _tensor_meta(output_value.shape)
    input_spec = _spec_seed(sss, input_meta)
    output_spec = _spec_seed(sss, output_meta)
    seed = {
        input_node.name: StrategySeed(
            sss, OpSpecSeed(output_specs=input_spec, input_specs=(input_spec,))
        ),
        sum_node.name: StrategySeed(
            ppp,
            OpSpecSeed(
                output_specs=_spec_seed(ppp, output_meta),
                input_specs=(input_spec,),
            ),
        ),
        dtype_node.name: StrategySeed(
            sss,
            OpSpecSeed(output_specs=output_spec, input_specs=(output_spec,)),
        ),
        alias_node.name: StrategySeed(
            sss,
            OpSpecSeed(output_specs=output_spec, input_specs=(output_spec,)),
        ),
    }

    optimizer = ShardingOptimizer(
        gm,
        mesh,
        repeated_subgraphs=False,
        build_pulp=False,
        build_costs=True,
        strategy_seed=seed,
        strategy_radius=2,
    )
    nodes = {node.name: node for node in optimizer.graph.nodes}
    sum_strategies = optimizer.strats[nodes[sum_node.name]].strategies
    dtype_strategies = optimizer.strats[nodes[dtype_node.name]].strategies

    assert all(strategy.output_spec.placements != sss for strategy in sum_strategies)
    sum_seed_index = next(
        index
        for index, strategy in enumerate(sum_strategies)
        if strategy.output_spec.placements == ppp
        and strategy.input_specs[0].placements == sss
    )
    dtype_seed = next(
        strategy
        for strategy in dtype_strategies
        if strategy.output_spec.placements == sss
        and strategy.input_specs[0].placements == sss
    )
    assert len(dtype_seed.redistribute_cost[0]) == len(sum_strategies)
    assert math.isfinite(dtype_seed.redistribute_cost[0][sum_seed_index])


@apply_cuda_patches
def test_real_dsv3_split_dim_search_preserves_all_complete_witnesses():
    mesh, roles = build_moe_mesh(
        dp_replicate=1,
        dp_shard=4,
        cp=1,
        tp=2,
        ep=4,
    )
    config = make_dsv3_config(num_experts=8, max_seq_len=2048)
    with torch.device("meta"):
        model = DeepSeekV3Model(
            config,
            mesh=mesh,
            roles=roles,
            compute_dtype=torch.float32,
        )

    def input_fn():
        return torch.empty(8, 2048, dtype=torch.int64, device="cuda")

    placements = (Shard(0),) * mesh.ndim
    with AutoParallel(
        model,
        input_fn,
        mesh,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
        ),
        dynamic=True,
        solver="approx",
        lazy_costs=True,
        strategy_radius=2,
    ) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)
        autop.add_input_constraints([placements])
        autop.add_output_constraints([placements])
        autop._build_split_dim_optimizer()
        optimizer = autop.sharding_optimizer
        assert optimizer is not None
        approximate = ApproximateShardingSolver(optimizer)
        solution = approximate.get_solution(verbose=False)

    assert solution
    assert math.isfinite(optimizer.profile["approximate"]["objective"])

    missing = []
    for node, strategies in optimizer.strats.items():
        if node.op == "output":
            continue
        node_seed = optimizer.strategy_seed.get(node.name)
        if node_seed is None or node_seed.witness is None:
            continue
        if not any(
            _strategy_matches_seed(strategy, node_seed.witness)
            for strategy in strategies.strategies
        ):
            missing.append(node.name)
    assert not missing

    sss = (Shard(0),) * mesh.ndim
    ppp = (Partial("sum"),) * mesh.ndim
    seeded_chain = None
    for dtype_node in optimizer.graph.nodes:
        if dtype_node.target != torch.ops.autoparallel.dtype_cast.default:
            continue
        input_nodes = optimizer._all_input_nodes(dtype_node)
        if len(input_nodes) != 1:
            continue
        sum_node = input_nodes[0]
        if sum_node.target != torch.ops.aten.sum.dim_IntList:
            continue
        sum_seed = optimizer.strategy_seed.get(sum_node.name)
        dtype_seed = optimizer.strategy_seed.get(dtype_node.name)
        if (
            sum_seed is not None
            and dtype_seed is not None
            and sum_seed.output_placements == ppp
            and dtype_seed.output_placements == sss
        ):
            seeded_chain = sum_node, dtype_node, sum_seed, dtype_seed
            break
    assert seeded_chain is not None

    sum_node, dtype_node, sum_seed, dtype_seed = seeded_chain
    assert sum_seed.witness is not None
    assert dtype_seed.witness is not None
    sum_strategy = next(
        strategy
        for strategy in optimizer.strats[sum_node].strategies
        if _strategy_matches_seed(strategy, sum_seed.witness)
    )
    dtype_strategy = next(
        strategy
        for strategy in optimizer.strats[dtype_node].strategies
        if _strategy_matches_seed(strategy, dtype_seed.witness)
    )
    assert sum_strategy.output_spec.placements == ppp
    assert dtype_strategy.input_specs[0].placements == sss
    assert dtype_strategy.output_spec.placements == sss
    assert math.isfinite(
        estimate_strategy_comms_cost(
            sum_strategy.output_spec, dtype_strategy.input_specs[0]
        )
    )


@pytest.mark.parametrize(
    ("strategy_radius", "message"),
    (
        (-1, "must be non-negative"),
        (3, "smaller than the mesh dimensionality"),
        (1.5, "must be an integer"),
        (True, "must be an integer"),
    ),
)
@apply_cuda_patches
def test_split_dim_public_options_validate(device_mesh_3d, strategy_radius, message):
    with torch.device("meta"):
        model = TinyMLP()

    with pytest.raises(ValueError, match=message):
        AutoParallel(
            model,
            _input_fn,
            device_mesh_3d,
            solver="approx",
            strategy_radius=strategy_radius,
        )


@apply_cuda_patches
def test_split_dim_large_radius_warns():
    with unset_fake_temporarily():
        mesh = init_device_mesh(
            "cuda", (2, 2, 2, 2), mesh_dim_names=("d0", "d1", "d2", "d3")
        )
    with torch.device("meta"):
        model = TinyMLP()

    with pytest.warns(UserWarning, match="greater than 2"):
        AutoParallel(model, _input_fn, mesh, solver="approx", strategy_radius=3)


@pytest.mark.parametrize("strategy_radius", (None, 0))
@apply_cuda_patches
def test_split_dim_can_be_disabled(device_mesh_3d, strategy_radius):
    with torch.device("meta"):
        model = TinyMLP()

    with AutoParallel(
        model,
        _input_fn,
        device_mesh_3d,
        repeated_subgraphs=False,
        solver="approx",
        strategy_radius=strategy_radius,
    ) as autop:
        placement = (Shard(0), Replicate(), Replicate())
        autop.add_input_constraints([placement])
        autop.add_output_constraints([placement])
        solution = autop.optimize_placement(verbose=False)

    assert solution


@apply_cuda_patches
def test_split_dim_radius_ignored_for_non_approx_and_2d(device_mesh_2d, device_mesh_3d):
    with torch.device("meta"):
        model = TinyMLP()

    AutoParallel(model, _input_fn, device_mesh_3d, solver="ilp", strategy_radius=-1)
    AutoParallel(model, _input_fn, device_mesh_2d, solver="approx", strategy_radius=2)


@apply_cuda_patches
def test_auto_parallel_split_dim_search(device_mesh_3d):
    with torch.device("meta"):
        model = TinyMLP()

    placement = (Shard(0), Replicate(), Replicate())
    with AutoParallel(
        model,
        _input_fn,
        device_mesh_3d,
        repeated_subgraphs=False,
        solver="approx",
    ) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)
        autop.add_input_constraints([placement])
        autop.add_output_constraints([placement])
        solution = autop.optimize_placement()

    assert solution
    profile = autop.sharding_optimizer.profile["approximate"]
    assert profile["status"] == "Heuristic"
    assert math.isfinite(profile["objective"])


def test_split_dim_projects_local_map_placements(device_mesh_3d):
    dim_names = device_mesh_3d.mesh_dim_names
    assert dim_names is not None
    roles = MoEMeshRoles(ep_axis_names=(dim_names[1], dim_names[2]), ep_group_name="ep")
    token, weight, count = build_moe_local_map_placements(dim_names, roles)

    graph = torch.fx.Graph()
    node = graph.placeholder("x")
    graph.output(node)
    gm = torch.fx.GraphModule({}, graph)
    local_map_kwargs = {
        "in_placements": (token, weight, None),
        "out_placements": (token, count),
        "device_mesh": device_mesh_3d,
    }
    node.meta["local_map_kwargs"] = local_map_kwargs

    for dim_idx, dim_name in enumerate(dim_names):
        mesh_1d = device_mesh_3d[dim_name]
        with _project_local_map_to_dim(gm, mesh_1d, dim_idx, len(dim_names)):
            projected = node.meta["local_map_kwargs"]
            assert projected["device_mesh"] == mesh_1d
            assert projected["in_placements"] == (
                (Shard(0),),
                ((Replicate(),) if dim_idx == 0 else (Shard(0),)),
                None,
            )
            assert projected["out_placements"] == (
                (Shard(0),),
                (Partial("sum"),),
            )
        assert node.meta["local_map_kwargs"] is local_map_kwargs

    shape = tuple(device_mesh_3d.shape)
    constraints = ((("R",),), (("R",),))
    assert _split_dim_seed_cache_key(
        shape[0], constraints, "roofline", shape, 0, fabric_aware=False
    ) != _split_dim_seed_cache_key(
        shape[1], constraints, "roofline", shape, 1, fabric_aware=False
    )


@apply_cuda_patches
def test_split_dim_seed_hamming_space_solves_with_ilp_and_lp():
    config = h100_topo_config(num_nodes=2, gpus_per_node=4)
    with unset_fake_temporarily():
        mesh = init_device_mesh(
            "cuda",
            (2, 2, 2),
            mesh_dim_names=("dp", "mid", "inner"),
        )

    with torch.device("meta"):
        model = TinyMLP()

    input_placement = (Shard(0), Replicate(), Replicate())
    one_d_cache = {}

    with AutoParallel(
        model,
        _input_fn,
        mesh,
        cost_model=config,
        repeated_subgraphs=False,
    ) as autop:
        seed = _build_split_dim_seed(
            autop.gm,
            tuple(mesh.shape),
            input_constraints=[input_placement],
            output_constraints=[input_placement],
            cost_model=config,
            repeated_subgraphs=False,
            one_d_cache=one_d_cache,
        )

        opt = ShardingOptimizer(
            autop.gm,
            mesh,
            repeated_subgraphs=False,
            strategy_seed=seed,
            strategy_radius=2,
        )
        opt.add_sharded_input_constraint([input_placement])
        opt.add_sharded_output_constraint([input_placement])
        opt.add_parameter_memory_constraint(0.0, 1.0)

        lp_result = opt.solve_lp_relaxation(extract=True)
        assert lp_result["status"] == "Optimal"
        assert math.isfinite(lp_result["objective"])

        solution = opt.get_solution(verbose=False)
        assert solution
        assert pulp.LpStatus[opt.prob.status] == "Optimal"
        assert math.isfinite(pulp.value(opt.prob.objective))
