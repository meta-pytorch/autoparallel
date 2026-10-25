# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import tempfile
from unittest.mock import patch

import torch
from conftest import apply_cuda_patches
from torch import nn
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel.api import AutoParallel
from autoparallel.cost_models.compute_estimation import reset_compute_cost_cache
from autoparallel.serialization import (
    _decode_replay_graph,
    _encode_replay_graph,
    _MeshPlaceholder,
    _patch_op_overload_pickle,
    _resolve_target,
)


class _SimpleModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.a = nn.Linear(dim, dim, bias=False)
        self.b = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.b(self.a(x))


class _RepeatedLayerModel(nn.Module):
    def __init__(self, dim, n_layers):
        super().__init__()
        self.embed = nn.Linear(dim, dim, bias=False)
        self.layers = nn.ModuleList(
            [nn.Linear(dim, dim, bias=False) for _ in range(n_layers)]
        )
        self.head = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def _setup_autop(model, dim, device_mesh, **kwargs):
    batch_size = 2 * device_mesh.shape[0]

    def input_fn():
        return torch.rand(batch_size, dim, device="cuda")

    return AutoParallel(
        model,
        input_fn,
        device_mesh,
        repeated_subgraphs=True,
        **kwargs,
    )


# ---- _MeshPlaceholder tests ----


def test_mesh_placeholder_attributes():
    mp = _MeshPlaceholder((32, 8), ("dp", "tp"))
    assert mp.shape == (32, 8)
    assert mp.mesh_dim_names == ("dp", "tp")
    assert mp.ndim == 2


def test_mesh_placeholder_none_dim_names():
    mp = _MeshPlaceholder((64,), None)
    assert mp.mesh_dim_names is None
    assert mp.ndim == 1


# ---- _resolve_target tests ----


def test_resolve_target_getitem():
    import operator

    result = _resolve_target("<built-in function getitem>")
    assert result is operator.getitem


def test_resolve_target_aten_op():
    result = _resolve_target("aten.mm.default")
    assert result is torch.ops.aten.mm.default


def test_resolve_target_unknown_raises():
    import pytest

    with pytest.raises(RuntimeError, match="Cannot resolve"):
        _resolve_target("nonexistent.op.name")


# ---- _patch_op_overload_pickle tests ----


def test_patch_op_overload_pickle_roundtrip():
    import pickle

    op = torch.ops.aten.mm.default
    with _patch_op_overload_pickle():
        data = pickle.dumps(op)
        restored = pickle.loads(data)
    assert restored is op


def test_patch_higher_order_operator_pickle_roundtrip():
    import pickle

    from torch._higher_order_ops.local_map import local_map_hop

    with _patch_op_overload_pickle():
        data = pickle.dumps(local_map_hop)
        restored = pickle.loads(data)
    assert restored is local_map_hop


def test_patch_op_overload_pickle_cleans_up():
    had_before = hasattr(torch._ops.OpOverload, "__reduce__")
    had_hop_before = hasattr(torch._ops.HigherOrderOperator, "__reduce__")
    with _patch_op_overload_pickle():
        assert hasattr(torch._ops.OpOverload, "__reduce__")
        assert hasattr(torch._ops.HigherOrderOperator, "__reduce__")
    assert hasattr(torch._ops.OpOverload, "__reduce__") == had_before
    assert hasattr(torch._ops.HigherOrderOperator, "__reduce__") == had_hop_before


def test_replay_graph_pickler_preserves_deep_graph():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    for _ in range(1500):
        value = graph.call_function(torch.ops.aten.relu.default, (value,))
    graph.output(value)
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

    encoded = _encode_replay_graph(graph_module)
    decoded = _decode_replay_graph(
        encoded,
        FakeTensorMode(shape_env=ShapeEnv(), static_shapes=False),
    ).graph

    original_nodes = list(graph.nodes)
    decoded_nodes = list(decoded.nodes)
    assert len(decoded_nodes) == len(original_nodes)
    for original, restored in zip(original_nodes, decoded_nodes):
        assert restored.name == original.name
        assert restored.op == original.op
        assert restored.target == original.target
        assert torch.fx.map_arg(restored.args, lambda node: node.name) == (
            torch.fx.map_arg(original.args, lambda node: node.name)
        )
        assert torch.fx.map_arg(restored.kwargs, lambda node: node.name) == (
            torch.fx.map_arg(original.kwargs, lambda node: node.name)
        )


# ---- save/load roundtrip tests ----


@apply_cuda_patches
def test_save_load_roundtrip(device_mesh_1d):
    dim = 64
    with torch.device("meta"):
        model = _SimpleModel(dim)

    autop = _setup_autop(model, dim, device_mesh_1d)
    with autop:
        autop.add_input_constraints([(Shard(0),)])
        autop.add_output_constraints([(Shard(0),)])
        opt = autop.sharding_optimizer
        opt.apply_prefetch_discount(0.25)
        opt.skipped_clusters = [{"reason": "test", "regions": [["a"], ["b"]]}]
        opt.get_solution()

    with tempfile.NamedTemporaryFile(suffix=".ap") as f:
        opt.save(f.name)
        loaded = type(opt).load(f.name)

    # Loaded optimizer should have the same graph structure
    assert len(loaded.nodes) == len(opt.nodes)
    assert len(loaded.decision_vars) == len(opt.decision_vars)
    assert set(loaded.prob.constraints) == set(opt.prob.constraints)

    # Should have a mesh placeholder
    assert isinstance(loaded.mesh, _MeshPlaceholder)
    assert loaded.mesh.shape == tuple(opt.mesh.shape)
    assert loaded._prefetch_discount == 0.25
    assert loaded.skipped_clusters == opt.skipped_clusters

    # Should be able to produce JSON
    data = loaded.get_json()
    assert "nodes" in data
    assert len(data["nodes"]) > 0


@apply_cuda_patches
def test_save_load_preserves_solution(device_mesh_1d):
    dim = 64
    with torch.device("meta"):
        model = _SimpleModel(dim)

    autop = _setup_autop(model, dim, device_mesh_1d)
    with autop:
        autop.add_input_constraints([(Shard(0),)])
        autop.add_output_constraints([(Shard(0),)])
        opt = autop.sharding_optimizer
        opt.get_solution()

    with tempfile.NamedTemporaryFile(suffix=".ap") as f:
        opt.save(f.name)
        loaded = type(opt).load(f.name)

    assert hasattr(loaded, "selected_keys")
    assert len(loaded.selected_keys) > 0


@apply_cuda_patches
def test_save_load_with_clusters(device_mesh_1d):
    dim = 64
    with torch.device("meta"):
        model = _RepeatedLayerModel(dim, n_layers=3)

    autop = _setup_autop(model, dim, device_mesh_1d)
    with autop:
        autop.add_input_constraints([(Shard(0),)])
        autop.add_output_constraints([(Shard(0),)])
        opt = autop.sharding_optimizer
        opt.get_solution()

    with tempfile.NamedTemporaryFile(suffix=".ap") as f:
        opt.save(f.name)
        loaded = type(opt).load(f.name)

    assert len(loaded.cluster_links) == len(opt.cluster_links)
    assert len(loaded.nodes) == len(opt.nodes)


@apply_cuda_patches
def test_save_load_replay_regenerates_optimizer(device_mesh_1d):
    dim = 64
    with torch.device("meta"):
        model = _RepeatedLayerModel(dim, n_layers=3)

    autop = _setup_autop(model, dim, device_mesh_1d, cost_model=None)
    with autop:
        autop.add_input_constraints([(Shard(0),)])
        autop.add_output_constraints([(Shard(0),)])
        opt = autop.sharding_optimizer
        opt.apply_prefetch_discount(0.25)
        opt.add_parameter_memory_constraint(0.0, 1.0)
        opt.get_solution()
        opt.remove_constraints(["memory_constraint_high", "memory_constraint_low"])
        expected_solution = opt.resolve()
        expected_summary = opt.get_json()["summary"]

    with tempfile.NamedTemporaryFile(suffix=".pt.gz") as f:
        opt.save_replay(f.name)
        with gzip.open(f.name, "rb") as replay_file:
            payload = torch.load(replay_file, weights_only=False)
        with (
            patch(
                "torch.cuda.get_device_name",
                side_effect=RuntimeError("No CUDA GPUs are available"),
            ),
            patch(
                "torch.cuda.device_count",
                side_effect=RuntimeError("No CUDA GPUs are available"),
            ),
        ):
            loaded = type(opt).load_replay(f.name)

    assert set(payload) == {
        "version",
        "torch_git_version",
        "graph_module",
        "mesh",
        "cost_model_device",
        "optimizer",
        "nccl_topology_config",
    }
    assert payload["version"] == 1
    assert payload["torch_git_version"] == torch.version.git_version
    assert isinstance(payload["graph_module"], bytes)
    assert payload["cost_model_device"] == {"name": "H100", "count": 8}
    assert payload["nccl_topology_config"] is None
    assert len(loaded.nodes) == len(opt.nodes)
    assert len(loaded.decision_vars) == len(opt.decision_vars)
    assert len(loaded.cluster_links) == len(opt.cluster_links)
    assert len(loaded.prob.constraints) == len(opt.prob.constraints)
    assert loaded._constraint_log == opt._constraint_log
    assert loaded._prefetch_discount == 0.25
    assert loaded._memory_constraint is None
    assert loaded.cost_model_device_name == "H100"
    assert loaded.cost_model_device_count == 8

    loaded_solution = loaded.get_solution()
    assert loaded.get_json()["summary"] == expected_summary
    expected_placements = {
        node.name: strategy.output_specs for node, strategy in expected_solution.items()
    }
    loaded_placements = {
        node.name: strategy.output_specs for node, strategy in loaded_solution.items()
    }
    assert loaded_placements == expected_placements


@apply_cuda_patches
def test_load_replay_resets_compute_cost_cache(device_mesh_1d, tmp_path):
    dim = 1024
    with torch.device("meta"):
        model = _SimpleModel(dim)

    autop = _setup_autop(model, dim, device_mesh_1d, cost_model=None)
    with autop:
        opt = autop.sharding_optimizer
        opt.get_solution()

    h100_path = tmp_path / "h100.pt.gz"
    a100_path = tmp_path / "a100.pt.gz"
    opt.save_replay(h100_path)
    with gzip.open(h100_path, "rb") as replay_file:
        payload = torch.load(replay_file, weights_only=False)
    payload["mesh"]["shape"] = [8]
    with gzip.open(h100_path, "wb") as replay_file:
        torch.save(payload, replay_file)
    payload["cost_model_device"]["name"] = "A100"
    with gzip.open(a100_path, "wb") as replay_file:
        torch.save(payload, replay_file)

    h100 = type(opt).load_replay(h100_path)
    h100.get_solution()
    a100 = type(opt).load_replay(a100_path)
    a100.get_solution()

    assert h100.cost_model_device_name == "H100"
    assert a100.cost_model_device_name == "A100"
    assert (
        h100.get_json()["summary"]["compute"] != a100.get_json()["summary"]["compute"]
    )


@apply_cuda_patches
def test_load_replay_resets_placement_options_cache(device_mesh_2d, tmp_path):
    dim = 64
    with torch.device("meta"):
        model = _SimpleModel(dim)

    autop = _setup_autop(model, dim, device_mesh_2d, cost_model=None)
    with autop:
        autop.add_input_constraints([(Shard(0), Replicate())])
        autop.add_output_constraints([(Shard(0), Replicate())])
        opt = autop.sharding_optimizer
        opt.get_solution()

    mesh_32x8_path = tmp_path / "mesh-32x8.pt.gz"
    mesh_16x16_path = tmp_path / "mesh-16x16.pt.gz"
    opt.save_replay(mesh_32x8_path)
    with gzip.open(mesh_32x8_path, "rb") as replay_file:
        payload = torch.load(replay_file, weights_only=False)
    payload["mesh"]["shape"] = [16, 16]
    with gzip.open(mesh_16x16_path, "wb") as replay_file:
        torch.save(payload, replay_file)

    mesh_32x8 = type(opt).load_replay(mesh_32x8_path)
    mesh_32x8.get_solution()
    mesh_16x16 = type(opt).load_replay(mesh_16x16_path)
    solution = mesh_16x16.get_solution()

    assert mesh_32x8.mesh.shape == (32, 8)
    assert mesh_16x16.mesh.shape == (16, 16)
    for strategy in solution.values():
        output_specs = strategy.output_specs
        if not isinstance(output_specs, tuple):
            output_specs = (output_specs,)
        for spec in output_specs:
            if spec is not None:
                assert spec.mesh.shape == (16, 16)


@apply_cuda_patches
def test_replay_explain_placement_uses_saved_device(device_mesh_1d, tmp_path):
    dim = 64
    with torch.device("meta"):
        model = _SimpleModel(dim)

    autop = _setup_autop(model, dim, device_mesh_1d, cost_model=None)
    with autop:
        opt = autop.sharding_optimizer
        opt.get_solution()

    replay_path = tmp_path / "optimizer-replay.pt.gz"
    opt.save_replay(replay_path)
    loaded = type(opt).load_replay(replay_path)
    loaded.get_solution()
    reset_compute_cost_cache()

    with patch(
        "torch.cuda.get_device_name",
        side_effect=RuntimeError("No CUDA GPUs are available"),
    ):
        result = loaded.explain_placement((Replicate(),))

    assert "explain_placement: target=R" in result


@apply_cuda_patches
def test_optimizer_construction_resets_comms_cost_cache(device_mesh_1d):
    dim = 64
    with torch.device("meta"):
        model = _SimpleModel(dim)

    autop = _setup_autop(model, dim, device_mesh_1d, cost_model=None)
    with patch("autoparallel.optimize_sharding.reset_comms_cost_cache") as reset_cache:
        with autop:
            pass

    reset_cache.assert_called_once_with()


@apply_cuda_patches
def test_save_replay_escapes_active_fake_tensor_mode(device_mesh_1d):
    dim = 64
    with torch.device("meta"):
        model = _SimpleModel(dim)

    autop = _setup_autop(model, dim, device_mesh_1d)
    with tempfile.NamedTemporaryFile(suffix=".pt.gz") as replay_file:
        with autop:
            autop.add_input_constraints([(Shard(0),)])
            autop.add_output_constraints([(Shard(0),)])
            autop.optimize_placement()
            autop.sharding_optimizer.save_replay(replay_file.name)

        loaded = type(autop.sharding_optimizer).load_replay(replay_file.name)

    assert len(loaded.nodes) == len(autop.sharding_optimizer.nodes)


# ---- save_placements/load_placements roundtrip tests ----


@apply_cuda_patches
def test_save_load_placements_roundtrip(device_mesh_1d):
    dim = 64
    with torch.device("meta"):
        model = _SimpleModel(dim)

    autop = _setup_autop(model, dim, device_mesh_1d)
    with autop:
        autop.add_input_constraints([(Shard(0),)])
        autop.add_output_constraints([(Shard(0),)])
        opt = autop.sharding_optimizer
        opt.get_solution()

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w") as f:
        opt.save_placements(f.name)
        solution = opt.load_placements(f.name)

    assert len(solution) > 0
    for node, strategy in solution.items():
        assert hasattr(strategy, "output_specs")


@apply_cuda_patches
def test_save_placements_is_valid_json(device_mesh_1d):
    dim = 64
    with torch.device("meta"):
        model = _SimpleModel(dim)

    autop = _setup_autop(model, dim, device_mesh_1d)
    with autop:
        autop.add_input_constraints([(Shard(0),)])
        autop.add_output_constraints([(Shard(0),)])
        opt = autop.sharding_optimizer
        opt.get_solution()

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w") as f:
        opt.save_placements(f.name)
        with open(f.name) as rf:
            data = json.load(rf)

    assert data["version"] == 1
    assert "mesh_shape" in data
    assert "placements" in data
    assert isinstance(data["placements"], dict)
    assert len(data["placements"]) > 0


@apply_cuda_patches
def test_loaded_optimizer_resolve_without_memory_constraint(device_mesh_1d):
    """A loaded optimizer that never had a memory constraint should be
    able to call resolve() without crashing."""
    dim = 64
    with torch.device("meta"):
        model = _SimpleModel(dim)

    autop = _setup_autop(model, dim, device_mesh_1d)
    with autop:
        autop.add_input_constraints([(Shard(0),)])
        autop.add_output_constraints([(Shard(0),)])
        opt = autop.sharding_optimizer
        opt.get_solution()

    with tempfile.NamedTemporaryFile(suffix=".ap") as f:
        opt.save(f.name)
        loaded = type(opt).load(f.name)

    assert loaded.solver_time_limit_seconds is None
    solution = loaded.resolve()
    assert len(solution) > 0
