# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import tempfile

import pytest
import torch
from conftest import apply_cuda_patches
from torch import nn
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel.api import AutoParallel
from autoparallel.collectives import flex_local_map, get_flex_local_map_alternatives
from autoparallel.serialization import (
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


def _clone_body(x):
    return (x.clone(),)


def _where_identity_body(x):
    return (torch.where(x == x, x, x).clone(),)


class _FlexModel(nn.Module):
    def __init__(self, mesh):
        super().__init__()
        placements = ((Replicate(),),)
        self.pointwise = flex_local_map(
            _clone_body,
            alternatives=[
                {
                    "name": "clone",
                    "in_placements": placements,
                    "out_placements": placements,
                    "cost_hint": 100.0,
                },
                {
                    "name": "where_identity",
                    "fn": _where_identity_body,
                    "in_placements": placements,
                    "out_placements": placements,
                    "cost_hint": 0.0,
                },
            ],
            device_mesh=mesh,
            redistribute_inputs=True,
        )

    def forward(self, x):
        return self.pointwise(x)[0]


def _setup_autop(model, dim, device_mesh):
    batch_size = 2 * device_mesh.shape[0]

    def input_fn():
        return torch.rand(batch_size, dim, device="cuda")

    return AutoParallel(model, input_fn, device_mesh, repeated_subgraphs=True)


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


def test_patch_op_overload_pickle_cleans_up():
    had_before = hasattr(torch._ops.OpOverload, "__reduce__")
    with _patch_op_overload_pickle():
        assert hasattr(torch._ops.OpOverload, "__reduce__")
    assert hasattr(torch._ops.OpOverload, "__reduce__") == had_before


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
        opt.get_solution()

    with tempfile.NamedTemporaryFile(suffix=".ap") as f:
        opt.save(f.name)
        loaded = type(opt).load(f.name)

    # Loaded optimizer should have the same graph structure
    assert len(loaded.nodes) == len(opt.nodes)
    assert len(loaded.decision_vars) == len(opt.decision_vars)

    # Should have a mesh placeholder
    assert isinstance(loaded.mesh, _MeshPlaceholder)
    assert loaded.mesh.shape == tuple(opt.mesh.shape)

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

    assert data["version"] == 2
    assert "mesh_shape" in data
    assert "placements" in data
    assert isinstance(data["placements"], dict)
    assert len(data["placements"]) > 0


@apply_cuda_patches
def test_flex_placements_roundtrip_and_apply(device_mesh_1d):
    shape = (512, 64)

    def input_fn():
        return torch.randn(*shape, device="cuda", requires_grad=True)

    with torch.device("meta"):
        model = _FlexModel(device_mesh_1d)

    with AutoParallel(model, input_fn, device_mesh_1d) as autop:
        replicated = (Replicate(),)
        autop.add_input_constraints([replicated])
        autop.add_output_constraints([replicated])
        solution = autop.optimize_placement()
        flex_specs = [
            spec
            for node, spec in solution.items()
            if get_flex_local_map_alternatives(node.meta.get("local_map_kwargs", {}))
            is not None
        ]
        assert flex_specs
        assert all(spec.flex_local_map_alternative_index == 1 for spec in flex_specs)

        opt = autop.sharding_optimizer
        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            opt.save_placements(f.name)
            with open(f.name) as rf:
                data = json.load(rf)

    flex_entries = [
        entry
        for entry in data["placements"].values()
        if "flex_local_map_alternative_index" in entry
    ]
    assert flex_entries
    assert all(entry["flex_local_map_alternative_index"] == 1 for entry in flex_entries)
    assert all(
        entry["flex_local_map_alternative_name"] == "where_identity"
        for entry in flex_entries
    )

    flex_node_name = next(
        name
        for name, entry in data["placements"].items()
        if "flex_local_map_alternative_index" in entry
    )
    invalid_cases = (
        ("flex_local_map_alternative_index", None, "no serialized alternative index"),
        ("flex_local_map_alternative_index", 99, "Invalid flex local_map"),
        ("flex_local_map_alternative_name", "renamed", "changed name"),
        ("output", "invalid placement", "does not match the saved placement"),
    )
    for key, value, error in invalid_cases:
        invalid = copy.deepcopy(data)
        entry = invalid["placements"][flex_node_name]
        if value is None:
            del entry[key]
        else:
            entry[key] = value
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w") as f:
            json.dump(invalid, f)
            f.flush()
            with pytest.raises(RuntimeError, match=error):
                opt.load_placements(f.name)

    with torch.device("meta"):
        fresh_model = _FlexModel(device_mesh_1d)
    with AutoParallel(fresh_model, input_fn, device_mesh_1d) as fresh_autop:
        replicated = (Replicate(),)
        fresh_autop.add_input_constraints([replicated])
        fresh_autop.add_output_constraints([replicated])
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w") as f:
            json.dump(data, f)
            f.flush()
            loaded = fresh_autop.sharding_optimizer.load_placements(f.name)
        loaded_flex_specs = [
            spec
            for node, spec in loaded.items()
            if get_flex_local_map_alternatives(node.meta.get("local_map_kwargs", {}))
            is not None
        ]
        assert loaded_flex_specs
        assert all(
            spec.flex_local_map_alternative_index == 1
            and spec.flex_local_map_alternative_name == "where_identity"
            for spec in loaded_flex_specs
        )
        parallel_model = fresh_autop.apply_placement(loaded)

    actual_input = torch.randn(*shape, device="cuda", requires_grad=True)
    expected_input = actual_input.detach().clone().requires_grad_(True)
    actual = parallel_model(actual_input)
    expected = _where_identity_body(expected_input)[0]
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_input.grad, expected_input.grad)


@apply_cuda_patches
def test_load_placements_rejects_v1(device_mesh_1d):
    dim = 64
    with torch.device("meta"):
        model = _SimpleModel(dim)

    autop = _setup_autop(model, dim, device_mesh_1d)
    with autop:
        autop.add_input_constraints([(Shard(0),)])
        autop.add_output_constraints([(Shard(0),)])
        opt = autop.sharding_optimizer
        opt.get_solution()
        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            opt.save_placements(f.name)
            with open(f.name) as rf:
                data = json.load(rf)
            data["version"] = 1
            with open(f.name, "w") as wf:
                json.dump(data, wf)
            with pytest.raises(
                RuntimeError, match="Unsupported placements file version"
            ):
                opt.load_placements(f.name)


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

    solution = loaded.resolve()
    assert len(solution) > 0
