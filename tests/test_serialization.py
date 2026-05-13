# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import json
import tempfile

import torch
from torch import nn
from torch.distributed.tensor.placement_types import Shard

from autoparallel.api import AutoParallel
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


def test_resolve_target_unknown_falls_back():
    result = _resolve_target("nonexistent.op.name")
    assert result is torch.ops.aten.alias.default


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


# ---- save_solution/load_solution roundtrip tests ----


def test_save_load_solution_roundtrip(device_mesh_1d):
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
        opt.save_solution(f.name)
        solution = opt.load_solution(f.name)

    assert len(solution) > 0
    for node, strategy in solution.items():
        assert hasattr(strategy, "output_specs")


def test_save_solution_is_valid_json(device_mesh_1d):
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
        opt.save_solution(f.name)
        with open(f.name) as rf:
            data = json.load(rf)

    assert data["version"] == 1
    assert "mesh_shape" in data
    assert "placements" in data
    assert isinstance(data["placements"], dict)
    assert len(data["placements"]) > 0
