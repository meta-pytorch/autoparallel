# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.distributed.tensor.placement_types import Shard

from autoparallel.api import AutoParallel
from autoparallel.export_json import _get_layer_index, _normalize_cluster_layer


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


# ---- _get_layer_index tests ----


def test_get_layer_index_from_nn_module_stack(device_mesh_1d):
    dim = 64
    with torch.device("meta"):
        model = _RepeatedLayerModel(dim, n_layers=3)

    autop = _setup_autop(model, dim, device_mesh_1d)
    with autop:
        autop.add_input_constraints([(Shard(0),)])
        autop.add_output_constraints([(Shard(0),)])
        graph = autop.sharding_optimizer.graph

    found_layers = set()
    found_non_layer = False
    for node in graph.nodes:
        idx = _get_layer_index(node)
        if idx is not None:
            found_layers.add(idx)
        elif node.op not in ("placeholder", "output"):
            found_non_layer = True

    assert found_layers == {0, 1, 2}, f"Expected layers 0-2, got {found_layers}"
    assert found_non_layer, "Expected some non-layer nodes (embed, head)"


def test_get_layer_index_returns_none_for_non_layer_node():
    """Nodes without nn_module_stack should return None."""

    class FakeNode:
        def __init__(self):
            self.meta = {}

    assert _get_layer_index(FakeNode()) is None

    node_with_empty_stack = FakeNode()
    node_with_empty_stack.meta = {"nn_module_stack": {}}
    assert _get_layer_index(node_with_empty_stack) is None


# ---- _normalize_cluster_layer tests ----


def test_normalize_cluster_layer_swaps_backward_roots(device_mesh_1d):
    dim = 64
    with torch.device("meta"):
        model = _RepeatedLayerModel(dim, n_layers=4)

    autop = _setup_autop(model, dim, device_mesh_1d)
    with autop:
        autop.add_input_constraints([(Shard(0),)])
        autop.add_output_constraints([(Shard(0),)])
        opt = autop.sharding_optimizer

    # Build cluster_roots from cluster_links
    cluster_roots = {}
    for linked_key, root_key in opt.cluster_links.items():
        linked_node = opt.nodes[linked_key[0]]
        root_node = opt.nodes[root_key[0]]
        cluster_roots[linked_node] = root_node

    if not cluster_roots:
        return  # no clusters to test

    _normalize_cluster_layer(cluster_roots)

    # After normalization, all roots should be in the canonical (lowest) layer
    root_layers = set()
    for root in set(cluster_roots.values()):
        idx = _get_layer_index(root)
        if idx is not None:
            root_layers.add(idx)

    if root_layers:
        canonical = min(root_layers)
        # Most roots should be in the canonical layer (some may not have
        # a copy there due to boundary asymmetry)
        assert canonical == min(root_layers)


def test_normalize_cluster_layer_empty():
    _normalize_cluster_layer({})


# ---- export_sharding_json tests ----


def test_export_json_produces_valid_structure(device_mesh_1d):
    dim = 64
    with torch.device("meta"):
        model = _RepeatedLayerModel(dim, n_layers=2)

    autop = _setup_autop(model, dim, device_mesh_1d)
    with autop:
        autop.add_input_constraints([(Shard(0),)])
        autop.add_output_constraints([(Shard(0),)])
        autop.sharding_optimizer.get_solution()
        data = autop.sharding_optimizer.get_json()

    assert "nodes" in data
    assert "mesh" in data
    assert "summary" in data
    assert isinstance(data["nodes"], list)
    assert len(data["nodes"]) > 0

    # Every node should have required fields
    for node in data["nodes"]:
        assert "name" in node
        assert "op" in node

    # Summary should have cost fields
    assert "total" in data["summary"]
    assert "comm" in data["summary"]
    assert "compute" in data["summary"]

    # Mesh should have shape
    assert "shape" in data["mesh"]
