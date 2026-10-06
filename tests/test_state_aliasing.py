# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pytest
import torch
from conftest import apply_cuda_patches
from torch import nn
from torch._functorch._aot_autograd.descriptors import BufferAOTInput, ParamAOTInput
from torch._functorch._aot_autograd.fx_utils import get_named_buffer_nodes
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel.api import AutoParallel
from autoparallel.apply_sharding import _shard_params_and_buffers
from autoparallel.optimize_sharding import ShardingOptimizer


def _make_alias_optimizer(mesh):
    graph = torch.fx.Graph()
    embedding = graph.placeholder("embedding")
    decoder = graph.placeholder("decoder")
    result = graph.call_function(torch.ops.aten.add.Tensor, (embedding, decoder))
    output = graph.output((result,))

    for node in (embedding, decoder, result):
        node.meta["val"] = torch.empty(512, 2, device="meta")
    embedding.meta["desc"] = ParamAOTInput("embedding.weight")
    decoder.meta["desc"] = ParamAOTInput("decoder.weight")
    output.meta["desc"] = [None]

    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    aliases = {"decoder.weight": "embedding.weight"}
    optimizer = ShardingOptimizer(gm, mesh, persistent_aliases=aliases)
    return optimizer, embedding, decoder


@apply_cuda_patches
def test_aliased_parameters_cannot_select_different_placements(device_mesh_1d):
    optimizer, embedding, decoder = _make_alias_optimizer(device_mesh_1d)
    optimizer.add_node_constraint(embedding, (Shard(0),))
    optimizer.add_node_constraint(decoder, (Replicate(),))

    with pytest.raises(RuntimeError, match="could not find a feasible solution"):
        optimizer._solve()


@apply_cuda_patches
def test_aliased_parameter_memory_is_counted_once(device_mesh_1d):
    optimizer, _, _ = _make_alias_optimizer(device_mesh_1d)
    optimizer.add_parameter_memory_constraint(0.0, 1.0)
    optimizer._apply_memory_constraint()

    (nodes,) = optimizer._get_persistent_alias_groups(ParamAOTInput).values()
    root = nodes[0]
    constraint = optimizer.prob.constraints["memory_constraint_high"]
    assert len(constraint) == len(optimizer.strats[root].strategies)
    assert all(f"n={root.name}" in variable.name for variable in constraint)


@apply_cuda_patches
def test_parameter_memory_constraint_is_weighted_by_bytes(device_mesh_1d):
    graph = torch.fx.Graph()
    large = graph.placeholder("large")
    small = graph.placeholder("small")
    output = graph.output((large, small))
    large.meta["val"] = torch.empty(1024, device="meta")
    small.meta["val"] = torch.empty(256, device="meta")
    large.meta["desc"] = ParamAOTInput("large")
    small.meta["desc"] = ParamAOTInput("small")
    output.meta["desc"] = [None, None]

    optimizer = ShardingOptimizer(
        torch.fx.GraphModule(torch.nn.Module(), graph), device_mesh_1d
    )
    optimizer.add_parameter_memory_constraint(0.0, 1.0)
    optimizer._apply_memory_constraint()
    constraint = optimizer.prob.constraints["memory_constraint_high"]

    coefficients = {}
    for node in (large, small):
        node = optimizer._normalize_node(node)
        replicate_idx = next(
            index
            for index, strategy in enumerate(optimizer.strats[node].strategies)
            if strategy.output_specs.placements == (Replicate(),)
        )
        variable = optimizer._resolve_decision_var(
            (optimizer.node_map[node], 0, replicate_idx, 0)
        ).var
        coefficients[node.name] = constraint[variable]

    assert coefficients["large"] == 4 * coefficients["small"]

    optimizer.get_solution()
    summary = optimizer.get_json()["summary"]["parameter_storage"]
    assert summary["global_bytes"] == (1024 + 256) * 4
    assert summary["tensor_count"] == 2
    assert summary["local_to_global_fraction"] is not None


def test_aliased_state_is_materialized_once():
    graph = torch.fx.Graph()
    parameter = graph.placeholder("parameter")
    buffer = graph.placeholder("buffer")
    output = graph.output((parameter, buffer))
    parameter.meta["desc"] = ParamAOTInput("embedding.weight")
    buffer.meta["desc"] = BufferAOTInput("observer.enabled")
    output.meta["desc"] = [None, None]
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    param_aliases = {"decoder.weight": "embedding.weight"}
    buffer_aliases = {"activation_post_process.enabled": "observer.enabled"}
    physical_placements = {parameter: object(), buffer: object()}

    with patch(
        "autoparallel.apply_sharding.shard_node_given_placements",
        side_effect=[torch.ones(2), torch.zeros(2)],
    ) as shard:
        params, buffers = _shard_params_and_buffers(
            gm,
            physical_placements,
            ["embedding.weight", "decoder.weight"],
            ["observer.enabled", "activation_post_process.enabled"],
            param_aliases,
            buffer_aliases,
        )

    assert shard.call_count == 2
    assert params["embedding.weight"] is params["decoder.weight"]
    assert buffers["observer.enabled"] is buffers["activation_post_process.enabled"]


@apply_cuda_patches
def test_tied_weight_executes_with_summed_gradient(device_mesh_1d):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(16, 8)
            self.decoder = nn.Linear(8, 16, bias=False)
            self.decoder.weight = self.embedding.weight

        def forward(self, tokens):
            return self.decoder(self.embedding(tokens))

        def init_weights(self):
            nn.init.ones_(self.embedding.weight)

    with torch.device("meta"):
        model = Model()

    def input_fn():
        return torch.randint(0, 16, (device_mesh_1d.size(), 2), device="cuda")

    with AutoParallel(
        model, input_fn, device_mesh_1d, repeated_subgraphs=False
    ) as autop:
        autop.add_input_constraints([(Shard(0),)])
        autop.add_output_constraints([(Shard(0),)])
        solution = autop.optimize_placement()
        parallel_model = autop.apply_placement(solution)

    parallel_model.to_empty(device="cuda")
    parallel_model.init_weights()
    assert parallel_model.embedding.weight is parallel_model.decoder.weight

    tokens = torch.tensor([[1, 2]], device="cuda")
    actual = parallel_model(tokens)
    actual.sum().backward()

    weight_1 = torch.ones(16, 8, device="cuda", requires_grad=True)
    weight_2 = torch.ones(16, 8, device="cuda", requires_grad=True)
    expected = nn.functional.linear(nn.functional.embedding(tokens, weight_1), weight_2)
    expected.sum().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        parallel_model.embedding.weight.grad.to_local(),
        weight_1.grad + weight_2.grad,
    )


class AliasedBufferModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("freqs", torch.zeros(512, 8))
        self.rope = nn.Module()
        self.rope.register_buffer("cache", self.freqs)

    def forward(self, x):
        return x + self.freqs + self.rope.cache

    def init_weights(self):
        self.freqs.fill_(1)


@apply_cuda_patches
def test_aliased_buffers_share_placement_and_runtime_object(device_mesh_1d):
    def input_fn():
        return torch.randn(512, 8, device="cuda")

    with torch.device("meta"):
        model = AliasedBufferModel()
    with AutoParallel(
        model, input_fn, device_mesh_1d, repeated_subgraphs=False
    ) as autop:
        buffer_nodes = get_named_buffer_nodes(autop.sharding_optimizer.graph)
        assert set(buffer_nodes) == {"freqs", "rope.cache"}
        autop.sharding_optimizer.add_node_constraint(buffer_nodes["freqs"], (Shard(0),))
        autop.sharding_optimizer.add_node_constraint(
            buffer_nodes["rope.cache"], (Replicate(),)
        )
        with pytest.raises(RuntimeError, match="could not find a feasible solution"):
            autop.optimize_placement()

    with torch.device("meta"):
        model = AliasedBufferModel()
    with AutoParallel(
        model, input_fn, device_mesh_1d, repeated_subgraphs=False
    ) as autop:
        autop.add_input_constraints([(Replicate(),)])
        autop.add_output_constraints([(Replicate(),)])
        solution = autop.optimize_placement()
        parallel_model = autop.apply_placement(solution)

    assert parallel_model.freqs is parallel_model.rope.cache
    parallel_model.to_empty(device="cuda")
    parallel_model.init_weights()
    assert parallel_model.freqs is parallel_model.rope.cache

    x = torch.randn(512, 8, device="cuda")
    torch.testing.assert_close(parallel_model(x), x + 2)
