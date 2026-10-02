# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from conftest import apply_cuda_patches
from torch import nn
from torch.distributed.tensor._op_schema import OpSpec
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel.api import AutoParallel
from autoparallel.apply_sharding import _lower_to_parallel_graph
from autoparallel.shardings.placement_options import get_placement_options
from autoparallel.shardings.propagation_rules import _create_all_options


def _mesh(request, fixture):
    return request.getfixturevalue(fixture)


@pytest.mark.parametrize("fixture", ["device_mesh_1d", "device_mesh_2d"])
def test_unsafe_index_strategy_is_replicated(request, fixture):
    mesh = _mesh(request, fixture)
    data = torch.empty(256, 16, device="meta")
    index = torch.empty(256, dtype=torch.int64, device="meta")
    data_strategy = _create_all_options(mesh, data.shape, tensor=data)
    index_strategy = _create_all_options(mesh, index.shape, tensor=index)

    strategy = get_placement_options(
        mesh,
        torch.ops.aten._unsafe_index.Tensor,
        (data_strategy, [index_strategy, None]),
        (data, [index, None]),
        {},
    )

    assert len(strategy.strategies) == 1
    option = strategy.strategies[0]
    replicated = (Replicate(),) * mesh.ndim
    assert option.output_specs.placements == replicated
    assert all(spec.placements == replicated for spec in option.input_specs)


@pytest.mark.parametrize("fixture", ["device_mesh_1d", "device_mesh_2d"])
@apply_cuda_patches
def test_every_unbind_strategy_lowers(request, fixture):
    mesh = _mesh(request, fixture)
    shape = (256, 4, 16)
    value = torch.empty(shape, device="meta")
    input_strategy = _create_all_options(mesh, shape, tensor=value)
    strategy = get_placement_options(
        mesh,
        torch.ops.aten.unbind.int,
        (input_strategy, 1),
        (value, 1),
        {},
    )

    assert strategy.strategies
    for option in strategy.strategies:
        input_spec = option.input_specs[0]
        assert not any(placement.is_shard(1) for placement in input_spec.placements)
        expected_placements = tuple(
            Shard(placement.dim - 1)
            if placement.is_shard() and placement.dim > 1
            else placement
            for placement in input_spec.placements
        )
        assert len(option.output_specs) == shape[1]
        assert all(
            output_spec.placements == expected_placements
            for output_spec in option.output_specs
        )

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        unbound = graph.call_function(torch.ops.aten.unbind.int, (x, 1))
        graph.output(unbound)
        gm = torch.fx.GraphModule(nn.Module(), graph)
        x.meta["val"] = value
        unbound.meta["val"] = tuple(
            torch.empty(256, 16, device="meta") for _ in range(shape[1])
        )
        placements = {
            x: OpSpec(input_spec, input_specs=(input_spec,)),
            unbound: option,
        }

        local_shape = list(shape)
        for mesh_size, placement in zip(mesh.shape, input_spec.placements):
            if placement.is_shard():
                dim = placement.dim
                local_shape[dim] = (local_shape[dim] + mesh_size - 1) // mesh_size
        local = torch.randn(local_shape, device="cuda")
        parallel_gm = _lower_to_parallel_graph(
            gm, placements, [local], param_placement_order={}
        )
        outputs = parallel_gm(local)
        assert len(outputs) == shape[1]
        assert all(
            output.shape == tuple(local_shape[:1] + local_shape[2:])
            for output in outputs
        )


class UnsafeIndexModel(nn.Module):
    def forward(self, x, index):
        return torch.ops.aten._unsafe_index.Tensor(x, [index, None])


@apply_cuda_patches
def test_unsafe_index_lowers_and_executes(device_mesh_1d):
    with torch.device("meta"):
        model = UnsafeIndexModel()

    def input_fn():
        return (
            torch.randn(256, 16, device="cuda", requires_grad=True),
            torch.arange(255, -1, -1, device="cuda"),
        )

    replicated = (Replicate(),)
    with AutoParallel(
        model, input_fn, device_mesh_1d, repeated_subgraphs=False
    ) as autop:
        autop.add_input_constraints([replicated, replicated])
        autop.add_output_constraints([replicated])
        solution = autop.optimize_placement()
        parallel_model = autop.apply_placement(solution)

    x = torch.randn(256, 16, device="cuda", requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    index = torch.arange(255, -1, -1, device="cuda")
    actual = parallel_model(x, index)
    expected = reference_x[index]
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad)


class UnbindModel(nn.Module):
    def forward(self, x):
        parts = torch.unbind(x, dim=1)
        return parts[0] + parts[-1]


@apply_cuda_patches
def test_unbind_lowers_and_executes(device_mesh_1d):
    with torch.device("meta"):
        model = UnbindModel()

    def input_fn():
        return torch.randn(512, 4, 16, device="cuda", requires_grad=True)

    sharded = (Shard(0),)
    with AutoParallel(
        model, input_fn, device_mesh_1d, repeated_subgraphs=False
    ) as autop:
        autop.add_input_constraints([sharded])
        autop.add_output_constraints([sharded])
        solution = autop.optimize_placement()
        parallel_model = autop.apply_placement(solution)

    x = torch.randn(2, 4, 16, device="cuda", requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    actual = parallel_model(x)
    expected = reference_x[:, 0] + reference_x[:, -1]
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad)
