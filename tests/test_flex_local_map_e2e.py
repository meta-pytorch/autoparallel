# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from autoparallel import flex_local_map
from autoparallel.api import AutoParallel
from autoparallel.collectives import get_flex_local_map_alternatives


def _default_body(x):
    return (torch.sin(x),)


def _replacement_body(x):
    return (torch.cos(x),)


class _BoundaryModel(nn.Module):
    def __init__(self, mesh, selected_index):
        super().__init__()
        self.scale = nn.Parameter(torch.empty(()))
        high_cost = 1_000_000.0
        self.pointwise = flex_local_map(
            _default_body,
            alternatives=[
                {
                    "name": "replicated_sin",
                    "in_placements": ((Replicate(),),),
                    "out_placements": ((Replicate(),),),
                    "cost_hint": 0.0 if selected_index == 0 else high_cost,
                },
                {
                    "name": "sharded_cos",
                    "fn": _replacement_body,
                    "in_placements": ((Shard(0),),),
                    "out_placements": ((Shard(0),),),
                    "cost_hint": 0.0 if selected_index == 1 else high_cost,
                },
            ],
            device_mesh=mesh,
            redistribute_inputs=True,
        )

    def forward(self, x):
        return self.pointwise(x * self.scale)[0]


def _local(value):
    return value.to_local() if isinstance(value, DTensor) else value


def _gather(value, sharded, world_size):
    gathered = [torch.empty_like(value) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, value)
    return torch.cat(gathered, dim=0) if sharded else gathered


def _run_case(mesh, rank, world_size, selected_index):
    shape = (512, 128)

    def input_fn():
        return torch.randn(*shape, device="cuda", requires_grad=True)

    with torch.device("meta"):
        model = _BoundaryModel(mesh, selected_index)

    placement = (Replicate(),) if selected_index == 0 else (Shard(0),)
    with AutoParallel(model, input_fn, mesh) as autop:
        autop.add_input_constraints([placement])
        autop.add_output_constraints([placement])
        solution = autop.optimize_placement()
        flex_nodes = [
            node
            for node in solution
            if get_flex_local_map_alternatives(node.meta.get("local_map_kwargs", {}))
            is not None
        ]
        assert flex_nodes
        assert {
            solution[node].flex_local_map_alternative_index for node in flex_nodes
        } == {selected_index}
        forward = next(
            node
            for node in flex_nodes
            if node.meta.get("partitioner_tag") != "is_backward"
        )
        parallel_model = autop.apply_placement(solution)
        selected_kwargs = forward.meta["local_map_kwargs"]
        assert get_flex_local_map_alternatives(selected_kwargs) is None
        assert selected_kwargs["in_placements"] == (placement,)
        assert selected_kwargs["out_placements"] == (placement,)

    parallel_model.to_empty(device="cuda")
    with torch.no_grad():
        parallel_model.scale.fill_(1.25)

    generator = torch.Generator(device="cuda").manual_seed(20260728)
    full_input = torch.randn(*shape, generator=generator, device="cuda")
    local_input = (
        full_input
        if selected_index == 0
        else full_input.chunk(world_size, dim=0)[rank].contiguous()
    )
    actual_input = local_input.detach().clone().requires_grad_(True)
    expected_input = local_input.detach().clone().requires_grad_(True)
    expected_scale = torch.tensor(1.25, device="cuda", requires_grad=True)

    actual = parallel_model(actual_input)
    body = _default_body if selected_index == 0 else _replacement_body
    expected = body(expected_input * expected_scale)[0]
    actual.sum().backward()
    expected.sum().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_input.grad, expected_input.grad)

    expected_scale_grad = expected_scale.grad.detach().clone()
    scale_grad = parallel_model.scale.grad
    if selected_index == 1:
        torch.distributed.all_reduce(expected_scale_grad)
    torch.testing.assert_close(_local(scale_grad), expected_scale_grad)

    gathered_output = _gather(actual.detach(), selected_index == 1, world_size)
    gathered_input_grad = _gather(
        actual_input.grad.detach(), selected_index == 1, world_size
    )
    if rank == 0:
        global_input = full_input.detach().clone().requires_grad_(True)
        global_scale = torch.tensor(1.25, device="cuda", requires_grad=True)
        global_expected = body(global_input * global_scale)[0]
        global_expected.sum().backward()
        if selected_index == 0:
            for output, input_grad in zip(gathered_output, gathered_input_grad):
                torch.testing.assert_close(output, global_expected)
                torch.testing.assert_close(input_grad, global_input.grad)
        else:
            torch.testing.assert_close(gathered_output, global_expected)
            torch.testing.assert_close(gathered_input_grad, global_input.grad)

    torch.distributed.barrier(device_ids=[rank])


class TestFlexLocalMapE2E(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @pytest.mark.skipif(
        torch.cuda.device_count() < 4,
        reason="flex_local_map E2E test requires at least 4 GPUs",
    )
    @with_comms
    def test_alternative_body_and_boundary_forward_backward(self):
        mesh = init_device_mesh(
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("dp",),
        )
        for selected_index in (0, 1):
            _run_case(mesh, self.rank, self.world_size, selected_index)


if __name__ == "__main__":
    run_tests()
