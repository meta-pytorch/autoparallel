# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from conftest import apply_cuda_patches
from torch import nn
from torch._functorch._aot_autograd.descriptors import (
    BufferAOTInput,
    InputMutationAOTOutput,
)
from torch._functorch._aot_autograd.fx_utils import (
    get_all_input_and_grad_nodes,
    get_all_output_and_tangent_nodes,
)
from torch.ao.quantization.fake_quantize import FusedMovingAvgObsFakeQuantize
from torch.ao.quantization.observer import MovingAveragePerChannelMinMaxObserver
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel.api import AutoParallel


class SliceMutation(nn.Module):
    def forward(self, x):
        with torch.no_grad():
            x[:, :2].add_(1)
        return x * x


class BufferMutation(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("state", torch.zeros(8))

    def forward(self, x):
        self.state.add_(x.mean(0))
        return x + self.state


class PerChannelQATLinear(nn.Linear):
    def __init__(self):
        super().__init__(4, 8)
        self.weight_fake_quant = FusedMovingAvgObsFakeQuantize(
            observer=MovingAveragePerChannelMinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
        )

    def forward(self, x):
        return nn.functional.linear(x, self.weight_fake_quant(self.weight), self.bias)


def _get_mutation_pair(optimizer):
    inputs = get_all_input_and_grad_nodes(optimizer.graph)
    outputs = get_all_output_and_tangent_nodes(optimizer.graph)
    desc, (mutation_node, _tangent) = next(
        (desc, pair)
        for desc, pair in outputs.items()
        if isinstance(desc, InputMutationAOTOutput)
    )
    input_node, _grad = inputs[desc.mutated_input]
    return input_node, mutation_node


@apply_cuda_patches
def test_input_mutation_rejects_incompatible_placement(device_mesh_1d):
    with torch.device("meta"):
        model = SliceMutation()

    def input_fn():
        return torch.randn(
            device_mesh_1d.size() * 2,
            8,
            device="cuda",
            requires_grad=True,
        )

    with AutoParallel(
        model, input_fn, device_mesh_1d, repeated_subgraphs=False
    ) as autop:
        input_node, mutation_node = _get_mutation_pair(autop.sharding_optimizer)
        mutation_placements = {
            strategy.output_specs.placements
            for strategy in autop.sharding_optimizer.strats[mutation_node].strategies
        }
        assert (Shard(0),) in mutation_placements

        autop.add_input_constraints([(Replicate(),)])
        autop.sharding_optimizer.add_node_constraint(mutation_node, (Shard(0),))
        with pytest.raises(RuntimeError, match="could not find a feasible solution"):
            autop.optimize_placement()


@apply_cuda_patches
def test_input_mutation_executes_with_input_placement(device_mesh_1d):
    with torch.device("meta"):
        model = SliceMutation()

    def input_fn():
        return torch.randn(
            device_mesh_1d.size() * 2,
            8,
            device="cuda",
            requires_grad=True,
        )

    with AutoParallel(
        model, input_fn, device_mesh_1d, repeated_subgraphs=False
    ) as autop:
        input_node, mutation_node = _get_mutation_pair(autop.sharding_optimizer)
        autop.add_input_constraints([(Replicate(),)])
        autop.add_output_constraints([(Replicate(),)])
        solution = autop.optimize_placement()
        assert solution[
            autop.sharding_optimizer._concrete_to_orig[input_node]
        ].output_specs.placements == (Replicate(),)
        assert solution[
            autop.sharding_optimizer._concrete_to_orig[mutation_node]
        ].output_specs.placements == (Replicate(),)
        parallel_model = autop.apply_placement(solution)

    actual_input = torch.randn(
        device_mesh_1d.size() * 2,
        8,
        device="cuda",
        requires_grad=True,
    )
    expected_input = actual_input.detach().clone().requires_grad_()
    expected = SliceMutation()(expected_input)
    actual = parallel_model(actual_input)

    torch.testing.assert_close(actual_input, expected_input)
    torch.testing.assert_close(actual, expected)

    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(actual_input.grad, expected_input.grad)


@apply_cuda_patches
def test_buffer_mutation_is_constrained(device_mesh_1d):
    with torch.device("meta"):
        model = BufferMutation()

    def input_fn():
        return torch.randn(device_mesh_1d.size() * 2, 8, device="cuda")

    with AutoParallel(
        model, input_fn, device_mesh_1d, repeated_subgraphs=False
    ) as autop:
        outputs = get_all_output_and_tangent_nodes(autop.sharding_optimizer.graph)
        mutation_desc = next(
            desc for desc in outputs if isinstance(desc, InputMutationAOTOutput)
        )
        assert isinstance(mutation_desc.mutated_input, BufferAOTInput)
        assert any(
            "input_mutation_constraint" in name
            for name in autop.sharding_optimizer.prob.constraints
        )


@apply_cuda_patches
def test_per_channel_qat_mutation_uses_channel_sized_buffers(device_mesh_1d):
    with torch.device("meta"):
        model = PerChannelQATLinear()

    def input_fn():
        return torch.randn(device_mesh_1d.size() * 2, 4, device="cuda")

    with AutoParallel(
        model, input_fn, device_mesh_1d, repeated_subgraphs=False
    ) as autop:
        inputs = get_all_input_and_grad_nodes(autop.sharding_optimizer.graph)
        outputs = get_all_output_and_tangent_nodes(autop.sharding_optimizer.graph)
        shapes = []
        for desc, (mutation_node, _tangent) in outputs.items():
            if not isinstance(desc, InputMutationAOTOutput) or not isinstance(
                desc.mutated_input, BufferAOTInput
            ):
                continue
            if "weight_fake_quant" not in desc.mutated_input.target:
                continue
            input_node, _grad = inputs[desc.mutated_input]
            shapes.append(
                (input_node.meta["val"].shape, mutation_node.meta["val"].shape)
            )

        assert shapes == [((8,), (8,))] * 4
