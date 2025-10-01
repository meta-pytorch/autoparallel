# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel.api import AutoParallel
from autoparallel.ordered_sharding import compute_optimal_placement_order_for_parameters


@pytest.fixture(scope="module", autouse=True)
def init_pg():
    world_size = 256
    fake_store = FakeStore()
    if torch.distributed.is_initialized():
        return
    torch.distributed.init_process_group(
        "fake", store=fake_store, rank=0, world_size=world_size
    )


@pytest.fixture(scope="module")
def device_mesh_2d():
    world_size = torch.distributed.get_world_size()
    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda",
        (world_size // 8, 8),
        mesh_dim_names=(
            "dp",
            "tp",
        ),
    )
    return mesh


class ModelWithNonTrainableParams(nn.Module):
    """A model with both trainable and non-trainable parameters to test the grad is None case."""

    def __init__(self, dim):
        super().__init__()
        # Trainable parameter (requires_grad=True by default)
        self.linear = nn.Linear(dim, dim, bias=False)

        # Non-trainable parameters (requires_grad=False)
        self.register_parameter(
            "non_trainable_weight",
            nn.Parameter(torch.randn(dim, dim), requires_grad=False),
        )
        self.register_buffer("buffer", torch.randn(dim))

        # Another trainable parameter
        self.linear2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        # Use both trainable and non-trainable parameters
        x = self.linear(x)
        x = x + torch.mm(x, self.non_trainable_weight)  # Use non-trainable parameter
        x = x + self.buffer  # Use buffer
        x = self.linear2(x)
        return x


class ModelWithAllNonTrainableParams(nn.Module):
    """A model where all parameters don't require gradients."""

    def __init__(self, dim):
        super().__init__()
        # Create linear layers but set requires_grad=False for all params
        self.linear1 = nn.Linear(dim, dim, bias=False)
        self.linear2 = nn.Linear(dim, dim, bias=False)

        # Set all parameters to not require gradients
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


@patch("torch.cuda.device_count", lambda: 8)
@patch("torch.cuda.get_device_name", lambda device: "H100")
def test_compute_optimal_placement_order_with_non_trainable_params(device_mesh_2d):
    """Test that compute_optimal_placement_order_for_parameters handles parameters with grad=None."""

    dim = 128
    device = "cuda"

    def model_fn():
        return ModelWithNonTrainableParams(dim)

    def input_fn():
        return torch.randn(512, dim, device=device, requires_grad=True)

    with torch.device("meta"):
        model = model_fn()

    # Verify our test setup: some params should have requires_grad=False
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    non_trainable_params = [p for p in model.parameters() if not p.requires_grad]

    assert (
        len(trainable_params) > 0
    ), "Test setup error: should have some trainable params"
    assert (
        len(non_trainable_params) > 0
    ), "Test setup error: should have some non-trainable params"

    with AutoParallel(model, input_fn, device_mesh_2d) as autop:
        autop.add_parameter_memory_constraint(low=0, high=None)
        sharding_placement = autop.optimize_placement()

        # This should not raise an exception due to grad=None
        # Before the fix, this would fail when trying to process non-trainable parameters
        placement_order = compute_optimal_placement_order_for_parameters(
            autop.gm, sharding_placement
        )

        # The function should return successfully
        assert isinstance(placement_order, dict)
        assert len(placement_order) == 0

        # Verify we can examine the graph structure to understand param/grad relationships
        from torch._functorch._aot_autograd.fx_utils import get_param_and_grad_nodes

        param_and_grad_nodes = list(get_param_and_grad_nodes(autop.gm.graph).values())

        # Should have param/grad pairs where some grads are None
        assert len(param_and_grad_nodes) > 0

        # At least one should have grad=None (the non-trainable param)
        has_none_grad = any(grad is None for param, grad in param_and_grad_nodes)
        assert has_none_grad, "Expected at least one parameter to have grad=None"

        # At least one should have a valid grad (the trainable param)
        has_valid_grad = any(grad is not None for param, grad in param_and_grad_nodes)
        assert (
            has_valid_grad
        ), "Expected at least one parameter to have a valid gradient"


@patch("torch.cuda.device_count", lambda: 8)
@patch("torch.cuda.get_device_name", lambda device: "H100")
def test_compute_optimal_placement_order_with_all_non_trainable_params(device_mesh_2d):
    """Test edge case where ALL parameters don't require gradients."""

    dim = 64
    device = "cuda"

    def model_fn():
        return ModelWithAllNonTrainableParams(dim)

    def input_fn():
        return torch.randn(256, dim, device=device, requires_grad=True)

    with torch.device("meta"):
        model = model_fn()

    # Verify test setup: all params should have requires_grad=False
    non_trainable_params = [p for p in model.parameters() if not p.requires_grad]
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    assert (
        len(non_trainable_params) > 0
    ), "Test setup error: should have non-trainable params"
    assert (
        len(trainable_params) == 0
    ), "Test setup error: should have NO trainable params"

    with AutoParallel(model, input_fn, device_mesh_2d) as autop:
        autop.add_parameter_memory_constraint(low=0, high=None)
        sharding_placement = autop.optimize_placement()

        # This should not raise an exception even when ALL gradients are None
        placement_order = compute_optimal_placement_order_for_parameters(
            autop.gm, sharding_placement
        )

        # Should return successfully with empty or minimal result
        assert isinstance(placement_order, dict)
        assert len(placement_order) == 0
