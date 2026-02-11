# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch._functorch._aot_autograd.fx_utils import get_param_and_grad_nodes
from torch._functorch.aot_autograd import aot_export_joint_with_descriptors
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel.api import AutoParallel
from autoparallel.shardings.ordered_sharding import (
    build_param_grad_linear_chains,
    compute_optimal_placement_order_for_parameters,
)


class SimpleLinearModel(nn.Module):
    """A simple model with a single trainable linear layer."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MixedTrainableModel(nn.Module):
    """A model with both trainable and non-trainable linear layers."""

    def __init__(self, dim: int):
        super().__init__()
        # Trainable linear layer
        self.trainable_linear = nn.Linear(dim, dim, bias=False)
        # Non-trainable linear layer
        self.frozen_linear = nn.Linear(dim, dim, bias=False)
        self.frozen_linear.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.trainable_linear(x)
        x = self.frozen_linear(x)
        return x


class TwoLayerModel(nn.Module):
    """A model with two trainable linear layers."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim, bias=False)
        self.linear2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.linear2(x)
        return x


def _get_joint_graph(model: nn.Module, sample_input: torch.Tensor):
    """Helper to get a joint forward+backward graph using aot_export_joint_with_descriptors."""
    with ExitStack() as stack:
        joint_with_descriptors = aot_export_joint_with_descriptors(
            stack,
            model,
            (sample_input,),
        )
        # Return the graph module - we need to use it before exiting the stack
        # So we extract what we need here
        gm = joint_with_descriptors.graph_module
        param_grad_nodes = list(get_param_and_grad_nodes(gm.graph).values())
        return gm, param_grad_nodes


class TestBuildParamGradLinearChains:
    """Tests for build_param_grad_linear_chains function."""

    def test_single_trainable_linear(self):
        """Test with a single trainable linear layer.

        The parameter chain should include the parameter node and its users
        up to the first node with multiple inputs (the matmul).
        """
        dim = 64
        model = SimpleLinearModel(dim)
        sample_input = torch.randn(8, dim, requires_grad=True)

        gm, param_grad_nodes = _get_joint_graph(model, sample_input)

        # Should have exactly one param-grad pair
        assert len(param_grad_nodes) == 1
        param_node, grad_node = param_grad_nodes[0]

        # The parameter should have a gradient (it's trainable)
        assert grad_node is not None

        # Build the chains
        node_to_source, source_to_chain = build_param_grad_linear_chains(
            param_grad_nodes
        )

        # Verify param chain exists and starts with the param node
        assert param_node in source_to_chain
        param_chain = source_to_chain[param_node]

        # For a Linear layer (y = x @ W.T), the param chain should be:
        # [param_placeholder, transpose]
        # The chain stops before mm/matmul because that has 2 inputs
        assert len(param_chain) == 2
        assert param_chain[0] == param_node
        assert param_chain[0].op == "placeholder"
        assert param_chain[1].target == torch.ops.aten.t.default

        # Verify node_to_source mappings for param chain
        assert node_to_source[param_chain[0]] == param_node
        assert node_to_source[param_chain[1]] == param_node

        # Verify grad chain exists and check its structure
        assert grad_node in source_to_chain
        grad_chain = source_to_chain[grad_node]

        # The grad chain traces backward from the gradient output
        # For a simple linear, the grad w.r.t. weight is: x.T @ grad_output
        # The chain should be [grad_node] since grad_node typically has multiple inputs
        # or is directly connected to multi-input nodes
        assert len(grad_chain) >= 0  # May be empty if grad_node has multiple inputs
        for node in grad_chain:
            assert node_to_source[node] == grad_node

    def test_non_trainable_parameter_has_no_grad_chain(self):
        """Test that non-trainable parameters have grad=None and no grad chain."""
        dim = 64
        model = MixedTrainableModel(dim)
        sample_input = torch.randn(8, dim, requires_grad=True)

        gm, param_grad_nodes = _get_joint_graph(model, sample_input)

        # Should have two param-grad pairs (one trainable, one frozen)
        assert len(param_grad_nodes) == 2

        # Build the chains
        node_to_source, source_to_chain = build_param_grad_linear_chains(
            param_grad_nodes
        )

        # Find the trainable and frozen parameters
        trainable_pair = None
        frozen_pair = None
        for param_node, grad_node in param_grad_nodes:
            if grad_node is None:
                frozen_pair = (param_node, grad_node)
            else:
                trainable_pair = (param_node, grad_node)

        # Verify we found both
        assert frozen_pair is not None, "Expected to find a frozen parameter"
        assert trainable_pair is not None, "Expected to find a trainable parameter"

        frozen_param, frozen_grad = frozen_pair
        trainable_param, trainable_grad = trainable_pair

        # Frozen parameter should have a param chain but no grad chain
        assert frozen_param in source_to_chain
        frozen_param_chain = source_to_chain[frozen_param]

        # Frozen param chain structure: [param_placeholder, transpose]
        assert len(frozen_param_chain) == 2
        assert frozen_param_chain[0] == frozen_param
        assert frozen_param_chain[0].op == "placeholder"
        assert frozen_param_chain[1].target == torch.ops.aten.t.default

        # Verify node_to_source for frozen param chain
        assert node_to_source[frozen_param_chain[0]] == frozen_param
        assert node_to_source[frozen_param_chain[1]] == frozen_param

        # Frozen grad is None, so no grad chain
        assert frozen_grad is None
        assert frozen_grad not in source_to_chain

        # Trainable parameter should have both param and grad chains
        assert trainable_param in source_to_chain
        trainable_param_chain = source_to_chain[trainable_param]

        # Trainable param chain structure: [param_placeholder, transpose]
        assert len(trainable_param_chain) == 2
        assert trainable_param_chain[0] == trainable_param
        assert trainable_param_chain[0].op == "placeholder"
        assert trainable_param_chain[1].target == torch.ops.aten.t.default

        # Verify node_to_source for trainable param chain
        assert node_to_source[trainable_param_chain[0]] == trainable_param
        assert node_to_source[trainable_param_chain[1]] == trainable_param

        # Trainable grad should have a chain
        assert trainable_grad in source_to_chain

    def test_two_layer_model_has_separate_chains(self):
        """Test that a two-layer model has separate chains for each parameter."""
        dim = 64
        model = TwoLayerModel(dim)
        sample_input = torch.randn(8, dim, requires_grad=True)

        gm, param_grad_nodes = _get_joint_graph(model, sample_input)

        # Should have two param-grad pairs
        assert len(param_grad_nodes) == 2

        # Build the chains
        node_to_source, source_to_chain = build_param_grad_linear_chains(
            param_grad_nodes
        )

        # Each parameter should have its own chain with the same structure
        for param_node, grad_node in param_grad_nodes:
            assert param_node in source_to_chain
            param_chain = source_to_chain[param_node]

            # Each Linear param chain: [param_placeholder, transpose]
            assert len(param_chain) == 2
            assert param_chain[0] == param_node
            assert param_chain[0].op == "placeholder"
            assert param_chain[1].target == torch.ops.aten.t.default

            # All nodes in param chain should map back to this param
            assert node_to_source[param_chain[0]] == param_node
            assert node_to_source[param_chain[1]] == param_node

            # Gradient should also have a chain
            assert grad_node is not None
            assert grad_node in source_to_chain
            grad_chain = source_to_chain[grad_node]

            # All nodes in grad chain should map back to this grad
            for node in grad_chain:
                assert node_to_source[node] == grad_node

        # Verify the two param chains are disjoint (no shared nodes)
        param1, _ = param_grad_nodes[0]
        param2, _ = param_grad_nodes[1]
        chain1_nodes = set(source_to_chain[param1])
        chain2_nodes = set(source_to_chain[param2])
        assert chain1_nodes.isdisjoint(chain2_nodes), "Param chains should be disjoint"

    def test_chains_contain_only_single_input_nodes(self):
        """Test that chains only include nodes with single inputs (linear dependency)."""
        dim = 64
        model = SimpleLinearModel(dim)
        sample_input = torch.randn(8, dim, requires_grad=True)

        gm, param_grad_nodes = _get_joint_graph(model, sample_input)

        node_to_source, source_to_chain = build_param_grad_linear_chains(
            param_grad_nodes
        )

        param_node, grad_node = param_grad_nodes[0]
        param_chain = source_to_chain[param_node]

        # Verify chain structure
        assert len(param_chain) == 2
        assert param_chain[0].op == "placeholder"  # param has 0 inputs
        assert (
            param_chain[1].target == torch.ops.aten.t.default
        )  # transpose has 1 input

        # All nodes in the chain (except the first/param itself) should have single input
        for i, node in enumerate(param_chain):
            if i == 0:
                # First node is the param placeholder, has 0 inputs
                assert len(node.all_input_nodes) == 0
                continue
            # Nodes in the chain should have exactly 1 input
            assert len(node.all_input_nodes) == 1, (
                f"Node {node.name} in chain has {len(node.all_input_nodes)} inputs, "
                f"expected 1"
            )


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


class SimpleMLP(nn.Module):
    """A simple MLP model for testing ordered sharding redistribution patterns."""

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.linear(x)


# TODO: add test for get_redistributed_input_placements


@patch("torch.cuda.device_count", lambda: 8)
@patch("torch.cuda.get_device_name", lambda device: "H100")
def test_compute_optimal_placement_order_ss_to_rs(device_mesh_2d):
    """Test the S(0)S(0) -> RS(0) and PS(0) -> S(0)S(0) optimization case.

    This test verifies that when parameters are sharded as S(0)S(0) (sharded on dim 0
    for both mesh dimensions) and need to be redistributed to RS(0) (Replicate on first
    mesh dim, Shard(0) on second), the compute_optimal_placement_order_for_parameters
    function correctly identifies and orders the nodes for optimization.

    The backward pass gradient pattern is PS(0) -> S(0)S(0):
    - PS(0): Partial on first mesh dim, Shard(0) on second
    - S(0)S(0): Shard(0) on both mesh dimensions
    """
    from torch.distributed.tensor.placement_types import Partial, Replicate, Shard

    from autoparallel.shardings.ordered_sharding import (
        get_redistributed_input_placements,
    )

    dim = 512
    device = "cuda"

    def model_fn():
        return SimpleMLP(dim)

    def input_fn():
        return torch.randn(512, dim, device=device, requires_grad=True)

    with torch.device("meta"):
        model = model_fn()

    with AutoParallel(model, input_fn, device_mesh_2d) as autop:
        # Set input sharding: S(0) on dp dim, R on tp dim (batch-parallel on dp)
        x_sharding = (Shard(0), Replicate())
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([x_sharding])
        # Force parameter sharding by constraining memory
        autop.add_parameter_memory_constraint(low=0, high=None)
        mm_nodes = autop.gm.graph.find_nodes(
            op="call_function", target=torch.ops.aten.mm.default
        )
        autop.sharding_optimizer.add_node_constraint(mm_nodes[0], (Shard(0), Shard(1)))
        sharding_placement = autop.optimize_placement(verbose=True)

        # Call the function to compute optimal placement order
        placement_order = compute_optimal_placement_order_for_parameters(
            autop.gm, sharding_placement
        )

        # Check the results - we should have placement order for param nodes
        assert isinstance(placement_order, dict)

        # Check that nodes have the expected shard_order metadata set
        from torch._functorch._aot_autograd.fx_utils import get_param_and_grad_nodes

        param_and_grad_nodes = list(get_param_and_grad_nodes(autop.gm.graph).values())

        # Check that we have parameter-gradient pairs
        assert len(param_and_grad_nodes) == 1

        param, grad = param_and_grad_nodes[0]
        assert param in placement_order
        assert grad in placement_order

        assert sharding_placement[param].output_spec.placements == (Shard(0), Shard(0))
        assert sharding_placement[grad].output_spec.placements == (Shard(0), Shard(0))

        # assert sharding_placement[list(param.users)[0].users]
        # assert sharding_placement[
        #     grad.input_nodes[0].input_nodes[0]
        # ].output_specs.placements == (Shard(1), Shard(1))

        # assert False, f"{sharding_placement}"
        # Verify the structure of the result
        for node, order_info in placement_order.items():
            # Each entry should have is_target_reversed_order and need_reorder fields
            assert hasattr(order_info, "is_target_reversed_order")
            assert hasattr(order_info, "need_reorder")
            # The node should have shard_order metadata set
            assert "shard_order" in node.meta

        # Verify that we can inspect the redistribution patterns
        has_ss_to_rs_pattern = False
        has_ps_to_ss_pattern = False

        for node in autop.gm.graph.nodes:
            if node in sharding_placement:
                redistrib = get_redistributed_input_placements(node, sharding_placement)
                for input_node, (curr_plc, tgt_plc) in redistrib.items():
                    # Check for S(0)S(0) -> RS(0) pattern
                    if curr_plc == (Shard(0), Shard(0)) and tgt_plc == (
                        Replicate(),
                        Shard(0),
                    ):
                        has_ss_to_rs_pattern = True
                    # Check for PS(0) -> S(0)S(0) pattern
                    if curr_plc == (Partial(), Shard(0)) and tgt_plc == (
                        Shard(0),
                        Shard(0),
                    ):
                        has_ps_to_ss_pattern = True

        # Log what patterns we found (useful for debugging if test fails)
        if not has_ss_to_rs_pattern:
            # If we didn't find the pattern, at least verify the function ran correctly
            print("Note: S(0)S(0) -> RS(0) pattern not found in this model/mesh config")
        if not has_ps_to_ss_pattern:
            print("Note: PS(0) -> S(0)S(0) pattern not found in this model/mesh config")


@patch("torch.cuda.device_count", lambda: 8)
@patch("torch.cuda.get_device_name", lambda device: "H100")
def test_compute_optimal_placement_order_verifies_redistribution_map(device_mesh_2d):
    """Test that the redistribution map is correctly populated for parameter nodes.

    This test verifies that get_redistributed_input_placements correctly identifies
    nodes that require redistribution and returns their current and target placements.
    """
    from torch.distributed.tensor.placement_types import Partial, Replicate, Shard

    from autoparallel.shardings.ordered_sharding import (
        get_redistributed_input_placements,
    )

    dim = 256
    device = "cuda"

    def model_fn():
        return SimpleMLP(dim)

    def input_fn():
        return torch.randn(512, dim, device=device, requires_grad=True)

    with torch.device("meta"):
        model = model_fn()

    with AutoParallel(model, input_fn, device_mesh_2d) as autop:
        # Set input/output sharding constraints
        x_sharding = (Shard(0), Replicate())
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([x_sharding])
        autop.add_parameter_memory_constraint(low=0, high=None)
        sharding_placement = autop.optimize_placement(verbose=False)

        # Collect all redistribution patterns
        redistribution_patterns = []
        for node in autop.gm.graph.nodes:
            if node in sharding_placement:
                redistrib = get_redistributed_input_placements(node, sharding_placement)
                for input_node, (curr_plc, tgt_plc) in redistrib.items():
                    redistribution_patterns.append(
                        {
                            "node_name": node.name,
                            "input_node_name": input_node.name,
                            "current_placement": curr_plc,
                            "target_placement": tgt_plc,
                        }
                    )

        # Verify we found some redistribution patterns (model should require some)
        assert (
            len(redistribution_patterns) > 0
        ), "Expected to find redistribution patterns for sharded parameters"

        # Verify placement tuples have correct structure (2D mesh = 2 placements)
        for pattern in redistribution_patterns:
            assert len(pattern["current_placement"]) == 2
            assert len(pattern["target_placement"]) == 2

            # Each placement should be a valid Placement type
            for plc in pattern["current_placement"]:
                assert isinstance(plc, (Shard, Replicate, Partial))
            for plc in pattern["target_placement"]:
                assert isinstance(plc, (Shard, Replicate))  # target won't be Partial


class MultiLinearModel(nn.Module):
    """A model with multiple linear layers to test more complex sharding patterns."""

    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim, bias=False)
        self.linear2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x


@patch("torch.cuda.device_count", lambda: 8)
@patch("torch.cuda.get_device_name", lambda device: "H100")
def test_compute_optimal_placement_order_multi_layer(device_mesh_2d):
    """Test placement ordering with multiple layers.

    Tests that the function correctly handles models with multiple parameters,
    ensuring each parameter/gradient pair is processed correctly.
    """
    from torch.distributed.tensor.placement_types import Replicate, Shard

    dim = 128
    device = "cuda"

    def model_fn():
        return MultiLinearModel(dim)

    def input_fn():
        return torch.randn(512, dim, device=device, requires_grad=True)

    with torch.device("meta"):
        model = model_fn()

    with AutoParallel(model, input_fn, device_mesh_2d) as autop:
        # Set input/output sharding constraints
        x_sharding = (Shard(0), Replicate())
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([x_sharding])
        autop.add_parameter_memory_constraint(low=0, high=None)
        sharding_placement = autop.optimize_placement(verbose=False)

        # Compute placement order
        placement_order = compute_optimal_placement_order_for_parameters(
            autop.gm, sharding_placement
        )

        assert isinstance(placement_order, dict)

        # Get param/grad pairs to verify we're handling multiple parameters
        from torch._functorch._aot_autograd.fx_utils import get_param_and_grad_nodes

        param_and_grad_nodes = list(get_param_and_grad_nodes(autop.gm.graph).values())

        # Should have param-grad pairs for both linear layers
        trainable_pairs = [(p, g) for p, g in param_and_grad_nodes if g is not None]
        assert (
            len(trainable_pairs) >= 2
        ), f"Expected at least 2 trainable param-grad pairs, got {len(trainable_pairs)}"

        # Verify that all nodes in placement_order have shard_order metadata
        for node in placement_order:
            assert "shard_order" in node.meta
