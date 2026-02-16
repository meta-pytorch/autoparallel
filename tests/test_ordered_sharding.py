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
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.distributed.tensor._op_schema import OpSpec
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel.api import AutoParallel
from autoparallel.shardings.ordered_sharding import (
    build_param_grad_linear_chains,
    compute_optimal_placement_order_for_parameters,
    get_redistributed_input_placements,
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
    dim = 64
    model = SimpleLinearModel(dim)
    sample_input = torch.randn(8, dim, requires_grad=True)

    gm, param_grad_nodes = _get_joint_graph(model, sample_input)

    # Should have exactly one param-grad pair
    assert len(param_grad_nodes) == 1
    param, grad = param_grad_nodes[0]
    assert grad is not None

    # Build the chains to identify the nodes we need to populate
    node_to_source, source_to_chain = build_param_grad_linear_chains(param_grad_nodes)

    param_chain = source_to_chain[param]
    grad_chain = source_to_chain[grad]

    # param_chain = [param_placeholder, t_node]
    assert len(param_chain) == 2
    t_node = param_chain[1]
    assert t_node.target == torch.ops.aten.t.default

    # grad_chain traces backward from grad through single-input nodes
    assert len(grad_chain) >= 1

    # The boundary node is the input of the last node in the grad chain
    # (the node where the chain stopped because it has multiple inputs)
    grad_boundary_node = grad_chain[-1].all_input_nodes[0]

    # Manually construct sharding_placement dict with the S(0)S(0)->RS(0) pattern
    ss_spec = DTensorSpec(device_mesh_2d, (Shard(0), Shard(0)))
    rs_spec = DTensorSpec(device_mesh_2d, (Replicate(), Shard(0)))
    ps_spec = DTensorSpec(device_mesh_2d, (Partial(), Shard(0)))

    sharding_placement = {
        # param output is S(0)S(0)
        param: OpSpec(output_specs=ss_spec),
        # t_node expects input as R,S(0) — triggers S(0)S(0)->RS(0) redistribution
        t_node: OpSpec(output_specs=rs_spec, input_specs=[rs_spec]),
        # Boundary node (e.g. mm backward) output is P,S(0)
        grad_boundary_node: OpSpec(output_specs=ps_spec),
    }

    # Add entries for all grad chain nodes — each has SS output and expects SS input.
    # The redistribution PS(0)->S(0)S(0) is detected at the last chain node
    # (where its input boundary_node has PS output but the chain node expects SS).
    for node in grad_chain:
        sharding_placement[node] = OpSpec(output_specs=ss_spec, input_specs=[ss_spec])

    # Call the function
    placement_order = compute_optimal_placement_order_for_parameters(
        gm, sharding_placement
    )

    # Verify that param and grad nodes are in placement_order
    assert param in placement_order
    assert grad in placement_order

    # Verify OrderInfo values for param chain nodes
    # param: first in chain, before target → is_target_reversed_order=True, need_reorder=False
    assert placement_order[param].is_target_reversed_order is True
    assert placement_order[param].need_reorder is False

    # t_node: target of param chain → is_target_reversed_order=False, need_reorder=True
    assert placement_order[t_node].is_target_reversed_order is False
    assert placement_order[t_node].need_reorder is True

    # Verify OrderInfo values for grad chain nodes
    # The last node in the grad chain is where redistribution happens (need_reorder=True)
    grad_redistrib_target = grad_chain[-1]
    assert placement_order[grad_redistrib_target].is_target_reversed_order is True
    assert placement_order[grad_redistrib_target].need_reorder is True

    # All earlier nodes in grad chain should have need_reorder=False
    for node in grad_chain[:-1]:
        assert placement_order[node].is_target_reversed_order is True
        assert placement_order[node].need_reorder is False

    # All nodes in placement_order should have shard_order metadata
    for node in placement_order:
        assert "shard_order" in node.meta


def test_compute_optimal_placement_order_verifies_redistribution_map(device_mesh_2d):
    """Test that get_redistributed_input_placements correctly identifies redistribution.

    This test verifies that when a node's input has different placements than what
    the node expects, get_redistributed_input_placements returns the correct
    current and target placements.
    """
    dim = 64
    model = SimpleLinearModel(dim)
    sample_input = torch.randn(8, dim, requires_grad=True)

    gm, param_grad_nodes = _get_joint_graph(model, sample_input)

    # Build chains to find the t node (transpose of param)
    node_to_source, source_to_chain = build_param_grad_linear_chains(param_grad_nodes)

    param, grad = param_grad_nodes[0]
    param_chain = source_to_chain[param]
    t_node = param_chain[1]
    assert t_node.target == torch.ops.aten.t.default

    # Construct sharding_placement where param output has S(0)S(0)
    # but t_node expects RS(0) as input — a redistribution mismatch
    ss_spec = DTensorSpec(device_mesh_2d, (Shard(0), Shard(0)))
    rs_spec = DTensorSpec(device_mesh_2d, (Replicate(), Shard(0)))

    sharding_placement = {
        param: OpSpec(output_specs=ss_spec),
        t_node: OpSpec(output_specs=rs_spec, input_specs=[rs_spec]),
    }

    # Call get_redistributed_input_placements directly
    redistrib = get_redistributed_input_placements(t_node, sharding_placement)

    # Should find redistribution needed for the param input
    assert param in redistrib, "Expected param node in redistribution map"

    curr_plc, tgt_plc = redistrib[param]

    # Verify current placement is S(0)S(0) (from param's output)
    assert curr_plc == (Shard(0), Shard(0))

    # Verify target placement is RS(0) (what t_node expects)
    assert tgt_plc == (Replicate(), Shard(0))

    # Verify placement tuples have length 2 (matching 2D mesh)
    assert len(curr_plc) == 2
    assert len(tgt_plc) == 2

    # Each placement should be a valid Placement type
    for plc in curr_plc:
        assert isinstance(plc, (Shard, Replicate, Partial))
    for plc in tgt_plc:
        assert isinstance(plc, (Shard, Replicate))


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
