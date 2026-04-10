# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import nn
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel import AutoParallel, with_sharding_constraint
from autoparallel.collectives import local_map
from autoparallel.ops import permutation


def get_local_map_nodes(graph, is_backward=False):
    nodes = []
    for node in graph.nodes:
        if "local_map_kwargs" in node.meta:
            node_is_backward = node.meta.get("partitioner_tag", "") == "is_backward"
            if node_is_backward == is_backward:
                nodes.append(node)
    return nodes


def verify_local_map_placements(sharding_placement, node, expected_placements):
    spec = sharding_placement[node]
    if isinstance(spec.output_specs, tuple):
        output_spec = spec.output_specs[0]
    else:
        output_spec = spec.output_specs
    assert (
        output_spec.placements == expected_placements
    ), f"Expected placements {expected_placements}, got {output_spec.placements}"


class TestWithShardingConstraint:
    """Tests for the with_sharding_constraint operator."""

    def test_with_sharding_constraint_explicit_mesh(self, device_mesh_1d):
        """Test with_sharding_constraint with an explicit device mesh."""
        dim = 128

        class Model(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.linear1 = nn.Linear(dim, dim, bias=False)
                self.linear2 = nn.Linear(dim, dim, bias=False)

            def forward(self, x):
                x = self.linear1(x)
                # Constrain intermediate result to be sharded
                x = with_sharding_constraint(x, (Shard(0),), device_mesh_1d)
                x = self.linear2(x)
                return x

        def input_fn():
            return torch.rand(512, dim, device="cuda")

        with torch.device("meta"):
            model = Model(dim)

        with AutoParallel(model, input_fn, device_mesh_1d) as autop:
            autop.add_input_constraints([(Shard(0),)])
            autop.add_output_constraints([(Shard(0),)])
            sharding_placement = autop.optimize_placement()

            # Verify the with_sharding_constraint node has correct placement
            local_map_nodes = get_local_map_nodes(autop.gm.graph, is_backward=False)
            assert len(local_map_nodes) == 1, "Expected 1 forward local_map node"
            verify_local_map_placements(
                sharding_placement, local_map_nodes[0], (Shard(0),)
            )

            parallel_mod = autop.apply_placement(sharding_placement)

        assert parallel_mod is not None

    def test_with_sharding_constraint_between_local_maps(self, device_mesh_1d):
        """Test with_sharding_constraint between local_map regions."""
        dim = 128

        @local_map(
            out_placements=((Shard(0),),),
            in_placements=((Shard(0),),),
            redistribute_inputs=True,
            device_mesh=device_mesh_1d,
        )
        def compute1(x):
            return x + 1

        @local_map(
            out_placements=((Shard(0),),),
            in_placements=((Shard(0),),),
            redistribute_inputs=True,
            device_mesh=device_mesh_1d,
        )
        def compute2(x):
            return x * 2

        class Model(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.linear = nn.Linear(dim, dim, bias=False)

            def forward(self, x):
                x = self.linear(x)
                x = compute1(x)
                # Constraint applied between local_map regions (at DTensor level)
                x = with_sharding_constraint(x, (Shard(0),), device_mesh_1d)
                x = compute2(x)
                return x

        def input_fn():
            return torch.rand(512, dim, device="cuda")

        with torch.device("meta"):
            model = Model(dim)

        with AutoParallel(model, input_fn, device_mesh_1d) as autop:
            autop.add_input_constraints([(Shard(0),)])
            autop.add_output_constraints([(Shard(0),)])
            sharding_placement = autop.optimize_placement()

            # Verify all local_map nodes have correct placement
            # There are 3 forward local_map nodes: compute1, with_sharding_constraint, compute2
            local_map_nodes = get_local_map_nodes(autop.gm.graph, is_backward=False)
            assert (
                len(local_map_nodes) == 3
            ), f"Expected 3 forward local_map nodes, got {len(local_map_nodes)}"
            for node in local_map_nodes:
                verify_local_map_placements(sharding_placement, node, (Shard(0),))

            parallel_mod = autop.apply_placement(sharding_placement)

        assert parallel_mod is not None

    def test_with_sharding_constraint_replicate(self, device_mesh_1d):
        """Test with_sharding_constraint to force replication."""
        dim = 128

        class Model(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.linear1 = nn.Linear(dim, dim, bias=False)
                self.linear2 = nn.Linear(dim, dim, bias=False)

            def forward(self, x):
                x = self.linear1(x)
                # Force intermediate to be replicated
                x = with_sharding_constraint(x, (Replicate(),), device_mesh_1d)
                x = self.linear2(x)
                return x

        def input_fn():
            return torch.rand(512, dim, device="cuda")

        with torch.device("meta"):
            model = Model(dim)

        with AutoParallel(model, input_fn, device_mesh_1d) as autop:
            autop.add_input_constraints([(Shard(0),)])
            autop.add_output_constraints([(Shard(0),)])
            sharding_placement = autop.optimize_placement()

            # Verify the with_sharding_constraint node forces Replicate
            local_map_nodes = get_local_map_nodes(autop.gm.graph, is_backward=False)
            assert len(local_map_nodes) == 1, "Expected 1 forward local_map node"
            verify_local_map_placements(
                sharding_placement, local_map_nodes[0], (Replicate(),)
            )

            parallel_mod = autop.apply_placement(sharding_placement)

        assert parallel_mod is not None

    def test_with_sharding_constraint_2d_mesh(self, device_mesh_2d):
        """Test with_sharding_constraint on a 2D mesh."""
        dim = 128

        class Model(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.linear1 = nn.Linear(dim, dim, bias=False)
                self.linear2 = nn.Linear(dim, dim, bias=False)

            def forward(self, x):
                x = self.linear1(x)
                # Shard along batch dim on dp, replicate on tp
                x = with_sharding_constraint(x, (Shard(0), Replicate()), device_mesh_2d)
                x = self.linear2(x)
                return x

        def input_fn():
            return torch.rand(512, dim, device="cuda")

        with torch.device("meta"):
            model = Model(dim)

        with AutoParallel(model, input_fn, device_mesh_2d) as autop:
            autop.add_input_constraints([(Shard(0), Replicate())])
            autop.add_output_constraints([(Shard(0), Replicate())])
            sharding_placement = autop.optimize_placement()

            # Verify the with_sharding_constraint node has correct 2D placement
            local_map_nodes = get_local_map_nodes(autop.gm.graph, is_backward=False)
            assert len(local_map_nodes) == 1, "Expected 1 forward local_map node"
            verify_local_map_placements(
                sharding_placement, local_map_nodes[0], (Shard(0), Replicate())
            )

            parallel_mod = autop.apply_placement(sharding_placement)

        assert parallel_mod is not None

    def test_with_sharding_constraint_multiple(self, device_mesh_1d):
        """Test multiple with_sharding_constraint calls in sequence."""
        dim = 128

        class Model(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.linear1 = nn.Linear(dim, dim, bias=False)
                self.linear2 = nn.Linear(dim, dim, bias=False)
                self.linear3 = nn.Linear(dim, dim, bias=False)

            def forward(self, x):
                x = self.linear1(x)
                x = with_sharding_constraint(x, (Shard(0),), device_mesh_1d)
                x = self.linear2(x)
                x = with_sharding_constraint(x, (Replicate(),), device_mesh_1d)
                x = self.linear3(x)
                x = with_sharding_constraint(x, (Shard(0),), device_mesh_1d)
                return x

        def input_fn():
            return torch.rand(512, dim, device="cuda")

        with torch.device("meta"):
            model = Model(dim)

        with AutoParallel(model, input_fn, device_mesh_1d) as autop:
            autop.add_input_constraints([(Shard(0),)])
            autop.add_output_constraints([(Shard(0),)])
            sharding_placement = autop.optimize_placement()

            # Verify all 3 with_sharding_constraint nodes have correct placements
            local_map_nodes = get_local_map_nodes(autop.gm.graph, is_backward=False)
            assert (
                len(local_map_nodes) == 3
            ), f"Expected 3 forward local_map nodes, got {len(local_map_nodes)}"

            # Nodes should be in order: Shard(0), Replicate(), Shard(0)
            expected_placements = [(Shard(0),), (Replicate(),), (Shard(0),)]
            for node, expected in zip(local_map_nodes, expected_placements):
                verify_local_map_placements(sharding_placement, node, expected)

            parallel_mod = autop.apply_placement(sharding_placement)

        assert parallel_mod is not None

    def test_with_sharding_constraint_no_mesh_outside_local_map_raises(self):
        """Test that with_sharding_constraint raises error when no mesh is available."""
        x = torch.rand(10, 10)
        with pytest.raises(RuntimeError, match="No device mesh is currently active"):
            with_sharding_constraint(x, (Shard(0),))


class TestPermutation:
    def test_shape_preserved(self):
        """Permutation should preserve tensor shape."""
        x = torch.randn(5, 10, 3)
        result = permutation(x, axis=0)
        assert result.shape == x.shape

        result = permutation(x, axis=1)
        assert result.shape == x.shape

        result = permutation(x, axis=2)
        assert result.shape == x.shape

    def test_elements_preserved(self):
        """Permutation should preserve all elements (just reordered)."""
        x = torch.arange(24).reshape(4, 6)
        result = permutation(x, axis=0)

        # Sort along axis and compare
        x_sorted = x.sort(dim=0).values
        result_sorted = result.sort(dim=0).values
        assert torch.equal(x_sorted, result_sorted)

    def test_elements_preserved_axis1(self):
        """Permutation along axis=1 should preserve all elements."""
        x = torch.arange(24).reshape(4, 6)
        result = permutation(x, axis=1)

        x_sorted = x.sort(dim=1).values
        result_sorted = result.sort(dim=1).values
        assert torch.equal(x_sorted, result_sorted)

    def test_independent_false_same_permutation(self):
        """With independent=False, the same permutation is applied to all slices."""
        torch.manual_seed(42)
        x = torch.arange(12).reshape(3, 4)
        result = permutation(x, axis=0, independent=False)

        # argsort gives the indices that would sort each column
        # If the same permutation is applied to all columns, all columns
        # should have identical argsort indices
        sort_indices = result.argsort(0)
        assert (sort_indices == sort_indices[:, :1]).all()

    def test_independent_true_different_permutations(self):
        """With independent=True, different permutations for each slice."""
        torch.manual_seed(42)
        # Use a larger tensor to make it very unlikely all columns get same permutation
        x = torch.arange(100).reshape(10, 10)
        result = permutation(x, axis=0, independent=True)

        # Elements should still be preserved per column
        for col in range(x.shape[1]):
            x_col_sorted = x[:, col].sort().values
            result_col_sorted = result[:, col].sort().values
            assert torch.equal(x_col_sorted, result_col_sorted)

        # With independent=True, at least some columns should have different orderings
        # Check by looking at the relative positions
        col0_order = result[:, 0].argsort()
        different_order_found = False
        for col in range(1, x.shape[1]):
            col_order = result[:, col].argsort()
            if not torch.equal(col0_order, col_order):
                different_order_found = True
                break
        assert (
            different_order_found
        ), "Expected different permutations for different columns"

    def test_1d_tensor(self):
        """Permutation works on 1D tensors."""
        x = torch.arange(10)
        result = permutation(x, axis=0)
        assert result.shape == x.shape
        assert set(result.tolist()) == set(x.tolist())

    def test_negative_axis(self):
        """Permutation works with negative axis."""
        x = torch.randn(3, 4, 5)
        result = permutation(x, axis=-1)
        assert result.shape == x.shape

    def test_device_preserved(self):
        """Result should be on same device as input."""
        x = torch.randn(5, 10)
        result = permutation(x)
        assert result.device == x.device

    def test_dtype_preserved(self):
        """Result should have same dtype as input."""
        for dtype in [torch.float32, torch.float64, torch.int64, torch.int32]:
            if dtype.is_floating_point:
                x = torch.randn(5, 10, dtype=dtype)
            else:
                x = torch.randint(0, 100, (5, 10), dtype=dtype)
            result = permutation(x)
            assert result.dtype == dtype

    def test_reproducibility_with_seed(self):
        """Same seed should produce same permutation."""
        x = torch.arange(20).reshape(4, 5)

        torch.manual_seed(123)
        result1 = permutation(x, axis=0)

        torch.manual_seed(123)
        result2 = permutation(x, axis=0)

        assert torch.equal(result1, result2)

    def test_independent_reproducibility(self):
        """Same seed should produce same result with independent=True."""
        x = torch.arange(20).reshape(4, 5)

        torch.manual_seed(456)
        result1 = permutation(x, axis=0, independent=True)

        torch.manual_seed(456)
        result2 = permutation(x, axis=0, independent=True)

        assert torch.equal(result1, result2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_tensor(self):
        """Permutation works on CUDA tensors."""
        x = torch.randn(5, 10, device="cuda")
        result = permutation(x, axis=0)
        assert result.device == x.device
        assert result.shape == x.shape
