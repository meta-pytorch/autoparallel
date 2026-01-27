# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for communication cost estimation module."""

import unittest

import torch
from torch.distributed.tensor import DeviceMesh, Replicate, Shard
from torch.distributed.tensor._collective_utils import MeshTopoInfo
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel.cost_models.collective_runtime_estimation import (
    all_to_all_cost,
    estimate_strategy_comms_cost,
    redistribute_cost,
)


def extract_tensor_meta(t: torch.Tensor) -> TensorMeta:
    return TensorMeta(t.shape, t.stride(), t.dtype)


class TestCollectiveRuntimeEstimation(unittest.TestCase):
    """Test communication cost estimation functions."""

    @classmethod
    def setUpClass(cls):
        """Set up fake distributed environment."""
        cls.world_size = 4
        store = FakeStore()
        torch.distributed.init_process_group(
            backend="fake",
            rank=0,
            world_size=cls.world_size,
            store=store,
        )
        cls.mesh = DeviceMesh("cuda", list(range(cls.world_size)))

    @classmethod
    def tearDownClass(cls):
        """Tear down distributed environment."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def test_redistribute_cost_basic(self):
        """Test basic redistribute cost estimation."""
        shard_placement = (Shard(0),)
        replica_placement = (Replicate(),)

        tensor = torch.randn(8, 8)
        tensor_meta = extract_tensor_meta(tensor)

        shard_spec = DTensorSpec(self.mesh, shard_placement, tensor_meta)
        replica_spec = DTensorSpec(self.mesh, replica_placement, tensor_meta)

        # Same spec should have zero cost
        cost_same = redistribute_cost(shard_spec, shard_spec)
        self.assertEqual(cost_same, 0)

        # Shard -> Replicate should have positive cost
        cost_allgather = redistribute_cost(shard_spec, replica_spec)
        self.assertGreater(cost_allgather, 0)

    def test_redistribute_cost_with_compute_overhead(self):
        """Test that non-dim-0 shards include compute overhead."""
        shard0_placement = (Shard(0),)
        shard1_placement = (Shard(1),)
        replica_placement = (Replicate(),)

        tensor = torch.randn(8, 8)
        tensor_meta = extract_tensor_meta(tensor)

        shard0_spec = DTensorSpec(self.mesh, shard0_placement, tensor_meta)
        shard1_spec = DTensorSpec(self.mesh, shard1_placement, tensor_meta)
        replica_spec = DTensorSpec(self.mesh, replica_placement, tensor_meta)

        # Shard(0) -> Replicate (no reshuffle needed)
        cost_dim0 = redistribute_cost(shard0_spec, replica_spec)
        # Shard(1) -> Replicate (reshuffle needed)
        cost_dim1 = redistribute_cost(shard1_spec, replica_spec)

        # Shard(1) -> Replicate should be more expensive due to reshuffle
        self.assertGreater(cost_dim1, cost_dim0)

    def test_all_to_all_cost(self):
        """Test all_to_all_cost function."""
        mesh_topo = MeshTopoInfo.build_from_mesh(self.mesh)

        # Test with 1MB
        cost = all_to_all_cost(0.001, mesh_topo, 0)
        self.assertGreater(cost, 0)

        # Larger tensor should have higher cost
        cost_larger = all_to_all_cost(0.01, mesh_topo, 0)
        self.assertGreater(cost_larger, cost)

    def test_shard_to_shard_uses_all_to_all(self):
        """Test that shard->shard transitions have reasonable cost."""
        shard0_placement = (Shard(0),)
        shard1_placement = (Shard(1),)

        tensor = torch.randn(8, 8)
        tensor_meta = extract_tensor_meta(tensor)

        shard0_spec = DTensorSpec(self.mesh, shard0_placement, tensor_meta)
        shard1_spec = DTensorSpec(self.mesh, shard1_placement, tensor_meta)

        # Shard(0) -> Shard(1) should use all_to_all
        cost = redistribute_cost(shard0_spec, shard1_spec)
        self.assertGreater(cost, 0)
        self.assertNotEqual(cost, float("inf"))

    def test_estimate_strategy_comms_cost(self):
        """Test estimate_strategy_comms_cost wrapper."""
        shard_placement = (Shard(0),)
        replica_placement = (Replicate(),)

        tensor = torch.randn(8, 8)
        tensor_meta = extract_tensor_meta(tensor)

        shard_spec = DTensorSpec(self.mesh, shard_placement, tensor_meta)
        replica_spec = DTensorSpec(self.mesh, replica_placement, tensor_meta)

        cost = estimate_strategy_comms_cost(shard_spec, replica_spec)
        expected_cost = redistribute_cost(shard_spec, replica_spec)
        self.assertEqual(cost, expected_cost)

    def test_order_parameter_deprecated(self):
        """Test that order parameter is accepted but ignored."""
        shard_placement = (Shard(0),)
        replica_placement = (Replicate(),)

        tensor = torch.randn(8, 8)
        tensor_meta = extract_tensor_meta(tensor)

        shard_spec = DTensorSpec(self.mesh, shard_placement, tensor_meta)
        replica_spec = DTensorSpec(self.mesh, replica_placement, tensor_meta)

        # Should accept order parameter without error
        cost_with_order = redistribute_cost(shard_spec, replica_spec, order=[0])
        cost_without_order = redistribute_cost(shard_spec, replica_spec)
        # Results should be the same (order is ignored)
        self.assertEqual(cost_with_order, cost_without_order)


if __name__ == "__main__":
    unittest.main()
