# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import nn
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel import AutoParallel, with_sharding_constraint
from autoparallel.collectives import local_map


@pytest.fixture(scope="module", autouse=True)
def init_pg():
    world_size = 64
    fake_store = FakeStore()
    if torch.distributed.is_initialized():
        return
    torch.distributed.init_process_group(
        "fake", store=fake_store, rank=0, world_size=world_size
    )


@pytest.fixture(scope="module")
def device_mesh_1d():
    world_size = torch.distributed.get_world_size()
    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda", (world_size,), mesh_dim_names=("dp",)
    )
    return mesh


@pytest.fixture(scope="module")
def device_mesh_2d():
    world_size = torch.distributed.get_world_size()
    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda", (world_size // 8, 8), mesh_dim_names=("dp", "tp")
    )
    return mesh


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
            parallel_mod = autop.apply_placement(sharding_placement)

        assert parallel_mod is not None

    def test_with_sharding_constraint_no_mesh_outside_local_map_raises(self):
        """Test that with_sharding_constraint raises error when no mesh is available."""
        import autoparallel.collectives as collectives

        original_mesh = collectives._local_map_device_mesh
        collectives._local_map_device_mesh = None

        try:
            x = torch.rand(10, 10)
            with pytest.raises(RuntimeError, match="No mesh found"):
                with_sharding_constraint(x, (Shard(0),))
        finally:
            # Restore original state
            collectives._local_map_device_mesh = original_mesh
