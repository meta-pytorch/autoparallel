# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard

from autoparallel.api import auto_parallel


def test_auto_parallel_basic(device_mesh_1d):
    """Test basic auto_parallel usage with DTensor input."""
    dim = 128
    batch_size = 512
    local_batch_size = batch_size // device_mesh_1d.size()

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)

        def forward(self, x):
            return self.linear(x)

        def init_weights(self):
            nn.init.ones_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    with torch.device("meta"):
        model = Model(dim)

    # Create DTensor input with sharding
    x = DTensor.from_local(
        torch.rand(local_batch_size, dim, device="cuda"),
        device_mesh_1d,
        [Shard(0)],
    )

    parallel_model = auto_parallel(
        model,
        device_mesh_1d,
        sample_inputs=(x,),
        out_shardings=(Shard(0),),
    )

    # Verify model was created
    assert parallel_model is not None
    assert hasattr(parallel_model, "linear")

    # Initialize and verify
    parallel_model.to_empty(device="cuda")
    parallel_model.init_weights()

    assert torch.equal(
        parallel_model.get_parameter("linear.weight").full_tensor(),
        torch.ones(dim, dim, device="cuda"),
    )


def test_auto_parallel_tuple_inputs(device_mesh_1d):
    """Test auto_parallel with multiple DTensor inputs as tuple."""
    dim = 128
    batch_size = 512
    local_batch_size = batch_size // device_mesh_1d.size()

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear1 = nn.Linear(dim, dim)
            self.linear2 = nn.Linear(dim, dim)

        def forward(self, x, y):
            return self.linear1(x) + self.linear2(y)

    with torch.device("meta"):
        model = Model(dim)

    # Create DTensor inputs
    x = DTensor.from_local(
        torch.rand(local_batch_size, dim, device="cuda"),
        device_mesh_1d,
        [Shard(0)],
    )
    y = DTensor.from_local(
        torch.rand(local_batch_size, dim, device="cuda"),
        device_mesh_1d,
        [Shard(0)],
    )

    parallel_model = auto_parallel(
        model,
        device_mesh_1d,
        sample_inputs=(x, y),
        out_shardings=(Shard(0),),
    )

    assert parallel_model is not None


def test_auto_parallel_multiple_outputs(device_mesh_1d):
    """Test auto_parallel with multiple outputs and pytree out_shardings."""
    dim = 128
    batch_size = 512
    local_batch_size = batch_size // device_mesh_1d.size()

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear1 = nn.Linear(dim, dim, bias=True)
            self.linear2 = nn.Linear(dim, dim, bias=True)

        def forward(self, x):
            return self.linear1(x), self.linear2(x)

    with torch.device("meta"):
        model = Model(dim)

    x = DTensor.from_local(
        torch.rand(local_batch_size, dim, device="cuda"),
        device_mesh_1d,
        [Shard(0)],
    )

    # Pytree out_shardings matching tuple output
    parallel_model = auto_parallel(
        model,
        device_mesh_1d,
        sample_inputs=(x,),
        out_shardings=((Shard(0),), (Shard(0),)),
    )

    assert parallel_model is not None


def test_auto_parallel_replicated_input(device_mesh_1d):
    """Test auto_parallel with regular tensor (assumed Replicate)."""
    dim = 128
    batch_size = 512

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)

        def forward(self, x):
            return self.linear(x)

    with torch.device("meta"):
        model = Model(dim)

    # Regular tensor - will be assumed Replicate
    # Output is sharded so the optimizer can find a valid solution
    x = torch.rand(batch_size, dim, device="cuda")

    parallel_model = auto_parallel(
        model,
        device_mesh_1d,
        sample_inputs=(x,),
        out_shardings=(Shard(0),),  # Shard output for valid solution
    )

    assert parallel_model is not None


def test_auto_parallel_callable_inputs(device_mesh_1d):
    """Test auto_parallel with callable sample_inputs."""
    dim = 128
    batch_size = 512
    local_batch_size = batch_size // device_mesh_1d.size()

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)

        def forward(self, x):
            return self.linear(x)

    with torch.device("meta"):
        model = Model(dim)

    def sample_inputs():
        return (
            DTensor.from_local(
                torch.rand(local_batch_size, dim, device="cuda"),
                device_mesh_1d,
                [Shard(0)],
            ),
        )

    parallel_model = auto_parallel(
        model,
        device_mesh_1d,
        sample_inputs=sample_inputs,
        out_shardings=(Shard(0),),
    )

    assert parallel_model is not None


def test_auto_parallel_with_mp_policy(device_mesh_1d):
    """Test auto_parallel with mixed precision policy."""
    from torch.distributed.fsdp import MixedPrecisionPolicy

    dim = 128
    batch_size = 512
    local_batch_size = batch_size // device_mesh_1d.size()

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)

        def forward(self, x):
            return self.linear(x)

    with torch.device("meta"):
        model = Model(dim)

    x = DTensor.from_local(
        torch.rand(local_batch_size, dim, device="cuda"),
        device_mesh_1d,
        [Shard(0)],
    )

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )

    parallel_model = auto_parallel(
        model,
        device_mesh_1d,
        sample_inputs=(x,),
        out_shardings=(Shard(0),),
        mp_policy=mp_policy,
    )

    assert parallel_model is not None
