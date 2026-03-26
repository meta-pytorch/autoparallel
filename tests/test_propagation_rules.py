# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Shard

from autoparallel.api import AutoParallel


def test_permute_layernorm_stride_handling(device_mesh_1d):
    """Test that permute + layernorm handles non-contiguous to contiguous stride transitions.

    This test reproduces the stride mismatch bug in ConvNeXt-style architectures where:
    1. First permute creates a non-contiguous tensor (view) with stride (301056, 56, 1, 3136)
    2. LayerNorm receives non-contiguous input but returns a contiguous tensor
    3. Second permute creates another non-contiguous tensor (view)
    """

    class PermuteLayerNormNet(nn.Module):
        """Network with permute -> LayerNorm -> permute."""

        def __init__(self, channels):
            super().__init__()
            self.norm = nn.LayerNorm(channels, eps=1e-6)

        def forward(self, x):
            # (N, C, H, W) -> (N, H, W, C)
            x = x.permute(0, 2, 3, 1)
            # LayerNorm on last dim (C)
            x = self.norm(x)
            # (N, H, W, C) -> (N, C, H, W)
            x = x.permute(0, 3, 1, 2)
            return x

    batch_size = 256
    channels = 96
    height = 56
    width = 56

    def input_fn():
        return torch.rand(batch_size, channels, height, width, device="cuda")

    # Create model on meta device
    with torch.device("meta"):
        model = PermuteLayerNormNet(channels=channels)

    # Mixed precision policy
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32, reduce_dtype=torch.float32
    )

    # This should not raise an AssertionError about tensor_meta stride mismatch.
    with AutoParallel(
        model, input_fn, device_mesh_1d, mp_policy, compile=True
    ) as autop:
        x_sharding = (Shard(0),)
        y_sharding = (Shard(0),)

        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([y_sharding])

        sharding_placement = autop.optimize_placement()

        # Apply the optimized placement
        parallel_mod = autop.apply_placement(sharding_placement)

    # Initialize the parallel module
    parallel_mod.to_empty(device="cuda")

    for name, param in parallel_mod.named_parameters():
        if "weight" in name:
            torch.nn.init.ones_(param)
        elif "bias" in name:
            torch.nn.init.zeros_(param)

    # Test forward pass execution works
    local_batch_size = batch_size // torch.distributed.get_world_size()
    x_test = torch.rand(local_batch_size, channels, height, width, device="cuda")
    out = parallel_mod(x_test)

    # Verify output shape (should match input after permute -> norm -> permute)
    assert out.shape == (local_batch_size, channels, height, width)
    # Output may be non-contiguous due to final permute (view operation)

    # Verify forward execution produces correct output
    assert out.abs().sum() > 0


def test_iota(device_mesh_1d):
    """End-to-end test: model with torch.arange (decomposes to prims.iota)."""
    seq_len = 256
    dim = 64

    class ArangeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(seq_len, dim)

        def forward(self, x):
            positions = torch.arange(x.shape[1], device=x.device)
            return x + self.embed(positions)

    batch_size = 256

    def input_fn():
        return torch.rand(batch_size, seq_len, dim, device="cuda")

    with torch.device("meta"):
        model = ArangeModel()

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32, reduce_dtype=torch.float32
    )

    with AutoParallel(
        model, input_fn, device_mesh_1d, mp_policy, compile=True
    ) as autop:
        autop.add_input_constraints([(Shard(0),)])
        autop.add_output_constraints([(Shard(0),)])
        sharding_placement = autop.optimize_placement()
        parallel_mod = autop.apply_placement(sharding_placement)

    parallel_mod.to_empty(device="cuda")
    torch.nn.init.ones_(parallel_mod.embed.weight)

    local_batch = batch_size // torch.distributed.get_world_size()
    x = torch.ones(local_batch, seq_len, dim, device="cuda")
    out = parallel_mod(x)

    assert out.shape == (local_batch, seq_len, dim)

    # embed(positions) with all-ones weight gives all-ones, plus all-ones input = 2.0
    assert torch.allclose(out, torch.full_like(out, 2.0))

    out.sum().backward()
    assert parallel_mod.embed.weight.grad is not None


def test_index_put(device_mesh_1d):
    """Test that aten.index_put with List[Tensor] args works through the solver.

    Advanced indexing (e.g. `out[:, idx] = x[:, idx]`) decomposes into
    aten.index_put, whose `indices` argument is List[Optional[Tensor]].
    In autoparallel's placement_options.py, list-of-OpStrategy args become
    TupleStrategy. The _try_single_dim_strategy path must unwrap these
    TupleStrategy children into DTensorSpecs when computing tensor meta,
    otherwise _propagate_tensor_meta_non_cached sees TupleStrategy where
    it expects a list of tensors.
    """

    class IndexPutModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(32, 64))

        def forward(self, x):
            out = torch.zeros_like(x)
            idx = torch.arange(x.shape[1], device=x.device)
            out[:, idx] = x[:, idx]
            return out @ self.weight

    batch_size = 256

    def input_fn():
        return torch.randn(batch_size, 32, device="cuda")

    with torch.device("meta"):
        model = IndexPutModel()

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )

    with AutoParallel(
        model, input_fn, device_mesh_1d, mp_policy, compile=True
    ) as autop:
        autop.add_input_constraints([(Shard(0),)])
        sharding_placement = autop.optimize_placement()
        autop.apply_placement(sharding_placement)
