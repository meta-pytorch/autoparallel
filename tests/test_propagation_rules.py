# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from conftest import apply_cuda_patches
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import OpSpec, OpStrategy
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel.api import AutoParallel
from autoparallel.compile import autoparallel_backend
from autoparallel.shardings.placement_options import get_placement_options


def _replicated_meta_input(mesh, shape, dtype=torch.bfloat16):
    tensor = torch.empty(shape, dtype=dtype, device="meta")
    tensor_meta = TensorMeta(tensor.shape, tensor.stride(), tensor.dtype)
    spec = DTensorSpec(mesh, (Replicate(), Replicate()), tensor_meta=tensor_meta)
    return tensor, OpStrategy([OpSpec(spec)])


def _muse_cudnn_forward_options(mesh, bias_shape):
    query, query_strategy = _replicated_meta_input(mesh, (8, 32, 8192, 128))
    key, key_strategy = _replicated_meta_input(mesh, (8, 2, 8192, 128))
    value, value_strategy = _replicated_meta_input(mesh, (8, 2, 8192, 128))
    if bias_shape is None:
        bias = bias_strategy = None
    else:
        bias, bias_strategy = _replicated_meta_input(mesh, bias_shape, torch.bool)

    args = (query, key, value, bias, True, 0.0, False, False)
    specs = (
        query_strategy,
        key_strategy,
        value_strategy,
        bias_strategy,
        True,
        0.0,
        False,
        False,
    )
    return get_placement_options(
        mesh,
        torch.ops.aten._scaled_dot_product_cudnn_attention.default,
        specs,
        args,
        {"scale": 128**-0.5},
    )


def _find_muse_dp_tp_head_strategy(options):
    placement = (Shard(0), Shard(1))
    return [
        strategy
        for strategy in options.strategies
        if all(spec.placements == placement for spec in strategy.input_specs[:3])
    ]


@apply_cuda_patches
@pytest.mark.parametrize(
    "bias_shape,expected_bias_placement",
    [
        ((8, 1, 8192, 8192), (Shard(0), Replicate())),
        ((8, 32, 8192, 8192), (Shard(0), Shard(1))),
        ((8192, 8192), (Replicate(), Replicate())),
        (None, None),
    ],
)
def test_cudnn_sdpa_projects_score_sharding_onto_broadcast_bias(
    bias_shape, expected_bias_placement
):
    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda", (8, 2), mesh_dim_names=("dp", "tp")
    )
    matches = _find_muse_dp_tp_head_strategy(
        _muse_cudnn_forward_options(mesh, bias_shape)
    )

    assert len(matches) == 1
    strategy = matches[0]
    if expected_bias_placement is None:
        assert len(strategy.input_specs) == 3
    else:
        assert strategy.input_specs[3].placements == expected_bias_placement
    assert strategy.output_specs[0].placements == (Shard(0), Shard(1))


@apply_cuda_patches
def test_cudnn_sdpa_backward_projects_broadcast_bias():
    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda", (8, 2), mesh_dim_names=("dp", "tp")
    )
    input_shapes_and_dtypes = [
        ((8, 32, 8192, 128), torch.bfloat16),
        ((8, 32, 8192, 128), torch.bfloat16),
        ((8, 2, 8192, 128), torch.bfloat16),
        ((8, 2, 8192, 128), torch.bfloat16),
        ((8, 32, 8192, 128), torch.bfloat16),
        ((8, 32, 8192, 1), torch.float32),
        ((), torch.int64),
        ((), torch.int64),
        ((8, 1, 8192, 8192), torch.bool),
    ]
    inputs = [
        _replicated_meta_input(mesh, shape, dtype)
        for shape, dtype in input_shapes_and_dtypes
    ]
    args = tuple(tensor for tensor, _ in inputs) + (
        None,
        None,
        8192,
        8192,
        0.0,
        False,
    )
    specs = tuple(strategy for _, strategy in inputs) + (
        None,
        None,
        8192,
        8192,
        0.0,
        False,
    )
    options = get_placement_options(
        mesh,
        torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default,
        specs,
        args,
        {"scale": 128**-0.5},
    )

    matches = _find_muse_dp_tp_head_strategy(options)
    assert len(matches) == 1
    strategy = matches[0]
    assert all(
        spec.placements == (Shard(0), Shard(1)) for spec in strategy.input_specs[:6]
    )
    assert all(
        spec.placements == (Replicate(), Replicate())
        for spec in strategy.input_specs[6:8]
    )
    assert strategy.input_specs[8].placements == (Shard(0), Replicate())
    assert all(
        spec.placements == (Shard(0), Shard(1)) for spec in strategy.output_specs
    )


@apply_cuda_patches
def test_cudnn_sdpa_rejects_tp_larger_than_kv_heads():
    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda", (8, 4), mesh_dim_names=("dp", "tp")
    )
    options = _muse_cudnn_forward_options(mesh, (8, 1, 8192, 8192))

    assert not _find_muse_dp_tp_head_strategy(options)


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
    with AutoParallel(model, input_fn, device_mesh_1d, mp_policy) as autop:
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

    parallel_mod = torch.compile(parallel_mod, backend=autoparallel_backend())

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

    with AutoParallel(model, input_fn, device_mesh_1d, mp_policy) as autop:
        autop.add_input_constraints([(Shard(0),)])
        autop.add_output_constraints([(Shard(0),)])
        sharding_placement = autop.optimize_placement()
        parallel_mod = autop.apply_placement(sharding_placement)

    parallel_mod.to_empty(device="cuda")
    torch.nn.init.ones_(parallel_mod.embed.weight)

    parallel_mod = torch.compile(parallel_mod, backend=autoparallel_backend())

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

    with AutoParallel(model, input_fn, device_mesh_1d, mp_policy) as autop:
        autop.add_input_constraints([(Shard(0),)])
        sharding_placement = autop.optimize_placement()
        autop.apply_placement(sharding_placement)
