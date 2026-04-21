# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import OpSchema, OpSpec, OpStrategy
from torch.distributed.tensor.placement_types import Replicate, Shard, _StridedShard

from autoparallel.api import AutoParallel
from autoparallel.shardings import dtensor_sharding_helpers
from autoparallel.shardings.dtensor_sharding_helpers import get_op_strategy


@pytest.fixture
def enable_single_dim_mm_family(monkeypatch):
    """Opt-in toggle: route mm-family ops through upstream single-dim path."""
    monkeypatch.setattr(dtensor_sharding_helpers, "ENABLE_SINGLE_DIM_MM_FAMILY", True)
    yield


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


def _mk_input_strategy(mesh, shape, placements):
    meta = TensorMeta(
        shape=torch.Size(shape),
        stride=(1,) * len(shape),
        dtype=torch.float32,
    )
    spec = DTensorSpec(mesh=mesh, placements=tuple(placements), tensor_meta=meta)
    return OpStrategy([OpSpec(output_specs=spec, input_specs=(spec,))])


def test_mm_strategy_enumerates_strided_shard(
    device_mesh_2d, enable_single_dim_mm_family
):
    """mm with a _StridedShard-bearing input must yield strategies that carry
    _StridedShard on the output. This is the capability that lets AP represent
    batch-on-mesh0 + seq-on-mesh1 through a view -> mm -> view decomposition
    without the einsum rewrite (see PLAN_dtensor_native_linear.md Phase 1).
    """
    mesh = device_mesh_2d
    split_factor = 8

    flat_in = _mk_input_strategy(
        mesh,
        [32 * split_factor, 16],
        [Shard(0), _StridedShard(0, split_factor=split_factor)],
    )
    weight = _mk_input_strategy(mesh, [16, 32], [Replicate(), Replicate()])

    schema = OpSchema(
        torch.ops.aten.mm.default,
        args_schema=(flat_in, weight),
        kwargs_schema={},
    )

    result = get_op_strategy(torch.ops.aten.mm.default, schema)

    strided_out_count = 0
    matched_sf = False
    for op_spec in result.strategies:
        for p in op_spec.output_spec.placements:
            if isinstance(p, _StridedShard):
                strided_out_count += 1
                if p.split_factor == split_factor:
                    matched_sf = True
                break

    assert strided_out_count > 0, (
        "Expected at least one mm strategy with _StridedShard on output; "
        f"got {len(result.strategies)} strategies, none strided. "
        "AP did not reach the single-dim path."
    )
    assert matched_sf, (
        f"Expected a _StridedShard(sf={split_factor}) variant matching the "
        "upstream input. Placeholder expansion is not propagating the "
        "split_factor from input strategies."
    )


def test_mm_strategy_plain_shard_still_present(
    device_mesh_2d, enable_single_dim_mm_family
):
    """Regression: enabling _StridedShard variants must not drop the plain
    Shard strategies. The solver still needs those for cases where the upstream
    chain hasn't introduced any _StridedShard.
    """
    mesh = device_mesh_2d

    lhs = _mk_input_strategy(mesh, [256, 16], [Shard(0), Replicate()])
    rhs = _mk_input_strategy(mesh, [16, 32], [Replicate(), Replicate()])

    schema = OpSchema(
        torch.ops.aten.mm.default,
        args_schema=(lhs, rhs),
        kwargs_schema={},
    )
    result = get_op_strategy(torch.ops.aten.mm.default, schema)

    has_plain_shard = any(
        any(
            isinstance(p, Shard) and not isinstance(p, _StridedShard)
            for p in s.output_spec.placements
        )
        for s in result.strategies
    )
    assert has_plain_shard, (
        "Expected at least one plain-Shard output strategy for mm with "
        "non-strided inputs."
    )


def test_mm_strategy_backward_grad_weight_strided(
    device_mesh_2d, enable_single_dim_mm_family
):
    """Backward grad-weight mm form: grad_out @ input where both operands
    carry _StridedShard on the contracting dim (the flattened batch*seq).

    Pattern in the autograd-generated backward:
        grad_out: [B, S, N] -> view -> [B*S, N] -> permute -> [N, B*S]
        input   : [B, S, K] -> view -> [B*S, K]
        mm(permuted_grad_out, flat_input) -> [N, K]

    If both inputs carry _StridedShard on the contracting dim (flat M),
    the mm strategy should produce at least one strategy where both inputs
    are _StridedShard on the contracting dim and the output is Partial
    (the usual contracting-dim pattern, specialized with split_factor).
    """
    mesh = device_mesh_2d
    split_factor = 8
    flat_m = 32 * split_factor

    # grad_out after permute: [N, M] with _StridedShard(1, sf) on M
    grad_out_p = _mk_input_strategy(
        mesh,
        [32, flat_m],
        [Shard(1), _StridedShard(1, split_factor=split_factor)],
    )
    # input after flatten: [M, K] with _StridedShard(0, sf) on M
    flat_input = _mk_input_strategy(
        mesh,
        [flat_m, 16],
        [Shard(0), _StridedShard(0, split_factor=split_factor)],
    )

    schema = OpSchema(
        torch.ops.aten.mm.default,
        args_schema=(grad_out_p, flat_input),
        kwargs_schema={},
    )
    result = get_op_strategy(torch.ops.aten.mm.default, schema)

    matched = False
    for op_spec in result.strategies:
        in1 = op_spec.input_specs[0].placements
        in2 = op_spec.input_specs[1].placements
        out = op_spec.output_spec.placements
        # Contracting-dim strided pair produces Partial output.
        has_strided_in1 = any(
            isinstance(p, _StridedShard) and p.split_factor == split_factor for p in in1
        )
        has_strided_in2 = any(
            isinstance(p, _StridedShard) and p.split_factor == split_factor for p in in2
        )
        has_partial = any(p.is_partial() for p in out)
        if has_strided_in1 and has_strided_in2 and has_partial:
            matched = True
            break

    assert matched, (
        "Expected at least one backward-mm strategy with _StridedShard on "
        f"both contracting inputs (sf={split_factor}) and Partial output. "
        "Phase 1 extension is not propagating _StridedShard through the "
        "contracting-dim pattern."
    )
