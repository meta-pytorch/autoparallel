from __future__ import annotations

import json

import torch
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import OpSpec, OpStrategy
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel.shardings.placement_options import get_placement_options


def _replicated_meta_input(mesh, shape, dtype=torch.bfloat16):
    tensor = torch.empty(shape, dtype=dtype, device="meta")
    tensor_meta = TensorMeta(tensor.shape, tensor.stride(), tensor.dtype)
    spec = DTensorSpec(mesh, (Replicate(), Replicate()), tensor_meta=tensor_meta)
    return tensor, OpStrategy([OpSpec(spec)])


def _forward_options(mesh, *, batch_size: int, bias_shape):
    query, query_strategy = _replicated_meta_input(mesh, (batch_size, 32, 4096, 128))
    key, key_strategy = _replicated_meta_input(mesh, (batch_size, 2, 4096, 128))
    value, value_strategy = _replicated_meta_input(mesh, (batch_size, 2, 4096, 128))
    if bias_shape is None:
        bias = bias_strategy = None
    else:
        bias, bias_strategy = _replicated_meta_input(mesh, bias_shape, torch.bool)
    return get_placement_options(
        mesh,
        torch.ops.aten._scaled_dot_product_cudnn_attention.default,
        (
            query_strategy,
            key_strategy,
            value_strategy,
            bias_strategy,
            True,
            0.0,
            False,
            False,
        ),
        (query, key, value, bias, True, 0.0, False, False),
        {"scale": 128**-0.5},
    )


def _head_sharded_matches(options):
    placement = (Shard(0), Shard(1))
    return [
        strategy
        for strategy in options.strategies
        if all(spec.placements == placement for spec in strategy.input_specs[:3])
    ]


def _validate_forward(mesh, *, batch_size: int, bias_shape, expected_bias):
    options = _forward_options(mesh, batch_size=batch_size, bias_shape=bias_shape)
    matches = _head_sharded_matches(options)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one head-sharded strategy, got {len(matches)} of "
            f"{len(options.strategies)}"
        )
    strategy = matches[0]
    if expected_bias is None:
        if len(strategy.input_specs) != 3:
            raise RuntimeError("no-bias strategy unexpectedly has a bias input spec")
    elif strategy.input_specs[3].placements != expected_bias:
        raise RuntimeError(
            f"unexpected bias placement: {strategy.input_specs[3].placements}"
        )
    if strategy.output_specs[0].placements != (Shard(0), Shard(1)):
        raise RuntimeError("unexpected forward output placement")
    return {
        "strategy_count": len(options.strategies),
        "bias_placement": None
        if expected_bias is None
        else repr(strategy.input_specs[3].placements),
        "output_placement": repr(strategy.output_specs[0].placements),
    }


def _validate_backward(mesh):
    shapes_and_dtypes = [
        ((16, 32, 4096, 128), torch.bfloat16),
        ((16, 32, 4096, 128), torch.bfloat16),
        ((16, 2, 4096, 128), torch.bfloat16),
        ((16, 2, 4096, 128), torch.bfloat16),
        ((16, 32, 4096, 128), torch.bfloat16),
        ((16, 32, 4096, 1), torch.float32),
        ((), torch.int64),
        ((), torch.int64),
        ((16, 1, 4096, 4096), torch.bool),
    ]
    inputs = [
        _replicated_meta_input(mesh, shape, dtype) for shape, dtype in shapes_and_dtypes
    ]
    args = tuple(tensor for tensor, _ in inputs) + (
        None,
        None,
        4096,
        4096,
        0.0,
        False,
    )
    specs = tuple(strategy for _, strategy in inputs) + (
        None,
        None,
        4096,
        4096,
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
    matches = _head_sharded_matches(options)
    if len(matches) != 1:
        raise RuntimeError("expected one backward head-sharded strategy")
    strategy = matches[0]
    expected_head = (Shard(0), Shard(1))
    expected_replicate = (Replicate(), Replicate())
    if not all(spec.placements == expected_head for spec in strategy.input_specs[:6]):
        raise RuntimeError("unexpected backward activation placements")
    if not all(
        spec.placements == expected_replicate for spec in strategy.input_specs[6:8]
    ):
        raise RuntimeError("unexpected Philox placements")
    if strategy.input_specs[8].placements != (Shard(0), Replicate()):
        raise RuntimeError("unexpected backward bias placement")
    if not all(spec.placements == expected_head for spec in strategy.output_specs):
        raise RuntimeError("unexpected backward output placements")
    return {
        "strategy_count": len(options.strategies),
        "bias_placement": repr(strategy.input_specs[8].placements),
        "output_placements": [repr(spec.placements) for spec in strategy.output_specs],
    }


def main() -> None:
    torch.distributed.init_process_group(
        "fake", store=FakeStore(), rank=0, world_size=16
    )
    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda", (8, 2), mesh_dim_names=("dp", "tp")
    )
    results = {
        "torch": torch.__version__,
        "torch_commit": torch.version.git_version,
        "mesh": [8, 2],
        "per_gpu_bs1_broadcast_bias": _validate_forward(
            mesh,
            batch_size=16,
            bias_shape=(16, 1, 4096, 4096),
            expected_bias=(Shard(0), Replicate()),
        ),
        "full_head_bias": _validate_forward(
            mesh,
            batch_size=16,
            bias_shape=(16, 32, 4096, 4096),
            expected_bias=(Shard(0), Shard(1)),
        ),
        "rank_two_bias": _validate_forward(
            mesh,
            batch_size=16,
            bias_shape=(4096, 4096),
            expected_bias=(Replicate(), Replicate()),
        ),
        "no_bias": _validate_forward(
            mesh,
            batch_size=16,
            bias_shape=None,
            expected_bias=None,
        ),
        "backward": _validate_backward(mesh),
    }
    tp4_mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda", (4, 4), mesh_dim_names=("dp", "tp")
    )
    tp4_options = _forward_options(
        tp4_mesh,
        batch_size=16,
        bias_shape=(16, 1, 4096, 4096),
    )
    if _head_sharded_matches(tp4_options):
        raise RuntimeError("TP4 must be rejected because Muse has two KV heads")
    results["tp4_rejected"] = True
    results["status"] = "passed"
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
