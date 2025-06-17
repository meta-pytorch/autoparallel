# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.utils._pytree import tree_map_only
from torch.utils.flop_counter import FlopCounterMode


def _get_device_tflops(dtype):
    # for some reason the function from PyTorch is giving
    # wildly different TFlops compared to the specs. I'm
    # using had-coded values for now that I pulled from xFormers
    # https://github.com/fairinternal/xformers/blob/main/xformers/profiler/device_limits.py
    # TODO: fix PyTorch's implementation
    # from torch._inductor.utils import get_device_tflops

    device = None
    device_name = torch.cuda.get_device_name(device)
    assert "H100" in device_name, f"Only H100 supported from now, got {device_name}"

    return {
        torch.float64: 67,
        # NOTE: NVIDIA gives all numbers "with 2:4 sparsity"
        # but we want the full GEMM numbers
        torch.float32: 989 // 2,
        torch.float16: 1979 // 2,
        torch.bfloat16: 1979 // 2,
        torch.int8: 3958 // 2,
    }[dtype]


def _get_sharded_shape(spec):
    mesh = spec.mesh
    tensor_shape = spec.tensor_meta.shape
    # TODO: take dtype into account as well
    # tensor_dtype = spec.tensor_meta.dtype
    placements = spec.placements
    # TODO: find a better heuristic other than
    # running DTensor
    new_tensor_shape = list(tensor_shape)
    for mesh_size, placement in zip(mesh.shape, placements):
        if placement.is_shard():
            dim = placement.dim
            new_tensor_shape[dim] = (new_tensor_shape[dim] + mesh_size - 1) // mesh_size
    return new_tensor_shape


def estimate_strategy_runtime_cost(node, strategy):
    if node.op != "call_function":
        return 0
    # suppose only matmul-like ops
    if not isinstance(node.target, torch._ops.OpOverload):
        return 0

    assert not isinstance(node.target, torch._ops.OpOverloadPacket), f"{node.target}"

    if node.target.is_view:
        return 0

    args = tree_map_only(torch.fx.Node, lambda x: x.meta["val"], node.args)
    kwargs = tree_map_only(torch.fx.Node, lambda x: x.meta["val"], node.kwargs)

    fake_mode = torch._guards.detect_fake_mode(args)

    if len(kwargs) > 0:
        for k, v in kwargs.items():
            assert not isinstance(v, torch.Tensor), f"{node} {v}"
    args_shapes = tuple(_get_sharded_shape(spec) for spec in strategy.input_specs)

    counter = 0
    args = list(args)
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            with fake_mode:
                args[i] = torch.empty(
                    args_shapes[counter], device=arg.device, dtype=arg.dtype
                )
            counter += 1

    # TODO: maybe cache the flop_counter to avoid recreating it
    # all the time
    with FlopCounterMode(display=False) as flop_counter:
        node.target(*args, **kwargs)

    flops = flop_counter.get_total_flops()

    # TODO: fix this
    dtype = strategy.input_specs[0].tensor_meta.dtype

    # TODO: better handle this case
    if dtype.is_complex:
        return 0
    # TODO: use PyTorch's version once it's giving correct results
    gpu_flops = _get_device_tflops(dtype) * 10**12

    # suppose 50% efficiency for the operator
    factor = 1 / 0.5
    compute_time = factor * flops / gpu_flops * 1e6  # us

    return compute_time
