# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch.distributed.tensor._collective_utils import redistribute_cost
from torch.utils._pytree import tree_flatten, tree_map_only
from torch.utils.flop_counter import FlopCounterMode


@dataclass
class DeviceLimit:
    """GPU device specifications for compute estimation.

    Attributes:
        name: Device name (e.g., "H100", "A100")
        ref: URL reference to official datasheet
        sm: Compute capability version (major, minor)
        gmem_bandwidth: Global memory bandwidth in bytes/second
        gemm_tflops: GEMM throughput in TFLOPS for different data types
    """

    name: str
    ref: str
    sm: Tuple[int, int]
    gmem_bandwidth: float
    gemm_tflops: Dict[torch.dtype, float]


# For f32, we assume we can use tf32
DEVICE_LIMITS: Tuple[DeviceLimit, ...] = (
    DeviceLimit(
        "H100",
        "https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet",
        sm=(9, 0),
        gmem_bandwidth=3.35 * (1024**4),  # NOTE: PCIe is 2 TB/s
        gemm_tflops={
            torch.float64: 67,
            # NOTE: NVIDIA gives all numbers "with 2:4 sparsity"
            # but we want the full GEMM numbers
            torch.float32: 989 // 2,
            torch.float16: 1979 // 2,
            torch.bfloat16: 1979 // 2,
            torch.int8: 3958 // 2,
        },
    ),
    DeviceLimit(
        "A100",
        "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf",
        sm=(8, 0),
        gmem_bandwidth=2 * (1024**4),  # NOTE: PCIe is 1.5 TB/s
        gemm_tflops={
            torch.float64: 19.5,
            torch.float32: 156,
            torch.float16: 312,
            torch.bfloat16: 312,
            torch.int8: 624,
        },
    ),
    DeviceLimit(
        "A30",
        "https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/products/a30-gpu/pdf/a30-datasheet.pdf",
        sm=(8, 0),
        gmem_bandwidth=933 * (1024**3),
        gemm_tflops={
            torch.float64: 10.3,
            torch.float32: 82,
            torch.float16: 165,
            torch.bfloat16: 165,
            torch.int8: 330,
        },
    ),
    DeviceLimit(
        "A10G",
        "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a10/pdf/a10-datasheet.pdf",
        sm=(8, 0),
        gmem_bandwidth=933 * (1024**3),
        gemm_tflops={
            torch.float32: 31.2,
            torch.float16: 125,
            torch.bfloat16: 125,
            torch.int8: 250,
        },
    ),
    DeviceLimit(
        "T4",
        "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf",
        sm=(7, 5),
        gmem_bandwidth=300 * (1024**3),
        gemm_tflops={
            torch.float32: 8.1,
            torch.float16: 65,
            torch.int8: 130,
        },
    ),
    # Assuming SXM2
    DeviceLimit(
        "V100",
        "https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf",
        sm=(7, 0),
        gmem_bandwidth=900 * (1024**3),
        gemm_tflops={
            torch.float64: 7.8,
            torch.float32: 15.7,
            torch.float16: 125,
        },
    ),
    DeviceLimit(
        "P100",
        "https://images.nvidia.com/content/tesla/pdf/nvidia-tesla-p100-datasheet.pdf",
        sm=(6, 0),
        gmem_bandwidth=732 * (1024**3),
        gemm_tflops={
            torch.float64: 5.3,
            torch.float32: 10.6,
            torch.float16: 21.2,
        },
    ),
)


def _get_device_limit():
    device = None
    device_name = torch.cuda.get_device_name(device)

    # Find matching device limit
    device_limit = None
    for limit in DEVICE_LIMITS:
        if limit.name in device_name or (
            limit.name == "A100" and "PG509" in device_name
        ):
            device_limit = limit
            break

    if device_limit is None:
        raise ValueError(
            f"Unsupported device: {device_name}. Supported devices: {[limit.name for limit in DEVICE_LIMITS]}"
        )
    return device_limit


def _get_device_tflops(dtype):
    # for some reason the function from PyTorch is giving
    # wildly different TFlops compared to the specs. I'm
    # using hard-coded values for now that I pulled from xFormers
    # https://github.com/fairinternal/xformers/blob/main/xformers/profiler/device_limits.py
    # TODO: fix PyTorch's implementation
    # from torch._inductor.utils import get_device_tflops

    device_limit = _get_device_limit()
    if dtype not in device_limit.gemm_tflops:
        raise ValueError(
            f"Dtype {dtype} not supported on {device_limit.name}. Supported dtypes: {list(device_limit.gemm_tflops.keys())}"
        )

    return device_limit.gemm_tflops[dtype]


def _get_device_gmem_bandwidth():
    device_limit = _get_device_limit()
    return device_limit.gmem_bandwidth


def _get_sharded_shape_stride(spec):
    mesh = spec.mesh
    tensor_shape = spec.tensor_meta.shape
    # TODO: take dtype into account as well
    # tensor_dtype = spec.tensor_meta.dtype
    placements = spec.placements
    # TODO: find a better heuristic other than
    # running DTensor
    new_tensor_shape = list(tensor_shape)
    new_tensor_stride = list(spec.tensor_meta.stride)
    for mesh_size, placement in zip(mesh.shape, placements):
        if placement.is_shard():
            dim = placement.dim
            new_tensor_shape[dim] = (new_tensor_shape[dim] + mesh_size - 1) // mesh_size
            if dim - 1 > 0:
                new_tensor_stride[dim - 1] = (
                    new_tensor_stride[dim - 1] + mesh_size - 1
                ) // mesh_size
    return new_tensor_shape, new_tensor_stride


def compute_memory_cost(op, args, outs):
    def tensor_bytes(data):
        return [
            x.numel() * x.element_size()
            for x in tree_flatten(data)[0]
            if isinstance(x, torch.Tensor)
        ]

    read_bytes = sum(tensor_bytes(args))
    write_bytes = sum(tensor_bytes(outs))
    return read_bytes + write_bytes


def estimate_strategy_comms_cost(src_spec, tgt_spec):
    # TODO: need to use optimal redistribution cost instead
    comms_cost = redistribute_cost(src_spec, tgt_spec)
    compute_cost = 0

    src_sizes, _ = _get_sharded_shape_stride(src_spec)
    tgt_sizes, _ = _get_sharded_shape_stride(tgt_spec)

    gpu_memory_bandwidth = _get_device_gmem_bandwidth()

    for src_plc, tgt_plc in zip(src_spec.placements, tgt_spec.placements):
        if src_plc.is_partial() and tgt_plc.is_shard() and tgt_plc.dim != 0:
            # add cost of additional cat on full size
            # *2 because we need to count input and output reads
            read_write_bytes = (
                math.prod(src_sizes) * 2 * src_spec.tensor_meta.dtype.itemsize
            )
            compute_cost += read_write_bytes / gpu_memory_bandwidth * 1e6  # us
        elif src_plc.is_shard() and src_plc.dim != 0 and tgt_plc.is_replicate():
            # add cost of additional cat on full size
            # *2 because we need to count input and output reads
            read_write_bytes = (
                math.prod(tgt_sizes) * 2 * tgt_spec.tensor_meta.dtype.itemsize
            )
            compute_cost += read_write_bytes / gpu_memory_bandwidth * 1e6  # us
        elif src_plc.is_replicate() and tgt_plc.is_partial():
            # forbit  R -> P case as this doesn't make sense for us
            compute_cost += math.inf

    return comms_cost + compute_cost


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
    args_sizes_strides = tuple(
        _get_sharded_shape_stride(spec) for spec in strategy.input_specs
    )

    counter = 0
    args = list(args)
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            with fake_mode:
                sizes, strides = args_sizes_strides[counter]
                args[i] = torch.empty_strided(
                    sizes, strides, device=arg.device, dtype=arg.dtype
                )
            counter += 1

    # TODO: maybe cache the flop_counter to avoid recreating it
    # all the time
    with FlopCounterMode(display=False) as flop_counter:
        out = node.target(*args, **kwargs)

    flops = flop_counter.get_total_flops()

    read_write_bytes = compute_memory_cost(node.target, args, out)
    gpu_memory_bandwidth = _get_device_gmem_bandwidth()
    read_write_time = read_write_bytes / gpu_memory_bandwidth * 1e6  # us

    # TODO: fix this
    dtype = strategy.input_specs[0].tensor_meta.dtype

    # TODO: better handle this case
    if dtype.is_complex:
        return read_write_time
    # TODO: use PyTorch's version once it's giving correct results
    gpu_flops = _get_device_tflops(dtype) * 10**12

    # suppose 50% efficiency for the operator
    factor = 1 / 0.5
    compute_time = factor * flops / gpu_flops * 1e6  # us

    return max(compute_time, read_write_time)
