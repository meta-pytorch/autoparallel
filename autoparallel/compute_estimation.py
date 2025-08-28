# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.utils._mode_utils import no_dispatch
from torch.utils._pytree import tree_flatten, tree_map_only
from torch.utils.flop_counter import FlopCounterMode, register_flop_formula


@register_flop_formula(torch.ops.aten.einsum, get_raw=True)
def einsum_flop(equation, tensors, out=None, **kwargs) -> int:
    # from torch.distributed.tensor._ops._einsum_strategy import EinsumDims
    assert len(tensors) == 2
    a_shape, b_shape = [x.shape for x in tensors]

    # parse einop equation and extract dims
    # TODO: generalize
    # input_dims, output_dim = EinsumDims.parse_equation(equation)
    # edims = EinsumDims.parse_dims(input_dims, output_dim)

    if len(a_shape) == 3 and len(b_shape) == 3:
        b, m, k = a_shape
        b1, n, k2 = b_shape
        assert b == b1
        assert m == n
        flop = (b * m) * k * k2 * 2
    elif len(a_shape) == 3 and len(b_shape) == 2:
        b, m, k = a_shape
        k2, n = b_shape
        assert k == k2
        flop = b * m * n * k * 2
    else:
        raise NotImplementedError(f"Unsupported einsum shapes: {a_shape} {b_shape}")
    return flop


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


def _shard_args_for_node(node, strategy, rand_init=False):
    args = tree_map_only(torch.fx.Node, lambda x: x.meta["val"], node.args)
    kwargs = tree_map_only(torch.fx.Node, lambda x: x.meta["val"], node.kwargs)

    # TODO: handle kwargs as well, for now we assume all tensors are
    # in args
    if len(kwargs) > 0:
        for k, v in kwargs.items():
            assert not isinstance(v, torch.Tensor), f"{node} {v}"
    args_sizes_strides = tuple(
        _get_sharded_shape_stride(spec) for spec in strategy.input_specs
    )

    flat_args, treespec = tree_flatten(args)
    new_flat_args = []
    counter = 0
    for x in flat_args:
        if isinstance(x, torch.Tensor):
            sizes, strides = args_sizes_strides[counter]
            x = torch.empty_strided(sizes, strides, device=x.device, dtype=x.dtype)
            if rand_init:
                if x.dtype.is_floating_point:
                    x.normal_()
                else:
                    x.random_(0, 256)
            counter += 1
        new_flat_args.append(x)
    args = treespec.unflatten(new_flat_args)
    return args, kwargs


def _has_zero_cost(node):
    if node.op != "call_function":
        return True

    if not isinstance(node.target, torch._ops.OpOverload):
        return True

    assert not isinstance(node.target, torch._ops.OpOverloadPacket), f"{node.target}"

    if node.target.is_view:
        return True

    return False


def _compute_flops(fn, *args, **kwargs):
    # TODO: maybe cache the flop_counter to avoid recreating it
    # all the time
    with FlopCounterMode(display=False) as flop_counter:
        out = fn(*args, **kwargs)
    return flop_counter.get_total_flops(), out


def estimate_strategy_runtime_cost(node, strategy):
    """
    This function estimates the runtime cost of a given strategy
    for a given node. It does this by computing the flop count
    and input-output memory cost of the node after sharding the
    inputs according to the strategy. It then uses the device
    specifications to estimate the runtime cost of the node.
    """
    if _has_zero_cost(node):
        return 0

    args, kwargs = _shard_args_for_node(node, strategy)

    flops, out = _compute_flops(node.target, *args, **kwargs)

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


def benchmark_strategy_runtime_cost(node, strategy):
    """
    This is the counterpart for estimate_strategy_runtime_cost
    but it actually runs the node on the GPU to get the runtime
    cost. This is useful for debugging the cost model and for
    cases where estimate_strategy_runtime_cost is not accurate.
    """
    if _has_zero_cost(node):
        return 0

    with unset_fake_temporarily(), no_dispatch():
        args, kwargs = _shard_args_for_node(node, strategy, rand_init=True)
        mean_op_time_us = benchmark_fn(node.target, *args, **kwargs)

    return mean_op_time_us


def benchmark_fn(fn, *args, **kwargs):
    n_warmup = 3
    for _ in range(n_warmup):
        fn(*args, **kwargs)

    num_iters = 10
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record(torch.cuda.current_stream())
    for _ in range(num_iters):
        fn(*args, **kwargs)
    end_event.record(torch.cuda.current_stream())
    torch.cuda.synchronize()
    total_op_time = start_event.elapsed_time(end_event)
    mean_op_time_ms = total_op_time / num_iters
    mean_op_time_us = mean_op_time_ms * 1e3
    return mean_op_time_us


def compare_estimated_with_benchmarked_throughput(
    graph, sharding_placement, tgt_ops="mm"
):
    if tgt_ops == "mm":
        tgt_ops = [torch.ops.aten.mm.default, torch.ops.aten.bmm.default]
    data = {}
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if tgt_ops is not None and node.target not in tgt_ops:
            continue
        strategy = sharding_placement[node]
        args, kwargs = _shard_args_for_node(node, strategy)
        flops = _compute_flops(node.target, *args, **kwargs)[0]
        real_t_us = benchmark_strategy_runtime_cost(node, strategy)  # us
        real_t_s = real_t_us / 1e6  # s
        throughput = flops / real_t_s * 10**-12  # TFLOPS / s

        dtype = strategy.input_specs[0].tensor_meta.dtype
        gpu_tflops = _get_device_tflops(dtype)
        efficiency = throughput / gpu_tflops

        est_t = estimate_strategy_runtime_cost(node, strategy)
        data[node] = (real_t_us, est_t, throughput, gpu_tflops, efficiency)
    return data
