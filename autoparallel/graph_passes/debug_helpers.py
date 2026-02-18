# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import json
import re
from contextlib import ExitStack
from typing import Any, Callable

import torch
from torch._functorch.aot_autograd import aot_export_joint_with_descriptors
from torch._inductor.fx_passes.bucketing import is_wait_tensor
from torch._inductor.fx_passes.overlap_scheduling import (
    get_group_name,
    schedule_overlap_bucketing,
)
from torch.utils._dtype_abbrs import dtype_abbrs

from autoparallel.cost_models.collective_runtime_estimation import (
    MeshTopoInfo,
    allgather_cost,
    allreduce_cost,
    reduce_scatter_cost,
)
from autoparallel.cost_models.compute_estimation import estimate_strategy_runtime_cost


def parse_tensor_annotation(annotation: str) -> torch.Tensor:
    """
    Parse a tensor annotation string and create a PyTorch tensor.

    Format: dtype[shape][strides]device
    Example: f32[384][1]cuda:0

    Args:
        annotation: String in format "dtype[shape][strides]device"

    Returns:
        A PyTorch tensor with the specified properties
    """
    # Parse the annotation string
    # Pattern: dtype[shape][strides]device
    pattern = r"([a-z0-9]+)(\[[\d,\s]*\])(\[[\d,\s]*\])(.+)"
    match = re.match(pattern, annotation)

    if not match:
        raise ValueError(f"Invalid tensor annotation format: {annotation}")

    dtype_str, shape_str, strides_str, device_str = match.groups()

    # Map dtype string to PyTorch dtype
    dtype_map = {v: k for k, v in dtype_abbrs.items()}

    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    dtype = dtype_map[dtype_str]

    # Parse shape: [384] or [384,512] -> (384,) or (384, 512)
    shape = (
        tuple(map(int, shape_str.strip("[]").split(",")))
        if shape_str.strip("[]")
        else ()
    )

    # Parse strides: [1] or [512,1] -> (1,) or (512, 1)
    strides = (
        tuple(map(int, strides_str.strip("[]").split(",")))
        if strides_str.strip("[]")
        else ()
    )

    # Parse device
    device = torch.device(device_str)

    # Create tensor with specified properties
    # We create an empty tensor and then use as_strided to set custom strides
    if shape:
        tensor = torch.empty_strided(shape, stride=strides, dtype=dtype, device=device)
        if tensor.dtype.is_floating_point:
            tensor.uniform_()
        else:
            try:
                tensor.random_()
                tensor = tensor % 128
            except NotImplementedError:
                tensor.fill_(0)
    else:
        # Scalar tensor
        tensor = torch.empty((), dtype=dtype, device=device)

    return tensor


def build_arguments(fn):
    sig = inspect.signature(fn)
    args = {}
    for k, v in sig.parameters.items():
        if k == "self":
            continue
        anno = v.annotation
        args[k] = parse_tensor_annotation(anno)
    return args


def _is_communication_node(node):
    if not node.op == "call_function":
        return False
    if not isinstance(node.target, torch._ops.OpOverload):
        return False

    return node.target.namespace == "_c10d_functional"


def make_custom_runtime_estimation(mesh):
    def custom_runtime_estimation(node: torch.fx.Node, override_size=None):
        if not node.op == "call_function":
            return 0
        if not isinstance(node.target, torch._ops.OpOverload):
            return 0

        if _is_communication_node(node):
            target = node.target
            if target == torch.ops._c10d_functional.wait_tensor.default:
                return 0
            # TODO: figure out mesh without reading from global scope
            mesh_topo = MeshTopoInfo.build_from_mesh(mesh)
            groups_name = tuple(g.group_name for g in mesh.get_all_groups())
            group_name = get_group_name(node)
            mesh_dim = groups_name.index(group_name)
            t = node.args[0].meta["val"]  # type: ignore[union-attr]
            comm_bytes_gb = t.numel() * t.itemsize / 2**30
            if override_size is not None:
                comm_bytes_gb = override_size
            if target in {
                torch.ops._c10d_functional.all_gather_into_tensor.default,
                torch.ops._c10d_functional.all_gather_into_tensor_out.default,
            }:
                comm_bytes_gb *= mesh.shape[mesh_dim]
                return allgather_cost(comm_bytes_gb, mesh_topo, mesh_dim)
            elif target == torch.ops._c10d_functional.reduce_scatter_tensor.default:
                return reduce_scatter_cost(comm_bytes_gb, mesh_topo, mesh_dim)
            elif target == torch.ops._c10d_functional.all_reduce.default:
                return allreduce_cost(comm_bytes_gb, mesh_topo, mesh_dim)
            else:
                # TODO: add all_to_all cost
                return 0
        return estimate_strategy_runtime_cost(node, None)

    return custom_runtime_estimation


def get_graph_module(gm, args):
    stack = ExitStack()
    with stack:
        joint_with_descriptors = aot_export_joint_with_descriptors(
            stack,
            gm,
            tuple(x for x in args.values()),
        )
    return joint_with_descriptors.graph_module


def apply_schedule_overlap_bucket(gm, custom_runtime_estimation):
    new_gm = schedule_overlap_bucketing(
        gm,
        collective_bucketing=False,
        custom_runtime_estimation=custom_runtime_estimation,
        max_compute_pre_fetch=5,
        max_in_flight_gb=2.0,
    )
    new_gm.recompile()
    return new_gm


def _get_tid(node):
    if _is_communication_node(node):
        if node.target == torch.ops._c10d_functional.wait_tensor.default:
            return 0
        return f"group-{node.args[-1]}"
    return 0


def _get_category(node):
    """Get trace category for node."""
    if _is_communication_node(node):
        return "nccl"
    if node.op == "call_function":
        target_str = str(node.target)
        if any(x in target_str.lower() for x in ["mm", "bmm", "matmul", "addmm"]):
            return "gemm"
        if any(x in target_str.lower() for x in ["flash", "attention", "sdpa"]):
            return "attention"
    return "kernel"


def _get_collective_type(node):
    """Get the type of collective operation."""
    if not _is_communication_node(node):
        return None
    target_str = str(node.target)
    if "all_gather" in target_str:
        return "AllGather"
    elif "reduce_scatter" in target_str:
        return "ReduceScatter"
    elif "all_reduce" in target_str:
        return "AllReduce"
    elif "wait_tensor" in target_str:
        return "Wait"
    return "Unknown"


def get_repr(arg, mode="full"):
    def get_dtype_repr(dtype):
        return dtype_abbrs[dtype]

    if isinstance(arg, torch.Tensor):
        out = {}
        out["shape"] = tuple(arg.shape)
        out["dtype"] = get_dtype_repr(arg.dtype)
        return out

    if isinstance(arg, (int, float, str)):
        return arg

    if isinstance(arg, torch.dtype):
        return get_dtype_repr(arg)

    if isinstance(arg, torch.fx.Node):
        if mode == "name_only" or "val" not in arg.meta:
            return f"fx node {arg.name}"
        elif mode == "full":
            return {"name": arg.name, "data": get_repr(arg.meta["val"])}
        elif mode == "content_only":
            return get_repr(arg.meta["val"])
        else:
            raise ValueError(f"Unknown mode {mode}")

    if isinstance(arg, (list, tuple)):
        return [get_repr(x, mode="name_only") for x in arg]

    if isinstance(arg, dict):
        return {k: get_repr(v, mode="name_only") for k, v in arg.items()}

    return f"arg {type(arg)}"


def create_execution_trace(
    gm: torch.fx.GraphModule,
    runtime_estimator: Callable[[torch.fx.Node], float],
    name: str = "fake_trace",
    file_path: str | None = None,
) -> dict[str, Any]:
    """
    Create a perfetto trace from a GraphModule representing its execution
    trace. This is useful for inspecting communication-computation overlapping
    for different reordering strategies.
    """
    launch_overhead = 1  # 1us
    ms_to_us = 1000

    trace_events = []
    curr_time: dict[int | str, float] = {0: 0}
    global_time: dict[torch.fx.Node, float] = {}

    for node_idx, node in enumerate(gm.graph.nodes):
        dur = runtime_estimator(node) * ms_to_us
        tid = _get_tid(node)
        if tid not in curr_time:
            curr_time[tid] = curr_time[0]

        cat = _get_category(node)
        coll_type = _get_collective_type(node)
        node_name = f"nccl:{coll_type}:{node.name}" if coll_type else str(node)

        event = {"ph": "X", "cat": cat, "name": node_name, "pid": 0, "tid": tid}

        if _is_communication_node(node):
            if tid == 0 and is_wait_tensor(node) and node.args[0].op != "placeholder":
                # if it's wait tensor, walk up chained waits to find the collective
                # Now this may happen for all_to_all in dsv3 (wait(wait(all_to_all)))
                comm_node = node.args[0]
                while is_wait_tensor(comm_node):
                    comm_node = comm_node.args[0]
                comm_end_time = global_time[comm_node]
                curr_time[tid] = max(curr_time[tid], comm_end_time)
            else:
                curr_time[tid] = max(curr_time[0], curr_time[tid])

        event["ts"] = curr_time[tid]
        event["dur"] = dur
        curr_time[tid] += dur + launch_overhead
        if tid != 0:
            curr_time[0] += launch_overhead
            # keep track of when a given collective will finish
            global_time[node] = curr_time[tid]

        event["args"] = {
            "order": node_idx,
            "output": get_repr(node, mode="content_only"),
            "inputs": [get_repr(arg) for arg in node.args],
        }

        if dur > 0.0:
            trace_events.append(event)

    trace = {
        "traceEvents": trace_events,
        "traceName": f"{name}_trace.json",
        "displayTimeUnit": "us",
    }

    if file_path is not None:
        with open(file_path, "w") as fp:
            json.dump(trace, fp, indent=2)

    return trace
