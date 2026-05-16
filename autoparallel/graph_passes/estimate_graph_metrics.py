# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch._inductor.fx_passes.memory_estimator import MemoryTracker

from autoparallel.graph_passes.debug_helpers import (
    _get_tid,
    _is_communication_node,
    _is_sync_collective,
    _is_wait_tensor,
    _resolve_comm_node,
)


@dataclass
class GraphMetrics:
    total_time: float  # critical-path time (max across all streams), microseconds
    compute_time: float  # sum of compute node durations, microseconds
    communication_time: float  # sum of comm node durations (excl. wait_tensor), microseconds
    exposed_comm_time: float  # time the compute stream stalls at wait_tensor nodes, microseconds
    peak_memory: int  # peak live tensor memory in bytes


def estimate_graph_metrics(
    gm: torch.fx.GraphModule,
    runtime_estimator: Callable[[torch.fx.Node], float],
) -> GraphMetrics:
    """
    Estimate runtime and memory metrics for a sharded GraphModule.

    Uses the same multi-stream timing simulation as create_execution_trace
    (tid=0 for compute, tid=N for comms, wait_tensor syncs) and PyTorch's
    MemoryTracker for liveness-based peak memory tracking.
    """
    tracker = MemoryTracker(gm.graph)

    curr_time: dict[int, float] = {0: 0.0}
    global_time: dict[torch.fx.Node, float] = {}
    compute_time = 0.0
    communication_time = 0.0
    exposed_comm_time = 0.0

    for node in gm.graph.nodes:
        tracker.schedule_node(node)

        dur = float(runtime_estimator(node))
        tid = _get_tid(node)
        if tid not in curr_time:
            curr_time[tid] = curr_time[0]

        if _is_communication_node(node):
            if tid == 0 and _is_wait_tensor(node) and node.args[0].op != "placeholder":
                comm_node = _resolve_comm_node(node, global_time)
                if comm_node in global_time:
                    comm_end_time = global_time.pop(comm_node)
                else:
                    comm_end_time = curr_time[0]
                stall = max(0.0, comm_end_time - curr_time[0])
                exposed_comm_time += stall
                curr_time[tid] = max(curr_time[tid], comm_end_time)
            else:
                curr_time[tid] = max(curr_time[0], curr_time[tid])
                communication_time += dur
        else:
            compute_time += dur

        launch_overhead = 1.0  # 1us
        curr_time[tid] += dur + launch_overhead
        if tid != 0:
            curr_time[0] += launch_overhead
            global_time[node] = curr_time[tid]
            if _is_sync_collective(node):
                stall = max(0.0, curr_time[tid] - curr_time[0])
                exposed_comm_time += stall
                curr_time[0] = max(curr_time[0], curr_time[tid])

    return GraphMetrics(
        total_time=max(curr_time.values()),
        compute_time=compute_time,
        communication_time=communication_time,
        exposed_comm_time=exposed_comm_time,
        peak_memory=tracker.peak_memory,
    )
