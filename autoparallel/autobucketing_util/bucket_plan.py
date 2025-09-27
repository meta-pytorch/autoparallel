# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import functools

# mypy: ignore-errors
from collections import defaultdict
from typing import Any, Dict

import torch
from torch._C._distributed_c10d import ReduceOp
from torch._inductor import scheduler
from torch._inductor.utils import is_collective

from .bucket_utils import (
    _schedule_fallback_operation,
    check_ir_node_bucketable,
    estimate_bucketed_snode_runtime,
    get_data_size,
    get_snode_process_group_info,
    get_snode_tensor_info,
)
from .estimation import benchmark_and_sync_runtime


def get_dynamic_memory_threshold(
    peak_memory,
    peak_memory_per_step_dict,
    current_step,
) -> int:
    """
    this function calculates the memory gap from the current step's memory to the peak memory
    """
    left_peak_memory = 0
    right_peak_memory = 0
    for idx, memory in peak_memory_per_step_dict.items():
        if idx <= current_step:
            left_peak_memory = max(memory, left_peak_memory)
        if idx >= current_step:
            right_peak_memory = max(memory, right_peak_memory)
    current_peak_memory = min(left_peak_memory, right_peak_memory)
    return peak_memory - current_peak_memory


def get_simplefsdp_auto_plan(
    sched: "scheduler.Scheduler",
    snodes: list["scheduler.BaseSchedulerNode"],
    name_to_buf: Dict[str, "scheduler.SchedulerBuffer"],
    name_to_fused_node: Dict[str, "scheduler.BaseSchedulerNode"],
    bucketable_nodes: set[str],
    configs: Any,
    verbose: bool = True,
) -> tuple[
    list[Dict[tuple[Any, ...], list["scheduler.BaseSchedulerNode"]]],
    list[Dict[tuple[Any, ...], list["scheduler.BaseSchedulerNode"]]],
]:
    """
    This function implements a greedy algorithm, which decides if the node could be bucketed
    with the previous one based on several criteria below:
        FWD Pass:
            (1) the bucketed AG communication could be overlapped by the previous computation;
            (2) the bucketed AG memory doesn't exceed peak memory;
            (3) bucketed AG communication size doesn't exceed 0.2*sum(fwd_ag_tensor_list), such
                that the estimated AG communication time is always in the calibration bound.
        BWD Pass:
            (1) the bucketed AG + RS communication could be overlapped by the previous computation;
            (2) the bucketed AG+RS memory doesn't exceed peak memory;
            (3) RS always have future compute to overlap it, such that its final exposed communication is small;
            (4) bucketed AG/RS communication size doesn't exceed 0.2* sum(fwd_ag_tensor_list) & 0.2* sum(bwd_rs_tensor_list),
                such that the estimated AG/RS communication time is always in the calibration bound.
    """
    all_gather_plan = []
    reduce_scatter_plan = []
    current_ag_bucket: Dict[
        tuple[Any, ...], list["scheduler.BaseSchedulerNode"]
    ] = defaultdict(list)
    current_rs_bucket: Dict[
        tuple[Any, ...], list["scheduler.BaseSchedulerNode"]
    ] = defaultdict(list)
    schedule_fallback_operation = functools.partial(
        _schedule_fallback_operation,
        scheduler=sched,
        name_to_buf=name_to_buf,
        name_to_fused_node=name_to_fused_node,
    )

    heuristic_info = {
        # time info
        "last_step_rs_comm_time": 0.0,
        "this_step_comp_time": 0.0,
        "this_step_rs_comm_time": 0.0,
        "next_step_comp_time": 0.0,
        "next_step_nonfsdp_comm_time": 0.0,
        # memory info
        "accumulated_gradient_size": 0,
        "last_step_rs_comm_size": 0,
        "this_step_rs_comm_out_size": 0,
        "this_step_rs_comm_inp_size": 0,
        "this_step_memory": 0,
        "next_step_memory": 0,
    }

    # sync runtime info across ranks
    (
        comm_cache,
        comp_time_dict,
        memory_dict,
        peak_memory_per_step_dict,
    ) = benchmark_and_sync_runtime(
        sched, snodes, name_to_buf, name_to_fused_node, bucketable_nodes, configs
    )
    future_comp_time = sum(comp_time_dict.values())
    peak_memory = max(peak_memory_per_step_dict.values()) + configs.peak_memory_offset

    # autobucket algorithm
    bucketable_ag_idx = -1
    seen_new_bucketable_ag = True
    for _, snode in enumerate(snodes):
        if is_collective(
            snode.node, op=torch.ops._c10d_functional.all_gather_into_tensor.default
        ) and check_ir_node_bucketable(snode.node, bucketable_nodes):
            bucketable_ag_idx += 1
            seen_new_bucketable_ag = True
            future_comp_time -= comp_time_dict[bucketable_ag_idx]

            ag_node_info = get_snode_tensor_info(
                snode, return_data_size=False
            ) + get_snode_process_group_info(
                snode,
                expected_op=torch.ops._c10d_functional.all_gather_into_tensor.default,
                resolve_pg=False,
            )
            current_ag_bucket[ag_node_info].append(snode)
            (
                estimated_comm,
                comm_size_inp,
                comm_size_out,
            ) = estimate_bucketed_snode_runtime(
                current_ag_bucket,
                schedule_fallback_operation,
                name_to_buf,
                "torch.ops._c10d_functional.all_gather_into_tensor.default",
                comm_cache,
            )

            # Check if current bucketing breaks the greedy criteria
            # (1) Overlappping criteria
            comp_time = heuristic_info["this_step_comp_time"] * (
                1 + configs.relax_ratio
            )
            comm_time = estimated_comm + heuristic_info["last_step_rs_comm_time"]
            break_overlap_criteria = comp_time < comm_time

            # (2) Memory criteria
            memory_threshold = get_dynamic_memory_threshold(
                peak_memory,
                peak_memory_per_step_dict,
                bucketable_ag_idx,
            )
            # the buckted AG/RS are created on-the-fly, whose memory was not captured by the
            # estimate_peak_memory function. The bucketed_comm_memory consists of:
            # in FWD pass:
            #   (1) all-gather copy-in (comm_size_inp): smaller buffers for dtype_conversion + bigger buffer to copy_in smaller buffers
            #       thus, we have comm_size_inp*2
            #   (2) all-gather copy-out (comm_size_out): bigger buffer to copy_out from ag_wait + split out smaller buffers for compute
            #       thus, we have comm_size_out*2
            # in BWD pass:
            # TODO (ruisizhang123): we need to double check this. From memory trace, we can clearly see
            #   these three regions stack together at a certain moment
            # due to reordering, the peak memory occurs at the end of current step's all-gather when last step & this step's reduce-scatter
            # are not cleared in time
            #   (1) all-gather copy-in/copy-out (like FWD pass)
            #   (2) last step's reduce-scatter: bigger buffer containts gradient
            #   (3) next step's reduce-scatter: smaller buffers for dtype_conversion + bigger buffer to copy_in gradient
            bucketed_comm_memory = (
                2 * comm_size_inp
                + 2 * comm_size_out
                + heuristic_info["this_step_rs_comm_inp_size"] * 2
                + heuristic_info["last_step_rs_comm_size"]
            )
            break_memory_criteria = (
                memory_threshold
                < heuristic_info["next_step_memory"] + bucketed_comm_memory
            )

            # (3) Communication size criteria
            break_comm_size_criteria = comm_cache.ag_max_inp_size < comm_size_inp
            if comm_cache.rs_max_out_size > 0:
                break_comm_size_criteria = (
                    break_comm_size_criteria
                    or comm_cache.rs_max_out_size
                    < heuristic_info["this_step_rs_comm_out_size"]
                )

            if (
                break_overlap_criteria
                or break_memory_criteria
                or break_comm_size_criteria
            ):
                if heuristic_info["this_step_comp_time"] > 0:
                    # if bucketing breaks the greedy criteria, pop the last node out
                    overflow_ag = current_ag_bucket[ag_node_info].pop()
                    all_gather_plan.append(current_ag_bucket)
                    current_ag_bucket: Dict[
                        tuple[Any, ...], list["scheduler.BaseSchedulerNode"]
                    ] = defaultdict(list)
                    current_ag_bucket[ag_node_info].append(overflow_ag)
                else:
                    # if there is no compute, we have to keep the all_gather to avoid deadlock
                    all_gather_plan.append(current_ag_bucket)
                    current_ag_bucket: Dict[
                        tuple[Any, ...], list["scheduler.BaseSchedulerNode"]
                    ] = defaultdict(list)

                if verbose:
                    print(
                        "break_overlap_criteria",
                        break_overlap_criteria,
                    )
                    print("Current comm time", comm_time, "comp time", comp_time)
                    print(
                        "break_memory_criteria",
                        break_memory_criteria,
                    )
                    print(
                        "memory_threshold",
                        memory_threshold,
                        "total memory",
                        heuristic_info["next_step_memory"] + bucketed_comm_memory,
                    )
                    print(
                        "break_comm_size_criteria",
                        break_comm_size_criteria,
                    )
                    print("current_ag_bucket", all_gather_plan[-1])

                # bucket reduce scatters if there are any
                if len(current_rs_bucket) > 0:
                    (
                        current_estimated_rs,
                        rs_comm_size_inp,
                        rs_comm_size_out,
                    ) = estimate_bucketed_snode_runtime(
                        current_rs_bucket,
                        schedule_fallback_operation,
                        name_to_buf,
                        "torch.ops._c10d_functional.reduce_scatter_tensor.default",
                        comm_cache,
                        ReduceOp.AVG,
                    )
                    heuristic_info["last_step_rs_comm_time"] = current_estimated_rs
                    reduce_scatter_plan.append(current_rs_bucket)
                    heuristic_info["last_step_rs_comm_size"] = rs_comm_size_out
                    current_rs_bucket: Dict[
                        tuple[Any, ...], list["scheduler.BaseSchedulerNode"]
                    ] = defaultdict(list)

                # update heuristic info for the next step
                (
                    heuristic_info["this_step_comp_time"],
                    heuristic_info["this_step_memory"],
                ) = (
                    heuristic_info["next_step_comp_time"]
                    + heuristic_info["next_step_nonfsdp_comm_time"],
                    heuristic_info["next_step_memory"],
                )
                (
                    heuristic_info["next_step_comp_time"],
                    heuristic_info["next_step_memory"],
                ) = (
                    0,
                    0,
                )
        elif is_collective(
            snode.node, op=torch.ops._c10d_functional.reduce_scatter_tensor.default
        ) and check_ir_node_bucketable(snode.node, bucketable_nodes):
            node_info = get_snode_tensor_info(
                snode, return_data_size=False
            ) + get_snode_process_group_info(
                snode,
                expected_op=torch.ops._c10d_functional.reduce_scatter_tensor.default,
                resolve_pg=False,
            )
            current_rs_bucket[node_info].append(snode)

            (
                heuristic_info["this_step_rs_comm_time"],
                rs_comm_size_inp,
                rs_comm_size_out,
            ) = estimate_bucketed_snode_runtime(
                current_rs_bucket,
                schedule_fallback_operation,
                name_to_buf,
                "torch.ops._c10d_functional.reduce_scatter_tensor.default",
                comm_cache,
                ReduceOp.AVG,
            )
            heuristic_info["this_step_rs_comm_out_size"] = rs_comm_size_out
            heuristic_info["this_step_rs_comm_inp_size"] = rs_comm_size_inp
            heuristic_info["accumulated_gradient_size"] += get_data_size(
                snode.node.layout.size
            )

            # Check if current bucketing breaks the greedy criteria
            # (4) future compute to overlap RS criteria
            break_rs_overlap_criteria = (
                future_comp_time < heuristic_info["this_step_rs_comm_time"] * 5
            )
            if break_rs_overlap_criteria:
                reduce_scatter_plan.append(current_rs_bucket)
                heuristic_info["last_step_rs_comm_time"] = heuristic_info[
                    "this_step_rs_comm_time"
                ]
                heuristic_info["this_step_rs_comm_time"] = 0
                current_rs_bucket: Dict[
                    tuple[Any, ...], list["scheduler.BaseSchedulerNode"]
                ] = defaultdict(list)
        else:
            # TODO (ruisizhang123): for now, we only consider FSDP + (TP & CP), whose comms are AG & RS & All_Reduce
            # For TP and CP, we consider the node as a "COMP" node with exposed communication as Comp time
            if is_collective(snode.node):
                current_comm = comm_cache.get_comm_time(
                    snode.node.inputs[0].layout.size,
                    snode.node.layout.size,
                    getattr(snode.node, "python_kernel_name", ""),
                    calibrated=True,
                )
                heuristic_info["next_step_nonfsdp_comm_time"] += current_comm
            else:
                if seen_new_bucketable_ag:
                    heuristic_info["next_step_memory"] += memory_dict[bucketable_ag_idx]
                    heuristic_info["next_step_comp_time"] += comp_time_dict[
                        bucketable_ag_idx
                    ]
                    seen_new_bucketable_ag = False

    if len(current_ag_bucket) > 0:
        all_gather_plan.append(current_ag_bucket)

    if len(current_rs_bucket) > 0:
        reduce_scatter_plan.append(current_rs_bucket)

    return all_gather_plan, reduce_scatter_plan
