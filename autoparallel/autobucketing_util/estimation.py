# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# mypy: ignore-errors
import os
import pickle
from collections import defaultdict
from typing import Any

import torch
import torch.distributed as c10d
from torch._inductor import memory, scheduler
from torch._inductor.utils import is_collective
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from .bucket_utils import (
    check_ir_node_bucketable,
    get_snode_process_group_info,
    get_snode_tensor_info,
)
from .estimation_utils import (
    CommPerfCache,
    CompPerfCache,
    benchmark_and_cache_comm_dicts,
    estimate_comp_time,
)


def sync_dict_across_ranks(runtime_dict, world_size, group=None):
    gathered_lists = [None for _ in range(world_size)]
    c10d.all_gather_object(gathered_lists, list(runtime_dict.values()), group=group)
    median_gathered_time = torch.median(torch.tensor(gathered_lists), dim=0).values
    for idx, (key, value) in enumerate(runtime_dict.items()):
        runtime_dict[key] = median_gathered_time[idx]
    return runtime_dict


def benchmark_and_sync_runtime(
    sched: "scheduler.Scheduler",
    snodes: list["scheduler.BaseSchedulerNode"],
    name_to_buf: dict[str, "scheduler.SchedulerBuffer"],
    name_to_fused_node: dict[str, "scheduler.BaseSchedulerNode"],
    bucketable_nodes: set[str],
    configs: Any,
):
    world_size = c10d.distributed_c10d.get_world_size()

    fsdp_ag_input_size_dict = defaultdict(list)
    fsdp_rs_output_size_dict = defaultdict(list)
    non_fsdp_ag_input_size_dict = defaultdict(list)
    non_fsdp_rs_input_size_dict = defaultdict(list)
    all_reduce_input_size_dict = defaultdict(list)
    all_to_all_input_size_dict = defaultdict(list)
    comp_cache, comm_cache = CompPerfCache(), CommPerfCache()

    cali_num_samples = configs.calibrate_number
    comp_time_dict = defaultdict(float)
    memory_dict = defaultdict(int)
    peak_memory_per_step_dict = defaultdict(int)
    fsdp_ag_idx = -1
    release_steps = [0]

    graph_outputs = OrderedSet(V.graph.get_output_names())
    graph_inputs = OrderedSet(V.graph.graph_inputs.keys())
    _, name_to_freeable_input_buf = memory.prepare_planning_info(
        snodes,
        name_to_buf,
        name_to_fused_node,
        graph_inputs,
        graph_outputs,
    )
    _, memories_at_nodes = memory.estimate_peak_memory(
        snodes, name_to_freeable_input_buf, graph_outputs
    )
    # ensure memory offset is always positive
    if min(memories_at_nodes) < 0:
        shift_value = abs(min(memories_at_nodes))
        memories_at_nodes = [x + shift_value for x in memories_at_nodes]

    for idx, snode in enumerate(snodes):
        if is_collective(
            snode.node, op=torch.ops._c10d_functional.all_gather_into_tensor.default
        ):
            fsdp_ag_idx += 1
            release_steps.append(idx)
            node_tensor_info = get_snode_tensor_info(snode, return_data_size=True)
            node_pg_info = get_snode_process_group_info(
                snode,
                expected_op=torch.ops._c10d_functional.all_gather_into_tensor.default,
                resolve_pg=True,
            )
            if node_pg_info is None:
                continue
            node_info = node_tensor_info[:-2] + node_pg_info
            input_size = node_tensor_info[-2]
            if check_ir_node_bucketable(snode.node, bucketable_nodes):
                # For FSDP, we assume they have all have the
                fsdp_ag_input_size_dict[node_info].append(input_size)
            else:
                non_fsdp_ag_input_size_dict[node_info].append(input_size)
        elif is_collective(
            snode.node, op=torch.ops._c10d_functional.reduce_scatter_tensor.default
        ):
            node_tensor_info = get_snode_tensor_info(snode, return_data_size=True)
            node_pg_info = get_snode_process_group_info(
                snode,
                expected_op=torch.ops._c10d_functional.reduce_scatter_tensor.default,
                resolve_pg=True,
            )
            if node_pg_info is None:
                continue
            node_info = node_tensor_info[:-2] + node_pg_info
            output_size = node_tensor_info[-1]
            if check_ir_node_bucketable(snode.node, bucketable_nodes):
                # For FSDP, we assume they have all have the same group size
                fsdp_rs_output_size_dict[node_info].append(output_size)
            else:
                non_fsdp_rs_input_size_dict[node_info].append(output_size)
        elif is_collective(
            snode.node, op=torch.ops._c10d_functional.all_reduce_.default
        ):
            node_tensor_info = get_snode_tensor_info(snode, return_data_size=True)
            node_pg_info = get_snode_process_group_info(
                snode,
                expected_op=torch.ops._c10d_functional.all_reduce_.default,
                resolve_pg=True,
            )
            if node_pg_info is None:
                continue
            node_info = node_tensor_info[:-2] + node_pg_info
            input_size = node_tensor_info[-2]
            all_reduce_input_size_dict[node_info].append(input_size)
        elif is_collective(
            snode.node, op=torch.ops._c10d_functional.all_to_all_single.default
        ):
            node_tensor_info = get_snode_tensor_info(snode, return_data_size=True)
            node_pg_info = get_snode_process_group_info(
                snode,
                expected_op=torch.ops._c10d_functional.all_to_all_single.default,
                resolve_pg=True,
            )
            if node_pg_info is None:
                continue
            node_info = node_tensor_info[:-2] + node_pg_info
            input_size = node_tensor_info[-2]
            all_to_all_input_size_dict[node_info].append(input_size)
        else:
            if not is_collective(snode.node):
                comp_time = estimate_comp_time(
                    sched, snode, verbose=False, comp_cache=comp_cache
                )
                comp_time_dict[fsdp_ag_idx] += comp_time
                memory_dict[fsdp_ag_idx] = max(
                    abs(
                        memories_at_nodes[idx + 1]
                        - memories_at_nodes[release_steps[-1]]
                    ),
                    memory_dict[fsdp_ag_idx],
                )
                peak_memory_per_step_dict[fsdp_ag_idx] = max(
                    memories_at_nodes[idx + 1], peak_memory_per_step_dict[fsdp_ag_idx]
                )
            else:
                print(
                    "[Relaxed Setting] untracked communication",
                    snode.node.python_kernel_name,
                )

    # Sync total compute time
    comp_time_dict = sync_dict_across_ranks(comp_time_dict, world_size)
    memory_dict = sync_dict_across_ranks(memory_dict, world_size)
    peak_memory_per_step_dict = sync_dict_across_ranks(
        peak_memory_per_step_dict, world_size
    )

    if configs.load_cache and os.path.exists(configs.save_estimation_path):
        with open(configs.save_estimation_path, "rb") as file:
            cache = pickle.load(file)
            comm_cache.cache = cache
            comm_cache._update_max_size()
        return comm_cache, comp_time_dict, memory_dict, peak_memory_per_step_dict

    benchmark_params = [
        (
            fsdp_ag_input_size_dict,
            "torch.ops._c10d_functional.all_gather_into_tensor.default",
            cali_num_samples,
        ),
        (
            fsdp_rs_output_size_dict,
            "torch.ops._c10d_functional.reduce_scatter_tensor.default",
            cali_num_samples,
        ),
        (
            non_fsdp_ag_input_size_dict,
            "torch.ops._c10d_functional.all_gather_into_tensor.default",
            3,
        ),
        (
            non_fsdp_rs_input_size_dict,
            "torch.ops._c10d_functional.reduce_scatter_tensor.default",
            3,
        ),
        (
            all_reduce_input_size_dict,
            "torch.ops._c10d_functional.all_reduce_.default",
            3,
        ),
        (
            all_to_all_input_size_dict,
            "torch.ops._c10d_functional.all_to_all_single.default",
            3,
        ),
    ]
    for input_size_dict, op_name, num_samples in benchmark_params:
        if len(input_size_dict) > 0:
            benchmark_and_cache_comm_dicts(
                comm_cache, input_size_dict, op_name, num_samples
            )

    median_runtimes = sync_dict_across_ranks(comm_cache.cache, world_size)
    comm_cache.cache = median_runtimes
    comm_cache._update_max_size()
    with open(configs.save_estimation_path, "wb") as file:
        pickle.dump(comm_cache.cache, file)
    return comm_cache, comp_time_dict, memory_dict, peak_memory_per_step_dict
