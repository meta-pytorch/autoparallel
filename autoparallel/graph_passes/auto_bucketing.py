# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import torch
from torch._inductor.fx_passes.overlap_scheduling import schedule_overlap_bucketing

from .autobucketing_inductor import bucket_func, bucket_plan, bucket_utils, reorder


class simplefsdp_autobucketing_config:
    """
    Config for simplefsdp's autobucketing pass, which by default would give good performance.
    To make the results tunable, we expose the following parameters:
    - relax_ratio: relax comp time to include more comm in one bucket
                with this config, comp is updated as comp * (1 + relax_ratio)
    - peak_memory_offset: relax peak_memory to include more comm in one bucket
                with this config, peak_memory is updated as (peak_memory + peak_memory_offset)
    - load_cache: set to True to load cache from save_estimation_path
    - enable_bucket_ir: set to True to bucket all_gather/reduce_scatter
    - enable_reorder_ir: set to True to reorder all_gather/reduce_satter
    - calibrate_number: number of samples to calibrate during comm estimation
    """

    relax_ratio = 0
    peak_memory_offset = 0
    load_cache = False
    save_estimation_path = "/mnt/mffuse/cache_ruisi/estimation_mast.pkl"
    enable_bucket_ir = True
    enable_reorder_ir = True
    calibrate_number = 40


def simple_fsdp_autobucketing_reordering_pass(
    snodes: list["torch._inductor.scheduler.BaseSchedulerNode"],
    configs: "simplefsdp_autobucketing_config",
) -> list["torch._inductor.scheduler.BaseSchedulerNode"]:
    scheduler = snodes[0].scheduler
    bucketable_nodes = bucket_utils.get_bucketable_ir_nodes(
        snodes, scheduler.name_to_fused_node, scheduler.name_to_buf
    )

    assert (
        not torch._inductor.config.allow_buffer_reuse
    ), "bucketing algorithm requires torch._inductor.config.allow_buffer_reuse to be False"

    if configs.enable_bucket_ir:
        all_gather_plan, reduce_scatter_plan = bucket_plan.get_simplefsdp_auto_plan(
            scheduler,
            snodes,
            scheduler.name_to_buf,
            scheduler.name_to_fused_node,
            bucketable_nodes,
            configs,
        )

        snodes = bucket_func.bucket_fsdp_all_gather_with_plan(
            scheduler,
            snodes,
            scheduler.name_to_buf,
            scheduler.name_to_fused_node,
            all_gather_plan,
            bucketable_nodes,
        )
        if len(reduce_scatter_plan) > 0:
            snodes = bucket_func.bucket_fsdp_reduce_scatter_with_plan(
                scheduler,
                snodes,
                scheduler.name_to_buf,
                scheduler.name_to_fused_node,
                reduce_scatter_plan,
                bucketable_nodes,
            )

    if configs.enable_reorder_ir:
        print("Reorder scheduler nodes with autobucketing algroithm")
        node_length = len(snodes)
        snodes = reorder.reorder_all_gather(
            snodes, bucketable_nodes, all_gather_before_last_wait=False
        )
        assert node_length == len(
            snodes
        ), f"Missed nodes in reordering all gather: expected {node_length}, but got {len(snodes)}"
        snodes = reorder.reorder_reduce_scatter(snodes, bucketable_nodes)
        assert node_length == len(
            snodes
        ), f"Missed nodes in reordering reduce scatter: expected {node_length}, but got {len(snodes)}"

    return snodes


class aten_autobucketing_config:
    """
    Config for aten level autobucketing pass from stacked PR: https://github.com/pytorch/pytorch/pull/163960
    - max_in_flight_gb: maximum GB of concurrent collective data
    - compute_overlap_multipler: scale factor for compute time used to hide collectives
    - max_coll_distance: maximum node distance for overlap consideration
    """

    max_in_flight_gb = 2.0
    compute_overlap_multipler = 1.0
    max_coll_distance = 100
    custom_runtime_estimation = None
    max_compute_pre_fetch = 5
    collective_bucketing = False
    save_trace = True
    _counter = 0


def aten_autobucketing_reordering_pass(
    gm: torch.fx.Graph, configs: "aten_autobucketing_config"
) -> torch.fx.GraphModule:
    new_gm = schedule_overlap_bucketing(
        gm.owning_module,
        collective_bucketing=configs.collective_bucketing,
        max_compute_pre_fetch=configs.max_compute_pre_fetch,
        custom_runtime_estimation=configs.custom_runtime_estimation,
        compute_overlap_multipler=configs.compute_overlap_multipler,
        max_in_flight_gb=configs.max_in_flight_gb,
        max_coll_distance=configs.max_coll_distance,
    )
    new_gm.recompile()

    if configs.save_trace:
        from autoparallel.graph_passes.debug_helpers import create_execution_trace

        assert configs.custom_runtime_estimation is not None

        create_execution_trace(
            new_gm,
            configs.custom_runtime_estimation,
            f"fake_trace_{configs._counter}.json",
        )
        configs._counter += 1
    return new_gm


class overlap_scheduling_config:
    """
    Config for overlap scheduling pass (autoparallel's own solver).

    - solver: "greedy" (fast, memory-aware) or "ilp" (optimal via PuLP/CBC)
    - save_trace: write Perfetto-compatible before/after traces
    - custom_runtime_estimation: callable(node) -> cost in microseconds
    - max_in_flight: per comm stream, max dispatched comms with unconsumed
      results.  Leaf comms (no downstream consumers) don't count.
      Controls the AG/compute interleaving depth.
    """

    solver = "greedy"
    save_trace = True
    custom_runtime_estimation = None
    memory_budget_bytes = None
    max_in_flight = 2
    _counter = 0


def overlap_scheduling_reordering_pass(
    graph: torch.fx.Graph, configs: "overlap_scheduling_config"
) -> None:
    from autoparallel.graph_passes.debug_helpers import (
        _is_communication_node,
        create_execution_trace,
    )
    from autoparallel.overlap_scheduling import (
        COMPUTE_STREAM,
        NodeInfo,
        NodeKind,
        reorder_for_overlap,
    )

    runtime_estimator = configs.custom_runtime_estimation
    assert runtime_estimator is not None

    gm = graph.owning_module

    def _output_memory_bytes(node: torch.fx.Node) -> int:
        val = node.meta.get("val")
        if isinstance(val, torch.Tensor):
            return val.numel() * val.element_size()
        if isinstance(val, (list, tuple)):
            total = 0
            for v in val:
                if isinstance(v, torch.Tensor):
                    total += v.numel() * v.element_size()
            return total
        return 0

    def classify(node: torch.fx.Node) -> NodeInfo:
        if node.op in ("placeholder", "output", "get_attr"):
            return NodeInfo(NodeKind.SKIP, COMPUTE_STREAM, 0.0)
        if node.op != "call_function":
            return NodeInfo(NodeKind.SKIP, COMPUTE_STREAM, 0.0)
        mem = _output_memory_bytes(node)
        if _is_communication_node(node):
            if node.target == torch.ops._c10d_functional.wait_tensor.default:
                return NodeInfo(NodeKind.COMM_WAIT, COMPUTE_STREAM, 0.0)
            pg_name = str(node.args[-1])
            duration_ms = runtime_estimator(node) / 1000.0
            return NodeInfo(NodeKind.COMM_START, pg_name, duration_ms, mem)
        duration_ms = runtime_estimator(node) / 1000.0
        return NodeInfo(NodeKind.COMPUTE, COMPUTE_STREAM, duration_ms, mem)

    counter = configs._counter
    configs._counter += 1

    if configs.save_trace:
        create_execution_trace(
            gm, runtime_estimator, f"overlap_trace_{counter}_before.json"
        )

    import time

    t = time.time()
    schedule = reorder_for_overlap(
        graph,
        classify,
        solver=configs.solver,
        memory_budget_bytes=configs.memory_budget_bytes,
        max_in_flight=configs.max_in_flight,
    )
    print(
        f"Reordering took {time.time() - t:.2f} seconds, "
        f"solver makespan: {schedule.makespan:.0f} ms"
    )
    gm.recompile()

    if configs.save_trace:
        create_execution_trace(
            gm, runtime_estimator, f"overlap_trace_{counter}_after.json"
        )


def configure_inductor_for_autobucketing(mode: str = "aten"):
    # allow configuring inductor comms optimizations from torchtitan commandline
    if mode == "aten":
        torch._inductor.config.aten_distributed_optimizations.enable_overlap_scheduling = (
            True
        )
        torch._inductor.config.aten_distributed_optimizations.collective_bucketing = (
            True
        )
        torch._inductor.config.aten_distributed_optimizations.insert_overlap_deps = True
        torch._inductor.config.aten_distributed_optimizations.max_compute_pre_fetch = 10
    elif mode == "inductor":
        from autoparallel.graph_passes.auto_bucketing import (
            simple_fsdp_autobucketing_reordering_pass,
            simplefsdp_autobucketing_config,
        )

        torch._inductor.config.allow_buffer_reuse = False
        torch._inductor.config.reorder_for_peak_memory = False
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        simplefsdp_autobucketing_config.calibrate_number = 5
        simplefsdp_autobucketing_config.save_estimation_path = "./estimation_mast.pkl"
        simple_fsdp_autobucketing_reordering_pass = partial(
            simple_fsdp_autobucketing_reordering_pass,
            configs=simplefsdp_autobucketing_config,  # type: ignore
        )
        torch._inductor.config.reorder_for_compute_comm_overlap_passes = [
            simple_fsdp_autobucketing_reordering_pass
        ]
    elif mode == "none":
        torch._inductor.config.reorder_for_peak_memory = False
        torch._inductor.config.reorder_for_compute_comm_overlap = False
    else:
        raise ValueError(f"Unknown comms bucket reorder strategy: {mode}")
