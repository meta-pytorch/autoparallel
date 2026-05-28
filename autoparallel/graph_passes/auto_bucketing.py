# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import Counter, defaultdict
from functools import partial

import torch
from torch._inductor.fx_passes.overlap_scheduling import schedule_overlap_bucketing
from torch.utils._ordered_set import OrderedSet

from .autobucketing_inductor import bucket_func, bucket_plan, bucket_utils, reorder

logger = logging.getLogger(__name__)


def _patch_fsdp_bucketing():
    """Patch PyTorch's FSDP bucketing for better multi-group handling.

    Two fixes:
    1. Primary-group-only: only include the group with the most FSDP
       all-gathers in fsdp_groups, preventing minority groups (tp, combined)
       from limiting dp bucketing aggressiveness.
    2. Non-adjacent bucketing: allow collectives to be bucketed even when
       interleaved with non-FSDP collectives on other groups. Only
       descendant conflicts prevent bucketing, not graph position.
    """
    import torch._inductor.fx_passes.bucketing as bucketing_mod
    import torch._inductor.fx_passes.fsdp as fsdp_mod
    from torch._inductor.fx_passes.bucketing import (
        collect_node_descendants,
        is_wait_tensor,
    )
    from torch._inductor.fx_passes.fsdp import (
        _find_all_gathers,
        _get_group_name,
        _get_group_size_from_node,
        is_fsdp_all_gather,
    )

    def _patched_identify_fsdp_groups(gm):
        fsdp_counts_by_group = Counter()
        group_size = None
        for n in _find_all_gathers(gm.graph):
            if is_fsdp_all_gather(n):
                gn = _get_group_name(n)
                fsdp_counts_by_group[gn] += 1
                if group_size is None:
                    group_size = _get_group_size_from_node(n)

        if fsdp_counts_by_group:
            primary_group = fsdp_counts_by_group.most_common(1)[0][0]
            fsdp_groups = OrderedSet([primary_group])
        else:
            fsdp_groups = OrderedSet()

        logger.debug(
            "identify_fsdp_groups (patched): fsdp_groups=%s, all_counts=%s",
            list(fsdp_groups),
            dict(fsdp_counts_by_group),
        )
        return fsdp_groups, group_size

    def _patched_greedy_bucket(
        gm,
        bucket_cap_mb_by_bucket_idx,
        filter_node,
        node_group_key,
        filter_wait_node=None,
    ):
        g = gm.graph
        groups = defaultdict(list)
        for node in g.nodes:
            if is_wait_tensor(node) and filter_node(node.args[0]):
                if (filter_wait_node is None) or filter_wait_node(node):
                    coll_node = node.args[0]
                    key = node_group_key(coll_node)
                    groups[key].append(coll_node)

        if not groups:
            return []

        node_descendents = collect_node_descendants(g)

        buckets = []
        for key, nodes in groups.items():
            cur_bucket = []
            cur_bucket_descendents = OrderedSet()
            cur_bucket_size_bytes = 0
            cur_bucket_id = 0
            bucket_size_bytes = int(
                bucket_cap_mb_by_bucket_idx(cur_bucket_id) * 1024 * 1024
            )
            for node in nodes:
                if node in cur_bucket_descendents:
                    continue
                n_val = node.meta["val"]
                out_size_bytes = n_val.numel() * n_val.element_size()
                n_input_val = node.all_input_nodes[0].meta["val"]
                in_size_bytes = n_input_val.numel() * n_input_val.element_size()
                size_bytes = max(out_size_bytes, in_size_bytes)
                if (
                    cur_bucket_size_bytes + size_bytes > bucket_size_bytes
                    and cur_bucket
                ):
                    if len(cur_bucket) > 1:
                        buckets.append(cur_bucket)
                    cur_bucket = []
                    cur_bucket_size_bytes = 0
                    cur_bucket_id += 1
                    cur_bucket_descendents = OrderedSet()
                    bucket_size_bytes = int(
                        bucket_cap_mb_by_bucket_idx(cur_bucket_id) * 1024 * 1024
                    )
                cur_bucket_size_bytes += size_bytes
                cur_bucket.append(node)
                cur_bucket_descendents |= node_descendents[node]
            if len(cur_bucket) > 1:
                buckets.append(cur_bucket)
        return buckets

    fsdp_mod.identify_fsdp_groups = _patched_identify_fsdp_groups
    bucketing_mod.greedy_bucket_collective_by_mb = _patched_greedy_bucket


_patch_fsdp_bucketing()


def _cap_compute_batch_size(
    graph, original_compute_names, original_rs_after_compute, max_consecutive=8
):
    """Break up long compute segments between ReduceScatter operations.

    After overlap scheduling + FSDP bucketing, compute nodes may be reordered
    so that many layers' matmuls execute before any ReduceScatter fires. This
    inflates peak memory because all layers' activations are alive
    simultaneously.

    Instead of restoring the full original order (which kills comm/compute
    overlap), this function only intervenes when the number of compute nodes
    between consecutive ReduceScatter ops exceeds max_consecutive. For each
    oversized segment, it sorts compute nodes by original order, splits into
    chunks, then:
    1. Chains consecutive compute nodes within each chunk (so they move
       together during topological sort).
    2. Adds a dep from the first compute node of each chunk to an RS node
       that originally appeared between the previous chunk and this one.

    Args:
        graph: The post-scheduled FX graph.
        original_compute_names: List of compute node names in original order.
        original_rs_after_compute: Dict mapping compute node name to the list
            of RS node names that appeared between it and the next compute node
            in the original (pre-scheduling) graph.
        max_consecutive: Maximum compute nodes allowed between RS ops.
    """
    from torch._dynamo.graph_deduplication import _stable_topological_sort
    from torch._inductor.fx_passes.overlap_scheduling import is_compute_node

    def _is_rs(node):
        if node.op != "call_function":
            return False
        name = str(node.target)
        return "reduce_scatter" in name and "wait" not in name

    original_rank = {name: rank for rank, name in enumerate(original_compute_names)}
    node_by_name = {n.name: n for n in graph.nodes}

    scheduled_rs_names = {
        n.name for n in graph.nodes if n.op == "call_function" and _is_rs(n)
    }

    # Collect compute nodes between RS ops in the post-scheduled graph.
    segments: list[tuple[list[torch.fx.Node], torch.fx.Node | None]] = []
    current_compute: list[torch.fx.Node] = []
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if is_compute_node(node):
            current_compute.append(node)
        elif _is_rs(node):
            segments.append((current_compute, node))
            current_compute = []
    if current_compute:
        segments.append((current_compute, None))

    additional_deps: dict[torch.fx.Node, OrderedSet] = defaultdict(OrderedSet)

    for compute_nodes, _seg_rs_node in segments:
        if len(compute_nodes) <= max_consecutive:
            continue

        sorted_nodes = sorted(
            compute_nodes,
            key=lambda n: original_rank.get(n.name, float("inf")),
        )

        # Split into chunks of max_consecutive.
        chunks = []
        for i in range(0, len(sorted_nodes), max_consecutive):
            chunks.append(sorted_nodes[i : i + max_consecutive])

        # Chain consecutive compute nodes within each chunk.
        for chunk in chunks:
            for j in range(1, len(chunk)):
                additional_deps[chunk[j]].add(chunk[j - 1])

        # At each chunk boundary, find an RS that originally appeared between
        # the last compute of the previous chunk and the first compute of
        # the current chunk, then add dep: first_of_chunk after RS.
        for ci in range(1, len(chunks)):
            prev_chunk = chunks[ci - 1]
            curr_chunk = chunks[ci]

            last_in_prev = prev_chunk[-1]
            first_in_curr = curr_chunk[0]
            last_rank = original_rank.get(last_in_prev.name, -1)
            first_rank = original_rank.get(
                first_in_curr.name, len(original_compute_names)
            )

            found_rs_node = None
            for r in range(last_rank, first_rank):
                cname = (
                    original_compute_names[r]
                    if r < len(original_compute_names)
                    else None
                )
                if cname is None:
                    continue
                for rs_name in original_rs_after_compute.get(cname, []):
                    if rs_name in scheduled_rs_names:
                        rs_obj = node_by_name.get(rs_name)
                        if rs_obj is not None:
                            found_rs_node = rs_obj
                            break
                if found_rs_node is not None:
                    break

            if found_rs_node is not None:
                additional_deps[first_in_curr].add(found_rs_node)

    if not additional_deps:
        return

    try:
        _stable_topological_sort(graph, additional_deps)
    except AssertionError:
        logger.warning(
            "Failed to cap compute batch size (cycle detected), "
            "falling back to uncapped ordering"
        )


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
        logger.debug("Reorder scheduler nodes with autobucketing algroithm")
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
    max_compute_pre_fetch = 50
    collective_bucketing = False
    save_trace = True
    _counter = 0


def aten_autobucketing_reordering_pass(
    gm: torch.fx.Graph, configs: "aten_autobucketing_config"
) -> torch.fx.GraphModule:
    assert gm.owning_module is not None

    # Record compute + RS interleaving before bucketing + overlap scheduling.
    from torch._inductor.fx_passes.overlap_scheduling import is_compute_node

    def _is_rs_node(node):
        if node.op != "call_function":
            return False
        name = str(node.target)
        return "reduce_scatter" in name and "wait" not in name

    original_compute_names = []
    original_rs_after_compute: dict[str, list[str]] = {}
    last_compute_name = None
    for n in gm.owning_module.graph.nodes:
        if n.op != "call_function":
            continue
        if is_compute_node(n):
            original_compute_names.append(n.name)
            last_compute_name = n.name
        elif _is_rs_node(n) and last_compute_name is not None:
            original_rs_after_compute.setdefault(last_compute_name, []).append(n.name)

    new_gm = schedule_overlap_bucketing(
        gm.owning_module,
        collective_bucketing=configs.collective_bucketing,
        max_compute_pre_fetch=configs.max_compute_pre_fetch,
        custom_runtime_estimation=configs.custom_runtime_estimation,
        compute_overlap_multipler=configs.compute_overlap_multipler,
        max_in_flight_gb=configs.max_in_flight_gb,
        max_coll_distance=configs.max_coll_distance,
    )

    _cap_compute_batch_size(
        new_gm.graph, original_compute_names, original_rs_after_compute
    )
    new_gm.recompile()

    if configs.save_trace:
        from autoparallel.graph_passes.debug_helpers import create_execution_trace

        assert configs.custom_runtime_estimation is not None

        create_execution_trace(
            new_gm,
            configs.custom_runtime_estimation,
            file_path=f"fake_trace_{configs._counter}.json",
        )
        configs._counter += 1
    return new_gm


def configure_inductor_for_autobucketing(mode: str = "aten"):
    # allow configuring inductor comms optimizations from torchtitan commandline
    if mode == "aten":
        torch._inductor.config.aten_distributed_optimizations.enable_overlap_scheduling = (
            torch.cuda.is_available()  # Disable overlap scheduling for non-CUDA devices
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
