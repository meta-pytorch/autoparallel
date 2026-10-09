# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
from collections import Counter, defaultdict
from contextvars import ContextVar
from functools import partial
from typing import Any

import torch
from torch._inductor.fx_passes.overlap_scheduling import schedule_overlap_bucketing
from torch._logging import trace_structured
from torch.utils._ordered_set import OrderedSet

from .autobucketing_inductor import bucket_func, bucket_plan, bucket_utils, reorder

logger = logging.getLogger(__name__)

_active_bucketing_diagnostics: ContextVar[dict[str, Any] | None] = ContextVar(
    "autoparallel_active_bucketing_diagnostics", default=None
)


def _record_greedy_bucket_diagnostic(diagnostic: dict[str, Any]) -> None:
    active = _active_bucketing_diagnostics.get()
    if active is not None:
        active["greedy_bucket_calls"].append(diagnostic)


def _collective_counts(graph: torch.fx.Graph) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        target = str(node.target)
        if "all_gather_into_tensor" in target:
            counts["all_gather"] += 1
        elif "reduce_scatter_tensor" in target:
            counts["reduce_scatter"] += 1
        elif "all_reduce" in target:
            counts["all_reduce"] += 1
        elif "wait_tensor" in target:
            counts["wait_tensor"] += 1
    return dict(sorted(counts.items()))


def _patch_fsdp_bucketing():
    """Patch PyTorch's FSDP bucketing for better multi-group handling.

    Two fixes:
    1. All-reduce group selection: by default, only all-reduces on the group
       with the most FSDP all-gathers are bucketed. An optional policy can admit
       other all-reduce process groups. Upstream all-gather and reduce-scatter
       bucketing uses structural filters and is unaffected by this selection.
    2. Topo-span-bounded bucketing: allow collectives to be bucketed even
       when interleaved with non-FSDP collectives on other groups, but
       close the bucket once its topo-span (rank of latest member minus
       rank of first member) exceeds aten_autobucketing_config.max_topo_span.

       Without a span bound, merging collectives that are far apart in the
       graph rewires the dependency graph so that stable_topological_sort
       can pull compute from late layers forward, batching many MMs before
       any RS fires (the MM*N problem). The span bound caps how far compute
       can be displaced by any one bucket merge.
    """
    import torch._inductor.fx_passes.bucketing as bucketing_mod
    import torch._inductor.fx_passes.fsdp as fsdp_mod
    from torch._inductor.fx_passes.bucketing import (
        collect_node_descendants,
        is_wait_tensor,
    )
    from torch._inductor.fx_passes.fsdp import (
        _find_all_gathers,
        _find_all_reduces,
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
            selection_mode = "primary"
            if aten_autobucketing_config.bucket_all_all_reduce_groups:
                fsdp_groups.update(
                    _get_group_name(node) for node in _find_all_reduces(gm.graph)
                )
                selection_mode = "all-all-reduce-groups"
        else:
            fsdp_groups = OrderedSet()
            primary_group = None
            selection_mode = "none"

        active = _active_bucketing_diagnostics.get()
        if active is not None:
            active["group_selection"] = {
                "mode": selection_mode,
                "selection_scope": "process_group",
                "primary_group": primary_group,
                "selected_groups": list(fsdp_groups),
                "fsdp_all_gather_counts": [
                    {"group": group, "count": count}
                    for group, count in fsdp_counts_by_group.items()
                ],
            }

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
        # Snapshot ranks before any bucketing mutates the graph. Used to
        # bound each bucket's topo-span, which bounds how far compute can
        # be displaced by stable_topological_sort after merging.
        ranks = {n.name: i for i, n in enumerate(g.nodes)}

        groups = defaultdict(list)
        for node in g.nodes:
            if is_wait_tensor(node) and filter_node(node.args[0]):
                if (filter_wait_node is None) or filter_wait_node(node):
                    coll_node = node.args[0]
                    key = node_group_key(coll_node)
                    groups[key].append(coll_node)

        if not groups:
            _record_greedy_bucket_diagnostic(
                {
                    "candidate_count": 0,
                    "group_count": 0,
                    "bucket_count": 0,
                    "bucketed_member_count": 0,
                    "dependency_blocked_count": 0,
                    "unbucketed_candidate_count": 0,
                }
            )
            return []

        node_descendents = collect_node_descendants(g)
        max_topo_span = aten_autobucketing_config.max_topo_span

        buckets = []
        # Metrics aggregated across all groups.
        n_close_bytes = 0
        n_close_span = 0
        max_observed_span = 0
        candidate_count = 0
        candidate_bytes = 0
        largest_candidate_bytes = 0
        dependency_blocked_count = 0
        bucketed_member_count = 0
        group_diagnostics = []

        for key, nodes in groups.items():
            group_candidate_bytes = 0
            group_dependency_blocked = 0
            group_bucket_count = 0
            group_bucketed_member_count = 0
            cur_bucket = []
            cur_bucket_descendents = OrderedSet()
            cur_bucket_size_bytes = 0
            cur_bucket_start_rank = None
            cur_bucket_id = 0
            bucket_size_bytes = int(
                bucket_cap_mb_by_bucket_idx(cur_bucket_id) * 1024 * 1024
            )
            for node in nodes:
                n_val = node.meta["val"]
                out_size_bytes = n_val.numel() * n_val.element_size()
                n_input_val = node.all_input_nodes[0].meta["val"]
                in_size_bytes = n_input_val.numel() * n_input_val.element_size()
                size_bytes = max(out_size_bytes, in_size_bytes)
                candidate_count += 1
                candidate_bytes += size_bytes
                group_candidate_bytes += size_bytes
                largest_candidate_bytes = max(largest_candidate_bytes, size_bytes)
                if node in cur_bucket_descendents:
                    dependency_blocked_count += 1
                    group_dependency_blocked += 1
                    continue

                node_rank = ranks.get(node.name, 0)
                would_span = (
                    node_rank - cur_bucket_start_rank
                    if cur_bucket_start_rank is not None
                    else 0
                )

                close_for_bytes = (
                    cur_bucket_size_bytes + size_bytes > bucket_size_bytes
                    and cur_bucket
                )
                close_for_span = (
                    max_topo_span is not None
                    and would_span > max_topo_span
                    and cur_bucket
                )

                if close_for_bytes or close_for_span:
                    if len(cur_bucket) > 1:
                        buckets.append(cur_bucket)
                        bucketed_member_count += len(cur_bucket)
                        group_bucket_count += 1
                        group_bucketed_member_count += len(cur_bucket)
                        if close_for_bytes:
                            n_close_bytes += 1
                        if close_for_span:
                            n_close_span += 1
                        observed_span = (
                            ranks.get(cur_bucket[-1].name, 0) - cur_bucket_start_rank
                        )
                        max_observed_span = max(max_observed_span, observed_span)
                    cur_bucket = []
                    cur_bucket_size_bytes = 0
                    cur_bucket_id += 1
                    cur_bucket_descendents = OrderedSet()
                    cur_bucket_start_rank = None
                    bucket_size_bytes = int(
                        bucket_cap_mb_by_bucket_idx(cur_bucket_id) * 1024 * 1024
                    )
                cur_bucket_size_bytes += size_bytes
                cur_bucket.append(node)
                cur_bucket_descendents |= node_descendents[node]
                if cur_bucket_start_rank is None:
                    cur_bucket_start_rank = node_rank
            if len(cur_bucket) > 1:
                buckets.append(cur_bucket)
                bucketed_member_count += len(cur_bucket)
                group_bucket_count += 1
                group_bucketed_member_count += len(cur_bucket)
                observed_span = (
                    ranks.get(cur_bucket[-1].name, 0) - cur_bucket_start_rank
                )
                max_observed_span = max(max_observed_span, observed_span)
            group_diagnostics.append(
                {
                    "group_key": [str(component) for component in key],
                    "candidate_count": len(nodes),
                    "candidate_bytes": group_candidate_bytes,
                    "bucket_count": group_bucket_count,
                    "bucketed_member_count": group_bucketed_member_count,
                    "dependency_blocked_count": group_dependency_blocked,
                }
            )

        if buckets:
            logger.info(
                "greedy_bucket: %d buckets, max_span=%d, "
                "closed (bytes=%d, span=%d, max_topo_span=%s)",
                len(buckets),
                max_observed_span,
                n_close_bytes,
                n_close_span,
                max_topo_span,
            )
        _record_greedy_bucket_diagnostic(
            {
                "candidate_count": candidate_count,
                "group_count": len(groups),
                "groups": group_diagnostics,
                "candidate_bytes": candidate_bytes,
                "largest_candidate_bytes": largest_candidate_bytes,
                "bucket_count": len(buckets),
                "bucketed_member_count": bucketed_member_count,
                "dependency_blocked_count": dependency_blocked_count,
                "unbucketed_candidate_count": candidate_count - bucketed_member_count,
                "closed_for_bytes_count": n_close_bytes,
                "closed_for_span_count": n_close_span,
                "maximum_bucket_span": max_observed_span,
                "max_topo_span": max_topo_span,
            }
        )
        return buckets

    fsdp_mod.identify_fsdp_groups = _patched_identify_fsdp_groups
    bucketing_mod.greedy_bucket_collective_by_mb = _patched_greedy_bucket


_patch_fsdp_bucketing()


def _max_consec_compute_between_rs(graph):
    """Return the maximum consecutive compute nodes between RS ops.

    Useful as a regression metric: bucketing followed by topo sort can
    pull compute from late layers forward, batching many MMs before any
    RS fires. This metric grows linearly with the size of such batches.
    """
    from torch._inductor.fx_passes.overlap_scheduling import is_compute_node

    max_run = cur = 0
    for n in graph.nodes:
        if n.op != "call_function":
            continue
        target = str(n.target)
        if is_compute_node(n):
            cur += 1
            max_run = max(max_run, cur)
        elif "reduce_scatter" in target and "wait" not in target:
            cur = 0
    return max_run


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
    - max_topo_span: maximum number of graph positions a single bucket may
        span. Bounds how far compute can be displaced when bucketing rewires
        the dep graph and stable_topological_sort runs afterwards. Set to
        None to disable the span bound (only bytes cap applies).
    - bucket_all_all_reduce_groups: include every process group containing an
        all-reduce in all-reduce bucketing. Does not affect all-gather or
        reduce-scatter selection.
    """

    bucket_all_all_reduce_groups = False
    max_in_flight_gb = 2.0
    compute_overlap_multipler = 1.0
    max_coll_distance = 100
    custom_runtime_estimation = None
    max_compute_pre_fetch = 50
    max_topo_span: int | None = 1500
    collective_bucketing = False
    save_trace = True
    _counter = 0


def aten_autobucketing_reordering_pass(
    gm: torch.fx.Graph, configs: "aten_autobucketing_config"
) -> torch.fx.GraphModule:
    assert gm.owning_module is not None

    diagnostics: dict[str, Any] = {
        "collective_counts_before": _collective_counts(gm),
        "group_selection": None,
        "greedy_bucket_calls": [],
    }
    token = _active_bucketing_diagnostics.set(diagnostics)
    try:
        new_gm = schedule_overlap_bucketing(
            gm.owning_module,
            collective_bucketing=configs.collective_bucketing,
            max_compute_pre_fetch=configs.max_compute_pre_fetch,
            custom_runtime_estimation=configs.custom_runtime_estimation,
            compute_overlap_multipler=configs.compute_overlap_multipler,
            max_in_flight_gb=configs.max_in_flight_gb,
            max_coll_distance=configs.max_coll_distance,
        )
    finally:
        _active_bucketing_diagnostics.reset(token)

    diagnostics["collective_counts_after"] = _collective_counts(new_gm.graph)
    new_gm.meta["autoparallel_bucketing_diagnostics"] = diagnostics
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "autoparallel_bucketing_diagnostics",
            "encoding": "json",
        },
        payload_fn=lambda: json.dumps(diagnostics, sort_keys=True),
    )

    logger.info(
        "aten_autobucketing_reordering_pass: post-pass "
        "max_consec_compute_between_rs=%d",
        _max_consec_compute_between_rs(new_gm.graph),
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
