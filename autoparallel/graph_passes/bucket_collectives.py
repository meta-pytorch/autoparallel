# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import defaultdict

import torch
from torch._inductor.fx_passes.bucketing import (
    collect_node_descendants,
    is_all_gather_into_tensor,
    is_all_reduce_tensor,
    is_reduce_scatter_tensor,
    merge_all_gather_bucket,
    merge_all_reduce_bucket,
    merge_reduce_scatter_bucket,
)
from torch._inductor.fx_passes.post_grad import stable_topological_sort
from torch.utils._ordered_set import OrderedSet

from .graph_utils import build_param_derived_set, build_terminal_derived_set

logger = logging.getLogger(__name__)

# Meta key used to tag collectives eligible for bucketing.
# Set to "param" for param-derived (forward all-gathers) or
# "terminal" for terminal-derived (backward reduce-scatters / all-reduces).
AP_BUCKET_KEY = "ap_bucket_group"


def tag_collectives_for_bucketing(graph: torch.fx.Graph) -> None:
    """Tag FSDP/DDP collectives on the joint graph for later bucketing.

    Must run on the joint graph where placeholder metadata is available.
    The tags survive partitioning into fw/bw subgraphs via node_copy's
    shallow copy of node.meta.
    """
    param_derived = build_param_derived_set(graph)
    terminal_derived = build_terminal_derived_set(graph)

    n_ag = 0
    n_rs = 0
    n_ar = 0
    for node in graph.nodes:
        if is_all_gather_into_tensor(node) and node in param_derived:
            node.meta[AP_BUCKET_KEY] = "param"
            n_ag += 1
        elif is_reduce_scatter_tensor(node) and node in terminal_derived:
            node.meta[AP_BUCKET_KEY] = "terminal"
            n_rs += 1
        elif is_all_reduce_tensor(node) and node in terminal_derived:
            node.meta[AP_BUCKET_KEY] = "terminal"
            n_ar += 1

    logger.info(
        "Tagged collectives for bucketing: AG %d, RS %d, AR %d",
        n_ag,
        n_rs,
        n_ar,
    )


def _group_key(node: torch.fx.Node) -> tuple:
    """Extract group key for any collective type.

    The key ensures only collectives on the same process group / reduce op /
    dtype can be bucketed together — the same constraints enforced by
    PyTorch's merge functions.
    """
    if is_all_gather_into_tensor(node):
        _, group_size, group_name = node.args
        return (group_name,)
    elif is_reduce_scatter_tensor(node):
        _, reduce_op, group_size, group_name = node.args
        dtype = node.meta["val"].dtype
        return (group_name, reduce_op, dtype)
    elif is_all_reduce_tensor(node):
        _, reduce_op, group_name = node.args
        dtype = node.meta["val"].dtype
        return (group_name, reduce_op, dtype)
    else:
        raise ValueError(f"Unsupported collective type: {node.target}")


def _greedy_bucket(
    graph: torch.fx.Graph,
    coll_nodes: list[torch.fx.Node],
    bucket_cap_bytes: int,
    node_descendants: dict[torch.fx.Node, OrderedSet[torch.fx.Node]],
) -> list[list[torch.fx.Node]]:
    """Group collectives into buckets up to bucket_cap_bytes.

    Unlike PyTorch's greedy_bucket_collective_by_mb, this does not require
    collectives to be adjacent in the graph — it only requires that no
    collective is a descendant of another in the same bucket (to avoid
    creating cycles when merged).
    """
    if not coll_nodes:
        return []

    groups: dict[tuple, list[torch.fx.Node]] = defaultdict(list)
    for node in coll_nodes:
        groups[_group_key(node)].append(node)

    buckets: list[list[torch.fx.Node]] = []
    for nodes in groups.values():
        cur_bucket: list[torch.fx.Node] = []
        cur_bucket_descendants: OrderedSet[torch.fx.Node] = OrderedSet()
        cur_bucket_bytes = 0

        for node in nodes:
            if node in cur_bucket_descendants:
                continue

            val = node.meta["val"]
            out_bytes = val.numel() * val.element_size()
            in_val = node.all_input_nodes[0].meta["val"]
            in_bytes = in_val.numel() * in_val.element_size()
            size_bytes = max(out_bytes, in_bytes)

            if cur_bucket_bytes + size_bytes > bucket_cap_bytes and cur_bucket:
                if len(cur_bucket) > 1:
                    buckets.append(cur_bucket)
                cur_bucket = []
                cur_bucket_bytes = 0
                cur_bucket_descendants = OrderedSet()

            cur_bucket_bytes += size_bytes
            cur_bucket.append(node)
            cur_bucket_descendants |= node_descendants[node]

        if len(cur_bucket) > 1:
            buckets.append(cur_bucket)

    return buckets


def bucket_collectives(
    gm: torch.fx.GraphModule,
    bucket_cap_mb: float = 25.0,
) -> None:
    """Bucket FSDP/DDP collectives before the overlap scheduler runs.

    Reads the tags set by tag_collectives_for_bucketing() to identify
    which collectives to merge. Merges per-parameter collectives into
    larger bucketed collectives so the overlap scheduler sees fewer,
    larger ops.
    """
    graph = gm.graph

    fsdp_all_gathers: list[torch.fx.Node] = []
    fsdp_reduce_scatters: list[torch.fx.Node] = []
    ddp_all_reduces: list[torch.fx.Node] = []

    for node in graph.nodes:
        bucket_group = node.meta.get(AP_BUCKET_KEY)
        if bucket_group is None:
            continue
        if is_all_gather_into_tensor(node):
            fsdp_all_gathers.append(node)
        elif is_reduce_scatter_tensor(node):
            fsdp_reduce_scatters.append(node)
        elif is_all_reduce_tensor(node):
            ddp_all_reduces.append(node)

    total = len(fsdp_all_gathers) + len(fsdp_reduce_scatters) + len(ddp_all_reduces)
    if total < 2:
        return

    node_descendants = collect_node_descendants(graph)
    bucket_cap_bytes = int(bucket_cap_mb * 1024 * 1024)

    ag_buckets = _greedy_bucket(
        graph, fsdp_all_gathers, bucket_cap_bytes, node_descendants
    )
    rs_buckets = _greedy_bucket(
        graph, fsdp_reduce_scatters, bucket_cap_bytes, node_descendants
    )
    ar_buckets = _greedy_bucket(
        graph, ddp_all_reduces, bucket_cap_bytes, node_descendants
    )

    n_merged = sum(len(b) for b in ag_buckets + rs_buckets + ar_buckets)
    n_buckets = len(ag_buckets) + len(rs_buckets) + len(ar_buckets)
    if n_buckets == 0:
        return

    logger.info(
        "Bucketing %d collectives into %d buckets "
        "(AG: %d->%d, RS: %d->%d, AR: %d->%d)",
        n_merged,
        n_buckets,
        len(fsdp_all_gathers),
        len(fsdp_all_gathers) - sum(len(b) for b in ag_buckets) + len(ag_buckets),
        len(fsdp_reduce_scatters),
        len(fsdp_reduce_scatters) - sum(len(b) for b in rs_buckets) + len(rs_buckets),
        len(ddp_all_reduces),
        len(ddp_all_reduces) - sum(len(b) for b in ar_buckets) + len(ar_buckets),
    )

    for bucket in ag_buckets:
        merge_all_gather_bucket(graph, bucket)
    for bucket in rs_buckets:
        merge_reduce_scatter_bucket(graph, bucket)
    for bucket in ar_buckets:
        merge_all_reduce_bucket(graph, bucket)

    # Bucketing can place new nodes (concat, split, etc.) at positions that
    # break use-before-def ordering. Re-sort to fix this — same as PyTorch's
    # post_grad_passes does after its own FX bucketing.
    stable_topological_sort(graph)
    gm.recompile()
