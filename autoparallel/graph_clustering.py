# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import math
from collections import defaultdict
from typing import Optional

import torch
from torch._dynamo.graph_region_tracker import (
    Any,
    IdenticalNodes,
    InputPickler,
    Node,
    Region,
    _populate_recursive_ancestor_map,
    fully_expand_region_group,
    operator,
    tree_flatten,
)
from torch._inductor.codecache import sha256_hash


def _extract_args(arg: Any) -> Any:
    if isinstance(arg, Node):
        return arg.meta.get("val")
    elif isinstance(arg, (torch.Tensor, int)):
        return arg
    else:
        return None


def _normalize_args(
    node: Node,
) -> tuple[tuple[str, ...], tuple[Optional[Any], ...]]:
    flat_args, _ = tree_flatten(node.args)
    sorted_kwargs = sorted(node.kwargs.items(), key=operator.itemgetter(0))
    sorted_keys = tuple(sorted(node.kwargs.keys()))
    flat_kwargs, _ = tree_flatten(sorted_kwargs)
    all_args = flat_args + flat_kwargs
    return (sorted_keys, tuple(_extract_args(arg) for arg in all_args))


def _hash_node(node, input_pickler):
    key = (
        node.meta.get("stack_trace"),
        _normalize_args(node),
    )
    return sha256_hash(input_pickler.dumps(key))


def get_identical_regions(graph: torch.fx.Graph) -> list[list[Region]]:
    """
    This function is responsible for extracting the largest regions of identical nodes from the given graph.
    **Note**: This function assumes the nodes that have been tracked with track_node are in the provided graph argument.

    The algorithm proceeds as follows:
    The nodes tracked via track_node above are organized into region groups. The initial region groups look like this:
    [[IdenticalNode1], [IdenticalNode2], [IdenticalNode3]] and each sublist is called a region. For each region group
    (starting at the topologically latest region group), the inner regions are gradually expanded one node at time from
    the flattened args and kwargs of the node in each region provided that for all regions in the group, the nodes being
    added are also identical (ie have the same key computed by track_node). This is checked by verifying that the two
    nodes have the same identical node list in node_to_duplicates.
    """
    topological_ranking = {node: i for i, node in enumerate(graph.nodes)}
    region_groups_with_rank = []
    # needed to detect if replacing a region will create cycles
    node_to_recursive_ancestors = _populate_recursive_ancestor_map(graph)

    input_pickler = InputPickler()
    hash_to_duplicates: dict[str, IdenticalNodes] = defaultdict(list)
    node_to_duplicates: dict[Node, IdenticalNodes] = {}
    for node in graph.nodes:
        if node.op == "placeholder":
            continue

        duplicates = hash_to_duplicates[_hash_node(node, input_pickler)]
        duplicates.append(node)
        node_to_duplicates[node] = duplicates

    def _is_identical(n0: Node, n1: Node) -> bool:
        return (
            n0 in node_to_duplicates
            and n1 in node_to_duplicates
            and node_to_duplicates[n0] is node_to_duplicates[n1]
            and n0 is not n1
        )

    # Create region groups; a region group is a group
    # of regions that are all identical. In this initial state
    # each region in the group is a single node, and we discard
    # groups that are only a single region.
    # We track the topological ranking to start with groups later in the graph
    # the reason for this is that we will necessarily create the largest groups first.
    for group in hash_to_duplicates.values():
        if len(group) > 1:
            region_group = []
            min_rank = math.inf
            for node in group:
                # some nodes aren't in the topo ranking?
                if node in topological_ranking:
                    min_rank = min(min_rank, topological_ranking[node])
                    region_group.append([node])

            if len(region_group) > 1:
                region_groups_with_rank.append((region_group, min_rank))

    region_groups_with_rank.sort(key=lambda rg: -rg[1])
    region_groups = [rg for rg, _ in region_groups_with_rank]

    # We start from regions later in the graph and expand them earlier
    # as a result, we will create the largest regions first and they won't
    # overlap.
    seen_nodes: set[Node] = set()
    for region_group in region_groups:
        fully_expand_region_group(
            region_group,
            seen_nodes,
            node_to_recursive_ancestors,
            _is_identical,
        )
        # sort topologically
        for region in region_group:
            region.sort(key=lambda n: topological_ranking[n])

    region_groups = [
        region_group for region_group in region_groups if len(region_group[0]) > 1
    ]
    for region_group in region_groups:
        region_group.sort(key=lambda rg: topological_ranking[rg[0]])
    region_groups.sort(key=lambda rg: topological_ranking[rg[0][0]])
    return region_groups
