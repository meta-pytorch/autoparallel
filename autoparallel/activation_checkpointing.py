# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import logging
import operator
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch._functorch.partitioners import _has_tag_is_backward, _size_of
from torch.utils._ordered_set import OrderedSet
from torch.utils.checkpoint import CheckpointPolicy

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# add an arbitrarily large graph id. I'm assuming 100000 here, which should be fine
# and is the same we add for the all-gather nodes
AP_AC_GRAPH_ID = 100000


# reimplement torch._functorch.partitioners.must_recompute
# to only check for MUST_RECOMPUTE tag, and not PREFER_RECOMPUTE
# For now there isn't any distinction in the partitioner between both
# and I think this is a bug
def must_recompute(node: torch.fx.Node) -> bool:
    return node.meta.get("recompute", None) is CheckpointPolicy.MUST_RECOMPUTE


def is_graph_input(node: torch.fx.Node) -> bool:
    return node.op == "placeholder"


def is_wait_tensor(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == torch.ops._c10d_functional.wait_tensor.default
    )


def is_all_gather_into_tensor(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == torch.ops._c10d_functional.all_gather_into_tensor.default
    )


def is_wait_tensor_from_fsdp(node: torch.fx.Node) -> bool:
    """
    Returns True if the node is a wait_tensor node that is the result of an all_gather
    that can be arbitrarily prefetched, i.e., if all its recursive inputs are
    single-input operators that leads to a graph input.
    """
    if is_wait_tensor(node) and is_all_gather_into_tensor(node.args[0]):
        n: torch.fx.Node = node.all_input_nodes[0]
        while len(n.all_input_nodes) == 1:
            if is_graph_input(n.all_input_nodes[0]):
                return True
            n = n.all_input_nodes[0]
    return False


# mypy: ignore-errors


def force_recompute_fsdp_all_gather(graph: torch.fx.Graph) -> None:
    """
    Force recompute all_gather nodes from simple fsdp in the graph.
    This pass should be added in torch._inductor.config.joint_custom_post_pass
    """

    def force_recompute_node(node):
        node.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
        # ac_graph_id is used in the partitioner to decide
        # if two nodes which have AC applied come from a different
        # AC regions. This is needed because  nodes in the boundary
        # of two AC regions are marked as MUST_SAVE. In our case
        # we just add a large value of ac_graph_id so that
        # all nodes we tag for recomputation do indeed get recomputed
        # and are not influenced by other nodes in the graph with
        # nearby ac_graph_id values
        node.meta["ac_graph_id"] = AP_AC_GRAPH_ID

    # Make all-gather nodes (and related nodes) recomputable, to circumvent
    # https://github.com/pytorch/pytorch/issues/136433
    for node in graph.nodes:
        if is_wait_tensor_from_fsdp(node):
            ag_node = node.args[0]
            force_recompute_node(ag_node)  # all_gather
            force_recompute_node(node)  # wait_tensor
            # Force-recompute slice that comes after wait
            for user in node.users:
                if (
                    user.op == "call_function"
                    and user.target == torch.ops.aten.slice.Tensor
                ):
                    force_recompute_node(user)
            # Force-recompute potential dtype casts from all_gather
            if (
                ag_node.all_input_nodes[0].op == "call_function"
                and ag_node.args[0].target
                == torch.ops.prims.convert_element_type.default
            ):
                force_recompute_node(ag_node.all_input_nodes[0])


INT_INF = int(1e9)


# NOTE: this is taken from PyTorch partitioner
def get_required_fwd_nodes(
    joint_graph: torch.fx.Graph,
) -> OrderedSet[torch.fx.Node]:
    """
    Return the set of nodes that are required in the forward graph.
    NOTE: this is doing similar things as classify_nodes() in _functorch/partitioenrs.py
            where nodes are classified into three types -- fwd, bwd, and unclaimed
            both bwd and unclaimed nodes have partitioner_tag equal to "is_backward"
    """
    required_fwd_nodes: OrderedSet[torch.fx.Node] = OrderedSet()
    for node in joint_graph.nodes:
        if node.op == "placeholder" and "tangents" in node.target:
            continue
        if node.op == "output":
            continue
        if _has_tag_is_backward(node):
            continue
        required_fwd_nodes.add(node)
    return required_fwd_nodes


# NOTE: this is taken from PyTorch partitioner
def get_node_distance_to_bwd(
    joint_graph: torch.fx.Graph,
    get_required_fwd_nodes: OrderedSet[torch.fx.Node],
) -> dict[torch.fx.Node, int]:
    """
    Compute and return the distance of all nodes to the closest backward node.
    If a node is not an ancestor of a backward node, then its distance is INT_INF.
    NOTE: this is adapted from
    https://github.com/pytorch/pytorch/blob/3196a3aca0f16792820158cfd451cb977f99ac7e/torch/_functorch/partitioners.py#L2089-L2097
    """
    dist_from_bw = {}
    for node in reversed(joint_graph.nodes):
        if node.op == "output":
            dist_from_bw[node] = INT_INF
        elif node not in get_required_fwd_nodes:
            dist_from_bw[node] = 0
        else:
            dist_from_bw[node] = INT_INF
            for user in node.users:
                dist_from_bw[node] = min(dist_from_bw[node], dist_from_bw[user] + 1)
    return dist_from_bw


# NOTE: this is taken from PyTorch partitioner
def get_all_recomputable_forward_nodes(
    joint_graph: torch.fx.Graph,
) -> OrderedSet[torch.fx.Node]:
    """
    Return the set of all forward nodes that are recomputable
    """
    required_fwd_nodes = get_required_fwd_nodes(joint_graph)
    dist_from_bw = get_node_distance_to_bwd(joint_graph, required_fwd_nodes)
    fwd_recomputable_nodes: OrderedSet[torch.fx.Node] = OrderedSet()
    for node in joint_graph.nodes:
        if (
            node in required_fwd_nodes
            and dist_from_bw[node] < INT_INF
            and node.op != "placeholder"
        ):
            fwd_recomputable_nodes.add(node)
    return fwd_recomputable_nodes


def _mark_nodes_as_must_save(must_save_nodes: list[torch.fx.Node]) -> None:
    """
    Given a list of nodes, mark them as must save.
    """
    skipped_nodes = {}
    for node in must_save_nodes:
        if (
            node.meta.get("recompute", None) is not None
            and node.meta["ac_graph_id"] != AP_AC_GRAPH_ID
        ):
            # Let user annotations take precedence
            skipped_nodes[node] = node.meta["recompute"]
            continue
        node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
    print(f"mark_nodes_as_must_save, attempting to mark nodes: {must_save_nodes}")
    print(f"mark_nodes_as_must_save, skipping already marked nodes: {skipped_nodes}")


def mark_nodes_as_must_save_to_stage_recomputation(
    joint_graph: torch.fx.Graph,
    stage_size_in_GiB: Optional[Union[float, str]] = "auto",
) -> None:
    """
    Marks specific nodes as "must save" to optimize memory usage during recomputation.
    With aggressive recomputation strategies, we often encounter situations where long chains
    of forward nodes must be recomputed before executing backward pass nodes, causing high
    peak memory usage. This function breaks these recomputation chains into smaller stages
    based by periodically saving itermediate nodes, keeping peak memory usage below.
    Args:
        joint_graph: The joint graph containing both forward and backward nodes
        stage_size_in_GiB: Target memory size per stage in GiB. None means no stage
            recomputation, "auto" means we use sqrt(total_used_memory) as stage size.
    """
    if stage_size_in_GiB is None:
        return

    fwd_recomputable_nodes = get_all_recomputable_forward_nodes(joint_graph)

    # Initialize all nodes as 'prefer recompute' and then adjust only the must-save ones below
    for node in fwd_recomputable_nodes:
        if node.meta.get("recompute", None) is not None:
            # do not mess with allgather nodes that have already been marked recompute!
            continue
        if node.target is operator.getitem:
            # we need to be a bit careful: we are trying to manually emulate setting "precompute" tags
            # in the same way that compiel does when it encounters userland SAC.
            #
            # torch.compile does this by using TorchDispatchModes to intercept ops as they are traced,
            # and setting their "recompute" tag.
            #
            # However, TorchDispatchModes *only* intercept OpOverloads (and HOPs)
            # getitem is neither, and so in vanilla torch.compile usage,
            # getitem nodes recieve no tags.
            #
            # What happens if we blindly set all nodes to PREFER_RECOMPUTE? Example bad outcome:
            # - user is using attention, so we see this series of ops in the joint graph:
            #     attention_fw -> getitem -> attention_bw (the getitem is an output used for the bw)
            # - user runs SAC, and marks attention_fw as MUST_SAVE
            # - if we mark getitem as PREFER_RECOMPUTE, and attention_fw as MUST_SAVE,
            #   the partitioner ends up generating an invalid graph.
            #   Today the partitioner relies on the fact that getitem's recompute behavior
            #   is implicitly determined by the recompute behavior of the multi-output op preceding it.
            continue
        node.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        node.meta["ac_graph_id"] = AP_AC_GRAPH_ID

    # get the mapping between node name and node
    name_to_node_mapping = {}
    for node in fwd_recomputable_nodes:
        name_to_node_mapping[node.name] = node

    # populate node_to_predecessors, accounting for must_recompute nodes. In particular,
    # if a node is marked as must recompute, then for its users, their predecessors should
    # be updated to be instead the predecessors of the must recompute node.
    node_to_predecessors = defaultdict(OrderedSet)
    for node in fwd_recomputable_nodes:
        node_to_predecessors[node] = OrderedSet(
            [pred for pred in node.all_input_nodes if pred in fwd_recomputable_nodes]
        )
    for node in fwd_recomputable_nodes:
        if not must_recompute(node):
            continue
        for user in node.users:
            if user in fwd_recomputable_nodes:
                node_to_predecessors[user].remove(node)
                node_to_predecessors[user].update(node_to_predecessors[node])

    # populate node_to_last_usage
    # if A is last used by B, then A \in node_to_last_usage[B]
    node_to_last_usage = defaultdict(OrderedSet)
    last_used_by = {}
    for node in fwd_recomputable_nodes:
        last_used_by[node] = node
        for pred in node_to_predecessors[node]:
            last_used_by[pred] = node
    for producer, consumer in last_used_by.items():
        node_to_last_usage[consumer].add(producer)

    # loop through nodes in order of the forward graph and keep track of the following:
    # for each node, right before its execution, the output of what nodes are in memory.
    @dataclass
    class NodeCutScore:
        tot_mem: float
        alive_node_names: OrderedSet[str]

    alive_nodes = OrderedSet()
    node2score = {}
    for node in fwd_recomputable_nodes:
        if not must_recompute(node):
            alive_nodes.add(node)
            for a in node_to_last_usage[node]:
                alive_nodes.remove(a)
        tot_mem = sum(_size_of(node) for node in alive_nodes)
        node2score[node] = NodeCutScore(
            tot_mem, OrderedSet([n.name for n in alive_nodes])
        )

    # divide the graph into stages with roughly equal memory usage.
    stages = defaultdict(OrderedSet)
    cum_mem_so_far = 0
    curr_stage_idx = 0

    if stage_size_in_GiB == "auto":
        total_used_memory = sum(
            _size_of(node)
            for node in fwd_recomputable_nodes
            if not must_recompute(node)
        )
        total_used_memory_in_GiB = total_used_memory / 2**30
        stage_size_in_GiB = total_used_memory_in_GiB**0.5
        print(f"Computed stage_size {stage_size_in_GiB=}")

    target_mem = stage_size_in_GiB * 2**30
    for node in fwd_recomputable_nodes:
        stages[curr_stage_idx].add(node)
        if not must_recompute(node):
            cum_mem_so_far += _size_of(node)
        if cum_mem_so_far >= target_mem:
            curr_stage_idx += 1
            cum_mem_so_far = 0

    # loop through each stage and pick the best node to cut on, and save
    # the nodes that will be marked as must save.
    nodes_to_save = OrderedSet()
    for stage in stages.values():
        best_node = min(stage, key=lambda x: node2score[x].tot_mem)
        nodes_to_save.update(node2score[best_node].alive_node_names)
    _mark_nodes_as_must_save([name_to_node_mapping[n] for n in nodes_to_save])


def _apply_ac_policy(joint_graph: torch.fx.Graph, save_list: set[torch.ops.OpOverload]):
    """
    This is not very generic, and just applies an AC policy similar to what
    TorchTitan is doing. I think we should just replace this altogether with
    torch._functorch.config.activation_memory_budget
    """
    fwd_recomputable_nodes = get_all_recomputable_forward_nodes(joint_graph)
    must_save_nodes = []
    counter = 0
    for node in fwd_recomputable_nodes:
        if node.target in save_list:
            if node.target == torch.ops.aten.mm.default:
                if counter % 2 == 0:
                    counter += 1
                else:
                    counter += 1
                    continue
            must_save_nodes.append(node)
    _mark_nodes_as_must_save(must_save_nodes)


def ac_joint_pass(graph: torch.fx.Graph, ac_stage_size_in_GiB: float = 2.0):
    force_recompute_fsdp_all_gather(graph)
    mark_nodes_as_must_save_to_stage_recomputation(
        graph, stage_size_in_GiB=ac_stage_size_in_GiB
    )

    # TODO: we need to also enable sdpa perfectly mimic the TorchTitan
    # policy, but this is not working yet
    save_list = {
        torch.ops.aten.mm.default,
        torch.ops.aten._scaled_dot_product_efficient_attention.default,
        torch.ops.aten._scaled_dot_product_flash_attention.default,
    }
    _apply_ac_policy(graph, save_list=save_list)
