# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# mypy: ignore-errors
from typing import Any, Callable, Dict

import torch
from torch._inductor import scheduler
from torch._inductor.dependencies import WeakDep
from torch._inductor.utils import buf_name_to_fused_snode, is_collective
from torch.utils._ordered_set import OrderedSet


def _find_recursive_deps_of_snode(
    snode: "scheduler.BaseSchedulerNode",
    collected_node_set: OrderedSet["scheduler.BaseSchedulerNode"],
    name_to_buf: Dict[str, "scheduler.SchedulerBuffer"],
    name_to_fused_node: Dict[str, "scheduler.BaseSchedulerNode"],
    criteria_cb: Callable[[Any], bool] = lambda snode: False,
    allow_weak_dep: bool = True,
):
    if criteria_cb(snode):
        return
    collected_node_set.add(snode)
    for dep in snode.unmet_dependencies:
        if isinstance(dep, WeakDep) and not allow_weak_dep:
            continue
        defining_op_for_dep = buf_name_to_fused_snode(
            dep.name, name_to_buf, name_to_fused_node
        )
        if defining_op_for_dep in collected_node_set:
            continue
        _find_recursive_deps_of_snode(
            defining_op_for_dep,
            collected_node_set,
            name_to_buf,
            name_to_fused_node,
            criteria_cb=criteria_cb,
        )


def _find_recursive_users_of_snode(
    snode: "scheduler.BaseSchedulerNode",
    collected_node_set: OrderedSet["scheduler.BaseSchedulerNode"],
    name_to_buf: Dict[str, "scheduler.SchedulerBuffer"],
    name_to_fused_node: Dict[str, "scheduler.BaseSchedulerNode"],
    criteria_cb: Callable[[Any], bool] = lambda snode: False,
):
    if criteria_cb(snode):
        return
    collected_node_set.add(snode)
    for o in snode.get_outputs():
        for user in o.users:
            assert user.node is not None
            if user.node.get_name() == "OUTPUT":
                continue
            if user.node.get_name() not in name_to_fused_node:
                continue
            user_op = name_to_fused_node[user.node.get_name()]
            if user_op in collected_node_set:
                continue
            _find_recursive_users_of_snode(
                user_op,
                collected_node_set,
                name_to_buf,
                name_to_fused_node,
                criteria_cb=criteria_cb,
            )


def get_bucketable_ir_nodes(
    snodes: list["torch._inductor.scheduler.BaseSchedulerNode"],
    name_to_fused_node: Dict[str, "scheduler.BaseSchedulerNode"],
    name_to_buf: Dict[str, "scheduler.SchedulerBuffer"],
) -> set[str]:
    """
    This function selects the ir nodes' names that are bucketable
    From first principle, only all-gathers that gather parameters and reduce-scatters
    that update model gradients could be bucketed together.
    Thus, bucketable all-gathers's deps are (1) computed buffer for dtype conversion (optional)
        (2) all-gather itself
    bucketable reduce-scatter wait's users are (1) reduce-scatter wait itself
    """
    bucketable_ir_nodes = set()
    for snode in snodes:
        if is_collective(
            snode.node, op=torch.ops._c10d_functional.all_gather_into_tensor.default
        ):
            ag_related_snode_set: OrderedSet[
                "torch._inductor.scheduler.BaseSchedulerNode"
            ] = OrderedSet()
            _find_recursive_deps_of_snode(
                snode,
                ag_related_snode_set,
                name_to_buf,
                name_to_fused_node,
                allow_weak_dep=False,
            )
            if len(ag_related_snode_set) <= 2:
                bucketable_ir_nodes.add(snode.node.get_name())
        elif is_collective(
            snode.node, op=torch.ops._c10d_functional.reduce_scatter_tensor.default
        ):
            wait_snode = snode.get_outputs()[0].users[0].node
            wait_snode_recursive_users: OrderedSet[
                "torch._inductor.scheduler.BaseSchedulerNode"
            ] = OrderedSet()
            _find_recursive_users_of_snode(
                wait_snode,
                wait_snode_recursive_users,
                name_to_buf,
                name_to_fused_node,
            )
            if len(wait_snode_recursive_users) <= 1:
                bucketable_ir_nodes.add(snode.node.get_name())

    return bucketable_ir_nodes
