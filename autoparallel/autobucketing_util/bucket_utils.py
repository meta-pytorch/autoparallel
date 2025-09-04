# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# mypy: ignore-errors
from functools import reduce
from typing import Any, Callable, Dict, Union

import torch
from torch._inductor import ir, scheduler
from torch._inductor.dependencies import WeakDep
from torch._inductor.ir import NoneLayout
from torch._inductor.utils import buf_name_to_fused_snode, is_collective, is_wait
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import _resolve_process_group
from torch.utils._ordered_set import OrderedSet


def get_data_size(size):
    return reduce(lambda x, y: x * y, size)


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


def check_ir_node_bucketable(
    ir_node: "ir.IRNode", bucketable_ir_nodes: set[str]
) -> bool:
    """
    Determine if the AG/RS & AG/RS wait node is from bucketable nodes or not
    """
    ir_node_origins = list(getattr(ir_node, "origins", None))
    if len(ir_node_origins) == 0:
        # bucketed AG and RS doesn't have origins
        return True

    if is_wait(ir_node):
        ir_node = ir_node.inputs[0]

    if is_collective(
        ir_node, op=torch.ops._c10d_functional.all_gather_into_tensor.default
    ):
        ir_node_name = ir_node.get_name()
    elif is_collective(
        ir_node, op=torch.ops._c10d_functional.reduce_scatter_tensor.default
    ):
        ir_node_name = ir_node.get_name()
    else:
        return False

    if ir_node_name in bucketable_ir_nodes:
        return True

    return False


def _get_fx_node(
    snode_or_ir_node: Union["scheduler.BaseSchedulerNode", "ir.IRNode"],
    expected_op: Callable[[Any]],
) -> torch.fx.Node:
    origins = None
    if isinstance(snode_or_ir_node, scheduler.BaseSchedulerNode):
        origins = snode_or_ir_node.node.get_origins()
    elif isinstance(snode_or_ir_node, ir.IRNode):
        origins = snode_or_ir_node.origins
    else:
        raise ValueError(
            f"Expected BaseSchedulerNode or IRNode, got {type(snode_or_ir_node)}. Offending value: {snode_or_ir_node}"
        )
    origins_with_expected_op = [o for o in origins if o.target == expected_op]
    if len(origins_with_expected_op) != 1:
        print(
            "[Get FX exception] origins_with_expected_op",
            origins_with_expected_op,
            "expected_op",
            expected_op,
            "snode_or_ir_node",
            snode_or_ir_node,
        )
        return None
    return origins_with_expected_op[0]


def get_snode_process_group_info(
    snode: "scheduler.BaseSchedulerNode",
    expected_op: Callable[[Any]],
    resolve_pg: bool = False,
) -> tuple[int, Union[str, ProcessGroup]]:
    fx_node = _get_fx_node(snode, expected_op=expected_op)
    # return None if the snode doesn't have a valid fx_node
    if fx_node is None:
        return None

    if expected_op == torch.ops._c10d_functional.all_gather_into_tensor.default:
        group_size, group_name = (
            snode.node.constant_args[0],
            snode.node.constant_args[1],
        )
    elif expected_op == torch.ops._c10d_functional.reduce_scatter_tensor.default:
        group_size, group_name = (
            snode.node.constant_args[1],
            snode.node.constant_args[2],
        )
    elif expected_op == torch.ops._c10d_functional.all_reduce_.default:
        group_size, group_name = fx_node.args[1], fx_node.args[2]
    elif expected_op == torch.ops._c10d_functional.all_to_all_single.default:
        group_size, group_name = 0, fx_node.args[3]
    else:
        raise ValueError(f"Unsupported op {expected_op}")

    if resolve_pg:
        group_name = _resolve_process_group(group_name)
    return group_size, group_name


def get_snode_tensor_info(
    snode: "scheduler.BaseSchedulerNode", return_data_size: bool = False
) -> tuple[Any, ...]:
    input_dtype, input_device = (
        snode.node.inputs[0].layout.dtype,
        snode.node.inputs[0].layout.device,
    )
    input_size = get_data_size(snode.node.inputs[0].layout.size)

    if not isinstance(snode.node.layout, NoneLayout):
        output_dtype, output_device = (
            snode.node.layout.dtype,
            snode.node.layout.device,
        )
        output_size = get_data_size(snode.node.layout.size)
    else:
        # In all_reduce, layout is NoneLayout
        # We set output info to be the same as input info as a special treatment
        output_dtype, output_device, output_size = input_dtype, input_device, input_size

    result = (input_dtype, input_device, output_dtype, output_device)
    if return_data_size:
        result += (input_size, output_size)
    return result
