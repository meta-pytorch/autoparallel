# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch

logger: logging.Logger = logging.getLogger(__name__)


def _is_permute_transpose(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == torch.ops.aten.permute.default
        and isinstance(node.args[1], (list, tuple))
        and list(node.args[1]) == [1, 0]
    )


def _is_all_gather(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == torch.ops._c10d_functional.all_gather_into_tensor.default
    )


def _is_wait_tensor(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == torch.ops._c10d_functional.wait_tensor.default
    )


def fuse_dp_tp_allgather(
    graph: torch.fx.Graph,
    full_group_size: int,
    full_group_name: str,
) -> int:
    """Fuse consecutive dp + tp allgather chains into a single full-mesh allgather.

    Detects the pattern::

        dp_ag    = all_gather(x, dp_size, dp_pg)
        dp_wait  = wait_tensor(dp_ag)
        p1       = permute(dp_wait, [1, 0])
        p2       = permute(p1, [1, 0])
        tp_ag    = all_gather(p2, tp_size, tp_pg)
        tp_wait  = wait_tensor(tp_ag)

    and replaces it with::

        full_ag   = all_gather(x, dp_size * tp_size, full_pg)
        full_wait = wait_tensor(full_ag)

    The two permutes must cancel (both ``[1, 0]`` transposes), the dp wait
    must have a single user (the first permute), and the dp and tp group
    sizes must multiply to ``full_group_size``.

    Returns the number of fusions performed.
    """
    fusions = 0

    # Snapshot the node list — we mutate the graph during iteration.
    all_nodes = list(graph.nodes)

    for tp_ag in all_nodes:
        if not _is_all_gather(tp_ag):
            continue

        # --- trace backwards through the expected chain ---

        p2 = tp_ag.args[0]
        if not isinstance(p2, torch.fx.Node) or not _is_permute_transpose(p2):
            continue
        if len(p2.users) != 1:
            continue

        p1 = p2.args[0]
        if not isinstance(p1, torch.fx.Node) or not _is_permute_transpose(p1):
            continue
        if len(p1.users) != 1:
            continue

        dp_wait = p1.args[0]
        if not isinstance(dp_wait, torch.fx.Node) or not _is_wait_tensor(dp_wait):
            continue
        if len(dp_wait.users) != 1:
            continue

        dp_ag = dp_wait.args[0]
        if not isinstance(dp_ag, torch.fx.Node) or not _is_all_gather(dp_ag):
            continue
        if len(dp_ag.users) != 1:
            continue

        # --- validate group sizes ---

        dp_group_size = dp_ag.args[1]
        tp_group_size = tp_ag.args[1]
        if dp_group_size * tp_group_size != full_group_size:
            continue

        # --- validate matching dtype ---

        dp_val = dp_ag.meta.get("val")
        tp_val = tp_ag.meta.get("val")
        if dp_val is not None and tp_val is not None and dp_val.dtype != tp_val.dtype:
            continue

        # --- find the tp wait (sole user of tp_ag) ---

        tp_wait = None
        for user in tp_ag.users:
            if _is_wait_tensor(user):
                tp_wait = user
                break
        if tp_wait is None:
            continue

        # --- build the fused allgather ---

        original_input = dp_ag.args[0]

        with graph.inserting_before(tp_ag):
            full_ag = graph.call_function(
                torch.ops._c10d_functional.all_gather_into_tensor.default,
                args=(original_input, full_group_size, full_group_name),
            )
            full_ag.meta.update(tp_ag.meta)

            full_wait = graph.call_function(
                torch.ops._c10d_functional.wait_tensor.default,
                args=(full_ag,),
            )
            full_wait.meta.update(tp_wait.meta)

        tp_wait.replace_all_uses_with(full_wait)
        fusions += 1

        logger.debug(
            "Fused dp_ag(%s, gs=%d) + tp_ag(gs=%d) -> full_ag(gs=%d)",
            original_input,
            dp_group_size,
            tp_group_size,
            full_group_size,
        )

    if fusions > 0:
        graph.eliminate_dead_code()
        logger.info(
            "Fused %d dp+tp allgather chains into full-mesh allgathers", fusions
        )

    return fusions
