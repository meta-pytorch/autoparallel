# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch

logger: logging.Logger = logging.getLogger(__name__)


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


def _is_nontrivial_dim_reorder(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    if node.target == torch.ops.aten.t.default:
        return True
    if node.target == torch.ops.aten.transpose.int:
        return node.args[1] != node.args[2]
    if node.target == torch.ops.aten.permute.default and isinstance(
        node.args[1], (list, tuple)
    ):
        dims = list(node.args[1])
        return dims != list(range(len(dims)))
    return False


def _is_identity_view_chain(start: torch.fx.Node, end: torch.fx.Node) -> bool:
    """Check that the view-op chain from start to end composes to the identity.

    Walks forward from ``start`` through single-user view ops and verifies
    that the composed transformation doesn't change the data layout.
    Uses FakeTensor metadata: if the output of ``start`` and the input of
    ``end`` have the same shape and stride, the chain is an identity
    (no data rearrangement, just metadata changes that cancel).

    Only allows ops that are true views (no data copy, no element removal):
    permute, transpose, t, view, reshape, expand, unsqueeze, squeeze.
    Rejects slice (can drop elements) and any non-view op.

    Returns False for empty chains or chains with no non-trivial dimension
    reorder, since consecutive allgathers on different subgroups have
    incompatible rank orderings without explicit layout reconciliation.
    """
    _ALLOWED_VIEW_OPS = frozenset(
        {
            torch.ops.aten.permute.default,
            torch.ops.aten.transpose.int,
            torch.ops.aten.t.default,
            torch.ops.aten.view.default,
            torch.ops.aten.reshape.default,
            torch.ops.aten.expand.default,
            torch.ops.aten.unsqueeze.default,
            torch.ops.aten.squeeze.default,
            torch.ops.aten.squeeze.dim,
        }
    )

    # Reject empty chains: no view ops means no layout reconciliation.
    users = list(start.users.keys())
    if len(users) == 1 and users[0] is end:
        return False

    start_val = start.meta.get("val")
    if start_val is None:
        return False
    start_stride = start_val.stride()

    # Walk forward from start to end, verifying all intermediate ops are
    # allowed views and that some op actually reorders dimensions.
    node = start
    saw_dim_reorder = False
    while node is not end:
        users = list(node.users.keys())
        if len(users) != 1:
            return False
        node = users[0]
        if node is end:
            break
        if node.op != "call_function" or node.target not in _ALLOWED_VIEW_OPS:
            return False
        if _is_nontrivial_dim_reorder(node):
            saw_dim_reorder = True

    if not saw_dim_reorder:
        return False

    # Verify the composed transformation is identity via FakeTensor metadata.
    ag2_input = end.args[0]
    end_val = (
        ag2_input.meta.get("val") if isinstance(ag2_input, torch.fx.Node) else None
    )

    if end_val is None:
        return False
    if start_val.shape != end_val.shape:
        return False
    if start_stride != end_val.stride():
        return False
    return True


def fuse_chained_allgathers(
    graph: torch.fx.Graph,
    full_group_size: int,
    full_group_name: str,
    subgroup_order: dict[str, int] | None = None,
    reversed_full_group_name: str | None = None,
) -> int:
    """Fuse consecutive allgather chains on different subgroups into a single allgather.

    Detects chains of two allgathers on different process groups connected
    either directly (wait1 feeds into ag2) or through single-user view ops
    that compose to the identity::

        ag1      = all_gather(x, size1, pg1)
        wait1    = wait_tensor(ag1)
        ...      = [optional identity_view_ops(wait1)]
        ag2      = all_gather(..., size2, pg2)
        wait2    = wait_tensor(ag2)

    and replaces them with::

        full_ag   = all_gather(x, size1 * size2, full_pg)
        full_wait = wait_tensor(full_ag)

    The correct flattened process group depends on the chain direction:
    - Descending mesh-dim order (tp before dp): ``full_group_name``
      (row-major flat mesh).
    - Ascending mesh-dim order (dp before tp): ``reversed_full_group_name``
      (column-major flat mesh).

    Direct chains (no view ops between AGs) are accepted when
    ``subgroup_order`` validates the direction.  Without ``subgroup_order``,
    an identity view chain is still required for safety.

    Returns the number of fusions performed.
    """
    fusions = 0
    all_nodes = list(graph.nodes)

    for ag2 in all_nodes:
        if not _is_all_gather(ag2):
            continue

        # Walk ag2's input backward through single-user nodes to find wait1.
        node = ag2.args[0]
        if not isinstance(node, torch.fx.Node):
            continue

        # Find the wait_tensor that starts the chain.
        wait1 = node
        while not _is_wait_tensor(wait1):
            if len(wait1.users) != 1:
                break
            if len(wait1.args) == 0:
                break
            inp = wait1.args[0]
            if not isinstance(inp, torch.fx.Node):
                break
            wait1 = inp

        if not _is_wait_tensor(wait1):
            continue
        if len(wait1.users) != 1:
            continue

        ag1 = wait1.args[0]
        if not isinstance(ag1, torch.fx.Node) or not _is_all_gather(ag1):
            continue
        if len(ag1.users) != 1:
            continue

        # Validate that the view chain between wait1 and ag2 is identity,
        # or that it's a direct chain (wait1 feeds ag2 with no intermediate ops).
        is_direct_chain = ag2.args[0] is wait1
        if not is_direct_chain and not _is_identity_view_chain(wait1, ag2):
            continue

        # Validate group sizes.
        ag1_group_size = ag1.args[1]
        ag2_group_size = ag2.args[1]
        if ag1_group_size * ag2_group_size != full_group_size:
            continue

        # Validate group names.
        ag1_group = ag1.args[2]
        ag2_group = ag2.args[2]
        assert isinstance(ag1_group, str)
        assert isinstance(ag2_group, str)
        if ag1_group == ag2_group:
            continue

        # Determine the correct flat mesh based on chain direction.
        if subgroup_order is not None:
            if ag1_group not in subgroup_order or ag2_group not in subgroup_order:
                continue
            ag1_dim = subgroup_order[ag1_group]
            ag2_dim = subgroup_order[ag2_group]
            if ag1_dim < ag2_dim:
                # Ascending (dp→tp): needs column-major flat mesh
                if reversed_full_group_name is None:
                    continue
                target_group_name = reversed_full_group_name
            elif ag1_dim > ag2_dim:
                # Descending (tp→dp): needs row-major flat mesh
                target_group_name = full_group_name
            else:
                continue
        else:
            # Without subgroup_order, require identity view chain for safety.
            if is_direct_chain:
                continue
            target_group_name = full_group_name

        # Validate matching dtype.
        ag1_val = ag1.meta.get("val")
        ag2_val = ag2.meta.get("val")
        if (
            ag1_val is not None
            and ag2_val is not None
            and ag1_val.dtype != ag2_val.dtype
        ):
            continue

        # Find wait2.
        wait2 = None
        for user in ag2.users:
            if _is_wait_tensor(user):
                wait2 = user
                break
        if wait2 is None:
            continue

        # Build the fused allgather.
        original_input = ag1.args[0]

        with graph.inserting_before(ag2):
            full_ag = graph.call_function(
                torch.ops._c10d_functional.all_gather_into_tensor.default,
                args=(original_input, full_group_size, target_group_name),
            )
            full_ag.meta.update(ag2.meta)

            full_wait = graph.call_function(
                torch.ops._c10d_functional.wait_tensor.default,
                args=(full_ag,),
            )
            full_wait.meta.update(wait2.meta)

        wait2.replace_all_uses_with(full_wait)
        fusions += 1

        logger.debug(
            "Fused ag(%s, gs=%d, pg=%s) + ag(gs=%d, pg=%s) -> ag(gs=%d, pg=%s)",
            original_input,
            ag1_group_size,
            ag1_group,
            ag2_group_size,
            ag2_group,
            full_group_size,
            target_group_name,
        )

    if fusions > 0:
        graph.eliminate_dead_code()
        logger.info(
            "Fused %d chained allgather pairs into full-mesh allgathers", fusions
        )

    return fusions
