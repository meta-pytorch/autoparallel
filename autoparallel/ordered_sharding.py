# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import Optional, Union

import torch
from torch._functorch._aot_autograd.fx_utils import get_param_and_grad_nodes
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.distributed.tensor._op_schema import OpSpec
from torch.distributed.tensor.placement_types import (  # noqa
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.utils._pytree import tree_flatten

from .dtensor_util.redistribute_tensor import redistribute_local_tensor


def _optimize_same_nd_sharding_as_1d(
    arg: torch.Tensor, curr_spec: DTensorSpec, tgt_spec: DTensorSpec
) -> torch.Tensor:
    """
    This function optimizes the case where the current and target placements
    have the same placements for all mesh dimensions. For example, if the
    current placement is S(0)S(0) and the target placement is RR, this
    function will perform a single collective, instead of two collectives.
    """
    curr_spec_first = curr_spec.placements[0]
    if not all(curr_spec_first == p for p in curr_spec.placements):
        return redistribute_local_tensor(arg, curr_spec, tgt_spec)
    tgt_spec_first = tgt_spec.placements[0]
    if not all(tgt_spec_first == p for p in tgt_spec.placements):
        return redistribute_local_tensor(arg, curr_spec, tgt_spec)

    # TODO: make this more general, I'm playing safe for now
    allowed_placements = [(Shard(0), Replicate()), (Partial(), Shard(0))]
    if (curr_spec_first, tgt_spec_first) not in allowed_placements:
        print(f"NOT doing optimization for {str(curr_spec)} -> {str(tgt_spec)}")
        return redistribute_local_tensor(arg, curr_spec, tgt_spec)

    print(f"Doing optimization for {str(curr_spec)} -> {str(tgt_spec)}")
    mesh = curr_spec.device_mesh
    # TODO: remove ndim == 1 special case once
    # DeviceMesh._flatten is fixed
    if mesh.ndim != 1:
        flat_mesh = mesh._flatten()
    else:
        flat_mesh = mesh
    flat_curr_spec = DTensorSpec(
        flat_mesh, (curr_spec_first,), tensor_meta=curr_spec.tensor_meta
    )
    flat_tgt_spec = DTensorSpec(
        flat_mesh, (tgt_spec_first,), tensor_meta=tgt_spec.tensor_meta
    )
    return redistribute_local_tensor(arg, flat_curr_spec, flat_tgt_spec)


def ordered_redistribute_local_tensor(
    arg: torch.Tensor,
    curr_spec: DTensorSpec,
    tgt_spec: DTensorSpec,
    src_placement_order=None,
    tgt_placement_order=None,
) -> torch.Tensor:
    """
    This is a simplified version of redistribute_local_tensor that optimizes
    a couple of specific cases by introducing an ordering information to the
    placements.

    The optimizations that we support for now are hard-coded, and we should
    generalize this in the future.
    """
    if src_placement_order:
        curr_spec.device_order = src_placement_order
    if tgt_placement_order:
        tgt_spec.device_order = tgt_placement_order
    if src_placement_order == tgt_placement_order:
        return _optimize_same_nd_sharding_as_1d(arg, curr_spec, tgt_spec)
    return redistribute_local_tensor(
        arg,
        curr_spec,
        tgt_spec,
    )


def get_redistributed_input_placements(
    node: torch.fx.Node, sharding_placement: dict[torch.fx.Node, OpSpec]
) -> dict[torch.fx.Node, tuple[tuple[Placement, ...], tuple[Placement, ...]]]:
    """
    This function returns a map of input nodes to their current and target
    placements, for the inputs that need to be redistributed.
    """
    # use this instead of node.all_input_nodes as it handles repeated nodes
    all_input_nodes = [
        x for x in tree_flatten(node.args)[0] if isinstance(x, torch.fx.Node)
    ]
    num_input_nodes = len(all_input_nodes)
    curr_specs: list[Union[DTensorSpec, tuple[Optional[DTensorSpec], ...]]] = [
        sharding_placement[n].output_specs for n in all_input_nodes
    ]  # FIXME ?
    if node.target == operator.getitem:
        # if getitem index is static, then there's no associated fx.Node
        assert (
            len(all_input_nodes) == 1
        ), "getitem with dynamic index not yet supported."
        assert len(curr_specs) == 1 and isinstance(curr_specs[0], (tuple, list))
        assert len(node.args) == 2
        index = node.args[1]
        assert isinstance(index, int)
        assert index < len(curr_specs[0])

        # This looks wrong, and it is wrong.
        # Basically, we need a refactor to properly support getitem.
        # It currently uses the wrong input_specs, see TODO in `getitem_rule`.
        curr_specs = [curr_specs[0][index]]  # type: ignore[assignment, list-item]

    tgt_specs: list[DTensorSpec] = [
        sharding_placement[node].input_specs[c] for c in range(num_input_nodes)  # type: ignore[index]
    ]
    assert len(curr_specs) == len(tgt_specs)

    res = {}
    for i, (curr_spec, tgt_spec) in enumerate(zip(curr_specs, tgt_specs)):
        tgt_placements = tuple(
            p if not p.is_partial() else Replicate() for p in tgt_spec.placements
        )
        if not isinstance(curr_spec, DTensorSpec):
            raise NotImplementedError(
                f"No support for ops with multiple outputs yet: {node.name}"
            )
        if curr_spec.placements != tgt_spec.placements:
            res[all_input_nodes[i]] = (curr_spec.placements, tgt_placements)
    return res


def compute_optimal_placement_order_for_parameters(module, sharding_placement):
    """
    This function computes the optimal placement order for parameters and
    gradients, based on the sharding placement. The optimal placement order is
    defined as the order in which the parameters and gradients should be
    placed, such that the number of communication steps is minimized.

    For now this function only optimizes the case where the parameters are
    distributed as S(0)S(0) -> RS(0) and the gradients are distributed
    as PS(0) -> S(0)S(0). We should generalize this in the future.
    """
    param_and_grad_nodes = list(get_param_and_grad_nodes(module.graph).values())
    # this is actually parameter users and gradient inputs
    # but well, naming is hard
    param_and_grad_users = {}
    param_grad_chain = {}
    for param, grad in param_and_grad_nodes:
        last_p = list(param.users)[0]
        p_chain = [param]
        # get all linear chain of users of the parameter
        while len(last_p.all_input_nodes) == 1:
            p_chain.append(last_p)
            # TODO: we need to handle the case where there are multiple users
            # maybe?
            last_p = list(last_p.users.keys())[0]
        for p in p_chain:
            param_and_grad_users[p] = param
        # order from source to dest
        param_grad_chain[param] = p_chain

        last_g = grad
        g_chain = []
        # get all linear chain of inputs that lead to the gradient
        while len(last_g.all_input_nodes) == 1:
            g_chain.append(last_g)
            last_g = last_g.all_input_nodes[0]
        for p in g_chain:
            param_and_grad_users[p] = grad
        # order from dest to source
        param_grad_chain[grad] = g_chain

    redistribution_map = {}
    mesh_ndim = None
    for user_node, param_or_grad_node in param_and_grad_users.items():
        d = get_redistributed_input_placements(user_node, sharding_placement)
        if d:
            redistribution_map[param_or_grad_node] = (user_node, d)
            if mesh_ndim is None:
                user_src_placement = list(d.values())[0][0]
                mesh_ndim = len(user_src_placement)

    param_grad_map = {p: g for p, g in param_and_grad_nodes}
    aligned_pg = []
    for param_or_grad_node in redistribution_map.keys():
        # just allow for arbitrary execution order if both param and grad
        # are in the map
        if param_or_grad_node in param_grad_map:
            param_node = param_or_grad_node
            grad_node = param_grad_map[param_node]
            if grad_node in redistribution_map:
                aligned_pg.append(
                    (
                        param_node,
                        grad_node,
                        list(redistribution_map[param_node][1].values())[0],
                        list(redistribution_map[grad_node][1].values())[0],
                    )
                )

    # map node from to (target order, need reorder?)
    redistribute_node_order = {}
    for (
        param_node,
        grad_node,
        (node_plc, node_tgt_plc),
        (grad_plc, grad_tgt_plc),
    ) in aligned_pg:
        if node_plc != grad_tgt_plc:
            # TODO: handle this
            print("Skipping", param_node, grad_node, node_plc, grad_tgt_plc)
            continue
        src_input = redistribution_map[param_node][0]
        src_grad = redistribution_map[grad_node][0]
        # Only support S(0)S(0) -> RS(0) and PS(0) -> S(0)S optimizations.
        if node_plc == (Shard(0), Shard(0)) and node_tgt_plc == (
            Replicate(),
            Shard(0),
        ):
            if grad_plc == (Partial(), Shard(0)) and grad_tgt_plc == (
                Shard(0),
                Shard(0),
            ):
                # last node with single input after param use order [0, 1].
                # note: we need to make all front nodes ordered as [1,0]
                # handle forward pass param related nodes
                param_node = param_and_grad_users[src_input]
                param_chain = param_grad_chain[param_node]
                # node that need to be reverse the order from (1,0) to (0,1)
                node_to_reorder = src_input
                # node between [param_and_grad_users[src_input], src_input) are under order [1,0],
                for p in param_chain:
                    if p == node_to_reorder:
                        redistribute_node_order[p] = ((0, 1), True)
                        break
                    else:
                        redistribute_node_order[p] = ((1, 0), False)

                # handle backward pass grad related nodes
                grad_node = param_and_grad_users[src_grad]
                grad_chain = param_grad_chain[grad_node]
                # node that need to be reverse the order from (0,1) to (1,0)
                node_to_reorder = src_grad
                # node between [param_and_grad_users[src_grad], src_grad) are under order [1,0],
                for p in grad_chain:
                    if p == node_to_reorder:
                        redistribute_node_order[p] = ((1, 0), True)
                        break
                    else:
                        # below is supposed not to be triggered
                        redistribute_node_order[p] = ((1, 0), False)
    for node, (order, _) in redistribute_node_order.items():
        node.meta["device_order"] = order
    return redistribute_node_order
