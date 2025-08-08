# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
from torch._functorch._aot_autograd.fx_utils import get_param_and_grad_nodes
from torch.distributed.tensor._redistribute import redistribute_local_tensor
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard  # noqa
from torch.utils._pytree import tree_flatten


def ordered_redistribute_local_tensor(arg, curr_spec, tgt_spec, placement_order=None):
    """
    This is a simplified version of redistribute_local_tensor that optimizes
    a couple of specific cases by introducing an ordering information to the
    placements.

    The optimizations that we support for now are hard-coded, and we should
    generalize this in the future.
    """
    canonical = tuple(reversed(range(len(curr_spec.placements))))
    if placement_order is None:
        placement_order = canonical
    if placement_order == canonical:
        return redistribute_local_tensor(arg, curr_spec, tgt_spec)
    assert placement_order == (0, 1), f"{placement_order}"
    if curr_spec.placements == (Shard(0), Shard(0)) and tgt_spec.placements == (
        Replicate(),
        Shard(0),
    ):
        # TODO: double-check in which cases this is valid
        x = curr_spec.placements[0]._to_replicate_tensor(
            arg, curr_spec.mesh, 0, curr_spec.shape
        )
    elif curr_spec.placements == (Partial(), Shard(0)) and tgt_spec.placements == (
        Shard(0),
        Shard(0),
    ):
        x = curr_spec.placements[0]._reduce_shard_value(
            arg, curr_spec.mesh, 0, tgt_spec.placements[0]
        )
    elif curr_spec.placements == (Partial(), Shard(1)) and tgt_spec.placements == (
        Replicate(),
        Shard(1),
    ):
        # from IPython import embed; embed(); sys.sdf
        raise NotImplementedError("Not implemented yet in here")
    else:
        raise ValueError("Shouldn't be here")
        x = redistribute_local_tensor(arg, curr_spec, tgt_spec)
    return x


def get_redistributed_input_placements(node, sharding_placement):
    """
    This function returns a map of input nodes to their current and target
    placements, for the inputs that need to be redistributed.
    """
    # use this instead of node.all_input_nodes as it handles repeated nodes
    all_input_nodes = [
        x for x in tree_flatten(node.args)[0] if isinstance(x, torch.fx.Node)
    ]
    num_input_nodes = len(all_input_nodes)
    curr_specs = [
        sharding_placement[n].output_specs for n in all_input_nodes
    ]  # FIXME ?
    tgt_specs = [
        sharding_placement[node].input_specs[c] for c in range(num_input_nodes)
    ]

    res = {}
    for i, (curr_spec, tgt_spec) in enumerate(zip(curr_specs, tgt_specs)):
        tgt_placements = tuple(
            p if not p.is_partial() else Replicate() for p in tgt_spec.placements
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
            param_and_grad_users[p] = grad

        last_g = grad
        g_chain = []
        # get all linear chain of inputs that lead to the gradient
        while len(last_g.all_input_nodes) == 1:
            g_chain.append(last_g)
            last_g = last_g.all_input_nodes[0]
        for p in reversed(g_chain):
            param_and_grad_users[p] = grad

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

    possible_orderings = list(itertools.permutations(range(mesh_ndim)))
    default_order = tuple(reversed(range(mesh_ndim)))
    param_placement_order = {}
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
        src_tgt_input = (
            redistribution_map[param_node][0],
            list(redistribution_map[param_node][1].keys())[0],
        )
        src_tgt_grad = (
            redistribution_map[grad_node][0],
            list(redistribution_map[grad_node][1].keys())[0],
        )
        param_placement_order[src_tgt_input] = default_order
        param_placement_order[src_tgt_grad] = default_order
        # Only support S(0)S(0) -> RS(0) and PS(0) -> S(0)S optimizations
        # for now, giving them (0, 1) ordering (instead of canonical (1, 0))
        if node_plc == (Shard(0), Shard(0)) and node_tgt_plc == (
            Replicate(),
            Shard(0),
        ):
            if grad_plc == (Partial(), Shard(0)) and grad_tgt_plc == (
                Shard(0),
                Shard(0),
            ):
                param_placement_order[src_tgt_input] = possible_orderings[0]
                param_placement_order[src_tgt_grad] = possible_orderings[0]
    return param_placement_order
