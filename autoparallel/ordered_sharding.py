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


def my_redistribute_local_tensor(arg, curr_spec, tgt_spec, perm=None):
    canonical = tuple(reversed(range(len(curr_spec.placements))))
    if perm is None:
        perm = canonical
    if perm == canonical:
        return redistribute_local_tensor(arg, curr_spec, tgt_spec)
    assert perm == (0, 1), f"{perm}"
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


def get_src_tgt_placements(node, sharding_placement):
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


def compute_node_directions(module, sharding_placement):
    param_and_grad_nodes = list(get_param_and_grad_nodes(module.graph).values())
    param_and_single_user = {}
    grad_and_single_inputs = {}
    for param, grad in param_and_grad_nodes:
        last_p = list(param.users)[0]
        p_chain = [param]
        while len(last_p.all_input_nodes) == 1:
            p_chain.append(last_p)
            # TODO: we need to handle the case where there are multiple users
            # maybe?
            last_p = list(last_p.users.keys())[0]
        for p in p_chain:
            param_and_single_user[p] = param

        last_g = grad
        g_chain = []
        while len(last_g.all_input_nodes) == 1:
            g_chain.append(last_g)
            last_g = last_g.all_input_nodes[0]
        for p in reversed(g_chain):
            grad_and_single_inputs[p] = grad

    combined = {**param_and_single_user, **grad_and_single_inputs}
    redist_map = {}
    mesh_dim = None
    for node, tgt_node in combined.items():

        d = get_src_tgt_placements(node, sharding_placement)
        if d:
            redist_map[tgt_node] = (node, d)
            if mesh_dim is None:
                plc = list(d.values())[0][0]
                mesh_dim = len(plc)

    param_map = {p: g for p, g in param_and_grad_nodes}
    aligned_pg = []
    for node in redist_map.keys():
        # just allow for arbitrary execution order if both param and grad
        # are in the map
        if node in param_map:
            grad_node = param_map[node]
            if grad_node in redist_map:
                aligned_pg.append(
                    (
                        node,
                        grad_node,
                        list(redist_map[node][1].values())[0],
                        list(redist_map[grad_node][1].values())[0],
                    )
                )

    possible_permutations = list(itertools.permutations(range(mesh_dim)))
    default_direction = tuple(reversed(range(mesh_dim)))
    node_directions = {}
    for (
        node,
        grad_node,
        (node_plc, node_tgt_plc),
        (grad_plc, grad_tgt_plc),
    ) in aligned_pg:
        # assert node_plc == grad_tgt_plc, f"{node}, {grad_node}, {node_plc} {grad_tgt_plc}"
        if node_plc != grad_tgt_plc:
            # TODO: handle this
            print("Skipping", node, grad_node, node_plc, grad_tgt_plc)
            continue
        src_tgt_input = (
            redist_map[node][0],
            list(redist_map[node][1].keys())[0],
        )
        src_tgt_grad = (
            redist_map[grad_node][0],
            list(redist_map[grad_node][1].keys())[0],
        )
        node_directions[src_tgt_input] = default_direction
        node_directions[src_tgt_grad] = default_direction
        if node_plc == (Shard(0), Shard(0)) and node_tgt_plc == (
            Replicate(),
            Shard(0),
        ):
            if grad_plc == (Partial(), Shard(0)) and grad_tgt_plc == (
                Shard(0),
                Shard(0),
            ):
                node_directions[src_tgt_input] = possible_permutations[0]
                node_directions[src_tgt_grad] = possible_permutations[0]
    return node_directions
