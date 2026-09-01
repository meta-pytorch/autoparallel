# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import operator
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch._functorch._aot_autograd.fx_utils import get_param_and_grad_nodes
from torch.distributed.tensor._dtensor_spec import (
    DTensorSpec,
    ShardOrder,
    ShardOrderEntry,
)
from torch.distributed.tensor._op_schema import OpSpec
from torch.distributed.tensor._redistribute import redistribute_local_tensor
from torch.distributed.tensor.placement_types import (  # noqa
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.utils._pytree import tree_flatten


@dataclass(frozen=True)
class OrderInfo:
    """Preferred physical shard order for a parameter and its gradient."""

    preferred_shard_order: ShardOrder


def _infer_fsdplike_storage_order(
    source: tuple[Placement, ...],
    target: tuple[Placement, ...],
) -> Optional[ShardOrder]:
    """Infer an order that keeps retained shards outside released shards."""
    if len(source) != len(target):
        return None

    retained: dict[int, list[int]] = defaultdict(list)
    released: dict[int, list[int]] = defaultdict(list)

    for mesh_dim, (src, dst) in enumerate(zip(source, target)):
        if isinstance(src, Shard) and isinstance(dst, Shard):
            if src.dim != dst.dim:
                return None
            retained[src.dim].append(mesh_dim)
        elif isinstance(src, Shard) and isinstance(dst, Replicate):
            released[src.dim].append(mesh_dim)
        elif isinstance(src, Replicate) and isinstance(dst, Replicate):
            continue
        else:
            return None

    if not released or any(not retained[tensor_dim] for tensor_dim in released):
        return None

    preferred_order = tuple(
        ShardOrderEntry(
            tensor_dim=tensor_dim,
            mesh_dims=tuple(
                sorted(retained[tensor_dim])
                + sorted(released[tensor_dim], reverse=True)
            ),
        )
        for tensor_dim in sorted(set(retained) | set(released))
        if retained[tensor_dim] or released[tensor_dim]
    )
    default_order = DTensorSpec.compute_default_shard_order(source)
    return preferred_order if preferred_order != default_order else None


def _matches_inverse_gradient_pattern(
    param_source: tuple[Placement, ...],
    param_target: tuple[Placement, ...],
    grad_source: tuple[Placement, ...],
    grad_target: tuple[Placement, ...],
) -> bool:
    """Check that the gradient redistribution reverses the parameter gather."""
    if grad_target != param_source:
        return False

    expected_grad_source = tuple(
        (
            Partial()
            if isinstance(param_src, Shard) and isinstance(param_dst, Replicate)
            else param_src
        )
        for param_src, param_dst in zip(param_source, param_target)
    )
    return grad_source == expected_grad_source


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
        return redistribute_local_tensor(arg, curr_spec, tgt_spec)

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
) -> torch.Tensor:
    """
    This is a simplified version of redistribute_local_tensor that optimizes
    a couple of specific cases by introducing an ordering information to the
    placements.

    The optimizations that we support for now are hard-coded, and we should
    generalize this in the future.
    """
    # _optimize_same_nd_sharding_as_1d flattens the 2D mesh into 1D and
    # performs a single collective (e.g. S(0)S(0)->RR becomes S(0)->R on
    # the flat mesh).  This is only correct when the data layout matches
    # the flat mesh's natural rank ordering, i.e. both specs use default
    # (ascending) shard_order.  We can't use structural equality here
    # because shard_order tuples have different lengths when the number of
    # sharded dims differs (e.g. S(0)S(0) has one entry while RR has none).
    # is_default_device_order checks ascending mesh_dims per entry (and
    # returns True for the empty tuple), which is the right consistency
    # criterion.
    both_default = DTensorSpec.is_default_device_order(
        curr_spec.shard_order
    ) and DTensorSpec.is_default_device_order(tgt_spec.shard_order)
    if both_default:
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
    curr_specs: list[Union[DTensorSpec, tuple[Optional[DTensorSpec], ...], None]] = [
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


def build_param_grad_linear_chains(
    param_and_grad_nodes: list[tuple[torch.fx.Node, Optional[torch.fx.Node]]],
) -> tuple[
    dict[torch.fx.Node, torch.fx.Node], dict[torch.fx.Node, list[torch.fx.Node]]
]:
    """
    Build linear dependency chains for parameters and their corresponding gradients.

    For each parameter node, traces forward through users to find a linear chain
    of nodes that depend on the parameter (i.e., nodes with single inputs).

    For each gradient node (if present), traces backward through inputs to find
    a linear chain of nodes that lead to the gradient.

    Args:
        param_and_grad_nodes: List of (parameter_node, gradient_node) pairs.
            gradient_node can be None for non-trainable parameters.

    Returns:
        A tuple of two dictionaries:
        - node_to_source: Maps each node in the chains back to its source
          (the original parameter or gradient node).
        - source_to_chain: Maps each parameter/gradient node to its linear chain
          of dependent nodes. For parameters, the chain goes from source to
          the last single-input user. For gradients, the chain goes from the
          gradient node backward to the first multi-input node.
    """
    node_to_source: dict[torch.fx.Node, torch.fx.Node] = {}
    source_to_chain: dict[torch.fx.Node, list[torch.fx.Node]] = {}

    for param, grad in param_and_grad_nodes:
        # Skip unused parameters — no chain to build
        if not param.users:
            continue
        # Build forward chain of users for the parameter
        last_p = list(param.users)[0]
        p_chain: list[torch.fx.Node] = [param]
        # get all linear chain of users of the parameter
        while len(last_p.all_input_nodes) == 1 and len(last_p.users) > 0:
            p_chain.append(last_p)
            # TODO: we need to handle the case where there are multiple users
            # maybe?
            last_p = list(last_p.users.keys())[0]
        for p in p_chain:
            node_to_source[p] = param
        # order from source to dest
        source_to_chain[param] = p_chain

        # TODO: optimize case where parameter doesn't require gradient
        if grad is None:
            continue

        # Build backward chain of inputs for the gradient
        last_g = grad
        g_chain: list[torch.fx.Node] = []
        # get all linear chain of inputs that lead to the gradient
        while len(last_g.all_input_nodes) == 1:
            g_chain.append(last_g)
            last_g = last_g.all_input_nodes[0]
        for p in g_chain:
            node_to_source[p] = grad
        # order from dest to source
        source_to_chain[grad] = g_chain

    return node_to_source, source_to_chain


def _assign_order_info_to_chain(
    chain: list[torch.fx.Node],
    target_node: torch.fx.Node,
    preferred_shard_order: ShardOrder,
    order_map: dict[torch.fx.Node, OrderInfo],
) -> None:
    """
    Assign OrderInfo to nodes in a chain up to and including the target node.

    Args:
        chain: List of nodes in the dependency chain.
        target_node: The redistribution boundary for this chain.
        preferred_shard_order: Parameter storage order to preserve along the chain.
        order_map: Dictionary to populate with OrderInfo for each node.
    """
    for node in chain:
        order_map[node] = OrderInfo(
            preferred_shard_order=preferred_shard_order,
        )
        if node == target_node:
            break


def compute_optimal_placement_order_for_parameters(
    module: torch.fx.GraphModule,
    sharding_placement: dict[torch.fx.Node, OpSpec],
) -> dict[torch.fx.Node, OrderInfo]:
    """
    Compute the optimal placement order for parameters and gradients.

    The optimal placement order minimizes communication needed to remove a
    subset of same-tensor-dimension parameter shards while retaining the rest.

    Args:
        module: The FX GraphModule containing parameter and gradient nodes.
        sharding_placement: Mapping from nodes to their sharding OpSpec.

    Returns:
        Dictionary mapping nodes to the preferred parameter storage order.
    """
    param_and_grad_nodes = list(get_param_and_grad_nodes(module.graph).values())

    node_to_source, source_to_chain = build_param_grad_linear_chains(
        param_and_grad_nodes
    )

    # Build map of source nodes (params/grads) to their redistribution info
    redistribution_map: dict[
        torch.fx.Node,
        tuple[
            torch.fx.Node,
            dict[torch.fx.Node, tuple[tuple[Placement, ...], tuple[Placement, ...]]],
        ],
    ] = {}
    for user_node, source_node in node_to_source.items():
        redistribution_info = get_redistributed_input_placements(
            user_node, sharding_placement
        )
        if redistribution_info and source_node not in redistribution_map:
            redistribution_map[source_node] = (user_node, redistribution_info)

    # Find param-grad pairs where both require redistribution
    param_to_grad_map = dict(param_and_grad_nodes)
    matched_param_grad_pairs: list[
        tuple[
            torch.fx.Node,
            torch.fx.Node,
            tuple[tuple[Placement, ...], tuple[Placement, ...]],
            tuple[tuple[Placement, ...], tuple[Placement, ...]],
        ]
    ] = []
    for source_node in redistribution_map.keys():
        if source_node not in param_to_grad_map:
            continue
        param_node = source_node
        grad_node = param_to_grad_map[param_node]
        if grad_node not in redistribution_map:
            continue
        # Extract (current_placement, target_placement) for param and grad
        param_placements = list(redistribution_map[param_node][1].values())[0]
        grad_placements = list(redistribution_map[grad_node][1].values())[0]
        matched_param_grad_pairs.append(
            (param_node, grad_node, param_placements, grad_placements)
        )

    redistribute_node_order: dict[torch.fx.Node, OrderInfo] = {}

    for (
        param_node,
        grad_node,
        (param_curr_plc, param_tgt_plc),
        (grad_curr_plc, grad_tgt_plc),
    ) in matched_param_grad_pairs:
        # Skip if param source placement doesn't match grad target placement
        if param_curr_plc != grad_tgt_plc:
            continue

        preferred_shard_order = _infer_fsdplike_storage_order(
            param_curr_plc, param_tgt_plc
        )
        if preferred_shard_order is None or not _matches_inverse_gradient_pattern(
            param_curr_plc,
            param_tgt_plc,
            grad_curr_plc,
            grad_tgt_plc,
        ):
            continue

        # Get the user nodes where redistribution occurs
        param_redistrib_node = redistribution_map[param_node][0]
        grad_redistrib_node = redistribution_map[grad_node][0]

        # Preserve the chosen storage order through the forward parameter chain.
        param_source = node_to_source[param_redistrib_node]
        param_chain = source_to_chain[param_source]
        _assign_order_info_to_chain(
            param_chain,
            target_node=param_redistrib_node,
            preferred_shard_order=preferred_shard_order,
            order_map=redistribute_node_order,
        )

        # Restore gradients to the same storage order in the backward chain.
        grad_source = node_to_source[grad_redistrib_node]
        grad_chain = source_to_chain[grad_source]
        _assign_order_info_to_chain(
            grad_chain,
            target_node=grad_redistrib_node,
            preferred_shard_order=preferred_shard_order,
            order_map=redistribute_node_order,
        )

    # Apply shard_order metadata to nodes
    for node, order_info in redistribute_node_order.items():
        node.meta["shard_order"] = order_info.preferred_shard_order

    return redistribute_node_order
