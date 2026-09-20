# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math
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
from torch.distributed.tensor._redistribute import (
    _gen_transform_infos_non_cached,
    _optimize_transform_infos,
    redistribute_local_tensor,
)
from torch.distributed.tensor.placement_types import (  # noqa
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.utils._pytree import tree_flatten

from ..cost_models.collective_runtime_estimation import redistribute_cost


@dataclass(frozen=True)
class OrderInfo:
    """Preferred physical shard order for a parameter and its gradient."""

    preferred_shard_order: ShardOrder


@dataclass(frozen=True)
class _FallbackPlan:
    operations: tuple[tuple[str, tuple[int, ...]], ...]
    cost: float

    @property
    def has_all_to_all(self) -> bool:
        return any(kind == "all_to_all" for kind, _ in self.operations)


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


def _matches_adjoint_gradient_pattern(
    param_source: tuple[Placement, ...],
    param_target: tuple[Placement, ...],
    grad_source: tuple[Placement, ...],
    grad_target: tuple[Placement, ...],
) -> bool:
    """Check that the gradient redistribution is the parameter gather's adjoint."""
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


# Keep the old private name for callers of the original PR #529 helper.
_matches_inverse_gradient_pattern = _matches_adjoint_gradient_pattern


def _project_shard_order(
    preferred_shard_order: ShardOrder,
    spec: DTensorSpec,
) -> ShardOrder:
    """Project a parameter storage order onto the shards present in ``spec``."""
    actual_shards = {
        (placement.dim, mesh_dim)
        for mesh_dim, placement in enumerate(spec.placements)
        if isinstance(placement, Shard)
    }
    covered_shards: set[tuple[int, int]] = set()
    projected_order = []

    for entry in preferred_shard_order:
        matching_mesh_dims = []
        for mesh_dim in entry.mesh_dims:
            if mesh_dim >= len(spec.placements):
                continue
            placement = spec.placements[mesh_dim]
            if isinstance(placement, Shard) and placement.dim == entry.tensor_dim:
                matching_mesh_dims.append(mesh_dim)
        mesh_dims = tuple(matching_mesh_dims)
        if mesh_dims:
            projected_order.append(
                ShardOrderEntry(tensor_dim=entry.tensor_dim, mesh_dims=mesh_dims)
            )
            covered_shards.update(
                (entry.tensor_dim, mesh_dim) for mesh_dim in mesh_dims
            )

    if covered_shards != actual_shards:
        assert spec.shard_order is not None
        return spec.shard_order
    return tuple(projected_order)


def _spec_with_shard_order(spec: DTensorSpec, shard_order: ShardOrder) -> DTensorSpec:
    return DTensorSpec(
        spec.mesh,
        spec.placements,
        tensor_meta=spec.tensor_meta,
        shard_order=shard_order,
        use_strided_shard_as_shard_order=False,
    )


def _flattened_specs(
    curr_spec: DTensorSpec, tgt_spec: DTensorSpec
) -> Optional[tuple[DTensorSpec, DTensorSpec]]:
    """Return the 1-D specs this redistribution collapses to, if it does.

    Mirrors the condition of ``_optimize_same_nd_sharding_as_1d`` so that the
    cost gate prices the same plan the runtime will emit.
    """
    curr_spec_first = curr_spec.placements[0]
    if not all(curr_spec_first == p for p in curr_spec.placements):
        return None
    tgt_spec_first = tgt_spec.placements[0]
    if not all(tgt_spec_first == p for p in tgt_spec.placements):
        return None

    # TODO: make this more general, I'm playing safe for now
    allowed_placements = [(Shard(0), Replicate()), (Partial(), Shard(0))]
    if (curr_spec_first, tgt_spec_first) not in allowed_placements:
        return None

    mesh = curr_spec.device_mesh
    # TODO: remove ndim == 1 special case once
    # DeviceMesh._flatten is fixed
    if mesh.ndim != 1:
        flat_mesh = mesh._flatten()
    else:
        flat_mesh = mesh
    return (
        DTensorSpec(flat_mesh, (curr_spec_first,), tensor_meta=curr_spec.tensor_meta),
        DTensorSpec(flat_mesh, (tgt_spec_first,), tensor_meta=tgt_spec.tensor_meta),
    )


def _fallback_plan(source: DTensorSpec, target: DTensorSpec) -> Optional[_FallbackPlan]:
    """Propagate one redistribution through DTensor and price its actual steps."""
    if source.mesh != target.mesh or source.tensor_meta is None:
        return None
    if DTensorSpec.is_default_device_order(
        source.shard_order
    ) and DTensorSpec.is_default_device_order(target.shard_order):
        flattened = _flattened_specs(source, target)
        if flattened is not None:
            flat_source, flat_target = flattened
            cost = float(redistribute_cost(flat_source, flat_target, [0]))
            if not math.isfinite(cost):
                return None
            kind = (
                "all_gather"
                if isinstance(flat_source.placements[0], Shard)
                else "reduce_scatter"
            )
            return _FallbackPlan(
                ((kind, tuple(range(len(source.placements)))),),
                cost,
            )
    try:
        transforms = _optimize_transform_infos(
            _gen_transform_infos_non_cached(
                source,
                target,
                use_graph_based_transform=True,
            ),
            source.mesh,
            source.placements,
            target.placements,
        )
        current_placements = list(source.placements)
        operations = []
        cost = 0.0
        for transform in transforms:
            mesh_dims = tuple(
                getattr(transform, "original_mesh_dims", (transform.mesh_dim,))
            )
            src, dst = transform.src_dst_placements
            if any(current_placements[mesh_dim] != src for mesh_dim in mesh_dims):
                return None
            next_placements = list(current_placements)
            for mesh_dim in mesh_dims:
                next_placements[mesh_dim] = dst
            current = DTensorSpec(
                source.mesh,
                tuple(current_placements),
                tensor_meta=source.tensor_meta,
            )
            next_spec = DTensorSpec(
                source.mesh,
                tuple(next_placements),
                tensor_meta=source.tensor_meta,
            )
            cost += float(redistribute_cost(current, next_spec, list(mesh_dims)))
            operations.append((transform._comm_type_key() or "local", mesh_dims))
            current_placements = next_placements
    except (AssertionError, RuntimeError, TypeError, ValueError):
        return None

    if tuple(current_placements) != target.placements or not math.isfinite(cost):
        return None
    return _FallbackPlan(tuple(operations), cost)


def _optimize_same_nd_sharding_as_1d(
    arg: torch.Tensor, curr_spec: DTensorSpec, tgt_spec: DTensorSpec
) -> torch.Tensor:
    """
    This function optimizes the case where the current and target placements
    have the same placements for all mesh dimensions. For example, if the
    current placement is S(0)S(0) and the target placement is RR, this
    function will perform a single collective, instead of two collectives.
    """
    flattened = _flattened_specs(curr_spec, tgt_spec)
    if flattened is None:
        return redistribute_local_tensor(arg, curr_spec, tgt_spec)
    flat_curr_spec, flat_tgt_spec = flattened
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


def _consumer_boundary_for_input(
    consumer: torch.fx.Node,
    input_node: torch.fx.Node,
    sharding_placement: dict[torch.fx.Node, OpSpec],
) -> Optional[tuple[tuple[Placement, ...], tuple[Placement, ...]]]:
    """The redistribution a multi-input consumer asks for on one of its inputs.

    ``build_param_grad_linear_chains`` stops before any consumer with more than
    one input, so a weight whose chain ends there has no redistribution between
    chain nodes: the only boundary is the consumer's input spec. That is the
    shape of a weight passed straight into a ``local_map`` region, which is how
    MoE expert weights reach their experts. Indexing by argument position keeps
    the consumer's other inputs out, which ``get_redistributed_input_placements``
    cannot do because it reports the whole node at once.
    """
    consumer_spec = sharding_placement.get(consumer)
    source_spec = sharding_placement.get(input_node)
    if consumer_spec is None or source_spec is None:
        return None
    if consumer_spec.input_specs is None:
        return None
    arg_nodes = [
        value
        for value in tree_flatten(consumer.args)[0]
        if isinstance(value, torch.fx.Node)
    ]
    positions = [i for i, node in enumerate(arg_nodes) if node is input_node]
    if len(positions) != 1 or positions[0] >= len(consumer_spec.input_specs):
        return None
    source = source_spec.output_specs
    target = consumer_spec.input_specs[positions[0]]
    if not isinstance(source, DTensorSpec) or not isinstance(target, DTensorSpec):
        return None
    if source.placements == target.placements:
        return None
    return (
        source.placements,
        tuple(
            Replicate() if placement.is_partial() else placement
            for placement in target.placements
        ),
    )


def _get_chain_redistributions(
    chain: list[torch.fx.Node],
    sharding_placement: dict[torch.fx.Node, OpSpec],
) -> Optional[list[tuple[torch.fx.Node, DTensorSpec, DTensorSpec]]]:
    """Return every unambiguous redistribution on a linear tensor chain."""
    result = []
    for node in chain:
        input_nodes = [
            value
            for value in tree_flatten(node.args)[0]
            if isinstance(value, torch.fx.Node)
        ]
        if not input_nodes:
            continue
        if len(input_nodes) != 1 or node.target == operator.getitem:
            return None
        node_spec = sharding_placement.get(node)
        input_spec = sharding_placement.get(input_nodes[0])
        if node_spec is None or input_spec is None or node_spec.input_specs is None:
            return None
        if len(node_spec.input_specs) != 1:
            return None
        source = input_spec.output_specs
        target = node_spec.input_specs[0]
        if not isinstance(source, DTensorSpec) or not isinstance(target, DTensorSpec):
            return None
        if source.placements == target.placements:
            continue
        target = DTensorSpec(
            target.mesh,
            tuple(
                Replicate() if isinstance(placement, Partial) else placement
                for placement in target.placements
            ),
            tensor_meta=target.tensor_meta,
        )
        result.append((node, source, target))
    return result


def _unambiguous_param_prefix(chain: list[torch.fx.Node]) -> list[torch.fx.Node]:
    """Return the part of a parameter chain the storage order can reach.

    ``_add_alias`` leaves a node with two users — the forward consumer and the
    backward one — in the middle of every weight chain, and
    ``build_param_grad_linear_chains`` walks straight past it.  Everything after
    that node reads a value the gate has already required to be back in default
    order, so only the prefix ending at it can carry the reordered layout.
    """
    prefix = chain[:1]
    for current, next_node in zip(chain, chain[1:]):
        if len(current.users) != 1 or next(iter(current.users)) is not next_node:
            break
        prefix = prefix + [next_node]
    return prefix


def _unambiguous_grad_prefix(chain: list[torch.fx.Node]) -> list[torch.fx.Node]:
    """Same, for a gradient chain, which runs from the gradient to its producer."""
    prefix = chain[:1]
    for current, next_node in zip(chain, chain[1:]):
        if current.all_input_nodes != [next_node] or len(next_node.users) != 1:
            break
        prefix = prefix + [next_node]
    return prefix


def _reordered_mesh_dims(
    placements: tuple[Placement, ...], preferred_order: ShardOrder
) -> frozenset[int]:
    default = {
        entry.tensor_dim: entry.mesh_dims
        for entry in DTensorSpec.compute_default_shard_order(placements)
    }
    preferred = {entry.tensor_dim: entry.mesh_dims for entry in preferred_order}
    return frozenset(
        mesh_dim
        for tensor_dim in default.keys() | preferred.keys()
        if default.get(tensor_dim) != preferred.get(tensor_dim)
        for mesh_dim in default.get(tensor_dim, ()) + preferred.get(tensor_dim, ())
    )


def _logical_collective_order(
    edges: list[tuple[torch.fx.Node, DTensorSpec, DTensorSpec]],
    relevant_mesh_dims: frozenset[int],
    *,
    backward: bool,
) -> Optional[tuple[int, ...]]:
    """Return relevant AG/RS axes, rejecting non-FSDP logical transitions."""
    order = []
    for _, source, target in edges:
        if source.mesh != target.mesh or len(source.placements) != len(
            target.placements
        ):
            return None
        for mesh_dim, (src, dst) in enumerate(
            zip(source.placements, target.placements)
        ):
            if src == dst:
                continue
            is_expected = (
                isinstance(src, Partial) and isinstance(dst, Shard)
                if backward
                else isinstance(src, Shard) and isinstance(dst, Replicate)
            )
            if not is_expected:
                return None
            if mesh_dim in relevant_mesh_dims:
                if mesh_dim in order:
                    return None
                order.append(mesh_dim)
    return tuple(order)


def _candidate_edge(
    edge: tuple[torch.fx.Node, DTensorSpec, DTensorSpec],
    preferred_order: ShardOrder,
) -> tuple[torch.fx.Node, DTensorSpec, DTensorSpec]:
    node, source, target = edge
    return (
        node,
        _spec_with_shard_order(source, _project_shard_order(preferred_order, source)),
        _spec_with_shard_order(target, _project_shard_order(preferred_order, target)),
    )


def _default_edge(
    edge: tuple[torch.fx.Node, DTensorSpec, DTensorSpec],
) -> tuple[torch.fx.Node, DTensorSpec, DTensorSpec]:
    node, source, target = edge
    return (
        node,
        _spec_with_shard_order(
            source, DTensorSpec.compute_default_shard_order(source.placements)
        ),
        _spec_with_shard_order(
            target, DTensorSpec.compute_default_shard_order(target.placements)
        ),
    )


def _plans_for_edges(
    edges: list[tuple[torch.fx.Node, DTensorSpec, DTensorSpec]],
) -> Optional[list[_FallbackPlan]]:
    plans = [_fallback_plan(source, target) for _, source, target in edges]
    if any(plan is None for plan in plans):
        return None
    return [plan for plan in plans if plan is not None]


def _planned_collective_order(
    plans: list[_FallbackPlan],
    relevant_mesh_dims: frozenset[int],
    expected_kind: str,
) -> Optional[tuple[int, ...]]:
    order = []
    for plan in plans:
        for kind, mesh_dims in plan.operations:
            for mesh_dim in mesh_dims:
                if mesh_dim not in relevant_mesh_dims:
                    continue
                if kind != expected_kind:
                    return None
                order.append(mesh_dim)
    return tuple(order)


def _multi_boundary_adjoint_improves_fallback(
    storage: DTensorSpec,
    param_chain: list[torch.fx.Node],
    grad_chain: list[torch.fx.Node],
    sharding_placement: dict[torch.fx.Node, OpSpec],
    preferred_order: ShardOrder,
) -> bool:
    """Gate PR #529's multi-boundary extension on the concrete fallback plans."""
    param_prefix = _unambiguous_param_prefix(param_chain)
    grad_prefix = _unambiguous_grad_prefix(grad_chain)
    if not param_prefix or not grad_prefix:
        return False
    forward = _get_chain_redistributions(param_prefix, sharding_placement)
    backward_from_storage = _get_chain_redistributions(grad_prefix, sharding_placement)
    if not forward or not backward_from_storage:
        return False
    backward = list(reversed(backward_from_storage))
    first_source = forward[0][1]
    last_source, last_target = backward[-1][1:]
    if first_source.placements != storage.placements:
        return False
    if last_target.placements != storage.placements:
        return False
    if first_source.tensor_meta is None or storage.tensor_meta is None:
        return False
    if (
        first_source.shape != storage.shape
        or first_source.stride != storage.stride
        or last_target.shape != storage.shape
        or last_target.stride != storage.stride
    ):
        return False

    reordered_mesh_dims = _reordered_mesh_dims(storage.placements, preferred_order)
    forward_order = _logical_collective_order(
        forward, reordered_mesh_dims, backward=False
    )
    backward_order = _logical_collective_order(
        backward, reordered_mesh_dims, backward=True
    )
    # The gradient may arrive already sharded on some of the axes the forward
    # gathered, so it only has to restore a subset of them.
    if (
        not forward_order
        or not backward_order
        or not set(backward_order).issubset(forward_order)
    ):
        return False

    expected_grad_source = tuple(
        Partial() if mesh_dim in backward_order else placement
        for mesh_dim, placement in enumerate(storage.placements)
    )
    if last_source.placements != expected_grad_source:
        return False

    baseline_forward = [_default_edge(edge) for edge in forward]
    baseline_backward = [_default_edge(edge) for edge in backward]
    candidate_forward = list(baseline_forward)
    candidate_backward = list(baseline_backward)
    candidate_forward[0] = _candidate_edge(forward[0], preferred_order)
    candidate_backward[-1] = _candidate_edge(backward[-1], preferred_order)

    # The non-default order must exist only at the two storage boundaries.
    if (
        candidate_forward[0][2].shard_order != baseline_forward[0][2].shard_order
        or candidate_backward[-1][1].shard_order != baseline_backward[-1][1].shard_order
    ):
        return False

    baseline_plans = _plans_for_edges(baseline_forward + baseline_backward)
    candidate_forward_plans = _plans_for_edges(candidate_forward)
    candidate_backward_plans = _plans_for_edges(candidate_backward)
    if (
        baseline_plans is None
        or candidate_forward_plans is None
        or candidate_backward_plans is None
    ):
        return False
    candidate_plans = candidate_forward_plans + candidate_backward_plans
    if not any(plan.has_all_to_all for plan in baseline_plans):
        return False
    if any(plan.has_all_to_all for plan in candidate_plans):
        return False
    if sum(plan.cost for plan in candidate_plans) >= sum(
        plan.cost for plan in baseline_plans
    ):
        return False
    planned_forward_order = _planned_collective_order(
        candidate_forward_plans, reordered_mesh_dims, "all_gather"
    )
    if planned_forward_order != forward_order:
        return False
    expected_backward_order = tuple(
        mesh_dim
        for mesh_dim in reversed(planned_forward_order)
        if mesh_dim in backward_order
    )
    if (
        _planned_collective_order(
            candidate_backward_plans, reordered_mesh_dims, "reduce_scatter"
        )
        != expected_backward_order
    ):
        return False

    try:
        physical_storage = DTensorSpec._convert_shard_order_to_StridedShard(
            preferred_order, storage.placements, storage.mesh
        )
        physical_grad = DTensorSpec._convert_shard_order_to_StridedShard(
            _project_shard_order(preferred_order, last_target),
            last_target.placements,
            last_target.mesh,
        )
    except (AssertionError, RuntimeError, TypeError, ValueError):
        return False
    return physical_storage == physical_grad


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

    # A parameter whose chain ends at a multi-input consumer has its only
    # boundary on that consumer's input spec, which the loop above cannot see.
    for param_node, _ in param_and_grad_nodes:
        if param_node in redistribution_map:
            continue
        chain = source_to_chain.get(param_node)
        if not chain:
            continue
        terminal = chain[-1]
        for consumer in terminal.users:
            boundary = _consumer_boundary_for_input(
                consumer, terminal, sharding_placement
            )
            if boundary is not None:
                redistribution_map[param_node] = (consumer, {terminal: boundary})
                break

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
        if preferred_shard_order is None:
            # A forward that releases every shard leaves the storage order
            # unconstrained, so derive it from the gradient boundary instead:
            # the axes the gradient keeps sharded must stay outside the ones it
            # restores from Partial.
            preferred_shard_order = _infer_fsdplike_storage_order(
                grad_tgt_plc,
                tuple(
                    Replicate() if isinstance(placement, Partial) else placement
                    for placement in grad_curr_plc
                ),
            )
        if preferred_shard_order is None:
            continue

        # Get the user nodes where redistribution occurs
        param_redistrib_node = redistribution_map[param_node][0]
        grad_redistrib_node = redistribution_map[grad_node][0]

        param_chain = source_to_chain[param_node]
        grad_chain = source_to_chain[grad_node]
        exact_adjoint = _matches_adjoint_gradient_pattern(
            param_curr_plc,
            param_tgt_plc,
            grad_curr_plc,
            grad_tgt_plc,
        )
        if not exact_adjoint:
            storage = sharding_placement[param_node].output_specs
            if not isinstance(storage, DTensorSpec):
                continue
            if not _multi_boundary_adjoint_improves_fallback(
                storage,
                param_chain,
                grad_chain,
                sharding_placement,
                preferred_shard_order,
            ):
                continue

        # Preserve the chosen storage order through the forward parameter chain.
        _assign_order_info_to_chain(
            param_chain,
            target_node=param_redistrib_node,
            preferred_shard_order=preferred_shard_order,
            order_map=redistribute_node_order,
        )

        # Restore gradients to the same storage order in the backward chain.
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
