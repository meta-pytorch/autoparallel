# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math
import operator
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional, Union

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
    _StridedShard,
)
from torch.utils._pytree import tree_flatten

from ..cost_models.collective_runtime_estimation import redistribute_cost


@dataclass(frozen=True)
class OrderInfo:
    """Preferred physical shard order for a parameter and its gradient."""

    preferred_shard_order: ShardOrder


@dataclass(frozen=True)
class _FallbackPlanSummary:
    """Communication chosen by DTensor for one logical redistribution."""

    operations: tuple[tuple[str, tuple[int, ...]], ...]
    cost: float

    @property
    def has_all_to_all(self) -> bool:
        return any(kind == "all_to_all" for kind, _ in self.operations)


def _can_optimize_same_nd_sharding_as_1d(
    source: tuple[Placement, ...],
    target: tuple[Placement, ...],
) -> bool:
    if not source or len(source) != len(target):
        return False

    src = source[0]
    dst = target[0]
    if not all(src == placement for placement in source):
        return False
    if not all(dst == placement for placement in target):
        return False

    return (src, dst) in {
        (Shard(0), Replicate()),
        (Partial(), Shard(0)),
    }


def _infer_pure_release_storage_order(
    source: tuple[Placement, ...],
    target: tuple[Placement, ...],
) -> Optional[ShardOrder]:
    """Infer storage order for a redistribution that only releases shards."""
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

    if (
        not released
        or any(not retained[tensor_dim] for tensor_dim in released)
        or _can_optimize_same_nd_sharding_as_1d(source, target)
    ):
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


def _spec_with_shard_order(spec: DTensorSpec, shard_order: ShardOrder) -> DTensorSpec:
    return DTensorSpec(
        spec.mesh,
        spec.placements,
        tensor_meta=spec.tensor_meta,
        shard_order=shard_order,
        use_strided_shard_as_shard_order=False,
    )


def _project_shard_order(
    preferred_shard_order: ShardOrder,
    spec: DTensorSpec,
) -> ShardOrder:
    """Project a storage order onto the shards present in ``spec``."""
    actual_shards = {
        (placement.dim, mesh_dim)
        for mesh_dim, placement in enumerate(spec.placements)
        if isinstance(placement, Shard)
    }
    covered_shards: set[tuple[int, int]] = set()
    projected_order = []

    for entry in preferred_shard_order:
        mesh_dims = tuple(
            mesh_dim
            for mesh_dim in entry.mesh_dims
            if mesh_dim < len(spec.placements)
            and isinstance(placement := spec.placements[mesh_dim], Shard)
            and placement.dim == entry.tensor_dim
        )
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


def _plan_cost(
    source: DTensorSpec,
    transforms: list[Any],
) -> Optional[float]:
    """Price the concrete DTensor fallback path with AutoParallel's cost model."""
    current_placements = list(source.placements)
    cost = 0.0
    try:
        for transform in transforms:
            mesh_dims = tuple(
                getattr(transform, "original_mesh_dims", (transform.mesh_dim,))
            )
            target_placement = transform.src_dst_placements[1]
            next_placements = list(current_placements)
            for mesh_dim in mesh_dims:
                next_placements[mesh_dim] = target_placement
            current = DTensorSpec(
                source.mesh,
                tuple(current_placements),
                tensor_meta=source.tensor_meta,
            )
            target = DTensorSpec(
                source.mesh,
                tuple(next_placements),
                tensor_meta=source.tensor_meta,
            )
            cost += float(redistribute_cost(current, target, list(mesh_dims)))
            current_placements = next_placements
    except (AssertionError, RuntimeError, TypeError, ValueError):
        return None
    return cost if math.isfinite(cost) else None


def _summarize_fallback_plan(
    source: DTensorSpec,
    target: DTensorSpec,
) -> Optional[_FallbackPlanSummary]:
    """Run the same graph-based propagation used by DTensor lowering."""
    if source.mesh != target.mesh or source.tensor_meta is None:
        return None
    try:
        raw = _gen_transform_infos_non_cached(
            source,
            target,
            use_graph_based_transform=True,
        )
        transforms = _optimize_transform_infos(
            raw,
            source.mesh,
            source.placements,
            target.placements,
        )
        operations = tuple(
            (
                transform._comm_type_key() or "local",
                tuple(
                    getattr(
                        transform,
                        "original_mesh_dims",
                        (transform.mesh_dim,),
                    )
                ),
            )
            for transform in transforms
        )
    except (AssertionError, RuntimeError, TypeError, ValueError):
        return None

    cost = _plan_cost(source, transforms)
    if cost is None:
        return None
    return _FallbackPlanSummary(operations=operations, cost=cost)


def _logical_plan_has_no_all_to_all(
    source: DTensorSpec,
    target: DTensorSpec,
) -> bool:
    """Return whether an AutoParallel placement edge requires no logical A2A."""
    if source.mesh != target.mesh or len(source.placements) != len(target.placements):
        return False
    for current, desired in zip(source.placements, target.placements):
        if (
            isinstance(current, Shard)
            and isinstance(desired, Shard)
            and current.dim != desired.dim
        ):
            return False
    try:
        logical_cost = float(
            redistribute_cost(source, target, list(range(source.mesh.ndim)))
        )
    except (AssertionError, RuntimeError, TypeError, ValueError):
        return False
    return math.isfinite(logical_cost)


def _same_logical_spec(left: DTensorSpec, right: DTensorSpec) -> bool:
    if left.mesh != right.mesh or left.placements != right.placements:
        return False
    if left.tensor_meta is None or right.tensor_meta is None:
        return left.tensor_meta is right.tensor_meta
    return (
        left.tensor_meta.shape == right.tensor_meta.shape
        and left.tensor_meta.stride == right.tensor_meta.stride
        and left.tensor_meta.dtype == right.tensor_meta.dtype
    )


def _candidate_specs(
    source: DTensorSpec,
    target: DTensorSpec,
    preferred_shard_order: ShardOrder,
) -> tuple[DTensorSpec, DTensorSpec]:
    return (
        _spec_with_shard_order(
            source,
            _project_shard_order(preferred_shard_order, source),
        ),
        _spec_with_shard_order(
            target,
            _project_shard_order(preferred_shard_order, target),
        ),
    )


def _physical_storage_is_checkpoint_compatible(
    storage: DTensorSpec,
    preferred_shard_order: ShardOrder,
) -> bool:
    """Check that the physical order is decodable and evenly sharded for DCP."""
    if storage.tensor_meta is None:
        return False
    try:
        global_shape = tuple(int(size) for size in storage.tensor_meta.shape)
        mesh_shape = tuple(int(size) for size in storage.mesh.shape)
        physical = DTensorSpec._convert_shard_order_to_StridedShard(
            preferred_shard_order,
            storage.placements,
            storage.mesh,
        )
        normalized, decoded_order = DTensorSpec._normalize_placements_into_shard_order(
            physical,
            storage.mesh,
        )
    except (AssertionError, RuntimeError, TypeError, ValueError):
        return False
    if normalized != storage.placements or decoded_order != preferred_shard_order:
        return False

    for tensor_dim, size in enumerate(global_shape):
        shard_factor = math.prod(
            mesh_shape[mesh_dim]
            for mesh_dim, placement in enumerate(physical)
            if isinstance(placement, (Shard, _StridedShard))
            and placement.dim == tensor_dim
        )
        if size % shard_factor:
            return False

    return True


def _fallback_improves_layout_only_redistribution(
    storage: DTensorSpec,
    forward_edges: list[tuple[DTensorSpec, DTensorSpec]],
    backward_edges: list[tuple[DTensorSpec, DTensorSpec]],
    preferred_shard_order: ShardOrder,
) -> bool:
    """Gate a storage-order change on the complete forward/backward paths."""
    if not forward_edges or not backward_edges:
        return False
    first_forward_source, first_forward_target = forward_edges[0]
    last_backward_source, last_backward_target = backward_edges[-1]
    if first_forward_source.placements != storage.placements:
        return False
    if not _same_logical_spec(storage, last_backward_target):
        return False

    if _matches_adjoint_gradient_pattern(
        first_forward_source.placements,
        first_forward_target.placements,
        last_backward_source.placements,
        last_backward_target.placements,
    ):
        forward_edges = [forward_edges[0]]
        backward_edges = [backward_edges[-1]]

    default_forward_edges = [
        (
            _spec_with_shard_order(
                source, DTensorSpec.compute_default_shard_order(source.placements)
            ),
            _spec_with_shard_order(
                target, DTensorSpec.compute_default_shard_order(target.placements)
            ),
        )
        for source, target in forward_edges
    ]
    default_backward_edges = [
        (
            _spec_with_shard_order(
                source, DTensorSpec.compute_default_shard_order(source.placements)
            ),
            _spec_with_shard_order(
                target, DTensorSpec.compute_default_shard_order(target.placements)
            ),
        )
        for source, target in backward_edges
    ]
    candidate_forward_edges = list(default_forward_edges)
    candidate_forward_edges[0] = _candidate_specs(
        first_forward_source,
        first_forward_target,
        preferred_shard_order,
    )
    candidate_backward_edges = list(default_backward_edges)
    candidate_backward_edges[-1] = _candidate_specs(
        last_backward_source,
        last_backward_target,
        preferred_shard_order,
    )

    # The non-default order is allowed only at storage boundaries.  All
    # intervening view/compute edges must return to ordinary DTensor order.
    if (
        candidate_forward_edges[0][1].shard_order
        != default_forward_edges[0][1].shard_order
        or candidate_backward_edges[-1][0].shard_order
        != default_backward_edges[-1][0].shard_order
    ):
        return False
    if any(
        not _same_logical_spec(default, candidate)
        for default_edges, candidate_edges in (
            (default_forward_edges, candidate_forward_edges),
            (default_backward_edges, candidate_backward_edges),
        )
        for default_edge, candidate_edge in zip(default_edges, candidate_edges)
        for default, candidate in zip(default_edge, candidate_edge)
    ):
        return False

    baseline_forward = [
        _summarize_fallback_plan(source, target)
        for source, target in default_forward_edges
    ]
    candidate_forward = [
        _summarize_fallback_plan(source, target)
        for source, target in candidate_forward_edges
    ]
    baseline_backward = [
        _summarize_fallback_plan(source, target)
        for source, target in default_backward_edges
    ]
    candidate_backward = [
        _summarize_fallback_plan(source, target)
        for source, target in candidate_backward_edges
    ]
    plans = [
        *baseline_forward,
        *candidate_forward,
        *baseline_backward,
        *candidate_backward,
    ]
    if any(plan is None for plan in plans):
        return False
    typed_baseline_forward = [plan for plan in baseline_forward if plan is not None]
    typed_candidate_forward = [plan for plan in candidate_forward if plan is not None]
    typed_baseline_backward = [plan for plan in baseline_backward if plan is not None]
    typed_candidate_backward = [plan for plan in candidate_backward if plan is not None]

    if not all(
        _logical_plan_has_no_all_to_all(source, target)
        for source, target in (*forward_edges, *backward_edges)
    ):
        return False
    if not any(
        plan.has_all_to_all
        for plan in (*typed_baseline_forward, *typed_baseline_backward)
    ):
        return False
    if any(
        plan.has_all_to_all
        for plan in (*typed_candidate_forward, *typed_candidate_backward)
    ):
        return False

    forward_operations = tuple(
        operation for plan in typed_candidate_forward for operation in plan.operations
    )
    backward_operations = tuple(
        operation for plan in typed_candidate_backward for operation in plan.operations
    )
    if any(kind != "all_gather" for kind, _ in forward_operations):
        return False
    if any(kind != "reduce_scatter" for kind, _ in backward_operations):
        return False
    if not (
        sum(plan.cost for plan in (*typed_candidate_forward, *typed_candidate_backward))
        < sum(plan.cost for plan in (*typed_baseline_forward, *typed_baseline_backward))
    ):
        return False

    released_mesh_dims = tuple(
        mesh_dim
        for mesh_dim, (stored, grad) in enumerate(
            zip(storage.placements, last_backward_source.placements)
        )
        if isinstance(stored, Shard) and isinstance(grad, Partial)
    )
    forward_gathers = tuple(
        mesh_dim
        for kind, mesh_dims in forward_operations
        if kind == "all_gather"
        for mesh_dim in mesh_dims
        if mesh_dim in released_mesh_dims
    )
    backward_scatters = tuple(
        mesh_dim
        for kind, mesh_dims in backward_operations
        if kind == "reduce_scatter"
        for mesh_dim in mesh_dims
        if mesh_dim in released_mesh_dims
    )
    if (
        len(forward_gathers) != len(released_mesh_dims)
        or set(forward_gathers) != set(released_mesh_dims)
        or backward_scatters != tuple(reversed(forward_gathers))
    ):
        return False

    if not _physical_storage_is_checkpoint_compatible(storage, preferred_shard_order):
        return False
    try:
        param_physical = DTensorSpec._convert_shard_order_to_StridedShard(
            preferred_shard_order,
            storage.placements,
            storage.mesh,
        )
        grad_physical = DTensorSpec._convert_shard_order_to_StridedShard(
            _project_shard_order(preferred_shard_order, last_backward_target),
            last_backward_target.placements,
            last_backward_target.mesh,
        )
    except (AssertionError, RuntimeError, TypeError, ValueError):
        return False
    return param_physical == grad_physical


def _optimize_same_nd_sharding_as_1d(
    arg: torch.Tensor, curr_spec: DTensorSpec, tgt_spec: DTensorSpec
) -> torch.Tensor:
    """
    This function optimizes the case where the current and target placements
    have the same placements for all mesh dimensions. For example, if the
    current placement is S(0)S(0) and the target placement is RR, this
    function will perform a single collective, instead of two collectives.
    """
    if not _can_optimize_same_nd_sharding_as_1d(
        curr_spec.placements, tgt_spec.placements
    ):
        return redistribute_local_tensor(arg, curr_spec, tgt_spec)

    curr_spec_first = curr_spec.placements[0]
    tgt_spec_first = tgt_spec.placements[0]

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


def _get_single_redistribution_specs(
    node: torch.fx.Node,
    sharding_placement: dict[torch.fx.Node, OpSpec],
) -> Optional[tuple[DTensorSpec, DTensorSpec]]:
    """Return concrete specs for an unambiguous, single-input redistribution."""
    if node.target == operator.getitem:
        return None
    input_nodes = [
        value
        for value in tree_flatten(node.args)[0]
        if isinstance(value, torch.fx.Node)
    ]
    if len(input_nodes) != 1:
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
        return None
    target_placements = tuple(
        Replicate() if isinstance(placement, Partial) else placement
        for placement in target.placements
    )
    normalized_target = DTensorSpec(
        target.mesh,
        target_placements,
        tensor_meta=target.tensor_meta,
    )
    return source, normalized_target


def _get_chain_redistributions(
    chain: list[torch.fx.Node],
    sharding_placement: dict[torch.fx.Node, OpSpec],
) -> list[tuple[torch.fx.Node, DTensorSpec, DTensorSpec]]:
    """Return every concrete redistribution on a linear chain."""
    result = []
    for node in chain:
        specs = _get_single_redistribution_specs(node, sharding_placement)
        if specs is not None:
            result.append((node, *specs))
    return result


def _redistribution_prefixes_are_unambiguous(
    param_chain: list[torch.fx.Node],
    param_boundary: torch.fx.Node,
    grad_chain: list[torch.fx.Node],
    grad_boundary: torch.fx.Node,
) -> bool:
    """Reject parameter/gradient paths that branch before the chosen boundary."""
    if not param_chain or not grad_chain:
        return False
    try:
        param_boundary_index = param_chain.index(param_boundary)
        grad_boundary_index = grad_chain.index(grad_boundary)
    except ValueError:
        return False

    for index, node in enumerate(param_chain[:param_boundary_index]):
        if len(node.users) != 1:
            return False
        if next(iter(node.users)) is not param_chain[index + 1]:
            return False
    for index, node in enumerate(grad_chain[:grad_boundary_index]):
        if len(node.all_input_nodes) != 1:
            return False
        input_node = node.all_input_nodes[0]
        if len(input_node.users) != 1:
            return False
        if input_node is not grad_chain[index + 1]:
            return False
    return True


def _prefix_preserves_storage_orientation(
    param_chain: list[torch.fx.Node],
    boundary: torch.fx.Node,
    storage: DTensorSpec,
    sharding_placement: dict[torch.fx.Node, OpSpec],
) -> bool:
    """Reject an order if a preceding view makes tensor-dim mapping ambiguous."""
    try:
        boundary_index = param_chain.index(boundary)
    except ValueError:
        return False
    if storage.tensor_meta is None:
        return False
    for node in param_chain[:boundary_index]:
        op_spec = sharding_placement.get(node)
        if op_spec is None or not isinstance(op_spec.output_specs, DTensorSpec):
            return False
        spec = op_spec.output_specs
        if spec.tensor_meta is None:
            return False
        if (
            spec.placements != storage.placements
            or spec.tensor_meta.shape != storage.tensor_meta.shape
            or spec.tensor_meta.stride != storage.tensor_meta.stride
        ):
            return False
    return True


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

    _, source_to_chain = build_param_grad_linear_chains(param_and_grad_nodes)

    redistribute_node_order: dict[torch.fx.Node, OrderInfo] = {}
    fallback_gate_cache: dict[tuple[Any, ...], bool] = {}

    for param_node, grad_node in param_and_grad_nodes:
        if grad_node is None:
            continue
        param_chain = source_to_chain.get(param_node)
        grad_chain = source_to_chain.get(grad_node)
        if param_chain is None or grad_chain is None:
            continue

        forward = _get_chain_redistributions(param_chain, sharding_placement)
        # grad_chain is stored from the returned gradient toward its producer;
        # reverse it to obtain backward execution order.
        backward = list(
            reversed(_get_chain_redistributions(grad_chain, sharding_placement))
        )
        if not forward or not backward:
            continue
        param_redistrib_node, param_source, param_target = forward[0]
        grad_redistrib_node, _, _ = backward[-1]

        param_storage_spec = sharding_placement[param_node].output_specs
        if not isinstance(param_storage_spec, DTensorSpec):
            continue
        # The preferred order describes the physical parameter storage.  Fail
        # closed if a view changed tensor dimensions before the first forward
        # redistribution; that requires explicit order remapping support.
        if param_source.placements != param_storage_spec.placements:
            continue

        preferred_shard_order = _infer_pure_release_storage_order(
            param_source.placements,
            param_target.placements,
        )
        if preferred_shard_order is None:
            continue

        if not _redistribution_prefixes_are_unambiguous(
            param_chain,
            param_redistrib_node,
            grad_chain,
            grad_redistrib_node,
        ):
            continue
        if not _prefix_preserves_storage_orientation(
            param_chain,
            param_redistrib_node,
            param_storage_spec,
            sharding_placement,
        ):
            continue
        forward_specs = [(source, target) for _, source, target in forward]
        backward_specs = [(source, target) for _, source, target in backward]
        gate_key = (
            param_storage_spec,
            tuple(forward_specs),
            tuple(backward_specs),
            preferred_shard_order,
        )
        try:
            accepted = fallback_gate_cache.get(gate_key)
        except TypeError:
            accepted = None
        if accepted is None:
            accepted = _fallback_improves_layout_only_redistribution(
                param_storage_spec,
                forward_specs,
                backward_specs,
                preferred_shard_order,
            )
            try:
                fallback_gate_cache[gate_key] = accepted
            except TypeError:
                pass
        if not accepted:
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
