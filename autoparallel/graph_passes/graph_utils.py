# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

import torch
from torch._functorch._aot_autograd.fx_utils import (
    get_param_and_grad_nodes,
    get_param_nodes,
)
from torch._functorch._aot_autograd.subclass_utils import create_subclass_meta
from torch._functorch.aot_autograd import JointWithDescriptors
from torch._inductor.fx_passes.joint_graph import patterns
from torch._inductor.fx_passes.post_grad import remove_assert_ops, remove_noop_ops
from torch._inductor.pattern_matcher import stable_topological_sort
from torch.fx import GraphModule
from torch.fx.experimental._backward_state import BackwardState
from torch.utils._pytree import tree_flatten


def cleanup_graph(gm: torch.fx.GraphModule, aggressive: bool = False) -> None:
    # TODO: we can switch the default "aggresive" to True and things should
    # be even better as we can remove more redundant nodes early on
    # I'm keeping compatibility with previous behavior for now, and will
    # switch the flag in the future

    # TODO: Make the DCE match exactly the AOTAutograd logic, I don't
    # think I trust the default FX DCE logic
    gm.graph.eliminate_dead_code()
    gm.recompile()
    remove_noop_ops(gm.graph)
    # TODO: We shouldn't actually remove these
    remove_assert_ops(gm.graph)
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()

    if aggressive:
        maybe_count = patterns.apply(gm)
        if maybe_count is not None:
            stable_topological_sort(gm.graph)
            gm.graph.lint()
            gm.recompile()


def update_joint_with_descriptors(
    joint_with_descriptors: JointWithDescriptors,
    updated_gm: GraphModule,
) -> None:
    """
    Assuming we have transformed updated_gm since the time it was captured,
    (e.g. by parallelizing it),
    this util updates the joint_with_descriptors struct to reference the new gm, and
    updates any copies of tensor meta/shape stored in joint_with_descriptors relating to input arguments,
    which may have changed shape since the initial trace.
    """
    # TODO: should we upstream a util like this?
    placeholders = [n for n in updated_gm.graph.nodes if n.op == "placeholder"]
    new_local_args = [n.meta["val"] for n in placeholders]
    joint_with_descriptors.graph_module = updated_gm
    joint_with_descriptors._aot_graph_capture.graph_module = updated_gm

    new_flat_args: list[Union[torch.Tensor, int, torch.SymInt, BackwardState]] = []
    for orig, new in zip(joint_with_descriptors._aot_state.flat_args, new_local_args):
        if isinstance(orig, torch.nn.Parameter):
            new_flat_args.append(torch.nn.Parameter(new))
        else:
            new_flat_args.append(new)

    tangent_idx = len(joint_with_descriptors._aot_state.flat_args)
    new_local_tangents = new_local_args[tangent_idx:]

    # For inference mode (no tangents), updated_flat_args should be a list.
    # For autograd mode (with tangents), it should be a tuple of (primals, tangents).
    if new_local_tangents:
        joint_with_descriptors._aot_graph_capture.updated_flat_args = (
            new_flat_args,
            new_local_tangents,
        )
    else:
        joint_with_descriptors._aot_graph_capture.updated_flat_args = new_flat_args

    joint_with_descriptors._aot_state.flat_args = new_flat_args  # type: ignore[assignment]
    joint_with_descriptors._aot_state.fw_metadata.traced_tangents = new_local_tangents
    # Regenerate subclass_tangent_meta from the updated local tangents so that
    # MemoryFormatMeta records the correct (local) sizes and strides.
    # Without this, the stale global-shaped metadata causes
    # coerce_to_expected_memory_format to broadcast the tangent back to global
    # shape, which then fails the inductor backward's assert_size_stride.
    joint_with_descriptors._aot_state.fw_metadata.subclass_tangent_meta = (
        create_subclass_meta(
            new_local_tangents, count_symints=False, with_memory_format=True
        )
    )


def _add_alias(gm, version="v1"):
    """
    Helper function to add alias nodes to every node in the graph
    this gives more configuration opportunities
    """
    graph = gm.graph

    nodes = list(graph.nodes)
    node_map = {node: idx for idx, node in enumerate(nodes)}

    def _insert_alias(node):
        first_user = nodes[min(node_map[n] for n in node.users)]
        with graph.inserting_before(first_user):
            alias_node = graph.call_function(torch.ops.aten.alias.default, args=(node,))
            # for some reason tangents have "partitioner_tag" marked as "is_forward"
            # which can mess up with the partitioner
            meta = {k: v for k, v in node.meta.items()}
            if "tangents" in node.name and "partitioner_tag" in meta:
                del meta["partitioner_tag"]
            alias_node.meta.update(meta)

            def delete_user_cb(n):
                return n != alias_node

            node.replace_all_uses_with(alias_node, delete_user_cb=delete_user_cb)

    if version == "v1":
        # only on inputs
        for node in graph.find_nodes(op="placeholder"):
            if len(node.users) == 0:
                # node is not used, don't add alias for it
                continue
            if (
                len(node.users) == 1
                and list(node.users)[0].target
                == torch.ops.autoparallel.dtype_cast.default
            ):
                node = list(node.users)[0]
            _insert_alias(node)
    elif version == "v2":
        # for every node that has more than one user
        for node in nodes:
            if len(node.users) < 2:
                continue
            # don't add alias for ops which return tuple for now
            if not isinstance(node.meta["val"], torch.Tensor):
                continue
            _insert_alias(node)
    else:
        raise ValueError(f"Unknown version {version}")

    """
    nodes = [n for n in graph.nodes if n.op == "call_function"]
    for node in nodes:
        # skip ops which return tuple
        if not isinstance(node.meta["val"], torch.Tensor):
            continue
        with graph.inserting_after(node):
            alias_node = graph.call_function(torch.ops.aten.alias.default, args=(node,))
            alias_node.meta.update(node.meta)

            def delete_user_cb(n):
                return n != alias_node

            node.replace_all_uses_with(alias_node, delete_user_cb=delete_user_cb)

    """

    for node in graph.find_nodes(op="output")[0].all_input_nodes:
        with graph.inserting_after(node):
            alias_node = graph.call_function(torch.ops.aten.alias.default, args=(node,))
            # for some reason tangents have "partitioner_tag" marked as "is_forward"
            # which can mess up with the partitioner
            meta = {k: v for k, v in node.meta.items()}
            if "tangents" in node.name and "partitioner_tag" in meta:
                del meta["partitioner_tag"]
            alias_node.meta.update(meta)

            def delete_user_cb(n):
                return n != alias_node

            node.replace_all_uses_with(alias_node, delete_user_cb=delete_user_cb)

    gm.recompile()
    return gm


def is_collective(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and isinstance(node.target, torch._ops.OpOverload)
        and node.target.namespace == "_c10d_functional"
    )


_SCATTER_OPS = {
    torch.ops.aten.diagonal_scatter.default,
    torch.ops.aten.select_scatter.default,
    torch.ops.aten.slice_scatter.default,
    torch.ops.aten.as_strided_scatter.default,
}


def fix_scatter_on_aliased_inputs(graph: torch.fx.Graph) -> None:
    """Insert clone before scatter ops whose input has zero strides (aliased from expand).

    Inductor's reinplace_inplaceable_ops decomposes scatter ops into copy_,
    which fails on aliased tensors. Cloning materializes the aliased view
    into a contiguous tensor.
    """
    for node in graph.nodes:
        if node.op != "call_function" or node.target not in _SCATTER_OPS:
            continue
        input_node = node.args[0]
        if not isinstance(input_node, torch.fx.Node):
            continue
        val = input_node.meta.get("val")
        if val is None or not isinstance(val, torch.Tensor):
            continue
        if any(s == 0 for s in val.stride()):
            with graph.inserting_before(node):
                clone = graph.call_function(torch.ops.aten.clone.default, (input_node,))
                clone.meta = input_node.meta.copy()
                clone.meta["val"] = val.clone()
                node.replace_input_with(input_node, clone)


def assert_has_no_collectives(gm: torch.fx.GraphModule):
    for node in gm.graph.nodes:
        if is_collective(node):
            raise RuntimeError(
                f"AutoParallel expects a single-GPU model "
                f"implementation with not collectives in it, but found {node} "
                f"operation in \n{node.meta['stack_trace']}.\n"
                f"If you want to manually add collectives in the model "
                f"(e.g., for optimization purposes), please wrap the region "
                f"of the code which contains the collectives in an "
                f"autoparallel.local_map_hop.apply_local_map, see "
                "examples/example_local_map.py for more information."
            )


# NOTE: [nn.Linear decomposition]
# PyTorch currently decomposes any 3d-input nn.Linear (and matmul) into a
# sequence of view -> mm -> view operations.
# This has as a consequence of breaking any type of sharding on both the
# batch and the sequence dimension, because the flattening that happens doesn't
# allow to preserve this sharding.
# While we wait for PyTorch to avoid decomposing nn.Linear, we instead take
# the route of pattern-matching the nn.Linear specific occurences, and we replace
# them with an einsum operator.
# We perform this pattern-matching replacement for both the forward as well as
# the backward pass.
# TODO: use graph_patterns to simplify writing this
def _batch_dims(n: int) -> str:
    """Return a string of `n` batch-dimension letters starting from 'a'."""
    assert 0 < n <= 25
    return "".join(chr(97 + i) for i in range(n))


def _is_canonical_flatten(input_shape, view_args):
    """Check that view_args represent [*batch, K] -> [prod(batch), K]."""
    if len(view_args) != 2:
        return False
    expected_flat = 1
    for d in input_shape[:-1]:
        expected_flat *= d
    return view_args[0] == expected_flat and view_args[1] == input_shape[-1]


def _is_canonical_unflatten(input_shape, view_args):
    """Check that view_args represent [prod(batch), N] -> [*batch, N]."""
    if len(view_args) < 3:
        return False
    batch_dims = view_args[:-1]
    expected_flat = 1
    for d in batch_dims:
        expected_flat *= d
    return expected_flat == input_shape[0] and view_args[-1] == input_shape[-1]


def _match_forward_linear(mm_node):
    """Match the forward pattern: view -> mm -> view.

    Verifies canonical flatten/unflatten shapes:
      input [*batch, K] -> view [prod(batch), K] -> mm [prod(batch), N] -> view [*batch, N]

    Returns (inputs, replaced_node, equation) or None.
    """
    first_input, second_input = mm_node.all_input_nodes
    if first_input.target != torch.ops.aten.view.default:
        return None
    view_input = first_input.all_input_nodes[0]
    input_shape = view_input.meta["val"].shape
    if input_shape.numel() == 0 or len(input_shape) < 3:
        return None
    # Verify the input view is a canonical flatten [*batch, K] -> [prod(batch), K]
    flatten_args = first_input.args[1]
    if not _is_canonical_flatten(input_shape, flatten_args):
        return None
    users = list(mm_node.users)
    if not (
        len(users) == 1
        and users[0].target == torch.ops.aten.view.default
        and second_input.meta["val"].ndim == 2
    ):
        return None
    output_view = users[0]
    output_shape = output_view.meta["val"].shape
    # Verify the output view is a canonical unflatten [prod(batch), N] -> [*batch, N]
    unflatten_args = output_view.args[1]
    mm_shape = mm_node.meta["val"].shape
    if not _is_canonical_unflatten(mm_shape, unflatten_args):
        return None
    # Verify batch dims match between input and output
    if input_shape[:-1] != output_shape[:-1]:
        return None
    # Verify weight shape is [K, N] matching the flatten dimensions
    weight_shape = second_input.meta["val"].shape
    if weight_shape[0] != input_shape[-1] or weight_shape[1] != output_shape[-1]:
        return None
    ndim = len(input_shape)
    dims = _batch_dims(ndim - 1)
    equation = f"{dims}k,kn->{dims}n"
    return [view_input, second_input], output_view, equation


def _match_backward_linear(mm_node):
    """Match the backward pattern: view -> permute -> mm -> permute.

    The backward of einsum "{batch}k,kn->{batch}n" produces a gradient-weight
    computation: permute(view(grad, [flat, N]), [1,0]) @ view(x, [flat, K]) -> [N, K],
    followed by permute([N, K], [1, 0]) -> [K, N].

    Verifies canonical flatten shapes and [1,0] permute orders.

    Returns (inputs, replaced_node, equation) or None.
    """
    first_input, second_input = mm_node.all_input_nodes
    if second_input.target != torch.ops.aten.view.default:
        return None
    if first_input.target != torch.ops.aten.permute.default:
        return None
    first_view = first_input.all_input_nodes[0]
    if first_view.target != torch.ops.aten.view.default:
        return None
    # Verify the input permute is [1, 0] (transpose)
    perm_dims = list(first_input.args[1])
    if perm_dims != [1, 0]:
        return None
    orig_first = first_view.all_input_nodes[0]
    orig_second = second_input.all_input_nodes[0]
    # Verify both views are canonical flattenings
    if not _is_canonical_flatten(orig_first.meta["val"].shape, first_view.args[1]):
        return None
    if not _is_canonical_flatten(orig_second.meta["val"].shape, second_input.args[1]):
        return None
    users = list(mm_node.users)
    if not (
        len(users) == 1
        and users[0].target == torch.ops.aten.permute.default
        and mm_node.meta["val"].ndim == 2
    ):
        return None
    # Verify the output permute is [1, 0]
    out_perm_dims = list(users[0].args[1])
    if out_perm_dims != [1, 0]:
        return None
    # Verify batch dims match
    if orig_first.meta["val"].shape[:-1] != orig_second.meta["val"].shape[:-1]:
        return None
    ndim = orig_first.meta["val"].ndim
    dims = _batch_dims(ndim - 1)
    equation = f"{dims}n,{dims}k->kn"
    return [orig_first, orig_second], users[0], equation


def _replace_view_mm_view_with_einsum(gm):
    mm_nodes = gm.graph.find_nodes(op="call_function", target=torch.ops.aten.mm.default)
    for node in mm_nodes:
        match = _match_forward_linear(node) or _match_backward_linear(node)
        if match is None:
            continue
        inputs, replaced_node, equation = match
        with gm.graph.inserting_before(node):
            new_node = gm.graph.call_function(
                torch.ops.aten.einsum.default,
                args=(equation, inputs),
            )
            new_node.meta.update(replaced_node.meta)
            # Preserve the mm node's seq_nr so that forward/backward
            # einsum pairs remain matched by autograd's sequence numbering.
            if "seq_nr" in node.meta:
                new_node.meta["seq_nr"] = node.meta["seq_nr"]
            replaced_node.replace_all_uses_with(new_node)
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()


def all_input_nodes(node: torch.fx.Node) -> list[torch.fx.Node]:
    """Variant of node.all_input_nodes that preserves duplicate nodes."""
    return [x for x in tree_flatten(node.args)[0] if isinstance(x, torch.fx.Node)]


def build_param_derived_set(graph: torch.fx.Graph) -> set[torch.fx.Node]:
    """Compute the set of nodes whose inputs are ALL parameter-derived.

    A node is parameter-derived if every one of its inputs is either a
    parameter placeholder or itself parameter-derived. This propagates
    through dtype_cast, views, aliases, etc.
    """
    param_derived = set(get_param_nodes(graph))
    for node in graph.nodes:
        inputs = node.all_input_nodes
        if inputs and all(inp in param_derived for inp in inputs):
            param_derived.add(node)
    return param_derived


def build_terminal_derived_set(graph: torch.fx.Graph) -> set[torch.fx.Node]:
    """Compute the set of nodes in the parameter gradient reduce-scatter chain.

    A node is in this set if every one of its users is either a parameter
    gradient node or itself in this set. This captures only the tail end
    of the gradient chain (alias/view nodes right before the output), not
    the backward matmuls that compute the gradients.
    """
    # Seed with the grad nodes themselves — these are the terminal outputs
    # for parameter gradients and their only user is the output node.
    terminal_derived: set[torch.fx.Node] = set()
    for param, grad in get_param_and_grad_nodes(graph).values():
        if grad is not None:
            terminal_derived.add(grad)

    # Walk backward: a node is terminal-derived if ALL its users are
    # already in the terminal_derived set.
    for node in reversed(list(graph.nodes)):
        if node.op == "output":
            continue
        if node.users and all(u in terminal_derived for u in node.users):
            terminal_derived.add(node)
    return terminal_derived
