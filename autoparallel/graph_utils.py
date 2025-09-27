# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

import torch
from torch._functorch.aot_autograd import JointWithDescriptors
from torch._inductor.fx_passes.joint_graph import patterns
from torch._inductor.fx_passes.post_grad import remove_assert_ops, remove_noop_ops
from torch._inductor.pattern_matcher import stable_topological_sort
from torch.fx import GraphModule
from torch.fx.experimental._backward_state import BackwardState


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
    joint_with_descriptors._aot_graph_capture.updated_flat_args = (
        new_flat_args,
        new_local_tangents,
    )
    joint_with_descriptors._aot_state.flat_args = new_flat_args
    joint_with_descriptors._aot_state.fw_metadata.traced_tangents = new_local_tangents


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
            alias_node.meta.update(node.meta)

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
            alias_node.meta.update(node.meta)

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
def _replace_view_mm_view_with_einsum(gm):
    mm_nodes = gm.graph.find_nodes(op="call_function", target=torch.ops.aten.mm.default)
    for node in mm_nodes:
        first_input, second_input = node.all_input_nodes
        if first_input.target == torch.ops.aten.view.default:
            view_input = first_input.all_input_nodes[0]
            users = list(node.users)
            if (
                len(users) == 1
                and users[0].target == torch.ops.aten.view.default
                and view_input.meta["val"].shape[:-1] == users[0].meta["val"].shape[:-1]
                and second_input.meta["val"].ndim == 2
            ):
                print(
                    f"Found matmul node {node}, {view_input.meta['val'].shape, second_input.meta['val'].shape}"
                )
                ndim = view_input.meta["val"].ndim
                assert 1 < ndim <= 10, "Only support up to 10D for now"

                # generate the leading dimensions as a, b, c, etc
                dims = "".join([chr(97 + i) for i in range(ndim - 1)])
                mm_equation = f"{dims}k,kn->{dims}n"
                with gm.graph.inserting_before(node):
                    new_node = gm.graph.call_function(
                        torch.ops.aten.einsum.default,
                        args=(mm_equation, [view_input, second_input]),
                    )
                    new_node.meta.update(users[0].meta)
                    users[0].replace_all_uses_with(new_node)

        elif second_input.target == torch.ops.aten.view.default:
            if first_input.target != torch.ops.aten.permute.default:
                continue
            if first_input.all_input_nodes[0].target != torch.ops.aten.view.default:
                continue
            orig_first = first_input.all_input_nodes[0].all_input_nodes[0]
            orig_second = second_input.all_input_nodes[0]
            users = list(node.users)
            if (
                len(users) == 1
                and users[0].target == torch.ops.aten.permute.default
                and orig_first.meta["val"].shape[:-1]
                == orig_second.meta["val"].shape[:-1]
                and node.meta["val"].ndim == 2
            ):
                print(
                    f"Found matmul node {node} {orig_first.meta['val'].shape, orig_second.meta['val'].shape}"
                )

                ndim = orig_first.meta["val"].ndim
                assert 1 < ndim <= 10, "Only support up to 10D for now"

                # generate the leading dimensions as a, b, c, etc
                dims = "".join([chr(97 + i) for i in range(ndim - 1)])
                mm_equation = f"{dims}n,{dims}k->kn"
                with gm.graph.inserting_before(node):
                    new_node = gm.graph.call_function(
                        torch.ops.aten.einsum.default,
                        args=(mm_equation, [orig_first, orig_second]),
                    )
                    new_node.meta.update(users[0].meta)
                    users[0].replace_all_uses_with(new_node)
    gm.graph.eliminate_dead_code()
    gm.recompile()
