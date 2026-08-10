# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import operator

import torch
from torch._dynamo.functional_export import _dynamo_graph_capture_for_export
from torch._subclasses.fake_tensor import FakeTensor, unset_fake_temporarily
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import OpSpec
from torch.distributed.tensor.placement_types import Replicate

from .cast_parametrization import set_dtype_cast
from .collectives import get_flex_local_map_alternatives, local_map
from .shardings.placement_options import _concretize_tensor_meta
from .tracing import enable_local_map_wrapping

_TRACE_INFO = "_autoparallel_flex_trace_info"


def _body(gm, node):
    if not node.args or not isinstance(node.args[0], torch.fx.Node):
        return None
    body_node = node.args[0]
    if body_node.op != "get_attr":
        return None
    try:
        body = gm.get_submodule(body_node.target)
    except AttributeError:
        return None
    return body if isinstance(body, torch.fx.GraphModule) else None


def _value(node):
    for key in ("example_value", "val"):
        if key in node.meta:
            return node.meta[key]
    raise RuntimeError(f"FX node {node.name} has no example value")


def prepare_flex_local_maps(gm):
    for node in gm.graph.nodes:
        body = _body(gm, node)
        if body is None:
            continue
        kwargs = body.meta.get("local_map_kwargs", {})
        if get_flex_local_map_alternatives(kwargs) is None:
            continue

        outputs = body.graph.find_nodes(op="output")[0].args[0]
        outputs = outputs if isinstance(outputs, (tuple, list)) else (outputs,)
        body.meta["local_map_kwargs"] = {
            **kwargs,
            _TRACE_INFO: {
                "input_requires_grad": tuple(
                    isinstance(
                        value := (
                            _value(arg) if isinstance(arg, torch.fx.Node) else arg
                        ),
                        torch.Tensor,
                    )
                    and value.requires_grad
                    for arg in node.args[1:]
                ),
                "grad_output_indices": tuple(
                    index
                    for index, output in enumerate(outputs)
                    if isinstance(output, torch.fx.Node)
                    and isinstance(output.meta.get("example_value"), torch.Tensor)
                    and output.meta["example_value"].requires_grad
                ),
            },
        }


def flex_local_map_pairs(nodes):
    forwards = []
    backwards = {}
    for node in nodes:
        kwargs = node.meta.get("local_map_kwargs")
        if kwargs is None:
            continue
        if node.meta.get("partitioner_tag") == "is_backward":
            seq_nr = node.meta.get("seq_nr")
            backwards.setdefault(seq_nr, []).append(node)
        elif get_flex_local_map_alternatives(kwargs) is not None:
            forwards.append(node)

    for forward in forwards:
        seq_nr = forward.meta.get("seq_nr")
        candidates = backwards.get(seq_nr, ())
        if len(candidates) > 1:
            raise RuntimeError(f"Multiple local_map backward nodes use seq_nr={seq_nr}")
        yield forward, candidates[0] if candidates else None


def normalize_flex_local_map_backward(gm):
    for forward, backward in flex_local_map_pairs(gm.graph.nodes):
        kwargs = forward.meta["local_map_kwargs"]
        alternatives = tuple(get_flex_local_map_alternatives(kwargs))
        kwargs["alternatives"] = alternatives
        if backward is None:
            continue

        grad_indices = kwargs[_TRACE_INFO]["grad_output_indices"]
        mirrored = tuple(
            {
                **alternative,
                "in_placements": tuple(
                    alternative["out_placements"][index] for index in grad_indices
                ),
                "out_placements": alternative["in_placements"],
            }
            for alternative in alternatives
        )
        backward_kwargs = backward.meta["local_map_kwargs"]
        if (
            mirrored[0]["in_placements"] != backward_kwargs["in_placements"]
            or mirrored[0]["out_placements"] != backward_kwargs["out_placements"]
        ):
            raise RuntimeError(
                f"flex_local_map seq_nr={forward.meta.get('seq_nr')} has an "
                "incompatible default backward placement contract"
            )
        backward_kwargs["alternatives"] = mirrored


def _trace_body(fn, kwargs, example_args):
    wrapped = local_map(
        fn,
        out_placements=kwargs["out_placements"],
        in_placements=kwargs["in_placements"],
        in_grad_placements=kwargs.get("in_grad_placements"),
        device_mesh=kwargs["device_mesh"],
        redistribute_inputs=kwargs.get("redistribute_inputs", False),
    )
    with unset_fake_temporarily():
        traced = _dynamo_graph_capture_for_export(wrapped)(*example_args)
    nodes = [node for node in traced.graph.nodes if _body(traced, node) is not None]
    if len(nodes) != 1:
        raise RuntimeError(
            "Tracing a selected flex_local_map body produced "
            f"{len(nodes)} local_map calls, expected one"
        )
    return _body(traced, nodes[0]), _value(nodes[0])


def _as_tuple(value):
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


def _activation_spec(mesh, value):
    if not isinstance(value, torch.Tensor):
        return None
    tensor_meta = _concretize_tensor_meta(value)
    if tensor_meta is None:
        return None
    return DTensorSpec(
        mesh,
        tuple(Replicate() for _ in range(mesh.ndim)),
        tensor_meta=tensor_meta,
    )


def _replace_region(
    gm,
    forward,
    forward_body,
    backward,
    backward_body,
    kwargs,
    activations,
    solution,
    mesh,
):
    old_num_activations = _body(gm, forward).meta.get("num_activations", 0)
    num_outputs = len(kwargs["out_placements"])
    if backward is None:
        old_activations = sorted(
            (
                user
                for user in forward.users
                if user.op == "call_function"
                and user.target == operator.getitem
                and isinstance(user.args[1], int)
                and user.args[1] >= num_outputs
            ),
            key=lambda node: node.args[1],
        )
    else:
        old_activations = list(backward.args[1 : 1 + old_num_activations])
    if len(old_activations) != old_num_activations:
        raise RuntimeError(
            f"flex_local_map node {forward.name} has {len(old_activations)} "
            f"activation getitems, expected {old_num_activations}"
        )

    with gm.graph.inserting_before(backward or forward.next):
        new_activations = [
            gm.graph.call_function(operator.getitem, (forward, index))
            for index in range(num_outputs, num_outputs + len(activations))
        ]
    for node, value in zip(new_activations, activations):
        node.meta.update(val=value, example_value=value, partitioner_tag="is_forward")

    forward_body.meta.update(
        local_map_kwargs=kwargs,
        num_activations=len(activations),
        is_backward=False,
    )
    gm.set_submodule(forward.args[0].target, forward_body)
    forward.meta.update(
        local_map_kwargs=kwargs,
        val=_as_tuple(_value(forward))[:num_outputs] + tuple(activations),
    )

    activation_specs = tuple(_activation_spec(mesh, value) for value in activations)
    forward_spec = solution[forward]
    forward_spec.output_specs = (
        tuple(forward_spec.output_specs[:num_outputs]) + activation_specs
    )
    for node, value, spec in zip(new_activations, activations, activation_specs):
        if isinstance(value, torch.Tensor):
            solution[node] = OpSpec(spec, (spec,), [[0.0]])

    if backward is not None:
        old_tensor_activations = sum(
            isinstance(_value(node), torch.Tensor) for node in old_activations
        )
        tensor_activation_specs = tuple(
            spec
            for value, spec in zip(activations, activation_specs)
            if isinstance(value, torch.Tensor)
        )
        backward_spec = solution[backward]
        backward_spec.input_specs = tensor_activation_specs + tuple(
            backward_spec.input_specs[old_tensor_activations:]
        )
        backward_spec.redistribute_cost = [
            [0.0] for _ in tensor_activation_specs
        ] + list(backward_spec.redistribute_cost[old_tensor_activations:])
        gm.set_submodule(backward.args[0].target, backward_body)
        backward_activations = [
            node
            for node, value in zip(new_activations, activations)
            if not isinstance(value, torch.Tensor)
        ] + [
            node
            for node, value in zip(new_activations, activations)
            if isinstance(value, torch.Tensor)
        ]
        backward.args = (
            backward.args[0],
            *backward_activations,
            *backward.args[1 + old_num_activations :],
        )
        backward.meta.update(
            local_map_kwargs=backward_body.meta["local_map_kwargs"],
        )

    for node in old_activations:
        if not node.users:
            solution.pop(node, None)
            gm.graph.erase_node(node)


def _example_args(forward, trace_info):
    masks = trace_info["input_requires_grad"]
    if len(masks) != len(forward.args) - 1:
        raise RuntimeError(f"flex_local_map node {forward.name} has invalid trace info")
    args = []
    for arg, requires_grad in zip(forward.args[1:], masks):
        value = _value(arg) if isinstance(arg, torch.fx.Node) else arg
        if isinstance(value, FakeTensor) and value.requires_grad != requires_grad:
            with value.fake_mode:
                value = value.detach().requires_grad_(requires_grad)
        args.append(value)
    return tuple(args)


def finalize_flex_local_maps(gm, solution, mesh):
    from torch._higher_order_ops.local_map import create_hop_fw_bw

    changed = False
    with (
        set_dtype_cast(True),
        enable_local_map_wrapping(),
        torch._dynamo.utils._disable_saved_tensors_hooks_during_tracing(),
    ):
        for forward, backward in list(flex_local_map_pairs(gm.graph.nodes)):
            kwargs = forward.meta["local_map_kwargs"]
            alternatives = get_flex_local_map_alternatives(kwargs)
            alternative_index = getattr(
                solution[forward], "flex_local_map_alternative_index", None
            )
            if alternative_index is None:
                raise RuntimeError(
                    f"flex_local_map node {forward.name} has no selected alternative"
                )
            alternative = alternatives[alternative_index]
            selected_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key not in ("alternatives", _TRACE_INFO)
            }
            selected_kwargs.update(
                in_placements=alternative["in_placements"],
                out_placements=alternative["out_placements"],
                device_mesh=mesh,
            )

            if alternative_index == 0:
                forward.meta["local_map_kwargs"] = selected_kwargs
                _body(gm, forward).meta["local_map_kwargs"] = selected_kwargs
                if backward is not None:
                    backward_kwargs = {
                        key: value
                        for key, value in backward.meta["local_map_kwargs"].items()
                        if key not in ("alternatives", _TRACE_INFO)
                    }
                    backward.meta["local_map_kwargs"] = backward_kwargs
                    _body(gm, backward).meta["local_map_kwargs"] = backward_kwargs
                continue

            trace_info = kwargs[_TRACE_INFO]
            example_args = _example_args(forward, trace_info)
            selected_body, selected_output = _trace_body(
                alternatives[alternative_index]["fn"],
                selected_kwargs,
                example_args,
            )
            selected_body.meta["local_map_kwargs"] = selected_kwargs

            if backward is None:
                selected_body.meta.update(num_activations=0, is_backward=False)
                gm.set_submodule(forward.args[0].target, selected_body)
                forward.meta.update(
                    local_map_kwargs=selected_kwargs,
                    val=_as_tuple(selected_output),
                )
                changed = True
                continue

            fake_mode = next(
                value.fake_mode
                for value in torch.utils._pytree.tree_leaves(example_args)
                if isinstance(value, FakeTensor)
            )
            with unset_fake_temporarily(), fake_mode:
                forward_body, backward_body, _, _, grad_indices = create_hop_fw_bw(
                    selected_body, *example_args
                )
            if grad_indices != set(trace_info["grad_output_indices"]):
                raise RuntimeError(
                    f"flex_local_map node {forward.name} changed its "
                    "differentiable outputs"
                )
            outputs = forward_body.graph.find_nodes(op="output")[0].args[0]
            outputs = outputs if isinstance(outputs, (tuple, list)) else (outputs,)
            outputs = tuple(
                _value(value) if isinstance(value, torch.fx.Node) else value
                for value in outputs
            )
            activations = outputs[len(selected_kwargs["out_placements"]) :]

            _replace_region(
                gm,
                forward,
                forward_body,
                backward,
                backward_body,
                selected_kwargs,
                activations,
                solution,
                mesh,
            )
            changed = True

    if changed:
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
