# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import operator
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch._subclasses.fake_tensor import (
    FakeTensor,
    FakeTensorMode,
    unset_fake_temporarily,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import OpSpec
from torch.distributed.tensor.placement_types import Replicate

from .shardings.placement_options import _concretize_tensor_meta

_DEFERRED_BODY_ATTR = "_autoparallel_deferred_local_map_body"
_TRACE_INFO = "_autoparallel_deferred_trace_info"
_TRACE_DEFERRED_BODIES = False


@dataclass(frozen=True)
class _DeferredLocalMapBody:
    surrogate_fn: Callable
    runtime_fn: Callable

    def __call__(self, *args, **kwargs):
        return self.runtime_fn(*args, **kwargs)

    def __deepcopy__(self, memo):
        # DeviceMesh owns ProcessGroup objects, which must not be deep-copied.
        return self


class _DeferredLocalMapOutPlacements(tuple):
    def __new__(cls, values, body, metadata=None):
        obj = super().__new__(cls, values)
        # Preserve metadata carried by wrappers such as flex_local_map.
        if metadata is None:
            metadata = getattr(values, "__dict__", {})
        obj.__dict__.update(metadata)
        setattr(obj, _DEFERRED_BODY_ATTR, body)
        return obj

    def __getnewargs__(self):
        metadata = {
            key: value
            for key, value in self.__dict__.items()
            if key != _DEFERRED_BODY_ATTR
        }
        return (tuple(self), getattr(self, _DEFERRED_BODY_ATTR), metadata)

    def __getitem__(self, item):
        result = super().__getitem__(item)
        if isinstance(item, slice):
            metadata = {
                key: value
                for key, value in self.__dict__.items()
                if key != _DEFERRED_BODY_ATTR
            }
            return type(self)(result, getattr(self, _DEFERRED_BODY_ATTR), metadata)
        return result


def deferred_local_map_out_placements(values, body):
    return _DeferredLocalMapOutPlacements(tuple(values), body)


def get_deferred_local_map_body(local_map_kwargs):
    for key in ("out_placements", "in_placements"):
        placements = local_map_kwargs.get(key)
        body = getattr(placements, _DEFERRED_BODY_ATTR, None)
        if body is not None:
            return body
    return None


def is_tracing_deferred_local_map_bodies() -> bool:
    return _TRACE_DEFERRED_BODIES


@contextmanager
def trace_deferred_local_map_bodies() -> Generator[None, None, None]:
    global _TRACE_DEFERRED_BODIES
    prior = _TRACE_DEFERRED_BODIES
    try:
        _TRACE_DEFERRED_BODIES = True
        yield
    finally:
        _TRACE_DEFERRED_BODIES = prior


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


def _as_tuple(value):
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


def _user_args(gm, node):
    return node.args[1:] if _body(gm, node) is not None else node.args


def _call_local_map(body, *args):
    from torch._higher_order_ops.local_map import local_map_hop

    return local_map_hop(body, *args)


def _call_deferred_body(body, *args):
    return _as_tuple(body.runtime_fn(*args))


def _set_body(gm, node, body):
    old_body = _body(gm, node)
    if old_body is not None:
        gm.set_submodule(node.args[0].target, body)
    else:
        attr = f"_deferred_{node.name}_body"
        gm.add_submodule(attr, body)
        with gm.graph.inserting_before(node):
            body_node = gm.graph.get_attr(attr)
        node.target = _call_local_map
        node.args = (body_node,) + node.args


def prepare_deferred_local_maps(gm) -> None:
    """Record the surrogate boundary contract before top-level AOTAutograd."""
    for node in gm.graph.nodes:
        body = _body(gm, node)
        if body is None:
            continue
        kwargs = body.meta.get("local_map_kwargs", {})
        if get_deferred_local_map_body(kwargs) is None:
            continue

        outputs = body.graph.find_nodes(op="output")[0].args[0]
        outputs = _as_tuple(outputs)
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


def _deferred_local_map_pairs(nodes):
    forwards = []
    backwards = {}
    for node in nodes:
        kwargs = node.meta.get("local_map_kwargs")
        if kwargs is None:
            continue
        if node.meta.get("partitioner_tag") == "is_backward":
            backwards.setdefault(node.meta.get("seq_nr"), []).append(node)
        elif get_deferred_local_map_body(kwargs) is not None:
            forwards.append(node)

    for forward in forwards:
        seq_nr = forward.meta.get("seq_nr")
        candidates = backwards.get(seq_nr, ())
        if len(candidates) > 1:
            raise RuntimeError(f"Multiple local_map backward nodes use seq_nr={seq_nr}")
        yield forward, candidates[0] if candidates else None


def _example_args(gm, forward, trace_info):
    masks = trace_info["input_requires_grad"]
    user_args = _user_args(gm, forward)
    if len(masks) != len(user_args):
        raise RuntimeError(
            f"local_map node {forward.name} has {len(user_args)} inputs "
            f"but trace info records {len(masks)}"
        )

    args = []
    for arg, requires_grad in zip(user_args, masks):
        value = _value(arg) if isinstance(arg, torch.fx.Node) else arg
        if isinstance(value, FakeTensor) and value.requires_grad != requires_grad:
            with value.fake_mode:
                value = value.detach().requires_grad_(requires_grad)
        args.append(value)
    return tuple(args)


def _clean_kwargs(kwargs):
    cleaned = {key: value for key, value in kwargs.items() if key != _TRACE_INFO}
    cleaned["out_placements"] = tuple(cleaned["out_placements"])
    return cleaned


def _concretize_fake_args(args):
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

    def concretize(value):
        if not isinstance(value, FakeTensor):
            return value

        def hint(dim):
            if not isinstance(dim, torch.SymInt):
                return dim
            if dim.node.hint is None:
                raise RuntimeError(
                    "Deferred local_map finalization requires hinted symbolic shapes"
                )
            return dim.node.hint

        with unset_fake_temporarily(), fake_mode:
            return torch.empty_strided(
                tuple(hint(dim) for dim in value.shape),
                tuple(hint(dim) for dim in value.stride()),
                dtype=value.dtype,
                device=value.device,
                requires_grad=value.requires_grad,
            )

    return torch.utils._pytree.tree_map(concretize, args)


def _trace_deferred_body(deferred, kwargs, example_args):
    from torch._higher_order_ops.local_map import redistribute_fw_inputs

    trace_args = _concretize_fake_args(example_args)
    fake_mode = next(
        value.fake_mode
        for value in torch.utils._pytree.tree_leaves(trace_args)
        if isinstance(value, FakeTensor)
    )
    with unset_fake_temporarily(), fake_mode:
        local_args = redistribute_fw_inputs(
            trace_args, kwargs["in_placements"], kwargs["device_mesh"]
        )

    # Keep the callable opaque until create_hop_fw_bw traces the joint graph.
    # This lets the DTensor CP dispatcher contribute both its rank-aware
    # forward and its custom backward without making DeviceMesh a HOP input.
    root = torch.nn.Module()
    root._deferred_body = deferred
    graph = torch.fx.Graph()
    body_node = graph.get_attr("_deferred_body")
    placeholders = []
    for index, value in enumerate(local_args):
        placeholder = graph.placeholder(f"arg{index}")
        placeholder.meta.update(val=value, example_value=value)
        placeholders.append(placeholder)
    output = graph.call_function(_call_deferred_body, (body_node, *placeholders))
    graph.output(output)
    traced = torch.fx.GraphModule(root, graph)
    traced.meta["local_map_kwargs"] = kwargs
    return traced, trace_args


def _assert_no_device_mesh(gm) -> None:
    for node in gm.graph.nodes:
        if node.op == "placeholder" and isinstance(
            node.meta.get("example_value"), DeviceMesh
        ):
            raise RuntimeError("DeviceMesh leaked into a finalized local_map input")
        if node.op == "get_attr" and isinstance(getattr(gm, node.target), DeviceMesh):
            raise RuntimeError("DeviceMesh leaked into a finalized local_map body")


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
        old_num_activations = len(old_activations)
    else:
        backward_user_args = _user_args(gm, backward)
        old_num_activations = len(backward_user_args) - len(
            backward.meta["local_map_kwargs"]["in_placements"]
        )
        old_activations = list(backward_user_args[:old_num_activations])
    if len(old_activations) != old_num_activations:
        raise RuntimeError(
            f"local_map node {forward.name} has {len(old_activations)} activation "
            f"getitems, expected {old_num_activations}"
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
    _set_body(gm, forward, forward_body)
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
        _set_body(gm, backward, backward_body)
        backward_activations = [
            node
            for node, value in zip(new_activations, activations)
            if not isinstance(value, torch.Tensor)
        ] + [
            node
            for node, value in zip(new_activations, activations)
            if isinstance(value, torch.Tensor)
        ]
        old_body = _body(gm, backward)
        body_prefix = (backward.args[0],) if old_body is not None else ()
        old_user_args = _user_args(gm, backward)
        backward.args = (
            body_prefix
            + tuple(backward_activations)
            + tuple(old_user_args[old_num_activations:])
        )
        backward.meta["local_map_kwargs"] = backward_body.meta["local_map_kwargs"]

    for node in old_activations:
        if not node.users:
            solution.pop(node, None)
            gm.graph.erase_node(node)


def finalize_deferred_local_maps(gm, solution, mesh) -> None:
    from torch._higher_order_ops.local_map import create_hop_fw_bw

    changed = False
    for forward, backward in list(_deferred_local_map_pairs(gm.graph.nodes)):
        original_kwargs = forward.meta["local_map_kwargs"]
        deferred = get_deferred_local_map_body(original_kwargs)
        if deferred is None:
            continue
        trace_info = original_kwargs[_TRACE_INFO]
        kwargs = _clean_kwargs(original_kwargs)
        example_args = _example_args(gm, forward, trace_info)
        import torch.distributed.config as dist_config

        # The CP dispatcher has Python rank-aware branches. Trace this child
        # region on each rank instead of turning mesh coordinates into the
        # unbacked symbols used by compile_on_one_rank.
        with dist_config.patch(compile_on_one_rank=False):
            traced, trace_args = _trace_deferred_body(deferred, kwargs, example_args)
            fake_mode = next(
                value.fake_mode
                for value in torch.utils._pytree.tree_leaves(trace_args)
                if isinstance(value, FakeTensor)
            )
            with unset_fake_temporarily(), fake_mode:
                forward_body, backward_body, _, _, grad_indices = create_hop_fw_bw(
                    traced, *trace_args
                )
        if grad_indices != set(trace_info["grad_output_indices"]):
            raise RuntimeError(
                f"local_map node {forward.name} changed its differentiable outputs"
            )
        _assert_no_device_mesh(forward_body)
        _assert_no_device_mesh(backward_body)

        outputs = forward_body.graph.find_nodes(op="output")[0].args[0]
        outputs = _as_tuple(outputs)
        num_outputs = len(kwargs["out_placements"])
        if len(outputs) < num_outputs:
            raise RuntimeError(
                f"local_map node {forward.name} changed its output arity"
            )
        activations = tuple(
            _value(value) if isinstance(value, torch.fx.Node) else value
            for value in outputs[num_outputs:]
        )

        _replace_region(
            gm,
            forward,
            forward_body,
            backward,
            backward_body,
            kwargs,
            activations,
            solution,
            mesh,
        )
        changed = True

    if changed:
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
