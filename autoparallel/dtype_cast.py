# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager
from functools import wraps

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.utils._pytree import tree_map


@torch.library.custom_op("autoparallel::dtype_cast", mutates_args=())
def dtype_cast(
    x: torch.Tensor, dtype: torch.dtype, grad_dtype: torch.dtype | None = None
) -> torch.Tensor:
    """
    This is a custom op that is used to cast the input tensor to the specified dtype.
    We use it to be able to specify a special compute cost for the cast operation,
    so that we always favor performing all-gather of small tensors in the smallest
    dtype.
    """
    return x.to(dtype, copy=True)


def setup_context(ctx, inputs, output) -> None:
    x, _, grad_dtype = inputs
    ctx.orig_dtype = x.dtype
    if grad_dtype is not None:
        ctx.set_output_grad_dtype(grad_dtype)


def backward(ctx, grad):
    dtype = ctx.orig_dtype
    if grad.dtype != dtype:
        grad = torch.ops.autoparallel.dtype_cast(grad, dtype)
    return grad, None, None


torch.library.register_autograd(
    "autoparallel::dtype_cast", backward, setup_context=setup_context
)


@dtype_cast.register_fake
def _(
    x: torch.Tensor, dtype: torch.dtype, grad_dtype: torch.dtype | None = None
) -> torch.Tensor:
    out = torch.empty_like(x, dtype=dtype)
    return out


def canonicalize_mp(mp_policy: MixedPrecisionPolicy) -> MixedPrecisionPolicy:
    # try and follow standard FSDP behavior
    # maybe this should be handled in the MixedPrecisionPolicy class itself
    param_dtype = mp_policy.param_dtype
    reduce_dtype = mp_policy.reduce_dtype or param_dtype
    output_dtype = mp_policy.output_dtype or param_dtype  # TODO: check if this is right
    cast_forward_inputs = mp_policy.cast_forward_inputs
    return MixedPrecisionPolicy(
        param_dtype, reduce_dtype, output_dtype, cast_forward_inputs
    )


@contextmanager
def make_dtype_cast_forward(model: torch.nn.Module, mp_policy: MixedPrecisionPolicy):
    """Scope parameter substitution and root precision boundaries to capture."""
    mp_policy = canonicalize_mp(mp_policy)
    original_forward = model.forward
    had_instance_forward = "forward" in model.__dict__
    original_instance_forward = model.__dict__.get("forward", original_forward)
    cast_dtypes = {}
    for name, param in model.named_parameters():
        dtype = mp_policy.param_dtype or param.dtype
        grad_dtype = mp_policy.reduce_dtype
        if dtype != param.dtype or (
            param.requires_grad
            and grad_dtype is not None
            and grad_dtype != param.grad_dtype
        ):
            cast_dtypes[name] = (dtype, grad_dtype)

    @wraps(original_forward)
    def precision_forward(*args, **kwargs):
        def cast_fn(x):
            if not isinstance(x, torch.Tensor) or not torch.is_floating_point(x):
                return x
            return x.to(mp_policy.param_dtype)

        if mp_policy.cast_forward_inputs:
            args, kwargs = tree_map(cast_fn, args), tree_map(cast_fn, kwargs)
        output = original_forward(*args, **kwargs)

        def cast_out_fn(x):
            if not isinstance(x, torch.Tensor) or not torch.is_floating_point(x):
                return x
            return x.to(mp_policy.output_dtype)

        return tree_map(cast_out_fn, output)

    @wraps(original_forward)
    def forward(*args, **kwargs):
        params = {}
        for name, param in model.named_parameters():
            if name in cast_dtypes:
                dtype, grad_dtype = cast_dtypes[name]
                param = torch.ops.autoparallel.dtype_cast(param, dtype, grad_dtype)
            params[name] = param
        return torch.func.functional_call(model, params, args, kwargs)

    model.forward = precision_forward
    try:
        yield forward
    finally:
        if had_instance_forward:
            model.forward = original_instance_forward
        else:
            del model.forward


def move_dtype_casts_to_first_use(gm: torch.fx.GraphModule) -> None:
    """Place captured casts at first use before AOT generates their backward."""
    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    for node in gm.graph.find_nodes(
        op="call_function", target=torch.ops.autoparallel.dtype_cast
    ):
        if node.users:
            first_user = min(node.users, key=order.__getitem__)
            first_user.prepend(node)
            # Clustering uses source attribution to distinguish call sites.
            for key in ("stack_trace", "nn_module_stack"):
                if key in first_user.meta:
                    node.meta[key] = first_user.meta[key]
    gm.graph.lint()
    gm.recompile()
