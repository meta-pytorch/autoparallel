# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from functools import wraps

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.utils._pytree import tree_map


@torch.library.custom_op("autoparallel::dtype_cast", mutates_args=())
def dtype_cast(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    This is a custom op that is used to cast the input tensor to the specified dtype.
    We use it to be able to specify a special compute cost for the cast operation,
    so that we always favor performing all-gather of small tensors in the smallest
    dtype.
    """
    return x.to(dtype)


def setup_context(ctx, inputs, output) -> None:
    x, _ = inputs
    ctx.orig_dtype = x.dtype


def backward(ctx, grad):
    dtype = ctx.orig_dtype
    return torch.ops.autoparallel.dtype_cast(grad, dtype), None


torch.library.register_autograd(
    "autoparallel::dtype_cast", backward, setup_context=setup_context
)


@dtype_cast.register_fake
def _(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
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


def make_dtype_cast_forward(model: torch.nn.Module, mp_policy: MixedPrecisionPolicy):
    """Cast parameters during capture without changing the model's module types."""
    mp_policy = canonicalize_mp(mp_policy)

    @wraps(model.forward)
    def forward(*args, **kwargs):
        params = {
            name: (
                torch.ops.autoparallel.dtype_cast(param, mp_policy.param_dtype)
                if mp_policy.param_dtype is not None
                and param.dtype != mp_policy.param_dtype
                else param
            )
            for name, param in model.named_parameters()
        }

        def cast_fn(x):
            if not isinstance(x, torch.Tensor) or not torch.is_floating_point(x):
                return x
            return x.to(mp_policy.param_dtype)

        if mp_policy.cast_forward_inputs:
            args, kwargs = tree_map(cast_fn, args), tree_map(cast_fn, kwargs)
        output = torch.func.functional_call(model, params, args, kwargs)

        def cast_out_fn(x):
            if not isinstance(x, torch.Tensor) or not torch.is_floating_point(x):
                return x
            return x.to(mp_policy.output_dtype)

        return tree_map(cast_out_fn, output)

    return forward
