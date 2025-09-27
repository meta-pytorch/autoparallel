# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import copy
import copyreg
from contextlib import contextmanager
from typing import Type

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.utils._pytree import tree_map


def make_getter(self, p_name, mp_policy):
    def getter(
        self_mod=self,
        _param_name=p_name,
        _dtype=mp_policy.param_dtype,
    ):
        _param = self_mod._parameters[_param_name]
        if not active_param():
            return _param
        return torch.ops.autoparallel.dtype_cast(_param, _dtype)

    return getter


# taken from PyTorch's parametrize module from
# https://github.com/pytorch/pytorch/blob/5d9653d90ee003173dd03f93e09fed236500ef06/torch/nn/utils/parametrize.py#L324-L351
# with some improvements
def default_deepcopy(self, memo):
    # Just emulate a standard deepcopy procedure when __deepcopy__ doesn't exist in the current class.
    obj = memo.get(id(self), None)
    if obj is not None:
        return obj
    replica = self.__new__(self.__class__)
    memo[id(self)] = replica
    replica.__dict__ = copy.deepcopy(self.__dict__, memo)

    # Fix the parametrization getters to point to the replica instead of the original
    if hasattr(replica, "_name_to_dtype_cast_managed_attr_getter") and hasattr(
        replica, "_mp_policy"
    ):
        # Recreate the getter functions to point to the replica
        param_properties = {}
        for p_name in list(replica._name_to_dtype_cast_managed_attr_getter.keys()):
            # Use a function factory to properly capture the loop variable
            # def make_getter(param_name):
            param_properties[p_name] = make_getter(replica, p_name, replica._mp_policy)
        replica._name_to_dtype_cast_managed_attr_getter = param_properties

    # Also save all slots if they exist.
    slots_to_save = copyreg._slotnames(self.__class__)  # type: ignore[attr-defined]
    for slot in slots_to_save:
        if hasattr(self, slot):
            setattr(replica, slot, copy.deepcopy(getattr(self, slot), memo))
    return replica


def getstate(self):
    raise RuntimeError(
        "Serialization of parametrized modules is only "
        "supported through state_dict(). See:\n"
        "https://pytorch.org/tutorials/beginner/saving_loading_models.html"
        "#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training"
    )


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


def create_dtype_cast_managed_attr(p_name):
    def getter(self):
        # TODO: if this function throws exception, how does it behave? add a unit test for it.
        return self._name_to_dtype_cast_managed_attr_getter[p_name]()

    def setter(self, value):
        raise RuntimeError(
            "Setting DTypeCast-managed attribute is not supported",
        )

    return property(getter, setter)


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


_active_param = False


def active_param():
    global _active_param
    return _active_param


@contextmanager
def set_dtype_cast(val):
    global _active_param
    prev = _active_param
    try:
        _active_param = val
        yield
    finally:
        _active_param = prev


# taken from https://www.internalfb.com/code/fbsource/[master][history]/fbcode/caffe2/torch/distributed/fb/simple_fsdp/simple_fsdp.py
# with minor modifications
def apply_dtype_cast(model, mp_policy: MixedPrecisionPolicy):
    mp_policy = canonicalize_mp(mp_policy)
    cls_key_to_dtype_cast_cls: dict[tuple[Type, str], Type] = {}

    for mod_name, mod in sorted(model.named_modules()):
        params_dict = dict(mod.named_parameters(recurse=False))

        # Create new class for this module with all parametrized parameters
        cls = mod.__class__
        param_properties_key = "#".join(sorted(params_dict.keys()))
        new_cls = cls_key_to_dtype_cast_cls.get((cls, param_properties_key), None)
        if not new_cls:
            namespace = {"__getstate__": getstate}
            # We don't allow serialization of parametrized modules but should still allow deepcopying.
            # Default 'deepcopy' function invokes __deepcopy__ method instead of __getstate__ when it exists.
            if not hasattr(cls, "__deepcopy__"):
                namespace["__deepcopy__"] = default_deepcopy  # type: ignore[assignment]

            for p_name in params_dict.keys():
                # NOTE: it's important to have this indirection, to make sure that:
                # Different instances of the same class can resolve their parameter access to instance-specific getters
                # (which contains unique objects used in that instance-specific parameter's unshard operation).
                namespace[p_name] = create_dtype_cast_managed_attr(p_name)
            cls_t = (DTypeCastModule, cls) if mod is model else (cls,)
            new_cls = type(f"DTypeCast{cls.__name__}", cls_t, namespace)
            cls_key_to_dtype_cast_cls[(cls, param_properties_key)] = new_cls
        mod.__class__ = new_cls

        param_properties = {}
        for p_name in params_dict.keys():
            param_properties[p_name] = make_getter(mod, p_name, mp_policy)

        mod._name_to_dtype_cast_managed_attr_getter = param_properties
        mod._mp_policy = mp_policy

    return model


class DTypeCastModule(torch.nn.Module):
    def forward(self, *args, **kwargs):
        def cast_fn(x):
            if not torch.is_floating_point(x):
                return x
            return x.to(self._mp_policy.param_dtype)

        if self._mp_policy.cast_forward_inputs:
            args, kwargs = tree_map(cast_fn, args), tree_map(cast_fn, kwargs)
        output = super().forward(*args, **kwargs)

        def cast_out_fn(x):
            return x.to(self._mp_policy.output_dtype)

        output = tree_map(cast_out_fn, output)
        return output
