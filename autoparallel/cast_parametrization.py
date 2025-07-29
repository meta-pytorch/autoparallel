# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager
from typing import Any, Type

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.utils._pytree import tree_map


def _unimplemented_deepcopy(*args: Any, **kwargs: Any):
    raise RuntimeError(
        "DTypeCast does not support deepcopy. Please use state dict for serialization.",
    )


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
        param_properties = {}
        for p_name, p in params_dict.items():

            def getter(
                self_mod=mod,
                _param_name=p_name,
                _dtype=mp_policy.param_dtype,
            ):
                _param = self_mod._parameters[_param_name]
                if not active_param():
                    return _param
                return _param.to(_dtype)

            param_properties[p_name] = getter

        cls = mod.__class__
        param_properties_key = "#".join(sorted(param_properties.keys()))
        new_cls = cls_key_to_dtype_cast_cls.get((cls, param_properties_key), None)
        if not new_cls:
            namespace = {"__deepcopy__": _unimplemented_deepcopy}
            for p_name in param_properties:
                # NOTE: it's important to have this indirection, to make sure that:
                # Different instances of the same class can resolve their parameter access to instance-specific getters
                # (which contains unique objects used in that instance-specific parameter's unshard operation).
                namespace[p_name] = create_dtype_cast_managed_attr(p_name)
            new_cls = type(
                f"DTypeCast{cls.__name__}", (DTypeCastModule, cls), namespace
            )
            cls_key_to_dtype_cast_cls[(cls, param_properties_key)] = new_cls
        mod.__class__ = new_cls
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
