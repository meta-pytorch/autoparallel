# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torch.distributed.tensor import DTensor


def _submod_setattr(model, fqn, value):
    module_path, _, buffer_name = fqn.rpartition(".")
    submod: torch.nn.Module = model.get_submodule(module_path)
    setattr(submod, buffer_name, value)


def _build_param_property(parallel_model, fqn):
    def getter(self, _fqn=fqn):
        param = parallel_model.get_parameter(_fqn)
        return param

    def setter(self, value):
        orig_value = parallel_model.get_parameter(fqn)
        new_value = DTensor.from_local(value, device_mesh=orig_value.device_mesh)
        if isinstance(orig_value, torch.nn.Parameter):
            new_value = torch.nn.Parameter(new_value)
        _submod_setattr(parallel_model, fqn, new_value)

    return property(getter, setter)


def _build_buffer_property(parallel_model, fqn):
    def getter(self):
        return parallel_model.get_buffer(fqn)

    def setter(self, value):
        orig_value = parallel_model.get_buffer(fqn)
        new_value = DTensor.from_local(value, device_mesh=orig_value.device_mesh)
        _submod_setattr(parallel_model, fqn, new_value)

    return property(getter, setter)


def hook_params_setters(model, parallel_model):
    """
    Replaces model's parameters with hooked properties that let us
     (a) return a new parameter (from our parallel_mod) instead of the one on the original model,
         similar to using stateless.reparametrize
     (b) also, detect if anyone tries to assign a new value to the parameter, e.g.
         self.layer.weight = nn.Parameter(torch.randn(10, 10))
         would not be properly captured if relying on parametrization alone

    Adds one 'property' (e.g. getter+setter) obj for each parameter name at the right spot in
    the module hierarchy.  For self.layer.weight, this would install a 'weight' property on the self.layer
    submodule.
    """
    for mod_name, mod in sorted(model.named_modules()):
        params_dict = dict(mod.named_parameters(recurse=False))
        buffers_dict = dict(mod.named_buffers(recurse=False))

        namespace = {}
        for p_name in params_dict:
            fqn = mod_name + "." + p_name
            namespace[p_name] = _build_param_property(parallel_model, fqn)

        for b_name in buffers_dict:
            fqn = mod_name + "." + b_name
            namespace[b_name] = _build_buffer_property(parallel_model, fqn)

        cls = mod.__class__
        if hasattr(cls, "init_weights"):
            namespace["init_weights"] = cls.init_weights
        # nn.Module.__setattr__ gets in the way
        namespace["__setattr__"] = object.__setattr__
        mod.__class__ = type(f"HookedInit{cls.__name__}", (cls,), namespace)

    return model
