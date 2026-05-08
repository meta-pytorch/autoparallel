# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable

import torch
from torch.export.unflatten import _AttrKind

from .cast_parametrization import DTypeCastModule
from .init_weights import wrap_init_weights

_NN_MODULE_KEYS = set(torch.nn.Module().__dict__.keys())


def _build_alias_map(
    named_iter_fn: Callable[..., Any],
) -> dict[str, str]:
    """Build a mapping from alias FQNs to canonical FQNs.

    named_parameters()/named_buffers() deduplicate by default, so when a model
    registers the same tensor under multiple FQNs only one survives. This
    function detects the aliases so they can be re-registered later.
    """
    canonical_by_id: dict[int, str] = {}
    canonical_fqns: set[str] = set()
    for fqn, tensor in named_iter_fn():
        canonical_by_id[id(tensor)] = fqn
        canonical_fqns.add(fqn)
    alias_map: dict[str, str] = {}
    for fqn, tensor in named_iter_fn(remove_duplicate=False):
        if fqn not in canonical_fqns and id(tensor) in canonical_by_id:
            alias_map[fqn] = canonical_by_id[id(tensor)]
    return alias_map


def _build_module_alias_map(model: torch.nn.Module) -> dict[str, str]:
    """Build a mapping from alias module FQNs to canonical module FQNs.

    When a model registers the same module under multiple names (e.g.
    self.model_ema = self.teacher), named_modules() deduplicates by default.
    This detects such aliases so they can be re-established on the parallel model.
    """
    canonical_by_id: dict[int, str] = {}
    canonical_fqns: set[str] = set()
    for fqn, mod in model.named_modules():
        if fqn == "":
            continue
        canonical_by_id[id(mod)] = fqn
        canonical_fqns.add(fqn)
    alias_map: dict[str, str] = {}
    for fqn, mod in model.named_modules(remove_duplicate=False):
        if fqn == "":
            continue
        if fqn not in canonical_fqns and id(mod) in canonical_by_id:
            alias_map[fqn] = canonical_by_id[id(mod)]
    return alias_map


def _assign_attr(
    attr: Any,
    target_module: torch.nn.Module,
    ref_module: torch.nn.Module,
    fqn: str,
    attr_kind: _AttrKind,
):
    """
    Custom version of torch.export._unlift._assign_attr that preserves the original
    module structure (e.g., nn.ModuleDict) from ref_module.

    Args:
        attr: The attribute to assign (parameter/buffer/module)
        target_module: The module to assign the attribute to
        ref_module: Reference module to check for original structure
        fqn: Fully qualified name of the attribute (e.g., "layers.0.weight")
        attr_kind: Type of attribute (PARAMETER, BUFFER, etc.)
    """
    *prefix, field = fqn.split(".")

    # Navigate to the parent module, creating submodules as needed
    curr_mod = target_module
    for i, attr_name in enumerate(prefix):
        if not hasattr(curr_mod, attr_name):
            # Check if we should create a module matching the ref_module type
            # Navigate to the same location in ref_module
            ref_curr_mod = ref_module
            for ref_attr_name in prefix[:i]:
                if hasattr(ref_curr_mod, ref_attr_name):
                    ref_curr_mod = getattr(ref_curr_mod, ref_attr_name)
                else:
                    ref_curr_mod = None  # type: ignore[assignment]
                    break

            # Create an instance of the same type as in ref_module
            if ref_curr_mod is not None and hasattr(ref_curr_mod, attr_name):
                ref_submod = getattr(ref_curr_mod, attr_name)
                cls = type(ref_submod)
                try:
                    cls = type(ref_submod)
                    new_inst = ref_submod.__new__(cls)
                    new_inst.__dict__ = ref_submod.__dict__.copy()
                    setattr(curr_mod, attr_name, new_inst)
                except Exception:
                    # Fall back to regular Module if instantiation fails
                    setattr(curr_mod, attr_name, torch.nn.Module())
            else:
                setattr(curr_mod, attr_name, torch.nn.Module())

        curr_mod = getattr(curr_mod, attr_name)

    # Set the final attribute
    if attr_kind == _AttrKind.PARAMETER:
        assert isinstance(attr, torch.nn.Parameter)
        curr_mod.register_parameter(field, attr)
    elif attr_kind == _AttrKind.BUFFER:
        assert isinstance(attr, torch.Tensor)
        ref_curr_mod = ref_module
        for attr_name in prefix:
            ref_curr_mod = getattr(ref_curr_mod, attr_name)
        curr_mod.register_buffer(
            field,
            attr,
            persistent=field not in ref_curr_mod._non_persistent_buffers_set,
        )
    else:
        setattr(curr_mod, field, attr)


def make_parallel_module(
    ref_model: torch.nn.Module,
    sharded_param_dict: dict[str, torch.nn.Parameter],
    sharded_buffer_dict: dict[str, torch.Tensor],
    forward_fn: Callable | None = None,
) -> torch.nn.Module:
    """Create a parallel module populated with sharded params/buffers.

    Handles UserModelClass resolution (stripping DTypeCastModule), user attribute
    copying, param/buffer registration, alias re-establishment, orphan submodule
    copying, and init_weights wrapping.

    Alias detection (param, buffer, and module aliases) is performed automatically
    from ref_model, which must preserve aliasing (e.g. move_to_fake does this).

    Args:
        ref_model: The original (possibly fake) model to mirror structure from.
        sharded_param_dict: FQN -> sharded Parameter mapping.
        sharded_buffer_dict: FQN -> sharded buffer mapping.
        forward_fn: If provided, used as the forward method on the new module.
    """
    param_alias_map = _build_alias_map(ref_model.named_parameters)
    buffer_alias_map = _build_alias_map(ref_model.named_buffers)
    module_alias_map = _build_module_alias_map(ref_model)
    UserModelClass = type(ref_model)
    if issubclass(UserModelClass, DTypeCastModule):
        UserModelClass = UserModelClass.__bases__[1]

    class ParallelModule(UserModelClass):  # type: ignore[valid-type,misc]
        def __init__(self):
            torch.nn.Module.__init__(self)

    if forward_fn is not None:
        ParallelModule.forward = forward_fn

    mod = ParallelModule()

    # Copy user-defined instance attributes (e.g. self.dim, self.config).
    # Uses __dict__ directly to avoid triggering nn.Module.__setattr__
    # which intercepts nn.Parameter/nn.Module assignments.
    for k, v in ref_model.__dict__.items():
        if k not in _NN_MODULE_KEYS:
            mod.__dict__[k] = v

    # Register sharded params and buffers, preserving original module
    # structure (e.g. nn.ModuleDict) from ref_model.
    for k, v in sharded_param_dict.items():
        _assign_attr(v, mod, ref_model, k, attr_kind=_AttrKind.PARAMETER)
    for k, v in sharded_buffer_dict.items():
        _assign_attr(v, mod, ref_model, k, attr_kind=_AttrKind.BUFFER)

    # Re-establish aliased params/buffers that were deduplicated during tracing.
    # The alias map's "canonical" may not match which FQN the tracer kept,
    # so we check both directions.
    for alias_fqn, canonical_fqn in param_alias_map.items():
        if canonical_fqn in sharded_param_dict:
            _assign_attr(
                mod.get_parameter(canonical_fqn),
                mod,
                ref_model,
                alias_fqn,
                attr_kind=_AttrKind.PARAMETER,
            )
        elif alias_fqn in sharded_param_dict:
            _assign_attr(
                mod.get_parameter(alias_fqn),
                mod,
                ref_model,
                canonical_fqn,
                attr_kind=_AttrKind.PARAMETER,
            )
    for alias_fqn, canonical_fqn in buffer_alias_map.items():
        if canonical_fqn in sharded_buffer_dict:
            _assign_attr(
                mod.get_buffer(canonical_fqn),
                mod,
                ref_model,
                alias_fqn,
                attr_kind=_AttrKind.BUFFER,
            )
        elif alias_fqn in sharded_buffer_dict:
            _assign_attr(
                mod.get_buffer(alias_fqn),
                mod,
                ref_model,
                canonical_fqn,
                attr_kind=_AttrKind.BUFFER,
            )

    # Re-establish module aliases (e.g. model_ema -> teacher) so that
    # both FQNs point to the same submodule on the parallel model.
    for alias_fqn, canonical_fqn in module_alias_map.items():
        canonical_mod: Any = mod
        for attr in canonical_fqn.split("."):
            canonical_mod = getattr(canonical_mod, attr, None)
            if canonical_mod is None:
                break
        if canonical_mod is None:
            continue
        *alias_prefix, alias_field = alias_fqn.split(".")
        alias_parent: Any = mod
        for attr in alias_prefix:
            alias_parent = getattr(alias_parent, attr, None)
            if alias_parent is None:
                break
        if alias_parent is not None:
            setattr(alias_parent, alias_field, canonical_mod)

    # Copy submodules that don't appear in any parameter/buffer FQN path
    # (e.g. self.rope when it has no traced params/buffers). These aren't
    # created by _assign_attr, but init_weights may need to access them.
    for k, v in ref_model._modules.items():
        if k not in mod._modules:
            mod._modules[k] = v

    wrap_init_weights(mod)
    return mod
