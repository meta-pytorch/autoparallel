# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable

import torch
from torch.export.unflatten import _AttrKind

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
        curr_mod.register_buffer(field, attr)
    else:
        setattr(curr_mod, field, attr)
