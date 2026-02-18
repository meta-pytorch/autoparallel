# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Union

import torch
from torch._dynamo.utils import warn_once
from torch.distributed.tensor import DTensor
from torch.utils._python_dispatch import TorchDispatchMode


def _submod_setattr(model: torch.nn.Module, fqn: str, value: Any):
    module_path, _, buffer_name = fqn.rpartition(".")
    submod: torch.nn.Module = model.get_submodule(module_path)
    setattr(submod, buffer_name, value)


def _copy_set_value_to_dtensor(
    fqn: str, parallel_value: DTensor, set_value: torch.Tensor
):
    # We expect the user wrote their module's init_weights in terms of a single-gpu model, so we do not expect
    # set_value to be a DTensor already (since this would imply init_weights was written in a 'distributed' way),
    # and we interpret it as a global tensor which we map to a Replicated DTensor.
    assert not isinstance(
        set_value, DTensor
    ), "Expected local/full tensor from setattr in init_weights, not DTensor."

    # This creates a replicated DTensor
    new_parallel_value = DTensor.from_local(
        set_value, device_mesh=parallel_value.device_mesh
    )
    if parallel_value.placements != new_parallel_value.placements:
        # no harm done if the parallel value is replicated, e.g. freqs_cis in llama3, but it would be
        # noticeably wasteful if we do this for all the sharded parameters.
        warn_once(
            f"init_weights set a new value for {fqn}, "
            f"but the existing value is already sharded ({parallel_value.placements=},  "
            "and it is wasteful to materialize the new value as a global tensor. "
            "Change init_weights to perform an inplace initialization instead if possible."
        )
    with torch.no_grad():
        # This ensures that we faithfully redistribute the replicated new_parallel_value into whatever placement
        # the autoparallel engine decided for parallel_value.  Note: this should in general be comm free, since it
        # would be going from Replicate -> Shard.
        parallel_value.copy_(new_parallel_value)


def _build_param_property(parallel_model: torch.nn.Module, fqn: str):
    def getter(self) -> torch.nn.Parameter:
        param = parallel_model.get_parameter(fqn)
        return param

    def setter(self, value: Union[torch.Tensor, torch.nn.Parameter]) -> None:
        parallel_value = parallel_model.get_parameter(fqn)
        assert isinstance(
            parallel_value, DTensor
        ), "Expected parallel_module params to be DTensors"
        _copy_set_value_to_dtensor(fqn, parallel_value, value)

    return property(getter, setter)


def _build_buffer_property(parallel_model: torch.nn.Module, fqn: str):
    def getter(self) -> torch.Tensor:
        return parallel_model.get_buffer(fqn)

    def setter(self, value: torch.Tensor) -> None:
        parallel_value = parallel_model.get_buffer(fqn)
        assert isinstance(
            parallel_value, DTensor
        ), "Expected parallel_module params to be DTensors"
        _copy_set_value_to_dtensor(fqn, parallel_value, value)

    return property(getter, setter)


class _InitWeightsDispatchMode(TorchDispatchMode):
    """Intercepts in-place copy operations on DTensor parameter local tensors
    during init_weights execution.

    When a user's init_weights does ``self.weight.data[:] = value``, the ``.data``
    accessor on a DTensor returns the local tensor (which has shard shape for
    sharded parameters). The ``[:] = value`` then dispatches ``aten.copy_`` from
    the global-shaped value into the shard-shaped local tensor, causing a shape
    mismatch. This mode intercepts such ``copy_`` calls and redirects them through
    ``_copy_set_value_to_dtensor`` for proper redistribution.
    """

    def __init__(self, parallel_model: torch.nn.Module):
        super().__init__()
        self.param_data_ptrs: dict[int, tuple[str, DTensor]] = {}
        for fqn, param in parallel_model.named_parameters():
            if isinstance(param, DTensor):
                self.param_data_ptrs[param.data_ptr()] = (fqn, param)
        for fqn, buf in parallel_model.named_buffers():
            if isinstance(buf, DTensor):
                self.param_data_ptrs[buf.data_ptr()] = (fqn, buf)
        self._handling = False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        if func == torch.ops.aten.copy_.default and not self._handling:
            dst = args[0]
            src = args[1]
            if dst.data_ptr() in self.param_data_ptrs and not isinstance(
                src, DTensor
            ):
                fqn, dtensor = self.param_data_ptrs[dst.data_ptr()]
                # Prevent re-entrant interception: _copy_set_value_to_dtensor
                # internally calls parallel_value.copy_(), which dispatches
                # further copy_ ops on the same local tensor.
                self._handling = True
                try:
                    _copy_set_value_to_dtensor(fqn, dtensor, src)
                finally:
                    self._handling = False
                return dst
        return func(*args, **(kwargs or {}))


def hook_params_setters(
    init_weights_model: torch.nn.Module, parallel_model: torch.nn.Module
) -> None:
    """
    Replaces init_weights_model's parameters with hooked properties that let us
     (a) return a new parameter (from our parallel_mod) instead of the one on the original model,
         similar to using stateless.reparametrize
     (b) also, detect if anyone tries to assign a new value to the parameter, e.g.
         self.layer.weight = nn.Parameter(torch.randn(10, 10))
         would not be properly captured if relying on parametrization alone

    Assumes init_weights_model is a deepcopy of the user's original model, with all fake params. This way we can
    modify the model to enable init_weights to work, without affecting the user's original model.

    Adds one 'property' (e.g. getter+setter) obj for each parameter name at the right spot in
    the module hierarchy.  For self.layer.weight, this would install a 'weight' property on the self.layer
    submodule.

    Also wraps init_weights_model.init_weights (if present) with a TorchDispatchMode
    to handle in-place data operations like ``self.weight.data[:] = value``.
    """
    for mod_name, mod in sorted(init_weights_model.named_modules()):
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
        # nn.Module.__setattr__ gets in the way
        namespace["__setattr__"] = object.__setattr__
        mod.__class__ = type(f"HookedInit{cls.__name__}", (cls,), namespace)

    # Wrap init_weights to activate a dispatch mode that intercepts in-place
    # copy operations (e.g. self.weight.data[:] = value) on DTensor local tensors.
    if hasattr(init_weights_model, "init_weights"):
        mode = _InitWeightsDispatchMode(parallel_model)
        original_init_weights = init_weights_model.init_weights

        def wrapped_init_weights(*args, **kwargs):  # type: ignore[no-untyped-def]
            with mode:
                return original_init_weights(*args, **kwargs)

        init_weights_model.init_weights = wrapped_init_weights
