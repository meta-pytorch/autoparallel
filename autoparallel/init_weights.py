# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
from contextlib import contextmanager

import torch
from torch._dynamo.utils import warn_once
from torch.distributed.tensor import DTensor
from torch.utils._python_dispatch import TorchDispatchMode


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


class _InitWeightsDispatchMode(TorchDispatchMode):
    """Intercepts in-place copy operations on DTensors during init_weights.

    When a user's init_weights does ``self.weight.data[:] = value``, the ``.data``
    accessor on a DTensor returns a (detached) DTensor.  The ``[:] = value`` then
    dispatches ``aten.copy_`` with a DTensor dst and a plain-tensor src, which
    DTensor rejects as mixed types.  This mode intercepts such calls, wraps the
    src as a Replicated DTensor, and performs a proper DTensor-to-DTensor copy
    that handles redistribution automatically.
    """

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        if func == torch.ops.aten.copy_.default:
            dst = args[0]
            src = args[1]
            if isinstance(dst, DTensor) and not isinstance(src, DTensor):
                # Interpret src as a full/global tensor and wrap it as
                # Replicated, then copy into dst which redistributes
                # automatically (e.g. Replicate → Shard).
                new_src = DTensor.from_local(src, device_mesh=dst.device_mesh)
                with torch.no_grad():
                    dst.copy_(new_src)
                return dst
        return func(*args, **(kwargs or {}))


@contextmanager
def _init_weights_context(parallel_model):
    """Context manager that intercepts parameter/buffer assignments during init_weights.

    Temporarily overrides ``nn.Module.__setattr__`` so that assignments like
    ``self.weight = nn.Parameter(torch.ones(...))`` copy the new value into the
    existing DTensor (preserving its placement) rather than replacing it.

    Also activates ``_InitWeightsDispatchMode`` to handle in-place data
    operations like ``self.weight.data[:] = value``.
    """
    # Build map of (module_id, attr_name) -> (fqn, dtensor) for all DTensor
    # params and buffers. We iterate per-module with recurse=False to capture
    # aliased params/buffers that named_parameters() would deduplicate.
    dtensor_map: dict[tuple[int, str], tuple[str, DTensor]] = {}
    for mod_name, mod in parallel_model.named_modules():
        for p_name, param in mod.named_parameters(recurse=False):
            if isinstance(param, DTensor):
                fqn = f"{mod_name}.{p_name}" if mod_name else p_name
                dtensor_map[(id(mod), p_name)] = (fqn, param)
        for b_name, buf in mod.named_buffers(recurse=False):
            if isinstance(buf, DTensor):
                fqn = f"{mod_name}.{b_name}" if mod_name else b_name
                dtensor_map[(id(mod), b_name)] = (fqn, buf)

    original_setattr = torch.nn.Module.__setattr__

    def _patched_setattr(self, name, value):  # type: ignore[no-untyped-def]
        key = (id(self), name)
        if key in dtensor_map:
            if isinstance(value, DTensor):
                # Already a DTensor (e.g. aliased buffer re-assignment).
                original_setattr(self, name, value)
                return
            set_value = value.data if isinstance(value, torch.nn.Parameter) else value
            fqn, existing_dtensor = dtensor_map[key]
            _copy_set_value_to_dtensor(fqn, existing_dtensor, set_value)
            return
        original_setattr(self, name, value)

    # Guard against `dtensor.data = value` which silently fails (the C++ storage
    # swap bypasses __torch_dispatch__ entirely, so the assignment is lost).
    # We shadow the inherited C++ `.data` descriptor with a Python property on
    # DTensor that raises a clear error on `__set__`.
    _original_data_descriptor = DTensor.__dict__.get("data", None)

    @property  # type: ignore[misc]
    def _guarded_data(self: DTensor) -> torch.Tensor:
        return torch.Tensor.data.__get__(self)  # type: ignore[attr-defined]

    @_guarded_data.setter
    def _guarded_data(self: DTensor, value: torch.Tensor) -> None:
        raise RuntimeError(
            "Cannot use `.data = ...` on a DTensor during init_weights — "
            "the assignment is silently lost because it bypasses DTensor dispatch. "
            "Use `self.<name> = value` or `self.<name>.data[:] = value` instead."
        )

    torch.nn.Module.__setattr__ = _patched_setattr  # type: ignore[assignment]
    DTensor.data = _guarded_data  # type: ignore[assignment]
    try:
        with _InitWeightsDispatchMode():
            yield
    finally:
        torch.nn.Module.__setattr__ = original_setattr  # type: ignore[assignment]
        if _original_data_descriptor is not None:
            DTensor.data = _original_data_descriptor  # type: ignore[assignment]
        else:
            del DTensor.data  # restore inheritance from torch.Tensor


def wrap_init_weights(parallel_model):
    """Wraps ``parallel_model.init_weights`` with DTensor-aware interception.

    After calling this, ``parallel_model.init_weights()`` will automatically
    handle parameter/buffer assignments and in-place data operations so that
    the user's single-GPU init_weights code works on the sharded parallel model.
    """
    if not hasattr(parallel_model, "init_weights"):
        return

    original_init_weights = parallel_model.init_weights

    def wrapped_init_weights(*args, **kwargs):  # type: ignore[no-untyped-def]
        with _init_weights_context(parallel_model):
            return original_init_weights(*args, **kwargs)

    parallel_model.init_weights = wrapped_init_weights  # type: ignore[assignment]
