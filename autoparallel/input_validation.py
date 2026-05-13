# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable

import torch
from torch.distributed.tensor import DeviceMesh


def _compute_expected_inputs(traced_inputs, input_placements, mesh):
    """Compute expected runtime inputs by applying sharding to traced global shapes.

    traced_inputs may contain pytree structures (dicts, nested tuples, etc.).
    We flatten them to tensor leaves before applying sharding, since the compiled
    graph operates on flat tensor args.

    Args:
        traced_inputs: The inputs used during tracing (may contain pytrees).
        input_placements: A placement tuple for each tensor leaf, as determined
            by the solver's solution.
        mesh: The device mesh.

    Returns:
        A tuple of (expected_inputs, dynamic_dims) where expected_inputs is a
        flat list of meta tensors (for tensor inputs) or raw values (for
        non-tensor inputs), and dynamic_dims is a set of (arg_index, dim) pairs
        indicating which dimensions are dynamic and should not be checked.
    """
    import torch.utils._pytree as pytree
    from torch.distributed.tensor.placement_types import Shard

    flat_inputs, _ = pytree.tree_flatten(traced_inputs)

    result = []
    dynamic_dims: set[tuple[int, int]] = set()
    result_idx = 0
    tensor_idx = 0
    for inp in flat_inputs:
        if isinstance(inp, torch.Tensor):
            placements = input_placements[tensor_idx]
            local_shape = list(inp.shape)
            for d, s in enumerate(inp.shape):
                if isinstance(s, torch.SymInt) and not s.node.expr.is_number:
                    dynamic_dims.add((result_idx, d))
            for mesh_dim, placement in enumerate(placements):
                if isinstance(placement, Shard):
                    dim = placement.dim
                    mesh_dim_size = mesh.size(mesh_dim)
                    local_shape[dim] = (
                        local_shape[dim] + mesh_dim_size - 1
                    ) // mesh_dim_size
            result.append(torch.empty(local_shape, dtype=inp.dtype, device="meta"))
            tensor_idx += 1
        else:
            result.append(inp)
        result_idx += 1

    return result, dynamic_dims


def _check_forward_args(args, expected_inputs, dynamic_dims=frozenset()):
    """Validate that forward() args match the shapes/dtypes used during tracing.

    Dynamic dimensions (identified by dynamic_dims) accept any size.
    Concrete dimensions are checked exactly.

    Args:
        args: The actual forward arguments (flat list).
        expected_inputs: Expected shapes from _compute_expected_inputs.
        dynamic_dims: Set of (arg_index, dim) pairs for dynamic dimensions.
    """
    if len(args) != len(expected_inputs):
        raise ValueError(
            f"AutoParallel: expected {len(expected_inputs)} arguments "
            f"but got {len(args)}"
        )
    for i, (arg, expected) in enumerate(zip(args, expected_inputs)):
        if isinstance(expected, torch.Tensor):
            if not isinstance(arg, torch.Tensor):
                raise TypeError(
                    f"AutoParallel: argument {i} should be a Tensor "
                    f"but got {type(arg).__name__}"
                )
            if len(arg.shape) != len(expected.shape):
                raise ValueError(
                    f"AutoParallel: argument {i} has {len(arg.shape)} dims "
                    f"but expected {len(expected.shape)} dims"
                )
            for dim, (actual, exp) in enumerate(zip(arg.shape, expected.shape)):
                if (i, dim) in dynamic_dims:
                    continue
                if isinstance(exp, torch.SymInt) and not exp.node.expr.is_number:
                    continue
                if actual != exp:
                    raise ValueError(
                        f"AutoParallel: argument {i} has shape {tuple(arg.shape)} "
                        f"but expected {tuple(expected.shape)}"
                    )
            if arg.dtype != expected.dtype:
                raise ValueError(
                    f"AutoParallel: argument {i} has dtype {arg.dtype} "
                    f"but expected {expected.dtype}"
                )
        else:
            if arg != expected:
                raise ValueError(
                    f"AutoParallel: argument {i} has value {arg!r} "
                    f"but was traced with {expected!r}"
                )


def _extract_input_info(
    sample_inputs: Any, mesh: DeviceMesh
) -> tuple[
    list[tuple[int, ...]],
    list[torch.dtype],
    list[tuple[Any, ...]],
    Any,
    list[torch.device],
]:
    """
    Extract tensor metadata and placements from sample inputs (supports pytrees).

    For DTensor inputs, extracts global shape, dtype, and placements.
    For regular Tensor inputs, uses shape/dtype and assumes Replicate.

    Does NOT materialize tensors - just extracts metadata.

    Returns:
        - List of shapes (global shapes for DTensors)
        - List of dtypes
        - List of placement tuples for each tensor leaf
        - TreeSpec for reconstructing the pytree structure
        - List of devices for each tensor leaf
    """
    import torch.utils._pytree as pytree
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.placement_types import Replicate

    flat_inputs, treespec = pytree.tree_flatten(sample_inputs)

    shapes = []
    dtypes = []
    input_placements = []
    devices = []

    for inp in flat_inputs:
        if isinstance(inp, DTensor):
            # DTensor.shape returns the global shape
            shapes.append(tuple(inp.shape))
            dtypes.append(inp.dtype)
            input_placements.append(tuple(inp.placements))
            devices.append(inp.device)
        elif isinstance(inp, torch.Tensor):
            shapes.append(tuple(inp.shape))
            dtypes.append(inp.dtype)
            input_placements.append(tuple(Replicate() for _ in range(mesh.ndim)))
            devices.append(inp.device)
        else:
            raise TypeError(
                f"sample_inputs leaves must be Tensor or DTensor, got {type(inp)}"
            )

    return shapes, dtypes, input_placements, treespec, devices


def _make_input_fn(
    shapes: list[tuple[int, ...]],
    dtypes: list[torch.dtype],
    treespec: Any,
    devices: list[torch.device],
) -> Callable[[], tuple[Any, ...]]:
    """
    Create an input_fn that creates tensors with the given shapes/dtypes/devices.

    The returned function should be called inside FakeTensorMode.
    It creates new tensors (which will be fake tensors when called in FakeTensorMode).

    Returns:
        Callable that returns inputs as a tuple.
    """
    import torch.utils._pytree as pytree

    def input_fn() -> tuple[Any, ...]:
        # Create tensors inside FakeTensorMode - they'll be fake tensors
        tensors = [
            torch.empty(shape, dtype=dtype, device=device)
            for shape, dtype, device in zip(shapes, dtypes, devices)
        ]
        result = pytree.tree_unflatten(tensors, treespec)

        # AutoParallel expects input_fn to return a tuple
        if isinstance(result, tuple):
            return result
        else:
            return (result,)

    return input_fn


def _flatten_out_shardings(
    out_shardings: Any,
) -> list[tuple[Any, ...]]:
    """
    Flatten out_shardings to a list of placement tuples.

    The out_shardings should match the structure of the model output.
    Each leaf should be a tuple of Placements.

    Handles nested structures by recursively walking until we find placement tuples.
    """
    from torch.distributed.tensor.placement_types import Placement

    def is_placement_tuple(obj: Any) -> bool:
        if not isinstance(obj, tuple):
            return False
        if len(obj) == 0:
            return False
        return all(isinstance(p, Placement) for p in obj)

    def collect_placement_tuples(obj: Any, result: list) -> None:
        """Recursively collect placement tuples from a nested structure."""
        if is_placement_tuple(obj):
            result.append(obj)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                collect_placement_tuples(item, result)
        elif isinstance(obj, dict):
            for item in obj.values():
                collect_placement_tuples(item, result)
        else:
            raise TypeError(
                f"out_shardings must contain tuples of Placements, "
                f"got {type(obj)}: {obj}"
            )

    result: list[tuple[Any, ...]] = []
    collect_placement_tuples(out_shardings, result)

    if not result:
        raise ValueError("out_shardings must contain at least one placement tuple")

    return result
