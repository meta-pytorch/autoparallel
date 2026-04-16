# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable

import torch
from torch.distributed.tensor import DeviceMesh


def _get_expected_dim_value(exp):
    """Return a concrete expected dim value when a symbolic dim collapsed to one."""
    if not isinstance(exp, torch.SymInt):
        return exp
    expr = exp.node.expr
    if expr.is_number:
        return int(expr)
    return None


def _compute_expected_inputs(traced_inputs, input_constraints, mesh):
    """Compute expected runtime inputs by applying sharding to traced global shapes.

    Returns a list of meta tensors (for tensor inputs) or raw values (for non-tensor inputs).
    """
    from torch.distributed.tensor.placement_types import Replicate, Shard

    default_placement = (Shard(0),) + (Replicate(),) * (mesh.ndim - 1)

    result = []
    tensor_idx = 0
    for inp in traced_inputs:
        if isinstance(inp, torch.Tensor):
            if input_constraints is None:
                placements = default_placement
            else:
                placements = input_constraints[tensor_idx]
                if placements is None:
                    placements = default_placement

            local_shape = list(inp.shape)
            for mesh_dim, placement in enumerate(placements):
                if isinstance(placement, Shard):
                    local_shape[placement.dim] //= mesh.size(mesh_dim)
            result.append(torch.empty(local_shape, dtype=inp.dtype, device="meta"))
            tensor_idx += 1
        else:
            result.append(inp)

    return result


def _check_forward_args(args, expected_inputs):
    """Validate that forward() args match the shapes/dtypes used during tracing.

    When dynamic shapes are enabled, dimensions that are SymInt in the expected
    shape accept any size. Concrete dimensions are checked exactly.
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
                expected_dim = _get_expected_dim_value(exp)
                if expected_dim is None:
                    continue
                if actual != expected_dim:
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
) -> tuple[list[tuple[int, ...]], list[torch.dtype], list[tuple[Any, ...]], Any]:
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
    """
    import torch.utils._pytree as pytree
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.placement_types import Replicate

    flat_inputs, treespec = pytree.tree_flatten(sample_inputs)

    shapes = []
    dtypes = []
    input_placements = []

    for inp in flat_inputs:
        if isinstance(inp, DTensor):
            # DTensor.shape returns the global shape
            shapes.append(tuple(inp.shape))
            dtypes.append(inp.dtype)
            input_placements.append(tuple(inp.placements))
        elif isinstance(inp, torch.Tensor):
            shapes.append(tuple(inp.shape))
            dtypes.append(inp.dtype)
            input_placements.append(tuple(Replicate() for _ in range(mesh.ndim)))
        else:
            raise TypeError(
                f"sample_inputs leaves must be Tensor or DTensor, got {type(inp)}"
            )

    return shapes, dtypes, input_placements, treespec


def _make_input_fn(
    shapes: list[tuple[int, ...]],
    dtypes: list[torch.dtype],
    treespec: Any,
) -> Callable[[], tuple[Any, ...]]:
    """
    Create an input_fn that creates tensors with the given shapes/dtypes.

    The returned function should be called inside FakeTensorMode.
    It creates new tensors (which will be fake tensors when called in FakeTensorMode).

    Returns:
        Callable that returns inputs as a tuple.
    """
    import torch.utils._pytree as pytree

    def input_fn() -> tuple[Any, ...]:
        # Create tensors inside FakeTensorMode - they'll be fake tensors
        tensors = [
            torch.empty(shape, dtype=dtype, device="cuda")
            for shape, dtype in zip(shapes, dtypes)
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
