# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import functools
import math
from collections.abc import Callable, Sequence
from typing import Any, Optional, Tuple

import torch
import torch.distributed.distributed_c10d as c10d
from torch.distributed._tensor.experimental import local_map as _local_map
from torch.distributed.device_mesh import DeviceMesh, _mesh_resources
from torch.distributed.distributed_c10d import GroupName
from torch.distributed.tensor.placement_types import Placement

_FLEX_LOCAL_MAP_ALTERNATIVES_ATTR = "_autoparallel_flex_local_map_alternatives"


# Dynamo's local_map HOP only preserves placement kwargs in FX metadata.
# Carry flex metadata on out_placements so slicing in the HOP wrapper keeps it.
class _FlexLocalMapOutPlacements(tuple):
    def __new__(cls, values, alternatives):
        obj = super().__new__(cls, values)
        setattr(obj, _FLEX_LOCAL_MAP_ALTERNATIVES_ATTR, alternatives)
        return obj

    def __getnewargs__(self):
        # AutoParallel deep-copies the user model (api.py), which reconstructs
        # this tuple subclass via copyreg.__newobj__ -> __new__(cls, *args).
        # The two-arg __new__ requires we surface `alternatives` here too.
        return (tuple(self), getattr(self, _FLEX_LOCAL_MAP_ALTERNATIVES_ATTR))

    def __getitem__(self, item):
        result = super().__getitem__(item)
        if isinstance(item, slice):
            return type(self)(result, getattr(self, _FLEX_LOCAL_MAP_ALTERNATIVES_ATTR))
        return result


def get_flex_local_map_alternatives(local_map_kwargs):
    alternatives = local_map_kwargs.get("alternatives")
    if alternatives is not None:
        return alternatives

    for key in ("out_placements", "in_placements"):
        placements = local_map_kwargs.get(key)
        alternatives = getattr(placements, _FLEX_LOCAL_MAP_ALTERNATIVES_ATTR, None)
        if alternatives is not None:
            return alternatives

    return None


def _normalize_flex_local_map_alternatives(
    default_fn: Callable,
    alternatives: Sequence[dict[str, Any]],
):
    if not alternatives:
        raise ValueError("flex_local_map requires at least one alternative")

    normalized = []
    for idx, alternative in enumerate(alternatives):
        if not isinstance(alternative, dict):
            raise TypeError(
                f"flex_local_map alternative {idx} must be a dict, got "
                f"{type(alternative).__name__}"
            )
        for key in ("in_placements", "out_placements"):
            if key not in alternative:
                raise ValueError(f"flex_local_map alternative {idx} requires {key}")
            if alternative[key] is None:
                raise ValueError(
                    f"flex_local_map alternative {idx} {key} must not be None"
                )

        fn = alternative.get("fn", default_fn)
        if not callable(fn):
            raise TypeError(f"flex_local_map alternative {idx} fn must be callable")
        try:
            cost_hint = float(alternative.get("cost_hint", 0.0))
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"flex_local_map alternative {idx} cost_hint must be a number"
            ) from error
        if not math.isfinite(cost_hint) or cost_hint < 0:
            raise ValueError(
                f"flex_local_map alternative {idx} cost_hint must be finite and "
                "non-negative"
            )
        normalized.append(
            {
                **alternative,
                "fn": fn,
                "cost_hint": cost_hint,
                "name": alternative.get("name", getattr(fn, "__name__", f"alt_{idx}")),
            }
        )

    input_arity = len(normalized[0]["in_placements"])
    output_arity = len(normalized[0]["out_placements"])
    for idx, alternative in enumerate(normalized[1:], start=1):
        if len(alternative["in_placements"]) != input_arity:
            raise ValueError(
                f"flex_local_map alternative {idx} has a different input arity"
            )
        if len(alternative["out_placements"]) != output_arity:
            raise ValueError(
                f"flex_local_map alternative {idx} has a different output arity"
            )

    return tuple(normalized)


def flex_local_map(
    func: Callable | None = None,
    *,
    alternatives: Sequence[dict[str, Any]],
    device_mesh: DeviceMesh,
    in_grad_placements=None,
    redistribute_inputs: bool = False,
):
    """Like ``local_map``, but declares several placement alternatives for one region so
    the AutoParallel solver can choose among them (e.g. MoE DP->EP vs DP+TP->EP+ETP).

    Each entry of ``alternatives`` is a dict with ``in_placements`` and ``out_placements``
    (required) and optional ``fn`` (defaults to ``func``), ``name``, and ``cost_hint`` (a
    non-negative solver cost hint, default 0.0). ``device_mesh`` must be explicit.
    ``in_grad_placements`` is reserved for parity with ``local_map`` but is not yet
    supported by AutoParallel.

    Apply it outside ``forward`` (e.g. in ``__init__``) with an explicit ``device_mesh``,
    and call the returned callable inside ``forward``.
    """
    if in_grad_placements is not None:
        raise NotImplementedError(
            "flex_local_map does not yet support in_grad_placements"
        )

    if func is None:
        return functools.partial(
            flex_local_map,
            alternatives=alternatives,
            device_mesh=device_mesh,
            in_grad_placements=in_grad_placements,
            redistribute_inputs=redistribute_inputs,
        )

    normalized_alternatives = _normalize_flex_local_map_alternatives(func, alternatives)
    default_alternative = normalized_alternatives[0]
    out_placements = _FlexLocalMapOutPlacements(
        tuple(default_alternative["out_placements"]),
        normalized_alternatives,
    )

    return local_map(
        default_alternative["fn"],
        out_placements=out_placements,
        in_placements=default_alternative["in_placements"],
        in_grad_placements=in_grad_placements,
        device_mesh=device_mesh,
        redistribute_inputs=redistribute_inputs,
    )


def with_sharding_constraint(
    x: torch.Tensor,
    shardings: Tuple[Placement, ...],
    device_mesh: Optional[DeviceMesh] = None,
) -> torch.Tensor:
    """Constrain the sharding of an intermediate tensor.

    Similar to JAX's with_sharding_constraint, this constrains the sharding
    of a tensor to a specific placement. This is useful for controlling
    intermediate tensor shardings within a computation.

    Args:
        x: The tensor to constrain.
        shardings: Tuple of placements specifying how the tensor should be
            sharded across each mesh dimension.
        device_mesh: The device mesh to use. If None, uses the mesh from
            the enclosing local_map region.

    Returns:
        The tensor with the specified sharding constraint applied.

    Example:
        >>> from torch.distributed.tensor.placement_types import Shard, Replicate
        >>> # Inside a local_map region or with explicit mesh:
        >>> x = with_sharding_constraint(x, (Shard(0), Replicate()))
    """
    if device_mesh is None:
        device_mesh = get_mesh_from_global()

    @_local_map(
        out_placements=(shardings,),
        in_placements=(shardings,),
        redistribute_inputs=True,
        device_mesh=device_mesh,
    )
    def identity(t):
        # clone() is required because local_map HOP doesn't support
        # input-to-output aliasing during dynamo tracing
        return t.clone()

    return identity(x)


def local_map(*args, **kwargs):
    # TODO: upstream this fallback into PyTorch's local_map, matching
    # DTensor.from_local and distribute_tensor which already do this.
    if kwargs.get("device_mesh", None) is None:
        kwargs["device_mesh"] = _mesh_resources.get_current_mesh()
    return _local_map(*args, **kwargs)


def get_mesh_from_global():
    return _mesh_resources.get_current_mesh()


def _get_group_name_from_axis_name(mesh_name):
    mesh = get_mesh_from_global()
    group = mesh.get_group(mesh_name)
    return group.group_name


def axis_size(axis_name):
    mesh = get_mesh_from_global()
    assert axis_name in mesh.mesh_dim_names
    axis_dim = mesh.mesh_dim_names.index(axis_name)
    return mesh.size(axis_dim)


def axis_index(axis_name):
    mesh = get_mesh_from_global()
    return mesh.get_local_rank(mesh_dim=axis_name)


def _all_gather_tensor(
    x: torch.Tensor,
    gather_dim: int,
    group_name: GroupName,
) -> torch.Tensor:
    x = x.contiguous()
    group_size = c10d._get_group_size_by_name(group_name)
    tensor = torch.ops._c10d_functional.all_gather_into_tensor(
        x, group_size, group_name
    )
    res = torch.ops._c10d_functional.wait_tensor(tensor)
    if gather_dim != 0:
        # torch.cat access the data so we already need to wait here, first do wait
        # and then chunk + cat avoid us going through ACT dispatching logic again
        res = torch.cat(torch.chunk(res, group_size, dim=0), dim=gather_dim)
    return res


def _reduce_scatter_tensor(
    self: torch.Tensor, reduceOp: str, scatter_dim: int, group_name: GroupName
):
    group_size = c10d._get_group_size_by_name(group_name)

    assert (
        self.size(scatter_dim) % group_size == 0
    ), f"input dimension 0 ({self.size(0)} must be a multiple of group_size {group_size})"
    if scatter_dim != 0:
        tensor_list = torch.chunk(self, group_size, dim=scatter_dim)
        self = torch.cat(tensor_list)

    tensor = torch.ops._c10d_functional.reduce_scatter_tensor(
        self,
        reduceOp.lower(),
        group_size,
        group_name,
    )
    res = torch.ops._c10d_functional.wait_tensor(tensor)
    return res


def _all_reduce(self: torch.Tensor, reduceOp: str, group_name: GroupName):
    tensor = torch.ops._c10d_functional.all_reduce(self, reduceOp.lower(), group_name)
    res = torch.ops._c10d_functional.wait_tensor(tensor)
    return res


def _all_to_all(
    self: torch.Tensor,
    output_split_sizes: Optional[list[int]],
    input_split_sizes: Optional[list[int]],
    group_name: GroupName,
):
    group_size = c10d._get_group_size_by_name(group_name)
    if output_split_sizes is None or input_split_sizes is None:
        assert output_split_sizes is None and input_split_sizes is None, (
            "output_split_sizes and input_split_sizes must either be "
            "specified together or both set to None"
        )
        output_split_sizes = [self.shape[0] // group_size] * group_size
        input_split_sizes = output_split_sizes

    tensor = torch.ops._c10d_functional.all_to_all_single(
        self, output_split_sizes, input_split_sizes, group_name
    )
    res = torch.ops._c10d_functional.wait_tensor(tensor)
    return res


class _AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, gather_dim: int, axis_name: str):
        group_name = _get_group_name_from_axis_name(axis_name)
        ctx.group_name = group_name
        ctx.gather_dim = gather_dim
        return _all_gather_tensor(x, gather_dim, group_name)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):  # type: ignore[override]
        return (
            _reduce_scatter_tensor(grad_output, "sum", ctx.gather_dim, ctx.group_name),
            None,
            None,
        )


class _ReduceScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, scatter_dim: int, axis_name: str):
        group_name = _get_group_name_from_axis_name(axis_name)
        ctx.group_name = group_name
        ctx.scatter_dim = scatter_dim
        return _reduce_scatter_tensor(x, "sum", scatter_dim, group_name)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):  # type: ignore[override]
        return (
            _all_gather_tensor(grad_output, ctx.scatter_dim, ctx.group_name),
            None,
            None,
        )


class _AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, axis_name: str):
        group_name = _get_group_name_from_axis_name(axis_name)
        ctx.group_name = group_name
        return _all_reduce(x, "sum", group_name)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):  # type: ignore[override]
        # TODO: split this into a function that does all-reduce and one which is the identity
        return _all_reduce(grad_output, "sum", ctx.group_name), None


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        output_split_sizes: Optional[list[int]],
        input_split_sizes: Optional[list[int]],
        axis_name: str,
    ):
        group_name = _get_group_name_from_axis_name(axis_name)
        ctx.group_name = group_name
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        return _all_to_all(x, output_split_sizes, input_split_sizes, group_name)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):  # type: ignore[override]
        return _all_to_all(
            grad_output, ctx.input_split_sizes, ctx.output_split_sizes, ctx.group_name
        )


all_gather = _AllGather.apply
all_reduce = _AllReduce.apply
reduce_scatter = _ReduceScatter.apply
all_to_all = _AllToAll.apply
