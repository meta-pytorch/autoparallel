from typing import cast, Optional

import torch
import torch.distributed as dist
from autoparallel.propagation_rules import register_opschema_rule

from torch.distributed._functional_collectives import all_to_all_single, RANK_TYPES
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._op_schema import OpSchema, OpStrategy
from torch.distributed.tensor._ops._matrix_ops import _mm_like_strategy
from torch.distributed.tensor._ops.utils import register_op_strategy
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard
from torch.utils.flop_counter import register_flop_formula

from .moe_placements import PartitionedShard
from .moe_utils import (
    batched_generate_permute_indices,
    batched_permute_and_pad,
    batched_unpermute_and_unpad,
    TOKEN_GROUP_ALIGN_SIZE_M,
)


@torch.library.custom_op("autoparallel::batched_mm", mutates_args=())
def batched_mm(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
) -> torch.Tensor:
    assert mat1.ndim == 3
    assert mat2.ndim == 2 or mat2.ndim == 3
    if mat2.ndim == 2:
        assert mat1.shape[2] == mat2.shape[0]
        mat2_expanded = mat2.expand(mat1.shape[0], -1, -1)
    else:
        assert mat1.shape[0] == mat2.shape[0]
        assert mat1.shape[2] == mat2.shape[1]
        mat2_expanded = mat2
    out = torch.bmm(mat1, mat2_expanded)
    return out


def setup_context_batched_mm(ctx, inputs, output):
    mat1, mat2 = inputs
    ctx.save_for_backward(mat1, mat2)


def backward_batched_mm(ctx, grad):
    assert grad.ndim == 3
    mat1, mat2 = ctx.saved_tensors
    grad1 = batched_mm(grad, mat2.transpose(-2, -1))
    grad2 = batched_mm(mat1.transpose(-2, -1), grad)
    if mat2.ndim == 2:
        grad2 = torch.sum(grad2, dim=0)
    return grad1, grad2


torch.library.register_autograd(
    "autoparallel::batched_mm",
    backward_batched_mm,
    setup_context=setup_context_batched_mm,
)


@batched_mm.register_fake
def batched_mm_meta(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
) -> torch.Tensor:
    assert mat1.ndim == 3
    assert mat2.ndim == 3 or mat2.ndim == 2

    if mat2.ndim == 2:
        assert mat1.shape[2] == mat2.shape[0]
    else:
        assert mat1.shape[2] == mat2.shape[1]
    out = torch.empty(
        mat1.shape[0],
        mat1.shape[1],
        mat2.shape[-1],
        dtype=mat1.dtype,
        device=mat1.device,
    )
    return out


@register_flop_formula(torch.ops.autoparallel.batched_mm)
def batched_mm_flop_count(mat1_shape, mat2_shape, *args, out_shape=None, **kwargs):
    """
    Count floating-point operations for batched matrix multiplication.

    This operation performs matrix multiplication between mat1 and mat2, where mat1 is
    a batched tensor and mat2 is a regular matrix. The output is also a batched tensor.

    Args:
        mat1_shape: Shape of first input matrix
        mat2_shape: Shape of second input matrix
    Returns:
        Total number of floating-point operations
    """
    assert len(mat1_shape) == 3
    assert len(mat2_shape) == 2 or len(mat2_shape) == 3
    # Parse mat1 dimensions
    b, m, n = mat1_shape
    # Parse mat2 dimensions
    n2, k = mat2_shape[-2], mat2_shape[-1]
    assert n == n2, f"Dimension mismatch: {n} vs {n2}"
    # Calculate FLOPs for matrix multiplication: C = A @ B
    return b * m * n * k * 2


@register_op_strategy(torch.ops.autoparallel.batched_mm.default)
def batched_matmul_rule(op_schema: OpSchema):
    mesh = op_schema.get_mesh_from_args()
    mat2_strategy = cast(OpStrategy, op_schema.args_schema[1])
    if len(mat2_strategy.shape) == 2:
        mm_equation = "bmk,kn->bmn"
    else:
        assert len(mat2_strategy.shape) == 3, "mat2 must be 2D or 3D"
        mm_equation = "bmk,bkn->bmn"
    # dispatch to mm_like_strategy
    return _mm_like_strategy(mm_equation, mesh, op_schema)


def _validate_batched_args(b_args: list[int]) -> None:
    assert b_args in [
        [
            0,
        ],
        [
            1,
        ],
        [0, 1],
    ], f"Invalid batched_args: {b_args}"


def _transform_grouped_mm_args(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    batched_args: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if batched_args == [
        0,
    ]:
        mat1_exp = mat1
        mat2_exp = mat2.expand(mat1.shape[0], -1, -1, -1)
    elif batched_args == [
        1,
    ]:
        mat1_exp = mat1.expand(mat2.shape[0], -1, -1, -1)
        mat2_exp = mat2
    else:
        mat1_exp = mat1
        mat2_exp = mat2
    return mat1_exp, mat2_exp


def _validate_batched_grouped_mm_shapes(
    mat1_shape: torch.Size,
    mat2_shape: torch.Size,
    offs_shape: torch.Size,
) -> None:
    # Case 1: mat1 and mat2 are both 3D batched tensors
    # grouped_mm 2Dx2D case
    # Case 2: mat1 is 3D batched tensor and mat2 is 4D batched tensor
    # grouped_mm 2Dx3D case
    # Case 3: mat1 is 4D batched tensor and mat2 is 3D batched tensor
    # grouped_mm 3Dx2D case
    valid_shapes = (
        (len(mat1_shape) == 3 and len(mat2_shape) == 3)
        or (len(mat1_shape) == 4 and len(mat2_shape) == 3)
        or (len(mat1_shape) == 3 and len(mat2_shape) == 4)
    )
    assert valid_shapes, (
        f"Invalid tensor dimensions: expected mat1 and mat2 to be either both 3D, "
        f"or mat1 4D and mat2 3D, or mat1 3D and mat2 4D. "
        f"Got mat1={mat1_shape} and mat2={mat2_shape}."
    )
    assert len(offs_shape) == 2, f"offs needs to be 2D, got {offs_shape}"
    # OB = batch size
    # IB = sum(num_tokens_per_expert)
    # D = embedding dim
    # H = hidden dim
    # E = number of experts
    if len(mat1_shape) == 3 and len(mat2_shape) == 3:
        OB1, _, IB1 = mat1_shape
        OB2, IB2, _ = mat2_shape
        assert OB1 == OB2, f"Batch size mismatch: {OB1} vs {OB2}"
        assert IB1 == IB2, f"Total tokens mismatch: {IB1} vs {IB2}"
    elif len(mat1_shape) == 4 and len(mat2_shape) == 3:
        OB1, E1, _, H1 = mat1_shape
        OB2, H2, _ = mat2_shape
        OB3, E2 = offs_shape
        assert (
            OB1 == OB2 and OB2 == OB3
        ), f"Batch size mismatch: {OB1} vs {OB2} vs {OB3}"
        assert H1 == H2, f"Contracting dimension mismatch: {H1} vs {H2}"
        assert E1 == E2, f"Number of experts mismatch: {E1} vs {E2}"
    else:
        OB1, IB1, D1 = mat1_shape
        OB2, E1, D2, _ = mat2_shape
        OB3, E2 = offs_shape
        assert (
            OB1 == OB2 and OB2 == OB3
        ), f"Batch size mismatch: {OB1} vs {OB2} vs {OB3}"
        assert D1 == D2, f"Contracting dimension mismatch: {D1} vs {D2}"
        assert E1 == E2, f"Number of experts mismatch: {E1} vs {E2}"


@torch.library.custom_op("autoparallel::batched_grouped_mm", mutates_args=())
def batched_grouped_mm(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    offs: torch.Tensor,
    batched_args: list[int],
) -> torch.Tensor:
    _validate_batched_args(batched_args)
    mat1_exp, mat2_exp = _transform_grouped_mm_args(mat1, mat2, batched_args)
    _validate_batched_grouped_mm_shapes(mat1_exp.shape, mat2_exp.shape, offs.shape)
    res = []
    for m1, m2, off in zip(mat1_exp, mat2_exp, offs):
        res.append(torch._grouped_mm(m1, m2, off))
    return torch.stack(res, 0)


def setup_context_batched_grouped_mm(ctx, inputs, output):
    mat1, mat2, offs, batched_args = inputs
    ctx.save_for_backward(mat1, mat2, offs)
    ctx.batched_args = batched_args


def _backward_batched_grouped_mm_mat1(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    grad: torch.Tensor,
    offs: torch.Tensor,
    batched_args: list[int],
) -> torch.Tensor:
    if mat1.stride(-2) == 1 and mat1.stride(-1) == mat1.size(-2):
        if batched_args == [
            0,
        ]:
            b_args = [
                1,
            ]
        else:
            b_args = [0, 1]
        # if input was column-major, return grad_input as column-order for efficiency
        grad_mat1 = batched_grouped_mm(
            mat2, grad.transpose(-2, -1), offs, b_args
        ).transpose(-2, -1)
    else:
        if batched_args == [
            0,
        ]:
            b_args = [
                0,
            ]
        else:
            b_args = [0, 1]
        grad_mat1 = batched_grouped_mm(grad, mat2.transpose(-2, -1), offs, b_args)
    return grad_mat1


def _backward_batched_grouped_mm_mat2(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    grad: torch.Tensor,
    offs: torch.Tensor,
    batched_args: list[int],
) -> torch.Tensor:
    if mat2.stride(-2) == 1 and mat2.stride(-1) == mat2.size(-2):
        if batched_args == [
            1,
        ]:
            b_args = [
                0,
            ]
        else:
            b_args = [0, 1]
        # if experts were column-major, return experts_grad as column-order for efficiency
        grad_mat2 = batched_grouped_mm(
            grad.transpose(-2, -1),
            mat1,
            offs,
            b_args,
        ).transpose(-2, -1)
    else:
        if batched_args == [
            1,
        ]:
            b_args = [
                1,
            ]
        else:
            b_args = [0, 1]
        grad_mat2 = batched_grouped_mm(
            mat1.transpose(-2, -1),
            grad,
            offs,
            b_args,
        )
    return grad_mat2


def backward_batched_grouped_mm(
    ctx, grad: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, None, None]:
    mat1, mat2, offs = ctx.saved_tensors
    batched_args = ctx.batched_args
    grad_mat1 = _backward_batched_grouped_mm_mat1(mat1, mat2, grad, offs, batched_args)
    grad_mat2 = _backward_batched_grouped_mm_mat2(mat1, mat2, grad, offs, batched_args)
    if batched_args == [
        0,
    ]:
        grad_mat2 = torch.sum(grad_mat2, dim=0)
    elif batched_args == [
        1,
    ]:
        grad_mat1 = torch.sum(grad_mat1, dim=0)

    return grad_mat1, grad_mat2, None, None


torch.library.register_autograd(
    "autoparallel::batched_grouped_mm",
    backward_batched_grouped_mm,
    setup_context=setup_context_batched_grouped_mm,
)


@batched_grouped_mm.register_fake
def batched_grouped_mm_meta(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    offs: torch.Tensor,
    batched_args: list[int],
) -> torch.Tensor:
    _validate_batched_args(batched_args)
    mat1, mat2 = _transform_grouped_mm_args(mat1, mat2, batched_args)
    _validate_batched_grouped_mm_shapes(mat1.shape, mat2.shape, offs.shape)
    if mat1.ndim == 3 and mat2.ndim == 3:
        # out_shape = (OB, E, H, D)
        out = torch.empty(
            mat1.shape[0],
            offs.shape[1],
            mat1.shape[1],
            mat2.shape[2],
            dtype=mat1.dtype,
            device=mat1.device,
        )
    elif mat1.ndim == 4 and mat2.ndim == 3:
        # out_shape = (OB, D, IB)
        out = torch.empty(
            mat1.shape[0],
            mat1.shape[2],
            mat2.shape[2],
            dtype=mat1.dtype,
            device=mat1.device,
        )
    else:
        # out_shape = (OB, IB, H)
        out = torch.empty(
            mat1.shape[0],
            mat1.shape[1],
            mat2.shape[3],
            dtype=mat1.dtype,
            device=mat1.device,
        )
    return out


@register_flop_formula(torch.ops.autoparallel.batched_grouped_mm)
def batched_grouped_mm_flop_count(
    mat1_shape: torch.Size,
    mat2_shape: torch.Size,
    offsets_shape: torch.Size,
    batched_args: list[int],
    bias_shape=None,
    out_shape=None,
    **kwargs,
) -> int:
    """
    Count floating-point operations for batched grouped matrix multiplication.

    This operation performs matrix multiplication between mat1 and mat2, where tokens
    are grouped by experts (common in MoE models). The offsets tensor defines how
    tokens are distributed across expert groups.

    Args:
        mat1_shape: Shape of first input matrix
        mat2_shape: Shape of second input matrix
        offsets_shape: Shape of offsets tensor

    Returns:
        Total number of floating-point operations

    """
    _validate_batched_args(batched_args)
    if batched_args == [
        0,
    ]:
        mat2_shape = torch.Size([mat1_shape[0], *mat2_shape])
    elif batched_args == [
        1,
    ]:
        mat1_shape = torch.Size([mat2_shape[0], *mat1_shape])
    _validate_batched_grouped_mm_shapes(mat1_shape, mat2_shape, offsets_shape)

    if len(mat1_shape) == 3 and len(mat2_shape) == 3:
        OB, D, IB = mat1_shape
        _, _, H = mat2_shape
    elif len(mat1_shape) == 4 and len(mat2_shape) == 3:
        OB, _, D, H = mat1_shape
        _, _, IB = mat2_shape
    else:
        OB, IB, D = mat1_shape
        _, _, _, H = mat2_shape
    # OB = batch size
    # IB = sum(num_tokens_per_expert)
    # D = embedding dim
    # H = hidden dim
    total_flops = OB * IB * D * H * 2
    return total_flops


@register_opschema_rule(torch.ops.autoparallel.batched_grouped_mm.default)
def batched_grouped_mm_strategy(mesh: DeviceMesh, op_schema: OpSchema):
    from torch.distributed.tensor._op_schema import PlacementList
    from torch.distributed.tensor._ops.utils import expand_to_full_mesh_op_strategy

    mat1_strategy = cast(OpStrategy, op_schema.args_schema[0])
    mat2_strategy = cast(OpStrategy, op_schema.args_schema[1])
    offs_strategy = cast(OpStrategy, op_schema.args_schema[2])
    batched_args = cast(list[int], op_schema.args_schema[3])
    _validate_batched_args(batched_args)
    assert offs_strategy.ndim == 2

    single_mesh_dim_strategies = []

    # placement list stores placements of [output, mat1, mat2, offs]
    # first we always have replicate all for inputs and output
    all_replicate: PlacementList = [Replicate()] * 4
    # TODO (@sanketpurandare): add sharding rules when one of the inputs will be partial (maybe not needed)
    single_mesh_dim_strategies.append(all_replicate)
    if batched_args == [
        0,
    ]:
        # Case: 2Dx3D
        # mat1 is [OB, IB, D]
        # mat2 is [E, D, H]
        # offs is [OB, E]
        # output is [OB, IB, H]
        assert mat1_strategy.ndim == 3 and mat2_strategy.ndim == 3
        strategies_2x3 = []
        # dp mesh
        # mat1 is sharded on outer batch, mat2 is replicated, offs is sharded on outer batch
        #  -> output is sharded on outer batch
        strategies_2x3.append([Shard(0), Shard(0), Replicate(), Shard(0)])
        # ep mesh
        # mat1 is sharded on inner batch, mat2 is replicated, offs is partial
        #  -> output is sharded on inner batch
        strategies_2x3.append([Shard(1), Shard(1), Replicate(), Partial()])
        # mat1 is dynamically sharded on row dim, mat2 is sharded on experts dim, offs is sharded on experts dim
        #  -> output is dynamically sharded on row dim
        strategies_2x3.append(
            [PartitionedShard(1), PartitionedShard(1), Shard(0), Shard(1)]
        )
        # tp mesh
        # mat2 is sharded on column dim, mat1 is replicated, offs is replicated
        #  -> output is sharded on column dim
        strategies_2x3.append([Shard(2), Replicate(), Shard(2), Replicate()])
        # mat1 is sharded on column dim, mat2 is sharded on row dim, offs is replicated
        #  -> output is partial
        strategies_2x3.append([Partial(), Shard(2), Shard(1), Replicate()])
        single_mesh_dim_strategies.extend(strategies_2x3)
    elif batched_args == [
        1,
    ]:
        # Case: 3Dx2D
        # mat1 is [E, H, D]
        # mat2 is [OB, D, IB]
        # offs is [OB, E]
        # output is [OB, H, IB]
        assert mat1_strategy.ndim == 3 and mat2_strategy.ndim == 3
        strategies_3x2 = []
        # dp mesh
        # mat1 is replicated, mat2 is sharded on outer batch, offs is sharded on outer batch
        # -> output is sharded on outer batch
        strategies_3x2.append([Shard(0), Replicate(), Shard(0), Shard(0)])
        # ep mesh
        # mat1 is replicated, mat2 is sharded on inner batch, offs is partial
        # -> output is sharded on inner batch
        strategies_3x2.append([Shard(2), Replicate(), Shard(2), Partial()])
        # mat1 is sharded on experts dim, mat2 is dynamically sharded on column dim, offs is sharded on experts dim
        # -> output is dynamically sharded on column dim
        strategies_3x2.append(
            [PartitionedShard(2), Shard(0), PartitionedShard(2), Shard(1)]
        )
        # tp mesh
        # mat1 is sharded on column dim, mat2 is sharded on row dim, offs is replicated
        # -> output is partial
        strategies_3x2.append([Partial(), Shard(2), Shard(2), Replicate()])
        # mat1 is sharded on row dim, mat2 is replicated, offs is replicated
        # output is sharded on row dim
        strategies_3x2.append([Shard(1), Shard(1), Replicate(), Replicate()])
        single_mesh_dim_strategies.extend(strategies_3x2)
    else:
        assert batched_args == [0, 1]
        if mat1_strategy.ndim == 3 and mat2_strategy.ndim == 3:
            # Case: 2Dx2D
            # mat1 is [OB, D, IB]
            # mat2 is [OB, IB, H]
            # offs is [OB, E]
            # output is [OB, E, D, H]
            strategies_2x2 = []
            # dp mesh
            # mat1 is sharded on outer batch, mat2 is sharded on outer batch, offs is sharded on outer batch
            # -> output is sharded on outer batch
            strategies_2x2.append([Shard(0), Shard(0), Shard(0), Shard(0)])
            # ep mesh
            # mat1 is sharded on inner batch, mat2 is sharded on inner batch, offs is partial
            # -> output is partial
            strategies_2x2.append([Partial(), Shard(2), Shard(1), Partial()])
            # mat1 is dynamically sharded on column dim, mat2 is dynamically sharded on row dim, offs is sharded on experts dim
            # -> output is sharded on experts dim
            strategies_2x2.append(
                [Shard(1), PartitionedShard(2), PartitionedShard(1), Shard(0)]
            )
            # tp mesh
            # mat1 is replicated, mat2 is sharded on column dim, offs is replicated
            # -> output is sharded on column dim
            strategies_2x2.append([Shard(3), Replicate(), Shard(2), Replicate()])
            # mat1 is sharded on row dim, mat2 is replicated, offs is replicated
            # -> output is sharded on row dim
            strategies_2x2.append([Shard(2), Shard(1), Replicate(), Replicate()])
            single_mesh_dim_strategies.extend(strategies_2x2)
        elif mat1_strategy.ndim == 4 and mat2_strategy.ndim == 3:
            # Case: 3Dx2D
            # mat1 is [OB, E, D, H]
            # mat2 is [OB, H, IB]
            # offs is [OB, E]
            # output is [OB, D, IB]
            strategies_batched_3x2 = []
            # dp mesh
            # mat1 is sharded on outer batch, mat2 is sharded on outer batch, offs is sharded on outer batch
            # -> output is sharded on outer batch
            strategies_batched_3x2.append([Shard(0), Shard(0), Shard(0), Shard(0)])
            # ep mesh
            # mat1 is replicated, mat2 is sharded on inner batch, offs is partial
            # -> output is sharded on inner batch
            strategies_batched_3x2.append([Shard(2), Replicate(), Shard(2), Partial()])
            # mat1 is sharded on experts dim, mat2 is dynamically sharded on column dim, offs is sharded on experts dim
            # -> output is dynamically sharded on column dim
            strategies_batched_3x2.append(
                [PartitionedShard(2), Shard(1), PartitionedShard(2), Shard(1)]
            )
            # tp mesh
            # mat1 is sharded on column dim, mat2 is sharded on row dim, offs is replicated
            # -> output is partial
            strategies_batched_3x2.append([Partial(), Shard(3), Shard(1), Replicate()])
            # mat1 is sharded on row dim, mat2 is replicated, offs is replicated
            # output is sharded on row dim
            strategies_batched_3x2.append(
                [Shard(1), Shard(2), Replicate(), Replicate()]
            )
            single_mesh_dim_strategies.extend(strategies_batched_3x2)

        elif mat1_strategy.ndim == 3 and mat2_strategy.ndim == 4:
            # Case: 2Dx3D
            # mat1 is [OB, IB, D]
            # mat2 is [OB, E, D, H]
            # offs is [OB, E]
            # output is [OB, IB, H]
            strategies_batched_2x3 = []
            # dp mesh
            # mat1 is sharded on outer batch, mat2 is sharded on outer batch, offs is sharded on outer batch
            #  -> output is sharded on outer batch
            strategies_batched_2x3.append([Shard(0), Shard(0), Shard(0), Shard(0)])
            # ep mesh
            # mat1 is sharded on inner batch, mat2 is replicated, offs is partial
            #  -> output is sharded on inner batch
            strategies_batched_2x3.append([Shard(1), Shard(1), Replicate(), Partial()])
            # mat1 is dynamically sharded on row dim, mat2 is sharded on experts dim, offs is sharded on experts dim
            #  -> output is dynamically sharded on row dim
            strategies_batched_2x3.append(
                [PartitionedShard(1), PartitionedShard(1), Shard(1), Shard(1)]
            )
            # tp mesh
            # mat1 is replicated, mat2 is sharded on column dim , offs is replicated
            #  -> output is sharded on column dim
            strategies_batched_2x3.append(
                [Shard(2), Replicate(), Shard(3), Replicate()]
            )
            # mat1 is sharded on column dim, mat2 is sharded on row dim, offs is replicated
            #  -> output is partial
            strategies_batched_2x3.append([Partial(), Shard(2), Shard(2), Replicate()])
            single_mesh_dim_strategies.extend(strategies_batched_2x3)
        else:
            raise RuntimeError("Invalid batched grouped mm strategy")

    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=1
    )


@torch.library.custom_op("autoparallel::batched_histc", mutates_args=())
def batched_histc(
    x: torch.Tensor, bins: int = 100, min_val: int = 0, max_val: int = 1
) -> torch.Tensor:
    assert x.ndim == 2
    out = []
    for t in x:
        out.append(torch.histc(t, bins, min_val, max_val))
    return torch.stack(out, 0)


@batched_histc.register_fake
def batched_histc_meta(
    x: torch.Tensor, bins: int = 100, min_val: int = 0, max_val: int = 1
) -> torch.Tensor:
    out = torch.empty((x.shape[0], bins), dtype=x.dtype, device=x.device)
    return out


@register_opschema_rule(torch.ops.autoparallel.batched_histc.default)
def batched_histc_strategy(mesh: DeviceMesh, op_schema: OpSchema):
    from torch.distributed.tensor._op_schema import PlacementList
    from torch.distributed.tensor._ops.utils import expand_to_full_mesh_op_strategy

    x_strategy = cast(OpStrategy, op_schema.args_schema[0])

    assert len(x_strategy.shape) == 2

    single_mesh_dim_strategies = []

    # placement list stores placements of [output, x]
    # first we always have replicate all for inputs and output
    all_replicate: PlacementList = [Replicate()] * 2
    single_mesh_dim_strategies.append(all_replicate)
    # x sharded on outer batch -> output is sharded on outer batch
    single_mesh_dim_strategies.append([Shard(0), Shard(0)])
    # x is sharded in inner batch -> output is partial
    single_mesh_dim_strategies.append([Partial(), Shard(1)])

    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=1
    )


def batched_all_to_all_single(
    batched_input: torch.Tensor,
    batched_out_splits: Optional[list[list[int]]] = None,
    batched_in_splits: Optional[list[list[int]]] = None,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    assert group is not None
    assert batched_input.ndim >= 2
    batched_out = []
    if batched_in_splits is not None:
        assert batched_out_splits is not None
        assert len(batched_in_splits) == batched_input.shape[0]
        assert len(batched_out_splits) == batched_input.shape[0]
        for input_t, in_splits, out_splits in zip(
            batched_input, batched_in_splits, batched_out_splits
        ):
            # TODO: We should coalesce all to alls to reduce overhead
            out_t = all_to_all_single(
                input_t.contiguous(), out_splits, in_splits, group=group
            )
            batched_out.append(out_t)
    else:
        assert batched_out_splits is None
        for input_t in batched_input:
            out_t = all_to_all_single(input_t.contiguous(), None, None, group=group)
            batched_out.append(out_t)
    return torch.stack(batched_out, dim=0)


@torch.library.custom_op("autoparallel::token_dispatch", mutates_args=())
def token_dispatch_op(
    x: torch.Tensor,
    top_scores: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    num_experts: int,
    top_k: int,
    score_before_experts: bool = False,
    ep_mesh: DeviceMesh | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[torch.Size],
    list[list[int]] | None,
    list[list[int]] | None,
]:  # (padded_permuted_routed_input, top_scores_sorted, token_indices_sorted, batch_permuted_indices, routed_input_shape_before_permute)
    """
    Token dispatch operator with custom backward.

    Forward: Dispatch tokens to experts based on their expert assignments.
    Backward: Use token combine logic to scatter gradients back to original positions.

    Args:
        x: Input tensor of shape (ob, ib * slen, dim)
        top_scores: Tensor of shape (ob, ib * slen, top_k)
        selected_experts_indices: Tensor of shape (ob, ib * slen, top_k)
        num_tokens_per_expert: Tensor of shape (ob, num_experts)
        num_experts: Number of experts
        top_k: Number of experts to route to
        score_before_experts: Whether to apply scores before expert processing
        ep_mesh: DeviceMesh for expert parallel token exchange

    Returns:
        padded_permuted_routed_input:
        (ob, ib * slen * top_k + (num_experts * TOKEN_GROUP_ALIGN_SIZE_M), dim) - padded and permuted routed input
        tokens_per_expert: (ob, num_experts) - number of tokens per expert
        top_scores_sorted: (ob, ib * slen * top_k) - sorted scores
        token_indices_sorted: (ob, ib * slen * top_k) - sorted token indices
        permuted_indices: (ob, ib * slen * top_k + (num_experts * TOKEN_GROUP_ALIGN_SIZE_M)) - permuted indices
        routed_input_shape_before_permute: list of shape before permute
        input_splits: list of input splits for all to all
        output_splits: list of output splits for all to all
    """
    assert x.ndim == 3
    ob, ib_slen, dim = x.shape

    # Flatten expert indices and scores
    expert_indices_flat = selected_experts_indices.reshape(
        ob, ib_slen * top_k
    )  # (ob, ib_slen * top_k)
    scores_flat = top_scores.reshape(ob, ib_slen * top_k)  # (ob, ib_slen * top_k)

    # Sort within each batch
    sort_indices = torch.argsort(
        expert_indices_flat, dim=1, stable=True
    )  # (ob, ib_slen * top_k)

    # Gather sorted scores
    batch_indices = torch.arange(ob, device=x.device).unsqueeze(1)  # (ob, 1)
    scores_sorted = scores_flat[batch_indices, sort_indices]  # (ob, ib_slen * top_k)

    # Convert to token indices
    token_indices = sort_indices // top_k  # (ob, ib_slen * top_k)

    # Expand for gather
    token_indices_expanded = token_indices.unsqueeze(-1).expand(
        -1, -1, dim
    )  # (ob, ib_slen * top_k, dim)

    # Gather tokens
    routed_input = torch.gather(
        x, dim=1, index=token_indices_expanded
    )  # (ob, ib_slen * top_k, dim)

    if score_before_experts:
        routed_input = (
            routed_input.to(torch.float32) * scores_sorted.unsqueeze(-1)
        ).to(
            x.dtype
        )  # (ob, ib_slen * top_k, dim)

    if ep_mesh is not None:
        # Need to perform an all to all to exchange tokens between ranks
        num_ep_ranks = ep_mesh.shape[0]
        exps_per_rank = num_experts // num_ep_ranks
        assert exps_per_rank * num_ep_ranks == num_experts
        with torch.no_grad():
            num_tokens_per_expert_group = batched_all_to_all_single(
                batched_input=num_tokens_per_expert, group=ep_mesh.get_group()
            )
            assert num_tokens_per_expert_group.shape == (
                ob,
                num_ep_ranks * exps_per_rank,
            )
            input_splits = torch.sum(
                num_tokens_per_expert.reshape(ob, num_ep_ranks, exps_per_rank),
                dim=2,
            ).to(torch.device("cpu"), non_blocking=True)
            assert input_splits.shape == (ob, num_ep_ranks)

            output_splits = torch.sum(
                num_tokens_per_expert_group.reshape(ob, num_ep_ranks, exps_per_rank),
                dim=2,
            ).to(torch.device("cpu"), non_blocking=True)
            assert output_splits.shape == (ob, num_ep_ranks)

            torch.cuda.current_stream().synchronize()
            input_splits = input_splits.tolist()
            output_splits = output_splits.tolist()

        routed_input = batched_all_to_all_single(
            batched_input=routed_input,
            batched_out_splits=output_splits,
            batched_in_splits=input_splits,
            group=ep_mesh.get_group(),
        )

    else:

        exps_per_rank = num_tokens_per_expert.shape[1]
        num_ep_ranks = num_experts // exps_per_rank
        assert num_ep_ranks == 1
        num_tokens_per_expert_group = num_tokens_per_expert
        input_splits = output_splits = None

    # Pad and permute routed input
    permuted_indices, tokens_per_expert, _ = batched_generate_permute_indices(
        batched_tokens_per_expert_group=num_tokens_per_expert_group,
        experts_per_rank=exps_per_rank,
        num_ranks=num_ep_ranks,
        batched_max_len=[
            routed_input[b].shape[0] + exps_per_rank * TOKEN_GROUP_ALIGN_SIZE_M
            for b in range(ob)
        ],
        alignment=TOKEN_GROUP_ALIGN_SIZE_M,
        use_cpu=False,
    )
    (
        padded_permuted_routed_input,
        routed_input_shapes_before_permute,
    ) = batched_permute_and_pad(
        batched_routed_input=routed_input,
        batched_permute_indices=permuted_indices,
    )
    return (
        padded_permuted_routed_input,
        tokens_per_expert,
        scores_sorted,
        token_indices,
        permuted_indices,
        routed_input_shapes_before_permute,
        input_splits,
        output_splits,
    )


@torch.library.custom_op("autoparallel::token_combine", mutates_args=())
def token_combine_op(
    base_output: torch.Tensor,
    padded_permuted_routed_output: torch.Tensor,
    top_scores_sorted: torch.Tensor,
    token_indices_sorted: torch.Tensor,
    permuted_indices: torch.Tensor,
    routed_input_shapes_before_permute: list[torch.Size],
    score_before_experts: bool = False,
    input_splits: list[list[int]] | None = None,
    output_splits: list[list[int]] | None = None,
    ep_mesh: DeviceMesh | None = None,
) -> torch.Tensor:
    """
    Token combine operator with custom backward.

    Forward: Combine expert outputs back to original token positions.
    Backward: Use token dispatch logic to gather gradients for expert processing.

    Args:
        base_output: Base output tensor (ob, ib * slen, dim) -
            e.g., from shared experts or zeros if no shared experts
        paddded_permuted_routed_output: Padded and permuted routed output
         (ob, ib * slender * top_k + (num_experts * TOKEN_GROUP_ALIGN_SIZE_M), dim)
        top_scores_sorted: Sorted scores (ob, ib * slen * top_k)
        token_indices_sorted: Sorted token indices (ob, ib * slen * top_k)
        permuted_indices: Permute indices (ob, ib * slen * top_k + (num_experts * TOKEN_GROUP_ALIGN_SIZE_M))
        routed_input_shape_before_permute: routed input shapes before permute
        score_before_experts: Whether scores were applied before experts
        input_splits: list of input splits for all to all
        output_splits: list of output splits for all to all
        ep_mesh: DeviceMesh for expert parallel token exchange

    Returns:
        combined_output: (ob, ib * slen, dim) - final combined output
    """
    dim = base_output.shape[-1]

    # Unpad and permute routed output
    routed_output = batched_unpermute_and_unpad(
        batched_padded_permuted_routed_output=padded_permuted_routed_output,
        batched_permute_indices=permuted_indices,
        batched_routed_input_shapes_before_permute=routed_input_shapes_before_permute,
    )
    if ep_mesh is not None:
        # Need to perform an all to all to exchange tokens between ranks
        assert input_splits is not None and output_splits is not None
        routed_output = batched_all_to_all_single(
            batched_input=routed_output,
            batched_out_splits=input_splits,
            batched_in_splits=output_splits,
            group=ep_mesh.get_group(),
        )

    # Apply scores if not applied before experts
    if not score_before_experts:
        routed_output = (
            routed_output.to(torch.float32) * top_scores_sorted.unsqueeze(-1)
        ).to(routed_output.dtype)

    # Expand token indices for scatter_add
    token_indices_expanded = token_indices_sorted.unsqueeze(-1).expand(-1, -1, dim)

    # Scatter and add expert outputs to base output
    combined_output = base_output.scatter_add(
        dim=1, index=token_indices_expanded, src=routed_output
    )

    return combined_output


def setup_token_dispatch_context(ctx, inputs, output):
    """Setup context for token_dispatch backward pass."""
    (
        x,
        top_scores,
        selected_experts_indices,
        num_tokens_per_expert,
        num_experts,
        top_k,
        score_before_experts,
        ep_mesh,
    ) = inputs

    (
        padded_permuted_routed_input,
        tokens_per_expert,
        top_scores_sorted,
        token_indices_sorted,
        batch_permuted_indices,
        routed_input_shape_before_permute,
        input_splits,
        output_splits,
    ) = output

    # Save tensors needed for backward
    if score_before_experts:
        # Save unscaled routed_input (before score multiplication)
        unscaled_routed_input = padded_permuted_routed_input / (
            top_scores_sorted.unsqueeze(-1) + 1e-8
        )
        ctx.save_for_backward(
            top_scores_sorted,
            token_indices_sorted,
            tokens_per_expert,
            unscaled_routed_input,
            batch_permuted_indices,
        )
    else:
        ctx.save_for_backward(
            top_scores_sorted,
            token_indices_sorted,
            tokens_per_expert,
            batch_permuted_indices,
        )

    ctx.num_experts = num_experts
    ctx.top_k = top_k
    ctx.score_before_experts = score_before_experts
    ctx.input_shape = x.shape
    ctx.routed_input_shape_before_permute = routed_input_shape_before_permute
    ctx.ep_mesh = ep_mesh
    ctx.input_splits = input_splits
    ctx.output_splits = output_splits


def token_dispatch_backward(
    ctx,
    grad_padded_permuted_routed_input,
    grad_tokens_per_expert,
    grad_top_scores_sorted,
    grad_token_indices_sorted,
    grad_batch_permuted_indices,
    grad_routed_input_shape_before_permute,
    grad_input_splits,
    grad_output_splits,
):
    """
    Backward pass for token_dispatch.

    Scatter gradients back to original token positions. Handles all-to-all communication
    for expert parallel case.
    """
    if ctx.score_before_experts:
        (
            top_scores_sorted,
            token_indices_sorted,
            num_tokens_per_expert,
            unscaled_routed_input,
            batch_permuted_indices,
        ) = ctx.saved_tensors
    else:
        (
            top_scores_sorted,
            token_indices_sorted,
            num_tokens_per_expert,
            batch_permuted_indices,
        ) = ctx.saved_tensors
        unscaled_routed_input = None

    ob, ib_slen, dim = ctx.input_shape

    if grad_padded_permuted_routed_input is None:
        return None, None, None, None, None, None, None, None

    # Step 1: Handle score gradients if scores were applied before experts
    grad_routed_input_for_processing = grad_padded_permuted_routed_input
    grad_top_scores = None

    if ctx.score_before_experts:
        assert unscaled_routed_input is not None

        # Unpad and unpermute both gradients and unscaled input for score gradient computation
        grad_routed_input_unpermuted = batched_unpermute_and_unpad(
            batched_padded_permuted_routed_output=grad_padded_permuted_routed_input,
            batched_permute_indices=batch_permuted_indices,
            batched_routed_input_shapes_before_permute=ctx.routed_input_shape_before_permute,
        )

        unscaled_routed_input_unpermuted = batched_unpermute_and_unpad(
            batched_padded_permuted_routed_output=unscaled_routed_input,
            batched_permute_indices=batch_permuted_indices,
            batched_routed_input_shapes_before_permute=ctx.routed_input_shape_before_permute,
        )

        # Chain rule: ∂L/∂scores = grad_routed_input * unscaled_routed_input
        grad_top_scores_flat = (
            grad_routed_input_unpermuted * unscaled_routed_input_unpermuted
        ).sum(dim=-1)
        grad_top_scores = grad_top_scores_flat.reshape(ob, ib_slen, ctx.top_k)

        # Remove score scaling from gradients for further processing
        grad_routed_input_for_processing = grad_padded_permuted_routed_input / (
            top_scores_sorted.unsqueeze(-1) + 1e-8
        )

    # Step 2: Unpad and unpermute gradients
    grad_routed_input = batched_unpermute_and_unpad(
        batched_padded_permuted_routed_output=grad_routed_input_for_processing,
        batched_permute_indices=batch_permuted_indices,
        batched_routed_input_shapes_before_permute=ctx.routed_input_shape_before_permute,
    )

    # Step 3: Handle all-to-all communication (inverse of forward pass)
    if ctx.ep_mesh is not None:
        # Reverse the all-to-all communication from forward pass
        grad_routed_input = batched_all_to_all_single(
            batched_input=grad_routed_input,
            batched_out_splits=ctx.input_splits,  # Reverse: use input_splits as out_splits
            batched_in_splits=ctx.output_splits,  # Reverse: use output_splits as in_splits
            group=ctx.ep_mesh.get_group(),
        )

    # Step 4: Scatter gradients back to original token positions (inverse of gather)
    grad_x = torch.zeros(
        (ob, ib_slen, dim),
        dtype=grad_routed_input.dtype,
        device=grad_routed_input.device,
    )
    token_indices_expanded = token_indices_sorted.unsqueeze(-1).expand(-1, -1, dim)
    grad_x = grad_x.scatter_add(
        dim=1, index=token_indices_expanded, src=grad_routed_input
    )

    return grad_x, grad_top_scores, None, None, None, None, None, None


def setup_token_combine_context(ctx, inputs, output):
    """Setup context for token_combine backward pass."""
    (
        base_output,
        padded_permuted_routed_output,
        top_scores_sorted,
        token_indices_sorted,
        batch_permuted_indices,
        routed_input_shapes_before_permute,
        num_tokens_per_expert,
        num_experts,
        top_k,
        score_before_experts,
        input_splits,
        output_splits,
        ep_mesh,
    ) = inputs

    # Save tensors needed for backward
    if not score_before_experts:
        # Save routed output before score multiplication for gradient computation
        unpadded_unscaled_routed_output = batched_unpermute_and_unpad(
            batched_padded_permuted_routed_output=padded_permuted_routed_output,
            batched_permute_indices=batch_permuted_indices,
            batched_routed_input_shapes_before_permute=routed_input_shapes_before_permute,
        )
        ctx.save_for_backward(
            top_scores_sorted,
            token_indices_sorted,
            num_tokens_per_expert,
            unpadded_unscaled_routed_output,
            batch_permuted_indices,
        )
    else:
        ctx.save_for_backward(
            top_scores_sorted,
            token_indices_sorted,
            num_tokens_per_expert,
            batch_permuted_indices,
        )

    ctx.num_experts = num_experts
    ctx.top_k = top_k
    ctx.score_before_experts = score_before_experts
    ctx.base_shape = base_output.shape
    ctx.routed_shape = padded_permuted_routed_output.shape
    ctx.routed_input_shapes_before_permute = routed_input_shapes_before_permute
    ctx.ep_mesh = ep_mesh
    ctx.input_splits = input_splits
    ctx.output_splits = output_splits


def token_combine_backward(
    ctx,
    grad_combined_output,
):
    """
    Backward pass for token_combine.

    token_combine forward: unpad_unpermute → all_to_all → scatter_add
    token_combine backward: gather → all_to_all → pad_permute
    """
    if not ctx.score_before_experts:
        (
            top_scores_sorted,
            token_indices_sorted,
            num_tokens_per_expert,
            unscaled_routed_output,
            batch_permuted_indices,
        ) = ctx.saved_tensors
    else:
        (
            top_scores_sorted,
            token_indices_sorted,
            num_tokens_per_expert,
            batch_permuted_indices,
        ) = ctx.saved_tensors
        unscaled_routed_output = None

    ob, ib_slen, dim = ctx.base_shape

    if grad_combined_output is None:
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    # Step 1: Base output gets the full gradient (residual connection)
    grad_base_output = grad_combined_output.clone()

    # Step 2: Gather gradients from original positions (inverse of scatter_add)
    token_indices_expanded = token_indices_sorted.unsqueeze(-1).expand(-1, -1, dim)
    grad_routed_output = torch.gather(
        grad_combined_output, dim=1, index=token_indices_expanded
    )

    # Step 3: Handle all-to-all communication (inverse of forward pass)
    if ctx.ep_mesh is not None:
        # Reverse the all-to-all communication from forward pass
        grad_routed_output = batched_all_to_all_single(
            batched_input=grad_routed_output,
            batched_out_splits=ctx.output_splits,  # Reverse: use output_splits as out_splits
            batched_in_splits=ctx.input_splits,  # Reverse: use input_splits as in_splits
            group=ctx.ep_mesh.get_group(),
        )

    # Step 4: Handle score gradients if scores were applied in forward pass
    grad_top_scores_sorted = None
    if not ctx.score_before_experts and unscaled_routed_output is not None:
        # Apply reverse all-to-all to unscaled_routed_output for correct gradient computation
        if ctx.ep_mesh is not None:
            unscaled_routed_output = batched_all_to_all_single(
                batched_input=unscaled_routed_output,
                batched_out_splits=ctx.output_splits,
                batched_in_splits=ctx.input_splits,
                group=ctx.ep_mesh.get_group(),
            )
        # Chain rule: ∂L/∂scores = grad_routed_output * unscaled_routed_output
        grad_top_scores_sorted = (grad_routed_output * unscaled_routed_output).sum(
            dim=-1
        )

    # Step 5: Apply permutation and padding (inverse of unpad_unpermute)
    (
        grad_padded_permuted_routed_output,
        _,  # Don't need the shapes
    ) = batched_permute_and_pad(
        batched_routed_input=grad_routed_output,
        batched_permute_indices=batch_permuted_indices,
    )

    return (
        grad_base_output,  # grad_base_output
        grad_padded_permuted_routed_output,  # grad_padded_permuted_routed_output
        grad_top_scores_sorted,  # grad_top_scores_sorted
        None,  # grad_token_indices_sorted
        None,  # grad_batch_permuted_indices
        None,  # grad_routed_input_shapes_before_permute
        None,  # grad_num_tokens_per_expert
        None,  # grad_num_experts
        None,  # grad_top_k
        None,  # grad_score_before_experts
        None,  # grad_input_splits
        None,  # grad_output_splits
        None,  # grad_ep_mesh
    )


# Register backward functions
torch.library.register_autograd(
    "autoparallel::token_dispatch",
    token_dispatch_backward,
    setup_context=setup_token_dispatch_context,
)

torch.library.register_autograd(
    "autoparallel::token_combine",
    token_combine_backward,
    setup_context=setup_token_combine_context,
)


# Register fake implementations for meta tensors
@token_dispatch_op.register_fake
def token_dispatch_meta(
    x: torch.Tensor,
    top_scores: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    num_experts: int,
    top_k: int,
    score_before_experts: bool = False,
    ep_mesh: DeviceMesh | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[torch.Size],
    list[list[int]] | None,
    list[list[int]] | None,
]:
    ob, ib_slen, dim = x.shape

    # Get maximum possible length based on max tokens per expert across all batches
    max_len = ib_slen * top_k + (num_experts * TOKEN_GROUP_ALIGN_SIZE_M)

    # Padded and permuted routed input with maximum possible size
    padded_permuted_routed_input = torch.empty(
        (ob, max_len, dim),
        dtype=x.dtype,
        device=x.device,
    )
    tokens_per_expert = torch.empty(
        (ob, num_experts),
        dtype=num_tokens_per_expert.dtype,
        device=num_tokens_per_expert.device,
    )
    scores_sorted = torch.empty(
        (ob, ib_slen * top_k), dtype=top_scores.dtype, device=top_scores.device
    )
    token_indices = torch.empty(
        (ob, ib_slen * top_k),
        dtype=selected_experts_indices.dtype,
        device=selected_experts_indices.device,
    )
    batch_permuted_indices = torch.empty(
        (ob, max_len),
        dtype=torch.long,
        device=x.device,
    )
    # Shapes before permute: each batch has (ib_slen * top_k + 1, dim) where +1 is the zero row
    routed_input_shape_before_permute = [
        torch.Size([ib_slen * top_k + 1, dim]) for _ in range(ob)
    ]

    # Mock splits for all-to-all (will be None if ep_mesh is None)
    input_splits = None
    output_splits = None

    return (
        padded_permuted_routed_input,
        tokens_per_expert,
        scores_sorted,
        token_indices,
        batch_permuted_indices,
        routed_input_shape_before_permute,
        input_splits,
        output_splits,
    )


@token_combine_op.register_fake
def token_combine_meta(
    base_output: torch.Tensor,
    padded_permuted_routed_output: torch.Tensor,
    top_scores_sorted: torch.Tensor,
    token_indices_sorted: torch.Tensor,
    batch_permuted_indices: torch.Tensor,
    routed_input_shapes_before_permute: list[torch.Size],
    num_tokens_per_expert: torch.Tensor,
    num_experts: int,
    top_k: int,
    score_before_experts: bool = False,
    input_splits: list[list[int]] | None = None,
    output_splits: list[list[int]] | None = None,
    ep_mesh: DeviceMesh | None = None,
) -> torch.Tensor:
    return torch.empty_like(base_output)


@register_opschema_rule(torch.ops.autoparallel.token_dispatch.default)
def _token_dispatch_strategy(mesh: DeviceMesh, op_schema: OpSchema):
    from torch.distributed.tensor._op_schema import PlacementList
    from torch.distributed.tensor._ops.utils import expand_to_full_mesh_op_strategy

    x_strategy = cast(OpStrategy, op_schema.args_schema[0])
    top_scores_strategy = cast(OpStrategy, op_schema.args_schema[1])
    selected_expert_indices_strategy = cast(OpStrategy, op_schema.args_schema[2])
    num_tokens_per_expert = cast(list[int], op_schema.args_schema[3])

    single_mesh_dim_strategies = []

    # placement list stores placements of [output, mat1, mat2, offs]
    # first we always have replicate all for inputs and output
    all_replicate: PlacementList = [Replicate()] * 4
    single_mesh_dim_strategies.append(all_replicate)
    # mat1 sharded on outer batch, mat2 is replicated, offs is sharded on outer batch
    #  -> output is sharded on outer batch
    # mat1 is replicated, mat2 is sharded on column dim, offs is replicated
    #  -> output is sharded on column dim
    single_mesh_dim_strategies.append([Shard(2), Replicate(), Shard(2), Replicate()])
    # mat1 is sharded on column dim, mat2 is sharded on row dim, offs is replicated
    #  -> output is partial
    single_mesh_dim_strategies.append([Partial(), Shard(2), Shard(1), Replicate()])
    # mat1 is dynamically sharded on row dim, mat2 is sharded on experts dim,
    # offs is sharded on experts dim -> output is dynamically sharded on row dim
    single_mesh_dim_strategies.append(
        [PartitionedShard(1), PartitionedShard(1), Shard(0), Shard(1)]
    )

    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=1
    )


@register_opschema_rule(torch.ops.autoparallel.token_combine.default)
def _token_combine_strategy(mesh: DeviceMesh, op_schema: OpSchema):
    from torch.distributed.tensor._op_schema import PlacementList
    from torch.distributed.tensor._ops.utils import expand_to_full_mesh_op_strategy

    mat1_strategy = cast(OpStrategy, op_schema.args_schema[0])
    mat2_strategy = cast(OpStrategy, op_schema.args_schema[1])
    offs_strategy = cast(OpStrategy, op_schema.args_schema[2])

    assert len(mat1_strategy.shape) == 3
    assert len(mat2_strategy.shape) == 3
    assert len(offs_strategy.shape) == 2
    assert mat1_strategy.shape[0] == offs_strategy.shape[0]
    assert mat2_strategy.shape[0] == offs_strategy.shape[1]
    assert mat1_strategy.shape[2] == mat2_strategy.shape[1]

    single_mesh_dim_strategies = []

    # placement list stores placements of [output, mat1, mat2, offs]
    # first we always have replicate all for inputs and output
    all_replicate: PlacementList = [Replicate()] * 4
    single_mesh_dim_strategies.append(all_replicate)
    # mat1 sharded on outer batch, mat2 is replicated, offs is sharded on outer batch
    #  -> output is sharded on outer batch
    single_mesh_dim_strategies.append([Shard(0), Shard(0), Replicate(), Shard(0)])
    # mat1 is replicated, mat2 is sharded on column dim, offs is replicated
    #  -> output is sharded on column dim
    single_mesh_dim_strategies.append([Shard(2), Replicate(), Shard(2), Replicate()])
    # mat1 is sharded on column dim, mat2 is sharded on row dim, offs is replicated
    #  -> output is partial
    single_mesh_dim_strategies.append([Partial(), Shard(2), Shard(1), Replicate()])
    # mat1 is dynamically sharded on row dim, mat2 is sharded on experts dim,
    # offs is sharded on experts dim -> output is dynamically sharded on row dim
    single_mesh_dim_strategies.append(
        [PartitionedShard(1), PartitionedShard(1), Shard(0), Shard(1)]
    )

    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=1
    )
