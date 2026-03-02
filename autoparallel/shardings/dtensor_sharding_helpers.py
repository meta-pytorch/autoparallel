# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
from contextlib import ExitStack, contextmanager
from typing import Optional

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpSpec,
    OpStrategy,
    StrategyType,
)
from torch.distributed.tensor._ops.utils import (
    generate_redistribute_costs,
    is_tensor_shardable,
    register_op_strategy,
)
from torch.distributed.tensor.debug import (
    _clear_fast_path_sharding_prop_cache,
    _clear_python_sharding_prop_cache,
)
from torch.distributed.tensor.placement_types import Placement, Replicate, Shard

try:
    from torch.utils._cxx_pytree import tree_leaves
except ImportError:
    from torch.utils._pytree import tree_leaves  # type: ignore[no-redef]


logger = logging.getLogger(__name__)

aten = torch.ops.aten

# reference to existing sharding_propagator DTensor upstream
propagator = DTensor._op_dispatcher.sharding_propagator

enable_implicit_replication = False
_current_stack = None

replicate_op_strategy = torch.distributed.tensor._ops.utils.replicate_op_strategy


# TODO: remove and refer to
# https://github.com/pytorch/pytorch/blob/9c107606629de6383f55e3b48b42e594d23407b1/test/distributed/tensor/test_op_strategy.py#L446
# once the function is moved outside of the test folder in upstream
@contextmanager
def op_strategy_context(op_overload, strategy_func, schema_info=None):
    """
    Context manager for setting and clearing op strategies.
    Args:
        op_overload: The operator overload to set or clear the strategy for.
        strategy_func: The strategy function to set for the operator overload.
        schema_info: Optional schema information for the operator overload.
    Yields:
        None
    """
    propagator = DTensor._op_dispatcher.sharding_propagator
    _origin_op_strategy_funcs = None
    _origin_op_strategy_schema = None
    try:
        # register the op strategy
        if op_overload in propagator.op_strategy_funcs:
            _origin_op_strategy_funcs = propagator.op_strategy_funcs[op_overload]
            del propagator.op_strategy_funcs[op_overload]
        if op_overload in propagator.op_to_schema_info:
            _origin_op_strategy_schema = propagator.op_to_schema_info[op_overload]
            del propagator.op_to_schema_info[op_overload]
        register_op_strategy(op_overload, schema_info=schema_info)(strategy_func)
        yield
    finally:
        # clear this op strategy cache
        if _origin_op_strategy_funcs is None:
            if op_overload in propagator.op_strategy_funcs:
                del propagator.op_strategy_funcs[op_overload]
        else:
            propagator.op_strategy_funcs[op_overload] = _origin_op_strategy_funcs
        if _origin_op_strategy_schema is None:
            if op_overload in propagator.op_to_schema_info:
                del propagator.op_to_schema_info[op_overload]
        else:
            propagator.op_to_schema_info[op_overload] = _origin_op_strategy_schema
        _clear_fast_path_sharding_prop_cache()
        _clear_python_sharding_prop_cache()


# -------------define universal op strategy-------------
def batch_shard_strategy(
    op_schema: OpSchema,
    input_shard_dim: list[Optional[int]],
    output_shard_dim: list[Optional[int]],
    enable_shard_batch_dim_over_multiple_axis: bool = False,
) -> OpStrategy:
    """
    Shard the input tensor over the specified dimensions. The strategy will map
    batch dim of input/output tensors to the same device mesh axis (or same
    multiple device axes). All input must either have one specified batch dim or
    no batch dim. If an input doesn't have batch dim, the strategy will assume
    the tensor will be broadcasted to batch dim and processed by the operator.
    For inputs specified with a batch dim, user need to make sure the batch dim
    size are all the same. Output should always have a batch dim.

    Args:
        op_schema (OpSchema): the op schema.

        input_shard_dim (list[Optional[int]]): the list of shard dimensions to
        consider for each input tensor argument. Use `None` if no batch dim of
        the input arg. If an arg is List[Tenor], we flatten it first and then
        match with input_shard_dim. Since the dim is not specific to the device
        mesh axis, it can be a combination of any device axes. Example 1: input
        tensor A[1024,64,8], B[1024,64,16], with input_shard_dim = [1,1], it can
        shard A's dim 0 over device axis X, and shard B's dim 0 over device axis
        X. X can be any of device axes. The output follow the same sharding as
        input. Example 2: input tensor A[64,8], B[64,16,1024], C[64,8], with
        input_shard_dim = [None,2,None], it will Replicate A,C over all device
        dim and only shard B's dim 2 over the device mesh. Assume the device
        mesh has 3 axis, then tensor B's placement can be (Shard(2), Shard(2),
        Replicate()), (Shard(2), Replicate(), Shard(2)), (Replicate(), Shard(2),
        Shard(2)).

        output_shard_dim (list[Optional[int]]): the list of shard dimensions to
        consider for each output tensor argument. Use `None` if no batch dim of
        the output arg. For example, if the output is a single tensor and is
        sharded on dim 0, pass in [0] then.

        enable_shard_batch_dim_over_multiple_axis (bool): if True, the strategy
        will try also map batch dim to multiple device axis. Default is False.

    Note: It is the user's responsibility to make sure the sharded tensor for
    processing is correct in shape.
    """
    output_type = [str(ret.type) for ret in op_schema.op._schema.returns]
    # TODO(zpcore): Confirm if view op can be handle properly or not. Prevent
    # handling view ops until confirmed.
    if op_schema.op.is_view:
        raise RuntimeError(
            "fallback strategy is unable to handle view ops until confirmed"
        )
    if "List[Tensor]" in output_type:
        raise RuntimeError(
            "fallback strategy is unable to handle ops with List[Tensor] output "
            "because size of the list may depend on the op's input value"
        )
    inputs_strategy = tree_leaves(op_schema.args_strategy)
    assert len(inputs_strategy) == len(input_shard_dim)
    output_strategy = OpStrategy([])
    mesh = inputs_strategy[0].mesh
    device_axis = list(range(mesh.ndim))
    use_how_many_axis = (
        [i + 1 for i in range(mesh.ndim)]
        if enable_shard_batch_dim_over_multiple_axis
        else [1]
    )
    # number of device axes to shard on for the batch dim
    for num_axis in use_how_many_axis:
        device_combinations = list(itertools.combinations(device_axis, num_axis))
        # e.g., if num_axis == 2, device_combinations = [(0,1), (0,2), (1,2),
        # ...]. Then One feasible strategy is to shard tensor dim on both axis
        # (0,1). We check all combinations in device_combinations below.
        for comb in device_combinations:
            input_specs_list: list[DTensorSpec] = []
            output_specs_list: list[DTensorSpec] = []
            is_shardable = True
            for op_stratgy, dim in zip(inputs_strategy, input_shard_dim):
                # create a new list of shard_dim_option
                new_placements: list[Placement] = [Replicate()] * mesh.ndim
                for axis in comb:
                    new_placements[axis] = Shard(dim) if dim else Replicate()
                tensor_meta = op_stratgy.strategies[0].output_spec.tensor_meta
                new_input_spec = DTensorSpec(
                    mesh,
                    tuple(new_placements),
                    tensor_meta=op_stratgy.strategies[0].output_spec.tensor_meta,
                )
                if not is_tensor_shardable(tensor_meta.shape, new_input_spec):
                    is_shardable = False
                    break
                input_specs_list.append(new_input_spec)
            if not is_shardable:
                continue
            for dim in output_shard_dim:
                new_placements = [Replicate()] * mesh.ndim
                for axis in comb:
                    new_placements[axis] = Shard(dim) if dim else Replicate()
                output_spec = DTensorSpec(
                    mesh,
                    tuple(new_placements),
                )
                output_specs_list.append(output_spec)

            output_specs = (
                output_specs_list[0]
                if len(output_specs_list) == 1
                else tuple(output_specs_list)
            )
            input_specs = input_specs_list
            redistribute_cost = [
                generate_redistribute_costs(strat, input_specs_list[i])
                for i, strat in enumerate(inputs_strategy)
            ]
            output_strategy.strategies.append(
                OpSpec(output_specs, input_specs, redistribute_cost)  # type: ignore
            )
    return output_strategy


def _try_single_dim_strategy(
    op: torch._ops.OpOverload, op_schema: OpSchema
) -> Optional[StrategyType]:
    """
    Check if the op has a single-dim strategy registered upstream and expand it.

    Upstream DTensor is migrating ops from register_op_strategy (which populates
    op_strategy_funcs) to register_single_dim_strategy (which populates
    op_single_dim_strategy_funcs). This function handles the new path so
    autoparallel can use strategies from either registry.
    """
    single_dim_info = propagator.op_single_dim_strategy_funcs.get(op)
    if single_dim_info is None:
        return None

    from torch.distributed.tensor._ops.single_dim_strategy import (
        _insert_single_dim_replication_strategy,
        _ShardingPlaceholder,
    )
    from torch.distributed.tensor._ops.utils import expand_to_full_mesh_op_strategy

    mesh = op_schema.args_strategy[0].mesh

    # Compute output tensor meta via the propagator's existing infra.
    spec_args = tuple(
        arg.strategies[0].output_spec if isinstance(arg, OpStrategy) else arg
        for arg in op_schema.args_schema
    )
    spec_schema = OpSchema(op, spec_args, op_schema.kwargs_schema)
    out_tensor_meta = propagator._propagate_tensor_meta_non_cached(spec_schema)

    # Get single-dim strategies and resolve placeholders.
    # Upstream _fill_single_dim_strategy_placeholders only expands placeholders
    # using shard types found in the runtime input placements. Since autoparallel
    # explores all placements (not a single runtime one), we always resolve
    # _ShardingPlaceholder(d) -> Shard(d).
    from torch.distributed.tensor._dtensor_spec import TensorMeta

    if out_tensor_meta is None:
        num_outputs = 0
    elif isinstance(out_tensor_meta, TensorMeta):
        num_outputs = 1
    else:
        num_outputs = len(out_tensor_meta)
    num_inputs = len(op_schema.args_strategy)
    strategies = single_dim_info.func(op, op_schema.args_meta, op_schema.kwargs_meta)
    strategies = _insert_single_dim_replication_strategy(
        strategies, num_outputs, num_inputs
    )
    resolved: list[list[Placement | None]] = []
    for s in strategies:
        resolved.append(
            [Shard(p.dim) if isinstance(p, _ShardingPlaceholder) else p for p in s]
        )

    result = expand_to_full_mesh_op_strategy(
        mesh,
        op_schema,
        resolved,
        input_index=num_outputs,
        output_tensor_meta=out_tensor_meta,
    )

    # Recompute redistribute costs against autoparallel's multi-entry input
    # OpStrategy args. The expansion computed costs against single-entry
    # strategies (the runtime placement), but autoparallel needs costs sized
    # to match each input's full set of placement options.
    inputs_strategy = op_schema.args_strategy
    for op_spec in result.strategies:
        assert op_spec.input_specs is not None
        op_spec.redistribute_cost = [
            generate_redistribute_costs(strategy, spec)
            for strategy, spec in zip(inputs_strategy, op_spec.input_specs)
        ]
    return result


def get_op_strategy(op: torch._ops.OpOverload, op_schema: OpSchema) -> StrategyType:
    global enable_implicit_replication, _current_stack

    if op not in propagator.op_strategy_funcs:
        # Check single-dim strategies (newer upstream DTensor registration path)
        single_dim_result = _try_single_dim_strategy(op, op_schema)
        if single_dim_result is not None:
            return single_dim_result

        if not enable_implicit_replication:
            raise NotImplementedError(
                f"Operator {op} does not have a sharding strategy registered."
            )
        else:
            # Use the current stack if available
            if _current_stack is not None:
                _current_stack.enter_context(
                    op_strategy_context(op, replicate_op_strategy)
                )
            else:
                # No stack available, just register permanently
                register_op_strategy(op)(replicate_op_strategy)
            logger.warning(
                f"implicitly registering `{op}` with `{replicate_op_strategy.__name__}`"
            )
    return propagator.op_strategy_funcs[op](op_schema)


@contextmanager
def with_implicit_strategies():
    """Context manager to enable implicit replication and clean up strategies."""
    global enable_implicit_replication, _current_stack

    # Create a fresh ExitStack for this context
    with ExitStack() as local_stack:
        # Store the stack as a global variable
        old_stack = _current_stack
        _current_stack = local_stack

        # Enable implicit replication
        old_value = enable_implicit_replication
        enable_implicit_replication = True
        try:
            yield
        finally:
            # Restore the original values
            _current_stack = old_stack
            enable_implicit_replication = old_value


# TODO: automatic generate redistribute cost for strategies. There exists a
# `fill_missing_redistribute_cost` in autoparallel/utils.py, which is a hack
# to generate redistribute cost given input specs, and only tested on
# certain ops. We can potentially make an improvement.
def fill_missing_redistribute_cost(op: torch._ops.OpOverload, op_schema: OpSchema):
    """
    Fill missing redistribute cost for strategies.
    """
    ...
