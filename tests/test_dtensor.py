# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import functools

import numpy as np
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import (
    OpInfo,
    OpSchema,
    OpSpec,
    OpStrategy,
    OutputSharding,
    OutputSpecType,
    RuntimeSchemaInfo,
    TupleStrategy,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from autoparallel.dtensor_util import (
    batch_shard_strategy,
    get_op_strategy,
    op_strategy_context,
    with_implicit_strategies,
)

propagator = DTensor._op_dispatcher.sharding_propagator

aten = torch.ops.aten

# -------------Test op strategy registration-------------
# custom op without List[Tensor] as input
# reference: https://docs.pytorch.org/docs/stable/library.html#torch.library.register_autograd


@torch.library.custom_op("mylib::numpy_sin", mutates_args=())
def numpy_sin(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()
    out_np = np.sin(x_np) + np.sin(y_np)
    return torch.from_numpy(out_np).to(device=x.device)


def setup_context(ctx, inputs, output):
    (x, y) = inputs
    ctx.save_for_backward(x, y)


def backward(ctx, grad):
    (x, y) = ctx.saved_tensors
    return grad * x.cos(), grad * y.cos()


@numpy_sin.register_fake
def _fw(x, y):
    return torch.empty_like(x)


torch.library.register_autograd(
    "mylib::numpy_sin", backward, setup_context=setup_context
)


# custom op with List[Tensor] as input
@torch.library.custom_op("mylib::numpy_tuple_sin", mutates_args=())
def numpy_tuple_sin(
    x: torch.Tensor, y: list[torch.Tensor], z: torch.Tensor
) -> torch.Tensor:
    x_np = x.cpu().numpy()
    y_np = [i.cpu().numpy() for i in y]
    z_np = z.cpu().numpy()

    out_np = np.sin(x_np) + np.sin(z_np) + sum(np.sin(i) for i in y_np)
    return torch.from_numpy(out_np).to(device=x.device)


def setup_tuple_context(ctx, inputs, output):
    (x, y, z) = inputs
    ctx.save_for_backward(x, y, z)


def tuple_backward(ctx, grad):
    (x, y, z) = ctx.saved_tensors
    return grad * x.cos(), [grad * i.cos() for i in y], grad * z.cos()


@numpy_tuple_sin.register_fake
def _fw_tuple(x, y, z):
    return torch.empty_like(x)


torch.library.register_autograd(
    "mylib::numpy_tuple_sin", tuple_backward, setup_context=setup_tuple_context
)


# Overwrite upstream `_op_dispatcher.sharding_propagator` with customized
# sharding_propagator. This is for testing purpose under eager mode and
# AutoParallel won't use the propagate function. The main changes are 1) Skip
# the LRU cache for strategies, which makes it easier to update/delete
# strategies; 2) don't necessarily trigger runtime error if an op doesn't have
# rule or strategy registered. Instead, check if we want to use implicit
# replication strategy fallback.
class CustomShardingPropagator(
    torch.distributed.tensor._sharding_prop.ShardingPropagator
):
    def __init__(self):
        super().__init__()
        self.propagate_op_sharding.cache.cache_clear()
        self.op_to_rules = propagator.op_to_rules
        self.op_strategy_funcs = propagator.op_strategy_funcs
        self.op_to_schema_info = propagator.op_to_schema_info

    def propagate(self, op_info: OpInfo) -> None:
        op_info.output_sharding = self.propagate_op_sharding_non_cached(op_info.schema)

    def propagate_op_sharding_non_cached(self, op_schema: OpSchema) -> OutputSharding:
        """
        Propagate the sharding for an operator given the op_schema.
        """
        # special case op, we don't need to propagate for local
        # scalar. TODO: figure out a better way to handle this
        if op_schema.op is aten._local_scalar_dense.default:
            return OutputSharding(None, op_schema)

        out_tensor_meta = self._propagate_tensor_meta_non_cached(op_schema)

        if op_schema.op in self.op_to_rules:
            # propagate the sharding with rule
            sharding_prop_func = self.op_to_rules[op_schema.op]

            # step 1. there's sharding propagation rule, run
            # sharding propagation to get the output sharding
            try:
                output_sharding = sharding_prop_func(op_schema)
            except NotImplementedError as e:
                raise e
            except Exception as e:
                raise RuntimeError(
                    f"Sharding propagation failed on op {op_schema}.\nError: {e}"
                ) from e

            # step 2. if can't get output_spec from sharding
            # propagation (i.e. no rules apply for input
            # placements), we return the output sharding
            # with schema suggestions, which can be used to
            # decide how to do redistribute on inputs
            if output_sharding.output_spec is None:
                if output_sharding.redistribute_schema is None:
                    raise RuntimeError(
                        f"Sharding propagation failed on op {op_schema}!"
                    )
                else:
                    # we do auto redistribute on inputs if necessary
                    # run sharding propagation again with suggested schema
                    propagation_res = sharding_prop_func(
                        output_sharding.redistribute_schema
                    )
                    # we set the output sharding with the new propagation result
                    # so that dispatching know both output_spec and redistribute_schema
                    # exist, which indicates a reshard is needed
                    output_sharding.output_spec = propagation_res.output_spec
                    output_sharding.needs_redistribute = True

            # associate the output sharding with the output tensor metadata
            self._wrap_output_spec_tensor_meta(
                op_schema.op, output_sharding.output_spec, out_tensor_meta
            )

            return output_sharding
        else:
            # wrap the op_schema with op strategy for sharding strategy propagation
            strategy_schema = self._wrap_with_op_strategy(op_schema)
            strategy_schema.schema_info = op_schema.schema_info

            # assign implicit strategy if enabled
            get_op_strategy(strategy_schema.op, strategy_schema)

            # run sharding strategy propagation/generation
            op_strategy = self.op_strategy_funcs[op_schema.op](strategy_schema)

            if isinstance(op_strategy, OpStrategy):
                # single Op strategy
                output_strategy = self._select_strategy(op_strategy)
                # check if we need to redistribute the input
                needs_redistribute = False
                expected_input_specs: list[DTensorSpec] = []

                # in case where the op does not specify input_specs and output_specs
                # is a DTensorSpec, we use output_specs as the spec for each DTensor
                # input arg.
                if output_strategy.input_specs is None:
                    assert isinstance(output_strategy.output_specs, DTensorSpec)

                for idx, input_spec in enumerate(op_schema.args_spec):
                    desired_spec = (
                        output_strategy.output_spec
                        if output_strategy.input_specs is None
                        else output_strategy.input_specs[idx]
                    )
                    expected_input_specs.append(
                        desired_spec.shallow_copy_with_tensor_meta(
                            input_spec.tensor_meta
                        )
                    )
                    if input_spec.placements != desired_spec.placements:
                        needs_redistribute = True

                suggestion_schema = None
                if needs_redistribute:
                    suggestion_schema = OpSchema(
                        op_schema.op, tuple(expected_input_specs), {}
                    )
                    suggestion_schema._inplace_rewrap_schema_suggestion(op_schema)

                # shape and stride args need to be modified for
                # view ops and new factory ops, potentially
                if op_schema.op in self.op_to_shape_and_stride_idx:
                    assert isinstance(output_strategy.output_spec, DTensorSpec)
                    # It happens when the output has the same shape as the input
                    # and the input placements are not all Replicate().
                    if output_strategy.output_spec.is_sharded():
                        schema = suggestion_schema or op_schema
                        assert isinstance(out_tensor_meta, TensorMeta)
                        suggestion_schema = self._adjust_shape_and_stride_args(
                            out_tensor_meta, schema, output_strategy.output_spec
                        )
                        needs_redistribute = True

                # construct output spec for the op
                if op_schema.return_type_tuple_tensor_like():
                    # for ops that return multiple tensors and the output_specs is not
                    # a tuple, we use a tuple of that single output spec as the new
                    # output_specs
                    output_specs: OutputSpecType = output_strategy.output_specs
                    if isinstance(output_specs, DTensorSpec):
                        output_specs = tuple(
                            [
                                # create a new DTensorSpec with the same placement as the
                                # output_specs in output_strategy
                                DTensorSpec(
                                    mesh=output_specs.mesh,
                                    placements=output_specs.placements,
                                    tensor_meta=output_specs.tensor_meta,
                                )
                                for _ in range(len(op_schema.op._schema.returns))
                            ]
                        )
                elif (
                    op_schema.return_type_tensor()
                    or op_schema.return_type_list_tensor_like()
                ):
                    output_specs = output_strategy.output_specs
                else:
                    output_specs = None

                output_sharding = OutputSharding(
                    output_specs,
                    suggestion_schema,
                    needs_redistribute=needs_redistribute,
                )
            elif isinstance(op_strategy, TupleStrategy):
                # tuple strategy output sharding processing
                # runtime select OpSpec for each TupleStrategy input arg
                selected_strategies: list[OpSpec] = []
                out_spec_list: list[DTensorSpec] = []
                for strategy in op_strategy.children:
                    assert isinstance(strategy, OpStrategy)
                    selected_strategy = self._select_strategy(strategy)
                    selected_strategies.append(selected_strategy)
                    out_spec_list.append(selected_strategy.output_spec)

                needs_redistribute = False
                suggestion_args: list[object] = []
                tensor_or_list_tensor_arg_idx = 0

                for arg in op_schema.args_schema:
                    if (
                        arg
                        and isinstance(arg, (list, tuple))
                        and isinstance(arg[0], DTensorSpec)
                    ):
                        expected_input_spec_list: list[DTensorSpec] = []
                        for idx, arg_spec in enumerate(arg):
                            expected_input_spec = selected_strategies[idx].input_spec(
                                tensor_or_list_tensor_arg_idx
                            )
                            expected_input_spec = (
                                expected_input_spec.shallow_copy_with_tensor_meta(
                                    arg_spec.tensor_meta
                                )
                            )
                            if arg_spec.placements != expected_input_spec.placements:
                                needs_redistribute = True
                            expected_input_spec_list.append(expected_input_spec)
                        suggestion_args.append(
                            tuple(expected_input_spec_list)
                            if isinstance(arg, tuple)
                            else expected_input_spec_list
                        )
                        tensor_or_list_tensor_arg_idx += 1

                    elif isinstance(arg, DTensorSpec):
                        expected_input_spec = selected_strategies[0].input_spec(
                            tensor_or_list_tensor_arg_idx
                        )
                        expected_input_spec = (
                            expected_input_spec.shallow_copy_with_tensor_meta(
                                arg.tensor_meta
                            )
                        )
                        if arg.placements != expected_input_spec.placements:
                            needs_redistribute = True
                        suggestion_args.append(expected_input_spec)
                        tensor_or_list_tensor_arg_idx += 1
                    else:
                        suggestion_args.append(arg)

                suggestion_schema = None
                if needs_redistribute:
                    suggestion_schema = OpSchema(
                        op_schema.op, tuple(suggestion_args), op_schema.kwargs_schema
                    )

                output_sharding = OutputSharding(
                    tuple(out_spec_list) if out_tensor_meta is not None else None,
                    suggestion_schema,
                    needs_redistribute=needs_redistribute,
                )
            else:
                raise ValueError("Unsupported op strategy type")
            # associate the output sharding with the output tensor metadata
            self._wrap_output_spec_tensor_meta(
                op_schema.op, output_sharding.output_spec, out_tensor_meta
            )
            return output_sharding


dispatcher = DTensor._op_dispatcher
# change to the customized sharding_propagator for testing implicit fallback
dispatcher.sharding_propagator = CustomShardingPropagator()


class ImplicitRegistrationTest(DTensorTestBase):
    @with_comms
    def test_implicit_registration(self):
        mesh = init_device_mesh(self.device_type, (2, self.world_size // 2))
        test_op = torch.ops.mylib.numpy_sin.default
        input_x = torch.randn([8, 16, 8], device=self.device_type)
        input_y = torch.randn([8, 16, 8], device=self.device_type)
        input_x_dt = distribute_tensor(input_x, mesh, [Shard(0), Shard(1)])
        input_y_dt = distribute_tensor(input_y, mesh, [Shard(1), Shard(0)])
        # 1. test_op strategy not registered test
        with self.assertRaisesRegex(
            NotImplementedError,
            f"Operator {test_op} does not have a sharding strategy registered",
        ):
            self._test_op_on_dtensor(test_op, input_x_dt, input_y_dt)

        # 2. test_op strategy implicitly registered under context manager
        with with_implicit_strategies():
            self._test_op_on_dtensor(test_op, input_x_dt, input_y_dt)

        # 3. remove registration after exiting the context manager
        with self.assertRaisesRegex(
            NotImplementedError,
            f"Operator {test_op} does not have a sharding strategy registered",
        ):
            self._test_op_on_dtensor(test_op, input_x_dt, input_y_dt)


class DimShardingTest(DTensorTestBase):
    @with_comms
    def test_simple_batch_sharding(self):
        # both input tensors batch on dim 0
        mesh = init_device_mesh(self.device_type, (2, self.world_size // 2))
        test_op = torch.ops.mylib.numpy_sin.default

        # 1. strategy that will try shard dim 0 into one devices axis.
        shard_first_dim_strategy = functools.partial(
            batch_shard_strategy, input_shard_dim=[0, 0], output_shard_dim=[0]
        )
        with op_strategy_context(test_op, shard_first_dim_strategy):
            # dim 0 is the batch dim. Here we shard 16 over one device axis
            input_x = torch.randn([16, 1, 4], device=self.device_type)
            input_y = torch.randn([16, 1, 4], device=self.device_type)
            # any sharding below should work
            input_x_dt = distribute_tensor(input_x, mesh, [Shard(1), Replicate()])
            input_y_dt = distribute_tensor(input_y, mesh, [Replicate(), Shard(0)])
            self._test_op_on_dtensor(test_op, input_x_dt, input_y_dt)

        # 2. strategy that will try shard dim 0 into multiple devices.
        shard_first_dim_to_multiple_devices_strategy = functools.partial(
            batch_shard_strategy,
            input_shard_dim=[0, 0],
            output_shard_dim=[0],
            enable_shard_batch_dim_over_multiple_axis=True,
        )
        with op_strategy_context(test_op, shard_first_dim_to_multiple_devices_strategy):
            # dim 0 is the batch dim. Here we potentially shard 16 over multiple device axes
            input_x = torch.randn([16, 1, 4], device=self.device_type)
            input_y = torch.randn([16, 1, 4], device=self.device_type)
            # any sharding below should work
            input_x_dt = distribute_tensor(input_x, mesh, [Shard(1), Replicate()])
            input_y_dt = distribute_tensor(input_y, mesh, [Replicate(), Shard(0)])
            self._test_op_on_dtensor(test_op, input_x_dt, input_y_dt)

    @with_comms
    def test_broadcast_batch_sharding(self):
        # Not recommended, user need to make sure the op supports input with
        # broadcast first. If not supported, try unsqueeze inputs first to match
        # each other's dimensions and and use the example in the
        # test_simple_batch_sharding test.
        mesh = init_device_mesh(self.device_type, (2, self.world_size // 2))
        test_op = torch.ops.mylib.numpy_sin.default
        shard_on_first_dim_strategy = functools.partial(
            batch_shard_strategy, input_shard_dim=[None, 0], output_shard_dim=[0]
        )
        with op_strategy_context(test_op, shard_on_first_dim_strategy):
            input_x = torch.randn([1, 4], device=self.device_type)
            # input_y's 16 locates on the batch dim
            input_y = torch.randn([16, 1, 4], device=self.device_type)
            # any sharding below should work as long as the tensor dim it is shardable
            input_x_dt = distribute_tensor(input_x, mesh, [Shard(1), Replicate()])
            input_y_dt = distribute_tensor(input_y, mesh, [Replicate(), Shard(0)])

            output_dt = test_op(input_x_dt, input_y_dt)

            # split the batch dim to test correctness
            input_y_chucks = torch.chunk(input_y, 4, dim=0)
            output = torch.cat(
                [test_op(input_x, input_y_part) for input_y_part in input_y_chucks]
            )
            self.assertEqual(output_dt.full_tensor(), output)

            # or we can test directly since the op support broadcast.
            self._test_op_on_dtensor(test_op, input_x_dt, input_y_dt)

    @with_comms
    def test_simple_tuple_batch_sharding(self):
        # both input tensors batch on dim 0
        mesh = init_device_mesh(self.device_type, (2, self.world_size // 2))
        test_op = torch.ops.mylib.numpy_tuple_sin.default

        # 1. strategy that will try shard dim 0 into one devices axis.
        shard_first_dim_to_multiple_devices_strategy = functools.partial(
            batch_shard_strategy,
            input_shard_dim=[0, 0, 0, 0, 0],  # flatten input_y
            output_shard_dim=[0],
            enable_shard_batch_dim_over_multiple_axis=True,
        )
        with op_strategy_context(
            test_op,
            shard_first_dim_to_multiple_devices_strategy,
            schema_info=RuntimeSchemaInfo(needs_pytree=True),
        ):
            # dim 0 is the batch dim. Here we shard 16 over one device axis
            input_x = torch.randn([16, 8, 4], device=self.device_type)
            input_y = [
                torch.randn([16, 8, 4], device=self.device_type) for _ in range(3)
            ]
            input_z = torch.randn([16, 8, 4], device=self.device_type)
            # any sharding below should work as long as the tensor dim it is shardable
            input_x_dt = distribute_tensor(input_x, mesh, [Shard(0), Shard(1)])
            input_y_dt = [
                distribute_tensor(i, mesh, [Shard(1), Shard(1)]) for i in input_y
            ]
            input_z_dt = distribute_tensor(input_z, mesh, [Shard(1), Shard(0)])
            self._test_op_on_dtensor(test_op, input_x_dt, input_y_dt, input_z_dt)


if __name__ == "__main__":
    run_tests()
