# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import copy
import functools
import logging
import time
from contextlib import ExitStack
from typing import Any, Callable, Optional, Union

import torch
from torch._dynamo.functional_export import _dynamo_graph_capture_for_export
from torch._functorch.aot_autograd import (
    aot_compile_joint_with_descriptors,
    aot_export_joint_with_descriptors,
    boxed_nop_preserve_node_meta,
)
from torch._inductor.compile_fx import compile_fx_inner
from torch._logging import trace_structured
from torch._subclasses import FakeTensorMode
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DeviceMesh
from torch.export._trace import _restore_state_dict
from torch.export.unflatten import _AttrKind
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from .apply_sharding import apply_sharding_to_model
from .cast_parametrization import (
    DTypeCastModule,
    apply_dtype_cast,
    canonicalize_mp,
    set_dtype_cast,
)
from .graph_passes.activation_checkpointing import ac_joint_pass
from .graph_passes.graph_utils import (
    _add_alias,
    _replace_view_mm_view_with_einsum,
    assert_has_no_collectives,
    cleanup_graph,
    update_joint_with_descriptors,
)
from .init_weights import wrap_init_weights
from .input_validation import (
    _check_forward_args,
    _compute_expected_inputs,
    _extract_input_info,
    _flatten_out_shardings,
    _make_input_fn,
)
from .module_construction import (
    _NN_MODULE_KEYS,
    _assign_attr,
    _build_alias_map,
    _build_module_alias_map,
)
from .optimize_sharding import ShardingOptimizer
from .shardings.placement_options import (
    NumericsLogger,
    _get_device_from_mesh,
    debug_boxed_nop_preserve_node_meta,
)
from .tracing import (
    _add_unused_params_and_buffers,
    _get_decomp_table,
    enable_local_map_wrapping,
    move_to_fake,
)

_APPLY_VIEW_MM_VIEW_PATTERN = False

logger = logging.getLogger(__name__)


class AutoParallel:
    """
    Args:
        mesh: Defines placement options.
        The meta model is moved to a fake device based on mesh.device_type.
    """

    def __init__(
        self,
        model,
        input_fn,
        mesh: DeviceMesh,
        mp_policy: Optional[MixedPrecisionPolicy] = None,
        compile: bool = False,
        enable_ac: bool = True,
        # None means 'auto'
        ac_stage_size_in_GiB: Optional[Union[float, str]] = "auto",
        reshard_after_forward: bool = True,
        dynamic: bool = False,
        numerics_logger: NumericsLogger | None = None,
        **kwargs,
    ):
        self.stack = ExitStack()
        self.fake_mode = (
            FakeTensorMode()
        )  # TODO: maybe need to reuse the model's fake mode
        # self.fake_mode.allow_scalar_outputs = True
        device = _get_device_from_mesh(mesh)
        if mp_policy is not None:
            mp_policy = canonicalize_mp(mp_policy)
        self.mp_policy = mp_policy
        self.cost_model = kwargs.pop("cost_model", None)
        self.kwargs = kwargs
        # copy user model to avoid modifying it in-place
        # in dtype casting and move_to_fake
        model = copy.deepcopy(model)

        # Capture parameter and buffer alias info before move_to_fake breaks
        # aliasing. named_parameters()/named_buffers() deduplicate by default,
        # so aliases are dropped. We record alias_fqn -> canonical_fqn so we
        # can re-register them later.
        self._param_alias_map = _build_alias_map(model.named_parameters)
        self._buffer_alias_map = _build_alias_map(model.named_buffers)
        self._module_alias_map = _build_module_alias_map(model)

        if self.mp_policy is not None:
            apply_dtype_cast(model, self.mp_policy)

        self.model = move_to_fake(model, self.fake_mode, device)
        self.input_fn = input_fn
        self.mesh = mesh
        if compile:
            self.compiler_fn = compile_fx_inner
        elif numerics_logger:
            self.compiler_fn = functools.partial(
                debug_boxed_nop_preserve_node_meta, numerics_logger=numerics_logger
            )
        else:
            self.compiler_fn = boxed_nop_preserve_node_meta  # type: ignore[assignment]
        self.enable_ac = enable_ac
        self.ac_stage_size_in_GiB = ac_stage_size_in_GiB
        self.reshard_after_forward = reshard_after_forward

        if dynamic:
            self.fake_mode.shape_env = ShapeEnv()
            self.fake_mode.static_shapes = False

        # NB: rest of the construction happens in __enter__
        self.active = False

    def __enter__(self):
        assert self.active is False

        # build_model_graph and the code below push context managers
        # (including FakeTensorMode) onto self.stack via
        # aot_export_joint_with_descriptors. If anything raises, __exit__
        # won't be called (Python only calls __exit__ if __enter__
        # succeeds), so we must unwind the stack ourselves.
        try:
            from .cost_models.collective_runtime_estimation import (
                get_nccl_topo_config,
                set_nccl_topo_config,
            )
            from .cost_models.nccl_cost_model import (
                NCCLTopoConfig,
                detect_nccl_topo_config,
            )

            self._prev_nccl_config = get_nccl_topo_config()
            if isinstance(self.cost_model, NCCLTopoConfig):
                set_nccl_topo_config(self.cost_model)
            elif self.cost_model == "nccl":
                set_nccl_topo_config(detect_nccl_topo_config(self.mesh))
            else:
                set_nccl_topo_config(None)

            self.build_model_graph()
            self.old_inductor_comprehensive_padding = (
                torch._inductor.config.comprehensive_padding
            )
            torch._inductor.config.comprehensive_padding = False

            rescale_grad_comm_cost_for_mp = 1.0
            if self.mp_policy is not None:
                param_size = self.mp_policy.param_dtype.itemsize
                reduce_size = self.mp_policy.reduce_dtype.itemsize
                if param_size != reduce_size:
                    rescale_grad_comm_cost_for_mp = reduce_size / param_size
                    # Tiebreak, favoring performing the comms in the largest
                    # dtype
                    rescale_grad_comm_cost_for_mp *= 1.1
            sharding_optimizer = ShardingOptimizer(
                self.gm,
                self.mesh,
                rescale_grad_comm_cost_for_mp,
                repeated_subgraphs=self.kwargs.get("repeated_subgraphs", False),
            )

            self.sharding_optimizer = sharding_optimizer

            self.input_constraints = None
            self.output_constraints = None

            self.active = True

            self.stack.__enter__()
        except BaseException:
            self.stack.__exit__(None, None, None)
            raise

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        from .cost_models.collective_runtime_estimation import set_nccl_topo_config

        set_nccl_topo_config(self._prev_nccl_config)
        torch._inductor.config.comprehensive_padding = (
            self.old_inductor_comprehensive_padding
        )
        self.active = None
        return self.stack.__exit__(exc_type, exc_val, exc_tb)

    def _assert_entered(self):
        if self.active is False:
            raise RuntimeError(
                "You must use AutoParallel as a context manager: with AutoParallel() as p: ..."
            )
        if self.active is None:
            raise RuntimeError(
                "AutoParallel is not reentrant, please file a bug report if you need this functionality"
            )

    def build_model_graph(self):
        t0 = time.perf_counter()
        decomp_table = _get_decomp_table()

        with self.fake_mode:
            raw_inputs = self.input_fn()

        formatted_inputs = (
            raw_inputs if isinstance(raw_inputs, tuple) else (raw_inputs,)
        )

        # Capture traced inputs for runtime validation in forward().
        self._traced_inputs = list(formatted_inputs)

        with set_dtype_cast(
            True
        ), enable_local_map_wrapping(), torch._dynamo.utils._disable_saved_tensors_hooks_during_tracing():
            torch_ir_with_fqn = _dynamo_graph_capture_for_export(self.model)(
                *formatted_inputs
            )
            _restore_state_dict(self.model, torch_ir_with_fqn)
            _add_unused_params_and_buffers(self.model, torch_ir_with_fqn)
            # TODO Can't use fake mode here because it clashes with the user level
            # fake mode. Ideally dynamo should reuse the user level fake mode.
            self.joint_with_descriptors = aot_export_joint_with_descriptors(
                self.stack,
                torch_ir_with_fqn,
                formatted_inputs,
                decompositions=decomp_table,
            )
        gm = self.joint_with_descriptors.graph_module
        assert_has_no_collectives(gm)

        cleanup_graph(gm)
        if _APPLY_VIEW_MM_VIEW_PATTERN:
            _replace_view_mm_view_with_einsum(gm)
        # now add aliases nodes to the graph to
        # give more room for optimizations
        _add_alias(gm, version="v2")
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "autoparallel_joint_graph",
                "encoding": "string",
            },
            payload_fn=lambda: gm.print_readable(
                print_output=False, include_stride=True, include_device=True
            ),
        )

        self.gm = gm
        logger.info("Graph tracing took %.3fs", time.perf_counter() - t0)

    # TODO: Specify what the low/high meaning is (percentage?)
    def add_parameter_memory_constraint(self, low=None, high=None):
        self._assert_entered()

        # by default, divide the parameters by the world size
        if low is None:
            low = 0.0
        if high is None:
            high = 1.0 / self.mesh.size()

        assert low <= high, f"low should be <= high, got low{low}, high={high}"

        self.sharding_optimizer.add_parameter_memory_constraint(low, high)

    def add_input_constraints(self, constraints):
        self._assert_entered()

        assert self.input_constraints is None, "Input constraints have already been set"
        self.sharding_optimizer.add_sharded_input_constraint(constraints)
        self.input_constraints = constraints

    def add_output_constraints(self, constraints):
        self._assert_entered()

        assert (
            self.output_constraints is None
        ), "Output constraints have already been set"
        # forces sharding of fwd output to be S(0) on first dimension and R on others
        self.sharding_optimizer.add_sharded_output_constraint(constraints)
        self.output_constraints = constraints

    def optimize_placement(self, verbose=True):
        self._assert_entered()

        self.sharding_placement = self.sharding_optimizer.get_solution(verbose=False)

        if verbose:
            print(self.sharding_optimizer.get_log(verbose=True))

        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "autoparallel_sharding_optimizer_log",
                "encoding": "string",
            },
            payload_fn=lambda: self.sharding_optimizer.get_log(colored=False),
        )

        if self.sharding_optimizer.prob.status == -1:
            raise RuntimeError("Didn't find solution")

        return self.sharding_placement

    def _apply_placement_common(self, sharding_placement):
        t0 = time.perf_counter()
        self._assert_entered()

        if sharding_placement is None:
            sharding_placement = self.sharding_placement
        # TODO: what kind of updates do we have to do?
        #  - graph obvs
        #  - flat_args / updated_flat_args
        # OTHER THINGS
        #  - subclass_meta
        #  - wrappers
        #    - contains another instance of subclass info in self
        #    - quite a lot of use of runtime_metadata
        #
        from torch._subclasses.fake_tensor import unset_fake_temporarily

        with unset_fake_temporarily():
            # creates a new mesh and caches it internally
            # we don't need to keep a reference to it
            # TODO: remove ndim == 1 special case once
            # DeviceMesh._flatten is fixed
            mesh = self.mesh
            if mesh.ndim != 1:
                mesh._flatten()
        with self.fake_mode:
            (
                parallel_gm,
                sharded_param_dict,
                sharded_buffer_dict,
            ) = apply_sharding_to_model(
                self.gm,
                sharding_placement,
                self.joint_with_descriptors.params_spec,
                self.joint_with_descriptors.buffers_spec,
            )
        t_apply = time.perf_counter()
        # clean it up by removing the added aliases from previous pass
        # as well as redundant views
        cleanup_graph(parallel_gm, aggressive=True)
        t_cleanup = time.perf_counter()

        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "autoparallel_parallel_graph",
                "encoding": "string",
            },
            payload_fn=lambda: parallel_gm.print_readable(
                print_output=False, include_stride=True, include_device=True
            ),
        )
        t_trace = time.perf_counter()

        if self.enable_ac:
            ac_joint_pass(
                parallel_gm.graph, self.ac_stage_size_in_GiB, self.reshard_after_forward
            )
        t_ac = time.perf_counter()
        # now rename input/param/tangent/output/grad_param/grad_input nodes following
        # our convention
        # apply_node_renaming(
        #    parallel_gm, self.params_len, self.buffer_len, self.metadata
        # )
        self.parallel_gm = parallel_gm
        update_joint_with_descriptors(self.joint_with_descriptors, parallel_gm)
        # NB: so this function takes in the parameters at the beginning

        # let's remove those otherwise we can't clean the backward graph properly
        # NB: This is VERY important for good memory use!
        # TODO: This is VERY VERY NAUGHTY, need to do this in a scoped way
        if (
            torch.ops._c10d_functional.wait_tensor
            in torch.fx.node._side_effectful_functions
        ):
            torch.fx.node._side_effectful_functions.remove(
                torch.ops._c10d_functional.wait_tensor
            )
        if (
            torch.ops._c10d_functional.wait_tensor.default
            in torch.fx.node._side_effectful_functions
        ):
            torch.fx.node._side_effectful_functions.remove(
                torch.ops._c10d_functional.wait_tensor.default
            )
        logger.info(
            "Apply placements took %.3fs "
            "(apply_sharding=%.3fs, cleanup=%.3fs, trace=%.3fs, ac=%.3fs)",
            time.perf_counter() - t0,
            t_apply - t0,
            t_cleanup - t_apply,
            t_trace - t_cleanup,
            t_ac - t_trace,
        )
        return (
            sharded_param_dict,
            sharded_buffer_dict,
        )

    def _register_params_and_init_weights(
        self, sharded_param_dict, sharded_buffer_dict
    ):

        # We construct an unflattened structure on parallel_mod,
        # e.g. _assign_attr(v, parallel_model, k="layers.0.weight") will literally
        # create empty nn.Modules recursively and then stash 'v' so it shows up in the right spot
        # We pass self.model as reference to preserve the original module structure (e.g., nn.ModuleDict)
        for k, v in sharded_param_dict.items():
            _assign_attr(
                v,
                self.parallel_model,
                self.model,
                k,
                attr_kind=_AttrKind.PARAMETER,
            )

        for k, v in sharded_buffer_dict.items():
            _assign_attr(
                v,
                self.parallel_model,
                self.model,
                k,
                attr_kind=_AttrKind.BUFFER,
            )

        # Register aliased params/buffers that were deduplicated during tracing.
        # e.g. if the original model has rope.cache and freqs_cis pointing to
        # the same tensor, only one survives in the sharded dict. We register
        # the missing alias so the parallel model mirrors the original structure.
        # Note: the alias map's "canonical" may not match which FQN the tracer
        # kept, so we check both directions.
        for alias_fqn, canonical_fqn in self._param_alias_map.items():
            if canonical_fqn in sharded_param_dict:
                _assign_attr(
                    self.parallel_model.get_parameter(canonical_fqn),
                    self.parallel_model,
                    self.model,
                    alias_fqn,
                    attr_kind=_AttrKind.PARAMETER,
                )
            elif alias_fqn in sharded_param_dict:
                _assign_attr(
                    self.parallel_model.get_parameter(alias_fqn),
                    self.parallel_model,
                    self.model,
                    canonical_fqn,
                    attr_kind=_AttrKind.PARAMETER,
                )
        for alias_fqn, canonical_fqn in self._buffer_alias_map.items():
            if canonical_fqn in sharded_buffer_dict:
                _assign_attr(
                    self.parallel_model.get_buffer(canonical_fqn),
                    self.parallel_model,
                    self.model,
                    alias_fqn,
                    attr_kind=_AttrKind.BUFFER,
                )
            elif alias_fqn in sharded_buffer_dict:
                _assign_attr(
                    self.parallel_model.get_buffer(alias_fqn),
                    self.parallel_model,
                    self.model,
                    canonical_fqn,
                    attr_kind=_AttrKind.BUFFER,
                )

        # Re-establish module aliases (e.g. model_ema -> teacher) so that
        # both FQNs point to the same submodule on the parallel model.
        for alias_fqn, canonical_fqn in self._module_alias_map.items():
            # Find the canonical module on the parallel model
            canonical_mod = self.parallel_model
            for attr in canonical_fqn.split("."):
                canonical_mod = getattr(canonical_mod, attr, None)
                if canonical_mod is None:
                    break
            if canonical_mod is None:
                continue
            # Navigate to the alias's parent and re-establish the alias
            *alias_prefix, alias_field = alias_fqn.split(".")
            alias_parent = self.parallel_model
            for attr in alias_prefix:
                alias_parent = getattr(alias_parent, attr, None)
                if alias_parent is None:
                    break
            if alias_parent is not None:
                setattr(alias_parent, alias_field, canonical_mod)

        # Wrap init_weights (inherited from UserModelClass) so that parameter/buffer
        # assignments and in-place data operations are intercepted and correctly
        # applied to the sharded DTensor parameters.
        wrap_init_weights(self.parallel_model)

    def apply_placement(self, sharding_placement=None):

        sharded_param_dict, sharded_buffer_dict = self._apply_placement_common(
            sharding_placement
        )

        self.parallel_model_fn = parallel_model_fn = aot_compile_joint_with_descriptors(
            self.joint_with_descriptors,
            fw_compiler=self.compiler_fn,
            bw_compiler=self.compiler_fn,
        )

        # TODO: this probably belongs in the AOTAutograd API
        # TODO: pytree handling
        # Capture the exact FQNs the compiled graph expects as primals.
        # This avoids issues with aliased params/buffers where identity-based
        # dedup can break after init_weights reassigns tensors.
        graph_param_fqns = list(self.joint_with_descriptors.params_spec)
        graph_buffer_fqns = list(self.joint_with_descriptors.buffers_spec)

        expected_inputs = _compute_expected_inputs(
            self._traced_inputs, self.input_constraints, self.mesh
        )

        # apply_dtype_cast swaps self.model.__class__ to a dynamic subclass
        # with property descriptors for each parameter (e.g. 'weight'). We need
        # the original user class so those properties don't conflict with
        # register_parameter in _register_params_and_init_weights.
        UserModelClass = type(self.model)
        if issubclass(UserModelClass, DTypeCastModule):
            # DTypeCast bases are (DTypeCastModule, OriginalClass)
            UserModelClass = UserModelClass.__bases__[1]

        class AutoParallelModule(UserModelClass):
            def __init__(self):
                torch.nn.Module.__init__(self)

            def forward(self, *args):
                _check_forward_args(args, expected_inputs)
                # NB: don't close over the parameters/buffers, as the user may
                # reassign the module!
                # Use the exact param/buffer FQNs that the compiled graph
                # expects, matching the primals order from tracing.
                params = [
                    self.get_parameter(fqn).to_local() for fqn in graph_param_fqns
                ] + [self.get_buffer(fqn).to_local() for fqn in graph_buffer_fqns]
                boxed_args = [*params, *args]
                del params
                # NB: don't do self.parallel_model_fn work around Dynamo bug
                out = parallel_model_fn(boxed_args)
                return out

        self.parallel_model = AutoParallelModule()
        # Copy user-defined instance attributes (e.g. self.dim, self.config)
        # from the original model. Uses __dict__ directly to avoid triggering
        # nn.Module.__setattr__ which intercepts nn.Parameter/nn.Module assignments.
        for k, v in self.model.__dict__.items():
            if k not in _NN_MODULE_KEYS:
                self.parallel_model.__dict__[k] = v
        self._register_params_and_init_weights(sharded_param_dict, sharded_buffer_dict)
        # Copy submodules that don't appear in any parameter/buffer FQN path
        # (e.g. self.rope when it has no traced params/buffers). These aren't
        # created by _assign_attr, but init_weights may need to access them.
        for k, v in self.model._modules.items():
            if k not in self.parallel_model._modules:
                self.parallel_model._modules[k] = v
        return self.parallel_model


####################
# Simple API start #
####################


def auto_parallel(
    model: torch.nn.Module,
    mesh: DeviceMesh,
    sample_inputs: Union[Any, Callable[[], Any]],
    out_shardings: Any,
    *,
    mp_policy: Optional[MixedPrecisionPolicy] = None,
    compile: bool = True,
    parameter_memory_budget: Optional[tuple[Optional[float], Optional[float]]] = None,
) -> torch.nn.Module:
    """
    Parallelize a model with automatic sharding optimization.

    This is a simplified API that wraps the full AutoParallel context manager.
    For more control, use the AutoParallel class directly.

    Args:
        model: Model to parallelize. Can be on meta device for large models.
        mesh: Device mesh defining the distributed topology.
        sample_inputs: Sample inputs for tracing. Supports pytrees (tuples, dicts,
            nested structures). Leaves can be:
            - DTensor: Sharding extracted from placements
            - Tensor: Assumed Replicate on all mesh dimensions
            Can also be a callable that returns the above.
        out_shardings: Output sharding specification as a pytree matching the
            model output structure. Each leaf should be a tuple of Placements.
            For a single output, can be just the placement tuple.
            Examples:
                - Single output: (Shard(0), Replicate())
                - Tuple output: ((Shard(0),), (Shard(0),))
                - Dict output: {"logits": (Shard(0),), "loss": (Replicate(),)}
        mp_policy: Optional mixed precision policy.
        compile: Whether to use torch.compile (default: True).
        parameter_memory_budget: Optional (low, high) bounds for parameter memory.
            Each bound is a float multiplier or None for unbounded.

    Returns:
        Parallelized module. Call to_empty(device="cuda") and init_weights()
        before use.

    Example with DTensor:
        >>> mesh = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
        >>> x = DTensor.from_local(
        ...     torch.rand(local_bs, seq_len, dim),
        ...     mesh,
        ...     [Shard(0), Replicate()],
        ... )
        >>> parallel_model = auto_parallel(
        ...     model, mesh,
        ...     sample_inputs=(x,),
        ...     out_shardings=(Shard(0), Replicate()),
        ...     mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
        ... )
        >>> parallel_model.to_empty(device="cuda")
        >>> parallel_model.init_weights()

    Example with dict inputs:
        >>> sample_inputs = {
        ...     "input_ids": DTensor.from_local(ids, mesh, [Shard(0)]),
        ...     "attention_mask": DTensor.from_local(mask, mesh, [Shard(0)]),
        ... }
        >>> parallel_model = auto_parallel(model, mesh, sample_inputs, out_shardings=...)
    """
    # Handle callable sample_inputs
    if callable(sample_inputs):
        raw_inputs = sample_inputs()
    else:
        raw_inputs = sample_inputs

    # Extract metadata and placements (does not materialize tensors)
    shapes, dtypes, input_placements, treespec = _extract_input_info(raw_inputs, mesh)

    # Flatten out_shardings to list
    output_placements = _flatten_out_shardings(out_shardings)

    # Create input_fn that will be called inside FakeTensorMode
    # It creates fresh tensors (which become fake tensors inside FakeTensorMode)
    input_fn = _make_input_fn(shapes, dtypes, treespec)

    # Use AutoParallel context manager
    with AutoParallel(
        model,
        input_fn,
        mesh,
        mp_policy=mp_policy,
        compile=compile,
        # enable_ac=True,
        enable_ac=False,
    ) as autop:
        # Add constraints
        autop.add_input_constraints(input_placements)
        if parameter_memory_budget is not None:
            autop.add_parameter_memory_constraint(
                low=parameter_memory_budget[0], high=parameter_memory_budget[1]
            )
        autop.add_output_constraints(output_placements)

        # Optimize and apply
        sharding_placement = autop.optimize_placement(verbose=False)
        parallel_model = autop.apply_placement(sharding_placement)

    return parallel_model


##################
# Simple API end #
##################
