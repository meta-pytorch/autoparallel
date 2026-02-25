# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import copy
import functools
import itertools
from contextlib import ExitStack, contextmanager
from types import MethodType
from typing import Any, Callable, Optional, Union

import torch
from torch._dynamo.functional_export import _dynamo_graph_capture_for_export
from torch._functorch.aot_autograd import (
    aot_compile_joint_with_descriptors,
    aot_export_joint_with_descriptors,
    boxed_nop_preserve_node_meta,
)
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.decomposition import select_decomp_table
from torch._logging import trace_structured
from torch._subclasses import FakeTensorMode
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DeviceMesh
from torch.export._trace import _restore_state_dict
from torch.export.unflatten import _AttrKind
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from .apply_sharding import apply_sharding_to_model
from .cast_parametrization import apply_dtype_cast, canonicalize_mp, set_dtype_cast
from .graph_passes.activation_checkpointing import ac_joint_pass
from .graph_passes.graph_utils import (
    _add_alias,
    _replace_view_mm_view_with_einsum,
    assert_has_no_collectives,
    cleanup_graph,
    update_joint_with_descriptors,
)
from .init_weights import hook_params_setters
from .optimize_sharding import ShardingOptimizer
from .shardings.placement_options import (
    NumericsLogger,
    _get_device_from_mesh,
    debug_boxed_nop_preserve_node_meta,
)

_APPLY_VIEW_MM_VIEW_PATTERN = False


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


def _get_decomp_table():
    decomp_table = copy.copy(select_decomp_table())
    # TODO: removing those as they cause missing DTensor propagation rules
    decomp_table.pop(torch.ops.aten.full_like.default)
    decomp_table.pop(torch.ops.aten.empty_like.default)
    decomp_table.pop(torch.ops.aten.threshold_backward.default)
    decomp_table.pop(torch.ops.aten.native_layer_norm.default)
    decomp_table.pop(torch.ops.aten.embedding_dense_backward.default)
    decomp_table.pop(torch.ops.aten.native_layer_norm_backward.default)
    decomp_table.pop(torch.ops.aten._softmax_backward_data.default)
    decomp_table.pop(torch.ops.aten._softmax.default)
    decomp_table.pop(torch.ops.aten.stack.default)

    # decompose addmm to allow for TP on mm
    decomp_table.pop(torch.ops.aten.addmm.default)

    def addmm_decomp(self, mat1, mat2, beta=1, alpha=1):
        return self + mat1 @ mat2

    decomp_table[torch.ops.aten.addmm.default] = addmm_decomp
    # decomp_table = None

    return decomp_table


def move_to_fake(model: torch.nn.Module, mode: FakeTensorMode, device: torch.device):
    """
    Move the model to the fake mode and move the weights to the fake device
    """

    def assert_is_meta_tensor(name, t):
        assert isinstance(t, torch.Tensor) and t.device == torch.device(
            "meta"
        ), f"tensor {name} must be on meta device, not {t.device}"

    def _move_to_fake(module, k, device, parameter=True):
        # lots of ways you might try to swap params with fake params do not work, but this one does
        submod = module
        while len(k.split(".")) > 1:
            submod_name, k = k.split(".", 1)
            submod = getattr(submod, submod_name)

        fake_tensor = mode.from_tensor(getattr(submod, k)).to(device)
        if parameter:
            fake_tensor = torch.nn.Parameter(
                fake_tensor, requires_grad=fake_tensor.requires_grad
            )

        setattr(submod, k, fake_tensor)

    with mode:
        for k, p in model.named_parameters():
            assert_is_meta_tensor(k, p)
            _move_to_fake(model, k, device, parameter=True)
        for k, b in model.named_buffers():
            assert_is_meta_tensor(k, b)
            _move_to_fake(model, k, device, parameter=False)

    return model


@contextmanager
def enable_local_map_wrapping():
    from torch._dynamo.variables.higher_order_ops import (
        LocalMapWrappedHigherOrderVariable as vt_cls,
    )
    from torch._higher_order_ops import local_map as local_map_module

    with vt_cls.enable(), local_map_module.defer_inlining():
        yield


def _export(
    model: torch.nn.Module, model_wrapper: Callable, inputs: tuple[Any, ...]
) -> torch.fx.GraphModule:
    """
    Capture a model graph via Dynamo and restore parameter/buffer metadata.

    We need both `model` and `model_wrapper` because:
    - `model_wrapper` is the actual callable that gets traced by Dynamo. It may wrap
      the model with additional logic (e.g., adding a loss function on top of the model's
      forward pass, or preparing inputs in a specific way).
    - `model` is the original nn.Module needed to restore the correct fully-qualified
      names (FQNs) for parameters and buffers in the traced graph. Without this, the
      captured graph would lose the original parameter naming structure.

    Args:
        model: Original nn.Module with parameter/buffer metadata to restore
        model_wrapper: Callable to trace (may wrap model with additional logic)
        inputs: Input tensors for tracing

    Returns:
        GraphModule with restored parameter FQNs and calling convention

    TODO:
    1) Use bytecode for calling convention instead of pytree for more seamless UX
    2) Attach guards
    3) Be more careful about tensor constants names
    """
    with torch._dynamo.config.patch(install_free_tensors=True):
        gm = _dynamo_graph_capture_for_export(model_wrapper)(*inputs)
        _restore_state_dict(model, gm)
        return gm


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

        # keep a separate copy of the fake orig model to customize for supporting init_weights
        self.init_weights_model = move_to_fake(
            copy.deepcopy(model), self.fake_mode, device
        )

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

        # makes sharding of params and gradients the same
        sharding_optimizer.add_grad_param_constraints()
        self.sharding_optimizer = sharding_optimizer

        self.input_constraints = None
        self.output_constraints = None

        self.active = True

        self.stack.__enter__()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
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

    def _prepare_model_wrapper_and_inputs(
        self, raw_inputs: Any
    ) -> tuple[Callable, tuple[Any, ...]]:
        """
        Prepare the model wrapper and formatted inputs for tracing.

        Args:
            raw_inputs: The raw inputs from input_fn()

        Returns:
            A tuple of (model_wrapper, formatted_inputs) where:
            - model_wrapper is a callable that will be traced
            - formatted_inputs are the inputs to pass to model_wrapper
        """
        # No loss function, inputs are just model inputs
        formatted_inputs = (
            raw_inputs if isinstance(raw_inputs, tuple) else (raw_inputs,)
        )

        def model_wrapper(*model_inputs) -> Any:
            output = self.model(*model_inputs)
            return output

        return model_wrapper, formatted_inputs

    def build_model_graph(self):
        decomp_table = _get_decomp_table()

        with self.fake_mode:
            raw_inputs = self.input_fn()

        # Prepare model wrapper and inputs for tracing
        model_wrapper, formatted_inputs = self._prepare_model_wrapper_and_inputs(
            raw_inputs
        )

        with set_dtype_cast(
            True
        ), enable_local_map_wrapping(), torch._dynamo.utils._disable_saved_tensors_hooks_during_tracing():
            torch_ir_with_fqn = _export(self.model, model_wrapper, formatted_inputs)
            # TODO Cna't use fake mode here because it clashes with the user level
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

        if self.input_constraints is None:
            # forces sharding of input to be S(0) on first dimension and R on others
            self.add_input_constraints(None)

        if self.output_constraints is None:
            # forces sharding of fwd output to be S(0) on first dimension and R on others
            self.add_output_constraints(None)

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
        # clean it up by removing the added aliases from previous pass
        # as well as redundant views
        cleanup_graph(parallel_gm, aggressive=True)

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

        if self.enable_ac:
            ac_joint_pass(
                parallel_gm.graph, self.ac_stage_size_in_GiB, self.reshard_after_forward
            )
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
        for alias_fqn, canonical_fqn in self._param_alias_map.items():
            if canonical_fqn in sharded_param_dict:
                _assign_attr(
                    self.parallel_model.get_parameter(canonical_fqn),
                    self.parallel_model,
                    self.model,
                    alias_fqn,
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

        # Right now we require a convention that the user model provides an init_weights method,
        # although we could snoop for other methods too.
        hook_params_setters(self.init_weights_model, self.parallel_model)
        if hasattr(self.model, "init_weights"):

            def init_weights(_self, *args, **kwargs):
                # this is now a deep-fake-copy of orig mod, so we don't have to use reparametrize
                return self.init_weights_model.init_weights(*args, **kwargs)

            # assign an init_weights method onto the output mod.
            # all it does is sneakily run the original user mod's init_weights method,
            # but with our new DTensor sharded params attached to the user module.
            self.parallel_model.init_weights = MethodType(
                init_weights, self.parallel_model
            )

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
        class AutoParallelModule(torch.nn.Module):
            def forward(self, *args):
                # NB: don't close over the parameters/buffers, as the user may
                # reassign the module!
                # TODO: It's this to just exactly match
                # prepare_aot_module_simplified, this seems like an API gap
                params = [
                    v.to_local()
                    for k, v in
                    # TODO: this is very slow
                    itertools.chain(
                        dict(self.named_parameters(remove_duplicate=False)).items(),
                        dict(self.named_buffers(remove_duplicate=False)).items(),
                    )
                ]
                boxed_args = [*params, *args]
                del params
                # NB: don't do self.parallel_model_fn work around Dynamo bug
                out = parallel_model_fn(boxed_args)
                return out

        self.parallel_model = AutoParallelModule()
        self._register_params_and_init_weights(sharded_param_dict, sharded_buffer_dict)
        return self.parallel_model


####################
# Simple API start #
####################


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


def auto_parallel(
    model: torch.nn.Module,
    mesh: DeviceMesh,
    sample_inputs: Union[Any, Callable[[], Any]],
    out_shardings: Any,
    *,
    mp_policy: Optional[MixedPrecisionPolicy] = None,
    compile: bool = True,
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
        # autop.add_parameter_memory_constraint(low=None, high=None)
        autop.add_input_constraints(input_placements)
        autop.add_output_constraints(output_placements)

        # Optimize and apply
        sharding_placement = autop.optimize_placement(verbose=False)
        parallel_model = autop.apply_placement(sharding_placement)

    return parallel_model


##################
# Simple API end #
##################
