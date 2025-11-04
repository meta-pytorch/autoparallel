# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import copy
import itertools
import warnings
from contextlib import ExitStack, contextmanager
from types import MethodType
from typing import Any, Optional, Union

import torch
from torch._dynamo.functional_export import dynamo_graph_capture_for_export
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
from torch.export._unlift import _assign_attr
from torch.export.unflatten import _AttrKind
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from autoparallel._passes.graph_partition import partition_joint_with_descriptors

from .activation_checkpointing import ac_joint_pass
from .apply_sharding import apply_sharding_to_model
from .cast_parametrization import apply_dtype_cast, canonicalize_mp, set_dtype_cast
from .graph_utils import (
    _add_alias,
    _replace_view_mm_view_with_einsum,
    assert_has_no_collectives,
    cleanup_graph,
    update_joint_with_descriptors,
)
from .init_weights import hook_params_setters
from .optimize_sharding import ShardingOptimizer
from .utils import _get_device_from_mesh

_APPLY_VIEW_MM_VIEW_PATTERN = False


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


# Export runs some asserts on the exported program to ensure that it is serializable,
# and some safety checks e.g. whether the graph metadata is consistent with what's been traced.
#
# In autoparallel, we don't care about the serializability of this initial
# trace, but we do want those same safety checks. In the short term, we
# can patch the verification logic.
@contextmanager
def monkey_patch_export_verifier():
    from torch._export.verifier import SpecViolationError, Verifier, final

    prior = Verifier._check_graph_module

    # Export validates the output module to ensure metadata isn't missing, that it is serializable, etc.
    # We don't need them for the most part, please allowlist them here:
    def expected_error(e: Exception):
        okay = [
            "Operator 'autoparallel.dtype_cast' is not an allowed operator type",
            "call_local_map",
        ]
        e_str = str(e)
        for msg in okay:
            if msg in e_str:
                return True
        return False

    @final
    def _try_check_graph_module(self: Verifier, gm: torch.fx.GraphModule) -> None:
        try:
            return prior(self, gm)
        except SpecViolationError as e:
            if not expected_error(e):
                raise
            warnings.warn(f"Ignoring strict-mode export verifier error: {e}")

    try:
        Verifier._check_graph_module = _try_check_graph_module
        yield
    finally:
        Verifier._check_graph_module = prior


@contextmanager
def enable_local_map_wrapping():
    from torch._dynamo.variables.higher_order_ops import (
        LocalMapWrappedHigherOrderVariable as vt_cls,
    )
    from torch._higher_order_ops import local_map as local_map_module

    with vt_cls.enable(), local_map_module.defer_inlining():
        yield


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

        # keep a separate copy of the fake orig model to customize for supporting init_weights
        self.init_weights_model = move_to_fake(
            copy.deepcopy(model), self.fake_mode, device
        )

        if self.mp_policy is not None:
            apply_dtype_cast(model, self.mp_policy)

        self.model = move_to_fake(model, self.fake_mode, device)
        self.input_fn = input_fn
        self.mesh = mesh
        self.compiler_fn = compile_fx_inner if compile else boxed_nop_preserve_node_meta
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

    def build_model_graph(self):
        decomp_table = _get_decomp_table()

        with self.fake_mode:
            inputs = self.input_fn()
            if not isinstance(inputs, tuple):
                inputs = (inputs,)

        with set_dtype_cast(
            True
        ), enable_local_map_wrapping(), torch._dynamo.utils._disable_saved_tensors_hooks_during_tracing():
            torch_ir_with_fqn = dynamo_graph_capture_for_export(self.model)(*inputs)
            # TODO Cna't use fake mode here because it clashes with the user level
            # fake mode. Ideally dynamo should reuse the user level fake mode.
            self.joint_with_descriptors = aot_export_joint_with_descriptors(
                self.stack,
                torch_ir_with_fqn,
                inputs,
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
            print(self.sharding_optimizer.get_log())

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
        for k, v in sharded_param_dict.items():
            _assign_attr(v, self.parallel_model, k, attr_kind=_AttrKind.PARAMETER)

        for k, v in sharded_buffer_dict.items():
            _assign_attr(v, self.parallel_model, k, attr_kind=_AttrKind.BUFFER)

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


########################
# Pipeline stuff start #
########################
class AutoParallelPPModule(torch.nn.Module):
    def __init__(
        self,
        sharded_param_dict: dict[str, torch.nn.Parameter],
        sharded_buffer_dict: dict[str, torch.Tensor],
        init_weights_model: torch.nn.Module,
    ):
        super().__init__()
        self._register_params_and_buffers(sharded_param_dict, sharded_buffer_dict)

        # Right now we require a convention that the user model provides an init_weights method,
        # although we could snoop for other methods too.
        if hasattr(init_weights_model, "init_weights"):
            hook_params_setters(init_weights_model, self)

            def init_weights(_self, *args, **kwargs):
                # this is now a deep-fake-copy of orig mod, so we don't have to use reparametrize
                return init_weights_model.init_weights(*args, **kwargs)

            # assign an init_weights method onto the output mod.
            # all it does is sneakily run the original user mod's init_weights method,
            # but with our new DTensor sharded params attached to the user module.
            self.init_weights = MethodType(init_weights, self)

    def _register_params_and_buffers(self, sharded_param_dict, sharded_buffer_dict):

        # We construct an unflattened structure on parallel_mod,
        # e.g. _assign_attr(v, parallel_model, k="layers.0.weight") will literally
        # create empty nn.Modules recursively and then stash 'v' so it shows up in the right spot
        for k, v in sharded_param_dict.items():
            _assign_attr(v, self, k, attr_kind=_AttrKind.PARAMETER)

        for k, v in sharded_buffer_dict.items():
            _assign_attr(v, self, k, attr_kind=_AttrKind.BUFFER)

    def forward(self, *args):
        raise NotImplementedError("This is a placeholder for the pipeline model")


class AutoParallelPP(AutoParallel):
    def apply_placement_pp(
        self, sharding_placement=None, graph_passes: list[str] = []
    ) -> dict[str, Any]:
        assert all(
            g_pass in ["split_fsdp_collectives", "split_dI_dW"]
            for g_pass in graph_passes
        ), "Only split_fsdp_collectives and split_dI_dW_graph are supported"
        sharded_param_dict, sharded_buffer_dict = self._apply_placement_common(
            sharding_placement
        )
        num_params = len(sharded_param_dict)
        num_buffers = len(sharded_buffer_dict)
        (
            fw_module,
            bw_module,
            num_params_buffers,
            num_user_outputs,
            num_mutate_inputs,
            num_fw_outs_saved_for_bw,
            num_symints_saved_for_bw,
            _indices_of_inps_to_detach,
            adjusted_flat_args,
        ) = partition_joint_with_descriptors(self.joint_with_descriptors)
        assert num_params_buffers == (
            num_params + num_buffers
        ), f"num_params_buffers: {num_params_buffers}, num_params: {num_params}, num_buffers: {num_buffers}"
        print(
            f"num_params_buffers: {num_params_buffers}\n"
            f"num_user_outputs: {num_user_outputs}\n"
            f"num_mutate_inputs: {num_mutate_inputs}\n"
            f"num_fw_outs_saved_for_bw: {num_fw_outs_saved_for_bw}\n"
            f"num_symints_saved_for_bw: {num_symints_saved_for_bw}"
        )

        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "autoparallel_pp_fwd_graph",
                "encoding": "string",
            },
            payload_fn=lambda: fw_module.print_readable(
                print_output=False, include_stride=True, include_device=True
            ),
        )
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "autoparallel_pp_bwd_graph",
                "encoding": "string",
            },
            payload_fn=lambda: bw_module.print_readable(
                print_output=False, include_stride=True, include_device=True
            ),
        )
        unshard_module: Optional[torch.fx.GraphModule] = None
        reduce_grad_module: Optional[torch.fx.GraphModule] = None
        if "split_fsdp_collectives" in graph_passes:
            assert (
                not self.reshard_after_forward
            ), "reshard_after_forward should be False to disable FSDP all_gather in the backward pass"
            from autoparallel._passes.split_fsdp_collectives import (
                split_fsdp_prefetch,
                split_fsdp_reduce_scatters_epilogue,
            )

            unshard_module, fw_module = split_fsdp_prefetch(fw_module, num_params)
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "autoparallel_pp_unshard_graph",
                    "encoding": "string",
                },
                payload_fn=lambda: unshard_module.print_readable(
                    print_output=False, include_stride=True, include_device=True
                ),
            )
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "autoparallel_pp_fwd_no_fsdp_graph",
                    "encoding": "string",
                },
                payload_fn=lambda: fw_module.print_readable(
                    print_output=False, include_stride=True, include_device=True
                ),
            )
            bw_module, reduce_grad_module = split_fsdp_reduce_scatters_epilogue(
                bw_module, num_params
            )
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "autoparallel_pp_bwd_no_fsdp_graph",
                    "encoding": "string",
                },
                payload_fn=lambda: bw_module.print_readable(
                    print_output=False, include_stride=True, include_device=True
                ),
            )
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "autoparallel_pp_reduce_grad_graph",
                    "encoding": "string",
                },
                payload_fn=lambda: reduce_grad_module.print_readable(
                    print_output=False, include_stride=True, include_device=True
                ),
            )

        bw_dI_module: Optional[torch.fx.GraphModule] = None
        bw_dW_module: Optional[torch.fx.GraphModule] = None
        num_input_grads = 0
        if "split_dI_dW" in graph_passes:
            from autoparallel._passes.split_di_dw_graph import split_di_dw_graph

            bw_dI_module, bw_dW_module, num_input_grads = split_di_dw_graph(
                bw_module,
                num_weight_gradients=num_params_buffers,
            )
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "autoparallel_pp_bw_dI_graph",
                    "encoding": "string",
                },
                payload_fn=lambda: bw_dI_module.print_readable(
                    print_output=False, include_stride=True, include_device=True
                ),
            )
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "autoparallel_pp_bw_dW_graph",
                    "encoding": "string",
                },
                payload_fn=lambda: bw_dW_module.print_readable(
                    print_output=False, include_stride=True, include_device=True
                ),
            )
            if all(
                x is None
                for x in bw_dI_module.graph.find_nodes(op="output")[0].args[0][
                    :num_input_grads
                ]
            ):
                raise RuntimeError(
                    "attempted to run split dI/dW pass on a graph that has no input gradients"
                )

        graph_meta: dict[str, int] = {
            "num_mutate_inputs": num_mutate_inputs,
            "num_user_outputs": num_user_outputs,
            "num_symints_saved_for_bw": num_symints_saved_for_bw,
            "num_params": num_params,
            "num_buffers": num_buffers,
            "num_input_grads": num_input_grads,
        }

        graph_modules: dict[str, Optional[torch.fx.GraphModule]] = {
            "fw": fw_module,
            "full_bw": bw_module,
            "bw_dI": bw_dI_module,
            "bw_dW": bw_dW_module,
            "unshard": unshard_module,
            "reduce_grad": reduce_grad_module,
        }
        self.parallel_model = AutoParallelPPModule(
            sharded_param_dict,
            sharded_buffer_dict,
            self.init_weights_model,
        )
        return {
            "graph_callables": graph_modules,
            "graph_meta": graph_meta,
            "sharded_param_dict": sharded_param_dict,
            "sharded_buffer_dict": sharded_buffer_dict,
        }


######################
# Pipeline stuff end #
######################
