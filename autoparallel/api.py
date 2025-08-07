# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import copy
import itertools
from contextlib import ExitStack
from types import MethodType
from typing import Optional, Union

import torch
from torch._functorch.aot_autograd import (
    JointWithDescriptors,
    aot_compile_joint_with_descriptors,
    aot_export_joint_with_descriptors,
    boxed_nop_preserve_node_meta,
)
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.decomposition import select_decomp_table
from torch._inductor.fx_passes.joint_graph import joint_graph_passes
from torch._inductor.fx_passes.post_grad import remove_assert_ops
from torch._logging import trace_structured
from torch._subclasses import FakeTensorMode
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DeviceMesh
from torch.export._unlift import _assign_attr
from torch.export.unflatten import _AttrKind
from torch.fx import GraphModule
from torch.fx.experimental._backward_state import BackwardState

from .activation_checkpointing import ac_joint_pass
from .apply_sharding import apply_sharding_to_model
from .cast_parametrization import apply_dtype_cast, canonicalize_mp, set_dtype_cast
from .init_weights import hook_params_setters
from .optimize_sharding import ShardingOptimizer
from .utils import _get_device_from_mesh


def update_joint_with_descriptors(
    joint_with_descriptors: JointWithDescriptors,
    updated_gm: GraphModule,
) -> None:
    """
    Assuming we have transformed updated_gm since the time it was captured,
    (e.g. by parallelizing it),
    this util updates the joint_with_descriptors struct to reference the new gm, and
    updates any copies of tensor meta/shape stored in joint_with_descriptors relating to input arguments,
    which may have changed shape since the initial trace.
    """
    # TODO: should we upstream a util like this?
    placeholders = [n for n in updated_gm.graph.nodes if n.op == "placeholder"]
    new_local_args = [n.meta["val"] for n in placeholders]
    joint_with_descriptors.graph_module = updated_gm
    joint_with_descriptors._aot_graph_capture.graph_module = updated_gm

    new_flat_args: list[Union[torch.Tensor, int, torch.SymInt, BackwardState]] = []
    for orig, new in zip(joint_with_descriptors._aot_state.flat_args, new_local_args):
        if isinstance(orig, torch.nn.Parameter):
            new_flat_args.append(torch.nn.Parameter(new))
        else:
            new_flat_args.append(new)

    tangent_idx = len(joint_with_descriptors._aot_state.flat_args)
    new_local_tangents = new_local_args[tangent_idx:]
    joint_with_descriptors._aot_graph_capture.updated_flat_args = (
        new_flat_args,
        new_local_tangents,
    )
    joint_with_descriptors._aot_state.flat_args = new_flat_args
    joint_with_descriptors._aot_state.fw_metadata.traced_tangents = new_local_tangents


def _add_alias(gm):
    """
    Helper function to add alias nodes to every node in the graph
    this gives more configuration opportunities
    """
    graph = gm.graph

    nodes = [n for n in graph.nodes if n.op == "call_function"]
    node_map = {node: idx for idx, node in enumerate(nodes)}
    inputs = graph.find_nodes(op="placeholder")
    for node in inputs:
        if len(node.users) == 0:
            # node is not used, don't add alias for it
            continue
        first_user = nodes[min(node_map[n] for n in node.users)]
        with graph.inserting_before(first_user):
            alias_node = graph.call_function(torch.ops.aten.alias.default, args=(node,))
            alias_node.meta.update(node.meta)

            def delete_user_cb(n):
                return n != alias_node

            node.replace_all_uses_with(alias_node, delete_user_cb=delete_user_cb)

    """
    for node in nodes:
        # skip ops which return tuple
        if not isinstance(node.meta["val"], torch.Tensor):
            continue
        with graph.inserting_after(node):
            alias_node = graph.call_function(torch.ops.aten.alias.default, args=(node,))
            alias_node.meta.update(node.meta)

            def delete_user_cb(n):
                return n != alias_node

            node.replace_all_uses_with(alias_node, delete_user_cb=delete_user_cb)

    """

    for node in graph.find_nodes(op="output")[0].all_input_nodes:
        with graph.inserting_after(node):
            alias_node = graph.call_function(torch.ops.aten.alias.default, args=(node,))
            alias_node.meta.update(node.meta)

            def delete_user_cb(n):
                return n != alias_node

            node.replace_all_uses_with(alias_node, delete_user_cb=delete_user_cb)

    gm.recompile()
    return gm


def try_convert_fake_to_real(tensors):
    out = {}
    for k, t in tensors.items():
        out[k] = torch.distributed.tensor.randn(
            t.shape, dtype=t.dtype, device_mesh=t.device_mesh, placements=t.placements
        )
    return out


def _get_decomp_table():
    decomp_table = copy.copy(select_decomp_table())
    # TODO: removing those as they cause missing DTensor propagation rules
    decomp_table.pop(torch.ops.aten.full_like.default)
    decomp_table.pop(torch.ops.aten.empty_like.default)
    decomp_table.pop(torch.ops.aten.threshold_backward.default)
    decomp_table.pop(torch.ops.aten.native_layer_norm.default)
    decomp_table.pop(torch.ops.aten.embedding_dense_backward.default)
    decomp_table.pop(torch.ops.aten.native_layer_norm_backward.default)

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
            fake_tensor = torch.nn.Parameter(fake_tensor)

        setattr(submod, k, fake_tensor)

    with mode:
        for k, p in model.named_parameters():
            assert_is_meta_tensor(k, p)
            _move_to_fake(model, k, device, parameter=True)
        for k, b in model.named_buffers():
            assert_is_meta_tensor(k, b)
            _move_to_fake(model, k, device, parameter=False)

    return model


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
        ac_stage_size_in_GiB: Optional[float] = None,
    ):
        self.stack = ExitStack()
        self.fake_mode = (
            FakeTensorMode()
        )  # TODO: maybe need to reuse the model's fake mode
        device = _get_device_from_mesh(mesh)
        if mp_policy is not None:
            mp_policy = canonicalize_mp(mp_policy)
        self.mp_policy = mp_policy
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
            self.gm, self.mesh, rescale_grad_comm_cost_for_mp
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
        # needed because of https://github.com/pytorch/pytorch/issues/148977
        # TODO: Don't do a global setting for this, this will unpredictably
        # affect user code
        torch.__future__.set_swap_module_params_on_conversion(True)

        with self.fake_mode:
            inputs = self.input_fn()
            if not isinstance(inputs, tuple):
                inputs = (inputs,)

        with set_dtype_cast(True):
            ep = torch.export.export(self.model, inputs)
            self.joint_with_descriptors = aot_export_joint_with_descriptors(
                self.stack,
                ep.module(),
                inputs,
                decompositions=decomp_table,
                fw_compiler=self.compiler_fn,
                bw_compiler=self.compiler_fn,
            )
        gm = self.joint_with_descriptors.graph_module

        # cleanup graph
        # TODO: Make the DCE match exactly the AOTAutograd logic, I don't
        # think I trust the default FX DCE logic
        gm.graph.eliminate_dead_code()
        gm.recompile()
        # disable pattern_matcher as it gets on our way
        # we basically want to remove noops in here
        prev = torch._inductor.config.pattern_matcher
        torch._inductor.config.pattern_matcher = False
        try:
            # TODO: Double check if this is what we want to do
            gm = joint_graph_passes(gm)
        finally:
            torch._inductor.config.pattern_matcher = prev
        # TODO: We shouldn't actually remove these
        remove_assert_ops(gm.graph)
        gm.graph.eliminate_dead_code()
        gm.recompile()
        # now add aliases nodes to the graph to
        # give more room for optimizations
        _add_alias(gm)
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "autoparallel_joint_graph",
                "encoding": "string",
            },
            # TODO: Use print_readable instead with useful options
            payload_fn=lambda: str(gm.graph),
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

    def apply_placement(self, sharding_placement=None):
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
        parallel_gm = joint_graph_passes(parallel_gm)
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "autoparallel_parallel_graph",
                "encoding": "string",
            },
            payload_fn=lambda: str(parallel_gm.graph),
        )

        if self.enable_ac:
            ac_joint_pass(parallel_gm.graph, self.ac_stage_size_in_GiB)
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
        torch.fx.node._side_effectful_functions.remove(
            torch.ops._c10d_functional.wait_tensor
        )
        torch.fx.node._side_effectful_functions.remove(
            torch.ops._c10d_functional.wait_tensor.default
        )

        self.parallel_model_fn = parallel_model_fn = aot_compile_joint_with_descriptors(
            self.joint_with_descriptors
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

        return self.parallel_model
