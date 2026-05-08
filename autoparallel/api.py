# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
from torch._dynamo.functional_export import (
    _dynamo_graph_capture_for_export,
    dynamo_graph_capture_for_export,
)
from torch._functorch.aot_autograd import (
    aot_compile_joint_with_descriptors,
    aot_export_joint_with_descriptors,
)
from torch._logging import trace_structured
from torch._subclasses import FakeTensorMode
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DeviceMesh
from torch.export._trace import _restore_state_dict
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from .apply_sharding import apply_sharding_to_model
from .cast_parametrization import apply_dtype_cast, canonicalize_mp, set_dtype_cast
from .graph_passes.activation_checkpointing import mark_fsdp_all_gather_recomputation
from .graph_passes.graph_utils import (
    _add_alias,
    _replace_view_mm_view_with_einsum,
    assert_has_no_collectives,
    cleanup_graph,
    fix_scatter_on_aliased_inputs,
    update_joint_with_descriptors,
)
from .input_validation import (
    _check_forward_args,
    _compute_expected_inputs,
    _extract_input_info,
    _flatten_out_shardings,
    _make_input_fn,
)
from .module_construction import make_parallel_module
from .optimize_sharding import ShardingOptimizer
from .shardings.placement_options import _get_device_from_mesh
from .tracing import (
    _add_unused_params_and_buffers,
    _get_decomp_table,
    enable_local_map_wrapping,
    move_to_fake,
)

_APPLY_VIEW_MM_VIEW_PATTERN = True

logger = logging.getLogger(__name__)


def _boxed_nop_preserve_node_meta(fx_g, example_inputs):
    def run(args):
        with torch.fx.traceback.preserve_node_meta():
            return torch.fx.Interpreter(fx_g).boxed_run(args)

    run._boxed_call = True
    return run


@contextmanager
def _suppress_wait_tensor_side_effect():
    """Temporarily remove wait_tensor from the side-effectful set.

    This allows DCE to clean up unused wait_tensor nodes in the backward graph,
    which is important for memory. The entries are restored on exit.
    """
    ops = torch.fx.node._side_effectful_functions
    removed = set()
    for op in (
        torch.ops._c10d_functional.wait_tensor,
        torch.ops._c10d_functional.wait_tensor.default,
    ):
        if op in ops:
            ops.remove(op)
            removed.add(op)
    try:
        yield
    finally:
        ops.update(removed)


@dataclass
class JointGraphResult:
    gm: torch.fx.GraphModule
    joint_with_descriptors: Any
    traced_inputs: list[Any]


def _make_inputs_dynamic(
    inputs: tuple[Any, ...], fake_mode: FakeTensorMode
) -> tuple[Any, ...]:
    """Convert concrete FakeTensors to symbolic ones with all-dynamic dims.

    The ShapeEnv will automatically concretize non-batch dimensions as they
    interact with concrete parameter shapes during tracing.
    """
    from torch.fx.experimental.symbolic_shapes import (
        DimDynamic,
        StatelessSymbolicContext,
    )
    from torch.utils._pytree import tree_map_only

    def to_symbolic(t: torch.Tensor) -> torch.Tensor:
        sym_ctx: StatelessSymbolicContext = StatelessSymbolicContext(
            dynamic_sizes=[DimDynamic.DYNAMIC] * t.ndim,
        )
        meta = torch.empty(
            t.shape, dtype=t.dtype, device="meta", requires_grad=t.requires_grad
        )
        sym = fake_mode.from_tensor(meta, symbolic_context=sym_ctx)
        with fake_mode:
            return sym.to(t.device)

    return tree_map_only(torch.Tensor, to_symbolic, inputs)


def _post_trace_graph_passes(gm, artifact_name):
    """Common graph passes after tracing: collectives check, cleanup, view pattern, aliases, logging."""
    assert_has_no_collectives(gm)
    cleanup_graph(gm)
    if _APPLY_VIEW_MM_VIEW_PATTERN:
        _replace_view_mm_view_with_einsum(gm)
    _add_alias(gm, version="v2")
    trace_structured(
        "artifact",
        metadata_fn=lambda: {"name": artifact_name, "encoding": "string"},
        payload_fn=lambda: gm.print_readable(
            print_output=False, include_stride=True, include_device=True
        ),
    )


def _prepare_inputs(input_fn, fake_mode):
    """Prepare inputs for graph tracing: call input_fn, format, optionally make dynamic."""
    with fake_mode:
        raw_inputs = input_fn()
    formatted_inputs = raw_inputs if isinstance(raw_inputs, tuple) else (raw_inputs,)
    if fake_mode.shape_env is not None:
        formatted_inputs = _make_inputs_dynamic(formatted_inputs, fake_mode)
    return formatted_inputs


def _tracing_context():
    """Context managers used during dynamo graph capture."""
    from contextlib import contextmanager

    @contextmanager
    def ctx():
        with (
            set_dtype_cast(True),
            enable_local_map_wrapping(),
            torch._dynamo.utils._disable_saved_tensors_hooks_during_tracing(),
        ):
            yield

    return ctx()


def build_joint_graph(
    model: torch.nn.Module,
    input_fn: Callable,
    fake_mode: FakeTensorMode,
    stack: ExitStack,
) -> JointGraphResult:
    t0 = time.perf_counter()
    decomp_table = _get_decomp_table()
    formatted_inputs = _prepare_inputs(input_fn, fake_mode)
    traced_inputs = list(formatted_inputs)

    with _tracing_context():
        gm = _dynamo_graph_capture_for_export(model)(*formatted_inputs)
        _restore_state_dict(model, gm)
        _add_unused_params_and_buffers(model, gm)
        joint_with_descriptors = aot_export_joint_with_descriptors(
            stack,
            gm,
            formatted_inputs,
            decompositions=decomp_table,
        )
    gm = joint_with_descriptors.graph_module
    _post_trace_graph_passes(gm, "autoparallel_joint_graph")

    logger.info("Graph tracing took %.3fs", time.perf_counter() - t0)
    return JointGraphResult(
        gm=gm,
        joint_with_descriptors=joint_with_descriptors,
        traced_inputs=traced_inputs,
    )


def build_user_backward_graph(model, input_fn, fake_mode):
    """Build graph from a model whose forward() contains backward().

    Uses trace_autograd_ops=True to capture autograd.grad nodes, then
    prepare_aot_module_simplified to get a functional_call with proper
    descriptors and params-first arg ordering. The functional_call is
    then traced via 2-pass make_fx to decompose autograd.grad and
    backward formula ops into primitive aten ops.
    """
    from torch._functorch._aot_autograd.descriptors import PlainAOTInput, PlainAOTOutput
    from torch._functorch.aot_autograd import prepare_aot_module_simplified
    from torch.fx.experimental.proxy_tensor import make_fx

    t0 = time.perf_counter()
    decomp_table = _get_decomp_table()
    formatted_inputs = _prepare_inputs(input_fn, fake_mode)

    prev_trace_autograd = torch._dynamo.config.trace_autograd_ops
    torch._dynamo.config.trace_autograd_ops = True
    try:
        with _tracing_context():
            gm = dynamo_graph_capture_for_export(model)(*formatted_inputs)
    finally:
        torch._dynamo.config.trace_autograd_ops = prev_trace_autograd

    # Use prepare_aot_module_simplified to get:
    # - functional_call: a callable with args in [params, buffers, inputs] order
    # - full_args_descs: descriptors matching that order
    # - fake_flat_args: fake tensors in the same order
    # This handles param lifting, unused params, and FQN mapping.
    (
        functional_call,
        _params_buffers_flat,
        _params_spec,
        _buffers_spec,
        fake_flat_args,
        full_args_descs,
        _aot_config,
        prep_fake_mode,
        _shape_env,
        _in_spec,
        _out_spec,
        _act_input_indices,
    ) = prepare_aot_module_simplified(
        gm,
        formatted_inputs,
        None,
        decomp_table,
        False,
        False,
        flatten=True,
        force_non_lazy_backward_lowering=True,
    )

    # Separate tensor args from non-tensor constants (e.g. string "loss and bwd").
    # make_fx can only trace tensor args as placeholders. Non-tensor args are
    # baked into the traced function as constants.
    tensor_positions = [
        i for i, a in enumerate(fake_flat_args) if isinstance(a, torch.Tensor)
    ]
    non_tensor_args = {
        i: a for i, a in enumerate(fake_flat_args) if not isinstance(a, torch.Tensor)
    }
    tensor_flat_args = [fake_flat_args[i] for i in tensor_positions]

    def _functional_call_tensors_only(*tensor_args):
        # Reconstruct full args with non-tensor constants injected
        full_args = list(tensor_args)
        for pos, val in sorted(non_tensor_args.items()):
            full_args.insert(pos, val)
        return functional_call(*full_args)

    # Trace functional_call with make_fx. Since functional_call has args
    # in [params, buffers, inputs] order, the resulting graph's placeholders
    # will be in that order too — matching full_args_descs for a straight zip.
    with torch.fx.traceback.preserve_node_meta(), prep_fake_mode:
        # Pass 1: decompose autograd.grad into backward formula ops
        decomposed_gm = make_fx(
            _functional_call_tensors_only,
            decomposition_table=decomp_table,
            tracing_mode="fake",
        )(*tensor_flat_args)

        # Pass 2: decompose backward formula ops (nll_loss_backward,
        # slice_backward, etc.) into primitive aten ops.
        interp2 = torch.fx.Interpreter(decomposed_gm)
        decomposed_gm = make_fx(
            interp2.run,
            decomposition_table=decomp_table,
            tracing_mode="fake",
        )(*tensor_flat_args)

    gm = decomposed_gm

    # Assign descriptors to placeholders — straight zip since both are
    # in [params, buffers, inputs] order. Non-tensor args were filtered
    # before make_fx, so all placeholders are tensors.
    placeholder_nodes = [n for n in gm.graph.nodes if n.op == "placeholder"]
    tensor_descs = [
        d
        for d, a in zip(full_args_descs, fake_flat_args)
        if isinstance(a, torch.Tensor)
    ]
    assert len(placeholder_nodes) == len(
        tensor_descs
    ), f"Placeholder count {len(placeholder_nodes)} != tensor descriptor count {len(tensor_descs)}"
    for n, desc in zip(placeholder_nodes, tensor_descs):
        n.meta["desc"] = desc

    # Set the output descriptor list on the output node
    for n in gm.graph.nodes:
        if n.op == "output":
            output_args = (
                n.args[0] if isinstance(n.args[0], (list, tuple)) else (n.args[0],)
            )
            n.meta["desc"] = [PlainAOTOutput(idx=i) for i in range(len(output_args))]

    _post_trace_graph_passes(gm, "autoparallel_user_backward_graph")

    logger.info("User-backward graph tracing took %.3fs", time.perf_counter() - t0)

    # Build traced_inputs from PlainAOTInput placeholders.
    traced_inputs = [
        n.meta["val"]
        for n in gm.graph.nodes
        if n.op == "placeholder" and isinstance(n.meta.get("desc"), PlainAOTInput)
    ]

    return gm, traced_inputs


class AutoParallelBase:
    """Low-level API that accepts an FX graph with descriptors and a mesh.

    Provides sharding optimization (ILP) and application. The graph must have
    ``meta["desc"]`` set on all placeholder and output nodes (ParamAOTInput,
    BufferAOTInput, PlainAOTInput, GradAOTOutput, PlainAOTOutput, etc.).

    For the higher-level API that traces a model, use ``AutoParallel`` instead.
    """

    def __init__(
        self,
        gm,
        mesh,
        *,
        reshard_after_forward=True,
        cost_model="nccl",
        repeated_subgraphs=True,
        rescale_grad_comm_cost_for_mp=1.0,
    ):
        self.gm = gm
        self.mesh = mesh
        self.reshard_after_forward = reshard_after_forward

        from .cost_models.collective_runtime_estimation import (
            get_nccl_topo_config,
            set_nccl_topo_config,
        )
        from .cost_models.nccl_cost_model import NCCLTopoConfig, detect_nccl_topo_config

        self._prev_nccl_config = get_nccl_topo_config()
        if isinstance(cost_model, NCCLTopoConfig):
            set_nccl_topo_config(cost_model)
        elif cost_model == "nccl":
            set_nccl_topo_config(detect_nccl_topo_config(mesh))
        else:
            set_nccl_topo_config(None)

        self.sharding_optimizer = ShardingOptimizer(
            gm,
            mesh,
            rescale_grad_comm_cost_for_mp,
            repeated_subgraphs=repeated_subgraphs,
        )

        self.input_constraints = None
        self.output_constraints = None

    def add_parameter_memory_constraint(self, low=None, high=None):
        if low is None:
            low = 0.0
        if high is None:
            high = 1.0 / self.mesh.size()
        assert low <= high, f"low should be <= high, got low{low}, high={high}"
        self.sharding_optimizer.add_parameter_memory_constraint(low, high)

    def add_input_constraints(self, constraints):
        assert self.input_constraints is None, "Input constraints have already been set"
        self.sharding_optimizer.add_sharded_input_constraint(constraints)
        self.input_constraints = constraints

    def add_output_constraints(self, constraints):
        assert (
            self.output_constraints is None
        ), "Output constraints have already been set"
        self.sharding_optimizer.add_sharded_output_constraint(constraints)
        self.output_constraints = constraints

    def optimize_placement(self, verbose=True):
        self.sharding_placement = self.sharding_optimizer.get_solution(verbose=False)

        if verbose:
            logger.info(self.sharding_optimizer.get_log(verbose=True))

        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "autoparallel_sharding_optimizer_log",
                "encoding": "string",
            },
            payload_fn=lambda: self.sharding_optimizer.get_log(
                verbose=True, colored=False
            ),
        )

        if self.sharding_optimizer.prob.status == -1:
            raise RuntimeError(
                "The sharding optimizer could not find a feasible solution. "
                "This typically means the user-specified constraints are "
                "contradictory or the device mesh is too small for the requested "
                "sharding. Check the WARNING log for the list of violated "
                "constraints, and consider relaxing input/output constraints or "
                "using a larger mesh."
            )

        return self.sharding_placement

    def apply_sharding(
        self, sharding_placement, *, params_spec=None, buffers_spec=None
    ):
        """Apply sharding to the graph, returning the parallelized graph and sharded dicts.

        If params_spec/buffers_spec are not provided, they are derived from
        graph node descriptors.
        """
        from torch._functorch._aot_autograd.fx_utils import (
            get_named_buffer_nodes,
            get_named_param_nodes,
        )
        from torch._subclasses.fake_tensor import unset_fake_temporarily

        t0 = time.perf_counter()

        if params_spec is None:
            params_spec = get_named_param_nodes(self.gm.graph)
        if buffers_spec is None:
            buffers_spec = get_named_buffer_nodes(self.gm.graph)

        with unset_fake_temporarily():
            mesh = self.mesh
            if mesh.ndim != 1:
                mesh._flatten()

        # Extract fake_mode from graph placeholder nodes
        fake_mode = None
        for node in self.gm.graph.nodes:
            if node.op == "placeholder":
                val = node.meta.get("val")
                if hasattr(val, "fake_mode"):
                    fake_mode = val.fake_mode
                    break

        if fake_mode is not None:
            ctx = fake_mode
        else:
            from contextlib import nullcontext

            ctx = nullcontext()

        with ctx:
            (
                parallel_gm,
                sharded_param_dict,
                sharded_buffer_dict,
            ) = apply_sharding_to_model(
                self.gm,
                sharding_placement,
                params_spec,
                buffers_spec,
            )
        t_apply = time.perf_counter()

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

        from torch._inductor.fx_passes.post_grad import view_to_reshape

        view_to_reshape(parallel_gm)

        mark_fsdp_all_gather_recomputation(
            parallel_gm.graph, self.reshard_after_forward
        )
        t_ac = time.perf_counter()

        fix_scatter_on_aliased_inputs(parallel_gm.graph)

        logger.info(
            "Apply placements took %.3fs "
            "(apply_sharding=%.3fs, cleanup=%.3fs, trace=%.3fs, ac=%.3fs)",
            time.perf_counter() - t0,
            t_apply - t0,
            t_cleanup - t_apply,
            t_trace - t_cleanup,
            t_ac - t_trace,
        )
        self.parallel_gm = parallel_gm
        return parallel_gm, sharded_param_dict, sharded_buffer_dict

    def _restore_nccl_config(self):
        from .cost_models.collective_runtime_estimation import set_nccl_topo_config

        set_nccl_topo_config(self._prev_nccl_config)


class AutoParallel(AutoParallelBase):
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
        reshard_after_forward: bool = True,
        dynamic: bool = False,
        cost_model: Any = "nccl",
        repeated_subgraphs: bool = True,
    ):
        # Don't call super().__init__ yet — gm isn't built until __enter__
        self.stack = ExitStack()
        self.fake_mode = (
            FakeTensorMode()
        )  # TODO: maybe need to reuse the model's fake mode
        device = _get_device_from_mesh(mesh)
        if mp_policy is not None:
            mp_policy = canonicalize_mp(mp_policy)
        self.mp_policy = mp_policy
        self.cost_model = cost_model
        self.repeated_subgraphs = repeated_subgraphs
        # copy user model to avoid modifying it in-place
        # in dtype casting and move_to_fake
        model = copy.deepcopy(model)

        if self.mp_policy is not None:
            apply_dtype_cast(model, self.mp_policy)

        self.model = move_to_fake(model, self.fake_mode, device)
        self.input_fn = input_fn
        self.mesh = mesh
        self.compiler_fn = _boxed_nop_preserve_node_meta  # type: ignore[assignment]
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
            self.stack.enter_context(self.mesh)

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

            # Initialize the base class now that gm is available
            AutoParallelBase.__init__(
                self,
                self.gm,
                self.mesh,
                reshard_after_forward=self.reshard_after_forward,
                cost_model=self.cost_model,
                repeated_subgraphs=self.repeated_subgraphs,
                rescale_grad_comm_cost_for_mp=rescale_grad_comm_cost_for_mp,
            )

            self.active = True

            self.stack.__enter__()
        except BaseException:
            self.stack.__exit__(None, None, None)
            raise

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._restore_nccl_config()
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
        result = build_joint_graph(
            self.model, self.input_fn, self.fake_mode, self.stack
        )
        self.gm = result.gm
        self.joint_with_descriptors = result.joint_with_descriptors
        self._traced_inputs = result.traced_inputs

    # Override base constraint methods to add _assert_entered checks
    def add_parameter_memory_constraint(self, low=None, high=None):
        self._assert_entered()
        super().add_parameter_memory_constraint(low, high)

    def add_input_constraints(self, constraints):
        self._assert_entered()
        super().add_input_constraints(constraints)

    def add_output_constraints(self, constraints):
        self._assert_entered()
        super().add_output_constraints(constraints)

    def optimize_placement(self, verbose=True):
        self._assert_entered()
        return super().optimize_placement(verbose)

    def _apply_placement_common(self, sharding_placement):
        self._assert_entered()
        parallel_gm, sharded_param_dict, sharded_buffer_dict = self.apply_sharding(
            sharding_placement,
            params_spec=self.joint_with_descriptors.params_spec,
            buffers_spec=self.joint_with_descriptors.buffers_spec,
        )
        update_joint_with_descriptors(self.joint_with_descriptors, parallel_gm)
        # Allow DCE to remove unused wait_tensor nodes in the backward graph.
        # Pushed onto self.stack so it's restored in AutoParallel.__exit__.
        self.stack.enter_context(_suppress_wait_tensor_side_effect())
        return sharded_param_dict, sharded_buffer_dict

    def apply_placement(self, sharding_placement):
        sharded_param_dict, sharded_buffer_dict = self._apply_placement_common(
            sharding_placement
        )

        self.parallel_model_fn = parallel_model_fn = aot_compile_joint_with_descriptors(
            self.joint_with_descriptors,
            fw_compiler=self.compiler_fn,
            bw_compiler=self.compiler_fn,
        )

        # Build a forward-only graph for inference (no backward, no
        # activation saves).  Compilation is lazy — only paid on first
        # call under torch.no_grad().
        from .graph_passes.extract_forward import extract_forward_graph

        fw_metadata = self.joint_with_descriptors._aot_state.fw_metadata
        num_fwd_outputs = fw_metadata.num_forward_returns
        fwd_only_gm = extract_forward_graph(self.parallel_gm, num_fwd_outputs)
        compiler_fn = self.compiler_fn
        aot_config = self.joint_with_descriptors._aot_state.aot_config
        out_spec = self.joint_with_descriptors.out_spec

        _inference_fn_cache = None

        def _get_inference_fn():
            nonlocal _inference_fn_cache
            if _inference_fn_cache is not None:
                return _inference_fn_cache
            example_inputs = [
                n.meta["val"] for n in fwd_only_gm.graph.nodes if n.op == "placeholder"
            ]
            compiled = compiler_fn(fwd_only_gm, example_inputs)

            # Wrap with RuntimeWrapper to handle mutation write-back
            # (e.g. buffer updates like BatchNorm running stats), output
            # alias handling, and intermediate base stripping.
            from torch._functorch._aot_autograd.runtime_wrappers import RuntimeWrapper

            wrapped = RuntimeWrapper(
                indices_of_inps_to_detach=[],
                trace_joint=False,
                disable_amp=False,
            ).post_compile(compiled, aot_config, runtime_metadata=fw_metadata)

            def inference_fn(args):
                flat_outs = wrapped(args)
                return torch.utils._pytree.tree_unflatten(flat_outs, out_spec)

            _inference_fn_cache = inference_fn
            return _inference_fn_cache

        # TODO: this probably belongs in the AOTAutograd API
        # TODO: pytree handling
        # Capture the exact FQNs the compiled graph expects as primals.
        # This avoids issues with aliased params/buffers where identity-based
        # dedup can break after init_weights reassigns tensors.
        graph_param_fqns = list(self.joint_with_descriptors.params_spec)
        graph_buffer_fqns = list(self.joint_with_descriptors.buffers_spec)

        # Extract solved input placements from the solution dict.
        # This is the ground truth for what the compiled graph expects.
        from torch._functorch._aot_autograd.fx_utils import (
            get_plain_input_and_grad_nodes,
        )

        input_nodes = get_plain_input_and_grad_nodes(self.gm.graph)
        solved_input_placements = []
        for desc in sorted(input_nodes, key=lambda d: d.idx):
            node, _grad_node = input_nodes[desc]
            strategy = sharding_placement[node]
            solved_input_placements.append(tuple(strategy.output_specs.placements))

        expected_inputs = _compute_expected_inputs(
            self._traced_inputs, solved_input_placements, self.mesh
        )

        def forward(self, *args):
            # Flatten pytree args (e.g. dicts, nested structures) to tensor
            # leaves, matching how Dynamo flattened the inputs during tracing.
            flat_args, _ = torch.utils._pytree.tree_flatten(args)
            _check_forward_args(flat_args, expected_inputs)
            # NB: don't close over the parameters/buffers, as the user may
            # reassign the module!
            # Use the exact param/buffer FQNs that the compiled graph
            # expects, matching the primals order from tracing.
            params = [
                self.get_parameter(fqn).to_local() for fqn in graph_param_fqns
            ] + [self.get_buffer(fqn).to_local() for fqn in graph_buffer_fqns]
            boxed_args = [*params, *flat_args]
            del params
            if torch.is_grad_enabled():
                # NB: don't do self.parallel_model_fn work around Dynamo bug
                out = parallel_model_fn(boxed_args)
            else:
                out = _get_inference_fn()(boxed_args)
            return out

        self.parallel_model = make_parallel_module(
            self.model,
            sharded_param_dict,
            sharded_buffer_dict,
            forward_fn=forward,
        )
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
    parameter_memory_budget: Optional[tuple[Optional[float], Optional[float]]] = None,
    dynamic: bool = False,
) -> torch.nn.Module:
    """
    Parallelize a model with automatic sharding optimization.

    This is a simplified API that wraps the full AutoParallel context manager.
    For more control, use the AutoParallel class directly.

    The returned module runs eagerly. For compiled execution, use
    ``torch.compile(module, backend=autoparallel_backend())`` to compile with
    AP-optimized Inductor passes (activation checkpointing, overlap scheduling).

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
        parameter_memory_budget: Optional (low, high) bounds for parameter memory.
            Each bound is a float multiplier or None for unbounded.
        dynamic: If True, trace with symbolic batch dimensions so the parallel
            model accepts arbitrary batch sizes at runtime.

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
    shapes, dtypes, input_placements, treespec, devices = _extract_input_info(
        raw_inputs, mesh
    )

    # Flatten out_shardings to list
    output_placements = _flatten_out_shardings(out_shardings)

    # Create input_fn that will be called inside FakeTensorMode
    # It creates fresh tensors (which become fake tensors inside FakeTensorMode)
    input_fn = _make_input_fn(shapes, dtypes, treespec, devices=devices)

    # Use AutoParallel context manager
    with AutoParallel(
        model,
        input_fn,
        mesh,
        mp_policy=mp_policy,
        dynamic=dynamic,
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


class AutoParallelBackward(AutoParallel):
    """Parallelize a model whose forward() contains backward().

    Similar to ``AutoParallel`` but for models that call ``loss.backward()``
    inside ``forward()``. The backward is traced as part of the graph via
    ``torch.autograd.grad`` nodes. The graph is compiled as a single unit
    (no fwd/bwd split).

    Usage::

        with AutoParallelBackward(model, input_fn, mesh) as ap:
            ap.add_input_constraints(...)
            ap.add_output_constraints(...)
            placement = ap.optimize_placement()
            parallel_model = ap.apply_placement(placement)
    """

    def build_model_graph(self):
        self.gm, self._traced_inputs = build_user_backward_graph(
            self.model, self.input_fn, self.fake_mode
        )
        # Push dynamo's FakeTensorMode onto the stack so that
        # ShardingOptimizer's cost model runs under fake mode
        # (matching the forward-only path where aot_export pushes it).
        for n in self.gm.graph.nodes:
            if n.op == "placeholder":
                val = n.meta.get("val")
                if hasattr(val, "fake_mode"):
                    self.stack.enter_context(val.fake_mode)
                    break

    def apply_placement(self, sharding_placement):
        self._assert_entered()
        parallel_gm, sharded_param_dict, sharded_buffer_dict = self.apply_sharding(
            sharding_placement
        )
        self.stack.enter_context(_suppress_wait_tensor_side_effect())

        # Compile the full fwd+bwd graph as a single unit (no fwd/bwd split)
        compiler_fn = self.compiler_fn
        example_inputs = [
            n.meta["val"] for n in parallel_gm.graph.nodes if n.op == "placeholder"
        ]
        compiled_fn = compiler_fn(parallel_gm, example_inputs)

        from torch._functorch._aot_autograd.fx_utils import (
            get_named_buffer_nodes,
            get_named_param_nodes,
            get_plain_input_and_grad_nodes,
        )

        graph_param_fqns = list(get_named_param_nodes(self.gm.graph))
        graph_buffer_fqns = list(get_named_buffer_nodes(self.gm.graph))

        input_nodes = get_plain_input_and_grad_nodes(self.gm.graph)
        solved_input_placements = []
        for desc in sorted(input_nodes, key=lambda d: d.idx):
            node, _grad_node = input_nodes[desc]
            strategy = sharding_placement[node]
            solved_input_placements.append(tuple(strategy.output_specs.placements))

        expected_inputs = _compute_expected_inputs(
            self._traced_inputs, solved_input_placements, self.mesh
        )

        def forward(self, *args):
            # Flatten user args and keep only tensors — this matches the
            # PlainAOTInput order since functional_call puts them after params.
            flat_args, _ = torch.utils._pytree.tree_flatten(args)
            flat_args = [a for a in flat_args if isinstance(a, torch.Tensor)]
            _check_forward_args(flat_args, expected_inputs)
            # Graph is in [params, buffers, inputs] order (from functional_call)
            params = [
                self.get_parameter(fqn).to_local() for fqn in graph_param_fqns
            ] + [self.get_buffer(fqn).to_local() for fqn in graph_buffer_fqns]
            boxed_args = [*params, *flat_args]
            del params
            return compiled_fn(boxed_args)

        self.parallel_model = make_parallel_module(
            self.model,
            sharded_param_dict,
            sharded_buffer_dict,
            forward_fn=forward,
        )
        return self.parallel_model


def auto_parallel_with_backward(
    model,
    mesh,
    sample_inputs,
    out_shardings,
    *,
    mp_policy=None,
    parameter_memory_budget=None,
    dynamic=False,
):
    """Parallelize a model whose forward() contains backward().

    This is a simplified API that wraps ``AutoParallelBackward``.
    For more control, use ``AutoParallelBackward`` directly.

    Args:
        model: Model whose forward() includes backward(). Can be on meta device.
        mesh: Device mesh defining the distributed topology.
        sample_inputs: Sample inputs for tracing (same format as auto_parallel).
        out_shardings: Output sharding specification for non-gradient outputs.
        mp_policy: Optional mixed precision policy.
        parameter_memory_budget: Optional (low, high) bounds for parameter memory.
        dynamic: If True, trace with symbolic batch dimensions.

    Returns:
        Parallelized module. Call to_empty(device="cuda") and init_weights()
        before use.
    """
    if callable(sample_inputs):
        raw_inputs = sample_inputs()
    else:
        raw_inputs = sample_inputs

    shapes, dtypes, input_placements, treespec, devices = _extract_input_info(
        raw_inputs, mesh
    )
    output_placements = (
        _flatten_out_shardings(out_shardings) if out_shardings is not None else None
    )
    input_fn = _make_input_fn(shapes, dtypes, treespec, devices=devices)

    with AutoParallelBackward(
        model,
        input_fn,
        mesh,
        mp_policy=mp_policy,
        dynamic=dynamic,
    ) as ap:
        ap.add_input_constraints(input_placements)
        if parameter_memory_budget is not None:
            ap.add_parameter_memory_constraint(
                low=parameter_memory_budget[0], high=parameter_memory_budget[1]
            )
        if output_placements is not None:
            ap.add_output_constraints(output_placements)
        sharding_placement = ap.optimize_placement(verbose=False)
        parallel_model = ap.apply_placement(sharding_placement)

    return parallel_model


##################
# Simple API end #
##################
