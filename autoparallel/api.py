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
from torch._dynamo.functional_export import _dynamo_graph_capture_for_export
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
    functionalize_fresh_index_put_mutations,
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


def _get_device_from_device_type(device_type: str) -> torch.device:
    """Resolve a concrete device from a device type string (auto-mesh path)."""
    if device_type == "cpu":
        return torch.device("cpu")
    from torch.distributed.device_mesh import _get_device_handle

    device_handle = _get_device_handle(device_type)
    return torch.device(device_type, device_handle.current_device())


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


def build_joint_graph(
    model: torch.nn.Module,
    input_fn: Callable,
    fake_mode: FakeTensorMode,
    stack: ExitStack,
) -> JointGraphResult:
    t0 = time.perf_counter()
    decomp_table = _get_decomp_table()

    with fake_mode:
        raw_inputs = input_fn()

    formatted_inputs = raw_inputs if isinstance(raw_inputs, tuple) else (raw_inputs,)

    if fake_mode.shape_env is not None:
        formatted_inputs = _make_inputs_dynamic(formatted_inputs, fake_mode)

    traced_inputs = list(formatted_inputs)

    with (
        set_dtype_cast(True),
        enable_local_map_wrapping(),
        torch._dynamo.utils._disable_saved_tensors_hooks_during_tracing(),
    ):
        torch_ir_with_fqn = _dynamo_graph_capture_for_export(model)(*formatted_inputs)
        _restore_state_dict(model, torch_ir_with_fqn)
        _add_unused_params_and_buffers(model, torch_ir_with_fqn)
        # TODO Can't use fake mode here because it clashes with the user level
        # fake mode. Ideally dynamo should reuse the user level fake mode.
        joint_with_descriptors = aot_export_joint_with_descriptors(
            stack,
            torch_ir_with_fqn,
            formatted_inputs,
            decompositions=decomp_table,
        )
    gm = joint_with_descriptors.graph_module
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

    logger.info("Graph tracing took %.3fs", time.perf_counter() - t0)
    return JointGraphResult(
        gm=gm,
        joint_with_descriptors=joint_with_descriptors,
        traced_inputs=traced_inputs,
    )


class AutoParallel:
    """
    Args:
        mesh: Defines placement options. Either a concrete ``DeviceMesh`` or the
            string ``"auto"`` to enable automatic mesh discovery, in which case
            the mesh shape is searched over candidate factorizations of the GPU
            count (see ``autoparallel.mesh_discovery``). The meta model is moved
            to a fake device based on the mesh / ``device_type``.
        device_type: Device type used when ``mesh="auto"`` (default ``"cuda"``).
        world_size: Number of GPUs to search over when ``mesh="auto"``. Defaults
            to the initialized process group's world size.
        mesh_max_dims: Cap on mesh dimensionality during auto discovery.
        mesh_candidate_fn: Optional ``(n_gpus, topo_config) -> list[MeshCandidate]``
            override for candidate generation during auto discovery.
        mesh_constraint_fn: Optional ``(optimizer, mesh) -> None`` applied to each
            candidate's ILP during auto discovery so the search is constraint
            aware (e.g. to add the same input/output constraints).
        mesh_prune: When True (default), use the ILP's LP relaxation as a lower
            bound to prune candidate meshes during auto discovery.
    """

    def __init__(
        self,
        model,
        input_fn,
        mesh: Union[DeviceMesh, str],
        mp_policy: Optional[MixedPrecisionPolicy] = None,
        reshard_after_forward: bool = True,
        dynamic: bool = False,
        cost_model: Any = "nccl",
        repeated_subgraphs: bool = True,
        device_type: str = "cuda",
        world_size: Optional[int] = None,
        mesh_max_dims: int = 4,
        mesh_candidate_fn: Optional[Callable] = None,
        mesh_constraint_fn: Optional[Callable] = None,
        mesh_prune: bool = True,
        mesh_solve_time_limit: Optional[float] = 300.0,
    ):
        self.stack = ExitStack()
        self.fake_mode = (
            FakeTensorMode()
        )  # TODO: maybe need to reuse the model's fake mode
        # self.fake_mode.allow_scalar_outputs = True

        self.auto_mesh = isinstance(mesh, str)
        if isinstance(mesh, str):
            if mesh != "auto":
                raise ValueError(f"mesh must be a DeviceMesh or 'auto', got {mesh!r}")
            self.device_type = device_type
            device = _get_device_from_device_type(device_type)
        else:
            self.device_type = mesh.device_type
            device = _get_device_from_mesh(mesh)

        if mp_policy is not None:
            mp_policy = canonicalize_mp(mp_policy)
        self.mp_policy = mp_policy
        self.cost_model = cost_model
        self.repeated_subgraphs = repeated_subgraphs

        # Auto-mesh discovery configuration.
        self.world_size = world_size
        self.mesh_max_dims = mesh_max_dims
        self.mesh_candidate_fn = mesh_candidate_fn
        self.mesh_constraint_fn = mesh_constraint_fn
        self.mesh_prune = mesh_prune
        self.mesh_solve_time_limit = mesh_solve_time_limit
        # Populated by __enter__ when auto_mesh is enabled.
        self.mesh_discovery_result = None

        # copy user model to avoid modifying it in-place
        # in dtype casting and move_to_fake
        model = copy.deepcopy(model)

        if self.mp_policy is not None:
            apply_dtype_cast(model, self.mp_policy)

        self.model = move_to_fake(model, self.fake_mode, device)
        self.input_fn = input_fn
        # self.mesh is None until discovery picks one (auto mode).
        self.mesh = None if self.auto_mesh else mesh
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
            from .cost_models.collective_runtime_estimation import get_nccl_topo_config

            self._prev_nccl_config = get_nccl_topo_config()
            topo_config = self._resolve_topo_config()

            # The joint graph is reused across all candidate meshes. Tracing
            # still runs under an active mesh context because mesh-capturing
            # constructs (e.g. local_map) resolve the current mesh at trace
            # time; for auto discovery we use a full-size detection mesh.
            if self.auto_mesh:
                with self._detection_mesh():
                    self.build_model_graph()
            else:
                self.stack.enter_context(self.mesh)
                self.build_model_graph()

            self.old_inductor_comprehensive_padding = (
                torch._inductor.config.comprehensive_padding
            )
            torch._inductor.config.comprehensive_padding = False

            if self.auto_mesh:
                self.mesh, self.sharding_optimizer = self._discover_mesh(topo_config)
                # The chosen mesh drives downstream cost estimation and
                # apply_sharding; enter it as the active mesh context.
                self.stack.enter_context(self.mesh)
            else:
                self.sharding_optimizer = self._build_optimizer(self.mesh)

            self.input_constraints = None
            self.output_constraints = None

            self.active = True

            self.stack.__enter__()
        except BaseException:
            self.stack.__exit__(None, None, None)
            raise

        return self

    def _resolve_topo_config(self):
        """Resolve and install the NCCL topology config used for cost estimation.

        Returns the resolved NCCLTopoConfig (or None for the default cost
        model). For auto-mesh discovery the config only needs the GPU arch and
        gpus_per_node — per-dimension topology is derived from each candidate's
        mesh shape.
        """
        from .cost_models.collective_runtime_estimation import set_nccl_topo_config
        from .cost_models.nccl_cost_model import NCCLTopoConfig, detect_nccl_topo_config

        if isinstance(self.cost_model, NCCLTopoConfig):
            topo_config = self.cost_model
        elif self.cost_model == "nccl":
            topo_config = detect_nccl_topo_config(self._detection_mesh())
        else:
            topo_config = None
        set_nccl_topo_config(topo_config)
        return topo_config

    def _detection_mesh(self):
        """A mesh used only for arch/topology auto-detection (size matters)."""
        if not self.auto_mesh:
            return self.mesh
        from torch._subclasses.fake_tensor import unset_fake_temporarily
        from torch.distributed.device_mesh import init_device_mesh

        with unset_fake_temporarily():
            return init_device_mesh(self.device_type, (self._auto_world_size(),))

    def _auto_world_size(self) -> int:
        if self.world_size is not None:
            return self.world_size
        if torch.distributed.is_initialized():
            return torch.distributed.get_world_size()
        raise RuntimeError(
            "mesh='auto' requires either world_size= or an initialized "
            "process group to determine the number of GPUs."
        )

    def _build_optimizer(self, mesh):
        """Construct a ShardingOptimizer for the given mesh over the traced graph."""
        # The placement-options cache keys on op/spec fingerprints but not the
        # mesh, so it must be cleared between candidate meshes (auto discovery
        # builds several optimizers, one per mesh, over the same graph).
        from .shardings.placement_options import reset_placement_options_cache

        reset_placement_options_cache()

        force_grad_reduce_in_higher_precision = (
            self.mp_policy is not None
            and self.mp_policy.reduce_dtype is not None
            and self.mp_policy.param_dtype is not None
            and self.mp_policy.reduce_dtype.itemsize
            > self.mp_policy.param_dtype.itemsize
        )
        return ShardingOptimizer(
            self.gm,
            mesh,
            force_grad_reduce_in_higher_precision,
            repeated_subgraphs=self.repeated_subgraphs,
        )

    def _discover_mesh(self, topo_config):
        """Search candidate meshes and return ``(best_mesh, best_optimizer)``."""
        from .mesh_discovery import (
            build_device_mesh,
            discover_mesh,
            enumerate_candidate_meshes,
        )

        n_gpus = self._auto_world_size()
        candidate_fn = self.mesh_candidate_fn or enumerate_candidate_meshes
        candidates = candidate_fn(n_gpus, topo_config, self.mesh_max_dims)

        # Build each candidate's optimizer once and cache it so the LP
        # relaxation (lower bound) and the ILP solve reuse the same problem.
        optimizers: dict[tuple, Any] = {}
        meshes: dict[tuple, DeviceMesh] = {}

        def _get_optimizer(candidate):
            if candidate.shape not in optimizers:
                mesh = build_device_mesh(candidate, self.device_type)
                opt = self._build_optimizer(mesh)
                if self.mesh_constraint_fn is not None:
                    self.mesh_constraint_fn(opt, mesh)
                meshes[candidate.shape] = mesh
                optimizers[candidate.shape] = opt
            return optimizers[candidate.shape]

        def evaluate(candidate):
            opt = _get_optimizer(candidate)
            try:
                opt.get_solution(verbose=False, time_limit=self.mesh_solve_time_limit)
            except RuntimeError:
                # Infeasible, or the solver did not converge within the time
                # limit — either way this candidate is not selectable.
                return float("inf"), False
            import pulp

            return pulp.value(opt.prob.objective), True

        lower_bound = None
        if self.mesh_prune:

            def lower_bound(candidate):  # noqa: F811
                return _get_optimizer(candidate).relaxed_cost()

        result = discover_mesh(
            candidates, evaluate, lower_bound=lower_bound, verbose=True
        )
        self.mesh_discovery_result = result
        best_mesh = meshes[result.best.shape]
        best_opt = optimizers[result.best.shape]
        logger.info(
            "Auto-mesh discovery selected shape %s (cost=%.4f) out of %d "
            "candidates (evaluated %d, pruned %d)",
            result.best.shape,
            result.best_cost,
            len(candidates),
            result.n_evaluated,
            result.n_pruned,
        )
        return best_mesh, best_opt

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
        result = build_joint_graph(
            self.model, self.input_fn, self.fake_mode, self.stack
        )
        self.gm = result.gm
        self.joint_with_descriptors = result.joint_with_descriptors
        self._traced_inputs = result.traced_inputs

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

    def _apply_placement_common(self, sharding_placement):
        t0 = time.perf_counter()
        self._assert_entered()

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

        # Replace aten.view with aten.reshape unconditionally. Graph passes
        # (sharding redistributions, collective bucketing) can produce
        # non-contiguous tensors that break aten.view's contiguity requirement.
        from torch._inductor.fx_passes.post_grad import view_to_reshape

        view_to_reshape(parallel_gm)
        functionalize_fresh_index_put_mutations(parallel_gm)

        t_ac = time.perf_counter()
        # now rename input/param/tangent/output/grad_param/grad_input nodes following
        # our convention
        # apply_node_renaming(
        #    parallel_gm, self.params_len, self.buffer_len, self.metadata
        # )
        self.parallel_gm = parallel_gm
        update_joint_with_descriptors(self.joint_with_descriptors, parallel_gm)
        fix_scatter_on_aliased_inputs(parallel_gm.graph)
        # Allow DCE to remove unused wait_tensor nodes in the backward graph.
        # Pushed onto self.stack so it's restored in AutoParallel.__exit__.
        self.stack.enter_context(_suppress_wait_tensor_side_effect())
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

    def apply_placement(self, sharding_placement):
        sharded_param_dict, sharded_buffer_dict = self._apply_placement_common(
            sharding_placement
        )

        mark_fsdp_all_gather_recomputation(
            self.parallel_gm.graph, self.reshard_after_forward
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
        num_primals = len(fw_metadata.input_info)
        fwd_only_gm = extract_forward_graph(
            self.parallel_gm, num_fwd_outputs, num_primals
        )
        compiler_fn = self.compiler_fn
        aot_config = self.joint_with_descriptors._aot_state.aot_config
        out_spec = self.joint_with_descriptors.out_spec

        # Build inference function eagerly so that Dynamo doesn't try
        # to trace into compiler_fn / RuntimeWrapper.post_compile (both
        # on its skip list, causing graph breaks under fullgraph=True).
        _fwd_example_inputs = [
            n.meta["val"] for n in fwd_only_gm.graph.nodes if n.op == "placeholder"
        ]
        _compiled_fwd = compiler_fn(fwd_only_gm, _fwd_example_inputs)

        from torch._functorch._aot_autograd.runtime_wrappers import RuntimeWrapper

        _wrapped_fwd = RuntimeWrapper(
            indices_of_inps_to_detach=[],
            trace_joint=False,
            disable_amp=False,
        ).post_compile(_compiled_fwd, aot_config, runtime_metadata=fw_metadata)

        @torch._dynamo.nonstrict_trace
        def _inference_fn(args):
            flat_outs = _wrapped_fwd(args)
            return torch.utils._pytree.tree_unflatten(flat_outs, out_spec)

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

        expected_inputs, dynamic_dims = _compute_expected_inputs(
            self._traced_inputs, solved_input_placements, self.mesh
        )

        def forward(self, *args, **kwargs):
            # Flatten pytree args (e.g. dicts, nested structures) to tensor
            # leaves, matching how Dynamo flattened the inputs during tracing.
            flat_args, _ = torch.utils._pytree.tree_flatten(args)
            if len(flat_args) != len(expected_inputs):
                flat_args, _ = torch.utils._pytree.tree_flatten((args, kwargs))
            _check_forward_args(flat_args, expected_inputs, dynamic_dims)
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
                out = _inference_fn(boxed_args)
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


##################
# Simple API end #
##################
