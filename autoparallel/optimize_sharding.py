# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Sharding optimization using Integer Linear Programming (ILP).

This module solves the optimal sharding strategy problem by formulating it as an ILP
where each binary variable x_{i,a,o,j} ∈ {0,1} represents a choice of input placement j
and output placement o for operation i and argument a. The objective minimizes total cost:

    minimize: Σ_{i,a,o,j} c_{i,a,o,j} * x_{i,a,o,j}

where:
- x_{i,a,o,j}: binary decision variable (1 if strategy selected, 0 otherwise)
- c_{i,a,o,j}: total cost (communication + computation) for this strategy choice

subject to the following constraint categories:

1. UNIQUENESS CONSTRAINTS: Each operation-argument pair must select exactly one
   input-output placement combination.

   ∀i,a: Σ_{o,j} x_{i,a,o,j} = 1

   → Implemented in: add_unique_decision_constraint()

2. CONSISTENCY CONSTRAINTS: For multi-argument operations, all arguments must agree
   on the same output placement to ensure the operation can execute correctly.

   ∀i,o: Σ_j x_{i,0,o,j} = Σ_j x_{i,1,o,j} = ... = Σ_j x_{i,A_i-1,o,j}
   where A_i is the number of arguments for operation i.

   → Implemented in: add_same_output_across_args_constraint()

3. FLOW CONSTRAINTS: The output placement of producer operations must match the
   input placement of consumer operations (dataflow consistency).

   ∀(i→k): Σ_j x_{i,0,o,j} = Σ_j x_{k,a,j,o}
   where operation i feeds into operation k at argument position a.

   → Implemented in: add_output_input_consistent_constraint()

4. COST CONSTRAINTS: Variables with infinite cost (invalid configurations) are
   forced to zero. Efficiency penalties for inefficient collective operations
   (e.g., non-batch-dim shard-to-replicate) are embedded in the cost coefficients
   computed by the cost model (see cost_models/collective_runtime_estimation.py).

   ∀i,a,o,j: c_{i,a,o,j} = ∞ ⟹ x_{i,a,o,j} = 0

   → Implemented in: add_inf_cost_constraint()

5. USER CONSTRAINTS (optional): Force specific placements for inputs, outputs,
   parameters, or memory usage bounds.

   5a. Input/Output constraints: x_{i,a,o*,j*} = 1 for specified (o*,j*)
       → Implemented in: add_sharded_input_constraint(), add_sharded_output_constraint()

   5b. Memory constraints: Σ_{params} (size_ratio * x_{param}) ≤ memory_limit
       → Implemented in: add_parameter_memory_constraint()

   5c. Forward-backward consistency: x_{fwd} = x_{bwd} for paired nodes
       → Implemented in: add_forward_backward_consistency_constraints()

   5d. General node constraints: Force specific placement for any node
       → Implemented in: add_node_constraint()

The solver finds the globally optimal sharding strategy that minimizes total
runtime cost while satisfying all constraints.
"""

import contextlib
import logging
import math
import operator
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import pulp
import torch
from torch._functorch._aot_autograd.descriptors import PlainAOTInput, PlainAOTOutput
from torch._functorch._aot_autograd.fx_utils import (
    get_param_and_grad_nodes,
    get_param_nodes,
    get_plain_input_and_grad_nodes,
    get_plain_output_and_tangent_nodes,
)
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import OpSpec, OpStrategy
from torch.distributed.tensor.placement_types import Placement, Replicate, Shard
from torch.utils._pytree import tree_map_only

from .cost_models.collective_runtime_estimation import estimate_strategy_comms_cost
from .cost_models.compute_estimation import (
    _get_sharded_shape_stride,
    estimate_strategy_runtime_cost,
)
from .graph_passes.graph_clustering import get_identical_regions
from .graph_passes.graph_utils import (
    all_input_nodes,
    build_param_derived_set,
    build_terminal_derived_set,
)
from .shardings.placement_options import (
    get_placement_options_for_node,
    reset_placement_options_cache,
)
from .shardings.propagation_rules import (
    DTensorSpecSeed,
    OpSpecSeed,
    _create_all_options,
    remove_invalid_configs,
    set_current_seed_node,
)

logger = logging.getLogger(__name__)

# Supported cost models are nonnegative; invalid variables are fixed to zero.
_INVALID_COST = -1.0

# Strategy enumeration fills each OpSpec's redistribute_cost via torch's
# generate_redistribute_costs (an expensive per-strategy redistribute-plan
# computation, the dominant cost of build on large/3D meshes). But
# _build_decision_vars overwrites every edge with the NCCL-aware
# estimate_strategy_comms_cost, and nothing reads the enumeration costs in
# between (remove_invalid_configs / keep_unique_configs select on placements/
# shapes only). So during enumeration we replace torch's redistribute_cost with
# a structure-preserving dummy to skip the wasted work; the final decision_vars
# are byte-identical. Autoparallel's own cost model uses a separate
# redistribute_cost (collective_runtime_estimation) and is unaffected.


@contextlib.contextmanager
def _skip_enumeration_redistribute_cost(enabled):
    if not enabled:
        yield
        return

    try:
        import torch.distributed.tensor._ops.utils as dt_utils

        original = dt_utils.redistribute_cost
    except (ImportError, AttributeError):
        yield
        return

    dt_utils.redistribute_cost = lambda *args, **kwargs: 0.0
    try:
        yield
    finally:
        dt_utils.redistribute_cost = original


def concretize_symint(val):
    """Concretize a SymInt to a plain int, pass through other values.

    For unbacked SymInts (hint=None), returns the SymInt unchanged.
    """
    if isinstance(val, torch.SymInt):
        hint = val.node.hint
        return hint if hint is not None else val
    return val


def concretize_args(args):
    """Concretize all SymInts and symbolic FakeTensors in an args tree.

    Returns an args tree where:
    - SymInts are replaced with their concrete hint values (via concretize_symint)
    - FakeTensors with symbolic shapes are replaced with concrete FakeTensors
    - Other values are left unchanged
    """
    from torch._subclasses.fake_tensor import FakeTensor

    def concretize(x):
        if isinstance(x, torch.SymInt):
            return concretize_symint(x)
        if isinstance(x, FakeTensor) and any(
            isinstance(s, torch.SymInt) for s in x.shape
        ):
            concrete_shape = [concretize_symint(s) for s in x.shape]
            concrete_stride = [concretize_symint(s) for s in x.stride()]
            if any(not isinstance(s, int) for s in concrete_shape):
                return x
            with x.fake_mode:
                return torch.empty_strided(
                    concrete_shape, concrete_stride, dtype=x.dtype, device=x.device
                )
        return x

    return tree_map_only((torch.SymInt, FakeTensor), concretize, args)


def _produces_tensor(val):
    """Check if a node's meta value represents tensor output(s)."""
    if isinstance(val, torch.Tensor):
        return True
    if isinstance(val, (tuple, list)):
        return any(_produces_tensor(v) for v in val)
    return False


def _materialize_seed_spec(spec, mesh):
    if spec is None:
        return None
    if isinstance(spec, DTensorSpecSeed):
        return DTensorSpec(mesh, spec.placements, tensor_meta=spec.tensor_meta)
    return tuple(_materialize_seed_spec(item, mesh) for item in spec)


def _spec_matches_seed(spec, seed) -> bool:
    if seed is None:
        return spec is None
    if isinstance(seed, DTensorSpecSeed):
        return isinstance(spec, DTensorSpec) and spec.placements == seed.placements
    return (
        isinstance(spec, (tuple, list))
        and len(spec) == len(seed)
        and all(
            _spec_matches_seed(actual, expected) for actual, expected in zip(spec, seed)
        )
    )


def _strategy_matches_seed(strategy: OpSpec, seed: OpSpecSeed) -> bool:
    return _spec_matches_seed(
        strategy.output_specs, seed.output_specs
    ) and _spec_matches_seed(strategy.input_specs, seed.input_specs)


def _first_seed_output(seed: OpSpecSeed) -> DTensorSpecSeed | None:
    if isinstance(seed.output_specs, DTensorSpecSeed):
        return seed.output_specs
    if isinstance(seed.output_specs, tuple):
        return next(
            (spec for spec in seed.output_specs if isinstance(spec, DTensorSpecSeed)),
            None,
        )
    return None


def _seed_spec_is_well_formed(spec: DTensorSpecSeed, mesh_ndim: int) -> bool:
    if len(spec.placements) != mesh_ndim or spec.tensor_meta is None:
        return False
    rank = len(spec.tensor_meta.shape)
    return all(
        not isinstance(placement, Shard) or -rank <= placement.dim < rank
        for placement in spec.placements
    )


def concretize_gm(gm):
    """Create a structural copy of gm with all symbolic shapes concretized.

    Returns (concrete_gm, orig_to_concrete, concrete_to_orig) where:
    - concrete_gm has identical graph structure but concrete metadata
    - orig_to_concrete maps original nodes → concrete nodes
    - concrete_to_orig maps concrete nodes → original nodes

    The concretized graph is used for the optimization pipeline (ILP solver,
    placement options, cost estimation) which needs concrete shapes. The
    original graph is preserved for apply_sharding which needs symbolic shapes
    for runtime flexibility.
    """

    concrete_graph = torch.fx.Graph()
    orig_to_concrete: dict[torch.fx.Node, torch.fx.Node] = {}

    for node in gm.graph.nodes:
        new_node = concrete_graph.node_copy(node, lambda n: orig_to_concrete[n])
        orig_to_concrete[node] = new_node
        # node_copy does copy.copy(node.meta), so new_node.meta is a shallow
        # copy. Concretize meta["val"] in-place on the copy.
        val = new_node.meta.get("val")
        if val is not None:
            new_node.meta["val"] = concretize_args(val)

    concrete_gm = torch.fx.GraphModule(gm, concrete_graph)
    concrete_to_orig = {v: k for k, v in orig_to_concrete.items()}
    return concrete_gm, orig_to_concrete, concrete_to_orig


@dataclass(slots=True)
class DecisionVar:
    """A decision variable in the ILP, representing one (node, arg, output_placement,
    input_placement) choice with its associated costs and strategy metadata.

    slots=True: there are millions of these on large/3D meshes, so dropping the
    per-instance __dict__ materially cuts both build time and memory."""

    var: Any  # pulp.LpVariable
    cost: float
    compute_cost: float
    comm_cost: float
    sharding_transition_cost: float
    strategy: Any  # OpSpec
    output_spec: Any  # DTensorSpec or tuple[DTensorSpec | None, ...]
    input_spec: Any  # DTensorSpec


@dataclass
class LPRelaxationResult:
    objective: float
    status: str
    solve_s: float
    total_s: float


def _assert_has_tensor_meta(spec_or_specs, node, label):
    """Assert that all DTensorSpecs in a spec (possibly a tuple) have tensor_meta."""
    if isinstance(spec_or_specs, (list, tuple)):
        for spec in spec_or_specs:
            if spec:
                assert (
                    spec.tensor_meta is not None
                ), f"{node} {label} doesn't have a tensor_meta"
    elif spec_or_specs is not None:
        assert (
            spec_or_specs.tensor_meta is not None
        ), f"{node} {label} doesn't have a tensor_meta"


class ShardingOptimizer:
    def __init__(
        self,
        gm,
        mesh,
        force_grad_reduce_in_higher_precision=False,
        repeated_subgraphs=False,
        fast_build=True,
        build_pulp=True,
        build_costs=True,
        strategy_seed=None,
        strategy_radius=None,
    ):
        self.orig_gm = gm
        self.fast_build = fast_build
        # When False, skip building the per-edge decision-var costs (Phase A of
        # _build_decision_vars, the comm-cost estimate over millions of edges that
        # dominates build). Strategies + cluster_links are still built; the
        # approximate solver computes the costs it needs on demand (lazy build),
        # only for the branches TRW-S actually touches. Implies build_pulp=False
        # (a PuLP problem needs the costs in its objective).
        self.build_costs = build_costs
        # When False, skip creating PuLP variables and constraints entirely.
        # decision_var costs + strategies + cluster_links are still built, which
        # is all the approximate solver needs (it derives the constraint topology
        # directly). This avoids constructing millions of PuLP objects on large /
        # 3D meshes, where that dominates build time.
        self.build_pulp = build_pulp and build_costs
        self.prob = None
        # The optimizer works on a concretized copy of the graph where all
        # symbolic shapes are replaced with their concrete hint values. This
        # centralizes dynamic-shape handling: the optimization pipeline
        # (placement options, cost estimation, clustering, ILP) only sees
        # concrete metadata, while the original symbolic graph is preserved
        # for apply_sharding.
        concrete_gm, self._orig_to_concrete, self._concrete_to_orig = concretize_gm(gm)
        self.gm = concrete_gm
        self.graph = concrete_gm.graph
        self.mesh = mesh
        self.force_grad_reduce_in_higher_precision = (
            force_grad_reduce_in_higher_precision
        )
        self.strategy_seed = strategy_seed
        self.strategy_radius = strategy_radius
        self._constraint_log: list[tuple[str, dict]] = []
        self._memory_constraint: tuple[float, float] | None = None
        # Maps ILP constraint name → node_name for active node constraints,
        # so that _apply_memory_constraint can exclude constrained params and
        # remove_constraints can keep this in sync.
        self._node_constraint_names: dict[str, str] = {}
        self._name_counters: dict[str, int] = {}
        # Set by _build_decision_vars: the (node, arg, out, inp) keys whose
        # strategy edge has finite cost. Invalid (infinite-cost) edges are
        # pruned and get no variable. None means "no pruning filter".
        self._valid_keys: set[tuple] | None = None
        self.profile: dict[str, Any] = {"timings": {}}
        t_init_start = time.perf_counter()
        t0 = time.perf_counter()
        if self.strategy_seed is None:
            self.strats = self.build_sharding_metadata()
        else:
            from .shardings import propagation_rules as _propagation_rules

            _propagation_rules.set_strategy_seed(
                self.strategy_seed, self.strategy_radius
            )
            reset_placement_options_cache()
            try:
                self.strats = self.build_sharding_metadata()
            finally:
                _propagation_rules.set_strategy_seed(None, None)
                set_current_seed_node(None)
                reset_placement_options_cache()
        t_strategy = time.perf_counter() - t0
        self.profile["timings"]["strategy_enumeration_s"] = t_strategy
        logger.info("ShardingOptimizer: strategy enumeration took %.3fs", t_strategy)
        # nodes/node_map are derived from strats (not graph.nodes) so that
        # shape-computation nodes skipped by build_sharding_metadata don't
        # appear and indices stay consistent.
        self.nodes = list(self.strats.keys())
        self.node_map = {node: i for i, node in enumerate(self.nodes)}
        logger.debug("Placement options took %.3fs", t_strategy)
        from autoparallel.shardings.placement_options import get_placement_options_timer

        get_placement_options_timer().report()

        # Node-level: cluster-copy node idx -> root node idx (option indices are
        # identical between copy and root; resolved on demand via
        # _cluster_root_key / _linked_option_keys).
        self.cluster_links: dict[int, int] = {}
        self._root_to_copies: dict[int, list[int]] = defaultdict(list)

        if repeated_subgraphs:
            t = time.time()
            clusters = get_identical_regions(self.gm.graph, self.strats)
            logger.debug(f"Found {len(clusters)} clusters in {time.time() - t:.2f}s")
            self.create_cluster_links(clusters)

        t0 = time.perf_counter()
        self.decision_vars = self._build_decision_vars()
        t1 = time.perf_counter()
        self.validate()
        t2 = time.perf_counter()
        if self.build_pulp:
            self.prob = pulp.LpProblem("AutoParallel", pulp.LpMinimize)
            self.add_default_constraints()
        t3 = time.perf_counter()
        self.profile["timings"].update(
            {
                "decision_var_build_s": t1 - t0,
                "validation_s": t2 - t1,
                "constraint_construction_s": t3 - t2,
                "init_total_s": t3 - t_init_start,
            }
        )
        n_constraints = len(self.prob.constraints) if self.prob is not None else 0
        logger.info(
            "ShardingOptimizer: build done in %.3fs (%d unique vars, %d decision "
            "vars, %d constraints)",
            t3 - t_init_start,
            len(self.pulp_variables),
            len(self.decision_vars),
            n_constraints,
        )

    @staticmethod
    def _safe_float(value):
        try:
            return float(value)
        except Exception:
            return math.nan

    def _get_next_name(self, prefix):
        idx = self._name_counters.setdefault(prefix, 0)
        self._name_counters[prefix] += 1
        return prefix + f"_{idx:03}"

    def _normalize_node(self, node):
        """Map a node to its concrete-graph counterpart.

        Public methods that accept nodes should call this so callers can pass
        nodes from either the original or the concrete graph.
        """
        return self._orig_to_concrete.get(node, node)

    def _built_input_nodes(self, node, strats):
        result = []
        for input_node in all_input_nodes(node):
            if input_node in strats:
                result.append(input_node)
            elif input_node.op != "get_attr":
                value = input_node.meta.get("val")
                assert not isinstance(value, torch.Tensor), (
                    f"Tensor-producing node {input_node} (op={input_node.op}) "
                    "unexpectedly missing from strats"
                )
        return result

    def _ensure_seed_witness(self, node, op_strategy, strats):
        if self.strategy_seed is None:
            return op_strategy
        node_seed = self.strategy_seed.get(node.name)
        if node_seed is None or node_seed.witness is None:
            return op_strategy
        witness = node_seed.witness
        if any(
            _strategy_matches_seed(strategy, witness)
            for strategy in op_strategy.strategies
        ):
            return op_strategy

        first_output = _first_seed_output(witness)
        if (
            first_output is None
            or first_output.placements != node_seed.output_placements
            or not all(
                _seed_spec_is_well_formed(spec, self.mesh.ndim)
                for spec in self._seeded_specs(witness)
            )
        ):
            logger.debug("Skipping structurally invalid strategy seed for %s", node)
            return op_strategy

        input_nodes = self._built_input_nodes(node, strats)
        input_specs = witness.input_specs
        if input_nodes:
            if input_specs is None or len(input_specs) != len(input_nodes):
                logger.debug("Skipping input-incompatible strategy seed for %s", node)
                return op_strategy
            redistribute_cost = [
                [0.0] * len(strats[input_node].strategies) for input_node in input_nodes
            ]
        elif input_specs:
            if len(input_specs) != 1:
                logger.debug("Skipping input-incompatible strategy seed for %s", node)
                return op_strategy
            redistribute_cost = [[0.0]]
        else:
            redistribute_cost = []

        candidate = OpSpec(
            output_specs=_materialize_seed_spec(witness.output_specs, self.mesh),
            input_specs=_materialize_seed_spec(input_specs, self.mesh),
            redistribute_cost=redistribute_cost,
        )
        valid = remove_invalid_configs(OpStrategy([candidate]), self.mesh)
        if not valid.strategies:
            logger.debug("Skipping unshardable strategy seed for %s", node)
            return op_strategy
        op_strategy.strategies.append(candidate)
        return op_strategy

    @staticmethod
    def _seeded_specs(witness):
        output_specs = witness.output_specs
        if isinstance(output_specs, DTensorSpecSeed):
            yield output_specs
        elif isinstance(output_specs, tuple):
            yield from (spec for spec in output_specs if spec is not None)
        if witness.input_specs is not None:
            yield from (spec for spec in witness.input_specs if spec is not None)

    def build_sharding_metadata(self):
        strats = {}
        # Enumeration's redistribute_cost matrices are overwritten with real
        # costs in _build_decision_vars, so skip computing them here (see
        # _skip_enumeration_redistribute_cost).
        with _skip_enumeration_redistribute_cost(self.fast_build):
            for node in self.graph.nodes:
                set_current_seed_node(node.name)
                if node.op in ("placeholder", "get_attr"):
                    val = node.meta.get("val")
                    if isinstance(val, torch.Tensor):
                        strats[node] = _create_all_options(
                            self.mesh, val.shape, tensor=val
                        )
                    elif node.op == "placeholder":
                        # Non-tensor placeholders (e.g. baked-in booleans/strings):
                        # keep them in strats with empty-shape replicate options
                        # so the constraint system can reference them.
                        strats[node] = _create_all_options(self.mesh, ())
                    else:
                        # Non-tensor get_attr: GraphModule submodules used by
                        # HOPs — not added to strats, invisible to the ILP.
                        # _all_input_nodes filters them.
                        assert node.op == "get_attr"
                        assert any(
                            isinstance(u.target, torch._ops.HigherOrderOperator)
                            or "local_map" in u.name
                            for u in node.users
                        ), f"Non-tensor get_attr {node} is not used by a HOP"
                elif node.op == "call_function":
                    if not _produces_tensor(node.meta.get("val")):
                        # Shape-computation nodes (sym_size, operator.mul, etc.)
                        # produce scalars, not tensors — skip sharding.
                        continue
                    user_strats = tree_map_only(
                        torch.fx.Node,
                        lambda x: strats.get(x, x.meta.get("val")),
                        node.args,
                    )
                    user_args = tree_map_only(
                        torch.fx.Node, lambda x: x.meta.get("val"), node.args
                    )
                    user_kwargs = tree_map_only(
                        torch.fx.Node, lambda x: x.meta.get("val"), node.kwargs
                    )
                    strats[node] = get_placement_options_for_node(
                        self.mesh, node, user_strats, user_args, user_kwargs
                    )
                elif node.op == "output":
                    user_strats = tree_map_only(
                        torch.fx.Node, lambda x: strats[x], node.args
                    )
                    strats[node] = user_strats
                else:
                    raise ValueError(f"Unexpected node op: {node.op}")
                if node in strats and node.op != "output":
                    strats[node] = self._ensure_seed_witness(node, strats[node], strats)
        set_current_seed_node(None)
        return strats

    def create_cluster_links(self, clusters):
        """Map each cluster-copy node to its root node (node-level). The optimizer
        reuses the root's decision variable for every copy, and the per-(arg, out,
        inp) option index is identical between a copy and its root, so we store
        only the node->node map and reconstruct option keys on demand (see
        _cluster_root_key / _linked_option_keys). Materializing one dict entry per
        option-tuple instead costs tens of millions of entries (and seconds of
        build time) on large/3D meshes."""
        for cluster_group in clusters:
            cluster0 = cluster_group[0]
            for cluster_i in cluster_group[1:]:
                for n0, ni in zip(cluster0, cluster_i):
                    assert len(self.strats[n0].strategies) == len(
                        self.strats[ni].strategies
                    ), (
                        f"Problem with graph clustering: {n0} and {ni} don't have "
                        "the same number of strategies. Please report a bug"
                    )
                    self.cluster_links[self.node_map[ni]] = self.node_map[n0]

    def _cluster_root_key(self, key):
        """Resolve an option key to its cluster-root option key, or return it
        unchanged when the node is not a cluster copy. The (arg, out, inp) indices
        are identical between a copy and its root."""
        root_idx = self.cluster_links.get(key[0])
        return key if root_idx is None else (root_idx, key[1], key[2], key[3])

    def _linked_option_keys(self, root_key):
        """The option keys on the cluster copies of root_key's node (each a mirror
        of root_key with the copy's node index). A copy mirrors its root's
        per-option validity, so callers pass valid root keys only."""
        copies = self._root_to_copies.get(root_key[0])
        if not copies:
            return ()
        return [(c, root_key[1], root_key[2], root_key[3]) for c in copies]

    def _all_input_nodes(self, node):
        """Variant of node.all_input_nodes that preserves duplicate nodes.

        Filters out nodes not in self.strats:
        - get_attr: HOP submodule nodes (GraphModules)
        - call_function producing non-tensors: shape-computation nodes
          (sym_size, operator.mul, etc.)
        """
        return self._built_input_nodes(node, self.strats)

    def walk_over_options(self, node, constrain_arg=None):
        """Yield (argi, out_idx, inp_idx) for all valid strategy combinations."""
        if node not in self.strats:
            return
        op_strategy = self.strats[node]
        for argi in range(len(op_strategy.strategies[0].input_specs)):
            if constrain_arg is not None and argi != constrain_arg:
                continue
            for out_idx, strategy in enumerate(op_strategy.strategies):
                for inp_idx in range(len(strategy.redistribute_cost[argi])):
                    yield argi, out_idx, inp_idx

    def _create_pulp_variables(self, variable_category=pulp.LpBinary):
        """Create PuLP variables for all decision points, resolving cluster
        links so that identical nodes share the same variable.

        Returns a dict mapping root (node_idx, argi, out_idx, inp_idx) keys
        to their PuLP variables. Linked keys are not stored; use
        _get_pulp_variable() to resolve them through cluster_links.

        Keys whose strategy is invalid (infinite cost) are pruned: if
        self._valid_keys is set, only those keys get a variable. These
        variables would otherwise be forced to zero by an inf-cost
        constraint, so skipping them shrinks the ILP without changing the
        optimum (see _build_decision_vars).
        """
        if variable_category not in {pulp.LpBinary, pulp.LpContinuous}:
            raise ValueError(
                f"Unsupported variable_category={variable_category!r}; "
                "expected pulp.LpBinary or pulp.LpContinuous"
            )
        cluster_linked_node_idxs = set(self.cluster_links)

        pulp_variables = {}
        for node, _ in self.strats.items():
            if node.op == "output":
                continue
            node_idx = self.node_map[node]
            if node_idx in cluster_linked_node_idxs:
                continue
            for argi, out_idx, inp_idx in self.walk_over_options(node):
                key = (node_idx, argi, out_idx, inp_idx)
                if self._valid_keys is not None and key not in self._valid_keys:
                    continue
                root_node = self.nodes[node_idx]
                bounds = (
                    {"lowBound": 0, "upBound": 1}
                    if variable_category == pulp.LpContinuous
                    else {}
                )
                pulp_variables[key] = pulp.LpVariable(
                    f"n={root_node},s={node_idx},arg={argi},"
                    f"output_p={out_idx},input_p={inp_idx}",
                    cat=variable_category,
                    **bounds,
                )

        return pulp_variables

    def _get_pulp_variable(self, key):
        """Look up the PuLP variable for a key, resolving through
        cluster_links if the key belongs to a linked node.

        Returns None if the key was pruned (invalid/infinite-cost strategy).
        """
        root_key = self._cluster_root_key(key)
        return self.pulp_variables.get(root_key)

    def _compute_edge_costs(
        self,
        node,
        output_strategy,
        argi,
        inp_idx,
        default_comm_cost,
        producer_strategy,
    ):
        """Compute communication and sharding transition costs for transitioning
        from input placement inp_idx to the output_strategy's expected input at argi.

        The transition cost is a tie-breaker that penalizes unnecessary redistributions:
        when multiple redistribution chains have the same communication cost, we prefer
        fewer individual redistributions so they can be fused into single collectives.

        Returns (comm_cost, sharding_transition_cost).
        """
        comm_cost = default_comm_cost
        sharding_transition_cost = 0

        if producer_strategy is not None:
            src_spec = producer_strategy.strategies[inp_idx].output_specs
            # TODO: operator.getitem being special is something
            # we might want to change in the future
            if node.target == operator.getitem:
                src_spec = src_spec[node.args[1]]
            tgt_spec = output_strategy.input_specs[argi]
            if isinstance(src_spec, DTensorSpec) and isinstance(tgt_spec, DTensorSpec):
                comm_cost = estimate_strategy_comms_cost(src_spec, tgt_spec)
                if src_spec.placements != tgt_spec.placements:
                    sharding_transition_cost = 1

        return comm_cost, sharding_transition_cost

    def _build_decision_vars(self):
        """Build DecisionVar entries for every (node_idx, argi, out_idx, inp_idx)
        combination in the strategy space.

        Strategy edges whose total cost is infinite (invalid redistributions)
        are pruned: no variable is created for them. Such a variable would be
        forced to zero by an inf-cost constraint anyway, so dropping it leaves
        the optimum unchanged while removing ~30% of the variables and the
        corresponding ~80% of constraints that are pure ``var == 0`` bounds.

        When ``build_pulp`` is False (approximate solver only) no PuLP variables
        are created (``DecisionVar.var`` is None); the valid-key set is still
        built so the approximate solver can treat a key absent from
        ``decision_vars`` as forbidden.
        """
        if not self.build_costs:
            return self._build_decision_vars_lazy()

        # Precompute which node indices are cluster-linked so we can
        # copy costs from the root instead of recomputing them.
        self._cluster_linked_node_idxs = set(self.cluster_links)

        t_compute = 0.0
        t_edge = 0.0
        n_vars = 0
        n_pruned = 0
        n_cluster_copied = 0

        t_pulp_start = time.perf_counter()
        self.pulp_variables = {}
        self._valid_keys = set()
        decision_vars = {}
        strats_items = [
            (self.node_map[node], node, strat) for node, strat in self.strats.items()
        ]

        # Phase A: compute every root node's per-edge costs.
        root_idxs = [
            node_idx
            for node_idx, node, _ in strats_items
            if node.op != "output" and node_idx not in self._cluster_linked_node_idxs
        ]
        tc0 = time.perf_counter()
        node_results = [self._compute_node_edge_costs(idx) for idx in root_idxs]
        t_edge = time.perf_counter() - tc0

        # Phase B: assemble decision vars (and PuLP variables) from the computed
        # costs. Serial because PuLP vars and DecisionVars hold parent-owned
        # strategy objects; byte-identical to computing the costs inline. This
        # also writes the real costs back into each strat's redistribute_cost
        # (overwriting the enumeration dummies) for the cluster batch-copy and
        # _compute_solution_cost readers below.
        for node_idx, out_data in node_results:
            node = self.nodes[node_idx]
            op_strategy = self.strats[node]
            for out_idx, (per_arg_compute, arg_rows) in enumerate(out_data):
                output_strategy = op_strategy.strategies[out_idx]
                for argi, redist_costs in enumerate(output_strategy.redistribute_cost):
                    input_spec = output_strategy.input_specs[argi]
                    for inp_idx, (comm_cost, transition_cost) in enumerate(
                        arg_rows[argi]
                    ):
                        redist_costs[inp_idx] = comm_cost
                        cost = comm_cost + per_arg_compute + transition_cost
                        # Prune invalid (infinite-cost) edges: no variable, no
                        # DecisionVar. A key absent from decision_vars is treated
                        # as forbidden by both the ILP and the approximate solver.
                        if not math.isfinite(cost):
                            n_pruned += 1
                            continue

                        key = (node_idx, argi, out_idx, inp_idx)
                        if self.build_pulp:
                            var = pulp.LpVariable(
                                f"n={node},s={node_idx},arg={argi},"
                                f"output_p={out_idx},input_p={inp_idx}",
                                cat=pulp.LpBinary,
                            )
                            self.pulp_variables[key] = var
                        else:
                            var = None
                        self._valid_keys.add(key)
                        decision_vars[key] = DecisionVar(
                            var=var,
                            cost=cost,
                            compute_cost=per_arg_compute,
                            comm_cost=comm_cost,
                            sharding_transition_cost=transition_cost,
                            strategy=output_strategy,
                            output_spec=output_strategy.output_specs,
                            input_spec=input_spec,
                        )
                        n_vars += 1

        # Batch-copy redistribute_cost from root strats to linked strats.
        # The root pass above updated redistribute_cost in place with
        # edge-computed costs; linked strats need the same values for
        # _compute_solution_cost and other readers.
        for linked_node_idx, root_node_idx in self.cluster_links.items():
            linked_node = self.nodes[linked_node_idx]
            root_node = self.nodes[root_node_idx]
            linked_op = self.strats[linked_node]
            root_op = self.strats[root_node]
            for out_idx in range(len(root_op.strategies)):
                root_spec = root_op.strategies[out_idx]
                linked_spec = linked_op.strategies[out_idx]
                linked_spec.redistribute_cost = [
                    list(costs) for costs in root_spec.redistribute_cost
                ]
        n_cluster_copied = len(self.cluster_links)

        # Root node idx -> [copy node idxs]. Option keys are reconstructed on
        # demand (see _linked_option_keys); a copy mirrors its root's per-option
        # validity, so no per-option filtering is needed here.
        self._root_to_copies = defaultdict(list)
        for copy_idx, root_idx in self.cluster_links.items():
            self._root_to_copies[root_idx].append(copy_idx)

        t_pulp_end = time.perf_counter()
        logger.debug(
            "_build_decision_vars breakdown (%d vars, %d pruned-inf, %d cluster-copied): "
            "build=%.3fs, compute_cost=%.3fs, edge_cost=%.3fs",
            len(decision_vars),
            n_pruned,
            n_cluster_copied,
            t_pulp_end - t_pulp_start,
            t_compute,
            t_edge,
        )
        self.profile["timings"]["cost_estimation_s"] = t_compute + t_edge
        return decision_vars

    def _build_decision_vars_lazy(self):
        """Lazy build (build_costs=False): skip the expensive per-edge cost
        estimation (Phase A) and the DecisionVar/PuLP materialization (Phase B).

        Strategies, cluster_links and the redistribute_cost *shape* (the
        enumeration dummies) are preserved, which is all the approximate solver
        needs to derive the topology; it computes the comm/compute costs it
        actually uses on demand. Returns an empty decision_vars dict — callers
        that read it (the approx solver) must route through their lazy provider.
        """
        self._cluster_linked_node_idxs = set(self.cluster_links)
        self.pulp_variables = {}
        self._valid_keys = None
        self._root_to_copies = defaultdict(list)
        for copy_idx, root_idx in self.cluster_links.items():
            self._root_to_copies[root_idx].append(copy_idx)
        self.profile["timings"]["cost_estimation_s"] = 0.0
        return {}

    def _compute_node_edge_costs(self, node_idx):
        """Compute per-edge costs for one root node."""
        node = self.nodes[node_idx]
        op_strategy = self.strats[node]
        num_args = len(op_strategy.strategies[0].input_specs)
        input_nodes = self._all_input_nodes(node)
        producer_strategies = [self.strats[n] for n in input_nodes]
        out_data = []
        for output_strategy in op_strategy.strategies:
            per_arg_compute = (
                estimate_strategy_runtime_cost(node, output_strategy) / num_args
            )
            arg_rows = []
            for argi, redist_costs in enumerate(output_strategy.redistribute_cost):
                producer_strategy = (
                    producer_strategies[argi]
                    if argi < len(producer_strategies)
                    else None
                )
                arg_rows.append(
                    [
                        self._compute_edge_costs(
                            node,
                            output_strategy,
                            argi,
                            inp_idx,
                            default_comm_cost,
                            producer_strategy,
                        )
                        for inp_idx, default_comm_cost in enumerate(redist_costs)
                    ]
                )
            out_data.append((per_arg_compute, arg_rows))
        return node_idx, out_data

    def _resolve_decision_var(self, key):
        """Return a DecisionVar for key, reconstructing on the fly for linked keys."""
        dv = self.decision_vars.get(key)
        if dv is not None:
            return dv
        node_idx, argi, out_idx, _ = key
        strategy = self.strats[self.nodes[node_idx]].strategies[out_idx]
        root_key = self._cluster_root_key(key)
        root_dv = self.decision_vars.get(root_key)
        if root_dv is None:
            # Lazy build (build_costs=False): no DecisionVars were materialized.
            # The approximate solver scores via its own lazy cost provider; here
            # only .strategy / specs are needed (solution extraction), so the
            # cost fields are zero placeholders.
            return DecisionVar(
                var=None,
                cost=0.0,
                compute_cost=0.0,
                comm_cost=0.0,
                sharding_transition_cost=0.0,
                strategy=strategy,
                output_spec=strategy.output_specs,
                input_spec=strategy.input_specs[argi],
            )
        return DecisionVar(
            var=self._get_pulp_variable(key) if self.pulp_variables else None,
            cost=root_dv.cost,
            compute_cost=root_dv.compute_cost,
            comm_cost=root_dv.comm_cost,
            sharding_transition_cost=root_dv.sharding_transition_cost,
            strategy=strategy,
            output_spec=strategy.output_specs,
            input_spec=strategy.input_specs[argi],
        )

    def _find_decision_var(self, node_idx, argi, out_idx):
        """Return a DecisionVar for any surviving inp_idx of (node, arg, out),
        or None if every edge for that output strategy was pruned.

        compute_cost is identical across inp_idx for a given out_idx, so callers
        that only need per-strategy costs can use whichever edge survived.
        """
        strategy = self.strats[self.nodes[node_idx]].strategies[out_idx]
        n_inp = (
            len(strategy.redistribute_cost[argi]) if strategy.redistribute_cost else 1
        )
        for inp_idx in range(n_inp):
            key = (node_idx, argi, out_idx, inp_idx)
            if key in self.decision_vars:
                return self._resolve_decision_var(key)
            if (
                key[0] in self.cluster_links
                and self._cluster_root_key(key) in self.decision_vars
            ):
                return self._resolve_decision_var(key)
        return None

    def _collect_vars(self, node, node_idx, argi, group_by, resolve_clusters=False):
        """Collect PuLP variables for a node's options, grouped by strategy index.

        Args:
            group_by: "out_idx" to group by output strategy index,
                      "inp_idx" to group by input strategy index.
            resolve_clusters: If False, skip cluster-linked keys. If True,
                              resolve through cluster_links to root variables.
        """
        result = {}
        for _, out_idx, inp_idx in self.walk_over_options(node, argi):
            key = (node_idx, argi, out_idx, inp_idx)
            if key[0] in self.cluster_links:
                if not resolve_clusters:
                    continue
                var = self.pulp_variables.get(self._cluster_root_key(key))
            else:
                var = self.pulp_variables.get(key)
            if var is None:  # pruned (invalid/infinite-cost) strategy edge
                continue
            group_key = out_idx if group_by == "out_idx" else inp_idx
            result.setdefault(group_key, []).append(var)
        return result

    def validate(self):
        for node in self.graph.nodes:
            if node.op != "call_function":
                continue
            if node not in self.strats:
                continue
            # Cluster copies are structurally identical to their root (same
            # strategies and input structure, asserted in create_cluster_links),
            # so validating the root covers them.
            if self.node_map[node] in self.cluster_links:
                continue
            strat = self.strats[node]
            strat0 = strat.strategies[0]
            all_input_nodes = self._all_input_nodes(node)
            num_input_nodes = len(all_input_nodes)
            if len(strat0.redistribute_cost) != num_input_nodes:
                # only constructor functions allowed here
                assert num_input_nodes == 0, f"{num_input_nodes}"
                assert (
                    len(strat0.redistribute_cost) == 1
                ), f"{len(strat0.redistribute_cost)}"
            assert (len(strat0.redistribute_cost) == num_input_nodes) or (
                num_input_nodes == 0 and len(strat0.redistribute_cost) == 1
            ), f"{node}, {len(strat0.redistribute_cost)}, {num_input_nodes}"

            _assert_has_tensor_meta(strat0.output_specs, node, "output_specs")
            for input_spec in strat0.input_specs:
                _assert_has_tensor_meta(input_spec, node, "input_spec")

            for i, arg in enumerate(all_input_nodes):
                strat_arg = self.strats[arg]
                num_arg_strats = len(strat_arg.strategies)
                assert (
                    len(strat0.redistribute_cost[i]) == num_arg_strats
                ), f"{node}, {len(strat0.redistribute_cost[i])}, {num_arg_strats}"

    # ---- Default constraints ----

    def add_unique_decision_constraint(self):
        """UNIQUENESS (Category 1): Each (operation, argument) pair selects exactly
        one input-output placement combination.

        ∀i,a: Σ_{o,j} x_{i,a,o,j} = 1
        """
        for node in self.graph.nodes:
            if node.op not in {"placeholder", "call_function", "get_attr"}:
                continue
            if node not in self.node_map:
                continue
            node_idx = self.node_map[node]
            if node_idx in self._cluster_linked_node_idxs:
                continue
            arg_vars = {}
            num_args = len(self.strats[node].strategies[0].input_specs)
            for argi, out_idx, inp_idx in self.walk_over_options(node):
                key = (node_idx, argi, out_idx, inp_idx)
                var = self.pulp_variables.get(key)
                if var is None:  # pruned (invalid) strategy edge
                    continue
                arg_vars.setdefault(argi, []).append(var)
            for argi in range(num_args):
                self.prob += (
                    pulp.lpSum(arg_vars.get(argi, [])) == 1,
                    self._get_next_name("unique_decision"),
                )

    def add_same_output_across_args_constraint(self):
        """CONSISTENCY (Category 2): For multi-argument operations, all arguments
        agree on the same output placement.

        ∀i,o: Σ_j x_{i,0,o,j} = Σ_j x_{i,1,o,j} = ... = Σ_j x_{i,A_i-1,o,j}
        """
        for node in self.graph.nodes:
            if node.op != "call_function":
                continue
            if node not in self.node_map:
                continue
            node_idx = self.node_map[node]
            if node_idx in self._cluster_linked_node_idxs:
                continue
            if len(self._all_input_nodes(node)) <= 1:
                continue
            # Group vars by (argi, out_idx). Pruning can leave an arg with no
            # vars for a given out_idx, so we key explicitly by out_idx rather
            # than relying on positional alignment: a missing entry means an
            # empty sum (== 0), which correctly forbids that output strategy.
            num_args = len(self._all_input_nodes(node))
            vars_per_output: dict[tuple[int, int], list] = {}
            for argi, out_idx, inp_idx in self.walk_over_options(node):
                key = (node_idx, argi, out_idx, inp_idx)
                var = self.pulp_variables.get(key)
                if var is None:  # pruned (invalid) strategy edge
                    continue
                vars_per_output.setdefault((argi, out_idx), []).append(var)
            all_out_idxs = {oi for (_, oi) in vars_per_output}
            for out_idx in all_out_idxs:
                arg0_eq = pulp.lpSum(vars_per_output.get((0, out_idx), []))
                for argi in range(1, num_args):
                    self.prob += (
                        arg0_eq == pulp.lpSum(vars_per_output.get((argi, out_idx), [])),
                        self._get_next_name("same_across_args"),
                    )

    def add_output_input_consistent_constraint(self):
        """FLOW (Category 3): The output placement of producer operations matches
        the input placement of consumer operations.

        ∀(i→k): Σ_j x_{i,0,o,j} = Σ_j x_{k,a,j,o}
        """
        for node in self.graph.nodes:
            if node.op == "output":
                continue
            if node not in self.node_map:
                continue
            node_idx = self.node_map[node]
            producer_is_linked = node_idx in self._cluster_linked_node_idxs
            # All args agree on the same output (ensured by consistency constraint),
            # so we use arg 0 for the producer side.
            for user in node.users:
                if user.op == "output":
                    continue
                if user not in self.node_map:
                    continue
                user_idx = self.node_map[user]
                # Skip edges where both endpoints are cluster-linked;
                # the root-to-root edge already covers this.
                if producer_is_linked and user_idx in self._cluster_linked_node_idxs:
                    continue
                user_argi = [
                    i for i, n in enumerate(self._all_input_nodes(user)) if n == node
                ]
                assert len(user_argi) >= 1
                # Use the first matching arg; the same-output-across-args
                # constraint already ensures all args agree.
                user_argi = user_argi[0]

                vars_producer = self._collect_vars(
                    node, node_idx, argi=0, group_by="out_idx"
                )
                if not vars_producer:
                    vars_producer = self._collect_vars(
                        node,
                        node_idx,
                        argi=0,
                        group_by="out_idx",
                        resolve_clusters=True,
                    )

                vars_consumer = self._collect_vars(
                    user, user_idx, argi=user_argi, group_by="inp_idx"
                )
                if not vars_consumer:
                    vars_consumer = self._collect_vars(
                        user,
                        user_idx,
                        argi=user_argi,
                        group_by="inp_idx",
                        resolve_clusters=True,
                    )

                # Skip edges where the consumer arg has no sharding decision
                # (e.g. None input_specs for HOP SymInt args).
                if not vars_consumer or not vars_producer:
                    user_strat = self.strats[user]
                    assert (
                        user_argi < len(user_strat.strategies[0].input_specs)
                        and user_strat.strategies[0].input_specs[user_argi] is None
                    ), (
                        f"Missing variables for non-None input_spec at "
                        f"{user}[{user_argi}]"
                    )
                    continue

                # Pruning can leave a producer output strategy with no matching
                # consumer var (the consumer cannot accept that placement) or
                # vice versa. Iterate the union and treat a missing side as an
                # empty sum (== 0): this forbids the unmatched output strategy,
                # exactly as the old inf-cost (== 0) variables did.
                for k in vars_producer.keys() | vars_consumer.keys():
                    self.prob += (
                        pulp.lpSum(vars_producer.get(k, []))
                        == pulp.lpSum(vars_consumer.get(k, [])),
                        self._get_next_name("output_input_consistent"),
                    )

    def add_inf_cost_constraint(self):
        """COST (Category 4): Variables with infinite cost (invalid configurations)
        are forced to zero.

        ∀i,a,o,j: c_{i,a,o,j} = ∞ ⟹ x_{i,a,o,j} = 0

        Freshly built optimizers prune these edges in _build_decision_vars, so
        no variable exists and this is a no-op. It still runs for optimizers
        loaded from save files produced before pruning was introduced, whose
        decision_vars may still contain infinite-cost entries.
        """
        for key, dv in self.decision_vars.items():
            if not math.isfinite(dv.cost):
                dv.cost = _INVALID_COST
                self.prob += (dv.var == 0, self._get_next_name("inf_cases"))

    def add_default_constraints(self):
        self.add_unique_decision_constraint()
        self.add_same_output_across_args_constraint()
        self.add_output_input_consistent_constraint()
        self.add_inf_cost_constraint()
        self.add_forward_backward_consistency_constraints()
        self.add_grad_reduce_dtype_constraints()

    # ---- Prefetch overlap ----

    def apply_prefetch_discount(self, scale=0.0):
        """Discount communication costs for prefetchable edges.

        Scales down comm costs for:
        - Forward: edges where the producer is parameter-derived (FSDP
          all-gathers that can be prefetched ahead of compute)
        - Backward: edges into terminal-derived nodes (gradient
          reduce-scatters that can overlap with earlier backward compute)

        Must be called before get_solution(). A scale of 0.0 means fully
        overlapped (free); 1.0 means no discount.

        Returns the number of decision vars modified.
        """
        param_derived = build_param_derived_set(self.graph)
        terminal_derived = build_terminal_derived_set(self.graph)

        logger.debug(
            "terminal_derived: %d param nodes, %d grad nodes",
            len(get_param_nodes(self.graph)),
            sum(
                1
                for _, grad in get_param_and_grad_nodes(self.graph).values()
                if grad is not None
            ),
        )

        n_modified = 0
        for key, dv in self.decision_vars.items():
            node_idx, argi, out_idx, inp_idx = key
            node = self.nodes[node_idx]

            input_nodes = self._all_input_nodes(node)
            if not input_nodes:
                continue
            producer = input_nodes[argi]

            is_prefetchable = producer in param_derived or node in terminal_derived
            if is_prefetchable and dv.comm_cost > 0 and math.isfinite(dv.comm_cost):
                dv.comm_cost *= scale
                dv.cost = dv.comm_cost + dv.compute_cost + dv.sharding_transition_cost
                n_modified += 1

        logger.debug(
            "apply_prefetch_discount(scale=%.2f): modified %d decision vars "
            "(%d param-derived nodes, %d terminal-derived nodes)",
            scale,
            n_modified,
            len(param_derived),
            len(terminal_derived),
        )
        return n_modified

    # ---- Solution ----

    def _set_objective(self):
        """Add the cost minimization objective to the ILP.

        Idempotent: a no-op if the objective has already been set. This lets the
        approximate solver populate ``prob.objective`` (so its assignment can be
        scored with ``pulp.value(prob.objective)``) without clobbering or
        double-adding it, and keeps repeated get_solution() calls safe.
        """
        if self.prob.objective is not None:
            return
        terms = []
        for key, dv in self.decision_vars.items():
            multiplier = 1 + len(self._root_to_copies.get(key[0], ()))
            terms.append(dv.var * dv.cost * multiplier)
        self.prob += pulp.lpSum(terms)

    def get_lower_bound(self, verbose=False):
        """Solve the LP relaxation and return a lower bound on the ILP objective.

        This relaxes the existing binary PuLP variables to continuous variables
        in [0, 1], solves the current problem with all constraints already added,
        then restores the optimizer state. The result is a certificate only:
        fractional LP values are not valid sharding placements.
        """
        t0 = time.perf_counter()
        old_objective = self.prob.objective
        old_status = self.prob.status
        old_sol_status = getattr(self.prob, "sol_status", None)
        old_selected_keys_marker = object()
        old_selected_keys = getattr(self, "selected_keys", old_selected_keys_marker)
        var_states = {
            var: (var.cat, var.lowBound, var.upBound, var.varValue)
            for var in self.pulp_variables.values()
        }

        try:
            if self.prob.objective is None:
                self._set_objective()
            # The relaxation must include the parameter-memory constraint, or it
            # is a lower bound on a different (unconstrained) problem and can fall
            # below the true ILP optimum.
            self._apply_memory_constraint()

            for var in self.pulp_variables.values():
                var.cat = pulp.LpContinuous
                var.lowBound = 0
                var.upBound = 1
                var.varValue = None

            solver = pulp.PULP_CBC_CMD(msg=verbose)
            t_solve0 = time.perf_counter()
            with tempfile.TemporaryDirectory() as tmpdir:
                solver.tmpDir = tmpdir
                self.prob.solve(solver)
            solve_s = time.perf_counter() - t_solve0

            status = pulp.LpStatus.get(self.prob.status, self.prob.status)
            objective = self._safe_float(pulp.value(self.prob.objective))
            result = LPRelaxationResult(
                objective=objective,
                status=status,
                solve_s=solve_s,
                total_s=time.perf_counter() - t0,
            )
            logger.info(
                "ShardingOptimizer LP relaxation: status=%s objective=%.4f "
                "solve=%.3fs",
                result.status,
                result.objective,
                result.solve_s,
            )
            return result
        finally:
            for var, (cat, low_bound, up_bound, value) in var_states.items():
                var.cat = cat
                var.lowBound = low_bound
                var.upBound = up_bound
                var.varValue = value
            self.prob.objective = old_objective
            self.prob.status = old_status
            if old_sol_status is None:
                if hasattr(self.prob, "sol_status"):
                    delattr(self.prob, "sol_status")
            else:
                self.prob.sol_status = old_sol_status
            if old_selected_keys is old_selected_keys_marker:
                if hasattr(self, "selected_keys"):
                    delattr(self, "selected_keys")
            else:
                self.selected_keys = old_selected_keys

    def _solve(self, verbose=False):
        self._apply_memory_constraint()
        # The sharding ILP is flow-like and its LP relaxation is usually integral.
        # Disable CBC's integer preprocessing; branch-and-bound still runs when
        # the relaxation is fractional.
        # Pass as a single string: PuLP prefixes each options entry with "-",
        # so this becomes the CBC flag "-preprocess off".
        solver = pulp.PULP_CBC_CMD(msg=verbose, options=["preprocess off"])
        # Use a dedicated temp directory for PuLP's intermediate files (.mps,
        # .sol, etc.) so they are always cleaned up, even if the process is
        # killed.  Without this, leftover files can fill up /tmp (tmpfs).
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmpdir:
            solver.tmpDir = tmpdir
            self.prob.solve(solver)
        solve_s = time.perf_counter() - t0

        status = pulp.LpStatus.get(self.prob.status, self.prob.status)
        if status != "Optimal":
            if status == "Infeasible":
                logger.warning(self.get_violated_constraints_log())
                raise RuntimeError(
                    "The sharding optimizer could not find a feasible solution. "
                    "This typically means the user-specified constraints are "
                    "contradictory or the device mesh is too small for the requested "
                    "sharding. Check the WARNING log above for the list of violated "
                    "constraints, and consider relaxing input/output constraints or "
                    "using a larger mesh."
                )
            raise RuntimeError(
                f"The sharding optimizer solve ended with status={status}"
            )

        self.selected_keys = [
            key for key, dv in self.decision_vars.items() if dv.var.value() == 1
        ]
        for root_key in list(self.selected_keys):
            self.selected_keys.extend(self._linked_option_keys(root_key))

        return solve_s

    def solve_lp_relaxation(self, verbose=False, frac_tol=1e-6, extract=False):
        """Solve the continuous relaxation of the ILP (binary variables relaxed
        to [0, 1]) and report diagnostics, restoring the binary categories on
        exit so a later ILP solve is unaffected.

        Returns a dict with the relaxation objective (a lower bound on the ILP
        optimum), the solve time, the number/fraction of decision variables that
        came out fractional, and the solver status.  This is the lens for
        understanding why constraints (e.g. propagated annotations) speed up the
        ILP: a relaxation that is tighter (objective closer to the ILP optimum)
        and less fractional leaves branch-and-bound far less work.

        For this sharding problem the relaxation is empirically integral, so the
        relaxation optimum equals the ILP optimum.  With ``extract=True`` and an
        integral solution, the dict also contains a ``"solution"`` key with the
        per-node strategy dict (same form as :meth:`get_solution`) — i.e. the LP
        relaxation can be used as a much cheaper exact solve, skipping
        branch-and-bound.  ``"solution"`` is ``None`` when the relaxation came
        out fractional.

        The objective is initialized on demand and its prior state is restored.
        """
        old_objective = self.prob.objective
        if old_objective is None:
            self._set_objective()
        variables = self.prob.variables()
        original_cats = [v.cat for v in variables]
        self._apply_memory_constraint()
        t0 = time.perf_counter()
        try:
            for v in variables:
                v.cat = pulp.LpContinuous  # bounds are already [0, 1] for binaries
            solver = pulp.PULP_CBC_CMD(msg=verbose)
            with tempfile.TemporaryDirectory() as tmpdir:
                solver.tmpDir = tmpdir
                self.prob.solve(solver)
            solve_time = time.perf_counter() - t0
            status = pulp.LpStatus.get(self.prob.status, self.prob.status)
            objective = pulp.value(self.prob.objective)
            n_fractional = 0
            n_vars = 0
            for v in variables:
                val = v.value()
                if val is None:
                    continue
                n_vars += 1
                if min(val, 1.0 - val) > frac_tol:
                    n_fractional += 1
            solution = None
            if extract and status != "Optimal":
                raise RuntimeError(
                    f"LP relaxation did not solve to optimality: status={status}"
                )
            if extract and n_fractional == 0:
                for v in variables:
                    value = v.value()
                    if value is not None:
                        v.varValue = float(value > 0.5)
                violated = [
                    name
                    for name, constraint in self.prob.constraints.items()
                    if not constraint.valid(frac_tol)
                ]
                if violated:
                    raise RuntimeError(
                        "Integral LP solution is infeasible after rounding; "
                        f"violated constraints: {violated}"
                    )
                self.selected_keys = [
                    key
                    for key, dv in self.decision_vars.items()
                    if dv.var.value() == 1.0
                ]
                for root_key in list(self.selected_keys):
                    self.selected_keys.extend(self._linked_option_keys(root_key))
                solution = self._to_orig_solution(self._extract_and_validate_solution())
        finally:
            for v, cat in zip(variables, original_cats):
                v.cat = cat
            self.prob.objective = old_objective
        result = {
            "objective": objective,
            "solve_time": solve_time,
            "n_fractional": n_fractional,
            "n_vars": n_vars,
            "status": status,
            "solution": solution,
        }
        self.profile["lp_relaxation"] = {
            key: value for key, value in result.items() if key != "solution"
        }
        return result

    def _extract_and_validate_solution(self):
        """Validate the ILP solution and return the optimal strategy per node."""
        selected_by_node = {}
        for key in self.selected_keys:
            node = self.nodes[key[0]]
            selected_by_node.setdefault(node, []).append(
                self._resolve_decision_var(key)
            )

        # Validate: each (node, arg) pair has exactly one selection
        seen = set()
        for key in self.selected_keys:
            node_arg = (key[0], key[1])
            if node_arg in seen:
                node = self.nodes[key[0]]
                raise RuntimeError(
                    f"Multiple solutions for {node}, key={node_arg}, "
                    f"solutions: {[str(dv.strategy) for dv in selected_by_node[node]]}"
                )
            seen.add(node_arg)

        expected = {
            (self.node_map[node], argi)
            for node, strategies in self.strats.items()
            if node.op != "output"
            for argi in range(len(strategies.strategies[0].input_specs))
        }
        missing = expected - seen
        if missing:
            node_idx, argi = min(missing)
            raise RuntimeError(
                f"Missing solution for {self.nodes[node_idx]}, key={(node_idx, argi)}"
            )

        # Validate: all args of a node agree on the same strategy
        for node, dvs in selected_by_node.items():
            assert all(
                dvs[0].strategy == dv.strategy for dv in dvs
            ), f"{[dv.var for dv in dvs]}: {[str(dv.strategy) for dv in dvs]}"

        return {node: dvs[0].strategy for node, dvs in selected_by_node.items()}

    def _to_orig_solution(self, solution):
        """Translate a solution from concrete-graph nodes to original-graph nodes."""
        if not self._concrete_to_orig:
            return solution
        return {self._concrete_to_orig[node]: spec for node, spec in solution.items()}

    def _to_concrete_solution(self, solution):
        """Translate a solution from original-graph nodes to concrete-graph nodes."""
        if not self._orig_to_concrete:
            return solution
        return {self._orig_to_concrete[node]: spec for node, spec in solution.items()}

    def get_solution(self, verbose=False):
        t0 = time.perf_counter()
        self._set_objective()
        solve_s = self._solve(verbose)
        obj_value = self._safe_float(pulp.value(self.prob.objective))
        solution = self._to_orig_solution(self._extract_and_validate_solution())
        total_s = time.perf_counter() - t0
        self.profile["ilp"] = {
            "objective": obj_value,
            "status": pulp.LpStatus.get(self.prob.status, self.prob.status),
            "solve_s": solve_s,
            "total_s": total_s,
        }
        logger.info("ILP solve took %.3fs (objective=%.4f)", total_s, obj_value)
        return solution

    def resolve(self, verbose=False):
        """Re-solve the ILP after adding or removing constraints.

        Unlike get_solution(), this does not re-set the objective, so it can
        be called multiple times after modifying constraints.
        """
        t0 = time.perf_counter()
        solve_s = self._solve(verbose)
        obj_value = self._safe_float(pulp.value(self.prob.objective))
        solution = self._to_orig_solution(self._extract_and_validate_solution())
        total_s = time.perf_counter() - t0
        self.profile["ilp"] = {
            "objective": obj_value,
            "status": pulp.LpStatus.get(self.prob.status, self.prob.status),
            "solve_s": solve_s,
            "total_s": total_s,
        }
        logger.info(
            "ILP re-solve took %.3fs (objective=%.4f)",
            total_s,
            obj_value,
        )
        return solution

    def remove_constraints(self, names):
        """Remove constraints by name, allowing re-solve to revert to the
        unconstrained optimum."""
        memory_names = {"memory_constraint_high", "memory_constraint_low"}
        for name in names:
            if self.prob is not None:
                del self.prob.constraints[name]
            elif name not in memory_names:
                raise RuntimeError("Named constraint removal requires an ILP/LP build")
            self._node_constraint_names.pop(name, None)
            if name in memory_names:
                self._memory_constraint = None
                self._constraint_log = [
                    entry
                    for entry in self._constraint_log
                    if entry[0] != "add_parameter_memory_constraint"
                ]

    def diff_solutions(self, solution_a, solution_b):
        """Compare two solutions and report placement changes and cost diffs.

        Args:
            solution_a: baseline solution dict (node → OpSpec), e.g. from get_solution()
            solution_b: counterfactual solution dict, e.g. from resolve()
        """
        from torch.distributed.tensor._op_schema import _pretty_print_spec

        changes = defaultdict(list)
        unchanged = 0

        for node in solution_a:
            if node not in solution_b:
                continue
            spec_a = solution_a[node].output_specs
            spec_b = solution_b[node].output_specs
            if not isinstance(spec_a, DTensorSpec) or not isinstance(
                spec_b, DTensorSpec
            ):
                continue
            if spec_a.placements == spec_b.placements:
                unchanged += 1
            else:
                key = (
                    _pretty_print_spec(spec_a),
                    _pretty_print_spec(spec_b),
                )
                changes[key].append(node)

        # Compute objective values from selected_keys for current solution
        # (solution_b is the current state after last solve)
        # Translate to concrete nodes for internal cost computation
        cost_a = self._compute_solution_cost(self._to_concrete_solution(solution_a))
        cost_b = self._compute_solution_cost(self._to_concrete_solution(solution_b))

        lines = []
        lines.append(
            f"Objective: {cost_a['total']:.1f} -> {cost_b['total']:.1f} "
            f"({cost_b['total'] - cost_a['total']:+.1f})"
        )
        lines.append(
            f"  compute: {cost_a['compute']:.1f} -> {cost_b['compute']:.1f} "
            f"({cost_b['compute'] - cost_a['compute']:+.1f})"
        )
        lines.append(
            f"  comm:    {cost_a['comm']:.1f} -> {cost_b['comm']:.1f} "
            f"({cost_b['comm'] - cost_a['comm']:+.1f})"
        )
        lines.append(
            f"  trans:   {cost_a['transition']:.1f} -> {cost_b['transition']:.1f} "
            f"({cost_b['transition'] - cost_a['transition']:+.1f})"
        )
        lines.append("")
        lines.append(
            f"Placement changes ({sum(len(v) for v in changes.values())} nodes "
            f"changed, {unchanged} unchanged):"
        )
        for (old, new), nodes in sorted(changes.items(), key=lambda x: -len(x[1])):
            lines.append(f"  {old} -> {new}: {len(nodes)} nodes")

        result = "\n".join(lines)
        print(result)
        return result

    def _compute_solution_cost(self, solution):
        """Compute the total cost breakdown for a given solution."""
        total_compute = 0.0
        total_comm = 0.0
        total_transition = 0.0

        for node, strategy in solution.items():
            out_idx = None
            for i, s in enumerate(self.strats[node].strategies):
                if s is strategy:
                    out_idx = i
                    break
            if out_idx is None:
                continue

            node_idx = self.node_map[node]

            # Use pre-computed costs from decision vars instead of
            # estimate_strategy_runtime_cost, which needs node.meta["val"]
            # (absent on loaded optimizers). The (.,0,out_idx,0) edge may be
            # pruned, so find any surviving inp_idx for arg 0 (compute_cost is
            # identical across inp_idx for a given out_idx).
            dv = self._find_decision_var(node_idx, 0, out_idx)
            if dv is None:
                continue
            num_args = max(len(strategy.input_specs), 1)
            total_compute += dv.compute_cost * num_args

            input_nodes = self._all_input_nodes(node)
            for argi, pred in enumerate(input_nodes):
                pred_strategy = solution.get(pred)
                if pred_strategy is None:
                    continue
                pred_out_idx = None
                for i, s in enumerate(self.strats[pred].strategies):
                    if s is pred_strategy:
                        pred_out_idx = i
                        break
                if pred_out_idx is None:
                    continue
                comm = strategy.redistribute_cost[argi][pred_out_idx]
                if math.isfinite(comm):
                    total_comm += comm

                # Transition cost
                pred_spec = pred_strategy.output_specs
                input_spec = strategy.input_specs[argi]
                if (
                    isinstance(pred_spec, DTensorSpec)
                    and isinstance(input_spec, DTensorSpec)
                    and pred_spec.placements != input_spec.placements
                ):
                    total_transition += 1

        return {
            "total": total_compute + total_comm + total_transition,
            "compute": total_compute,
            "comm": total_comm,
            "transition": total_transition,
        }

    # ---- Logging ----

    def get_violated_constraints_log(self):
        if self.prob is None:
            return "Violated constraints: [] (no PuLP problem; lite build)"
        violated_constraints = [
            (k, c) for k, c in self.prob.constraints.items() if not c.valid()
        ]
        log_str = f"Violated constraints: {[x[0] for x in violated_constraints]}"
        for cname, c in violated_constraints:
            log_str += f"\n========= {cname} ============="
            for cc, v in c.items():
                log_str += f"\n{cc}, coeff={v}, value={cc.value()}"
        return log_str

    def get_log(self, colored=False, verbose=False):
        from autoparallel.log_formatting import format_sharding_log

        selected_by_node = {}
        for key in self.selected_keys:
            node = self.nodes[key[0]]
            selected_by_node.setdefault(node, []).append(
                self._resolve_decision_var(key)
            )

        return format_sharding_log(
            graph=self.graph,
            opt=selected_by_node,
            colored=colored,
            verbose=verbose,
            violated_constraints_log=self.get_violated_constraints_log(),
        )

    def get_json(self):
        from autoparallel.export_json import (
            _normalize_cluster_layer,
            export_sharding_json,
        )

        # Build selected DVs keyed by (node, argi) for explicit ordering.
        selected_by_node: dict[torch.fx.Node, dict[int, DecisionVar]] = {}
        for key in self.selected_keys:
            node_idx, argi, out_idx, inp_idx = key
            node = self.nodes[node_idx]
            selected_by_node.setdefault(node, {})[argi] = self._resolve_decision_var(
                key
            )

        # Build node-level cluster mapping: linked_node -> root_node
        cluster_roots: dict[torch.fx.Node, torch.fx.Node] = {}
        for copy_idx, root_idx in self.cluster_links.items():
            cluster_roots[self.nodes[copy_idx]] = self.nodes[root_idx]

        _normalize_cluster_layer(cluster_roots)

        return export_sharding_json(
            graph=self.graph,
            mesh=self.mesh,
            solution={
                node: next(iter(dvs_by_argi.values())).strategy
                for node, dvs_by_argi in selected_by_node.items()
            },
            selected_dvs=selected_by_node,
            cluster_roots=cluster_roots,
        )

    def get_strategy(self, node):
        """Look up the OpStrategy for a node.

        Accepts nodes from either the original or concrete graph.
        """
        node = self._normalize_node(node)
        return self.strats[node]

    # ---- Serialization ----

    def save(self, path):
        """Save the full ILP/LP optimizer state for interactive exploration."""
        from autoparallel.serialization import save_optimizer

        save_optimizer(self, path)

    @classmethod
    def load(cls, path):
        """Load optimizer state saved with save()."""
        from autoparallel.serialization import load_optimizer

        return load_optimizer(cls, path)

    def save_placements(self, path):
        """Save the current placement choices as a lightweight JSON file."""
        from autoparallel.serialization import save_placements

        save_placements(self, path)

    def load_placements(self, path):
        """Load placements from a JSON file and return a dict[Node, OpSpec]."""
        from autoparallel.serialization import load_placements

        return load_placements(self, path)

    def print_costs_for_node(self, node, arg=0, **kwargs):
        from tabulate import tabulate  # type: ignore
        from torch.distributed.tensor._op_schema import _pretty_print_spec

        node = self._normalize_node(node)
        tgt_strat = self.strats[node]
        src_strat = self.strats[self._all_input_nodes(node)[arg]]
        src_placements = [""] + [
            _pretty_print_spec(x.output_specs) for x in src_strat.strategies
        ]
        costs = [[str(x)] + x.redistribute_cost[arg] for x in tgt_strat.strategies]
        print(tabulate(costs, headers=src_placements, **kwargs))

    def explain_placement(self, target_placement, start_node=None, max_nodes=None):
        """Diagnose why the optimizer didn't choose a specific placement.

        Walks the graph in topological order and, for each node, reports whether
        target_placement is available, what its compute cost would be compared
        to the chosen placement, and flags the first node where the target
        becomes unavailable.

        Shows both compute and communication costs. Communication cost for the
        target assumes all predecessors also use target_placement (i.e., "what
        if the entire graph used this placement?").

        Must be called after get_solution().

        Args:
            target_placement: tuple of Placement objects, e.g. (Shard(0), Replicate())
            start_node: optional node to begin from (default: first graph node)
            max_nodes: optional limit on number of nodes to report
        """
        if start_node is not None:
            start_node = self._normalize_node(start_node)
        from tabulate import tabulate  # type: ignore
        from torch.distributed.tensor._op_schema import _pretty_print_spec

        # Build chosen out_idx per node from the solution
        chosen_out_idx = {}
        for key in self.selected_keys:
            node_idx, argi, out_idx, inp_idx = key
            if argi == 0:
                chosen_out_idx[node_idx] = out_idx

        target_placement = tuple(target_placement)
        target_str = _pretty_print_spec(DTensorSpec(self.mesh, target_placement))

        # Build target_out_idx per node: which strategy index produces
        # target_placement, if any.
        target_out_idx_map = {}
        for ni, node in enumerate(self.nodes):
            if node.op == "output":
                continue
            op_strategy = self.strats.get(node)
            if op_strategy is None:
                continue
            for i, strategy in enumerate(op_strategy.strategies):
                spec = strategy.output_specs
                if (
                    isinstance(spec, DTensorSpec)
                    and spec.placements == target_placement
                ):
                    target_out_idx_map[ni] = i
                    break

        started = start_node is None
        rows = []
        count = 0

        for node_idx, node in enumerate(self.nodes):
            if node.op == "output":
                continue

            if not started:
                if node is start_node:
                    started = True
                else:
                    continue

            if max_nodes is not None and count >= max_nodes:
                break

            op_strategy = self.strats[node]
            out_idx = chosen_out_idx.get(node_idx)
            if out_idx is None:
                continue

            chosen_strategy = op_strategy.strategies[out_idx]
            chosen_spec = chosen_strategy.output_specs

            # Skip multi-output nodes (tuples)
            if not isinstance(chosen_spec, DTensorSpec):
                continue

            chosen_placement_str = _pretty_print_spec(chosen_spec)
            chosen_compute = estimate_strategy_runtime_cost(node, chosen_strategy)

            # Compute communication cost for the chosen strategy: sum
            # redistribute_cost across all args using predecessors' chosen
            # out_idx.
            input_nodes = self._all_input_nodes(node)
            chosen_comm = 0.0
            for argi, pred in enumerate(input_nodes):
                pred_idx = self.node_map[pred]
                pred_out = chosen_out_idx.get(pred_idx, 0)
                chosen_comm += chosen_strategy.redistribute_cost[argi][pred_out]

            # Get output shape for display
            shape = chosen_spec.tensor_meta.shape if chosen_spec.tensor_meta else "?"
            shape_str = str(list(shape)) if shape != "?" else "?"

            # Search for target placement among available strategies
            target_out_idx = target_out_idx_map.get(node_idx)

            if target_out_idx is not None:
                target_strategy = op_strategy.strategies[target_out_idx]
                target_compute = estimate_strategy_runtime_cost(node, target_strategy)

                # Compute communication cost for the target strategy:
                # assume each predecessor also uses target_placement. If a
                # predecessor doesn't have it, fall back to its chosen placement.
                target_comm = 0.0
                for argi, pred in enumerate(input_nodes):
                    pred_idx = self.node_map[pred]
                    pred_tgt_out = target_out_idx_map.get(
                        pred_idx, chosen_out_idx.get(pred_idx, 0)
                    )
                    target_comm += target_strategy.redistribute_cost[argi][pred_tgt_out]

                chosen_total = chosen_compute + chosen_comm
                target_total = target_compute + target_comm

                # Determine status based on total cost
                if chosen_spec.placements == target_placement:
                    status = "CHOSEN"
                elif abs(chosen_total - target_total) < 0.01:
                    status = "TIE"
                elif target_total < chosen_total:
                    status = "TARGET CHEAPER"
                else:
                    status = "CHOSEN CHEAPER"

                # Flag kernel launch floor on compute
                if (
                    abs(target_compute - 7.0) < 0.01
                    and abs(chosen_compute - 7.0) < 0.01
                ):
                    status += " [floor]"

                target_compute_str = f"{target_compute:.1f}"
                target_comm_str = f"{target_comm:.1f}"
            else:
                target_compute_str = "N/A"
                target_comm_str = "N/A"
                status = "UNAVAILABLE"

            rows.append(
                [
                    str(node),
                    shape_str,
                    chosen_placement_str,
                    f"{chosen_compute:.1f}",
                    f"{chosen_comm:.1f}",
                    target_compute_str,
                    target_comm_str,
                    status,
                ]
            )
            count += 1

        headers = [
            "Node",
            "Shape",
            "Chosen",
            "Ch.Comp",
            "Ch.Comm",
            "Tgt.Comp",
            "Tgt.Comm",
            "Status",
        ]
        result = f"explain_placement: target={target_str}\n\n"
        result += tabulate(rows, headers=headers)
        print(result)
        return result

    # ---- User constraints ----

    def _add_node_constraint(
        self, node, output_constraint_indices, constraint_name=None
    ):
        if constraint_name is None:
            constraint_name = "user_constraint"
        node_idx = self.node_map[node]
        vars_per_arg = {}
        for argi, out_idx, inp_idx in self.walk_over_options(node):
            if out_idx in output_constraint_indices:
                var = self._get_pulp_variable((node_idx, argi, out_idx, inp_idx))
                if var is None:  # pruned (invalid) strategy edge
                    continue
                vars_per_arg.setdefault(argi, []).append(var)
        names = []
        for eqs in vars_per_arg.values():
            name = self._get_next_name(constraint_name)
            self.prob += (pulp.lpSum(eqs) == 1, name)
            names.append(name)
        return names

    def _add_paired_output_constraint(self, node_a, node_b, constraint_name):
        """Constrains two nodes to have matching output placements.

        For each output strategy of node_a that also exists in node_b, adds:
            Σ_j x_{a, 0, oi_a, j} == Σ_j x_{b, 0, oi_b, j}

        where oi_a and oi_b are output strategy indices with the same placement.
        """
        idx_a = self.node_map[node_a]
        idx_b = self.node_map[node_b]
        strat_a = [str(s.output_specs) for s in self.strats[node_a].strategies]
        strat_b = [str(s.output_specs) for s in self.strats[node_b].strategies]
        num_inp_a = len(self.strats[node_a].strategies[0].redistribute_cost[0])
        num_inp_b = len(self.strats[node_b].strategies[0].redistribute_cost[0])
        for out_idx, sp in enumerate(strat_a):
            if sp not in strat_b:
                # This placement exists in node_a but not in node_b.
                # Disable it: force sum of its decision variables to 0.
                v_a = [
                    v
                    for inp_idx in range(num_inp_a)
                    if (v := self._get_pulp_variable((idx_a, 0, out_idx, inp_idx)))
                    is not None
                ]
                self.prob += (
                    pulp.lpSum(v_a) == 0,
                    self._get_next_name(constraint_name + "_disable"),
                )
                continue
            out_idx_b = strat_b.index(sp)
            v_a = [
                v
                for inp_idx in range(num_inp_a)
                if (v := self._get_pulp_variable((idx_a, 0, out_idx, inp_idx)))
                is not None
            ]
            v_b = [
                v
                for inp_idx in range(num_inp_b)
                if (v := self._get_pulp_variable((idx_b, 0, out_idx_b, inp_idx)))
                is not None
            ]
            self.prob += (
                pulp.lpSum(v_b) == pulp.lpSum(v_a),
                self._get_next_name(constraint_name),
            )

    def add_forward_backward_consistency_constraints(self):
        """USER (Category 5c): Forward-backward consistency constraints.
        Ensures that paired forward/backward nodes have matching output placements
        for parameters, plain inputs, and plain outputs.

        Σ_j x_{fwd, 0, oi, j} = Σ_j x_{bwd, 0, oi', j}
        where oi and oi' have matching placements.
        """
        for param, grad in get_param_and_grad_nodes(self.graph).values():
            if grad is None:
                continue
            self._add_paired_output_constraint(param, grad, "grad_param_constraint")

        for node, grad_node in get_plain_input_and_grad_nodes(self.graph).values():
            if grad_node is None:
                continue
            self._add_paired_output_constraint(node, grad_node, "grad_input_constraint")

        for node, tangent_node in get_plain_output_and_tangent_nodes(
            self.graph
        ).values():
            if tangent_node is None:
                continue
            self._add_paired_output_constraint(
                node, tangent_node, "grad_output_constraint"
            )

    def add_grad_reduce_dtype_constraints(self):
        """Forbid redistribution on the wrong side of dtype_cast nodes.

        When MixedPrecisionPolicy inserts dtype_cast nodes to convert between
        storage dtype and compute dtype, we want collectives to run in the
        smaller dtype. This method enforces two independent constraints:

        **Forward** (unconditional when dtype_cast nodes exist):
        Forces allgather to happen after the cast (in param_dtype, e.g. bf16)
        rather than before it (in storage dtype, e.g. f32). Applies whenever
        the model stores parameters in a higher-precision dtype than
        param_dtype, regardless of reduce_dtype.

            primals (storage_dtype) -> [unary ops] -> dtype_cast (param_dtype)

        **Backward** (when reduce_dtype > param_dtype):
        Forces reduce-scatter to happen after the cast (in reduce_dtype, e.g.
        f32) rather than before it (in param_dtype, e.g. bf16).

            einsum (param_dtype) -> [unary ops] -> dtype_cast -> alias (grad_param)

        Both are enforced by disabling decision variables on pre-cast edges
        that would imply a placement change.
        """
        # --- Forward: forbid redistribution before param dtype_cast ---
        # Only when the cast reduces precision (storage_dtype > param_dtype),
        # so that allgather runs in the smaller param_dtype. When the cast
        # increases precision (e.g. bf16 storage → f32 compute), we want the
        # allgather in the smaller storage dtype, which is already the default.
        fwd_pre_cast_node_idxs: set[int] = set()
        for param, _grad in get_param_and_grad_nodes(self.graph).values():
            # Walk forward from param through single-user unary nodes
            n = param
            while True:
                if n.target == torch.ops.autoparallel.dtype_cast.default:
                    break
                users = list(n.users.keys())
                if len(users) != 1:
                    break
                child = users[0]
                if len(child.all_input_nodes) != 1:
                    break
                n = child

            if n.target != torch.ops.autoparallel.dtype_cast.default:
                continue

            # Only constrain if it downcasts
            storage_dtype = param.meta["val"].dtype
            cast_dtype = n.meta["val"].dtype
            if cast_dtype.itemsize >= storage_dtype.itemsize:
                continue

            # Mark dtype_cast and all nodes between param and dtype_cast
            # (exclusive of param itself since placeholders don't have
            # input edges)
            node = n
            while node != param:
                if node in self.node_map:
                    fwd_pre_cast_node_idxs.add(self.node_map[node])
                parent = node.all_input_nodes[0]
                node = parent

        for key, dv in self.decision_vars.items():
            node_idx, argi, out_idx, inp_idx = key
            if node_idx not in fwd_pre_cast_node_idxs:
                continue
            if dv.comm_cost > 0:
                self.prob += (
                    dv.var == 0,
                    self._get_next_name("fwd_param_dtype"),
                )

        # --- Backward: forbid redistribution before grad dtype_cast ---
        if not self.force_grad_reduce_in_higher_precision:
            return

        pre_cast_node_idxs: set[int] = set()
        for param, grad in get_param_and_grad_nodes(self.graph).values():
            if grad is None:
                continue

            # Walk backward from grad through the unary chain, excluding
            # the first non-unary producer.
            chain = [grad]
            n = grad
            while len(n.all_input_nodes) == 1:
                parent = n.all_input_nodes[0]
                if len(parent.all_input_nodes) != 1:
                    break
                chain.append(parent)
                n = parent

            # Find the dtype_cast node in the chain
            cast_idx = None
            for i, node in enumerate(chain):
                if node.target == torch.ops.autoparallel.dtype_cast.default:
                    cast_idx = i
                    break

            if cast_idx is None:
                continue

            # Mark dtype_cast and all pre-cast unary nodes
            for node in chain[cast_idx:]:
                if node in self.node_map:
                    pre_cast_node_idxs.add(self.node_map[node])

        # Disable decision vars on pre-cast edges that imply redistribution
        for key, dv in self.decision_vars.items():
            node_idx, argi, out_idx, inp_idx = key
            if node_idx not in pre_cast_node_idxs:
                continue
            if dv.comm_cost > 0:
                self.prob += (
                    dv.var == 0,
                    self._get_next_name("grad_reduce_dtype"),
                )

    def add_parameter_memory_constraint(
        self, memory_factor_low: float, memory_factor_high: float
    ):
        """USER (Category 5b): Constrain total parameter memory usage.

        Σ_{params} (size_ratio * x_{param}) ≤ memory_limit

        The actual ILP constraints are added lazily at solve time so that
        node constraints registered after this call are still respected.
        Parameters with user-defined node constraints are excluded to avoid
        infeasible problems.
        """
        self._constraint_log.append(
            (
                "add_parameter_memory_constraint",
                {
                    "memory_factor_low": memory_factor_low,
                    "memory_factor_high": memory_factor_high,
                },
            )
        )
        self._memory_constraint = (memory_factor_low, memory_factor_high)

    def _apply_memory_constraint(self):
        """Rebuild the parameter memory constraint in the ILP.

        Called on every solve so that node constraints added after
        add_parameter_memory_constraint() are always respected.
        """
        if self._memory_constraint is None:
            return
        if self.prob is None:
            return  # approx-only builds read the active memory state directly
        memory_factor_low, memory_factor_high = self._memory_constraint

        # Remove previous memory constraints before rebuilding
        for name in ("memory_constraint_high", "memory_constraint_low"):
            self.prob.constraints.pop(name, None)

        user_constrained_names = set(self._node_constraint_names.values())

        param_nodes: list[torch.fx.Node] = get_param_nodes(self.graph)
        elms: list[pulp.LpAffineExpression] = []
        budget_low: float = 0.0
        budget_high: float = 0.0
        for node in param_nodes:
            if node.name in user_constrained_names:
                continue
            node_idx = self.node_map[node]
            num_out_strat = len(self.strats[node].strategies)
            ratios: list[float] = []
            for out_idx in range(num_out_strat):
                dv = self._find_decision_var(node_idx, 0, out_idx)
                if dv is None:  # every edge for this strategy was pruned
                    continue
                spec: DTensorSpec = dv.input_spec
                assert spec.tensor_meta is not None
                tensor_shape: torch.Size = spec.tensor_meta.shape
                new_tensor_shape, _ = _get_sharded_shape_stride(spec)
                new_size: int = math.prod(new_tensor_shape)
                old_size: int = math.prod(tensor_shape)
                ratio = new_size / old_size
                ratios.append(ratio)
                elms.append(dv.var * ratio)
            best_ratio: float = min(ratios)
            budget_low += max(best_ratio, memory_factor_low)
            budget_high += max(best_ratio, memory_factor_high)

        self.prob += (pulp.lpSum(elms) <= budget_high, "memory_constraint_high")
        self.prob += (pulp.lpSum(elms) >= budget_low, "memory_constraint_low")

    def add_node_constraint(self, node, placement=None, constraint_name=None):
        """USER (Category 5d): Force a specific placement for a node.

        For nodes with tuple output_specs (e.g. SDPA), the placement is matched
        against the first DTensorSpec element in the tuple.
        """
        node = self._normalize_node(node)
        self._constraint_log.append(
            (
                "add_node_constraint",
                {
                    "node_name": node.name,
                    "placement": placement,
                    "constraint_name": constraint_name,
                },
            )
        )
        assert node in self.strats, (node, self.strats.keys())
        strat = self.strats[node]
        if placement is None:
            # default is Shard(0) to parallelize on the batch
            placement = (Shard(0),) + (Replicate(),) * (self.mesh.ndim - 1)
        output_constraint_indices = []
        for i, s in enumerate(strat.strategies):
            specs = s.output_specs
            if isinstance(specs, DTensorSpec):
                if specs.placements == placement:
                    output_constraint_indices.append(i)
            elif isinstance(specs, (list, tuple)):
                for spec in specs:
                    if isinstance(spec, DTensorSpec):
                        if spec.placements == placement:
                            output_constraint_indices.append(i)
                        break
        if len(output_constraint_indices) == 0:
            raise RuntimeError(
                f"Couldn't find appropriate constraint {node} {constraint_name} {placement}"
            )
        if self.prob is None:
            return []  # approx (lite) build replays this from _constraint_log
        names = self._add_node_constraint(
            node,
            output_constraint_indices=output_constraint_indices,
            constraint_name=constraint_name,
        )
        for name in names:
            self._node_constraint_names[name] = node.name
        return names

    def _add_io_placement_constraints(
        self,
        nodes_dict,
        placements,
        desc_type,
        constraint_name,
        error_message,
    ):
        """Shared implementation for add_sharded_input_constraint and
        add_sharded_output_constraint. Only constrains the forward-side node;
        the backward side is handled by add_forward_backward_consistency_constraints."""
        remaining = None
        if placements is not None:
            remaining = {i: p for i, p in enumerate(placements)}

        for desc, (node, _companion_node) in nodes_dict.items():
            if placements is None:
                placement = None
            else:
                assert isinstance(desc, desc_type)
                assert remaining is not None
                placement = remaining.pop(desc.idx)

            self.add_node_constraint(node, placement, constraint_name=constraint_name)

        ignored = []
        if remaining is not None:
            for i, p in remaining.items():
                if p is not None:
                    ignored.append(i)

        if ignored:
            raise RuntimeError(error_message.format(indices=ignored))

    def add_sharded_input_constraint(
        self, input_placements: Optional[list[Optional[tuple[Placement, ...]]]] = None
    ):
        """USER (Category 5a): Force specific placements for input nodes.
        The corresponding gradient inputs are automatically constrained via
        add_forward_backward_consistency_constraints()."""
        self._constraint_log.append(
            ("add_sharded_input_constraint", {"input_placements": input_placements})
        )
        self._add_io_placement_constraints(
            nodes_dict=get_plain_input_and_grad_nodes(self.graph),
            placements=input_placements,
            desc_type=PlainAOTInput,
            constraint_name="input_constraint",
            error_message=(
                "We were unable to respect placements for inputs at indices {indices}.  "
                "This is because the traced joint graph did not actually have a dedicated "
                "placeholder node for these inputs.  "
                "This typically occurs because some inputs aliased each other; inspect the "
                "joint graph from tlparse for more details.  "
                "You can either remove an explicit placement for this input (replace it with "
                "None) or clone the inputs before tracing to remove aliasing."
            ),
        )

    def add_sharded_output_constraint(self, output_placements=None):
        """USER (Category 5a): Force specific placements for output nodes.
        The corresponding tangent/gradient nodes are automatically constrained via
        add_forward_backward_consistency_constraints()."""
        self._constraint_log.append(
            ("add_sharded_output_constraint", {"output_placements": output_placements})
        )
        self._add_io_placement_constraints(
            nodes_dict=get_plain_output_and_tangent_nodes(self.graph),
            placements=output_placements,
            desc_type=PlainAOTOutput,
            constraint_name="output_constraint",
            error_message=(
                "We were unable to respect placements for outputs at indices {indices}.  "
                "This is because the traced joint graph did not actually have a dedicated "
                "output node for these inputs.  "
                "This typically occurs because some outputs aliased each other; inspect the "
                "joint graph from tlparse for more details.  "
                "You can either remove an explicit placement for this output (replace it with "
                "None), stop the model from returning aliases of the tensor or clone the "
                "outputs before returning them from the graph to avoid aliasing."
            ),
        )
