# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Save/load for ShardingOptimizer state.

Handles full optimizer serialization (save/load) and lightweight
solution-only serialization (save_solution/load_solution).
"""

import json
import logging
import operator
import time
from collections import defaultdict

import pulp
import torch

logger: logging.Logger = logging.getLogger(__name__)


class _MeshPlaceholder:
    """Lightweight stand-in for DeviceMesh when loading without a process group.

    Provides the attributes used by get_json() and add_node_constraint()
    without requiring distributed initialization.
    """

    def __init__(self, shape, dim_names):
        self.shape = tuple(shape)
        self.mesh_dim_names = tuple(dim_names) if dim_names else None
        self.ndim = len(shape)


def _resolve_target(target_str):
    """Resolve a serialized target string back to the callable."""
    if target_str == "<built-in function getitem>":
        return operator.getitem
    parts = target_str.split(".")
    if len(parts) >= 2:
        try:
            obj = getattr(torch.ops, parts[0])
            for p in parts[1:]:
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            pass
    return torch.ops.aten.alias.default


def _patch_op_overload_pickle():
    """Temporarily add pickle support to OpOverload so FX graphs with custom
    ops can be serialized. Returns a context manager that removes the patch
    on exit."""
    import contextlib

    @contextlib.contextmanager
    def _patch():
        had_reduce = hasattr(torch._ops.OpOverload, "__reduce__")
        old_reduce = getattr(torch._ops.OpOverload, "__reduce__", None)

        def _reduce(self):
            return (_resolve_target, (str(self),))

        torch._ops.OpOverload.__reduce__ = _reduce
        try:
            yield
        finally:
            if had_reduce:
                torch._ops.OpOverload.__reduce__ = old_reduce
            else:
                del torch._ops.OpOverload.__reduce__

    return _patch()


_SAVE_META_KEYS = {
    "desc",
    "partitioner_tag",
    "stack_trace",
    "tensor_meta",
    "module_path",
    "phase",
    "is_gradient_acc",
    "seq_nr",
}


def save_optimizer(opt, path):
    """Save the full optimizer state for later interactive exploration.

    The saved file contains everything needed to rebuild the ILP and
    re-solve without the original model code, DeviceMesh, or process group.
    Can be called before or after solving — if unsolved, the loaded
    optimizer can be solved in a notebook via get_solution() or resolve().
    """
    import copy

    import numpy as np

    from autoparallel.export_json import _extract_module_path, _get_phase

    t0 = time.perf_counter()

    # Convert selected_keys to node-name-based representation (if solved)
    selected_keys_by_name = None
    if hasattr(opt, "selected_keys"):
        selected_keys_by_name = {}
        for key in opt.selected_keys:
            node_idx, argi, out_idx, inp_idx = key
            node_name = opt.nodes[node_idx].name
            selected_keys_by_name.setdefault(node_name, []).append(
                (argi, out_idx, inp_idx)
            )

    # Re-key strats by node name, saving only root nodes (non-linked).
    # Linked nodes share identical strats with their root and are
    # reconstructed on load from cluster_links.
    linked_node_names = {opt.nodes[lk[0]].name for lk in opt.cluster_links}
    strats_by_name = {
        node.name: strat
        for node, strat in opt.strats.items()
        if node.name not in linked_node_names
    }

    # Save decision var costs as compact numpy arrays. Indices use int32,
    # costs use float64 to preserve full precision (matching the in-memory
    # representation used by the ILP solver).
    save_node_names = [n.name for n in opt.nodes]
    save_node_to_idx = {name: i for i, name in enumerate(save_node_names)}
    n_dvs = len(opt.decision_vars)
    dv_costs_keys = np.empty((n_dvs, 4), dtype=np.int32)
    dv_costs_vals = np.empty((n_dvs, 3), dtype=np.float64)
    for row_idx, (key, dv) in enumerate(opt.decision_vars.items()):
        node_idx, argi, out_idx, inp_idx = key
        save_node_idx = save_node_to_idx[opt.nodes[node_idx].name]
        dv_costs_keys[row_idx] = (save_node_idx, argi, out_idx, inp_idx)
        dv_costs_vals[row_idx] = (
            dv.compute_cost,
            dv.comm_cost,
            dv.sharding_transition_cost,
        )

    # Extract module paths from nn_module_stack before deepcopy strips it
    module_paths = {}
    phases = {}
    for node in opt.graph.nodes:
        mp = _extract_module_path(node)
        if mp is not None:
            module_paths[node.name] = mp
        if node.op not in ("placeholder", "output"):
            phases[node.name] = _get_phase(node)

    # Deepcopy the graph and keep only picklable metadata
    graph_copy = copy.deepcopy(opt.graph)
    for node in graph_copy.nodes:
        meta = {k: v for k, v in node.meta.items() if k in _SAVE_META_KEYS}
        if node.name in module_paths:
            meta["module_path"] = module_paths[node.name]
        if node.name in phases:
            meta["phase"] = phases[node.name]
        node.meta = meta

    save_dict = {
        "version": 1,
        "graph": graph_copy,
        "mesh_shape": list(opt.mesh.shape),
        "mesh_dim_names": (
            list(opt.mesh.mesh_dim_names) if opt.mesh.mesh_dim_names else None
        ),
        "force_grad_reduce_in_higher_precision": opt.force_grad_reduce_in_higher_precision,
        "strats_by_name": strats_by_name,
        "dv_costs_node_names": save_node_names,
        "dv_costs_keys": dv_costs_keys,
        "dv_costs_vals": dv_costs_vals,
        "cluster_links_node_by_name": {
            opt.nodes[lk[0]].name: opt.nodes[rk[0]].name
            for lk, rk in opt.cluster_links.items()
        },
        "constraint_log": opt._constraint_log,
        "selected_keys_by_name": selected_keys_by_name,
    }
    t1 = time.perf_counter()
    logger.debug("save: prepared save_dict in %.3fs", t1 - t0)
    with _patch_op_overload_pickle():
        torch.save(save_dict, path)
    logger.debug(
        "save: wrote %s in %.3fs (total %.3fs)",
        path,
        time.perf_counter() - t1,
        time.perf_counter() - t0,
    )


def load_optimizer(cls, path):
    """Load optimizer state saved with save().

    Returns a fully functional ShardingOptimizer that supports get_json(),
    get_log(), add_node_constraint(), resolve(), diff_solutions(), etc.
    No live DeviceMesh or process group is needed.
    """
    # Ensure custom ops are registered before loading
    import autoparallel.cast_parametrization  # noqa: F401
    from autoparallel.optimize_sharding import DecisionVar

    t0 = time.perf_counter()
    with _patch_op_overload_pickle():
        save_dict = torch.load(path, weights_only=False)
    t1 = time.perf_counter()
    logger.debug("load: torch.load took %.3fs", t1 - t0)
    assert save_dict["version"] == 1, f"Unsupported version: {save_dict['version']}"

    graph = save_dict["graph"]
    strats_by_name = save_dict["strats_by_name"]
    cluster_links_node_by_name = save_dict["cluster_links_node_by_name"]

    # Build node-name lookup from the graph
    nodes_by_name = {node.name: node for node in graph.nodes}

    # Build linked_node_name -> root_node_name mapping (already node-level)
    linked_to_root_name = dict(cluster_links_node_by_name)

    # Reconstruct strats: root strats are saved, linked strats copied
    strats = {}
    for node in graph.nodes:
        if node.name in strats_by_name:
            strats[node] = strats_by_name[node.name]
        elif node.name in linked_to_root_name:
            strats[node] = strats_by_name[linked_to_root_name[node.name]]

    # Create optimizer without calling __init__
    opt = cls.__new__(cls)
    opt.gm = None
    opt.graph = graph
    opt._orig_to_concrete = {}
    opt._concrete_to_orig = {}
    opt.strats = strats
    opt.nodes = list(strats.keys())
    opt.node_map = {node: i for i, node in enumerate(opt.nodes)}
    opt.force_grad_reduce_in_higher_precision = save_dict[
        "force_grad_reduce_in_higher_precision"
    ]
    opt._constraint_log = []
    opt._name_counters = {}

    # Reconstruct cluster_links by expanding the node-level mapping over
    # all (argi, out_idx, inp_idx) combinations.
    opt.cluster_links = {}
    for linked_name, root_name in cluster_links_node_by_name.items():
        linked_node = nodes_by_name[linked_name]
        root_node = nodes_by_name[root_name]
        linked_idx = opt.node_map[linked_node]
        root_idx = opt.node_map[root_node]
        for argi, out_idx, inp_idx in opt.walk_over_options(linked_node):
            opt.cluster_links[(linked_idx, argi, out_idx, inp_idx)] = (
                root_idx,
                argi,
                out_idx,
                inp_idx,
            )
    opt._cluster_linked_node_idxs = {key[0] for key in opt.cluster_links}

    # Mesh placeholder — provides shape/dim_names for get_json() and ndim
    # for add_node_constraint() default placement, without needing a PG
    opt.mesh = _MeshPlaceholder(save_dict["mesh_shape"], save_dict["mesh_dim_names"])

    # Rebuild PuLP variables and decision vars from saved costs.
    t2 = time.perf_counter()
    opt.pulp_variables = opt._create_pulp_variables()
    t3 = time.perf_counter()
    logger.debug(
        "load: _create_pulp_variables took %.3fs (%d vars)",
        t3 - t2,
        len(opt.pulp_variables),
    )
    # Reconstruct decision_vars from compact tensors.
    save_node_names = save_dict["dv_costs_node_names"]
    keys_t = save_dict["dv_costs_keys"].tolist()
    vals_t = save_dict["dv_costs_vals"].tolist()
    opt.decision_vars = {}
    for (save_node_idx, argi, out_idx, inp_idx), (
        compute_cost,
        comm_cost,
        transition_cost,
    ) in zip(keys_t, vals_t):
        node_name = save_node_names[save_node_idx]
        node = nodes_by_name[node_name]
        node_idx = opt.node_map[node]
        key = (node_idx, argi, out_idx, inp_idx)
        strategy = opt.strats[node].strategies[out_idx]
        opt.decision_vars[key] = DecisionVar(
            var=opt.pulp_variables[key],
            cost=compute_cost + comm_cost + transition_cost,
            compute_cost=compute_cost,
            comm_cost=comm_cost,
            sharding_transition_cost=transition_cost,
            strategy=strategy,
            output_spec=strategy.output_specs,
            input_spec=(
                strategy.input_specs[argi] if argi < len(strategy.input_specs) else None
            ),
        )
    t4 = time.perf_counter()
    logger.debug(
        "load: decision_vars rebuild took %.3fs (%d entries)",
        t4 - t3,
        len(opt.decision_vars),
    )

    opt._root_to_linked = defaultdict(list)
    for linked_key, root_key in opt.cluster_links.items():
        opt._root_to_linked[root_key].append(linked_key)

    opt.prob = pulp.LpProblem("AutoParallel", pulp.LpMinimize)
    opt.add_default_constraints()
    t5 = time.perf_counter()
    logger.debug("load: add_default_constraints took %.3fs", t5 - t4)

    # Replay user constraints
    for method_name, kwargs in save_dict["constraint_log"]:
        method = getattr(opt, method_name)
        # Resolve node names back to node objects
        if "node_name" in kwargs:
            kwargs = dict(kwargs)
            node_name = kwargs.pop("node_name")
            kwargs["node"] = nodes_by_name[node_name]
        method(**kwargs)
    t6 = time.perf_counter()
    logger.debug("load: constraint replay took %.3fs", t6 - t5)

    # Set objective and restore solution if one was saved
    opt._set_objective()
    t7 = time.perf_counter()
    logger.debug("load: _set_objective took %.3fs", t7 - t6)
    if save_dict["selected_keys_by_name"] is not None:
        _restore_solution(opt, save_dict["selected_keys_by_name"], nodes_by_name)
    logger.debug("load: total %.3fs", time.perf_counter() - t0)

    return opt


def _restore_solution(opt, selected_keys_by_name, nodes_by_name):
    """Restore selected_keys and PuLP variable values from a saved solution."""
    opt.selected_keys = []
    for node_name, key_parts in selected_keys_by_name.items():
        node = nodes_by_name[node_name]
        node_idx = opt.node_map[node]
        for argi, out_idx, inp_idx in key_parts:
            opt.selected_keys.append((node_idx, argi, out_idx, inp_idx))

    # Set PuLP variable values: selected = 1.0, all others default to 0.0
    selected_set = set(opt.selected_keys)
    for key in selected_set:
        dv = opt.decision_vars.get(key)
        if dv is not None:
            dv.var.varValue = 1.0

    # Expand cluster links
    for root_key in list(opt.selected_keys):
        opt.selected_keys.extend(opt._root_to_linked.get(root_key, []))


def save_solution(opt, path):
    """Save the current solution as a lightweight JSON file."""
    from torch.distributed.tensor._op_schema import _pretty_print_spec

    placements = {}
    solution = opt._extract_and_validate_solution()
    for node, strategy in solution.items():
        placements[node.name] = _pretty_print_spec(strategy.output_specs)

    save_dict = {
        "version": 1,
        "mesh_shape": list(opt.mesh.shape),
        "mesh_dim_names": (
            list(opt.mesh.mesh_dim_names) if opt.mesh.mesh_dim_names else None
        ),
        "placements": placements,
    }
    with open(path, "w") as f:
        json.dump(save_dict, f, indent=2)


def load_solution(opt, path):
    """Load a solution from a JSON file and return a dict[Node, OpSpec].

    Matches saved placement strings against the strategies in opt.strats.
    The returned dict can be passed directly to apply_placement().
    """
    from torch.distributed.tensor._op_schema import _pretty_print_spec

    with open(path) as f:
        save_dict = json.load(f)

    assert save_dict["version"] == 1

    nodes_by_name = {node.name: node for node in opt.nodes}
    solution = {}
    for node_name, placement_str in save_dict["placements"].items():
        if node_name not in nodes_by_name:
            raise RuntimeError(
                f"Node '{node_name}' from saved solution not found in current graph. "
                "The model may have changed since the solution was saved."
            )
        node = nodes_by_name[node_name]
        strat = opt.strats[node]
        matched = None
        for s in strat.strategies:
            if _pretty_print_spec(s.output_specs) == placement_str:
                matched = s
                break
        if matched is None:
            raise RuntimeError(
                f"Placement '{placement_str}' for node '{node_name}' not found "
                f"in available strategies. The model or mesh may have changed."
            )
        solution[node] = matched

    return solution
