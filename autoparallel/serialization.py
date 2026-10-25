# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Save/load for ShardingOptimizer state.

Handles full optimizer serialization (save/load), compact optimizer replay
(save_replay/load_replay), and lightweight placement serialization
(save_placements/load_placements).
"""

import gzip
import json
import logging
import math
import operator
import time
from collections import defaultdict
from dataclasses import dataclass

import pulp
import torch
from torch.utils._pytree import tree_map_only

logger: logging.Logger = logging.getLogger(__name__)


class _MeshPlaceholder:
    """Lightweight stand-in for DeviceMesh when loading without a process group.

    Provides the attributes used by get_json(), add_node_constraint(),
    and add_parameter_memory_constraint() without requiring distributed
    initialization.
    """

    def __init__(self, shape, dim_names):
        self.shape = tuple(shape)
        self.mesh_dim_names = tuple(dim_names) if dim_names else None
        self.ndim = len(shape)

    def size(self):
        result = 1
        for s in self.shape:
            result *= s
        return result


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
    raise RuntimeError(
        f"Cannot resolve serialized op target '{target_str}'. "
        "The model may require custom op registration before loading."
    )


def _resolve_higher_order_operator(name):
    try:
        return getattr(torch.ops.higher_order, name)
    except AttributeError as error:
        raise RuntimeError(
            f"Cannot resolve serialized higher-order operator {name!r}"
        ) from error


def _patch_op_overload_pickle():
    """Temporarily add pickle support to OpOverload so FX graphs with custom
    ops can be serialized. Returns a context manager that removes the patch
    on exit."""
    import contextlib

    @contextlib.contextmanager
    def _patch():
        had_reduce = hasattr(torch._ops.OpOverload, "__reduce__")
        old_reduce = getattr(torch._ops.OpOverload, "__reduce__", None)
        had_hop_reduce = hasattr(torch._ops.HigherOrderOperator, "__reduce__")
        old_hop_reduce = getattr(torch._ops.HigherOrderOperator, "__reduce__", None)

        def _reduce(self):
            return (_resolve_target, (str(self),))

        def _reduce_hop(self):
            return (_resolve_higher_order_operator, (self.name(),))

        torch._ops.OpOverload.__reduce__ = _reduce
        torch._ops.HigherOrderOperator.__reduce__ = _reduce_hop
        try:
            yield
        finally:
            if had_reduce:
                torch._ops.OpOverload.__reduce__ = old_reduce
            else:
                del torch._ops.OpOverload.__reduce__
            if had_hop_reduce:
                torch._ops.HigherOrderOperator.__reduce__ = old_hop_reduce
            else:
                del torch._ops.HigherOrderOperator.__reduce__

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
    "ac_graph_id",
    "recompute",
    "is_tensor_value",
}

_REPLAY_META_KEYS = _SAVE_META_KEYS | {"local_map_kwargs", "val"}


@dataclass(frozen=True)
class _ReplayMeshRef:
    pass


def _make_replay_graph_module(graph_module):
    from torch.distributed.device_mesh import DeviceMesh

    from autoparallel.export_json import _extract_module_path, _get_phase

    copied = torch.fx.Graph()
    node_map = {}
    for node in graph_module.graph.nodes:
        new_node = copied.node_copy(node, lambda input_node: node_map[input_node])
        meta = {
            key: tree_map_only(DeviceMesh, lambda _: _ReplayMeshRef(), value)
            for key, value in node.meta.items()
            if key in _REPLAY_META_KEYS
        }
        module_path = _extract_module_path(node)
        if module_path is not None:
            meta["module_path"] = module_path
        if node.op not in ("placeholder", "output"):
            meta["phase"] = _get_phase(node)
        new_node.meta = meta
        node_map[node] = new_node
    replay_graph_module = torch.fx.GraphModule(graph_module, copied)
    replay_graph_module.meta = {}
    return replay_graph_module


def _encode_replay_graph(graph_module):
    from torch._subclasses.fake_tensor import unset_fake_temporarily
    from torch.fx._graph_pickler import GraphPickler, Options

    replay_graph_module = _make_replay_graph_module(graph_module)
    with unset_fake_temporarily():
        return GraphPickler.dumps(
            replay_graph_module,
            Options(
                ops_filter=None,
                node_metadata_key_filter=lambda key: key in _REPLAY_META_KEYS,
            ),
        )


def _decode_replay_graph(data, fake_mode):
    from torch.fx._graph_pickler import GraphPickler

    graph_module = GraphPickler.loads(data, fake_mode)
    if not isinstance(graph_module, torch.fx.GraphModule):
        raise RuntimeError("Optimizer replay graph did not decode to an FX GraphModule")
    graph_module.graph.lint()
    return graph_module


def _replay_constraints(opt, saved_log):
    nodes_by_name = {node.name: node for node in opt.graph.nodes}
    for method_name, saved_kwargs in saved_log:
        # These composite calls are followed by the add_node_constraint calls
        # that they expanded into. Replaying both would duplicate the ILP rows.
        if method_name in {
            "add_sharded_input_constraint",
            "add_sharded_output_constraint",
        }:
            continue
        kwargs = dict(saved_kwargs)
        if method_name == "remove_constraints":
            memory_names = {"memory_constraint_high", "memory_constraint_low"}
            for name in kwargs["names"]:
                opt.prob.constraints.pop(name, None)
                opt._node_constraint_names.pop(name, None)
                if name in memory_names:
                    opt._memory_constraint = None
            continue
        if "node_name" in kwargs:
            node_name = kwargs.pop("node_name")
            kwargs["node"] = nodes_by_name[node_name]
        getattr(opt, method_name)(**kwargs)
    opt._constraint_log = list(saved_log)


def save_optimizer_replay(opt, path):
    """Save the graph and compact inputs needed to regenerate an optimizer."""
    module_nodes = [node.name for node in opt.graph.nodes if node.op == "call_module"]
    if module_nodes:
        raise RuntimeError(
            "Optimizer replay requires a functional FX graph, but found module "
            f"calls at nodes: {module_nodes}"
        )

    payload = {
        "version": 1,
        "torch_git_version": torch.version.git_version,
        "graph_module": _encode_replay_graph(opt.gm),
        "mesh": {
            "device_type": opt.mesh.device_type,
            "shape": list(opt.mesh.shape),
            "dim_names": (
                list(opt.mesh.mesh_dim_names) if opt.mesh.mesh_dim_names else None
            ),
        },
        "cost_model_device": {
            "name": opt.cost_model_device_name,
            "count": opt.cost_model_device_count,
        },
        "optimizer": {
            "force_grad_reduce_in_higher_precision": (
                opt.force_grad_reduce_in_higher_precision
            ),
            "repeated_subgraphs": opt.repeated_subgraphs,
            "persistent_aliases": opt.persistent_aliases,
            "solver_time_limit_seconds": opt.solver_time_limit_seconds,
            "prefetch_discount": opt._prefetch_discount,
            "constraint_log": opt._constraint_log,
        },
        "nccl_topology_config": opt.nccl_topology_config,
    }
    with gzip.open(path, "wb") as replay_file:
        torch.save(payload, replay_file)


def load_optimizer_replay(cls, path):
    """Regenerate a ShardingOptimizer from a compact graph replay seed."""
    from torch._subclasses.fake_tensor import FakeTensorMode, unset_fake_temporarily
    from torch.distributed.device_mesh import DeviceMesh
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    from autoparallel.cost_models.collective_runtime_estimation import (
        get_nccl_topo_config,
        override_cost_model_device_count,
        set_nccl_topo_config,
    )
    from autoparallel.cost_models.compute_estimation import (
        override_cost_model_device_name,
    )

    with gzip.open(path, "rb") as replay_file:
        payload = torch.load(replay_file, weights_only=False)
    version = payload.get("version")
    if version != 1:
        raise RuntimeError(f"Unsupported optimizer replay version: {version}")
    if payload.get("torch_git_version") != torch.version.git_version:
        raise RuntimeError(
            "Optimizer replay requires the capture PyTorch revision "
            f"({payload.get('torch_git_version')}), but the current revision is "
            f"{torch.version.git_version}"
        )

    mesh_config = payload["mesh"]
    shape = tuple(mesh_config["shape"])
    with unset_fake_temporarily():
        mesh_values = torch.arange(math.prod(shape)).reshape(shape)
        mesh = DeviceMesh(
            mesh_config["device_type"],
            mesh_values,
            mesh_dim_names=(
                tuple(mesh_config["dim_names"]) if mesh_config["dim_names"] else None
            ),
            _init_backend=False,
            _rank=0,
        )

    fake_mode = FakeTensorMode(shape_env=ShapeEnv(), static_shapes=False)
    graph_module = _decode_replay_graph(payload["graph_module"], fake_mode)
    for node in graph_module.graph.nodes:
        node.meta = tree_map_only(_ReplayMeshRef, lambda _: mesh, node.meta)

    config = payload["optimizer"]
    device_config = payload["cost_model_device"]
    previous_topology = get_nccl_topo_config()
    try:
        set_nccl_topo_config(payload["nccl_topology_config"])
        with (
            override_cost_model_device_name(device_config["name"]),
            override_cost_model_device_count(device_config["count"]),
            fake_mode,
        ):
            opt = cls(
                graph_module,
                mesh,
                config["force_grad_reduce_in_higher_precision"],
                repeated_subgraphs=config["repeated_subgraphs"],
                persistent_aliases=config["persistent_aliases"],
                solver_time_limit_seconds=config["solver_time_limit_seconds"],
            )
    finally:
        set_nccl_topo_config(previous_topology)

    _replay_constraints(opt, config["constraint_log"])
    if config["prefetch_discount"] != 1.0:
        opt.apply_prefetch_discount(config["prefetch_discount"])
    return opt


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

    # Convert selected_keys to node-name-based representation (if solved).
    # Only persist root keys — linked keys are re-expanded on load.
    selected_keys_by_name = None
    if hasattr(opt, "selected_keys"):
        selected_keys_by_name = {}
        linked = opt._cluster_linked_node_idxs
        for key in opt.selected_keys:
            node_idx, argi, out_idx, inp_idx = key
            if node_idx in linked:
                continue
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
        node.meta.setdefault(
            "is_tensor_value", isinstance(node.meta.get("val"), torch.Tensor)
        )
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
        "persistent_aliases": opt.persistent_aliases,
        "prefetch_discount": opt._prefetch_discount,
        "solver_time_limit_seconds": opt.solver_time_limit_seconds,
        "skipped_clusters": opt.skipped_clusters,
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
    if save_dict["version"] != 1:
        raise RuntimeError(f"Unsupported save file version: {save_dict['version']}")

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
    opt.persistent_aliases = save_dict.get("persistent_aliases", {})
    opt._prefetch_discount = save_dict.get("prefetch_discount", 1.0)
    opt.solver_time_limit_seconds = save_dict.get("solver_time_limit_seconds")
    opt._prefetchable_keys = None
    opt.skipped_clusters = save_dict.get("skipped_clusters", [])
    opt._constraint_log = []
    opt._memory_constraint = None
    opt._node_constraint_names = {}
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

    saved_log = list(save_dict["constraint_log"])
    _replay_constraints(opt, saved_log)
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
    """Restore selected_keys and PuLP variable values from a saved state."""
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


def save_placements(opt, path):
    """Save the current placement choices as a lightweight JSON file."""
    from torch.distributed.tensor._op_schema import _pretty_print_spec

    placements = {}
    solution = opt._extract_and_validate_solution()
    for node, strategy in solution.items():
        placements[node.name] = {
            "output": _pretty_print_spec(strategy.output_specs),
            "inputs": [_pretty_print_spec(s) for s in strategy.input_specs],
        }

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


def load_placements(opt, path):
    """Load placements from a JSON file and return a dict[Node, OpSpec].

    Matches saved placement strings against the strategies in opt.strats.
    The returned dict can be passed directly to apply_placement().
    """
    from torch.distributed.tensor._op_schema import _pretty_print_spec

    with open(path) as f:
        save_dict = json.load(f)

    if save_dict["version"] != 1:
        raise RuntimeError(
            f"Unsupported placements file version: {save_dict['version']}"
        )

    # Validate mesh compatibility
    saved_shape = tuple(save_dict["mesh_shape"])
    if tuple(opt.mesh.shape) != saved_shape:
        raise RuntimeError(
            f"Mesh shape mismatch: saved {list(saved_shape)}, "
            f"current {list(opt.mesh.shape)}"
        )
    saved_names = (
        tuple(save_dict["mesh_dim_names"]) if save_dict.get("mesh_dim_names") else None
    )
    current_names = tuple(opt.mesh.mesh_dim_names) if opt.mesh.mesh_dim_names else None
    if saved_names != current_names:
        raise RuntimeError(
            f"Mesh dim names mismatch: saved {list(saved_names or [])}, "
            f"current {list(current_names or [])}"
        )

    nodes_by_name = {node.name: node for node in opt.nodes}
    solution = {}
    for node_name, entry in save_dict["placements"].items():
        if node_name not in nodes_by_name:
            raise RuntimeError(
                f"Node '{node_name}' from saved placements not found in current graph. "
                "The model may have changed since the placements were saved."
            )
        output_str = entry["output"]
        input_strs = entry["inputs"]

        node = nodes_by_name[node_name]
        strat = opt.strats[node]
        matched = None
        for s in strat.strategies:
            if _pretty_print_spec(s.output_specs) != output_str:
                continue
            if [_pretty_print_spec(sp) for sp in s.input_specs] != input_strs:
                continue
            matched = s
            break
        if matched is None:
            raise RuntimeError(
                f"Placement '{output_str}' for node '{node_name}' not found "
                f"in available strategies. The model or mesh may have changed."
            )
        solution[node] = matched

    return opt._to_orig_solution(solution)
