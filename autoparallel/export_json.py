# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Structured JSON export of sharding optimization results.

Builds a JSON-serializable dict directly from the FX graph, DeviceMesh,
and optimizer solution — no text parsing required.
"""

import operator
import re
from typing import Any

import torch
import torch.fx

from autoparallel.graph_passes.graph_utils import all_input_nodes


def _extract_source_info(node: torch.fx.Node) -> dict | None:
    """Extract source location from node's stack_trace metadata."""
    stack_trace = node.meta.get("stack_trace")
    if not stack_trace:
        return None
    # stack_trace is a Python traceback string; extract the last frame.
    # Format: '  File "path/file.py", line 53, in func_name\n    code_line\n'
    frames = re.findall(
        r'File "([^"]+)", line (\d+), in (\w+)\n\s+(.*)',
        stack_trace,
    )
    if not frames:
        return None
    file, line, func, code = frames[-1]
    return {"file": file, "line": int(line), "func": func, "code": code.strip()}


def _extract_module_path(node: torch.fx.Node) -> str | None:
    """Extract the deepest nn.Module path from node metadata."""
    # Check for pre-computed string (set by save() for serialized graphs)
    mp = node.meta.get("module_path")
    if mp is not None:
        return mp
    stack = node.meta.get("nn_module_stack") or node.meta.get("fwd_nn_module_stack")
    if not stack:
        return None
    # nn_module_stack is an OrderedDict: key -> (qualified_name, module_class)
    # Take the last entry for the deepest module.
    last_value = list(stack.values())[-1]
    return last_value[0] if last_value else None


def _get_phase(node: torch.fx.Node) -> str:
    """Determine whether a node belongs to the forward or backward pass."""
    # Check for pre-computed phase (set by save() for serialized graphs)
    phase = node.meta.get("phase")
    if phase is not None:
        return phase
    if node.op == "placeholder":
        return "backward" if node.name.startswith("tangents") else "forward"
    if "nn_module_stack" in node.meta:
        return "forward"
    if "fwd_nn_module_stack" in node.meta:
        return "backward"
    # Heuristic: nodes after the first tangent placeholder are backward.
    found_tangent = False
    for n in node.graph.nodes:
        if n.op == "placeholder" and n.name.startswith("tangents"):
            found_tangent = True
        if n is node:
            return "backward" if found_tangent else "forward"
    return "forward"


def _extract_shape_dtype(node: torch.fx.Node) -> tuple[list | None, str | None]:
    """Extract shape and dtype from the node's fake tensor value."""
    val = node.meta.get("val")
    if val is None:
        # Fall back to tensor_meta (available on loaded optimizers where
        # val was stripped during save)
        tm = node.meta.get("tensor_meta")
        if tm is not None:
            return list(tm.shape), str(tm.dtype).removeprefix("torch.")
        return None, None
    if isinstance(val, torch.Tensor):
        return list(val.shape), str(val.dtype).removeprefix("torch.")
    if isinstance(val, (list, tuple)) and val:
        first = next((v for v in val if isinstance(v, torch.Tensor)), None)
        if first is not None:
            return list(first.shape), str(first.dtype).removeprefix("torch.")
    return None, None


def _pretty_print(spec_or_specs) -> str:
    from torch.distributed.tensor._dtensor_spec import DTensorSpec
    from torch.distributed.tensor._op_schema import _pretty_print_spec

    if isinstance(spec_or_specs, DTensorSpec):
        return _pretty_print_spec(spec_or_specs)
    if isinstance(spec_or_specs, (list, tuple)):
        parts = [
            _pretty_print_spec(s) if isinstance(s, DTensorSpec) else str(s)
            for s in spec_or_specs
        ]
        return f"({', '.join(parts)})"
    return str(spec_or_specs)


def _get_output_spec_placement(spec_or_specs) -> str:
    """Get the placement string for output specs, handling tuples."""
    return _pretty_print(spec_or_specs)


def export_sharding_json(
    graph: torch.fx.Graph,
    mesh: Any,
    solution: dict[torch.fx.Node, Any],
    selected_dvs: dict[torch.fx.Node, dict[int, Any]],
    cluster_roots: dict[torch.fx.Node, torch.fx.Node] | None = None,
) -> dict:
    """Export sharding optimization results as a JSON-serializable dict.

    Args:
        graph: The FX graph.
        mesh: DeviceMesh instance.
        solution: Maps node -> OpSpec (the chosen strategy).
        selected_dvs: Maps node -> {argi: DecisionVar} (one per argument),
            built from ShardingOptimizer.selected_keys.
        cluster_roots: Optional mapping from linked nodes to their cluster
            root node (from ShardingOptimizer.cluster_links). When present,
            linked nodes get a "cluster_root" field pointing to the root
            node name, and share the same "cluster_id".

    Returns:
        A dict with keys "mesh", "nodes", and "summary".
    """
    if cluster_roots is None:
        cluster_roots = {}

    # Assign stable cluster IDs: one per unique root node.
    root_to_cluster_id: dict[torch.fx.Node, int] = {}
    for root in cluster_roots.values():
        if root not in root_to_cluster_id:
            root_to_cluster_id[root] = len(root_to_cluster_id)

    mesh_info = {
        "shape": list(mesh.shape),
        "dim_names": list(mesh.mesh_dim_names) if mesh.mesh_dim_names else None,
    }

    nodes_list = []
    total_compute = 0.0
    total_comm = 0.0
    total_transition = 0.0

    for node in graph.nodes:
        if node.op == "output":
            # Include the output node to complete the DAG, but without
            # placement or costs (the optimizer doesn't solve for it).
            output_inputs = [{"name": inp.name} for inp in all_input_nodes(node)]
            nodes_list.append(
                {
                    "name": node.name,
                    "op": "output",
                    "inputs": output_inputs,
                }
            )
            continue
        if node not in solution:
            continue

        strategy = solution[node]
        dvs = selected_dvs[node]

        shape, dtype = _extract_shape_dtype(node)
        placement = _get_output_spec_placement(strategy.output_specs)

        # Build input edges with redistribution info
        input_nodes = all_input_nodes(node)
        inputs = []
        node_comm = 0.0
        node_transition = 0.0
        for argi, pred in enumerate(input_nodes):
            pred_strategy = solution.get(pred)
            if pred_strategy is not None:
                src_specs = pred_strategy.output_specs
                # For getitem nodes, resolve the tuple to the specific element
                if node.target is operator.getitem and isinstance(
                    src_specs, (list, tuple)
                ):
                    idx = node.args[1]
                    src_specs = src_specs[idx]
                src_placement = _get_output_spec_placement(src_specs)
            else:
                src_placement = None
            dv = dvs.get(argi)
            dst_placement = _pretty_print(dv.input_spec) if dv is not None else None
            comm_cost = dv.comm_cost if dv is not None else 0.0
            node_comm += comm_cost
            node_transition += dv.sharding_transition_cost if dv is not None else 0.0

            inputs.append(
                {
                    "name": pred.name,
                    "src_placement": src_placement,
                    "dst_placement": dst_placement,
                    "comm_cost": comm_cost,
                }
            )

        # Compute cost is split across args in the ILP; sum to get the full op cost.
        compute_cost = sum(dv.compute_cost for dv in dvs.values())
        total_compute += compute_cost
        total_comm += node_comm
        total_transition += node_transition

        entry: dict[str, Any] = {
            "name": node.name,
            "op": str(node.target) if node.op == "call_function" else node.op,
            "phase": _get_phase(node),
            "placement": placement,
            "inputs": inputs,
            "compute_cost": compute_cost,
        }

        if dtype is not None:
            entry["dtype"] = dtype
        if shape is not None:
            entry["shape"] = shape

        source = _extract_source_info(node)
        if source is not None:
            entry["source"] = source

        module_path = _extract_module_path(node)
        if module_path is not None:
            entry["module_path"] = module_path

        ac_graph_id = node.meta.get("ac_graph_id")
        recompute = node.meta.get("recompute")
        if ac_graph_id is not None:
            entry["ac_region"] = {
                "id": ac_graph_id,
                "type": recompute.name if recompute is not None else None,
            }

        if node in cluster_roots:
            root = cluster_roots[node]
            entry["cluster_id"] = root_to_cluster_id[root]
            entry["cluster_root"] = root.name
        elif node in root_to_cluster_id:
            entry["cluster_id"] = root_to_cluster_id[node]

        nodes_list.append(entry)

    return {
        "mesh": mesh_info,
        "nodes": nodes_list,
        "summary": {
            "total": total_compute + total_comm + total_transition,
            "comm": total_comm,
            "compute": total_compute,
            "transition": total_transition,
        },
    }
