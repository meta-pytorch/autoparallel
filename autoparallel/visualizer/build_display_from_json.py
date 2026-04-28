# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
HTML visualizer that consumes the JSON dict from export_json.py.

Supports cluster collapsing for multi-layer models, phase filtering,
redistribution highlighting, and hierarchical module tree view.
"""


def _classify_placement(placement):
    if not placement:
        return "unknown"
    p = placement.strip()
    if p in ("RR", "R"):
        return "replicated"
    parts = []
    i = 0
    while i < len(p):
        if p[i] == "S" and i + 1 < len(p) and p[i + 1] == "(":
            end = p.index(")", i)
            parts.append(p[i : end + 1])
            i = end + 1
        elif p[i] == "R":
            parts.append("R")
            i += 1
        elif p[i] == "P" and i + 1 < len(p) and p[i + 1] == "(":
            end = p.index(")", i)
            parts.append(p[i : end + 1])
            i = end + 1
        else:
            i += 1
    if len(parts) == 2:
        d0, d1 = parts
        if d0.startswith("S") and d1 == "R":
            return "fsdp"
        if d0 == "R" and d1.startswith("S"):
            return "tp"
        if d0.startswith("S") and d1.startswith("S"):
            return "fsdp+tp"
        if d0.startswith("P") or d1.startswith("P"):
            return "partial"
    return "other"


_STRATEGY_COLORS = {
    "fsdp": "#3B82F6",
    "tp": "#10B981",
    "fsdp+tp": "#8B5CF6",
    "replicated": "#6B7280",
    "partial": "#F59E0B",
    "other": "#EC4899",
    "unknown": "#D1D5DB",
}

_STRATEGY_BGS = {
    "fsdp": "#EFF6FF",
    "tp": "#ECFDF5",
    "fsdp+tp": "#F5F3FF",
    "replicated": "#F9FAFB",
    "partial": "#FFFBEB",
    "other": "#FDF2F8",
    "unknown": "#F9FAFB",
}

_MODULE_PATH_PREFIXES = ("L['self'].", "_export_root.")


def _infer_collective(src, dst):
    if not src or not dst or src == dst:
        return None
    if "P(sum)" in src and "S(" in dst:
        return "ReduceScatter"
    if "S(" in src and "R" in dst and "P" not in dst:
        return "AllGather"
    if "P(sum)" in src and "R" in dst:
        return "AllReduce"
    return "Redistribute"


def _node_total_comm(n):
    return sum(inp.get("comm_cost", 0) for inp in n.get("inputs", []))


def _node_total_cost(n):
    return _node_total_comm(n) + n.get("compute_cost", 0)


def _placement_tooltip(placement, shape, dtype, mesh):
    """Build a human-readable tooltip for a placement chip."""
    if not placement or not mesh:
        return ""
    dim_names = mesh.get("dim_names") or [f"dim{i}" for i in range(len(mesh["shape"]))]
    mesh_shape = mesh["shape"]

    # Parse placement parts
    parts = []
    i = 0
    p = placement.strip()
    while i < len(p):
        if p[i] == "S" and i + 1 < len(p) and p[i + 1] == "(":
            end = p.index(")", i)
            parts.append(p[i : end + 1])
            i = end + 1
        elif p[i] == "R":
            parts.append("R")
            i += 1
        elif p[i] == "P" and i + 1 < len(p) and p[i + 1] == "(":
            end = p.index(")", i)
            parts.append(p[i : end + 1])
            i = end + 1
        else:
            i += 1

    if not parts or not shape:
        return placement

    lines = []
    local_shape = list(shape)
    for mesh_dim, (part, dname, msize) in enumerate(
        zip(parts, dim_names, mesh_shape)
    ):
        if part == "R":
            lines.append(f"{dname}: Replicate ({msize} copies)")
        elif part.startswith("S("):
            shard_dim = int(part[2:-1])
            lines.append(f"{dname}: Shard dim {shard_dim} ({msize} ways)")
            if shard_dim < len(local_shape):
                local_shape[shard_dim] = local_shape[shard_dim] // msize
        elif part.startswith("P("):
            lines.append(f"{dname}: Partial ({msize}-way reduction pending)")

    shape_str = "\u00d7".join(str(s) for s in shape)
    local_str = "\u00d7".join(str(s) for s in local_shape)
    lines.insert(0, f"Global: {dtype}[{shape_str}]")
    lines.append(f"Local: {dtype}[{local_str}]")
    return "\n".join(lines)


def _placement_chip_html(placement, bg, color, shape, dtype, mesh):
    """Build a placement chip span with an optional tooltip child."""
    tooltip = _placement_tooltip(placement, shape, dtype, mesh)
    tooltip_span = ""
    if tooltip:
        tooltip_span = f'<span class="chip-tooltip">{_esc(tooltip)}</span>'
    return (
        f'<span class="placement-chip" style="background:{bg};color:{color}">'
        f'{_esc(placement)}{tooltip_span}</span>'
    )


def _esc(s):
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _strip_module_prefix(path):
    for prefix in _MODULE_PATH_PREFIXES:
        if path.startswith(prefix):
            path = path[len(prefix):]
    return path


class _ModuleTreeNode:
    __slots__ = ("name", "children", "own_nodes", "_all_nodes_cache")

    def __init__(self, name):
        self.name = name
        self.children = {}
        self.own_nodes = []
        self._all_nodes_cache = None

    def all_nodes(self):
        if self._all_nodes_cache is not None:
            return self._all_nodes_cache
        result = list(self.own_nodes)
        for child in self.children.values():
            result.extend(child.all_nodes())
        self._all_nodes_cache = result
        return result

    def invalidate_cache(self):
        self._all_nodes_cache = None
        for child in self.children.values():
            child.invalidate_cache()


def _build_module_tree(display_nodes, min_group_size=3):
    """Build a hierarchical tree from module_path, then prune it."""
    root = _ModuleTreeNode("")

    for n in display_nodes:
        raw_path = n.get("module_path", "")
        if not raw_path:
            if n.get("op") == "placeholder":
                key = "inputs"
            else:
                key = "other"
            root.children.setdefault(key, _ModuleTreeNode(key)).own_nodes.append(n)
            continue

        path = _strip_module_prefix(raw_path)
        parts = path.split(".")
        node = root
        for part in parts[:-1]:
            if part not in node.children:
                node.children[part] = _ModuleTreeNode(part)
            node = node.children[part]
        leaf_name = parts[-1]
        if leaf_name not in node.children:
            node.children[leaf_name] = _ModuleTreeNode(leaf_name)
        node.children[leaf_name].own_nodes.append(n)

    _prune_tree(root, min_group_size)
    return root


def _prune_tree(node, min_group_size):
    """Prune the tree bottom-up with two rules:
    1. Collapse single-child chains (when parent has no own_nodes)
    2. Promote small groups (absorb children with fewer than N total nodes)
    """
    # Recurse first (bottom-up)
    for child in list(node.children.values()):
        _prune_tree(child, min_group_size)

    # Rule 1: single-child collapse
    while len(node.children) == 1 and not node.own_nodes:
        only_key = next(iter(node.children))
        only_child = node.children[only_key]
        if node.name:
            node.name = f"{node.name}.{only_child.name}"
        else:
            node.name = only_child.name
        node.own_nodes = only_child.own_nodes
        node.children = only_child.children
        node.invalidate_cache()

    # Rule 2: promote small groups
    for key in list(node.children):
        child = node.children[key]
        if len(child.all_nodes()) < min_group_size:
            node.own_nodes.extend(child.all_nodes())
            del node.children[key]
            node.invalidate_cache()


def _tree_node_stats(tree_node):
    """Compute summary stats for a tree node (all descendants)."""
    all_n = tree_node.all_nodes()
    comm = sum(n["_total_comm"] for n in all_n)
    compute = sum(n.get("compute_cost", 0) for n in all_n)
    strategies = [n["_strategy"] for n in all_n]
    dominant = max(set(strategies), key=strategies.count) if strategies else "unknown"
    all_colls = []
    for n in all_n:
        all_colls.extend(n["_collectives"])
    return {
        "num_nodes": len(all_n),
        "comm_cost": comm,
        "compute_cost": compute,
        "dominant": dominant,
        "color": _STRATEGY_COLORS.get(dominant, "#D1D5DB"),
        "bg": _STRATEGY_BGS.get(dominant, "#F9FAFB"),
        "collectives": all_colls,
    }


def _render_tree_block(tree_node, cluster_counts, mesh, depth=0):
    """Recursively render a module tree node as nested HTML blocks."""
    stats = _tree_node_stats(tree_node)
    if stats["num_nodes"] == 0:
        return ""

    indent = depth * 20
    coll_html = ""
    for cname, ctype, cfrom, cto, ccost in stats["collectives"][:5]:
        coll_html += f'<span class="collective-badge">{_esc(ctype)}: {_esc(cfrom)}&rarr;{_esc(cto)} ({ccost:.0f})</span> '
    if len(stats["collectives"]) > 5:
        coll_html += f'<span class="collective-badge">+{len(stats["collectives"]) - 5} more</span>'

    # Detail table for own_nodes (nodes at this exact level)
    detail_rows = ""
    for n in tree_node.own_nodes:
        comm = n["_total_comm"]
        compute = n.get("compute_cost", 0)
        cost_class = "cost-high" if comm > 100 else "cost-med" if comm > 0 else "cost-low"
        shape_str = ",".join(str(s) for s in n.get("shape", []))
        dtype = n.get("dtype", "")
        placement = n.get("placement", "")
        phase = n.get("phase", "")
        row_class = "arch-row arch-bwd" if phase == "backward" else "arch-row"
        cluster_html = ""
        cid = n.get("cluster_id")
        if cid is not None and "cluster_root" not in n and cid in cluster_counts:
            cluster_html = f'<span class="cluster-badge">&times;{cluster_counts[cid] + 1}</span>'
        phase_badge = (
            '<span class="phase-badge phase-bwd">bwd</span>'
            if phase == "backward"
            else '<span class="phase-badge phase-fwd">fwd</span>'
        )

        chip = _placement_chip_html(placement, n["_bg"], n["_color"], n.get("shape"), dtype, mesh)

        detail_rows += (
            f'<tr class="{row_class}" data-phase="{phase}">'
            f'<td style="font-family:monospace">{_esc(n["name"])}{cluster_html}</td>'
            f'<td>{phase_badge}</td>'
            f'<td>{_esc(dtype)}[{shape_str}]</td>'
            f'<td>{chip}</td>'
            f'<td class="{cost_class}" style="font-family:monospace">{comm:.1f}</td>'
            f'<td style="font-family:monospace">{compute:.1f}</td>'
            f'</tr>'
        )

    has_children = bool(tree_node.children)
    has_detail = bool(detail_rows)

    # For leaf nodes or nodes with only own_nodes, click expands the detail table
    # For interior nodes, click expands the children
    toggle_target = "expanded" if (has_detail or has_children) else ""

    # Compute which phases this block contains
    all_phases = set(n.get("phase", "") for n in tree_node.all_nodes())
    phases_attr = " ".join(sorted(all_phases))

    html = f'''
<div class="func-block" style="background:{stats['bg']};border-color:{stats['color']};margin-left:{indent}px"
     data-phases="{phases_attr}"
     onclick="event.stopPropagation(); this.classList.toggle('expanded')">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <div>
      <span class="func-name">{_esc(tree_node.name)}</span>
      <span class="strategy-badge" style="background:{stats['color']}">{stats['dominant'].upper()}</span>
      {f'<span style="font-size:11px;color:#94a3b8;margin-left:4px">&#9660;</span>' if has_children else ''}
    </div>
    <div style="text-align:right;font-size:12px;color:#64748b">
      {stats['num_nodes']} ops &middot; comm: {stats['comm_cost']:.0f} &middot; compute: {stats['compute_cost']:.0f}
    </div>
  </div>
  <div class="func-meta">{coll_html if coll_html else "No redistributions"}</div>
  <div class="func-details">'''

    if has_detail:
        html += f'''
    <table><tr><th>Node</th><th>Phase</th><th>Shape</th><th>Placement</th><th>Comm</th><th>Compute</th></tr>
    {detail_rows}</table>'''

    if has_children:
        if has_detail:
            html += '<div style="margin-top:10px"></div>'
        # Render children in execution order (dict preserves insertion order)
        for child in tree_node.children.values():
            html += _render_tree_block(child, cluster_counts, mesh, depth + 1)

    html += '</div></div>'
    return html


def _flatten_tree_for_costs(tree_node, result=None, prefix=""):
    """Flatten the tree into a list of (name, stats) for the cost-by-module view.
    Only includes nodes that have children (interior nodes) or own_nodes (leaves)."""
    if result is None:
        result = []
    full_name = f"{prefix}.{tree_node.name}" if prefix else tree_node.name
    stats = _tree_node_stats(tree_node)
    if stats["num_nodes"] > 0:
        result.append((full_name or "root", stats))
    for child in tree_node.children.values():
        _flatten_tree_for_costs(child, result, full_name)
    return result


def generate_visualization_html(data: dict) -> str:
    """Generate interactive HTML visualization from the JSON export dict.

    Args:
        data: The dict returned by export_sharding_json() or
              ShardingOptimizer.get_json().

    Returns:
        A self-contained HTML string.
    """
    mesh = data["mesh"]
    nodes = data["nodes"]
    summary = data["summary"]

    # Precompute per-node derived fields
    for n in nodes:
        placement = n.get("placement", "")
        n["_strategy"] = _classify_placement(placement)
        n["_color"] = _STRATEGY_COLORS.get(n["_strategy"], "#D1D5DB")
        n["_bg"] = _STRATEGY_BGS.get(n["_strategy"], "#F9FAFB")
        n["_total_comm"] = _node_total_comm(n)
        # Collectives on input edges
        collectives = []
        for inp in n.get("inputs", []):
            coll = _infer_collective(
                inp.get("src_placement"), inp.get("dst_placement")
            )
            if coll:
                collectives.append(
                    (inp["name"], coll, inp["src_placement"], inp["dst_placement"], inp["comm_cost"])
                )
        n["_collectives"] = collectives

    # Filter out output node for most views
    compute_nodes = [n for n in nodes if n.get("op") != "output"]

    # Cluster info
    has_clusters = any("cluster_id" in n for n in compute_nodes)
    root_nodes = [n for n in compute_nodes if "cluster_root" not in n]
    # Count instances per cluster_id
    cluster_counts = {}
    for n in compute_nodes:
        cid = n.get("cluster_id")
        if cid is not None and "cluster_root" not in n:
            cluster_counts[cid] = 1
    for n in compute_nodes:
        cid = n.get("cluster_id")
        if cid is not None and "cluster_root" in n:
            cluster_counts[cid] = cluster_counts.get(cid, 0) + 1

    # Strategy distribution (root nodes only when clustered)
    display_nodes = root_nodes if has_clusters else compute_nodes
    strategy_counts = {}
    for n in display_nodes:
        s = n["_strategy"]
        strategy_counts[s] = strategy_counts.get(s, 0) + 1

    # Build hierarchical module tree
    module_tree = _build_module_tree(display_nodes)

    # Flat list for cost-by-module view
    module_cost_list = _flatten_tree_for_costs(module_tree)
    module_cost_list.sort(key=lambda x: x[1]["comm_cost"] + x[1]["compute_cost"], reverse=True)

    # Top costly nodes
    costly = sorted(display_nodes, key=_node_total_cost, reverse=True)
    top_costly = [n for n in costly[:20] if _node_total_cost(n) > 0]

    # Nodes with redistributions
    redist_nodes = [n for n in display_nodes if n["_collectives"]]

    # Mesh description
    mesh_shape = "×".join(str(d) for d in mesh["shape"])
    dim_names = mesh.get("dim_names")
    mesh_desc = f"{mesh_shape}"
    if dim_names:
        mesh_desc += f" ({', '.join(dim_names)})"

    # Per-layer cost breakdown (for repeated architectures)
    import re as _re

    layer_costs = {}  # layer_idx -> {"comm": float, "compute": float, "num_nodes": int}
    non_layer_cost = {"comm": 0.0, "compute": 0.0, "num_nodes": 0, "name": "non-layer"}
    for n in display_nodes:
        mp = n.get("module_path", "")
        m = _re.search(r"layers\.(\d+)", mp)
        comm = n["_total_comm"]
        compute = n.get("compute_cost", 0)
        if m:
            idx = int(m.group(1))
            if idx not in layer_costs:
                layer_costs[idx] = {"comm": 0.0, "compute": 0.0, "num_nodes": 0}
            layer_costs[idx]["comm"] += comm
            layer_costs[idx]["compute"] += compute
            layer_costs[idx]["num_nodes"] += 1
        else:
            non_layer_cost["comm"] += comm
            non_layer_cost["compute"] += compute
            non_layer_cost["num_nodes"] += 1

    total_cost = summary["total"] or 1
    comm_pct = summary["comm"] / total_cost * 100
    compute_pct = summary["compute"] / total_cost * 100
    trans_pct = summary["transition"] / total_cost * 100

    # ===== BUILD HTML =====
    html = f'''<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
.ap-viz {{ margin: 0; padding: 0; box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f8fafc; color: #1e293b; }}
.ap-viz *, .ap-viz *::before, .ap-viz *::after {{ box-sizing: border-box; }}
.ap-viz .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
.ap-viz h1 {{ font-size: 24px; font-weight: 700; margin: 0 0 4px 0; color: #1e293b; }}
.ap-viz .subtitle {{ color: #64748b; font-size: 14px; margin-bottom: 20px; }}
.ap-viz .stats-row {{ display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }}
.ap-viz .stat-card {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px 18px; flex: 1; min-width: 140px; color: #1e293b; }}
.ap-viz .stat-label {{ font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }}
.ap-viz .stat-value {{ font-size: 22px; font-weight: 700; margin-top: 2px; }}
.ap-viz .stat-detail {{ font-size: 12px; color: #64748b; margin-top: 2px; }}
.ap-viz .controls {{ display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; align-items: center; }}
.ap-viz .control-group {{ display: flex; gap: 4px; align-items: center; }}
.ap-viz .control-label {{ font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; margin-right: 4px; }}
.ap-viz .tabs {{ display: flex; gap: 0; border-bottom: 2px solid #e2e8f0; margin-bottom: 20px; }}
.ap-viz .tab {{ padding: 10px 20px; cursor: pointer; font-size: 14px; font-weight: 500; color: #64748b; border-bottom: 2px solid transparent; margin-bottom: -2px; transition: all 0.2s; background: transparent; }}
.ap-viz .tab:hover {{ color: #3b82f6; }}
.ap-viz .tab.active {{ color: #3b82f6; border-bottom-color: #3b82f6; }}
.ap-viz .tab-content {{ display: none; }}
.ap-viz .tab-content.active {{ display: block; }}
.ap-viz .card {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 18px; margin-bottom: 16px; color: #1e293b; }}
.ap-viz .card-title {{ font-size: 14px; font-weight: 600; margin-bottom: 12px; color: #1e293b; }}
.ap-viz .func-block {{ border-radius: 8px; padding: 14px; margin-bottom: 10px; border-left: 4px solid; cursor: pointer; transition: box-shadow 0.2s; }}
.ap-viz .func-block:hover {{ box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.ap-viz .func-name {{ font-weight: 600; font-size: 14px; color: #1e293b; }}
.ap-viz .func-meta {{ font-size: 12px; color: #64748b; margin-top: 4px; }}
.ap-viz .collective-badge {{ display: inline-block; background: #FEE2E2; color: #DC2626; font-size: 11px; padding: 2px 8px; border-radius: 10px; margin: 2px 2px; font-weight: 500; }}
.ap-viz .strategy-badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 500; color: white; }}
.ap-viz .cluster-badge {{ display: inline-block; padding: 1px 6px; border-radius: 8px; font-size: 10px; font-weight: 600; color: #6B7280; background: #F1F5F9; margin-left: 4px; }}
.ap-viz .func-details {{ display: none; margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(0,0,0,0.1); font-size: 12px; }}
.ap-viz .func-block.expanded > .func-details {{ display: block; }}
.ap-viz .cost-bar-container {{ margin-bottom: 10px; }}
.ap-viz .cost-bar-label {{ font-size: 12px; margin-bottom: 3px; display: flex; justify-content: space-between; color: #1e293b; }}
.ap-viz .cost-bar {{ height: 24px; border-radius: 4px; display: flex; overflow: hidden; background: #f1f5f9; }}
.ap-viz .cost-bar-segment {{ height: 100%; display: flex; align-items: center; justify-content: center; font-size: 10px; color: white; font-weight: 500; min-width: 2px; }}
.ap-viz table {{ width: 100%; border-collapse: collapse; font-size: 12px; color: #1e293b; }}
.ap-viz th {{ text-align: left; padding: 8px 10px; background: #f8fafc; border-bottom: 2px solid #e2e8f0; font-weight: 600; color: #64748b; text-transform: uppercase; font-size: 11px; letter-spacing: 0.3px; position: sticky; top: 0; }}
.ap-viz td {{ padding: 7px 10px; border-bottom: 1px solid #f1f5f9; background: white; }}
.ap-viz tr:hover td {{ background: #f8fafc; }}
.ap-viz .placement-chip {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-family: monospace; font-size: 11px; font-weight: 500; position: relative; cursor: default; }}
.ap-viz .placement-chip .chip-tooltip {{ display: none; position: absolute; left: 0; top: 100%; margin-top: 4px; background: #1e293b; color: white; padding: 6px 10px; border-radius: 6px; font-size: 11px; font-family: monospace; white-space: pre; z-index: 1000; pointer-events: none; box-shadow: 0 2px 8px rgba(0,0,0,0.2); }}
.ap-viz .placement-chip:hover .chip-tooltip {{ display: block; }}
.ap-viz .cost-high {{ color: #DC2626; font-weight: 600; }}
.ap-viz .cost-med {{ color: #F59E0B; }}
.ap-viz .cost-low {{ color: #64748b; }}
.ap-viz .filter-bar {{ display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; align-items: center; }}
.ap-viz .filter-btn {{ padding: 4px 12px; border-radius: 16px; border: 1px solid #e2e8f0; background: white; color: #1e293b; cursor: pointer; font-size: 12px; transition: all 0.15s; }}
.ap-viz .filter-btn:hover {{ border-color: #3b82f6; }}
.ap-viz .filter-btn.active {{ background: #3b82f6; color: white; border-color: #3b82f6; }}
.ap-viz .search-box {{ padding: 5px 12px; border-radius: 16px; border: 1px solid #e2e8f0; background: white; color: #1e293b; font-size: 12px; width: 220px; outline: none; transition: border-color 0.15s; }}
.ap-viz .search-box:focus {{ border-color: #3b82f6; }}
.ap-viz .table-scroll {{ max-height: 600px; overflow-y: auto; }}
.ap-viz th.sortable {{ cursor: pointer; user-select: none; }}
.ap-viz th.sortable:hover {{ color: #3b82f6; }}
.ap-viz th.sort-asc::after {{ content: " \\25B2"; font-size: 9px; }}
.ap-viz th.sort-desc::after {{ content: " \\25BC"; font-size: 9px; }}
.ap-viz .legend {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 16px; }}
.ap-viz .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 12px; color: #1e293b; }}
.ap-viz .legend-dot {{ width: 12px; height: 12px; border-radius: 3px; }}
.ap-viz tr.linked-row {{ opacity: 0.45; display: none; }}
.ap-viz tr.linked-row td {{ font-style: italic; }}
.ap-viz tr.arch-bwd {{ opacity: 0.6; }}
.ap-viz .phase-badge {{ display: inline-block; padding: 1px 6px; border-radius: 8px; font-size: 10px; font-weight: 600; }}
.ap-viz .phase-fwd {{ color: #10B981; background: #ECFDF5; }}
.ap-viz .phase-bwd {{ color: #6B7280; background: #F1F5F9; }}
</style></head><body>
<div class="ap-viz">
<div class="container">
<h1>AutoParallel Strategy Visualizer</h1>
<div class="subtitle">Mesh: {_esc(mesh_desc)} &middot; {len(compute_nodes)} nodes &middot; {len(redist_nodes)} redistributions'''

    if has_clusters:
        n_clusters = len(cluster_counts)
        max_instances = max(cluster_counts.values()) + 1 if cluster_counts else 1
        html += f' &middot; {n_clusters} clusters (&times;{max_instances} layers)'

    html += f'''</div>

<div class="stats-row">
  <div class="stat-card">
    <div class="stat-label">Total Cost</div>
    <div class="stat-value">{summary["total"]:,.0f}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Communication</div>
    <div class="stat-value" style="color:#DC2626">{summary["comm"]:,.0f}</div>
    <div class="stat-detail">{comm_pct:.0f}% of total</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Compute</div>
    <div class="stat-value" style="color:#10B981">{summary["compute"]:,.0f}</div>
    <div class="stat-detail">{compute_pct:.0f}% of total</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Transitions</div>
    <div class="stat-value" style="color:#F59E0B">{summary["transition"]:,.0f}</div>
    <div class="stat-detail">{trans_pct:.0f}% of total</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Redistributions</div>
    <div class="stat-value">{len(redist_nodes)}</div>
  </div>
</div>

<div class="legend">'''

    for strat in ("fsdp", "tp", "fsdp+tp", "partial", "replicated", "other"):
        count = strategy_counts.get(strat, 0)
        if count > 0:
            html += f'<div class="legend-item"><div class="legend-dot" style="background:{_STRATEGY_COLORS[strat]}"></div>{strat.upper()} ({count})</div>'

    html += '</div>'

    # Controls bar
    html += '''
<div class="controls">
  <div class="control-group">
    <span class="control-label">Phase</span>
    <button class="filter-btn active" onclick="filterPhase('all', this)">All</button>
    <button class="filter-btn" onclick="filterPhase('forward', this)">Forward</button>
    <button class="filter-btn" onclick="filterPhase('backward', this)">Backward</button>
  </div>'''

    if has_clusters:
        html += '''
  <div class="control-group">
    <span class="control-label">Detail View</span>
    <button class="filter-btn active" id="btn-cluster-collapse" onclick="toggleClusters(this)">Single Layer</button>
  </div>'''

    html += '</div>'

    # Tabs
    html += '''
<div class="tabs">
  <div class="tab active" onclick="switchTab('arch', this)">Architecture</div>
  <div class="tab" onclick="switchTab('cost', this)">Cost Breakdown</div>
  <div class="tab" onclick="switchTab('detail', this)">All Nodes</div>
</div>'''

    # ===== ARCHITECTURE TAB =====
    html += '''
<div data-tab="arch" class="tab-content active">
<div class="card"><div class="card-title">Computation Blocks by Module</div>'''

    # Render top-level tree children in execution order (dict preserves
    # insertion order from graph traversal)
    for child in module_tree.children.values():
        html += _render_tree_block(child, cluster_counts, mesh, depth=0)

    html += '</div></div>'

    # ===== COST BREAKDOWN TAB =====
    html += f'''
<div data-tab="cost" class="tab-content">
<div class="card"><div class="card-title">Cost Distribution</div>
<div class="cost-bar-container">
  <div class="cost-bar-label"><span>Overall Breakdown</span></div>
  <div class="cost-bar" style="height:32px">
    <div class="cost-bar-segment" style="width:{comm_pct}%;background:#DC2626">Comm {comm_pct:.0f}%</div>
    <div class="cost-bar-segment" style="width:{compute_pct}%;background:#10B981">Compute {compute_pct:.0f}%</div>
    <div class="cost-bar-segment" style="width:{trans_pct}%;background:#F59E0B">{trans_pct:.0f}%</div>
  </div>
</div></div>'''

    # Per-layer cost breakdown (only if layers exist)
    if layer_costs:
        all_layer_entries = [(f"Layer {idx}", lc) for idx, lc in sorted(layer_costs.items())]
        if non_layer_cost["num_nodes"] > 0:
            all_layer_entries.append(("Non-layer ops", non_layer_cost))
        layer_max = max((lc["comm"] + lc["compute"]) for _, lc in all_layer_entries) or 1

        # Check if all layers have the same cost (clustered)
        layer_vals = list(layer_costs.values())
        all_same = len(layer_vals) > 1 and all(
            abs(lc["comm"] - layer_vals[0]["comm"]) < 0.01
            and abs(lc["compute"] - layer_vals[0]["compute"]) < 0.01
            for lc in layer_vals[1:]
        )
        layer_note = ""
        if all_same and has_clusters:
            per_layer_total = layer_vals[0]["comm"] + layer_vals[0]["compute"]
            layer_note = f'<p style="font-size:12px;color:#64748b;margin-bottom:10px">All {len(layer_vals)} layers have identical costs ({per_layer_total:.0f} per layer = {per_layer_total * len(layer_vals):.0f} total)</p>'

        html += f'''
<div class="card"><div class="card-title">Cost by Layer</div>
{layer_note}'''
        for name, lc in all_layer_entries:
            total = lc["comm"] + lc["compute"]
            if total == 0:
                continue
            comm_w = lc["comm"] / layer_max * 100
            comp_w = lc["compute"] / layer_max * 100
            html += f'''<div class="cost-bar-container">
  <div class="cost-bar-label"><span style="font-weight:600">{_esc(name)}</span><span style="color:#64748b">{lc["num_nodes"]} ops &middot; comm: {lc["comm"]:.0f} &middot; compute: {lc["compute"]:.0f}</span></div>
  <div class="cost-bar">
    <div class="cost-bar-segment" style="width:{comm_w}%;background:#DC2626"></div>
    <div class="cost-bar-segment" style="width:{comp_w}%;background:#10B981"></div>
  </div>
</div>'''
        html += '</div>'

    html += '''
<div class="card"><div class="card-title">Top Costly Operations</div>'''

    if top_costly:
        top_max = _node_total_cost(top_costly[0]) or 1
        for n in top_costly:
            total = _node_total_cost(n)
            comm_w = n["_total_comm"] / top_max * 100
            comp_w = n.get("compute_cost", 0) / top_max * 100
            src = n.get("source")
            src_label = f'{src["func"]}: {src["code"]}' if src else n.get("op", "")
            html += f'''<div class="cost-bar-container">
  <div class="cost-bar-label"><span style="font-family:monospace">{_esc(n["name"])}</span><span style="color:#64748b">{_esc(src_label)} &middot; total: {total:.1f}</span></div>
  <div class="cost-bar">
    <div class="cost-bar-segment" style="width:{comm_w}%;background:#DC2626" title="comm: {n['_total_comm']:.1f}"></div>
    <div class="cost-bar-segment" style="width:{comp_w}%;background:#10B981" title="compute: {n.get('compute_cost', 0):.1f}"></div>
  </div>
</div>'''

    # Cost by module
    html += '</div><div class="card"><div class="card-title">Cost by Module</div>'
    func_max = max((s["comm_cost"] + s["compute_cost"]) for _, s in module_cost_list) if module_cost_list else 1
    for mname, stats in module_cost_list:
        total = stats["comm_cost"] + stats["compute_cost"]
        if total == 0:
            continue
        comm_w = stats["comm_cost"] / func_max * 100
        comp_w = stats["compute_cost"] / func_max * 100
        html += f'''<div class="cost-bar-container">
  <div class="cost-bar-label"><span style="font-weight:600">{_esc(mname)}</span><span style="color:#64748b">{stats["num_nodes"]} ops &middot; total: {total:.0f}</span></div>
  <div class="cost-bar">
    <div class="cost-bar-segment" style="width:{comm_w}%;background:#DC2626"></div>
    <div class="cost-bar-segment" style="width:{comp_w}%;background:#10B981"></div>
  </div>
</div>'''

    html += '</div></div>'

    # ===== ALL NODES TAB =====
    html += '''
<div data-tab="detail" class="tab-content">
<div class="card">
<div class="card-title">All Nodes</div>
<div class="filter-bar">
  <input type="text" class="search-box" placeholder="Search nodes..." oninput="searchNodes(this)">
  <button class="filter-btn active" onclick="filterNodes('all', this)">All</button>
  <button class="filter-btn" onclick="filterNodes('placeholder', this)">Params</button>
  <button class="filter-btn" onclick="filterNodes('forward', this)">Forward</button>
  <button class="filter-btn" onclick="filterNodes('backward', this)">Backward</button>
  <button class="filter-btn" onclick="filterNodes('redist', this)">Has Redistribution</button>
</div>
<div class="table-scroll">
<table data-role="node-table"><thead>
<tr><th class="sortable" onclick="sortTable(0, 'str', this)">Name</th><th class="sortable" onclick="sortTable(1, 'str', this)">Op</th><th class="sortable" onclick="sortTable(2, 'str', this)">Phase</th><th>Shape</th><th class="sortable" onclick="sortTable(4, 'str', this)">Placement</th><th class="sortable" onclick="sortTable(5, 'num', this)">Comm</th><th class="sortable" onclick="sortTable(6, 'num', this)">Compute</th><th>Redistribution</th><th class="sortable" onclick="sortTable(8, 'str', this)">Module</th></tr>
</thead><tbody>'''

    all_display = compute_nodes
    for n in all_display:
        comm = n["_total_comm"]
        compute = n.get("compute_cost", 0)
        cost_class = "cost-high" if comm > 100 else "cost-med" if comm > 0 else "cost-low"
        shape_str = ",".join(str(s) for s in n.get("shape", []))
        dtype = n.get("dtype", "")
        placement = n.get("placement", "")
        phase = n.get("phase", "")
        op_name = n.get("op", "")
        module = n.get("module_path", "")

        # Collective badges
        coll_html = ""
        for _, ctype, cfrom, cto, ccost in n["_collectives"]:
            coll_html += f'<span class="collective-badge">{_esc(ctype)}: {_esc(cfrom)}&rarr;{_esc(cto)} ({ccost:.0f})</span> '

        is_linked = "cluster_root" in n
        row_class = "node-row linked-row" if is_linked else "node-row"
        cluster_html = ""
        cid = n.get("cluster_id")
        if cid is not None and not is_linked and cid in cluster_counts:
            cluster_html = f'<span class="cluster-badge">&times;{cluster_counts[cid] + 1}</span>'

        data_attrs = (
            f'data-phase="{phase}" '
            f'data-op="{_esc(op_name)}" '
            f'data-comm="{comm}" '
            f'data-redist="{1 if n["_collectives"] else 0}" '
            f'data-linked="{1 if is_linked else 0}"'
        )

        chip = _placement_chip_html(placement, n['_bg'], n['_color'], n.get("shape"), dtype, mesh)

        html += f'''<tr class="{row_class}" {data_attrs}>
  <td style="font-family:monospace;font-size:11px">{_esc(n["name"])}{cluster_html}</td>
  <td style="font-size:11px">{_esc(op_name)}</td>
  <td style="font-size:11px">{phase}</td>
  <td style="font-size:11px">{_esc(dtype)}[{shape_str}]</td>
  <td>{chip}</td>
  <td class="{cost_class}" style="font-family:monospace">{comm:.1f}</td>
  <td style="font-family:monospace">{compute:.1f}</td>
  <td>{coll_html}</td>
  <td style="font-size:11px;color:#64748b">{_esc(module)}</td>
</tr>'''

    html += '''</tbody></table></div></div></div>

<script>
function _root(el) { return el.closest('.ap-viz'); }

function switchTab(id, btn) {
  var r = _root(btn);
  r.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  r.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  r.querySelector('.tab-content[data-tab="' + id + '"]').classList.add('active');
  btn.classList.add('active');
}

var clustersCollapsed = true;

function toggleClusters(btn) {
  var r = _root(btn);
  clustersCollapsed = !clustersCollapsed;
  btn.textContent = clustersCollapsed ? 'Single Layer' : 'All Layers';
  btn.classList.toggle('active', clustersCollapsed);
  r.querySelectorAll('.node-row').forEach(row => {
    if (row.dataset.linked === '1') {
      row.style.display = clustersCollapsed ? 'none' : '';
    }
  });
}

function filterPhase(phase, btn) {
  var r = _root(btn);
  r.querySelectorAll('.control-group:first-child .filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  r.querySelectorAll('.node-row').forEach(row => {
    if (phase === 'all') { row.style.display = (clustersCollapsed && row.dataset.linked === '1') ? 'none' : ''; return; }
    var match = row.dataset.phase === phase;
    row.style.display = (!match || (clustersCollapsed && row.dataset.linked === '1')) ? 'none' : '';
  });
  r.querySelectorAll('.arch-row').forEach(row => {
    if (phase === 'all') { row.style.display = ''; return; }
    row.style.display = row.dataset.phase === phase ? '' : 'none';
  });
  r.querySelectorAll('.func-block').forEach(block => {
    if (phase === 'all') { block.style.display = ''; return; }
    var phases = block.dataset.phases || '';
    block.style.display = phases.indexOf(phase) >= 0 ? '' : 'none';
  });
}

var currentSearch = '';

function searchNodes(el) {
  currentSearch = el.value.toLowerCase();
  _applyNodeFilters(_root(el));
}

function _applyNodeFilters(r) {
  r.querySelectorAll('.node-row').forEach(row => {
    if (clustersCollapsed && row.dataset.linked === '1') { row.style.display = 'none'; return; }
    var text = row.textContent.toLowerCase();
    row.style.display = (currentSearch === '' || text.indexOf(currentSearch) >= 0) ? '' : 'none';
  });
}

function filterNodes(type, btn) {
  var r = _root(btn);
  r.querySelectorAll('.filter-bar .filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  r.querySelectorAll('.node-row').forEach(row => {
    var show = true;
    if (type === 'placeholder') show = row.dataset.op === 'placeholder';
    else if (type === 'forward') show = row.dataset.phase === 'forward';
    else if (type === 'backward') show = row.dataset.phase === 'backward';
    else if (type === 'redist') show = row.dataset.redist === '1';
    if (clustersCollapsed && row.dataset.linked === '1') show = false;
    if (show && currentSearch !== '') {
      show = row.textContent.toLowerCase().indexOf(currentSearch) >= 0;
    }
    row.style.display = show ? '' : 'none';
  });
}

var sortState = {col: -1, dir: 'asc'};

function sortTable(colIdx, type, th) {
  var table = _root(th).querySelector('table[data-role="node-table"]');
  var tbody = table.tBodies[0];
  var rows = Array.from(tbody.querySelectorAll('tr.node-row'));

  if (sortState.col === colIdx) {
    sortState.dir = sortState.dir === 'asc' ? 'desc' : 'asc';
  } else {
    sortState.col = colIdx;
    sortState.dir = 'asc';
  }

  table.querySelectorAll('th').forEach(h => h.classList.remove('sort-asc', 'sort-desc'));
  th.classList.add('sort-' + sortState.dir);

  var dir = sortState.dir === 'asc' ? 1 : -1;
  rows.sort(function(a, b) {
    var aVal = a.cells[colIdx].textContent.trim();
    var bVal = b.cells[colIdx].textContent.trim();
    if (type === 'num') {
      return (parseFloat(aVal) - parseFloat(bVal)) * dir;
    }
    return aVal.localeCompare(bVal) * dir;
  });

  rows.forEach(function(row) { tbody.appendChild(row); });
}
</script>
</div></div></body></html>'''

    return html
