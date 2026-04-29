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


def _parse_placement_dims(p):
    """Parse a placement string into a list of per-mesh-dim placements.

    Returns a list of strings like ['R', 'S(0)', 'P(sum)'].
    """
    if not p:
        return []
    parts = []
    i = 0
    s = p.strip()
    while i < len(s):
        if s[i] == "S" and i + 1 < len(s) and s[i + 1] == "(":
            end = s.index(")", i)
            parts.append(s[i : end + 1])
            i = end + 1
        elif s[i] == "P" and i + 1 < len(s) and s[i + 1] == "(":
            end = s.index(")", i)
            parts.append(s[i : end + 1])
            i = end + 1
        elif s[i] == "R":
            parts.append("R")
            i += 1
        else:
            i += 1
    return parts


def _classify_dim_transition(src_dim, dst_dim):
    """Classify what happens on a single mesh dim from src to dst.

    Returns (collective_name, is_free) where collective_name is None
    if no communication happens.
    """
    if src_dim == dst_dim:
        return None, True
    if src_dim == "R" and dst_dim.startswith("S("):
        return None, True  # local slicing
    if src_dim.startswith("S(") and dst_dim == "R":
        return "AllGather", False
    if src_dim.startswith("P(") and dst_dim == "R":
        return "AllReduce", False
    if src_dim.startswith("P(") and dst_dim.startswith("S("):
        return "ReduceScatter", False
    if src_dim.startswith("S(") and dst_dim.startswith("S("):
        return "AllToAll", False
    return "Redistribute", False


def _infer_collective(src, dst):
    """Infer the collective(s) for a placement transition by analyzing
    each mesh dimension independently."""
    if not src or not dst or src == dst:
        return None
    src_dims = _parse_placement_dims(src)
    dst_dims = _parse_placement_dims(dst)
    if len(src_dims) != len(dst_dims) or not src_dims:
        # Fall back to the simple heuristic for unparseable placements
        if "P(sum)" in src and "S(" in dst:
            return "ReduceScatter"
        if "P(sum)" in src and "R" in dst:
            return "AllReduce"
        if "S(" in src and "R" in dst and "P" not in dst:
            return "AllGather"
        return "Redistribute"

    collectives = []
    for sd, dd in zip(src_dims, dst_dims):
        coll, _ = _classify_dim_transition(sd, dd)
        if coll is not None:
            collectives.append(coll)

    if not collectives:
        return None  # all transitions are free (no comm)
    if len(collectives) == 1:
        return collectives[0]
    return "+".join(collectives)


def _node_total_comm(n):
    return sum(inp.get("comm_cost", 0) for inp in n.get("inputs", []))


def _node_total_transition(n):
    return n.get("transition_cost", 0) or sum(
        inp.get("transition_cost", 0) for inp in n.get("inputs", [])
    )


def _node_total_cost(n):
    return (
        _node_total_comm(n)
        + n.get("compute_cost", 0)
        + _node_total_transition(n)
    )


def _fmt_us(us):
    """Format microseconds as a human-readable string."""
    if us >= 1000:
        return f"{us / 1000:.1f}ms"
    return f"{us:.0f}\u00b5s"


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
    transition = sum(n["_total_transition"] for n in all_n)
    # Pick the dominant strategy by cost (not count): the strategy with the
    # most expensive nodes wins, so a layer with one massive ReduceScatter
    # doesn't appear as "REPLICATED" just because most nodes are replicated.
    cost_by_strat = {}
    for n in all_n:
        c = n["_total_comm"] + n.get("compute_cost", 0) + n["_total_transition"]
        cost_by_strat[n["_strategy"]] = cost_by_strat.get(n["_strategy"], 0) + c
    if cost_by_strat and max(cost_by_strat.values()) > 0:
        dominant = max(cost_by_strat, key=cost_by_strat.get)
    elif all_n:
        # Fall back to count when all costs are zero
        strategies = [n["_strategy"] for n in all_n]
        dominant = max(set(strategies), key=strategies.count)
    else:
        dominant = "unknown"
    all_colls = []
    for n in all_n:
        all_colls.extend(n["_collectives"])
    return {
        "num_nodes": len(all_n),
        "comm_cost": comm,
        "compute_cost": compute,
        "transition_cost": transition,
        "dominant": dominant,
        "color": _STRATEGY_COLORS.get(dominant, "#D1D5DB"),
        "bg": _STRATEGY_BGS.get(dominant, "#F9FAFB"),
        "collectives": all_colls,
    }


def _summarize_collectives(collectives):
    grouped = {}
    for _, ctype, _, _, ccost in collectives:
        stats = grouped.setdefault(ctype, {"count": 0, "cost": 0.0})
        stats["count"] += 1
        stats["cost"] += ccost
    return sorted(grouped.items(), key=lambda item: (-item[1]["cost"], item[0]))


def _render_tree_block(tree_node, cluster_counts, mesh, depth=0):
    """Recursively render a module tree node as nested HTML blocks."""
    stats = _tree_node_stats(tree_node)
    if stats["num_nodes"] == 0:
        return ""

    indent = depth * 20
    coll_summary = _summarize_collectives(stats["collectives"])
    coll_html = ""
    for ctype, cstats in coll_summary[:4]:
        coll_html += (
            f'<span class="collective-badge">{_esc(ctype)} &times;{cstats["count"]} '
            f'&middot; {_fmt_us(cstats["cost"])}</span> '
        )
    if len(coll_summary) > 4:
        coll_html += (
            f'<span class="collective-badge">+{len(coll_summary) - 4} more types</span>'
        )

    # Detail table for own_nodes (nodes at this exact level)
    detail_rows = ""
    for n in sorted(tree_node.own_nodes, key=_node_total_cost, reverse=True):
        comm = n["_total_comm"]
        compute = n.get("compute_cost", 0)
        transition = n["_total_transition"]
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
            f'<td class="{cost_class}" style="font-family:monospace">{_fmt_us(comm)}</td>'
            f'<td style="font-family:monospace">{_fmt_us(compute)}</td>'
            f'<td style="font-family:monospace">{_fmt_us(transition)}</td>'
            f'</tr>'
        )

        # Sub-rows for input edges with redistributions
        if comm > 0 or transition > 0:
            for inp in n.get("inputs", []):
                inp_comm = inp.get("comm_cost", 0)
                inp_trans = inp.get("transition_cost", 0)
                if inp_comm == 0 and inp_trans == 0:
                    continue
                src_p = inp.get("src_placement", "")
                dst_p = inp.get("dst_placement", "")
                coll = _infer_collective(src_p, dst_p)
                coll_label = f'{coll}: ' if coll else ''
                detail_rows += (
                    f'<tr class="{row_class} arch-input-edge" data-phase="{phase}">'
                    f'<td style="font-family:monospace;padding-left:20px;color:#94a3b8">'
                    f'&larr; {_esc(inp["name"])}</td>'
                    f'<td colspan="3" style="color:#64748b;font-size:11px">'
                    f'{_esc(coll_label)}{_esc(src_p)} &rarr; {_esc(dst_p)}</td>'
                    f'<td style="font-family:monospace;color:#DC2626;font-size:11px">{_fmt_us(inp_comm) if inp_comm > 0 else ""}</td>'
                    f'<td></td>'
                    f'<td style="font-family:monospace;color:#F59E0B;font-size:11px">{_fmt_us(inp_trans) if inp_trans > 0 else ""}</td>'
                    f'</tr>'
                )

    has_children = bool(tree_node.children)
    has_detail = bool(detail_rows)

    # Compute which phases this block contains
    all_phases = set(n.get("phase", "") for n in tree_node.all_nodes())
    phases_attr = " ".join(sorted(all_phases))
    total_cost = (
        stats["comm_cost"] + stats["compute_cost"] + stats["transition_cost"]
    )
    zero_cost_attr = "1" if total_cost == 0 else "0"
    zero_cost_class = " zero-cost-group" if total_cost == 0 else ""

    html = f'''
<div class="func-block{zero_cost_class}" style="background:{stats['bg']};border-color:{stats['color']};margin-left:{indent}px"
     data-phases="{phases_attr}"
     data-zero-cost="{zero_cost_attr}"
     onclick="event.stopPropagation(); this.classList.toggle('expanded')">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <div>
      <span class="func-name">{_esc(tree_node.name)}</span>
      <span class="strategy-badge" style="background:{stats['color']}">{stats['dominant'].upper()}</span>
      {f'<span style="font-size:11px;color:#94a3b8;margin-left:4px">&#9660;</span>' if has_children else ''}
    </div>
    <div style="text-align:right;font-size:12px;color:#64748b">
      {stats['num_nodes']} ops &middot; comm: {_fmt_us(stats['comm_cost'])} &middot; compute: {_fmt_us(stats['compute_cost'])} &middot; trans: {_fmt_us(stats['transition_cost'])}
    </div>
  </div>
  <div class="func-meta">{coll_html if coll_html else "No redistributions"}</div>
  <div class="func-details">'''

    if has_detail:
        html += f'''
    <table><tr><th>Node</th><th>Phase</th><th>Shape</th><th>Placement</th><th>Comm</th><th>Compute</th><th>Trans</th></tr>
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


def _build_param_tree_html(param_nodes, mesh):
    root = _ModuleTreeNode("")

    for n in param_nodes:
        raw_path = _strip_module_prefix(n.get("module_path", "") or n["name"])
        parts = raw_path.split(".")
        if len(parts) >= 2 and parts[-1] in {"weight", "bias"}:
            parent_parts = parts[:-2]
            leaf_name = ".".join(parts[-2:])
        else:
            parent_parts = parts[:-1]
            leaf_name = parts[-1]
        n["_param_leaf_name"] = leaf_name
        node = root
        for part in parent_parts:
            if part not in node.children:
                node.children[part] = _ModuleTreeNode(part)
            node = node.children[part]
        node.own_nodes.append(n)

    def subtree_signature(tree_node):
        own = tuple(
            sorted(
                (
                    n.get("_param_leaf_name", _strip_module_prefix(n.get("module_path", "") or n["name"]).split(".")[-1]),
                    tuple(n.get("shape", [])),
                    n.get("dtype", ""),
                    n.get("placement", ""),
                )
                for n in tree_node.own_nodes
            )
        )
        children = tuple(
            (name, subtree_signature(child))
            for name, child in sorted(tree_node.children.items())
        )
        return (own, children)

    def repeated_layer_info(tree_node):
        child_items = list(tree_node.children.items())
        if len(child_items) < 2 or not all(name.isdigit() for name, _ in child_items):
            return None
        layer_ids = sorted(int(name) for name, _ in child_items)
        if layer_ids != list(range(layer_ids[0], layer_ids[-1] + 1)):
            return None
        first_name = str(layer_ids[0])
        first_sig = subtree_signature(tree_node.children[first_name])
        for idx in layer_ids[1:]:
            if subtree_signature(tree_node.children[str(idx)]) != first_sig:
                return None
        return first_name, len(layer_ids)

    def inline_desc(n):
        placement = n.get("placement", "")
        if not placement or not mesh:
            return ""
        dim_names = mesh.get("dim_names") or [f"dim{i}" for i in range(len(mesh["shape"]))]
        mesh_shape = mesh["shape"]
        parts = _parse_placement_dims(placement)
        if not parts:
            return _esc(placement)
        desc = []
        for dname, msize, part in zip(dim_names, mesh_shape, parts):
            if part == "R":
                desc.append(f"{dname}: Replicate")
            elif part.startswith("S("):
                desc.append(f"{dname}: Shard({int(part[2:-1])})")
            elif part.startswith("P("):
                desc.append(f"{dname}: Partial ({msize}-way)")
        return _esc(", ".join(desc))

    def subtree_has_any_own_nodes(tree_node):
        if tree_node.own_nodes:
            return True
        return any(subtree_has_any_own_nodes(child) for child in tree_node.children.values())

    rows = []

    def render(tree_node, depth=0):
        if not subtree_has_any_own_nodes(tree_node):
            return
        child_items = list(tree_node.children.items())
        repeated = repeated_layer_info(tree_node)
        if tree_node.name:
            badge = ""
            if repeated is not None:
                _, count = repeated
                badge = f'<span class="cluster-badge">&times;{count} layers (identical)</span>'
            module_indent = depth * 20
            rows.append(
                f'<tr class="param-module-row">'
                f'<td class="param-name-cell" style="padding-left:{module_indent}px">'
                f'<span class="param-node-icon param-module-icon"></span>'
                f'<span class="param-module-label">{_esc(tree_node.name)}{badge}</span>'
                f"</td><td></td><td></td><td></td></tr>"
            )
        if repeated is not None:
            child_items = [(repeated[0], tree_node.children[repeated[0]])]
        child_depth = depth if not tree_node.name else depth + 1
        for _, child in child_items:
            render(child, child_depth)
        for own_idx, n in enumerate(tree_node.own_nodes):
            chip = _placement_chip_html(
                n.get("placement", ""),
                n["_bg"],
                n["_color"],
                n.get("shape"),
                n.get("dtype", ""),
                mesh,
            )
            leaf_name = n.get(
                "_param_leaf_name",
                _strip_module_prefix(n.get("module_path", "") or n["name"]).split(".")[-1],
            )
            shape = f'{n.get("dtype", "")}[{",".join(str(s) for s in n.get("shape", []))}]'
            leaf_indent = (depth + 1) * 20 if tree_node.name else depth * 20
            rows.append(
                f'<tr><td class="param-name-cell" style="padding-left:{leaf_indent}px">'
                f'<span class="param-node-icon param-leaf-icon"></span>'
                f'<span class="param-leaf-name">{_esc(leaf_name)}</span></td>'
                f'<td>{_esc(shape)}</td><td>{chip}</td><td>{inline_desc(n)}</td></tr>'
            )

    render(root, 0)

    return '''<div class="card"><div class="card-title">Parameters and Buffers</div>
<div class="table-scroll param-tree"><table><colgroup>
<col style="width:42%">
<col style="width:16%">
<col style="width:18%">
<col style="width:24%">
</colgroup><thead>
<tr><th>Module/Param</th><th>Shape</th><th>Placement</th><th>Description</th></tr>
</thead><tbody>''' + "".join(rows) + "</tbody></table></div></div>"


def _build_param_ascii_dump(param_nodes):
    root = _ModuleTreeNode("")

    for n in param_nodes:
        raw_path = _strip_module_prefix(n.get("module_path", "") or n["name"])
        parts = raw_path.split(".")
        if len(parts) >= 2 and parts[-1] in {"weight", "bias"}:
            parent_parts = parts[:-2]
            leaf_name = ".".join(parts[-2:])
        else:
            parent_parts = parts[:-1]
            leaf_name = parts[-1]
        node = root
        for part in parent_parts:
            if part not in node.children:
                node.children[part] = _ModuleTreeNode(part)
            node = node.children[part]
        node.own_nodes.append((leaf_name, n))

    lines = []

    def subtree_signature(tree_node):
        own = tuple(
            sorted(
                (
                    leaf_name,
                    tuple(n.get("shape", [])),
                    n.get("dtype", ""),
                    n.get("placement", ""),
                )
                for leaf_name, n in tree_node.own_nodes
            )
        )
        children = tuple(
            (name, subtree_signature(child))
            for name, child in sorted(tree_node.children.items())
        )
        return (own, children)

    def repeated_layer_run(items, start_idx):
        name, child, kind = items[start_idx]
        if kind != "module" or not name.isdigit():
            return None
        run = [(name, child)]
        expected = int(name) + 1
        j = start_idx + 1
        while j < len(items):
            next_name, next_child, next_kind = items[j]
            if next_kind != "module" or not next_name.isdigit() or int(next_name) != expected:
                break
            run.append((next_name, next_child))
            expected += 1
            j += 1
        if len(run) < 2:
            return None
        sig = subtree_signature(run[0][1])
        if any(subtree_signature(child) != sig for _, child in run[1:]):
            return None
        return run, j

    def render(tree_node, prefix=""):
        child_items = list(tree_node.children.items())
        own_items = list(tree_node.own_nodes)
        items = [(name, child, "module") for name, child in child_items] + [
            (leaf_name, n, "leaf") for leaf_name, n in own_items
        ]
        idx = 0
        while idx < len(items):
            repeated = repeated_layer_run(items, idx)
            if repeated is not None:
                run, next_idx = repeated
                start_name, child = run[0]
                end_name = run[-1][0]
                count = len(run)
                branch = "└── " if next_idx == len(items) else "├── "
                lines.append(
                    f"{prefix}{branch}({start_name}-{end_name}): {count} x"
                )
                next_prefix = prefix + ("    " if next_idx == len(items) else "│   ")
                render(child, next_prefix)
                idx = next_idx
                continue

            name, item, kind = items[idx]
            is_last = idx == len(items) - 1
            branch = "└── " if is_last else "├── "
            next_prefix = prefix + ("    " if is_last else "│   ")
            if kind == "module":
                lines.append(f"{prefix}{branch}{name}")
                render(item, next_prefix)
            else:
                shape = ",".join(str(s) for s in item.get("shape", []))
                dtype = item.get("dtype", "")
                placement = item.get("placement", "")
                lines.append(
                    f"{prefix}{branch}{name}  {dtype}[{shape}]  {placement}"
                )
            idx += 1

    render(root)
    return "\n".join(lines)


def _summarize_placement_costs(nodes):
    placement_costs = {}
    for n in nodes:
        placement = n.get("placement", "")
        if not placement:
            continue
        placement_costs[placement] = placement_costs.get(placement, 0.0) + _node_total_cost(n)
    return sorted(placement_costs.items(), key=lambda item: (-item[1], item[0]))


def _collective_summary_from_inputs(nodes, stage_names, all_nodes_by_name):
    grouped = {}
    for n in nodes:
        for inp in n.get("inputs", []):
            coll = _infer_collective(inp.get("src_placement"), inp.get("dst_placement"))
            if not coll:
                continue
            src_node = all_nodes_by_name.get(inp.get("name"))
            if src_node is not None and src_node in nodes:
                continue
            stats = grouped.setdefault(coll, {"count": 0, "cost": 0.0})
            stats["count"] += 1
            stats["cost"] += inp.get("comm_cost", 0.0)
    return sorted(grouped.items(), key=lambda item: (-item[1]["cost"], item[0]))


def _key_ops_for_stage(nodes):
    def short_op_name(op):
        text = str(op or "")
        if "." in text:
            parts = text.split(".")
            if len(parts) >= 2:
                return parts[-2]
        return text

    def op_label(n):
        mp = _strip_module_prefix(n.get("module_path", "") or "")
        op_name = short_op_name(n.get("op", ""))
        if mp:
            return f"{mp} [{op_name}]"
        return f'{n.get("name", "")} [{op_name}]'

    def input_output_summary(n):
        inputs_desc = []
        for idx, inp in enumerate(n.get("inputs", [])):
            src = inp.get("src_placement") or "?"
            dst = inp.get("dst_placement") or src
            coll = _infer_collective(src, dst)
            inputs_desc.append(
                {
                    "arg": f"arg{idx}",
                    "src": src,
                    "dst": dst,
                    "coll": coll,
                    "changed": bool(coll and src != dst),
                }
            )
        output_desc = n.get("placement", "") or "?"
        return inputs_desc, output_desc

    compute_ranked = sorted(
        [n for n in nodes if n.get("compute_cost", 0) > 0],
        key=lambda n: (
            n.get("compute_cost", 0),
            _node_total_cost(n),
            n["_total_comm"],
        ),
        reverse=True,
    )
    comm_ranked = sorted(
        [n for n in nodes if (n["_total_comm"] + n["_total_transition"]) > 0],
        key=lambda n: (
            n["_total_comm"] + n["_total_transition"],
            _node_total_cost(n),
            n.get("compute_cost", 0),
        ),
        reverse=True,
    )
    ranked = compute_ranked[:3] + comm_ranked[:3] + sorted(
        nodes,
        key=lambda n: (
            _node_total_cost(n),
            n.get("compute_cost", 0),
            n["_total_comm"],
            n["_total_transition"],
        ),
        reverse=True,
    )
    result = []
    seen = set()
    for n in ranked:
        label = op_label(n)
        if label in seen:
            continue
        seen.add(label)
        inputs_desc, output_desc = input_output_summary(n)
        result.append(
            {
                "label": label,
                "placement": n.get("placement", ""),
                "inputs": inputs_desc,
                "output_summary": output_desc,
                "comm": n["_total_comm"],
                "compute": n.get("compute_cost", 0),
                "transition": n["_total_transition"],
            }
        )
        if len(result) >= 6:
            break
    return result


def _redistributions_for_stage(nodes, stage_names, all_nodes_by_name):
    events = []
    for n in nodes:
        label = _strip_module_prefix(n.get("module_path", "") or n.get("name", ""))
        for inp in n.get("inputs", []):
            coll = _infer_collective(inp.get("src_placement"), inp.get("dst_placement"))
            comm_cost = inp.get("comm_cost", 0.0)
            trans_cost = inp.get("transition_cost", 0.0)
            if not coll and comm_cost <= 0 and trans_cost <= 0:
                continue
            src_node = all_nodes_by_name.get(inp.get("name"))
            if src_node is not None and src_node in nodes:
                continue
            events.append(
                {
                    "target": label,
                    "collective": coll or "Transition",
                    "src": inp.get("src_placement") or "?",
                    "dst": inp.get("dst_placement") or "?",
                    "comm": comm_cost,
                    "transition": trans_cost,
                    "total": comm_cost + trans_cost,
                }
            )
    events.sort(key=lambda e: (-e["total"], -e["comm"], e["target"], e["collective"]))
    dedup = []
    seen = set()
    for event in events:
        key = (event["target"], event["collective"], event["src"], event["dst"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(event)
    return dedup


def _stage_summary(nodes, all_nodes_by_name):
    if not nodes:
        return None
    stage_names = {n["name"] for n in nodes}
    total_comm = sum(n["_total_comm"] for n in nodes)
    total_compute = sum(n.get("compute_cost", 0) for n in nodes)
    total_transition = sum(n["_total_transition"] for n in nodes)
    placements = _summarize_placement_costs(nodes)
    collective_summary = _collective_summary_from_inputs(
        nodes, stage_names, all_nodes_by_name
    )
    return {
        "nodes": nodes,
        "num_nodes": len(nodes),
        "comm": total_comm,
        "compute": total_compute,
        "transition": total_transition,
        "total": total_comm + total_compute + total_transition,
        "placements": placements[:3],
        "collectives": collective_summary[:3],
        "key_ops": _key_ops_for_stage(nodes),
        "redistributions": _redistributions_for_stage(
            nodes, stage_names, all_nodes_by_name
        ),
    }


def _build_strategy_overview_html(non_param_nodes, layer_costs):
    import re as _re

    nodes = [
        n
        for n in non_param_nodes
        if n.get("op") != "placeholder" and n.get("op") != "output"
    ]
    all_nodes_by_name = {n["name"]: n for n in non_param_nodes}
    layer_indices = sorted(layer_costs.keys())
    repr_idx = layer_indices[0] if layer_indices else None
    layer_count = len(layer_indices)
    all_same = bool(layer_indices) and len(layer_indices) > 1 and all(
        abs(layer_costs[idx]["comm"] - layer_costs[repr_idx]["comm"]) < 0.01
        and abs(layer_costs[idx]["compute"] - layer_costs[repr_idx]["compute"]) < 0.01
        and abs(layer_costs[idx]["transition"] - layer_costs[repr_idx]["transition"]) < 0.01
        for idx in layer_indices[1:]
    )

    def stage_key(n):
        mp = _strip_module_prefix(n.get("module_path", "") or "")
        if repr_idx is not None:
            if mp.startswith(f"layers.{repr_idx}.attention."):
                return "attention"
            if mp.startswith(f"layers.{repr_idx}.feed_forward."):
                return "feed_forward"
            if mp == f"layers.{repr_idx}.attention_norm.weight" or mp.startswith(
                f"layers.{repr_idx}.attention_norm"
            ):
                return "attention_norm"
            if mp == f"layers.{repr_idx}.ffn_norm.weight" or mp.startswith(
                f"layers.{repr_idx}.ffn_norm"
            ):
                return "ffn_norm"
            if mp.startswith(f"layers.{repr_idx}."):
                return "layer_other"
        if mp.startswith("tok_embeddings"):
            return "tok_embeddings"
        if mp == "freqs_cis" or mp.startswith("freqs_cis"):
            return "freqs_cis"
        if mp == "norm.weight" or mp.startswith("norm."):
            return "norm"
        if mp.startswith("output"):
            return "output"
        return "other"

    grouped = {}
    for n in nodes:
        key = stage_key(n)
        grouped.setdefault(key, []).append(n)

    stage_labels = {
        "tok_embeddings": "Token Embeddings",
        "freqs_cis": "RoPE / freqs_cis",
        "attention_norm": "Attention Norm",
        "attention": "Attention",
        "ffn_norm": "FFN Norm",
        "feed_forward": "Feed-Forward",
        "layer_other": "Other Layer Ops",
        "norm": "Final Norm",
        "output": "Output Head",
        "other": "Other Ops",
    }

    ordered_pre = ["tok_embeddings", "freqs_cis"]
    ordered_layer = ["attention_norm", "attention", "ffn_norm", "feed_forward", "layer_other"]
    ordered_post = ["norm", "output", "other"]

    def stage_block_html(key):
        summary = _stage_summary(grouped.get(key, []), all_nodes_by_name)
        if summary is None:
            return ""
        total = summary["total"] or 1
        comm_pct = summary["comm"] / total * 100
        compute_pct = summary["compute"] / total * 100
        trans_pct = summary["transition"] / total * 100

        placement_html = ""
        for placement, _ in summary["placements"]:
            strat = _classify_placement(placement)
            placement_html += _placement_chip_html(
                placement,
                _STRATEGY_BGS.get(strat, "#F9FAFB"),
                _STRATEGY_COLORS.get(strat, "#D1D5DB"),
                None,
                "",
                None,
            )
            placement_html += " "
        if not placement_html:
            placement_html = '<span style="color:#94a3b8">No dominant placement</span>'

        coll_html = ""
        for ctype, stats in summary["collectives"]:
            coll_html += (
                f'<span class="collective-badge">{_esc(ctype)} &times;{stats["count"]} '
                f'&middot; {_fmt_us(stats["cost"])}</span> '
            )
        if not coll_html:
            coll_html = '<span style="color:#94a3b8">No boundary redistributions</span>'

        key_ops_html = ""
        for op in summary["key_ops"]:
            strat = _classify_placement(op["placement"])
            chip = _placement_chip_html(
                op["placement"],
                _STRATEGY_BGS.get(strat, "#F9FAFB"),
                _STRATEGY_COLORS.get(strat, "#D1D5DB"),
                None,
                "",
                None,
            )
            input_lines = ""
            for entry in op["inputs"]:
                src_strat = _classify_placement(entry["src"])
                src_chip = _placement_chip_html(
                    entry["src"],
                    _STRATEGY_BGS.get(src_strat, "#F9FAFB"),
                    _STRATEGY_COLORS.get(src_strat, "#D1D5DB"),
                    None,
                    "",
                    None,
                )
                if entry["changed"]:
                    dst_strat = _classify_placement(entry["dst"])
                    dst_chip = _placement_chip_html(
                        entry["dst"],
                        _STRATEGY_BGS.get(dst_strat, "#F9FAFB"),
                        _STRATEGY_COLORS.get(dst_strat, "#D1D5DB"),
                        None,
                        "",
                        None,
                    )
                    values_html = (
                        f'{src_chip}<span class="overview-placement-arrow">&rarr;</span>'
                        f'{dst_chip}<span class="overview-placement-coll">({ _esc(entry["coll"]) })</span>'
                    )
                else:
                    values_html = src_chip
                input_lines += (
                    '<div class="overview-placement-row">'
                    f'<div class="overview-placement-label">{_esc(entry["arg"])}</div>'
                    f'<div class="overview-placement-values">{values_html}</div>'
                    '</div>'
                )
            if not input_lines:
                input_lines = (
                    '<div class="overview-placement-row">'
                    '<div class="overview-placement-label">inputs</div>'
                    '<div class="overview-placement-values"><span style="color:#94a3b8">none</span></div>'
                    '</div>'
                )
            key_ops_html += (
                '<div class="overview-list-row overview-keyop-row">'
                f'<div class="overview-list-label">{_esc(op["label"])}</div>'
                '<div class="overview-placement-flow">'
                f'{input_lines}'
                '<div class="overview-placement-row">'
                '<div class="overview-placement-label">output</div>'
                f'<div class="overview-placement-values">{chip}</div>'
                '</div>'
                '</div>'
                f'<div class="overview-list-meta">comm {_fmt_us(op["comm"])} &middot; '
                f'compute {_fmt_us(op["compute"])} &middot; trans {_fmt_us(op["transition"])}</div>'
                '</div>'
            )
        if key_ops_html:
            key_ops_html = (
                '<div class="overview-mini-title">Key Operators</div>'
                f'<div class="overview-list overview-keyop-list">{key_ops_html}</div>'
            )

        redist_events = summary["redistributions"]
        redist_html = ""
        for event in redist_events[:5]:
            redist_html += (
                '<div class="overview-list-row">'
                f'<div class="overview-list-label">{_esc(event["target"])}</div>'
                f'<div class="overview-list-meta">{_esc(event["src"])} &rarr; {_esc(event["dst"])}</div>'
                f'<div class="overview-list-meta">{_esc(event["collective"])} &middot; '
                f'comm {_fmt_us(event["comm"])} &middot; trans {_fmt_us(event["transition"])}</div>'
                '</div>'
            )
        if redist_html:
            full_redist_html = ""
            for event in redist_events:
                full_redist_html += (
                    '<div class="overview-list-row">'
                    f'<div class="overview-list-label">{_esc(event["target"])}</div>'
                    f'<div class="overview-list-meta">{_esc(event["src"])} &rarr; {_esc(event["dst"])}</div>'
                    f'<div class="overview-list-meta">{_esc(event["collective"])} &middot; '
                    f'comm {_fmt_us(event["comm"])} &middot; trans {_fmt_us(event["transition"])}</div>'
                    '</div>'
                )
            redist_html = (
                '<div class="overview-mini-title">Top Redistributive Boundaries</div>'
                f'<div class="overview-list">{redist_html}</div>'
            )
            if len(redist_events) > 5:
                redist_html += (
                    f'<details class="overview-details"><summary>Show all {len(redist_events)} redistributions</summary>'
                    f'<div class="overview-details-body"><div class="overview-list">{full_redist_html}</div></div></details>'
                )

        return f'''
<div class="overview-stage-card">
  <div class="overview-stage-header">
    <div>
      <div class="overview-stage-name">{_esc(stage_labels[key])}</div>
      <div class="overview-stage-sub">{summary["num_nodes"]} ops &middot; total {_fmt_us(summary["total"])}</div>
    </div>
    <div class="overview-stage-costs">
      <span>Comm {_fmt_us(summary["comm"])}</span>
      <span>Compute {_fmt_us(summary["compute"])}</span>
      <span>Trans {_fmt_us(summary["transition"])}</span>
    </div>
  </div>
  <div class="cost-bar" style="margin-top:8px">
    <div class="cost-bar-segment" style="width:{comm_pct}%;background:#DC2626"></div>
    <div class="cost-bar-segment" style="width:{compute_pct}%;background:#10B981"></div>
    <div class="cost-bar-segment" style="width:{trans_pct}%;background:#F59E0B"></div>
  </div>
  <div class="overview-stage-meta"><strong>Layouts:</strong> {placement_html}</div>
  <div class="overview-stage-meta"><strong>Boundary collectives:</strong> {coll_html}</div>
  {key_ops_html}
  {redist_html}
</div>'''

    html = '<div class="card"><div class="card-title">Representative Execution Structure</div>'
    if repr_idx is not None:
        note = f"Showing representative layer {repr_idx}"
        if all_same:
            note += f" &middot; all {layer_count} layers share the same cost structure"
        elif layer_count > 1:
            note += f" &middot; {layer_count} total layers"
        html += f'<div class="overview-note">{_esc(note)}</div>'
    else:
        html += '<div class="overview-note">No repeated layer structure detected.</div>'

    pre_html = "".join(stage_block_html(key) for key in ordered_pre if grouped.get(key))
    if pre_html:
        html += '<div class="overview-section-title">Model Inputs and Setup</div>' + pre_html

    if repr_idx is not None:
        html += (
            f'<div class="overview-section-title">Representative Transformer Layer'
            f'{f" &times;{layer_count}" if layer_count > 1 else ""}</div>'
        )
        html += "".join(stage_block_html(key) for key in ordered_layer if grouped.get(key))

    post_html = "".join(stage_block_html(key) for key in ordered_post if grouped.get(key))
    if post_html:
        html += '<div class="overview-section-title">Final Stages</div>' + post_html

    html += '</div>'
    return html


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
        n["_total_transition"] = _node_total_transition(n)
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
    param_nodes = [
        n
        for n in compute_nodes
        if n.get("placeholder_kind") in {"param", "buffer"}
    ]
    architecture_nodes = [
        n
        for n in display_nodes
        if n.get("placeholder_kind") not in {"param", "buffer"}
    ]
    strategy_counts = {}
    for n in architecture_nodes:
        s = n["_strategy"]
        strategy_counts[s] = strategy_counts.get(s, 0) + 1

    # Build hierarchical module tree
    module_tree = _build_module_tree(architecture_nodes)

    # Flat list for cost-by-module view
    module_cost_list = _flatten_tree_for_costs(module_tree)
    module_cost_list.sort(
        key=lambda x: (
            x[1]["comm_cost"] + x[1]["compute_cost"] + x[1]["transition_cost"]
        ),
        reverse=True,
    )

    # Top costly nodes
    costly = sorted(architecture_nodes, key=_node_total_cost, reverse=True)
    top_costly = [n for n in costly[:20] if _node_total_cost(n) > 0]

    # Nodes with redistributions
    redist_nodes = [n for n in architecture_nodes if n["_collectives"]]

    # Mesh description
    mesh_shape = "×".join(str(d) for d in mesh["shape"])
    dim_names = mesh.get("dim_names")
    mesh_desc = f"{mesh_shape}"
    if dim_names:
        mesh_desc += f" ({', '.join(dim_names)})"

    # Per-layer cost breakdown (for repeated architectures)
    import re as _re

    layer_costs = {}  # layer_idx -> {"comm": float, "compute": float, "transition": float, "num_nodes": int}
    non_layer_cost = {
        "comm": 0.0,
        "compute": 0.0,
        "transition": 0.0,
        "num_nodes": 0,
        "name": "non-layer",
    }
    # Detailed per-layer collective breakdown
    layer_collectives = {}  # layer_idx -> list of (module_suffix, coll_type, cost)
    for n in architecture_nodes:
        mp = n.get("module_path", "")
        m = _re.search(r"layers\.(\d+)", mp)
        comm = n["_total_comm"]
        compute = n.get("compute_cost", 0)
        transition = n["_total_transition"]
        if m:
            idx = int(m.group(1))
            if idx not in layer_costs:
                layer_costs[idx] = {
                    "comm": 0.0,
                    "compute": 0.0,
                    "transition": 0.0,
                    "num_nodes": 0,
                }
                layer_collectives[idx] = []
            layer_costs[idx]["comm"] += comm
            layer_costs[idx]["compute"] += compute
            layer_costs[idx]["transition"] += transition
            layer_costs[idx]["num_nodes"] += 1
            # Collect per-edge collectives for this node (skip zero-cost)
            for coll_name, coll_type, cfrom, cto, ccost in n["_collectives"]:
                if ccost <= 0:
                    continue
                # Extract module suffix after layers.N.
                suffix = _re.sub(r".*layers\.\d+\.?", "", _strip_module_prefix(mp))
                layer_collectives[idx].append((suffix or mp, coll_type, ccost))
        else:
            non_layer_cost["comm"] += comm
            non_layer_cost["compute"] += compute
            non_layer_cost["transition"] += transition
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
.ap-viz .notation-box {{ margin: -6px 0 16px 0; border: 1px solid #e2e8f0; border-radius: 8px; background: white; }}
.ap-viz .notation-summary {{ list-style: none; cursor: pointer; padding: 10px 14px; font-size: 12px; font-weight: 600; color: #334155; }}
.ap-viz .notation-summary::-webkit-details-marker {{ display: none; }}
.ap-viz .notation-body {{ padding: 0 14px 12px 14px; display: grid; gap: 6px; font-size: 12px; color: #475569; }}
.ap-viz .notation-code {{ display: inline-block; font-family: monospace; background: #f8fafc; border: 1px solid #e2e8f0; padding: 1px 6px; border-radius: 4px; color: #1e293b; }}
.ap-viz .ascii-dump {{ margin-top: 14px; padding: 14px; background: #0f172a; color: #e2e8f0; border-radius: 8px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 11px; line-height: 1.45; overflow-x: auto; white-space: pre; }}
.ap-viz .overview-note {{ font-size: 12px; color: #64748b; margin-bottom: 14px; }}
.ap-viz .overview-section-title {{ font-size: 13px; font-weight: 700; color: #334155; margin: 18px 0 10px 0; }}
.ap-viz .overview-stage-card {{ border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; margin-bottom: 12px; background: #fff; }}
.ap-viz .overview-stage-header {{ display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }}
.ap-viz .overview-stage-name {{ font-size: 14px; font-weight: 700; color: #1e293b; }}
.ap-viz .overview-stage-sub {{ font-size: 12px; color: #64748b; margin-top: 2px; }}
.ap-viz .overview-stage-costs {{ display: flex; flex-wrap: wrap; gap: 10px; font-size: 12px; color: #475569; justify-content: flex-end; }}
.ap-viz .overview-stage-meta {{ font-size: 12px; color: #475569; margin-top: 10px; }}
.ap-viz .overview-mini-title {{ font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.4px; margin: 12px 0 6px 0; }}
.ap-viz .overview-list {{ display: grid; gap: 6px; }}
.ap-viz .overview-list-row {{ display: grid; grid-template-columns: minmax(260px, 1.5fr) minmax(120px, 0.9fr) auto; gap: 10px; align-items: start; font-size: 12px; color: #334155; }}
.ap-viz .overview-keyop-list {{ gap: 0; }}
.ap-viz .overview-keyop-row {{ padding: 8px 0; border-top: 1px solid #e2e8f0; }}
.ap-viz .overview-keyop-row:first-child {{ border-top: 0; padding-top: 0; }}
.ap-viz .overview-keyop-row:last-child {{ padding-bottom: 0; }}
.ap-viz .overview-list-label {{ font-family: monospace; overflow-wrap: anywhere; }}
.ap-viz .overview-list-meta {{ color: #64748b; font-size: 11px; }}
.ap-viz .overview-placement-cell {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.ap-viz .overview-placement-flow {{ display: grid; gap: 4px; font-size: 11px; color: #475569; }}
.ap-viz .overview-placement-row {{ display: grid; grid-template-columns: 44px 1fr; gap: 8px; align-items: center; }}
.ap-viz .overview-placement-label {{ font-family: monospace; font-size: 11px; color: #64748b; text-align: right; }}
.ap-viz .overview-placement-values {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }}
.ap-viz .overview-placement-arrow {{ color: #94a3b8; font-size: 11px; }}
.ap-viz .overview-placement-coll {{ color: #64748b; font-size: 11px; }}
.ap-viz .overview-details {{ margin-top: 10px; border: 1px solid #e2e8f0; border-radius: 8px; background: #f8fafc; }}
.ap-viz .overview-details > summary {{ cursor: pointer; padding: 8px 10px; font-size: 12px; font-weight: 600; color: #334155; }}
.ap-viz .overview-details-body {{ padding: 0 10px 10px 10px; }}
.ap-viz .param-tree table {{ width: 100%; }}
.ap-viz .param-tree td {{ vertical-align: top; }}
.ap-viz .param-module-row td {{ background: #f8fafc; font-weight: 600; color: #334155; }}
.ap-viz .param-tree th:first-child, .ap-viz .param-tree td:first-child {{ min-width: 420px; }}
.ap-viz .param-module-label {{ display: inline-flex; align-items: center; gap: 6px; }}
.ap-viz .param-name-cell {{ white-space: nowrap; }}
.ap-viz .param-leaf-name {{ font-family: monospace; font-size: 11px; color: #1e293b; }}
.ap-viz .param-node-icon {{ display: inline-block; vertical-align: middle; margin-right: 8px; }}
.ap-viz .param-module-icon {{ width: 10px; height: 10px; border-radius: 999px; background: #94a3b8; }}
.ap-viz .param-leaf-icon {{ width: 10px; height: 2px; background: #cbd5e1; border-radius: 999px; }}
.ap-viz tr.linked-row {{ opacity: 0.45; display: none; }}
.ap-viz tr.linked-row td {{ font-style: italic; }}
.ap-viz tr.arch-bwd {{ opacity: 0.6; }}
.ap-viz tr.arch-input-edge td {{ background: #f8fafc; border-bottom: 1px solid #f1f5f9; }}
.ap-viz .phase-badge {{ display: inline-block; padding: 1px 6px; border-radius: 8px; font-size: 10px; font-weight: 600; }}
.ap-viz .phase-fwd {{ color: #10B981; background: #ECFDF5; }}
.ap-viz .phase-bwd {{ color: #6B7280; background: #F1F5F9; }}
.ap-viz .zero-cost-group {{ opacity: 0.72; }}
.ap-viz .zero-cost-group:not(.expanded) > .func-details {{ display: none; }}
</style></head><body>
<div class="ap-viz">
<div class="container">
<h1>AutoParallel Strategy Visualizer</h1>
<div class="subtitle">Mesh: {_esc(mesh_desc)} &middot; {len(compute_nodes)} nodes &middot; {len(redist_nodes)} redistributions'''

    if has_clusters:
        n_clusters = len(cluster_counts)
        max_instances = max(cluster_counts.values()) + 1 if cluster_counts else 1
        html += (
            f" &middot; {n_clusters} repeated clusters "
            f"(up to {max_instances} layer instances)"
        )

    html += f'''</div>

<div class="stats-row">
  <div class="stat-card">
    <div class="stat-label">Total Cost</div>
    <div class="stat-value">{_fmt_us(summary["total"])}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Communication</div>
    <div class="stat-value" style="color:#DC2626">{_fmt_us(summary["comm"])}</div>
    <div class="stat-detail">{comm_pct:.0f}% of total</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Compute</div>
    <div class="stat-value" style="color:#10B981">{_fmt_us(summary["compute"])}</div>
    <div class="stat-detail">{compute_pct:.0f}% of total</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Transitions</div>
    <div class="stat-value" style="color:#F59E0B">{_fmt_us(summary["transition"])}</div>
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

    html += '''</div>
<details class="notation-box">
  <summary class="notation-summary">Placement notation</summary>
  <div class="notation-body">
    <div><span class="notation-code">R</span> replicate across that mesh dimension</div>
    <div><span class="notation-code">S(i)</span> shard tensor dimension <strong>i</strong> across that mesh dimension</div>
    <div><span class="notation-code">P(sum)</span> partial value with a pending reduction</div>
    <div><span class="notation-code">S(0)S(1)</span> means the tensor is sharded on two mesh dimensions</div>
  </div>
</details>'''

    # Controls bar
    html += '''
<div class="controls">
  <div class="control-group">
    <span class="control-label">Phase</span>
    <button class="filter-btn active" data-filter-group="phase" data-filter-value="all" onclick="setFilterState('phase', 'all', this)">All</button>
    <button class="filter-btn" data-filter-group="phase" data-filter-value="forward" onclick="setFilterState('phase', 'forward', this)">Forward</button>
    <button class="filter-btn" data-filter-group="phase" data-filter-value="backward" onclick="setFilterState('phase', 'backward', this)">Backward</button>
  </div>'''

    if has_clusters:
        html += '''
  <div class="control-group">
    <span class="control-label">Detail View</span>
    <button class="filter-btn active" id="btn-cluster-collapse" onclick="toggleRepresentativeLayers(this)">Representative Layer</button>
  </div>'''

    html += '</div>'

    # Tabs
    html += '''
<div class="tabs">
  <div class="tab" onclick="switchTab('params', this)">Parameters</div>
  <div class="tab" onclick="switchTab('overview', this)">Execution Overview</div>
  <div class="tab active" onclick="switchTab('arch', this)">Architecture</div>
  <div class="tab" onclick="switchTab('cost', this)">Cost Breakdown</div>
  <div class="tab" onclick="switchTab('detail', this)">All Nodes</div>
</div>'''

    # ===== PARAMETERS TAB =====
    html += '''
<div data-tab="params" class="tab-content">'''
    if param_nodes:
        html += (
            '<div class="card"><div class="card-title">Parameters and Buffers</div>'
            f'<pre class="ascii-dump">{_esc(_build_param_ascii_dump(param_nodes))}</pre>'
            '</div>'
        )
    else:
        html += '<div class="card"><div class="card-title">Parameters and Buffers</div><div style="font-size:12px;color:#64748b">No parameter or buffer placeholder nodes available.</div></div>'
    html += '</div>'

    # ===== EXECUTION OVERVIEW TAB =====
    html += '''
<div data-tab="overview" class="tab-content">'''
    html += _build_strategy_overview_html(architecture_nodes, layer_costs)
    html += '</div>'

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
  <div class="cost-bar-label"><span>Cost Breakdown</span></div>
  <div class="cost-bar" style="height:32px">
    <div class="cost-bar-segment" style="width:{comm_pct}%;background:#DC2626">Comm {comm_pct:.0f}%</div>
    <div class="cost-bar-segment" style="width:{compute_pct}%;background:#10B981">Compute {compute_pct:.0f}%</div>
    <div class="cost-bar-segment" style="width:{trans_pct}%;background:#F59E0B">Trans {trans_pct:.0f}%</div>
  </div>
</div></div>'''

    # Per-layer cost breakdown (only if layers exist)
    if layer_costs:
        # Check if all layers have the same cost (clustered)
        layer_vals = list(layer_costs.values())
        all_same = len(layer_vals) > 1 and all(
            abs(lc["comm"] - layer_vals[0]["comm"]) < 0.01
            and abs(lc["compute"] - layer_vals[0]["compute"]) < 0.01
            and abs(lc["transition"] - layer_vals[0]["transition"]) < 0.01
            for lc in layer_vals[1:]
        )

        # Use a representative layer for the detailed breakdown
        repr_idx = sorted(layer_costs.keys())[0]
        repr_costs = layer_costs[repr_idx]
        repr_colls = layer_collectives.get(repr_idx, [])

        # Aggregate collectives by (module, type)
        coll_groups = {}
        for suffix, coll_type, ccost in repr_colls:
            key = (coll_type, suffix)
            if key not in coll_groups:
                coll_groups[key] = {"cost": 0.0, "count": 0}
            coll_groups[key]["cost"] += ccost
            coll_groups[key]["count"] += 1

        # Sort by cost descending
        sorted_colls = sorted(coll_groups.items(), key=lambda x: -x[1]["cost"])

        layer_title = "Cost by Layer"
        if all_same and has_clusters:
            n_layers = len(layer_vals)
            layer_title += f' (all {n_layers} layers identical, showing layer {repr_idx})'

        html += f'''
<div class="card"><div class="card-title">{_esc(layer_title)}</div>'''

        # Compute bar: full width reference
        bar_max = (
            repr_costs["compute"]
            + repr_costs["comm"]
            + repr_costs["transition"]
        ) or 1

        html += f'''<div class="cost-bar-container">
  <div class="cost-bar-label"><span style="font-weight:600">Compute</span><span style="color:#64748b">{_fmt_us(repr_costs["compute"])}</span></div>
  <div class="cost-bar">
    <div class="cost-bar-segment" style="width:{repr_costs["compute"] / bar_max * 100}%;background:#10B981"></div>
  </div>
</div>'''

        if repr_costs["transition"] > 0:
            html += f'''<div class="cost-bar-container">
  <div class="cost-bar-label"><span style="font-weight:600">Transitions</span><span style="color:#64748b">{_fmt_us(repr_costs["transition"])}</span></div>
  <div class="cost-bar">
    <div class="cost-bar-segment" style="width:{repr_costs["transition"] / bar_max * 100}%;background:#F59E0B"></div>
  </div>
</div>'''

        # Individual collective bars, scaled relative to compute
        for (coll_type, suffix), info in sorted_colls:
            if info["cost"] == 0:
                continue
            w = min(info["cost"] / bar_max * 100, 100)
            label = f'{coll_type}'
            if suffix:
                label += f' ({suffix})'
            html += f'''<div class="cost-bar-container">
  <div class="cost-bar-label"><span>{_esc(label)}</span><span style="color:#64748b">{_fmt_us(info["cost"])}</span></div>
  <div class="cost-bar">
    <div class="cost-bar-segment" style="width:{w}%;background:#DC2626"></div>
  </div>
</div>'''

        # Summary line
        total_comm = repr_costs["comm"]
        total_compute = repr_costs["compute"]
        total_transition = repr_costs["transition"]
        total = total_comm + total_compute + total_transition
        if total > 0:
            comm_frac = total_comm / total * 100
            n_layers = len(layer_costs)
            html += (
                f'<p style="font-size:12px;color:#64748b;margin-top:8px">'
                f'Per layer: compute={_fmt_us(total_compute)}, '
                f'comm={_fmt_us(total_comm)}, trans={_fmt_us(total_transition)} '
                f'({comm_frac:.0f}% communication)'
            )
            if n_layers > 1:
                html += f' &middot; {n_layers} layers total: {_fmt_us(total * n_layers)}'
            html += '</p>'

        # Non-layer ops
        if non_layer_cost["num_nodes"] > 0:
            nl_total = (
                non_layer_cost["comm"]
                + non_layer_cost["compute"]
                + non_layer_cost["transition"]
            )
            if nl_total > 0:
                html += f'''<div class="cost-bar-container" style="margin-top:12px">
  <div class="cost-bar-label"><span style="font-weight:600">Non-layer ops</span><span style="color:#64748b">{non_layer_cost["num_nodes"]} ops &middot; {_fmt_us(nl_total)}</span></div>
  <div class="cost-bar">
    <div class="cost-bar-segment" style="width:{min(non_layer_cost["comm"] / bar_max * 100, 100)}%;background:#DC2626"></div>
    <div class="cost-bar-segment" style="width:{min(non_layer_cost["compute"] / bar_max * 100, 100)}%;background:#10B981"></div>
    <div class="cost-bar-segment" style="width:{min(non_layer_cost["transition"] / bar_max * 100, 100)}%;background:#F59E0B"></div>
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
            trans_w = n["_total_transition"] / top_max * 100
            src = n.get("source")
            src_label = f'{src["func"]}: {src["code"]}' if src else n.get("op", "")
            html += f'''<div class="cost-bar-container">
  <div class="cost-bar-label"><span style="font-family:monospace">{_esc(n["name"])}</span><span style="color:#64748b">{_esc(src_label)} &middot; total: {_fmt_us(total)}</span></div>
  <div class="cost-bar">
    <div class="cost-bar-segment" style="width:{comm_w}%;background:#DC2626" title="comm: {_fmt_us(n['_total_comm'])}"></div>
    <div class="cost-bar-segment" style="width:{comp_w}%;background:#10B981" title="compute: {_fmt_us(n.get('compute_cost', 0))}"></div>
    <div class="cost-bar-segment" style="width:{trans_w}%;background:#F59E0B" title="trans: {_fmt_us(n['_total_transition'])}"></div>
  </div>
</div>'''

    # Cost by module
    html += '</div><div class="card"><div class="card-title">Cost by Module</div>'
    func_max = (
        max(
            (
                s["comm_cost"] + s["compute_cost"] + s["transition_cost"]
                for _, s in module_cost_list
            )
        )
        if module_cost_list
        else 1
    )
    for mname, stats in module_cost_list:
        total = (
            stats["comm_cost"] + stats["compute_cost"] + stats["transition_cost"]
        )
        if total == 0:
            continue
        comm_w = stats["comm_cost"] / func_max * 100
        comp_w = stats["compute_cost"] / func_max * 100
        trans_w = stats["transition_cost"] / func_max * 100
        html += f'''<div class="cost-bar-container">
  <div class="cost-bar-label"><span style="font-weight:600">{_esc(mname)}</span><span style="color:#64748b">{stats["num_nodes"]} ops &middot; total: {_fmt_us(total)}</span></div>
  <div class="cost-bar">
    <div class="cost-bar-segment" style="width:{comm_w}%;background:#DC2626"></div>
    <div class="cost-bar-segment" style="width:{comp_w}%;background:#10B981"></div>
    <div class="cost-bar-segment" style="width:{trans_w}%;background:#F59E0B"></div>
  </div>
</div>'''

    html += '</div></div>'

    # ===== ALL NODES TAB =====
    html += '''
<div data-tab="detail" class="tab-content">
<div class="card">
<div class="card-title">All Nodes</div>
<div class="filter-bar">
  <input type="text" class="search-box" placeholder="Search nodes..." oninput="setSearch(this)">
  <button class="filter-btn active" data-filter-group="node-type" data-filter-value="all" onclick="setFilterState('nodeType', 'all', this)">All</button>
  <button class="filter-btn" data-filter-group="node-type" data-filter-value="param" onclick="setFilterState('nodeType', 'param', this)">Parameters</button>
  <button class="filter-btn" data-filter-group="node-type" data-filter-value="input" onclick="setFilterState('nodeType', 'input', this)">Inputs</button>
  <button class="filter-btn" data-filter-group="node-type" data-filter-value="tangent" onclick="setFilterState('nodeType', 'tangent', this)">Tangents</button>
  <button class="filter-btn" data-filter-group="node-type" data-filter-value="forward" onclick="setFilterState('nodeType', 'forward', this)">Forward</button>
  <button class="filter-btn" data-filter-group="node-type" data-filter-value="backward" onclick="setFilterState('nodeType', 'backward', this)">Backward</button>
  <button class="filter-btn" data-filter-group="node-type" data-filter-value="redist" onclick="setFilterState('nodeType', 'redist', this)">Has Redistribution</button>
</div>
<div class="table-scroll">
<table data-role="node-table"><thead>
<tr><th class="sortable" onclick="sortTable(0, 'str', this)">Name</th><th class="sortable" onclick="sortTable(1, 'str', this)">Op</th><th class="sortable" onclick="sortTable(2, 'str', this)">Phase</th><th>Shape</th><th class="sortable" onclick="sortTable(4, 'str', this)">Placement</th><th class="sortable" onclick="sortTable(5, 'num', this)">Comm</th><th class="sortable" onclick="sortTable(6, 'num', this)">Compute</th><th class="sortable" onclick="sortTable(7, 'num', this)">Trans</th><th>Redistribution</th><th class="sortable" onclick="sortTable(9, 'str', this)">Module</th></tr>
</thead><tbody>'''

    all_display = compute_nodes
    for n in all_display:
        comm = n["_total_comm"]
        compute = n.get("compute_cost", 0)
        transition = n["_total_transition"]
        cost_class = "cost-high" if comm > 100 else "cost-med" if comm > 0 else "cost-low"
        shape_str = ",".join(str(s) for s in n.get("shape", []))
        dtype = n.get("dtype", "")
        placement = n.get("placement", "")
        phase = n.get("phase", "")
        op_name = n.get("op", "")
        module = n.get("module_path", "")
        placeholder_kind = n.get("placeholder_kind", "")

        # Collective badges
        coll_html = ""
        for _, ctype, cfrom, cto, ccost in n["_collectives"]:
            coll_html += f'<span class="collective-badge">{_esc(ctype)}: {_esc(cfrom)}&rarr;{_esc(cto)} ({_fmt_us(ccost)})</span> '

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
            f'data-linked="{1 if is_linked else 0}" '
            f'data-placeholder-kind="{_esc(placeholder_kind)}"'
        )

        chip = _placement_chip_html(placement, n['_bg'], n['_color'], n.get("shape"), dtype, mesh)

        html += f'''<tr class="{row_class}" {data_attrs}>
  <td style="font-family:monospace;font-size:11px">{_esc(n["name"])}{cluster_html}</td>
  <td style="font-size:11px">{_esc(op_name)}</td>
  <td style="font-size:11px">{phase}</td>
  <td style="font-size:11px">{_esc(dtype)}[{shape_str}]</td>
  <td>{chip}</td>
  <td class="{cost_class}" style="font-family:monospace" data-sort-value="{comm}">{_fmt_us(comm)}</td>
  <td style="font-family:monospace" data-sort-value="{compute}">{_fmt_us(compute)}</td>
  <td style="font-family:monospace" data-sort-value="{transition}">{_fmt_us(transition)}</td>
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

var filterState = {
  phase: 'all',
  nodeType: 'all',
  searchText: '',
  representativeLayersOnly: true,
};

function setFilterState(key, value, btn) {
  var r = _root(btn);
  filterState[key] = value;
  var group = btn.dataset.filterGroup;
  if (group) {
    r.querySelectorAll('[data-filter-group="' + group + '"]').forEach(function(b) {
      b.classList.toggle('active', b === btn);
    });
  }
  applyFilters(r);
}

function setSearch(el) {
  filterState.searchText = el.value.toLowerCase();
  applyFilters(_root(el));
}

function toggleRepresentativeLayers(btn) {
  filterState.representativeLayersOnly = !filterState.representativeLayersOnly;
  btn.textContent = filterState.representativeLayersOnly ? 'Representative Layer' : 'All Repeated Layers';
  btn.classList.toggle('active', filterState.representativeLayersOnly);
  applyFilters(_root(btn));
}

function _matchesNodeType(row) {
  var type = filterState.nodeType;
  if (type === 'all') return true;
  if (type === 'forward') return row.dataset.phase === 'forward';
  if (type === 'backward') return row.dataset.phase === 'backward';
  if (type === 'redist') return row.dataset.redist === '1';
  return row.dataset.placeholderKind === type;
}

function applyFilters(r) {
  r.querySelectorAll('.node-row').forEach(function(row) {
    var show = true;
    if (filterState.representativeLayersOnly && row.dataset.linked === '1') show = false;
    if (show && filterState.phase !== 'all') show = row.dataset.phase === filterState.phase;
    if (show) show = _matchesNodeType(row);
    if (show && filterState.searchText !== '') {
      show = row.textContent.toLowerCase().indexOf(filterState.searchText) >= 0;
    }
    row.style.display = show ? '' : 'none';
  });

  r.querySelectorAll('.arch-row').forEach(function(row) {
    var show = filterState.phase === 'all' || row.dataset.phase === filterState.phase;
    row.style.display = show ? '' : 'none';
  });

  r.querySelectorAll('.func-block').forEach(function(block) {
    var show = true;
    if (filterState.phase !== 'all') {
      var phases = block.dataset.phases || '';
      show = phases.indexOf(filterState.phase) >= 0;
    }
    block.style.display = show ? '' : 'none';
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
    var aCell = a.cells[colIdx];
    var bCell = b.cells[colIdx];
    var aVal = aCell.dataset.sortValue || aCell.textContent.trim();
    var bVal = bCell.dataset.sortValue || bCell.textContent.trim();
    if (type === 'num') {
      return (parseFloat(aVal) - parseFloat(bVal)) * dir;
    }
    return aVal.localeCompare(bVal) * dir;
  });

  rows.forEach(function(row) { tbody.appendChild(row); });
}

document.querySelectorAll('.ap-viz').forEach(function(root) {
  applyFilters(root);
});
</script>
</div></div></body></html>'''

    return html
