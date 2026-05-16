# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
HTML visualizer that consumes the JSON dict from export_json.py.

Supports cluster collapsing for multi-layer models, phase filtering,
redistribution highlighting, and hierarchical module tree view.
"""

import json


def _placement_style(placement):
    """Return (bg_color, text_color) for a placement chip."""
    if not placement:
        return "#F9FAFB", "#D1D5DB"
    p = placement.strip()
    if "P(" in p:
        return "#FFFBEB", "#F59E0B"  # amber — partial
    if "S(" in p:
        return "#EFF6FF", "#3B82F6"  # blue — sharded
    return "#F9FAFB", "#6B7280"  # gray — replicated


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


_PLACEMENT_CATEGORIES = [
    ("sharded", "#3B82F6", "#EFF6FF"),
    ("replicated", "#6B7280", "#F9FAFB"),
    ("partial", "#F59E0B", "#FFFBEB"),
]


def _placement_category(placement):
    """Classify a placement string into a coarse category."""
    if not placement:
        return "replicated"
    p = placement.strip()
    if "P(" in p:
        return "partial"
    if "S(" in p:
        return "sharded"
    return "replicated"


def _placement_distribution(nodes):
    """Compute coarse placement distribution with exact-string details."""
    counts = {"sharded": 0, "replicated": 0, "partial": 0}
    details = {"sharded": {}, "replicated": {}, "partial": {}}
    for n in nodes:
        p = n.get("placement", "")
        cat = _placement_category(p)
        counts[cat] += 1
        key = p or "(none)"
        details[cat][key] = details[cat].get(key, 0) + 1
    total = sum(counts.values())
    dominant = max(counts, key=counts.get) if total else "replicated"
    color_map = {c[0]: c[1] for c in _PLACEMENT_CATEGORIES}
    return {
        "dist_counts": counts,
        "dist_details": details,
        "dist_total": total,
        "dominant_color": color_map[dominant],
    }


def _tree_node_stats(tree_node):
    """Compute summary stats for a tree node (all descendants)."""
    all_n = tree_node.all_nodes()
    comm = sum(n["_total_comm"] for n in all_n)
    compute = sum(n.get("compute_cost", 0) for n in all_n)
    transition = sum(n["_total_transition"] for n in all_n)
    all_colls = []
    for n in all_n:
        all_colls.extend(n["_collectives"])
    result = {
        "num_nodes": len(all_n),
        "comm_cost": comm,
        "compute_cost": compute,
        "transition_cost": transition,
        "collectives": all_colls,
    }
    result.update(_placement_distribution(all_n))
    return result


def _summarize_collectives(collectives):
    grouped = {}
    for _, ctype, _, _, ccost in collectives:
        stats = grouped.setdefault(ctype, {"count": 0, "cost": 0.0})
        stats["count"] += 1
        stats["cost"] += ccost
    return sorted(grouped.items(), key=lambda item: (-item[1]["cost"], item[0]))


def _render_tree_block(tree_node, cluster_counts, mesh, depth=0, total_cost=1, comm_threshold=0):
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
    block_cost = (
        stats["comm_cost"] + stats["compute_cost"] + stats["transition_cost"]
    )
    zero_cost_attr = "1" if block_cost == 0 else "0"
    zero_cost_class = " zero-cost-group" if block_cost == 0 else ""
    hotspot_class = " cost-hotspot" if stats["comm_cost"] >= comm_threshold > 0 else ""

    # Build placement distribution bar
    dist_bar = ""
    if stats["dist_total"] > 0:
        segments = []
        for cat, color, bg in _PLACEMENT_CATEGORIES:
            cnt = stats["dist_counts"][cat]
            if cnt == 0:
                continue
            pct = cnt * 100.0 / stats["dist_total"]
            exact = stats["dist_details"][cat]
            tip_parts = [f"{p} \u00d7{c}" for p, c in sorted(exact.items(), key=lambda x: -x[1])]
            tip = f"{cat.title()}: {', '.join(tip_parts)}"
            segments.append(
                f'<div class="placement-bar-seg" style="width:{pct:.1f}%;background:{color}" title="{_esc(tip)}"></div>'
            )
        dist_bar = f'<div class="placement-bar">{"".join(segments)}</div>'

    # Build cost distribution bar (comm / compute / transition)
    cost_bar = ""
    if block_cost > 0:
        comm_w = stats["comm_cost"] * 100 / block_cost
        comp_w = stats["compute_cost"] * 100 / block_cost
        trans_w = stats["transition_cost"] * 100 / block_cost
        cost_bar = (
            '<div class="cost-bar" style="height:6px;margin-top:4px">'
            f'<div class="cost-bar-segment" style="width:{comm_w:.1f}%;background:#DC2626" title="comm: {_fmt_us(stats["comm_cost"])}"></div>'
            f'<div class="cost-bar-segment" style="width:{comp_w:.1f}%;background:#10B981" title="compute: {_fmt_us(stats["compute_cost"])}"></div>'
            f'<div class="cost-bar-segment" style="width:{trans_w:.1f}%;background:#F59E0B" title="trans: {_fmt_us(stats["transition_cost"])}"></div>'
            '</div>'
        )

    # Percentages relative to global total
    def _pct(v):
        return f" ({v * 100 / total_cost:.0f}%)" if total_cost > 0 else ""

    html = f'''
<div class="func-block{zero_cost_class}{hotspot_class}" style="margin-left:{indent}px;border-left-color:{stats['dominant_color']}"
     data-phases="{phases_attr}"
     data-zero-cost="{zero_cost_attr}"
     onclick="event.stopPropagation(); this.classList.toggle('expanded')">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <div>
      <span class="func-name">{_esc(tree_node.name)}</span>
      {f'<span style="font-size:11px;color:#94a3b8;margin-left:4px">&#9660;</span>' if has_children else ''}
    </div>
    <div style="text-align:right;font-size:12px;color:#64748b">
      {stats['num_nodes']} ops &middot; comm: {_fmt_us(stats['comm_cost'])}{_pct(stats['comm_cost'])} &middot; compute: {_fmt_us(stats['compute_cost'])}{_pct(stats['compute_cost'])} &middot; trans: {_fmt_us(stats['transition_cost'])}{_pct(stats['transition_cost'])}
    </div>
  </div>
  {dist_bar}
  {cost_bar}
  <div class="func-meta">{coll_html if coll_html else "No redistributions"}</div>
  <div class="func-details">'''

    if has_children:
        for child in tree_node.children.values():
            html += _render_tree_block(child, cluster_counts, mesh, depth + 1, total_cost, comm_threshold)

    if has_detail:
        if has_children:
            html += '<div style="margin-top:10px"></div>'
        html += f'''
    <table><tr><th>Node</th><th>Phase</th><th>Shape</th><th>Placement</th><th>Comm</th><th>Compute</th><th>Trans</th></tr>
    {detail_rows}</table>'''

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


def _treemap_data(tree_node, prefix=""):
    """Flatten module tree into leaf-level items for a flat treemap."""
    full_name = f"{prefix}.{tree_node.name}" if prefix else (tree_node.name or "root")
    items = []
    if tree_node.children:
        for child in tree_node.children.values():
            items.extend(_treemap_data(child, full_name))
        # Own nodes at this level (not in any child)
        if tree_node.own_nodes:
            own_comm = sum(n["_total_comm"] for n in tree_node.own_nodes)
            own_compute = sum(n.get("compute_cost", 0) for n in tree_node.own_nodes)
            own_trans = sum(n["_total_transition"] for n in tree_node.own_nodes)
            total = own_comm + own_compute + own_trans
            if total > 0:
                items.append({
                    "name": full_name, "comm": own_comm,
                    "compute": own_compute, "transition": own_trans,
                })
    else:
        stats = _tree_node_stats(tree_node)
        total = stats["comm_cost"] + stats["compute_cost"] + stats["transition_cost"]
        if total > 0:
            items.append({
                "name": full_name, "comm": stats["comm_cost"],
                "compute": stats["compute_cost"], "transition": stats["transition_cost"],
            })
    return items


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
                "name": n["name"],
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
                    "name": n["name"],
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


def _find_repeated_layer_container(tree_node):
    """Find a subtree whose children are numbered layers (e.g., 0, 1, 2, ...).

    Returns (container_node, full_dotted_name, first_layer_id) or None.
    With cluster deduplication, not all layer indices may be present, so
    we only require ≥2 numeric children.
    """
    for name, child in tree_node.children.items():
        child_items = list(child.children.items())
        numeric_keys = [n for n, _ in child_items if n.isdigit()]
        if len(numeric_keys) >= 2:
            first_id = min(int(n) for n in numeric_keys)
            return child, name, first_id
        result = _find_repeated_layer_container(child)
        if result is not None:
            container, inner_name, first_id = result
            full_name = f"{name}.{inner_name}" if name else inner_name
            return container, full_name, first_id
    return None


def _build_strategy_overview_html(non_param_nodes, layer_costs, module_tree):
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

    # Discover stage structure from module tree
    layer_info = _find_repeated_layer_container(module_tree)

    if layer_info is not None:
        layer_container, container_name, first_layer_id = layer_info
        repr_layer_key = str(repr_idx if repr_idx is not None else first_layer_id)
        repr_layer_node = layer_container.children.get(repr_layer_key)
        layer_stage_children = list(repr_layer_node.children.items()) if repr_layer_node else []
        layer_prefix = f"{container_name}.{repr_layer_key}."
    else:
        layer_stage_children = []
        container_name = None
        layer_prefix = None

    # Pre/post stages: top-level module_tree children excluding the layer container
    pre_stages = []
    post_stages = []
    seen_layer_container = False
    for name, child in module_tree.children.items():
        if layer_info is not None and child is layer_info[0]:
            seen_layer_container = True
            continue
        if not seen_layer_container:
            pre_stages.append(name)
        else:
            post_stages.append(name)

    ordered_layer = [name for name, _ in layer_stage_children]
    if ordered_layer:
        ordered_layer.append("layer_other")

    def _make_label(name):
        return name.replace("_", " ").title()

    stage_labels = {name: _make_label(name) for name in pre_stages}
    for name in ordered_layer:
        stage_labels[name] = "Other Layer Ops" if name == "layer_other" else _make_label(name)
    stage_labels.update({name: _make_label(name) for name in post_stages})
    stage_labels["other"] = "Other Ops"

    def stage_key(n):
        mp = _strip_module_prefix(n.get("module_path", "") or "")
        if layer_prefix is not None and mp.startswith(layer_prefix):
            suffix = mp[len(layer_prefix):]
            for child_name, _ in layer_stage_children:
                if suffix == child_name or suffix.startswith(child_name + "."):
                    return child_name
            return "layer_other"
        for name in pre_stages + post_stages:
            if mp == name or mp.startswith(name + "."):
                return name
        return "other"

    grouped = {}
    for n in nodes:
        key = stage_key(n)
        grouped.setdefault(key, []).append(n)

    ordered_pre = pre_stages
    ordered_post = post_stages + ["other"]

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
            bg, color = _placement_style(placement)
            placement_html += _placement_chip_html(
                placement, bg, color, None, "", None,
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
            bg, color = _placement_style(op["placement"])
            chip = _placement_chip_html(
                op["placement"], bg, color, None, "", None,
            )
            input_lines = ""
            for entry in op["inputs"]:
                src_bg, src_color = _placement_style(entry["src"])
                src_chip = _placement_chip_html(
                    entry["src"], src_bg, src_color, None, "", None,
                )
                if entry["changed"]:
                    dst_bg, dst_color = _placement_style(entry["dst"])
                    dst_chip = _placement_chip_html(
                        entry["dst"], dst_bg, dst_color, None, "", None,
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
                f'<div class="overview-list-label"><a href="#" class="node-link" onclick="jumpToNode(\'{_esc(op["name"])}\', this); return false">{_esc(op["label"])}</a></div>'
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
                f'<div class="overview-list-label"><a href="#" class="node-link" onclick="jumpToNode(\'{_esc(event["name"])}\', this); return false">{_esc(event["target"])}</a></div>'
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
                    f'<div class="overview-list-label"><a href="#" class="node-link" onclick="jumpToNode(\'{_esc(event["name"])}\', this); return false">{_esc(event["target"])}</a></div>'
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
      <div class="overview-stage-name">{_esc(stage_labels.get(key, key))}</div>
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
            f'<div class="overview-section-title">Representative Layer'
            f'{f" &times;{layer_count}" if layer_count > 1 else ""}</div>'
        )
        html += "".join(stage_block_html(key) for key in ordered_layer if grouped.get(key))

    post_html = "".join(stage_block_html(key) for key in ordered_post if grouped.get(key))
    if post_html:
        html += '<div class="overview-section-title">Final Stages</div>' + post_html

    html += '</div>'
    return html



def _perfetto_trace_payload(non_param_nodes):
    nodes = [
        n
        for n in non_param_nodes
        if n.get("op") not in {"placeholder", "output"}
    ]
    if not nodes:
        return None

    def collective_kind(inp):
        return _infer_collective(inp.get("src_placement"), inp.get("dst_placement"))

    def compute_tid_for_phase(phase):
        return "forward" if phase == "forward" else "backward"

    def _module_prefixes(path):
        if not path:
            return []
        parts = path.split(".")
        if len(parts) <= 1:
            return [path]
        return [".".join(parts[:i]) for i in range(1, len(parts))]

    def _path_prefixes(path):
        prefixes = _module_prefixes(path)
        if path:
            prefixes.append(path)
        return prefixes

    redist_by_target = {}

    metadata_events = [
        {
            "ph": "M",
            "pid": "compute",
            "name": "process_name",
            "args": {"name": "Compute"},
        },
        {
            "ph": "M",
            "pid": "compute",
            "tid": "forward",
            "name": "thread_name",
            "args": {"name": "Forward"},
        },
        {
            "ph": "M",
            "pid": "compute",
            "tid": "backward",
            "name": "thread_name",
            "args": {"name": "Backward"},
        },
        {
            "ph": "M",
            "pid": "compute",
            "tid": "redistribution",
            "name": "thread_name",
            "args": {"name": "Redistribution"},
        },
    ]

    launch_overhead = 1.0
    min_marker_dur = 0.001
    module_scope_margin = 0.01
    module_gap_tolerance = 2 * launch_overhead
    module_end_epsilon = 1e-5
    comm_tid = "redistribution"
    trace_events = []
    compute_events = []
    curr_time = {0: 0.0, comm_tid: 0.0}
    module_runs = {"forward": {}, "backward": {}}
    phase_start = {"forward": None, "backward": None}
    phase_end = {"forward": 0.0, "backward": 0.0}
    flow_id = 0

    for node_idx, n in enumerate(nodes):
        phase = n.get("phase") or "forward"
        compute_tid = compute_tid_for_phase(phase)
        path = _strip_module_prefix(n.get("module_path", "") or "")
        redistribution_end_times = []
        event_start = None
        event_end = None

        for inp_idx, inp in enumerate(n.get("inputs", [])):
            pred_name = inp.get("name")
            comm_cost = float(inp.get("comm_cost", 0.0) or 0.0)
            transition_cost = float(inp.get("transition_cost", 0.0) or 0.0)
            total = comm_cost + transition_cost
            coll = collective_kind(inp)
            if total <= 0 or not coll:
                continue

            src = inp.get("src_placement") or "?"
            dst = inp.get("dst_placement") or n.get("placement", "") or "?"
            curr_time[comm_tid] = max(curr_time[0], curr_time[comm_tid])
            start = curr_time[comm_tid]
            end = start + total
            curr_time[comm_tid] += total + launch_overhead
            curr_time[0] += launch_overhead
            redistribution_end_times.append(end)

            trace_events.append(
                {
                    "ph": "X",
                    "cat": "redistribution",
                    "name": f"{src} -> {dst}",
                    "pid": "compute",
                    "tid": comm_tid,
                    "ts": start,
                    "dur": total,
                    "args": {
                        "order": node_idx,
                        "path": path,
                        "target_node": n.get("name", ""),
                        "input": pred_name or "",
                        "input_index": inp_idx,
                        "phase": phase,
                        "collective": coll,
                        "collective_type": coll,
                        "src_placement": src,
                        "dst_placement": dst,
                        "comm_cost_us": comm_cost,
                        "transition_cost_us": transition_cost,

                        "cluster_id": n.get("cluster_id"),
                    },
                }
            )
            redist_by_target.setdefault((phase, n.get("name", "")), []).append(
                {"start": start, "end": end, "path": path}
            )
            event_start = start if event_start is None else min(event_start, start)
            event_end = max(event_end or 0.0, end)

            flow = flow_id
            flow_id += 1
            trace_events.append(
                {
                    "ph": "s",
                    "pid": "compute",
                    "tid": comm_tid,
                    "ts": start,
                    "id": flow,
                    "cat": "dependency",
                    "name": pred_name or "",
                }
            )
            trace_events.append(
                {
                    "ph": "f",
                    "pid": "compute",
                    "tid": compute_tid,
                    "ts": max(end, curr_time[0]),
                    "id": flow,
                    "cat": "dependency",
                    "name": n.get("name", ""),
                }
            )

        curr_time[0] = max(curr_time[0], max(redistribution_end_times, default=0.0))
        start = curr_time[0]
        dur = float(n.get("compute_cost", 0.0) or 0.0)
        marker_only = dur <= 0
        event_dur = dur if dur > 0 else min_marker_dur
        curr_time[0] += dur + launch_overhead

        compute_event = {
            "ph": "X",
            "cat": "compute",
            "name": n.get("op", "") or n.get("name", ""),
            "pid": "compute",
            "tid": compute_tid,
            "ts": start,
            "dur": event_dur,
            "args": {
                "order": node_idx,
                "path": path,
                "node": n.get("name", ""),
                "phase": phase,
                "module_path": path,
                "op": n.get("op", ""),
                "shape": n.get("shape"),
                "dtype": n.get("dtype"),
                "placement": n.get("placement", ""),
                "cluster_id": n.get("cluster_id"),
                "compute_cost_us": dur,
                "marker_only": marker_only,
                "inputs": [inp.get("name", "") for inp in n.get("inputs", [])],
            },
        }
        compute_events.append(compute_event)
        event_start = start if event_start is None else min(event_start, start)
        event_end = max(event_end or 0.0, start + event_dur)

        if phase_start[phase] is None:
            phase_start[phase] = event_start
        phase_end[phase] = max(phase_end[phase], event_end)

        module_end = max(event_end, max(redistribution_end_times, default=0.0))
        active_paths = set()
        for prefix in _path_prefixes(path):
            active_paths.add(prefix)
            runs = module_runs[phase].setdefault(prefix, [])
            if runs and event_start <= runs[-1]["end"] + module_gap_tolerance:
                runs[-1]["end"] = max(runs[-1]["end"], module_end)
            else:
                runs.append({"start": event_start, "end": module_end})

        for redist_event in redist_by_target.get((phase, n.get("name", "")), []):
            redist_path = redist_event.get("path", "")
            if not redist_path:
                continue
            redist_end = redist_event["end"]
            for prefix in _path_prefixes(redist_path):
                if prefix in active_paths:
                    continue
                runs = module_runs[phase].setdefault(prefix, [])
                if runs and redist_event["start"] <= runs[-1]["end"] + module_gap_tolerance:
                    runs[-1]["end"] = max(runs[-1]["end"], redist_end)
                else:
                    runs.append({"start": redist_event["start"], "end": redist_end})

    for phase in ("forward", "backward"):
        phase_runs = module_runs[phase]
        for prefix, runs in phase_runs.items():
            ancestors = []
            parts = prefix.split(".")
            for i in range(1, len(parts)):
                ancestor = ".".join(parts[:i])
                if ancestor in phase_runs:
                    ancestors.append(ancestor)
            if not ancestors:
                continue
            for run in runs:
                for ancestor in ancestors:
                    for parent_run in phase_runs[ancestor]:
                        if parent_run["start"] <= run["start"] <= parent_run["end"] + module_end_epsilon:
                            if parent_run["end"] < run["end"] <= parent_run["end"] + module_end_epsilon:
                                run["end"] = parent_run["end"]
                            break

    if not trace_events and not compute_events:
        return None

    for phase in ("forward", "backward"):
        phase_pid = "compute"
        phase_tid = compute_tid_for_phase(phase)
        if phase_start[phase] is not None:
            trace_events.append(
                {
                    "ph": "i",
                    "s": "g",
                    "pid": phase_pid,
                    "tid": phase_tid,
                    "ts": phase_start[phase],
                    "name": f"{phase}_start",
                }
            )
            trace_events.append(
                {
                    "ph": "i",
                    "s": "g",
                    "pid": phase_pid,
                    "tid": phase_tid,
                    "ts": phase_end[phase],
                    "name": f"{phase}_end",
                }
            )

        sorted_prefixes = sorted(
            module_runs[phase],
            key=lambda prefix: (prefix.count("."), prefix),
        )
        for prefix in sorted_prefixes:
            for run_idx, run in enumerate(module_runs[phase][prefix]):
                trace_events.append(
                    {
                        "ph": "X",
                        "cat": "module",
                        "name": prefix,
                        "pid": phase_pid,
                        "tid": phase_tid,
                        "ts": max(0.0, run["start"] - module_scope_margin),
                        "dur": max(
                            run["end"] - run["start"] + 2 * module_scope_margin,
                            min_marker_dur,
                        ),
                        "args": {
                            "path": prefix,
                            "phase": phase,
                            "run_index": run_idx,
                            "scope": "module",
                        },
                    }
                )

    deduped_trace_events = []
    for event in trace_events:
        if event.get("ph") != "X" or event.get("cat") != "module":
            deduped_trace_events.append(event)
            continue
        path = event.get("args", {}).get("path", "")
        event_ts = float(event.get("ts", 0.0) or 0.0)
        event_dur = float(event.get("dur", 0.0) or 0.0)
        event_end = event_ts + event_dur
        keep = True
        if path and "." in path:
            parts = path.split(".")
            ancestors = {".".join(parts[:i]) for i in range(1, len(parts))}
            for other in trace_events:
                if other is event:
                    continue
                if (
                    other.get("ph") != "X"
                    or other.get("cat") != "module"
                    or other.get("pid") != event.get("pid")
                    or other.get("tid") != event.get("tid")
                ):
                    continue
                other_path = other.get("args", {}).get("path", "")
                if other_path not in ancestors:
                    continue
                other_ts = float(other.get("ts", 0.0) or 0.0)
                other_dur = float(other.get("dur", 0.0) or 0.0)
                other_end = other_ts + other_dur
                if (
                    abs(other_ts - event_ts) <= module_end_epsilon
                    and abs(other_end - event_end) <= module_end_epsilon
                ):
                    keep = False
                    break
        if keep:
            deduped_trace_events.append(event)
    trace_events = deduped_trace_events

    for phase in ("forward", "backward"):
        phase_module_events = [
            event
            for event in trace_events
            if event.get("ph") == "X"
            and event.get("cat") == "module"
            and event.get("pid") == "compute"
            and event.get("tid") == compute_tid_for_phase(phase)
        ]
        phase_module_events.sort(
            key=lambda event: (
                event.get("args", {}).get("path", "").count("."),
                event.get("args", {}).get("path", ""),
                float(event.get("ts", 0.0) or 0.0),
            )
        )
        for event in phase_module_events:
            path = event.get("args", {}).get("path", "")
            if not path or "." not in path:
                continue
            event_ts = float(event.get("ts", 0.0) or 0.0)
            event_end = event_ts + float(event.get("dur", 0.0) or 0.0)
            parts = path.split(".")
            ancestors = [".".join(parts[:i]) for i in range(1, len(parts))]
            for parent in phase_module_events:
                parent_path = parent.get("args", {}).get("path", "")
                if parent_path not in ancestors:
                    continue
                parent_ts = float(parent.get("ts", 0.0) or 0.0)
                parent_end = parent_ts + float(parent.get("dur", 0.0) or 0.0)
                if parent_ts <= event_ts <= parent_end + module_end_epsilon:
                    if parent_end < event_end <= parent_end + module_end_epsilon:
                        event["dur"] = max(parent_end - event_ts, min_marker_dur)
                        event_end = parent_end

    # Convert module X events to B/E (begin/end) pairs.  X events cannot
    # overlap on the same (pid, tid) track, but nested module scopes
    # inherently do.  B/E uses a stack model designed for exactly this.
    # Tiny depth-based epsilon offsets ensure correct ordering even under
    # an unstable sort: deeper B events start slightly later (nest inside),
    # deeper E events end slightly earlier (close before parents).
    be_epsilon = 0.001
    be_events = []
    for event in trace_events:
        if event.get("ph") == "X" and event.get("cat") == "module":
            ts = float(event["ts"])
            dur = float(event["dur"])
            depth = event.get("args", {}).get("path", "").count(".")
            b_event = {
                k: v for k, v in event.items() if k != "dur"
            }
            b_event["ph"] = "B"
            b_event["ts"] = ts + depth * be_epsilon
            be_events.append(b_event)
            be_events.append(
                {
                    "ph": "E",
                    "cat": "module",
                    "name": event["name"],
                    "pid": event["pid"],
                    "tid": event["tid"],
                    "ts": ts + dur - depth * be_epsilon,
                    "args": event.get("args", {}),
                }
            )
        else:
            be_events.append(event)
    trace_events = be_events

    trace_events.extend(compute_events)

    def _sort_key(event):
        if event["ph"] == "M":
            return (0, 0.0, 0, 0, 0, "", "", "")
        ts = event.get("ts", 0.0)
        ph = event["ph"]
        # B before other events at the same ts (opens scope), E after (closes)
        # Within B at same ts: shallowest first (ascending depth)
        # Within E at same ts: deepest first (descending depth)
        depth = event.get("args", {}).get("path", "").count(".") if event.get("cat") == "module" else 0
        if ph == "B":
            ph_rank, depth_key = 1, depth
        elif ph == "E":
            ph_rank, depth_key = 6, -depth
        else:
            ph_rank = {"i": 2, "X": 3, "s": 4, "f": 5}.get(ph, 5)
            depth_key = 0
        cat_rank = {"module": 0, "compute": 1, "redistribution": 2, "dependency": 3}.get(
            event.get("cat", ""), 4,
        )
        return (1, ts, ph_rank, depth_key, cat_rank, str(event.get("pid", "")), str(event.get("tid", "")), event.get("name", ""))

    trace_events = metadata_events + sorted(trace_events, key=_sort_key)

    total_span = max(
        max(curr_time.values(), default=0.0),
        0.0,
    )
    return {
        "traceEvents": trace_events,
        "traceName": "autoparallel_perfetto_trace.json",
        "displayTimeUnit": "us",
        "metadata": {
            "source": "autoparallel.visualizer.build_display_from_json",
            "total_span_us": total_span,
            "num_events": len(trace_events),
        },
    }


def _build_perfetto_tab_html(non_param_nodes):
    if not non_param_nodes:
        return (
            '<div class="card"><div class="card-title">Perfetto Trace</div>'
            '<div style="font-size:12px;color:#64748b">No execution nodes available.</div></div>'
        )

    full_trace = _perfetto_trace_payload(non_param_nodes)
    full_total_span = full_trace.get("metadata", {}).get("total_span_us", 0.0) if full_trace else 0.0
    full_num_events = full_trace.get("metadata", {}).get("num_events", 0) if full_trace else 0
    nodes_json = json.dumps(non_param_nodes, separators=(",", ":")).replace("</", "<\\/")

    return f"""<div class="card"><div class="card-title">Perfetto Trace</div>
<div class="overview-note">Synthetic Perfetto-compatible timeline derived from exported compute and redistribution costs. Redistributions use dedicated streams keyed by <span class="notation-code">src -&gt; dst</span>.</div>
<div class="trace-controls" style="justify-content:space-between;flex-wrap:wrap">
  <div style="display:flex;gap:12px;flex-wrap:wrap">
    <span class="control-label" style="margin-right:0">Trace mode</span>
    <button class="filter-btn" data-filter-group="perfetto-mode" data-filter-value="representative" onclick="setPerfettoMode('representative', this)">Representative</button>
    <button class="filter-btn active" data-filter-group="perfetto-mode" data-filter-value="full" onclick="setPerfettoMode('full', this)">Full</button>
    <span class="overview-note perfetto-events" style="margin:0">Events: {full_num_events}</span>
    <span class="overview-note perfetto-span" style="margin:0">Total span: {_fmt_us(full_total_span)}</span>
  </div>
  <div style="display:flex;gap:8px;flex-wrap:wrap">
    <button class="filter-btn" onclick="loadPerfettoTrace(this)">Load trace</button>
    <button class="filter-btn" onclick="downloadPerfettoTrace(this)">Download trace</button>
    <a class="filter-btn" href="https://ui.perfetto.dev" target="_blank" rel="noopener noreferrer" style="text-decoration:none">Open Perfetto</a>
  </div>
</div>
<div class="perfetto-status" style="display:none"></div>
<div class="card" style="padding:0;overflow:hidden">
  <iframe class="perfetto-frame" data-src="https://ui.perfetto.dev" width="100%" height="800px" loading="lazy" referrerpolicy="strict-origin-when-cross-origin"></iframe>
</div>
<script type="application/json" class="perfetto-node-data">{nodes_json}</script>
</div>"""


def _render_styles():
    """Return the complete <style> block."""
    return r"""<style>
.ap-viz { margin: 0; padding: 0; box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f8fafc; color: #1e293b; }
.ap-viz *, .ap-viz *::before, .ap-viz *::after { box-sizing: border-box; }
.ap-viz .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
.ap-viz h1 { font-size: 24px; font-weight: 700; margin: 0 0 4px 0; color: #1e293b; }
.ap-viz .subtitle { color: #64748b; font-size: 14px; margin-bottom: 20px; }
.ap-viz .stats-row { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }
.ap-viz .stat-card { background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px 18px; flex: 1; min-width: 140px; color: #1e293b; }
.ap-viz .stat-label { font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }
.ap-viz .stat-value { font-size: 22px; font-weight: 700; margin-top: 2px; }
.ap-viz .stat-detail { font-size: 12px; color: #64748b; margin-top: 2px; }
.ap-viz .controls { display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; align-items: center; }
.ap-viz .control-group { display: flex; gap: 4px; align-items: center; }
.ap-viz .control-label { font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; margin-right: 4px; }
.ap-viz .tabs { display: flex; gap: 0; border-bottom: 2px solid #e2e8f0; margin-bottom: 20px; }
.ap-viz .tab { padding: 10px 20px; cursor: pointer; font-size: 14px; font-weight: 500; color: #64748b; border-bottom: 2px solid transparent; margin-bottom: -2px; transition: all 0.2s; background: transparent; }
.ap-viz .tab:hover { color: #3b82f6; }
.ap-viz .tab.active { color: #3b82f6; border-bottom-color: #3b82f6; }
.ap-viz .tab-content { display: none; }
.ap-viz .tab-content.active { display: block; }
.ap-viz .card { background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 18px; margin-bottom: 16px; color: #1e293b; }
.ap-viz .card-title { font-size: 14px; font-weight: 600; margin-bottom: 12px; color: #1e293b; }
.ap-viz .func-block { border-radius: 8px; padding: 14px; margin-bottom: 10px; border-left: 4px solid #cbd5e1; cursor: pointer; transition: box-shadow 0.2s; }
.ap-viz .func-block:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
.ap-viz .func-name { font-weight: 600; font-size: 14px; color: #1e293b; }
.ap-viz .func-meta { font-size: 12px; color: #64748b; margin-top: 4px; }
.ap-viz .placement-bar { height: 6px; border-radius: 3px; display: flex; overflow: hidden; margin-top: 6px; background: #f1f5f9; }
.ap-viz .placement-bar-seg { height: 100%; min-width: 2px; }
.ap-viz .collective-badge { display: inline-block; background: #FEE2E2; color: #DC2626; font-size: 11px; padding: 2px 8px; border-radius: 10px; margin: 2px 2px; font-weight: 500; }

.ap-viz .cluster-badge { display: inline-block; padding: 1px 6px; border-radius: 8px; font-size: 10px; font-weight: 600; color: #6B7280; background: #F1F5F9; margin-left: 4px; }
.ap-viz .func-details { display: none; margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(0,0,0,0.1); font-size: 12px; }
.ap-viz .func-block.expanded > .func-details { display: block; }
.ap-viz .cost-bar-container { margin-bottom: 10px; }
.ap-viz .cost-bar-label { font-size: 12px; margin-bottom: 3px; display: flex; justify-content: space-between; color: #1e293b; }
.ap-viz .cost-bar { height: 24px; border-radius: 4px; display: flex; overflow: hidden; background: #f1f5f9; }
.ap-viz .cost-bar-segment { height: 100%; display: flex; align-items: center; justify-content: center; font-size: 10px; color: white; font-weight: 500; min-width: 2px; }
.ap-viz .treemap-container { position: relative; height: 400px; background: #f8fafc; border-radius: 8px; overflow: hidden; }
.ap-viz .treemap-cell { position: absolute; overflow: hidden; cursor: pointer; border: 1px solid white; transition: opacity 0.15s; display: flex; align-items: center; justify-content: center; box-sizing: border-box; }
.ap-viz .treemap-cell:hover { opacity: 0.85; }
.ap-viz .treemap-cell span { font-size: 11px; color: white; font-weight: 500; text-shadow: 0 1px 2px rgba(0,0,0,0.5); pointer-events: none; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; padding: 0 4px; }
.ap-viz table { width: 100%; border-collapse: collapse; font-size: 12px; color: #1e293b; }
.ap-viz th { text-align: left; padding: 8px 10px; background: #f8fafc; border-bottom: 2px solid #e2e8f0; font-weight: 600; color: #64748b; text-transform: uppercase; font-size: 11px; letter-spacing: 0.3px; position: sticky; top: 0; z-index: 2; }
.ap-viz td { padding: 7px 10px; border-bottom: 1px solid #f1f5f9; background: white; }
.ap-viz tr:hover td { background: #f8fafc; }
.ap-viz .placement-chip { display: inline-block; padding: 2px 8px; border-radius: 4px; font-family: monospace; font-size: 11px; font-weight: 500; position: relative; cursor: default; }
.ap-viz .placement-chip .chip-tooltip { display: none; position: absolute; left: 0; top: 100%; margin-top: 4px; background: #1e293b; color: white; padding: 6px 10px; border-radius: 6px; font-size: 11px; font-family: monospace; white-space: pre; z-index: 1000; pointer-events: none; box-shadow: 0 2px 8px rgba(0,0,0,0.2); }
.ap-viz .placement-chip:hover .chip-tooltip { display: block; }
.ap-viz .cost-high { color: #DC2626; font-weight: 600; }
.ap-viz .cost-med { color: #F59E0B; }
.ap-viz .cost-low { color: #64748b; }
.ap-viz .filter-bar { display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; align-items: center; }
.ap-viz .filter-btn { padding: 4px 12px; border-radius: 16px; border: 1px solid #e2e8f0; background: white; color: #1e293b; cursor: pointer; font-size: 12px; transition: all 0.15s; }
.ap-viz .filter-btn:hover { border-color: #3b82f6; }
.ap-viz .filter-btn.active { background: #3b82f6; color: white; border-color: #3b82f6; }
.ap-viz .search-box { padding: 5px 12px; border-radius: 16px; border: 1px solid #e2e8f0; background: white; color: #1e293b; font-size: 12px; width: 220px; outline: none; transition: border-color 0.15s; }
.ap-viz .search-box:focus { border-color: #3b82f6; }
.ap-viz .table-scroll { max-height: 600px; overflow-y: auto; }
.ap-viz th.sortable { cursor: pointer; user-select: none; }
.ap-viz th.sortable:hover { color: #3b82f6; }
.ap-viz th.sort-asc::after { content: " \25B2"; font-size: 9px; }
.ap-viz th.sort-desc::after { content: " \25BC"; font-size: 9px; }

.ap-viz .notation-box { margin: -6px 0 16px 0; border: 1px solid #e2e8f0; border-radius: 8px; background: white; }
.ap-viz .notation-summary { list-style: none; cursor: pointer; padding: 10px 14px; font-size: 12px; font-weight: 600; color: #334155; }
.ap-viz .notation-summary::-webkit-details-marker { display: none; }
.ap-viz .notation-body { padding: 0 14px 12px 14px; display: grid; gap: 6px; font-size: 12px; color: #475569; }
.ap-viz .notation-code { display: inline-block; font-family: monospace; background: #f8fafc; border: 1px solid #e2e8f0; padding: 1px 6px; border-radius: 4px; color: #1e293b; }
.ap-viz .ascii-dump { margin-top: 14px; padding: 14px; background: #0f172a; color: #e2e8f0; border-radius: 8px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 11px; line-height: 1.45; overflow-x: auto; white-space: pre; }

.ap-viz .overview-note { font-size: 12px; color: #64748b; margin-bottom: 14px; }
.ap-viz .overview-section-title { font-size: 13px; font-weight: 700; color: #334155; margin: 18px 0 10px 0; }
.ap-viz .overview-stage-card { border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; margin-bottom: 12px; background: #fff; }
.ap-viz .overview-stage-header { display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }
.ap-viz .overview-stage-name { font-size: 14px; font-weight: 700; color: #1e293b; }
.ap-viz .overview-stage-sub { font-size: 12px; color: #64748b; margin-top: 2px; }
.ap-viz .overview-stage-costs { display: flex; flex-wrap: wrap; gap: 10px; font-size: 12px; color: #475569; justify-content: flex-end; }
.ap-viz .overview-stage-meta { font-size: 12px; color: #475569; margin-top: 10px; }
.ap-viz .overview-mini-title { font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.4px; margin: 12px 0 6px 0; }
.ap-viz .overview-list { display: grid; gap: 6px; }
.ap-viz .overview-list-row { display: grid; grid-template-columns: minmax(260px, 1.5fr) minmax(120px, 0.9fr) auto; gap: 10px; align-items: start; font-size: 12px; color: #334155; }
.ap-viz .overview-keyop-list { gap: 0; }
.ap-viz .overview-keyop-row { padding: 8px 0; border-top: 1px solid #e2e8f0; }
.ap-viz .overview-keyop-row:first-child { border-top: 0; padding-top: 0; }
.ap-viz .overview-keyop-row:last-child { padding-bottom: 0; }
.ap-viz .overview-list-label { font-family: monospace; overflow-wrap: anywhere; }
.ap-viz .overview-list-meta { color: #64748b; font-size: 11px; }
.ap-viz .overview-placement-cell { display: flex; flex-wrap: wrap; gap: 6px; }
.ap-viz .overview-placement-flow { display: grid; gap: 4px; font-size: 11px; color: #475569; }
.ap-viz .overview-placement-row { display: grid; grid-template-columns: 44px 1fr; gap: 8px; align-items: center; }
.ap-viz .overview-placement-label { font-family: monospace; font-size: 11px; color: #64748b; text-align: right; }
.ap-viz .overview-placement-values { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.ap-viz .overview-placement-arrow { color: #94a3b8; font-size: 11px; }
.ap-viz .overview-placement-coll { color: #64748b; font-size: 11px; }
.ap-viz .overview-details { margin-top: 10px; border: 1px solid #e2e8f0; border-radius: 8px; background: #f8fafc; }
.ap-viz .overview-details > summary { cursor: pointer; padding: 8px 10px; font-size: 12px; font-weight: 600; color: #334155; }
.ap-viz .overview-details-body { padding: 0 10px 10px 10px; }
.ap-viz .perfetto-frame { display: block; border: 0; background: #fff; }
.ap-viz .perfetto-status { margin-bottom: 12px; padding: 10px 12px; border-radius: 8px; border: 1px solid #e2e8f0; font-size: 12px; color: #475569; background: #f8fafc; }
.ap-viz .perfetto-status.error { color: #B91C1C; background: #FEF2F2; border-color: #FECACA; }
.ap-viz .param-tree table { width: 100%; }
.ap-viz .param-tree td { vertical-align: top; }
.ap-viz .param-module-row td { background: #f8fafc; font-weight: 600; color: #334155; }
.ap-viz .param-tree th:first-child, .ap-viz .param-tree td:first-child { min-width: 420px; }
.ap-viz .param-module-label { display: inline-flex; align-items: center; gap: 6px; }
.ap-viz .param-name-cell { white-space: nowrap; }
.ap-viz .param-leaf-name { font-family: monospace; font-size: 11px; color: #1e293b; }
.ap-viz .param-node-icon { display: inline-block; vertical-align: middle; margin-right: 8px; }
.ap-viz .param-module-icon { width: 10px; height: 10px; border-radius: 999px; background: #94a3b8; }
.ap-viz .param-leaf-icon { width: 10px; height: 2px; background: #cbd5e1; border-radius: 999px; }
.ap-viz tr.linked-row { opacity: 0.45; display: none; }
.ap-viz tr.linked-row td { font-style: italic; }
.ap-viz tr.arch-bwd { opacity: 0.6; }
.ap-viz tr.arch-input-edge td { background: #f8fafc; border-bottom: 1px solid #f1f5f9; }
.ap-viz .phase-badge { display: inline-block; padding: 1px 6px; border-radius: 8px; font-size: 10px; font-weight: 600; }
.ap-viz .phase-fwd { color: #10B981; background: #ECFDF5; }
.ap-viz .phase-bwd { color: #6B7280; background: #F1F5F9; }
.ap-viz .zero-cost-group { opacity: 0.72; }
.ap-viz .zero-cost-group:not(.expanded) > .func-details { display: none; }
.ap-viz .cost-hotspot { border-left-color: #DC2626 !important; background: #FEF2F2; }
.ap-viz .node-link { color: #3B82F6; text-decoration: none; cursor: pointer; }
.ap-viz .node-link:hover { text-decoration: underline; }
@keyframes nodeHighlight { from { background: #FEF08A; } to { background: transparent; } }
.ap-viz .node-highlight { animation: nodeHighlight 1.5s ease-out; }
.ap-viz .func-block.search-expanded > .func-details { display: block; }
</style>"""


def _render_script():
    """Return the complete <script> block."""
    return """<script>
function _root(el) { return el.closest('.ap-viz'); }

function switchTab(id, btn) {
  var r = _root(btn);
  if (id !== 'perfetto') {
    _cancelPerfettoLoad(r.querySelector('.tab-content[data-tab="perfetto"]'));
  }
  r.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  r.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  r.querySelector('.tab-content[data-tab="' + id + '"]').classList.add('active');
  btn.classList.add('active');
  if (id === 'perfetto') {
    loadPerfettoTrace(btn);
  }
  if (id === 'cost' && typeof initTreemap === 'function') {
    initTreemap();
  }
}

var filterState = {
  phase: 'all',
  nodeType: 'all',
  searchText: '',
  archSearchText: '',
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

function setArchSearch(el) {
  filterState.archSearchText = el.value.toLowerCase();
  var r = _root(el);
  r.querySelectorAll('.func-block.search-expanded').forEach(function(b) {
    b.classList.remove('search-expanded');
  });
  applyFilters(r);
}

function archExpandAll(btn) {
  _root(btn).querySelectorAll('.func-block').forEach(function(b) { b.classList.add('expanded'); });
}
function archCollapseAll(btn) {
  _root(btn).querySelectorAll('.func-block').forEach(function(b) {
    b.classList.remove('expanded');
    b.classList.remove('search-expanded');
  });
}

function jumpToNode(name, el) {
  var r = _root(el);
  // Switch to All Nodes tab
  var tabBtn = r.querySelector('.tab[data-tab-id="detail"]');
  if (tabBtn) switchTab('detail', tabBtn);
  // Reset filters so the row is visible
  filterState.nodeType = 'all';
  filterState.searchText = '';
  filterState.representativeLayersOnly = false;
  r.querySelectorAll('[data-filter-group="node-type"]').forEach(function(b) {
    b.classList.toggle('active', b.textContent.trim() === 'All');
  });
  var clusterBtn = r.querySelector('#btn-cluster-collapse');
  if (clusterBtn) {
    clusterBtn.textContent = 'All Repeated Layers';
    clusterBtn.classList.remove('active');
  }
  var searchBox = r.querySelector('.tab-content[data-tab="detail"] .search-box');
  if (searchBox) searchBox.value = '';
  applyFilters(r);
  // Find and highlight the row
  var row = r.querySelector('tr[data-node-name="' + name.replace(/'/g, "\\'") + '"]');
  if (row) {
    row.style.display = '';
    row.scrollIntoView({behavior: 'smooth', block: 'center'});
    row.classList.remove('node-highlight');
    void row.offsetWidth;
    row.classList.add('node-highlight');
  }
}

function jumpToArch(moduleName, el) {
  var r = _root(el);
  var tabBtn = r.querySelector('.tab[data-tab-id="arch"]');
  if (tabBtn) switchTab('arch', tabBtn);
  // Clear previous search state
  r.querySelectorAll('.func-block.search-expanded').forEach(function(b) {
    b.classList.remove('search-expanded');
  });
  filterState.archSearchText = moduleName.toLowerCase();
  var searchBox = r.querySelector('.tab-content[data-tab="arch"] .search-box');
  if (searchBox) searchBox.value = moduleName;
  applyFilters(r);
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

  var archText = filterState.archSearchText;

  r.querySelectorAll('.arch-row').forEach(function(row) {
    var show = filterState.phase === 'all' || row.dataset.phase === filterState.phase;
    if (show && archText) {
      show = row.textContent.toLowerCase().indexOf(archText) >= 0;
    }
    row.style.display = show ? '' : 'none';
  });

  r.querySelectorAll('.func-block').forEach(function(block) {
    var show = true;
    if (filterState.phase !== 'all') {
      var phases = block.dataset.phases || '';
      show = phases.indexOf(filterState.phase) >= 0;
    }
    if (show && archText) {
      show = block.textContent.toLowerCase().indexOf(archText) >= 0;
    }
    block.style.display = show ? '' : 'none';
  });

  if (archText) {
    // Ensure parents of visible blocks are visible and expanded
    r.querySelectorAll('.func-block').forEach(function(block) {
      if (block.style.display !== 'none') {
        block.classList.add('search-expanded');
        var parent = block.parentElement;
        while (parent) {
          if (parent.classList && parent.classList.contains('func-block')) {
            parent.style.display = '';
            parent.classList.add('search-expanded');
          }
          parent = parent.parentElement;
        }
      }
    });
  }
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

function _perfettoTabRoot(el) {
  return _root(el).querySelector('.tab-content[data-tab="perfetto"]');
}

function _perfettoStatus(tab) {
  return tab ? tab.querySelector('.perfetto-status') : null;
}

function _setPerfettoStatus(tab, message, isError) {
  var status = _perfettoStatus(tab);
  if (!status) return;
  if (!message) {
    status.style.display = 'none';
    status.textContent = '';
    status.classList.remove('error');
    return;
  }
  status.style.display = 'block';
  status.textContent = message;
  status.classList.toggle('error', !!isError);
}

function _cancelPerfettoLoad(tab) {
  if (!tab || !tab._perfettoLoadState) return;
  var state = tab._perfettoLoadState;
  state.cancelled = true;
  if (state.interval) clearInterval(state.interval);
  if (state.timeout) clearTimeout(state.timeout);
  if (state.onMessage) window.removeEventListener('message', state.onMessage);
  if (state.frame && state.onFrameLoad) state.frame.removeEventListener('load', state.onFrameLoad);
  tab._perfettoLoadState = null;
  var frame = tab.querySelector('.perfetto-frame');
  if (frame && frame.dataset.loading === '1') frame.dataset.loading = '0';
}

function _perfettoNodeData(tab) {
  var script = tab ? tab.querySelector('.perfetto-node-data') : null;
  if (!script) return [];
  if (!tab._perfettoNodeDataCache) {
    tab._perfettoNodeDataCache = JSON.parse(script.textContent);
  }
  return tab._perfettoNodeDataCache;
}

function _perfettoMode(tab) {
  return tab && tab.dataset.perfettoMode ? tab.dataset.perfettoMode : 'full';
}

function _perfettoNodesForMode(nodes, mode) {
  if (mode !== 'representative') return nodes;
  return nodes.filter(function(n) { return !('cluster_root' in n); });
}

function _perfettoInferCollective(src, dst) {
  if (!src || !dst || src === dst) return null;
  function dims(p) {
    var out = [];
    for (var i = 0; i < p.length;) {
      if (p[i] === 'S' && i + 1 < p.length && p[i + 1] === '(') {
        var endS = p.indexOf(')', i);
        out.push(p.slice(i, endS + 1));
        i = endS + 1;
      } else if (p[i] === 'P' && i + 1 < p.length && p[i + 1] === '(') {
        var endP = p.indexOf(')', i);
        out.push(p.slice(i, endP + 1));
        i = endP + 1;
      } else if (p[i] === 'R') {
        out.push('R');
        i += 1;
      } else {
        i += 1;
      }
    }
    return out;
  }
  var srcDims = dims(src);
  var dstDims = dims(dst);
  if (!srcDims.length || srcDims.length !== dstDims.length) {
    if (src.indexOf('P(sum)') >= 0 && dst.indexOf('S(') >= 0) return 'ReduceScatter';
    if (src.indexOf('P(sum)') >= 0 && dst.indexOf('R') >= 0) return 'AllReduce';
    if (src.indexOf('S(') >= 0 && dst.indexOf('R') >= 0 && dst.indexOf('P') < 0) return 'AllGather';
    return 'Redistribute';
  }
  var collectives = [];
  for (var i = 0; i < srcDims.length; i++) {
    var sd = srcDims[i], dd = dstDims[i], coll = null;
    if (sd === dd || (sd === 'R' && dd.indexOf('S(') === 0)) {
      coll = null;
    } else if (sd.indexOf('S(') === 0 && dd === 'R') {
      coll = 'AllGather';
    } else if (sd.indexOf('P(') === 0 && dd === 'R') {
      coll = 'AllReduce';
    } else if (sd.indexOf('P(') === 0 && dd.indexOf('S(') === 0) {
      coll = 'ReduceScatter';
    } else if (sd.indexOf('S(') === 0 && dd.indexOf('S(') === 0) {
      coll = 'AllToAll';
    } else {
      coll = 'Redistribute';
    }
    if (coll) collectives.push(coll);
  }
  if (!collectives.length) return null;
  if (collectives.length === 1) return collectives[0];
  return collectives.join('+');
}

function _perfettoStripModulePrefix(path) {
  if (!path) return '';
  if (path.indexOf("L['self'].") === 0) return path.slice("L['self'].".length);
  if (path.indexOf('_export_root.') === 0) return path.slice('_export_root.'.length);
  return path;
}

function _perfettoBuildTrace(nodes, mode) {
  nodes = _perfettoNodesForMode(nodes, mode).filter(function(n) {
    return n.op !== 'placeholder' && n.op !== 'output' && (!n.placeholder_kind || (n.placeholder_kind !== 'param' && n.placeholder_kind !== 'buffer'));
  });
  if (!nodes.length) return null;

  function computeTidForPhase(phase) {
    return phase === 'backward' ? 'backward' : 'forward';
  }
  function modulePrefixes(path) {
    if (!path) return [];
    var parts = path.split('.');
    if (parts.length <= 1) return [path];
    var prefixes = [];
    for (var i = 1; i < parts.length; i++) prefixes.push(parts.slice(0, i).join('.'));
    return prefixes;
  }
  function pathPrefixes(path) {
    var prefixes = modulePrefixes(path);
    if (path) prefixes.push(path);
    return prefixes;
  }

  var metadataEvents = [
    {ph: 'M', pid: 'compute', name: 'process_name', args: {name: 'Compute'}},
    {ph: 'M', pid: 'compute', tid: 'forward', name: 'thread_name', args: {name: 'Forward'}},
    {ph: 'M', pid: 'compute', tid: 'backward', name: 'thread_name', args: {name: 'Backward'}},
    {ph: 'M', pid: 'compute', tid: 'redistribution', name: 'thread_name', args: {name: 'Redistribution'}},
  ];

  var launchOverhead = 1.0;
  var minMarkerDur = 0.001;
  var moduleScopeMargin = 0.01;
  var moduleGapTolerance = 2 * launchOverhead;
  var moduleEndEpsilon = 1e-5;
  var commTid = 'redistribution';
  var traceEvents = [];
  var computeEvents = [];
  var currTime = {0: 0.0, redistribution: 0.0};
  var moduleRuns = {forward: {}, backward: {}};
  var phaseStart = {forward: null, backward: null};
  var phaseEnd = {forward: 0.0, backward: 0.0};
  var redistByTarget = {};

  nodes.forEach(function(n, nodeIdx) {
    var phase = n.phase || 'forward';
    var computeTid = computeTidForPhase(phase);
    var path = _perfettoStripModulePrefix(n.module_path || '');
    var redistributionEndTimes = [];
    var eventStart = null;
    var eventEnd = null;

    (n.inputs || []).forEach(function(inp, inpIdx) {
      var commCost = Number(inp.comm_cost || 0.0);
      var transitionCost = Number(inp.transition_cost || 0.0);
      var total = commCost + transitionCost;
      var coll = _perfettoInferCollective(inp.src_placement, inp.dst_placement);
      if (!(total > 0) || !coll) return;
      var src = inp.src_placement || '?';
      var dst = inp.dst_placement || n.placement || '?';
      currTime[commTid] = Math.max(currTime[0], currTime[commTid]);
      var start = currTime[commTid];
      var end = start + total;
      currTime[commTid] += total + launchOverhead;
      currTime[0] += launchOverhead;
      redistributionEndTimes.push(end);
      traceEvents.push({
        ph: 'X',
        cat: 'redistribution',
        name: src + ' -> ' + dst,
        pid: 'compute',
        tid: commTid,
        ts: start,
        dur: total,
        args: {
          order: nodeIdx,
          path: path,
          target_node: n.name || '',
          input: inp.name || '',
          input_index: inpIdx,
          phase: phase,
          collective: coll,
          collective_type: coll,
          src_placement: src,
          dst_placement: dst,
          comm_cost_us: commCost,
          transition_cost_us: transitionCost,
          cluster_id: n.cluster_id,
        }
      });
      var key = phase + '::' + (n.name || '');
      redistByTarget[key] = redistByTarget[key] || [];
      redistByTarget[key].push({start: start, end: end, path: path});
      eventStart = eventStart === null ? start : Math.min(eventStart, start);
      eventEnd = Math.max(eventEnd || 0.0, end);
    });

    currTime[0] = Math.max(currTime[0], redistributionEndTimes.length ? Math.max.apply(null, redistributionEndTimes) : 0.0);
    var start = currTime[0];
    var dur = Number(n.compute_cost || 0.0);
    var markerOnly = dur <= 0;
    var eventDur = dur > 0 ? dur : minMarkerDur;
    currTime[0] += dur + launchOverhead;

    computeEvents.push({
      ph: 'X',
      cat: 'compute',
      name: n.op || n.name || '',
      pid: 'compute',
      tid: computeTid,
      ts: start,
      dur: eventDur,
      args: {
        order: nodeIdx,
        path: path,
        node: n.name || '',
        phase: phase,
        module_path: path,
        op: n.op || '',
        shape: n.shape,
        dtype: n.dtype,
        placement: n.placement || '',
        cluster_id: n.cluster_id,
        compute_cost_us: dur,
        marker_only: markerOnly,
        inputs: (n.inputs || []).map(function(inp) { return inp.name || ''; }),
      }
    });

    eventStart = eventStart === null ? start : Math.min(eventStart, start);
    eventEnd = Math.max(eventEnd || 0.0, start + eventDur);
    if (phaseStart[phase] === null) phaseStart[phase] = eventStart;
    phaseEnd[phase] = Math.max(phaseEnd[phase], eventEnd);

    var moduleEnd = Math.max(eventEnd, redistributionEndTimes.length ? Math.max.apply(null, redistributionEndTimes) : 0.0);
    var activePaths = {};
    pathPrefixes(path).forEach(function(prefix) {
      activePaths[prefix] = true;
      moduleRuns[phase][prefix] = moduleRuns[phase][prefix] || [];
      var runs = moduleRuns[phase][prefix];
      if (runs.length && eventStart <= runs[runs.length - 1].end + moduleGapTolerance) {
        runs[runs.length - 1].end = Math.max(runs[runs.length - 1].end, moduleEnd);
      } else {
        runs.push({start: eventStart, end: moduleEnd});
      }
    });

    (redistByTarget[phase + '::' + (n.name || '')] || []).forEach(function(redistEvent) {
      var redistPath = redistEvent.path || '';
      if (!redistPath) return;
      var redistEnd = redistEvent.end;
      pathPrefixes(redistPath).forEach(function(prefix) {
        if (activePaths[prefix]) return;
        moduleRuns[phase][prefix] = moduleRuns[phase][prefix] || [];
        var runs = moduleRuns[phase][prefix];
        if (runs.length && redistEvent.start <= runs[runs.length - 1].end + moduleGapTolerance) {
          runs[runs.length - 1].end = Math.max(runs[runs.length - 1].end, redistEnd);
        } else {
          runs.push({start: redistEvent.start, end: redistEnd});
        }
      });
    });
  });

  ['forward', 'backward'].forEach(function(phase) {
    var phaseRuns = moduleRuns[phase];
    Object.keys(phaseRuns).forEach(function(prefix) {
      var ancestors = [];
      var parts = prefix.split('.');
      for (var i = 1; i < parts.length; i++) {
        var ancestor = parts.slice(0, i).join('.');
        if (phaseRuns[ancestor]) ancestors.push(ancestor);
      }
      if (!ancestors.length) return;
      phaseRuns[prefix].forEach(function(run) {
        ancestors.forEach(function(ancestor) {
          phaseRuns[ancestor].forEach(function(parentRun) {
            if (parentRun.start <= run.start && run.start <= parentRun.end + moduleEndEpsilon) {
              if (parentRun.end < run.end && run.end <= parentRun.end + moduleEndEpsilon) {
                run.end = parentRun.end;
              }
            }
          });
        });
      });
    });
  });

  if (!traceEvents.length && !computeEvents.length) return null;

  ['forward', 'backward'].forEach(function(phase) {
    var phasePid = 'compute';
    var phaseTid = computeTidForPhase(phase);
    if (phaseStart[phase] !== null) {
      traceEvents.push({ph: 'i', s: 'g', pid: phasePid, tid: phaseTid, ts: phaseStart[phase], name: phase + '_start'});
      traceEvents.push({ph: 'i', s: 'g', pid: phasePid, tid: phaseTid, ts: phaseEnd[phase], name: phase + '_end'});
    }
    Object.keys(moduleRuns[phase]).sort(function(a, b) {
      var da = (a.match(/\./g) || []).length;
      var db = (b.match(/\./g) || []).length;
      return da - db || a.localeCompare(b);
    }).forEach(function(prefix) {
      moduleRuns[phase][prefix].forEach(function(run, runIdx) {
        traceEvents.push({
          ph: 'X',
          cat: 'module',
          name: prefix,
          pid: phasePid,
          tid: phaseTid,
          ts: Math.max(0.0, run.start - moduleScopeMargin),
          dur: Math.max(run.end - run.start + 2 * moduleScopeMargin, minMarkerDur),
          args: {
            path: prefix,
            phase: phase,
            run_index: runIdx,
            scope: 'module',
          }
        });
      });
    });
  });

  traceEvents = traceEvents.filter(function(event, _, allEvents) {
    if (event.ph !== 'X' || event.cat !== 'module') return true;
    var path = (event.args && event.args.path) || '';
    if (!path || path.indexOf('.') < 0) return true;
    var eventTs = Number(event.ts || 0.0);
    var eventEnd = eventTs + Number(event.dur || 0.0);
    var parts = path.split('.');
    var ancestors = [];
    for (var i = 1; i < parts.length; i++) ancestors.push(parts.slice(0, i).join('.'));
    for (var j = 0; j < allEvents.length; j++) {
      var other = allEvents[j];
      if (other === event) continue;
      if (other.ph !== 'X' || other.cat !== 'module' || other.pid !== event.pid || other.tid !== event.tid) continue;
      var otherPath = (other.args && other.args.path) || '';
      if (ancestors.indexOf(otherPath) < 0) continue;
      var otherTs = Number(other.ts || 0.0);
      var otherEnd = otherTs + Number(other.dur || 0.0);
      if (Math.abs(otherTs - eventTs) <= moduleEndEpsilon && Math.abs(otherEnd - eventEnd) <= moduleEndEpsilon) {
        return false;
      }
    }
    return true;
  });

  ['forward', 'backward'].forEach(function(phase) {
    var phaseTid = computeTidForPhase(phase);
    var phaseModuleEvents = traceEvents.filter(function(event) {
      return event.ph === 'X'
        && event.cat === 'module'
        && event.pid === 'compute'
        && event.tid === phaseTid;
    }).sort(function(a, b) {
      var ap = (a.args && a.args.path) || '';
      var bp = (b.args && b.args.path) || '';
      var ad = (ap.match(/\./g) || []).length;
      var bd = (bp.match(/\./g) || []).length;
      return ad - bd || ap.localeCompare(bp) || (a.ts || 0) - (b.ts || 0);
    });
    phaseModuleEvents.forEach(function(event) {
      var path = (event.args && event.args.path) || '';
      if (!path || path.indexOf('.') < 0) return;
      var eventTs = Number(event.ts || 0.0);
      var eventEnd = eventTs + Number(event.dur || 0.0);
      var parts = path.split('.');
      var ancestors = [];
      for (var i = 1; i < parts.length; i++) ancestors.push(parts.slice(0, i).join('.'));
      phaseModuleEvents.forEach(function(parent) {
        var parentPath = (parent.args && parent.args.path) || '';
        if (ancestors.indexOf(parentPath) < 0) return;
        var parentTs = Number(parent.ts || 0.0);
        var parentEnd = parentTs + Number(parent.dur || 0.0);
        if (parentTs <= eventTs && eventTs <= parentEnd + moduleEndEpsilon) {
          if (parentEnd < eventEnd && eventEnd <= parentEnd + moduleEndEpsilon) {
            event.dur = Math.max(parentEnd - eventTs, minMarkerDur);
            eventEnd = parentEnd;
          }
        }
      });
    });
  });

  // Convert module X events to B/E pairs for proper nesting.
  // Depth-based epsilon offsets ensure correct ordering under unstable sort.
  var beEpsilon = 0.001;
  var beEvents = [];
  traceEvents.forEach(function(event) {
    if (event.ph === 'X' && event.cat === 'module') {
      var ts = Number(event.ts);
      var dur = Number(event.dur);
      var path = (event.args && event.args.path) || '';
      var depth = (path.match(/\./g) || []).length;
      var bEvent = {};
      Object.keys(event).forEach(function(k) { if (k !== 'dur') bEvent[k] = event[k]; });
      bEvent.ph = 'B';
      bEvent.ts = ts + depth * beEpsilon;
      beEvents.push(bEvent);
      beEvents.push({ph: 'E', cat: 'module', name: event.name, pid: event.pid, tid: event.tid, ts: ts + dur - depth * beEpsilon, args: event.args || {}});
    } else {
      beEvents.push(event);
    }
  });
  traceEvents = beEvents;

  traceEvents = metadataEvents.concat(traceEvents.concat(computeEvents).sort(function(a, b) {
    if (a.ph === 'M' && b.ph !== 'M') return -1;
    if (a.ph !== 'M' && b.ph === 'M') return 1;
    if (a.ph === 'M' && b.ph === 'M') return 0;
    var tsDiff = (a.ts || 0) - (b.ts || 0);
    if (tsDiff !== 0) return tsDiff;
    function phRank(ev) {
      if (ev.ph === 'B') return 1;
      if (ev.ph === 'E') return 6;
      return {i: 2, X: 3, s: 4, f: 5}[ev.ph] !== undefined ? {i: 2, X: 3, s: 4, f: 5}[ev.ph] : 5;
    }
    var phDiff = phRank(a) - phRank(b);
    if (phDiff !== 0) return phDiff;
    function depthKey(ev) {
      if (ev.cat !== 'module') return 0;
      var path = (ev.args && ev.args.path) || '';
      var d = (path.match(/\./g) || []).length;
      return ev.ph === 'B' ? d : (ev.ph === 'E' ? -d : 0);
    }
    var depthDiff = depthKey(a) - depthKey(b);
    if (depthDiff !== 0) return depthDiff;
    function catRank(ev) {
      return {module: 0, compute: 1, redistribution: 2, dependency: 3}[ev.cat] !== undefined ? {module: 0, compute: 1, redistribution: 2, dependency: 3}[ev.cat] : 4;
    }
    return catRank(a) - catRank(b)
      || String(a.pid || '').localeCompare(String(b.pid || ''))
      || String(a.tid || '').localeCompare(String(b.tid || ''))
      || String(a.name || '').localeCompare(String(b.name || ''));
  }));

  var totalSpan = Math.max(currTime[0] || 0.0, currTime[commTid] || 0.0, 0.0);
  return {
    traceEvents: traceEvents,
    traceName: 'autoparallel_perfetto_trace.json',
    displayTimeUnit: 'us',
    metadata: {
      source: 'autoparallel.visualizer.build_display_from_json',
      total_span_us: totalSpan,
      num_events: traceEvents.length,
      mode: mode,
    }
  };
}

function _fmtUsJs(us) {
  if (us >= 1000) return (us / 1000).toFixed(1) + 'ms';
  return Math.round(us) + 'µs';
}

function _updatePerfettoSummary(tab) {
  if (!tab) return;
  var trace = _perfettoBuildTrace(_perfettoNodeData(tab), _perfettoMode(tab));
  if (!trace) return;
  var eventsEl = tab.querySelector('.perfetto-events');
  var spanEl = tab.querySelector('.perfetto-span');
  if (eventsEl) eventsEl.textContent = 'Events: ' + ((trace.metadata && trace.metadata.num_events) || 0);
  if (spanEl) spanEl.textContent = 'Total span: ' + _fmtUsJs((trace.metadata && trace.metadata.total_span_us) || 0);
  tab._perfettoTraceMeta = trace.metadata || {};
}

function setPerfettoMode(mode, btn) {
  var tab = _perfettoTabRoot(btn);
  if (!tab) return;
  tab.dataset.perfettoMode = mode;
  _root(btn).querySelectorAll('[data-filter-group="perfetto-mode"]').forEach(function(b) {
    b.classList.toggle('active', b === btn);
  });
  _updatePerfettoSummary(tab);
  var frame = tab.querySelector('.perfetto-frame');
  var wasLoaded = frame && frame.dataset.loaded === '1';
  var wasLoading = frame && frame.dataset.loading === '1';
  _cancelPerfettoLoad(tab);
  if (frame) {
    frame.dataset.loaded = '0';
    frame.dataset.loading = '0';
  }
  _setPerfettoStatus(tab, '', false);
  if (wasLoaded || wasLoading) {
    loadPerfettoTrace(btn);
  }
}

function _perfettoTraceText(tab) {
  var trace = _perfettoBuildTrace(_perfettoNodeData(tab), _perfettoMode(tab));
  if (!trace) return '';
  tab._perfettoTraceMeta = trace.metadata || {};
  return JSON.stringify(trace);
}

function _perfettoArrayBuffer(traceText) {
  var encoded = new TextEncoder().encode(traceText);
  return encoded.buffer;
}

function _ensurePerfettoFrame(frame) {
  if (frame && !frame.getAttribute('src')) {
    frame.dataset.ready = '0';
    frame.setAttribute('src', frame.dataset.src || 'https://ui.perfetto.dev');
  }
}

function _perfettoSafePostMessage(handle, payload, targetOrigin) {
  if (!handle) return false;
  try {
    handle.postMessage(payload, targetOrigin);
    return true;
  } catch (err) {
    return false;
  }
}

function downloadPerfettoTrace(btn) {
  var tab = _perfettoTabRoot(btn);
  var traceText = _perfettoTraceText(tab);
  if (!traceText) return;
  var blob = new Blob([traceText], {type: 'application/json'});
  var url = URL.createObjectURL(blob);
  var link = document.createElement('a');
  link.href = url;
  link.download = 'autoparallel_perfetto_trace.json';
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(function() { URL.revokeObjectURL(url); }, 0);
}

function loadPerfettoTrace(btn) {
  var tab = _perfettoTabRoot(btn);
  var frame = tab ? tab.querySelector('.perfetto-frame') : null;
  if (!frame || frame.dataset.loaded === '1' || frame.dataset.loading === '1') return;
  var traceText = _perfettoTraceText(tab);
  if (!traceText) return;

  _cancelPerfettoLoad(tab);
  tab._perfettoLoadSeq = (tab._perfettoLoadSeq || 0) + 1;
  var mode = _perfettoMode(tab);
  _setPerfettoStatus(tab, 'Loading Perfetto trace...', false);
  frame.dataset.loading = '1';
  var targetOrigin = 'https://ui.perfetto.dev';
  var traceArrayBuffer = _perfettoArrayBuffer(traceText);
  var state = {
    id: tab._perfettoLoadSeq,
    interval: null,
    timeout: null,
    onMessage: null,
    onFrameLoad: null,
    frame: frame,
    cancelled: false,
    pingStarted: false,
    traceSent: false,
  };

  function isCurrent() {
    return !state.cancelled && tab._perfettoLoadState === state && tab._perfettoLoadSeq === state.id;
  }

  function cleanup() {
    state.cancelled = true;
    if (state.interval) clearInterval(state.interval);
    if (state.timeout) clearTimeout(state.timeout);
    if (state.onMessage) window.removeEventListener('message', state.onMessage);
    if (state.frame && state.onFrameLoad) state.frame.removeEventListener('load', state.onFrameLoad);
    if (tab._perfettoLoadState === state) tab._perfettoLoadState = null;
  }

  function startPingLoop() {
    if (!isCurrent() || state.pingStarted || frame.dataset.loading !== '1') return;
    state.pingStarted = true;
    frame.dataset.ready = '1';
    var handle = frame.contentWindow;
    _perfettoSafePostMessage(handle, 'PING', targetOrigin);
    state.interval = setInterval(function() {
      if (!isCurrent()) {
        if (state.interval) clearInterval(state.interval);
        state.interval = null;
        return;
      }
      _perfettoSafePostMessage(handle, 'PING', targetOrigin);
    }, 100);
  }

  state.onFrameLoad = function() {
    startPingLoop();
  };
  frame.addEventListener('load', state.onFrameLoad);
  _ensurePerfettoFrame(frame);
  if (frame.dataset.ready === '1' || frame.getAttribute('src')) {
    if (frame.dataset.ready === '1') startPingLoop();
  }

  state.onMessage = function(evt) {
    if (!isCurrent()) return;
    var handle = frame.contentWindow;
    if (evt.source !== handle || evt.origin !== targetOrigin || evt.data !== 'PONG') return;
    if (state.traceSent) return;
    if (state.interval) {
      clearInterval(state.interval);
      state.interval = null;
    }
    if (!_perfettoSafePostMessage(handle, {
      perfetto: {
        buffer: traceArrayBuffer,
        title: 'AutoParallel Trace (' + mode + ')',
        keepApiOpen: true,
      }
    }, targetOrigin)) {
      state.pingStarted = false;
      startPingLoop();
      return;
    }
    state.traceSent = true;
    cleanup();
    frame.dataset.loaded = '1';
    frame.dataset.loading = '0';
    _setPerfettoStatus(tab, '', false);
  };

  state.timeout = setTimeout(function() {
    if (!isCurrent() || state.traceSent) return;
    cleanup();
    frame.dataset.loading = '0';
    _setPerfettoStatus(
      tab,
      'Perfetto did not respond. You can use "Open Perfetto" to launch it in a new tab or download the trace JSON.',
      true
    );
  }, 30000);

  tab._perfettoLoadState = state;
  window.addEventListener('message', state.onMessage);
  if (frame.dataset.ready === '1') startPingLoop();
}


if (typeof initTreemap === 'function') initTreemap();
</script>"""


def _hotspot_table_html(hotspots):
    if not hotspots:
        return '<p style="font-size:12px;color:#94a3b8">No communication costs.</p>'
    rows = ""
    for h in hotspots:
        module_esc = _esc(h["module"]) if h["module"] else "(root)"
        rows += (
            f'<tr>'
            f'<td style="font-size:12px;font-weight:500">'
            f'<a href="#" class="node-link" onclick="jumpToArch(\'{_esc(h["module"])}\', this); return false">{module_esc}</a></td>'
            f'<td><span class="collective-badge">{_esc(h["coll"])}</span></td>'
            f'<td style="font-family:monospace;font-size:11px">{_esc(h["src"])} &rarr; {_esc(h["dst"])}</td>'
            f'<td style="text-align:center">&times;{h["count"]}</td>'
            f'<td style="font-family:monospace;font-weight:600;color:#DC2626">{_fmt_us(h["total_cost"])}</td>'
            f'</tr>'
        )
    return (
        '<div class="table-scroll" style="max-height:300px">'
        '<table><thead><tr>'
        '<th>Module</th><th>Collective</th><th>Placement Change</th><th>Occurrences</th><th>Total Comm</th>'
        f'</tr></thead><tbody>{rows}</tbody></table></div>'
    )



def _prepare_viz_data(data):
    """Extract and precompute all shared data from the JSON export dict."""
    mesh = data["mesh"]
    nodes = data["nodes"]
    summary = data["summary"]

    # Precompute per-node derived fields
    for n in nodes:
        placement = n.get("placement", "")
        n["_bg"], n["_color"] = _placement_style(placement)
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
    perfetto_nodes = [
        n
        for n in compute_nodes
        if n.get("placeholder_kind") not in {"param", "buffer"}
    ]

    # Build hierarchical module tree
    module_tree = _build_module_tree(architecture_nodes)

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

    # Per-layer cost breakdown (for Execution Overview)
    import re as _re

    layer_container_info = _find_repeated_layer_container(module_tree)
    if layer_container_info is not None:
        _lc_name = layer_container_info[1]
        _layer_pattern = _re.compile(rf"{_re.escape(_lc_name)}\.(\d+)")
    else:
        _layer_pattern = None

    layer_costs = {}
    for n in architecture_nodes:
        mp = n.get("module_path", "")
        m = _layer_pattern.search(mp) if _layer_pattern else None
        if m:
            idx = int(m.group(1))
            if idx not in layer_costs:
                layer_costs[idx] = {
                    "comm": 0.0,
                    "compute": 0.0,
                    "transition": 0.0,
                    "num_nodes": 0,
                }
            layer_costs[idx]["comm"] += n["_total_comm"]
            layer_costs[idx]["compute"] += n.get("compute_cost", 0)
            layer_costs[idx]["transition"] += n["_total_transition"]
            layer_costs[idx]["num_nodes"] += 1

    total_cost = summary["total"] or 1
    comm_pct = summary["comm"] / total_cost * 100
    compute_pct = summary["compute"] / total_cost * 100
    trans_pct = summary["transition"] / total_cost * 100

    return {
        "mesh": mesh, "nodes": nodes, "summary": summary,
        "compute_nodes": compute_nodes, "has_clusters": has_clusters,
        "cluster_counts": cluster_counts,
        "param_nodes": param_nodes, "architecture_nodes": architecture_nodes,
        "perfetto_nodes": perfetto_nodes, "module_tree": module_tree,
        "top_costly": top_costly, "redist_nodes": redist_nodes,
        "mesh_desc": mesh_desc, "layer_costs": layer_costs,
        "total_cost": total_cost, "comm_pct": comm_pct,
        "compute_pct": compute_pct, "trans_pct": trans_pct,
    }


def generate_visualization_html(data: dict) -> str:
    """Generate interactive HTML visualization from the JSON export dict.

    Args:
        data: The dict returned by export_sharding_json() or
              ShardingOptimizer.get_json().

    Returns:
        A self-contained HTML string.
    """
    ctx = _prepare_viz_data(data)
    mesh = ctx["mesh"]
    summary = ctx["summary"]
    compute_nodes = ctx["compute_nodes"]
    has_clusters = ctx["has_clusters"]
    cluster_counts = ctx["cluster_counts"]
    param_nodes = ctx["param_nodes"]
    architecture_nodes = ctx["architecture_nodes"]
    perfetto_nodes = ctx["perfetto_nodes"]
    module_tree = ctx["module_tree"]
    top_costly = ctx["top_costly"]
    redist_nodes = ctx["redist_nodes"]
    mesh_desc = ctx["mesh_desc"]
    layer_costs = ctx["layer_costs"]
    total_cost = ctx["total_cost"]
    comm_pct = ctx["comm_pct"]
    compute_pct = ctx["compute_pct"]
    trans_pct = ctx["trans_pct"]

    html = '<!DOCTYPE html>\n<html><head><meta charset="UTF-8">\n'
    html += _render_styles()
    html += f'''</head><body>
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
  <div class="tab" data-tab-id="params" onclick="switchTab('params', this)">Parameters</div>
  <div class="tab" data-tab-id="overview" onclick="switchTab('overview', this)">Execution Overview</div>
  <div class="tab" data-tab-id="perfetto" onclick="switchTab('perfetto', this)">Perfetto</div>
  <div class="tab" data-tab-id="arch" onclick="switchTab('arch', this)">Architecture</div>
  <div class="tab active" data-tab-id="cost" onclick="switchTab('cost', this)">Cost Breakdown</div>
  <div class="tab" data-tab-id="detail" onclick="switchTab('detail', this)">All Nodes</div>
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
    html += _build_strategy_overview_html(architecture_nodes, layer_costs, module_tree)
    html += '</div>'

    # ===== PERFETTO TAB =====
    html += '''
<div data-tab="perfetto" class="tab-content">'''
    html += _build_perfetto_tab_html(perfetto_nodes)
    html += '</div>'

    # ===== ARCHITECTURE TAB =====
    html += '''
<div data-tab="arch" class="tab-content">
<div class="card"><div class="card-title">Computation Blocks by Module</div>
<div class="filter-bar">
  <input type="text" class="search-box" placeholder="Search modules..." oninput="setArchSearch(this)">
  <button class="filter-btn" onclick="archExpandAll(this)">Expand All</button>
  <button class="filter-btn" onclick="archCollapseAll(this)">Collapse All</button>
</div>'''

    # Render top-level tree children in execution order (dict preserves
    # insertion order from graph traversal)
    flat_costs = _flatten_tree_for_costs(module_tree)
    comm_costs = sorted([s["comm_cost"] for _, s in flat_costs if s["comm_cost"] > 0])
    comm_threshold = comm_costs[int(len(comm_costs) * 0.9)] if len(comm_costs) >= 5 else 0
    for child in module_tree.children.values():
        html += _render_tree_block(child, cluster_counts, mesh, depth=0, total_cost=total_cost, comm_threshold=comm_threshold)

    html += '</div></div>'

    # ===== COST BREAKDOWN TAB =====
    html += f'''
<div data-tab="cost" class="tab-content active">
<div class="card"><div class="card-title">Cost Distribution</div>
<div class="cost-bar-container">
  <div class="cost-bar-label"><span>Cost Breakdown</span></div>
  <div class="cost-bar" style="height:32px">
    <div class="cost-bar-segment" style="width:{comm_pct}%;background:#DC2626">Comm {comm_pct:.0f}%</div>
    <div class="cost-bar-segment" style="width:{compute_pct}%;background:#10B981">Compute {compute_pct:.0f}%</div>
    <div class="cost-bar-segment" style="width:{trans_pct}%;background:#F59E0B">Trans {trans_pct:.0f}%</div>
  </div>
</div></div>'''

    # Communication hotspots — group by (module, collective, src, dst) per phase
    hotspot_by_phase = {"forward": {}, "backward": {}}
    phase_totals = {"forward": 0.0, "backward": 0.0}
    for n in architecture_nodes:
        module = _strip_module_prefix(n.get("module_path", "") or n.get("name", ""))
        # Use top-level module as the grouping scope
        module_prefix = module.split(".")[0] if module else ""
        phase = n.get("phase", "forward")
        cid = n.get("cluster_id")
        instances = (cluster_counts.get(cid, 0) + 1) if cid is not None else 1
        for inp_name, coll_type, src_p, dst_p, ccost in n["_collectives"]:
            if ccost <= 0:
                continue
            cost = ccost * instances
            phase_totals[phase] = phase_totals.get(phase, 0) + cost
            groups = hotspot_by_phase.get(phase, hotspot_by_phase["forward"])
            key = (module_prefix, coll_type, src_p, dst_p)
            if key not in groups:
                groups[key] = {
                    "coll": coll_type, "src": src_p, "dst": dst_p,
                    "module": module_prefix,
                    "total_cost": 0.0, "count": 0,
                }
            groups[key]["total_cost"] += cost
            groups[key]["count"] += instances


    fwd_hotspots = sorted(hotspot_by_phase["forward"].values(), key=lambda h: -h["total_cost"])[:15]
    bwd_hotspots = sorted(hotspot_by_phase["backward"].values(), key=lambda h: -h["total_cost"])[:15]

    fwd_total = phase_totals["forward"]
    bwd_total = phase_totals["backward"]
    fwd_shown = sum(h["total_cost"] for h in fwd_hotspots)
    bwd_shown = sum(h["total_cost"] for h in bwd_hotspots)
    fwd_note = "" if fwd_shown >= fwd_total * 0.99 else f" (top 15 shown, {_fmt_us(fwd_total)} total)"
    bwd_note = "" if bwd_shown >= bwd_total * 0.99 else f" (top 15 shown, {_fmt_us(bwd_total)} total)"

    html += f'''
<div class="card"><div class="card-title">Communication Hotspots — Forward <span style="font-weight:400;color:#64748b">{_fmt_us(fwd_total)}{fwd_note}</span></div>'''
    html += _hotspot_table_html(fwd_hotspots)
    html += f'''</div>
<div class="card"><div class="card-title">Communication Hotspots — Backward <span style="font-weight:400;color:#64748b">{_fmt_us(bwd_total)}{bwd_note}</span></div>'''
    html += _hotspot_table_html(bwd_hotspots)
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
  <div class="cost-bar-label"><span style="font-family:monospace"><a href="#" class="node-link" onclick="jumpToNode('{_esc(n["name"])}', this); return false">{_esc(n["name"])}</a></span><span style="color:#64748b">{_esc(src_label)} &middot; total: {_fmt_us(total)}</span></div>
  <div class="cost-bar">
    <div class="cost-bar-segment" style="width:{comm_w}%;background:#DC2626" title="comm: {_fmt_us(n['_total_comm'])}"></div>
    <div class="cost-bar-segment" style="width:{comp_w}%;background:#10B981" title="compute: {_fmt_us(n.get('compute_cost', 0))}"></div>
    <div class="cost-bar-segment" style="width:{trans_w}%;background:#F59E0B" title="trans: {_fmt_us(n['_total_transition'])}"></div>
  </div>
</div>'''

    # Cost by module — flat treemap with comm/compute mode toggle
    treemap_items = _treemap_data(module_tree)
    treemap_json = json.dumps(treemap_items)
    html += f'''</div><div class="card"><div class="card-title">Cost by Module</div>
<div class="filter-bar">
  <button class="filter-btn active" onclick="setTreemapMode('comm', this)">Communication</button>
  <button class="filter-btn" onclick="setTreemapMode('compute', this)">Compute</button>
</div>
<div class="treemap-container" id="treemap-container"></div>
<script>
function _tmFmtUs(us) {{ return us >= 1000 ? (us/1000).toFixed(1)+'ms' : Math.round(us)+'\\u00b5s'; }}

function _tmSquarify(items, rect) {{
  if (!items.length) return;
  var total = 0;
  for (var i = 0; i < items.length; i++) total += items[i].value;
  if (total === 0) return;
  _tmLayout(items, 0, rect, total, rect.w * rect.h);
}}

function _tmLayout(items, start, rect, total, area) {{
  if (start >= items.length) return;
  if (items.length - start === 1) {{
    items[start]._x = rect.x; items[start]._y = rect.y;
    items[start]._w = rect.w; items[start]._h = rect.h;
    return;
  }}
  var short = Math.min(rect.w, rect.h);
  var rowSum = 0, bestRatio = Infinity, end = start;
  for (var i = start; i < items.length; i++) {{
    rowSum += items[i].value;
    var frac = rowSum / total * area;
    var rowLen = frac / short;
    var last = items[i].value / total * area / rowLen;
    var ratio = Math.max(rowLen / last, last / rowLen);
    if (ratio > bestRatio) break;
    bestRatio = ratio;
    end = i;
  }}
  rowSum = 0;
  for (var i = start; i <= end; i++) rowSum += items[i].value;
  var stripFrac = rowSum / total;
  var pos = 0;
  for (var i = start; i <= end; i++) {{
    var itemFrac = items[i].value / rowSum;
    if (rect.w >= rect.h) {{
      var sw = rect.w * stripFrac;
      items[i]._x = rect.x; items[i]._y = rect.y + rect.h * pos;
      items[i]._w = sw; items[i]._h = rect.h * itemFrac;
    }} else {{
      var sh = rect.h * stripFrac;
      items[i]._x = rect.x + rect.w * pos; items[i]._y = rect.y;
      items[i]._w = rect.w * itemFrac; items[i]._h = sh;
    }}
    pos += itemFrac;
  }}
  if (end + 1 < items.length) {{
    var rem;
    if (rect.w >= rect.h) {{
      rem = {{x: rect.x + rect.w * stripFrac, y: rect.y, w: rect.w * (1 - stripFrac), h: rect.h}};
    }} else {{
      rem = {{x: rect.x, y: rect.y + rect.h * stripFrac, w: rect.w, h: rect.h * (1 - stripFrac)}};
    }}
    _tmLayout(items, end + 1, rem, total - rowSum, rem.w * rem.h);
  }}
}}

var _treemapInited = false;
var _treemapMode = 'comm';
var _treemapItems = {treemap_json};
function setTreemapMode(mode, btn) {{
  _treemapMode = mode;
  btn.parentElement.querySelectorAll('.filter-btn').forEach(function(b) {{ b.classList.remove('active'); }});
  btn.classList.add('active');
  if (_treemapInited) _treemapRender();
}}
function initTreemap() {{
  if (_treemapInited) return;
  _treemapInited = true;
  _treemapRender();
}}
function _treemapRender() {{
  var container = document.getElementById('treemap-container');
  container.innerHTML = '';
  var mode = _treemapMode;

  // Copy items and assign .value based on mode
  var items = [];
  for (var i = 0; i < _treemapItems.length; i++) {{
    var d = _treemapItems[i];
    var v = (mode === 'comm') ? d.comm : d.compute;
    if (v > 0) items.push({{name: d.name, comm: d.comm, compute: d.compute, transition: d.transition, value: v}});
  }}
  items.sort(function(a, b) {{ return b.value - a.value; }});

  if (!items.length) {{
    container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#94a3b8;font-size:14px">No ' + mode + ' cost</div>';
    return;
  }}

  var rect = {{x: 0, y: 0, w: container.offsetWidth, h: container.offsetHeight}};
  _tmSquarify(items, rect);
  var totalValue = 0;
  for (var i = 0; i < items.length; i++) totalValue += items[i].value;

  for (var i = 0; i < items.length; i++) {{
    var d = items[i];
    if (!d._w || !d._h) continue;

    // Intensity: selected cost / total cost of this module
    var t = d.comm + d.compute + d.transition;
    var intensity = t > 0 ? d.value / t : 0;
    var r, g, b;
    if (mode === 'comm') {{
      r = Math.round(255 - intensity * (255 - 0xDC));
      g = Math.round(255 - intensity * (255 - 0x26));
      b = Math.round(255 - intensity * (255 - 0x26));
    }} else {{
      r = Math.round(255 - intensity * (255 - 0x10));
      g = Math.round(255 - intensity * (255 - 0xB9));
      b = Math.round(255 - intensity * (255 - 0x81));
    }}
    var txtColor = intensity > 0.4 ? '#fff' : '#1e293b';

    var cell = document.createElement('div');
    cell.className = 'treemap-cell';
    cell.style.left = d._x + 'px'; cell.style.top = d._y + 'px';
    cell.style.width = d._w + 'px'; cell.style.height = d._h + 'px';
    cell.style.background = 'rgb(' + r + ',' + g + ',' + b + ')';

    var pct = (d.value / totalValue * 100).toFixed(1);
    cell.title = d.name + '\\ncomm: ' + _tmFmtUs(d.comm) + '  compute: ' + _tmFmtUs(d.compute) + '  trans: ' + _tmFmtUs(d.transition) + '\\n' + pct + '% of total ' + mode;

    if (d._w > 50 && d._h > 20) {{
      var span = document.createElement('span');
      span.textContent = d.name;
      span.style.color = txtColor;
      span.style.textShadow = 'none';
      cell.appendChild(span);
    }}
    // Mini stacked cost bar at bottom
    if (d._w > 20 && d._h > 16 && t > 0) {{
      var bar = document.createElement('div');
      bar.style.cssText = 'position:absolute;bottom:0;left:0;right:0;height:4px;display:flex;pointer-events:none';
      bar.innerHTML = '<div style="width:'+(d.comm/t*100)+'%;background:#DC2626;height:100%"></div>'
        + '<div style="width:'+(d.compute/t*100)+'%;background:#10B981;height:100%"></div>'
        + '<div style="width:'+(d.transition/t*100)+'%;background:#F59E0B;height:100%"></div>';
      cell.appendChild(bar);
    }}
    (function(name) {{
      cell.addEventListener('click', function() {{
        jumpToArch(name, cell);
      }});
    }})(d.name);
    container.appendChild(cell);
  }}
}}
</script>'''

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
            f'data-placeholder-kind="{_esc(placeholder_kind)}" '
            f'data-node-name="{_esc(n["name"])}"'
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

'''
    html += _render_script()
    html += '\n</div></div></body></html>'
    return html
