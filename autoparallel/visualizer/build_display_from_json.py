# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
HTML visualizer that consumes the JSON dict from export_json.py.

Supports cluster collapsing for multi-layer models, phase filtering,
and redistribution highlighting.
"""

import math


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


def _esc(s):
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


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

    # Group by module_path (fall back to source.func)
    def _group_key(n):
        if n.get("module_path"):
            return n["module_path"]
        src = n.get("source")
        if src and src.get("func"):
            return src["func"]
        if n.get("op") == "placeholder":
            return "inputs"
        return "other"

    module_groups = {}
    for n in display_nodes:
        key = _group_key(n)
        module_groups.setdefault(key, []).append(n)

    module_summary = []
    for mname, mnodes in module_groups.items():
        comm = sum(n["_total_comm"] for n in mnodes)
        compute = sum(n.get("compute_cost", 0) for n in mnodes)
        strategies = [n["_strategy"] for n in mnodes]
        dominant = max(set(strategies), key=strategies.count) if strategies else "unknown"
        all_colls = []
        for n in mnodes:
            all_colls.extend(n["_collectives"])
        module_summary.append({
            "name": mname,
            "nodes": mnodes,
            "num_nodes": len(mnodes),
            "comm_cost": comm,
            "compute_cost": compute,
            "dominant": dominant,
            "color": _STRATEGY_COLORS.get(dominant, "#D1D5DB"),
            "bg": _STRATEGY_BGS.get(dominant, "#F9FAFB"),
            "collectives": all_colls,
        })
    module_summary.sort(key=lambda m: m["comm_cost"] + m["compute_cost"], reverse=True)

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

    total_cost = summary["total"] or 1
    comm_pct = summary["comm"] / total_cost * 100
    compute_pct = summary["compute"] / total_cost * 100
    trans_pct = summary["transition"] / total_cost * 100

    # ===== BUILD HTML =====
    html = f'''<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f8fafc; color: #1e293b; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
h1 {{ font-size: 24px; font-weight: 700; margin-bottom: 4px; }}
.subtitle {{ color: #64748b; font-size: 14px; margin-bottom: 20px; }}
.stats-row {{ display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }}
.stat-card {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px 18px; flex: 1; min-width: 140px; }}
.stat-label {{ font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }}
.stat-value {{ font-size: 22px; font-weight: 700; margin-top: 2px; }}
.stat-detail {{ font-size: 12px; color: #64748b; margin-top: 2px; }}
.controls {{ display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; align-items: center; }}
.control-group {{ display: flex; gap: 4px; align-items: center; }}
.control-label {{ font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; margin-right: 4px; }}
.tabs {{ display: flex; gap: 0; border-bottom: 2px solid #e2e8f0; margin-bottom: 20px; }}
.tab {{ padding: 10px 20px; cursor: pointer; font-size: 14px; font-weight: 500; color: #64748b; border-bottom: 2px solid transparent; margin-bottom: -2px; transition: all 0.2s; }}
.tab:hover {{ color: #3b82f6; }}
.tab.active {{ color: #3b82f6; border-bottom-color: #3b82f6; }}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}
.card {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 18px; margin-bottom: 16px; }}
.card-title {{ font-size: 14px; font-weight: 600; margin-bottom: 12px; }}
.func-block {{ border-radius: 8px; padding: 14px; margin-bottom: 10px; border-left: 4px solid; cursor: pointer; transition: box-shadow 0.2s; }}
.func-block:hover {{ box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.func-name {{ font-weight: 600; font-size: 14px; }}
.func-meta {{ font-size: 12px; color: #64748b; margin-top: 4px; }}
.collective-badge {{ display: inline-block; background: #FEE2E2; color: #DC2626; font-size: 11px; padding: 2px 8px; border-radius: 10px; margin: 2px 2px; font-weight: 500; }}
.strategy-badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 500; color: white; }}
.cluster-badge {{ display: inline-block; padding: 1px 6px; border-radius: 8px; font-size: 10px; font-weight: 600; color: #6B7280; background: #F1F5F9; margin-left: 4px; }}
.func-details {{ display: none; margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(0,0,0,0.1); font-size: 12px; }}
.func-block.expanded .func-details {{ display: block; }}
.cost-bar-container {{ margin-bottom: 10px; }}
.cost-bar-label {{ font-size: 12px; margin-bottom: 3px; display: flex; justify-content: space-between; }}
.cost-bar {{ height: 24px; border-radius: 4px; display: flex; overflow: hidden; background: #f1f5f9; }}
.cost-bar-segment {{ height: 100%; display: flex; align-items: center; justify-content: center; font-size: 10px; color: white; font-weight: 500; min-width: 2px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
th {{ text-align: left; padding: 8px 10px; background: #f8fafc; border-bottom: 2px solid #e2e8f0; font-weight: 600; color: #64748b; text-transform: uppercase; font-size: 11px; letter-spacing: 0.3px; position: sticky; top: 0; }}
td {{ padding: 7px 10px; border-bottom: 1px solid #f1f5f9; }}
tr:hover td {{ background: #f8fafc; }}
.placement-chip {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-family: monospace; font-size: 11px; font-weight: 500; }}
.cost-high {{ color: #DC2626; font-weight: 600; }}
.cost-med {{ color: #F59E0B; }}
.cost-low {{ color: #64748b; }}
.filter-bar {{ display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }}
.filter-btn {{ padding: 4px 12px; border-radius: 16px; border: 1px solid #e2e8f0; background: white; cursor: pointer; font-size: 12px; transition: all 0.15s; }}
.filter-btn:hover {{ border-color: #3b82f6; }}
.filter-btn.active {{ background: #3b82f6; color: white; border-color: #3b82f6; }}
.table-scroll {{ max-height: 600px; overflow-y: auto; }}
.legend {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 16px; }}
.legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 12px; }}
.legend-dot {{ width: 12px; height: 12px; border-radius: 3px; }}
tr.linked-row {{ opacity: 0.45; }}
tr.linked-row td {{ font-style: italic; }}
</style></head><body>
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
    <span class="control-label">Layers</span>
    <button class="filter-btn active" id="btn-cluster-collapse" onclick="toggleClusters(this)">Single Layer</button>
  </div>'''

    html += '</div>'

    # Tabs
    html += '''
<div class="tabs">
  <div class="tab active" onclick="switchTab('arch')">Architecture</div>
  <div class="tab" onclick="switchTab('cost')">Cost Breakdown</div>
  <div class="tab" onclick="switchTab('detail')">All Nodes</div>
</div>'''

    # ===== ARCHITECTURE TAB =====
    html += '''
<div id="tab-arch" class="tab-content active">
<div class="card"><div class="card-title">Computation Blocks by Module</div>'''

    for ms in module_summary:
        coll_html = ""
        for cname, ctype, cfrom, cto, ccost in ms["collectives"][:5]:
            coll_html += f'<span class="collective-badge">{_esc(ctype)}: {_esc(cfrom)}&rarr;{_esc(cto)} ({ccost:.0f})</span> '
        if len(ms["collectives"]) > 5:
            coll_html += f'<span class="collective-badge">+{len(ms["collectives"]) - 5} more</span>'

        detail_rows = ""
        for n in ms["nodes"]:
            comm = n["_total_comm"]
            compute = n.get("compute_cost", 0)
            cost_class = "cost-high" if comm > 100 else "cost-med" if comm > 0 else "cost-low"
            shape_str = ",".join(str(s) for s in n.get("shape", []))
            dtype = n.get("dtype", "")
            placement = n.get("placement", "")
            phase = n.get("phase", "")
            cluster_html = ""
            cid = n.get("cluster_id")
            if cid is not None and "cluster_root" not in n and cid in cluster_counts:
                cluster_html = f'<span class="cluster-badge">&times;{cluster_counts[cid] + 1}</span>'

            detail_rows += (
                f'<tr class="arch-row" data-phase="{phase}">'
                f'<td style="font-family:monospace">{_esc(n["name"])}{cluster_html}</td>'
                f'<td>{_esc(dtype)}[{shape_str}]</td>'
                f'<td><span class="placement-chip" style="background:{n["_bg"]};color:{n["_color"]}">{_esc(placement)}</span></td>'
                f'<td class="{cost_class}" style="font-family:monospace">{comm:.1f}</td>'
                f'<td style="font-family:monospace">{compute:.1f}</td>'
                f'</tr>'
            )

        html += f'''
<div class="func-block" style="background:{ms['bg']};border-color:{ms['color']}" onclick="this.classList.toggle('expanded')">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <div>
      <span class="func-name">{_esc(ms['name'])}</span>
      <span class="strategy-badge" style="background:{ms['color']}">{ms['dominant'].upper()}</span>
    </div>
    <div style="text-align:right;font-size:12px;color:#64748b">
      {ms['num_nodes']} ops &middot; comm: {ms['comm_cost']:.0f} &middot; compute: {ms['compute_cost']:.0f}
    </div>
  </div>
  <div class="func-meta">{coll_html if coll_html else "No redistributions"}</div>
  <div class="func-details">
    <table><tr><th>Node</th><th>Shape</th><th>Placement</th><th>Comm</th><th>Compute</th></tr>
    {detail_rows}</table>
  </div>
</div>'''

    html += '</div></div>'

    # ===== COST BREAKDOWN TAB =====
    html += f'''
<div id="tab-cost" class="tab-content">
<div class="card"><div class="card-title">Cost Distribution</div>
<div class="cost-bar-container">
  <div class="cost-bar-label"><span>Overall Breakdown</span></div>
  <div class="cost-bar" style="height:32px">
    <div class="cost-bar-segment" style="width:{comm_pct}%;background:#DC2626">Comm {comm_pct:.0f}%</div>
    <div class="cost-bar-segment" style="width:{compute_pct}%;background:#10B981">Compute {compute_pct:.0f}%</div>
    <div class="cost-bar-segment" style="width:{trans_pct}%;background:#F59E0B">{trans_pct:.0f}%</div>
  </div>
</div></div>

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
    func_max = max((m["comm_cost"] + m["compute_cost"]) for m in module_summary) if module_summary else 1
    for ms in module_summary:
        total = ms["comm_cost"] + ms["compute_cost"]
        if total == 0:
            continue
        comm_w = ms["comm_cost"] / func_max * 100
        comp_w = ms["compute_cost"] / func_max * 100
        html += f'''<div class="cost-bar-container">
  <div class="cost-bar-label"><span style="font-weight:600">{_esc(ms["name"])}</span><span style="color:#64748b">{ms["num_nodes"]} ops &middot; total: {total:.0f}</span></div>
  <div class="cost-bar">
    <div class="cost-bar-segment" style="width:{comm_w}%;background:#DC2626"></div>
    <div class="cost-bar-segment" style="width:{comp_w}%;background:#10B981"></div>
  </div>
</div>'''

    html += '</div></div>'

    # ===== ALL NODES TAB =====
    html += '''
<div id="tab-detail" class="tab-content">
<div class="card">
<div class="card-title">All Nodes</div>
<div class="filter-bar">
  <button class="filter-btn active" onclick="filterNodes('all', this)">All</button>
  <button class="filter-btn" onclick="filterNodes('placeholder', this)">Params</button>
  <button class="filter-btn" onclick="filterNodes('forward', this)">Forward</button>
  <button class="filter-btn" onclick="filterNodes('backward', this)">Backward</button>
  <button class="filter-btn" onclick="filterNodes('redist', this)">Has Redistribution</button>
</div>
<div class="table-scroll">
<table id="node-table"><thead>
<tr><th>Name</th><th>Op</th><th>Phase</th><th>Shape</th><th>Placement</th><th>Comm</th><th>Compute</th><th>Redistribution</th><th>Module</th></tr>
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
            coll_html += f'<span class="collective-badge">{_esc(ctype)} ({ccost:.0f})</span> '

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

        html += f'''<tr class="{row_class}" {data_attrs}>
  <td style="font-family:monospace;font-size:11px">{_esc(n["name"])}{cluster_html}</td>
  <td style="font-size:11px">{_esc(op_name)}</td>
  <td style="font-size:11px">{phase}</td>
  <td style="font-size:11px">{_esc(dtype)}[{shape_str}]</td>
  <td><span class="placement-chip" style="background:{n['_bg']};color:{n['_color']}">{_esc(placement)}</span></td>
  <td class="{cost_class}" style="font-family:monospace">{comm:.1f}</td>
  <td style="font-family:monospace">{compute:.1f}</td>
  <td>{coll_html}</td>
  <td style="font-size:11px;color:#64748b">{_esc(module)}</td>
</tr>'''

    html += '''</tbody></table></div></div></div>

<script>
function switchTab(id) {
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-' + id).classList.add('active');
  event.target.classList.add('active');
}

var clustersCollapsed = true;

function toggleClusters(btn) {
  clustersCollapsed = !clustersCollapsed;
  btn.textContent = clustersCollapsed ? 'Single Layer' : 'All Layers';
  btn.classList.toggle('active', clustersCollapsed);
  document.querySelectorAll('.node-row').forEach(row => {
    if (row.dataset.linked === '1') {
      row.style.display = clustersCollapsed ? 'none' : '';
    }
  });
}

function filterPhase(phase, btn) {
  document.querySelectorAll('.control-group:first-child .filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.node-row').forEach(row => {
    if (phase === 'all') { row.style.display = (clustersCollapsed && row.dataset.linked === '1') ? 'none' : ''; return; }
    var match = row.dataset.phase === phase;
    row.style.display = (!match || (clustersCollapsed && row.dataset.linked === '1')) ? 'none' : '';
  });
  // Also filter architecture rows
  document.querySelectorAll('.arch-row').forEach(row => {
    if (phase === 'all') { row.style.display = ''; return; }
    row.style.display = row.dataset.phase === phase ? '' : 'none';
  });
}

function filterNodes(type, btn) {
  document.querySelectorAll('#tab-detail .filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.node-row').forEach(row => {
    var show = true;
    if (type === 'placeholder') show = row.dataset.op === 'placeholder';
    else if (type === 'forward') show = row.dataset.phase === 'forward';
    else if (type === 'backward') show = row.dataset.phase === 'backward';
    else if (type === 'redist') show = row.dataset.redist === '1';
    if (clustersCollapsed && row.dataset.linked === '1') show = false;
    row.style.display = show ? '' : 'none';
  });
}

// Initial state: hide linked rows if clusters exist
if (clustersCollapsed) {
  document.querySelectorAll('.node-row[data-linked="1"]').forEach(row => {
    row.style.display = 'none';
  });
}
</script>
</div></body></html>'''

    return html
