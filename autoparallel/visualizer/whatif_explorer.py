# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Interactive What-If Explorer for sharding optimization.

Displays a module-tree view with placement dropdowns. Users can pin
constraints on individual nodes, re-solve the ILP, and see the cost
impact — all inside a Jupyter notebook. Works with ShardingOptimizer.load()
(no live DeviceMesh or process group required).

Usage:
    from autoparallel.visualizer.whatif_explorer import WhatIfExplorer
    explorer = WhatIfExplorer("llama_8b.ap")
    explorer.show()
"""

from __future__ import annotations

import html as html_lib
from collections import defaultdict
from pathlib import Path

import torch

# Module-level reference so JS kernel.execute() can call back.
_active_explorer: WhatIfExplorer | None = None


def _parse_placement_string(s: str, ndim: int):
    """Parse a placement string like 'S(0)R' into a tuple of Placement objects.

    Handles standard placements (R, S(dim), P(sum)) and extended forms
    like MaskP(sum, None, 0) produced by _pretty_print_spec.
    """
    from torch.distributed.tensor.placement_types import Partial, Replicate, Shard

    placements: list = []
    i = 0
    while i < len(s):
        if s[i] == "R" and (i + 1 >= len(s) or s[i + 1] != "e"):
            # 'R' but not start of 'Replicate' or similar
            placements.append(Replicate())
            i += 1
        elif s[i] == "S" and i + 1 < len(s) and s[i + 1] == "(":
            end = s.index(")", i)
            dim = int(s[i + 2 : end])
            placements.append(Shard(dim))
            i = end + 1
        elif s[i] == "P" and i + 1 < len(s) and s[i + 1] == "(":
            end = s.index(")", i)
            placements.append(Partial())
            i = end + 1
        elif s[i:].startswith("MaskP("):
            # MaskP(sum, None, 0) — find the matching closing paren
            depth = 0
            j = i
            while j < len(s):
                if s[j] == "(":
                    depth += 1
                elif s[j] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            # For constraint purposes, treat MaskP as Partial
            placements.append(Partial())
            i = j + 1
        else:
            i += 1
    if len(placements) != ndim:
        raise ValueError(
            f"Parsed {len(placements)} placements from '{s}', expected {ndim}"
        )
    return tuple(placements)


def _spec_to_str(spec) -> str:
    """Convert an OpSpec's output_specs to a short placement string."""
    from torch.distributed.tensor._dtensor_spec import DTensorSpec
    from torch.distributed.tensor._op_schema import _pretty_print_spec

    if isinstance(spec, DTensorSpec):
        return _pretty_print_spec(spec)
    if isinstance(spec, (list, tuple)):
        for s in spec:
            if isinstance(s, DTensorSpec):
                return _pretty_print_spec(s)
    return str(spec)


class WhatIfExplorer:
    """Interactive sharding what-if explorer for Jupyter notebooks."""

    def __init__(self, path_or_optimizer):
        from autoparallel.optimize_sharding import ShardingOptimizer

        if isinstance(path_or_optimizer, (str, Path)):
            self.optimizer = ShardingOptimizer.load(str(path_or_optimizer))
        else:
            self.optimizer = path_or_optimizer

        self.baseline_solution = self.optimizer.get_solution()
        self.current_solution = dict(self.baseline_solution)
        self.pinned: dict[str, list[str]] = {}  # node_name -> constraint names
        self._display_handle = None
        self._error: str | None = None

        # Pre-compute node name -> available placements for dropdowns
        self._placement_options: dict[str, list[str]] = {}
        for node in self.optimizer.nodes:
            strat = self.optimizer.get_strategy(node)
            if not hasattr(strat, "strategies"):
                # Tuple-output nodes (e.g. SDPA) have a tuple of OpStrategy;
                # skip them for now — they can't be individually pinned.
                continue
            options = []
            for spec in strat.strategies:
                label = _spec_to_str(spec.output_specs)
                if label not in options:
                    options.append(label)
            self._placement_options[node.name] = options

        # Build baseline placement map keyed by node name
        self._baseline_placements: dict[str, str] = {}
        for node, spec in self.baseline_solution.items():
            self._baseline_placements[node.name] = _spec_to_str(spec.output_specs)

    def show(self):
        """Render the explorer widget in the current notebook cell."""
        from IPython.display import HTML, display

        global _active_explorer
        _active_explorer = self

        html = self._render()
        self._display_handle = display(HTML(html), display_id=True)

    def _on_pin(self, node_name: str, placement_str: str):
        """Pin a node to a specific placement, re-solve, and update display."""
        self._error = None
        # Remove existing pin on this node
        if node_name in self.pinned:
            self.optimizer.remove_constraints(self.pinned[node_name])
            del self.pinned[node_name]

        node = self._find_node(node_name)
        # Match placement_str against strategies directly (avoids parsing
        # extended placement types like MaskP).
        strat = self.optimizer.get_strategy(node)
        matching_indices = []
        for i, spec in enumerate(strat.strategies):
            if _spec_to_str(spec.output_specs) == placement_str:
                matching_indices.append(i)
        if not matching_indices:
            self._error = f"Placement '{placement_str}' not available for '{node_name}'"
            self._update_display()
            return
        names = self.optimizer._add_node_constraint(
            node,
            output_constraint_indices=matching_indices,
            constraint_name=f"whatif_{node_name}",
        )
        self.pinned[node_name] = names
        try:
            self.current_solution = self.optimizer.resolve()
        except RuntimeError as e:
            # Infeasible — revert the constraint
            self.optimizer.remove_constraints(names)
            del self.pinned[node_name]
            self._error = f"Infeasible: pinning '{node_name}' to '{placement_str}' conflicts with other constraints"
        self._update_display()

    def _on_unpin(self, node_name: str):
        """Remove the pin on a node, re-solve, and update display."""
        if node_name not in self.pinned:
            return
        self.optimizer.remove_constraints(self.pinned.pop(node_name))
        self.current_solution = self.optimizer.resolve()
        self._update_display()

    def _on_reset(self):
        """Remove all pins, re-solve, and update display."""
        for names in self.pinned.values():
            self.optimizer.remove_constraints(names)
        self.pinned.clear()
        self.current_solution = self.optimizer.resolve()
        self._update_display()

    def _find_node(self, name: str) -> torch.fx.Node:
        for n in self.optimizer.nodes:
            if n.name == name:
                return n
        raise KeyError(f"Node {name!r} not found in optimizer")

    def _update_display(self):
        from IPython.display import HTML

        if self._display_handle is not None:
            self._display_handle.update(HTML(self._render()))

    # -- Cost helpers --

    def _cost_breakdown(self, solution) -> dict[str, float]:
        concrete = self.optimizer._to_concrete_solution(solution)
        return self.optimizer._compute_solution_cost(concrete)

    def _current_placements(self) -> dict[str, str]:
        out = {}
        for node, spec in self.current_solution.items():
            out[node.name] = _spec_to_str(spec.output_specs)
        return out

    def _changed_nodes(self) -> set[str]:
        """Node names whose placement differs between baseline and current."""
        curr = self._current_placements()
        return {
            name
            for name, baseline_pl in self._baseline_placements.items()
            if curr.get(name, baseline_pl) != baseline_pl
        }

    # -- Module tree helpers --

    def _build_module_tree(self) -> dict[str, list[str]]:
        """Build module_path -> [node_name] mapping."""
        tree: dict[str, list[str]] = defaultdict(list)
        for node in self.optimizer.nodes:
            # Try to get module_path from meta
            path = node.meta.get("module_path", "")
            if not path:
                nn_stack = node.meta.get(
                    "nn_module_stack", node.meta.get("fwd_nn_module_stack")
                )
                if nn_stack:
                    path = ".".join(name for name, _ in nn_stack.values())
            tree[path or "(root)"].append(node.name)
        return dict(tree)

    # -- HTML rendering --

    def _render(self) -> str:
        cost_base = self._cost_breakdown(self.baseline_solution)
        cost_curr = self._cost_breakdown(self.current_solution)
        curr_placements = self._current_placements()
        changed = self._changed_nodes()
        module_tree = self._build_module_tree()

        parts: list[str] = []
        parts.append(self._css())
        parts.append('<div class="ap-whatif">')
        if self._error:
            parts.append(
                f'<div style="background:#fef2f2;border:1px solid #fca5a5;'
                f"border-radius:8px;padding:10px 14px;margin-bottom:12px;"
                f'color:#991b1b;font-size:13px">{html_lib.escape(self._error)}</div>'
            )
        parts.append(self._render_header(cost_base, cost_curr, changed))
        parts.append(self._render_controls())
        parts.append(self._render_module_tree(module_tree, curr_placements, changed))
        if changed:
            parts.append(self._render_diff_panel(changed, curr_placements))
        parts.append("</div>")
        parts.append(self._js())
        return "\n".join(parts)

    def _css(self) -> str:
        return """<style>
.ap-whatif, .ap-whatif *, .ap-whatif *::before, .ap-whatif *::after {
  box-sizing: border-box; margin: 0; padding: 0;
}
.ap-whatif {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  color: #1e293b; font-size: 13px; max-width: 1100px;
}
.ap-whatif .header-card {
  display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 16px;
}
.ap-whatif .cost-card {
  background: white; border: 1px solid #e2e8f0; border-radius: 8px;
  padding: 12px 16px; flex: 1; min-width: 150px;
}
.ap-whatif .cost-label { font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }
.ap-whatif .cost-value { font-size: 20px; font-weight: 700; margin-top: 2px; }
.ap-whatif .cost-delta { font-size: 12px; margin-top: 2px; }
.ap-whatif .cost-delta.better { color: #16a34a; }
.ap-whatif .cost-delta.worse { color: #dc2626; }
.ap-whatif .cost-delta.neutral { color: #64748b; }
.ap-whatif .controls {
  display: flex; gap: 8px; align-items: center; margin-bottom: 12px; flex-wrap: wrap;
}
.ap-whatif .btn {
  padding: 5px 14px; border-radius: 6px; border: 1px solid #e2e8f0;
  background: white; cursor: pointer; font-size: 12px; font-weight: 500;
}
.ap-whatif .btn:hover { border-color: #3b82f6; color: #3b82f6; }
.ap-whatif .btn-danger { color: #dc2626; border-color: #fca5a5; }
.ap-whatif .btn-danger:hover { background: #fef2f2; }
.ap-whatif .pin-count {
  font-size: 12px; color: #64748b; margin-left: 4px;
}
.ap-whatif .search-box {
  padding: 5px 10px; border: 1px solid #e2e8f0; border-radius: 6px;
  font-size: 12px; width: 200px;
}
.ap-whatif .module-group {
  background: white; border: 1px solid #e2e8f0; border-radius: 8px;
  margin-bottom: 8px; overflow: hidden;
}
.ap-whatif .module-header {
  padding: 8px 12px; background: #f8fafc; cursor: pointer;
  display: flex; justify-content: space-between; align-items: center;
  font-weight: 600; font-size: 13px; border-bottom: 1px solid #e2e8f0;
  user-select: none;
}
.ap-whatif .module-header:hover { background: #f1f5f9; }
.ap-whatif .module-header .arrow { transition: transform 0.15s; font-size: 10px; color: #94a3b8; }
.ap-whatif .module-group.collapsed .module-body { display: none; }
.ap-whatif .module-group.collapsed .arrow { transform: rotate(-90deg); }
.ap-whatif .module-header .changed-badge {
  background: #fef3c7; color: #92400e; font-size: 10px; font-weight: 600;
  padding: 1px 6px; border-radius: 8px; margin-left: 8px;
}
.ap-whatif .node-row {
  display: flex; align-items: center; gap: 8px; padding: 4px 12px;
  border-bottom: 1px solid #f1f5f9; font-size: 12px;
}
.ap-whatif .node-row:last-child { border-bottom: none; }
.ap-whatif .node-row.changed { background: #fffbeb; }
.ap-whatif .node-row.pinned { background: #eff6ff; }
.ap-whatif .node-name {
  font-family: monospace; font-size: 11px; min-width: 180px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.ap-whatif .node-baseline {
  font-family: monospace; font-size: 11px; color: #94a3b8; min-width: 80px;
}
.ap-whatif .node-dropdown select {
  font-family: monospace; font-size: 11px; padding: 2px 4px;
  border: 1px solid #e2e8f0; border-radius: 4px; background: white;
}
.ap-whatif .node-dropdown select.pinned-select {
  border-color: #3b82f6; background: #eff6ff;
}
.ap-whatif .node-dropdown select.changed-select {
  border-color: #f59e0b; background: #fffbeb;
}
.ap-whatif .unpin-btn {
  font-size: 10px; color: #dc2626; cursor: pointer; border: none;
  background: none; padding: 2px 4px;
}
.ap-whatif .unpin-btn:hover { text-decoration: underline; }
.ap-whatif .pin-badge {
  font-size: 9px; background: #dbeafe; color: #1d4ed8; padding: 1px 5px;
  border-radius: 6px; font-weight: 600;
}
.ap-whatif .diff-panel {
  background: white; border: 1px solid #e2e8f0; border-radius: 8px;
  padding: 12px 16px; margin-top: 12px;
}
.ap-whatif .diff-panel h3 {
  font-size: 13px; font-weight: 600; margin-bottom: 8px;
}
.ap-whatif .diff-group {
  margin-bottom: 6px; font-size: 12px;
}
.ap-whatif .diff-arrow { color: #64748b; }
.ap-whatif .diff-count { font-weight: 600; color: #1e293b; }
</style>"""

    def _render_header(
        self,
        cost_base: dict[str, float],
        cost_curr: dict[str, float],
        changed: set[str],
    ) -> str:
        def _card(label: str, key: str, color: str) -> str:
            base_val = cost_base[key]
            curr_val = cost_curr[key]
            delta = curr_val - base_val
            if abs(delta) < 0.01:
                delta_cls = "neutral"
                delta_str = "no change"
            elif delta < 0:
                delta_cls = "better"
                delta_str = (
                    f"{delta:+.1f} ({delta / base_val * 100:+.1f}%)"
                    if base_val
                    else f"{delta:+.1f}"
                )
            else:
                delta_cls = "worse"
                delta_str = (
                    f"{delta:+.1f} ({delta / base_val * 100:+.1f}%)"
                    if base_val
                    else f"{delta:+.1f}"
                )
            return f"""<div class="cost-card">
  <div class="cost-label">{label}</div>
  <div class="cost-value" style="color:{color}">{curr_val:.1f}</div>
  <div class="cost-delta {delta_cls}">{delta_str}</div>
</div>"""

        n_pinned = len(self.pinned)
        n_changed = len(changed)
        summary = f"""<div class="cost-card">
  <div class="cost-label">Status</div>
  <div class="cost-value" style="font-size:16px">{n_pinned} pinned</div>
  <div class="cost-delta neutral">{n_changed} nodes changed</div>
</div>"""

        return f"""<div class="header-card">
  {_card("Total Cost", "total", "#1e293b")}
  {_card("Compute", "compute", "#10b981")}
  {_card("Communication", "comm", "#dc2626")}
  {_card("Transitions", "transition", "#f59e0b")}
  {summary}
</div>"""

    def _render_controls(self) -> str:
        disabled = ' style="opacity:0.4;pointer-events:none"' if not self.pinned else ""
        return f"""<div class="controls">
  <input class="search-box" type="text" placeholder="Search nodes..."
    oninput="apWhatifSearch(this.closest('.ap-whatif'), this.value)">
  <button class="btn btn-danger" onclick="apWhatifReset(this)"{disabled}>Reset All</button>
  <span class="pin-count">{len(self.pinned)} pin(s) active</span>
</div>"""

    def _render_module_tree(
        self,
        module_tree: dict[str, list[str]],
        curr_placements: dict[str, str],
        changed: set[str],
    ) -> str:
        parts: list[str] = []

        # Sort modules: those with changes first, then alphabetically
        def sort_key(path):
            nodes = module_tree[path]
            has_change = any(n in changed for n in nodes)
            return (0 if has_change else 1, path)

        for mod_path in sorted(module_tree, key=sort_key):
            node_names = module_tree[mod_path]
            mod_changed = sum(1 for n in node_names if n in changed)
            mod_pinned = sum(1 for n in node_names if n in self.pinned)

            # Module header
            badge = ""
            if mod_changed:
                badge = f'<span class="changed-badge">{mod_changed} changed</span>'
            if mod_pinned:
                badge += f'<span class="pin-badge" style="margin-left:4px">{mod_pinned} pinned</span>'

            collapsed_cls = "" if (mod_changed or mod_pinned) else " collapsed"
            display_path = html_lib.escape(mod_path) if mod_path else "(root)"

            parts.append(
                f'<div class="module-group{collapsed_cls}" data-module="{html_lib.escape(mod_path)}">'
            )
            parts.append(
                f'<div class="module-header" onclick="this.parentElement.classList.toggle(\'collapsed\')">'
                f'<span><span class="arrow">&#9660;</span> {display_path} '
                f'<span style="color:#94a3b8;font-weight:400">({len(node_names)} nodes)</span>'
                f"{badge}</span></div>"
            )
            parts.append('<div class="module-body">')

            for node_name in node_names:
                baseline_pl = self._baseline_placements.get(node_name, "?")
                current_pl = curr_placements.get(node_name, baseline_pl)
                is_changed = node_name in changed
                is_pinned = node_name in self.pinned
                options = self._placement_options.get(node_name, [])

                row_cls = "node-row"
                if is_pinned:
                    row_cls += " pinned"
                elif is_changed:
                    row_cls += " changed"

                select_cls = (
                    "pinned-select"
                    if is_pinned
                    else ("changed-select" if is_changed else "")
                )

                # Build options HTML
                opts_html = ""
                for opt in options:
                    selected = " selected" if opt == current_pl else ""
                    label = opt
                    if opt == baseline_pl and opt != current_pl:
                        label += " (baseline)"
                    opts_html += f'<option value="{html_lib.escape(opt)}"{selected}>{html_lib.escape(label)}</option>'

                esc_name = html_lib.escape(node_name)

                unpin_html = ""
                if is_pinned:
                    unpin_html = f'<button class="unpin-btn" onclick="apWhatifUnpin(this, \'{esc_name}\')">unpin</button>'

                pin_html = f'<span class="pin-badge">PIN</span>' if is_pinned else ""

                baseline_display = (
                    f'<span class="node-baseline">{html_lib.escape(baseline_pl)}</span>'
                    if is_changed
                    else ""
                )

                parts.append(
                    f'<div class="{row_cls}" data-node="{esc_name}">'
                    f'<span class="node-name" title="{esc_name}">{esc_name}</span>'
                    f"{baseline_display}"
                    f'<span class="node-dropdown">'
                    f'<select class="{select_cls}" onchange="apWhatifPin(this, \'{esc_name}\', this.value)">'
                    f"{opts_html}</select></span>"
                    f"{pin_html}{unpin_html}"
                    f"</div>"
                )

            parts.append("</div></div>")

        return "\n".join(parts)

    def _render_diff_panel(
        self, changed: set[str], curr_placements: dict[str, str]
    ) -> str:
        # Group changes by (old -> new)
        groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        for name in changed:
            old = self._baseline_placements.get(name, "?")
            new = curr_placements.get(name, "?")
            groups[(old, new)].append(name)

        rows = []
        for (old, new), names in sorted(groups.items(), key=lambda x: -len(x[1])):
            rows.append(
                f'<div class="diff-group">'
                f'<span class="diff-count">{len(names)}</span> nodes: '
                f"<code>{html_lib.escape(old)}</code> "
                f'<span class="diff-arrow">&rarr;</span> '
                f"<code>{html_lib.escape(new)}</code></div>"
            )

        return f"""<div class="diff-panel">
  <h3>Placement Changes ({len(changed)} nodes)</h3>
  {"".join(rows)}
</div>"""

    def _js(self) -> str:
        return """<script>
(function() {
  function exec(code) {
    if (typeof IPython !== 'undefined' && IPython.Jupyter) {
      IPython.Jupyter.kernel.execute(code);
    } else if (typeof google !== 'undefined' && google.colab) {
      google.colab.kernel.invokeFunction('notebook.execute', [code]);
    }
  }
  window.apWhatifPin = function(el, nodeName, placement) {
    exec('from autoparallel.visualizer.whatif_explorer import _active_explorer; _active_explorer._on_pin("' + nodeName + '", "' + placement + '")');
  };
  window.apWhatifUnpin = function(el, nodeName) {
    exec('from autoparallel.visualizer.whatif_explorer import _active_explorer; _active_explorer._on_unpin("' + nodeName + '")');
  };
  window.apWhatifReset = function(el) {
    exec('from autoparallel.visualizer.whatif_explorer import _active_explorer; _active_explorer._on_reset()');
  };
  window.apWhatifSearch = function(root, query) {
    var q = query.toLowerCase();
    root.querySelectorAll('.module-group').forEach(function(g) {
      var rows = g.querySelectorAll('.node-row');
      var anyVisible = false;
      rows.forEach(function(r) {
        var name = r.dataset.node || '';
        var show = !q || name.toLowerCase().indexOf(q) >= 0;
        r.style.display = show ? '' : 'none';
        if (show) anyVisible = true;
      });
      g.style.display = anyVisible ? '' : 'none';
    });
  };
})();
</script>"""
