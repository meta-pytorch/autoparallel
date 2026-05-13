# Saving and Loading Optimizer State

AutoParallel provides three serialization APIs for different workflows:

| API | What it saves | Size | Use case |
|-----|--------------|------|----------|
| `save()` / `load()` | Full optimizer: graph, strategies, costs, constraints, solution | Large (~300MB for LLaMA-3 8B) | Offline exploration, re-solving, what-if analysis |
| `save_placements()` / `load_placements()` | Per-node placement choices (output + input specs) | Small (~100KB) | Reapplying a solution in a training script |
| `get_json()` | Rich export for visualization (nodes, edges, costs, clusters, source info) | Medium (~2MB) | Feeding the HTML visualizer |

## Workflow 1: Offline notebook exploration

Run the expensive tracing + optimization once, save the full state, then explore interactively without the model code or a process group.

```python
# === Script: trace and save ===
with AutoParallel(model, input_fn, mesh, mp_policy) as autop:
    autop.add_input_constraints([x_sharding])
    autop.add_output_constraints([out_sharding])
    autop.optimize_placement()
    autop.sharding_optimizer.save("model.ap")
```

```python
# === Notebook: load and explore ===
from autoparallel.optimize_sharding import ShardingOptimizer

opt = ShardingOptimizer.load("model.ap")

# Visualize
from autoparallel.visualizer.build_display_from_json import generate_visualization_html
from IPython.display import HTML
HTML(generate_visualization_html(opt.get_json()))

# Inspect a node
opt.print_costs_for_node(opt.nodes[42])

# What-if: constrain a node and re-solve
from torch.distributed.tensor.placement_types import Shard, Replicate
original = opt.get_solution()
names = opt.add_node_constraint(node, (Shard(0), Replicate()))
new = opt.resolve()
opt.diff_solutions(original, new)

# Revert
opt.remove_constraints(names)
opt.resolve()
```

No GPU, no model code, no process group needed in the notebook.

## Workflow 2: Reuse placements in a training script

Save the optimizer's placement choices as a lightweight JSON file, then load them in a different script to skip re-solving.

```python
# === First run: solve and save placements ===
with AutoParallel(model, input_fn, mesh, mp_policy) as autop:
    autop.add_input_constraints([x_sharding])
    autop.add_output_constraints([out_sharding])
    solution = autop.optimize_placement()
    autop.sharding_optimizer.save_placements("placements.json")
    module = autop.apply_placement(solution)
```

```python
# === Later run: load placements instead of re-solving ===
with AutoParallel(model, input_fn, mesh, mp_policy) as autop:
    autop.add_input_constraints([x_sharding])
    autop.add_output_constraints([out_sharding])
    solution = autop.sharding_optimizer.load_placements("placements.json")
    module = autop.apply_placement(solution)
```

The placements file is a small JSON with the mesh shape, dim names, and per-node output + input placement strings. `load_placements` validates that the mesh matches and finds the exact strategy by matching both output and input specs.

## Workflow 3: Generate visualization HTML

```python
with AutoParallel(model, input_fn, mesh, mp_policy) as autop:
    autop.add_input_constraints([x_sharding])
    autop.add_output_constraints([out_sharding])
    autop.optimize_placement()

    data = autop.sharding_optimizer.get_json()

from autoparallel.visualizer.build_display_from_json import generate_visualization_html
with open("viz.html", "w") as f:
    f.write(generate_visualization_html(data))
```

Or from a saved optimizer:

```python
opt = ShardingOptimizer.load("model.ap")
data = opt.get_json()
```

## Requirements and limitations

- **`save()` / `load()`**: Uses `torch.save` (pickle). Same-codebase, same-PyTorch-version only. Custom ops must be registered before loading (the loader imports `autoparallel.cast_parametrization` automatically).
- **`save_placements()` / `load_placements()`**: Plain JSON. Portable across runs, but the model graph and mesh must match (node names, mesh shape, mesh dim names are all validated on load).
- **`get_json()`**: Requires a solved optimizer (call `get_solution()` or `optimize_placement()` first). The output is a Python dict, not written to disk — serialize it yourself if needed.
