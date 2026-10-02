# AutoParallel Optimization Workflow

## Start from a working entry point

Reuse the model's current AutoParallel integration. Useful repository examples:

- `docs/api_walkthrough.md` for model, mesh, and input conventions;
- `examples/example_autoparallel.py` for minimal planning;
- `examples/example_llama3.py` for lowering and reordered metrics;
- `examples/notebook_explore_llama3.ipynb` and `docs/save_load.md` for offline
  exploration.

Use a real process group for execution evidence. A fake process group is useful
for capture and planning experiments, but does not show that real collectives
execute correctly.

Before capture, audit the forward with `global-spmd-modeling.md`. In particular,
code written around `batch_size_per_gpu`, local random permutations, or explicit
process-group reductions may change meaning when traced over a global logical
batch. Warn the user and propose an adaptation before proceeding when unclear.

Use a supplied `DeviceMesh`. Otherwise follow `mesh-selection.md`: confirm the
intended GPU and node allocation rather than inferring it from visible devices,
then record the inferred shape, dimension roles, and physical-link rationale.

Use the real training boundary when plan quality matters. If the supplied model
returns logits but the actual loss is outside the captured graph, state that the
plan excludes the loss and is provisional. Include the representative loss in
the captured module only when that matches the intended integration and the user
authorized the adaptation.

## Capture and optimize

The basic planning operation is:

```python
with AutoParallel(model, input_fn, mesh, mp_policy=mp_policy) as autop:
    autop.add_input_constraints(input_placements)
    autop.add_output_constraints(output_placements)

    if memory_bounds is not None:
        autop.add_parameter_memory_constraint(*memory_bounds)
    if prefetch_discount is not None:
        autop.sharding_optimizer.apply_prefetch_discount(
            scale=prefetch_discount
        )

    placement = autop.optimize_placement(verbose=True)
    optimizer_json = autop.sharding_optimizer.get_json()
    autop.sharding_optimizer.save_placements(placements_path)
    parallel_model = autop.apply_placement(placement)
```

Output constraints align with the flattened output pytree. Use `None` for
non-tensor leaves, for example `[(Shard(0),), None]` for `(logits, None)`.
Omit `add_input_constraints()` or `add_output_constraints()` entirely when the
boundary should be unconstrained. Passing `None` as the whole argument is not
an unconstrained request: the underlying node constraint interprets a tensor
placement of `None` as the default batch `Shard(0)` placement.

Omitting the parameter-memory call leaves replication legal. Calling it with no
arguments is different: it uses `low=0` and `high=1 / world_size`, forcing
sufficiently shardable parameters toward their minimum local fraction. The
current constraint averages per-tensor sharding ratios; for relaxed bounds,
compute the selected byte-weighted fraction separately rather than describing
the bound as a byte budget.

Apply a prefetch discount only as an explicit sensitivity assumption and always
record its scale. The operation mutates communication costs in place, so load a
fresh saved optimizer or recapture before comparing another scale. For a
full-shard plan, follow `full-shard-planning.md` rather than tuning the scale
until a desired placement appears.

Capture the placement, `get_json()["summary"]`, exact constraints, discount,
mesh, and communication cost model before leaving the context. Save the full
optimizer with `save()` when repeated counterfactual analysis is likely.

Repeated-subgraph clustering is optional compression of the ILP, not a reason
to discard an otherwise supported graph. If clustering asserts that two linked
nodes have different input/output strategy counts, preserve the assertion and
retry once with `repeated_subgraphs=False`. Do not use this retry for unrelated
capture or strategy failures.

`optimize_placement()` also emits `autoparallel_sharding_optimizer_log` and
`autoparallel_solution` structured trace artifacts. Set `TORCH_TRACE` to a trace
directory before the run to retain them; use `tlparse` when a parsed trace is
useful. This is optional and does not replace explicit placement or optimizer
artifacts.

## Keep one cost model through evaluation

`AutoParallel.__enter__` installs its communication topology, but `__exit__`
restores the previous global value. Compilation normally occurs afterward.
Without an outer lifetime, the ILP can therefore use the NCCL model while the
reordering pass and `estimate_graph_metrics()` silently use PyTorch's default
model.

Use the skill helper to keep the exact topology active across planning,
compilation, and metrics collection:

```python
import subprocess
import sys
from pathlib import Path

repo_root = Path(
    subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()
)
skill_scripts = repo_root / ".agents/skills/autoparallel-optimizer/scripts"
sys.path.insert(0, str(skill_scripts))
from collect_reordered_metrics import ReorderedMetricsCollector

collector = ReorderedMetricsCollector(mesh, trace_dir=None)

with collector:
    with AutoParallel(
        model,
        input_fn,
        mesh,
        mp_policy=mp_policy,
        cost_model=collector.cost_model,
    ) as autop:
        # Add the same intended constraints as the planning run.
        placement = autop.optimize_placement(verbose=True)
        optimizer_json = autop.sharding_optimizer.get_json()
        parallel_model = autop.apply_placement(placement)

    compiled = torch.compile(
        parallel_model,
        backend=autoparallel_backend(
            enable_ac=False,
            overlap_scheduling=False,
        ),
    )
    output = compiled(*local_inputs)
    model_specific_loss(output).backward()

collector.require_forward_backward()
collector.write_json("artifacts/metrics.json")
```

Resolve the checkout explicitly if the evaluation process starts outside its
Git worktree. Adapt the loss, inputs, activation-checkpointing setting, and
reordering options to the workload. Omitted reordering options inherit
`aten_autobucketing_config`; override them only intentionally.

The helper's ATen pass is the only overlap scheduler in this workflow. Keep
`overlap_scheduling=False`: Inductor otherwise runs its scheduler after the
custom post-grad pass, so the recorded metrics would describe an intermediate
graph and the graph would be scheduled twice. The collector raises if it sees
Inductor overlap scheduling enabled.

The helper detects the NCCL topology by default, passes that same object to
AutoParallel through `collector.cost_model`, records whether the result is NCCL
or a PyTorch fallback, and restores the prior global state. Pass an explicit
`NCCLTopoConfig` as `nccl_topology=` for non-standard hardware.
Do not treat a multi-node fallback result as a final deployment ranking; obtain
or construct the topology profile, or report the evaluation as provisional.

The helper refuses an already configured `post_grad_custom_post_pass`; compose a
model-specific pass explicitly instead of silently changing pass order. It
identifies backward graphs by `tangents_*` placeholders and treats an
all-`is_backward` graph as backward when no tangent placeholder is present.
Mixed partitioner tags can survive recomputation in a partitioned forward
graph, so they do not make the phase ambiguous by themselves. It records every
specialization and can emit optional forward/backward Perfetto traces with
`trace_dir=`. Metrics contain a compact placeholder count by default; pass
`include_placeholder_signatures=True` only when shapes and dtypes are needed to
diagnose specialization, because full signatures are large for parameter-heavy
graphs. Forward and backward must both execute before leaving `with collector:`
because backward compilation is lazy and the post-grad hook is restored when
the context exits.

## Explore the stable joint graph

Node names are stable while solving counterfactual placements:

```python
opt = autop.sharding_optimizer
nodes = {node.name: node for node in autop.gm.graph.nodes}
baseline = placement

constraint_names = opt.add_node_constraint(
    nodes[requested_node_name], requested_placement
)
candidate = opt.resolve()
diff = opt.diff_solutions(baseline, candidate)

opt.remove_constraints(constraint_names)
restored = opt.resolve()
```

Use `explain_placement()` for a placement family and
`print_costs_for_node()` for a redistribution matrix. Preserve their text when
it supports the answer. Explore before lowering unless the purpose is to
compare multiple candidates after scheduling.

For CPU-side exploration, capture on GPU and call
`autop.sharding_optimizer.save("model.ap")`. For ATen-only graphs,
`ShardingOptimizer.load("model.ap")` can later inspect and re-solve without
model code, a live mesh, a process group, or a GPU. Custom ops and some
`local_map` regions may require their registration modules to be importable
before loading. Treat the artifact as trusted pickle data and keep code and
PyTorch versions aligned.

## Validate and report

For execution, materialize or load initialized parameters after
`to_empty(device="cuda")`, run compiled forward and backward with local input
shapes, and check outputs and gradients for finite values. For numerical
comparison, use the same parameters, inputs, seed, dtype, and tolerances as the
unsharded reference. Call `torch.manual_seed(seed)` on every participating rank
before DTensor initialization or random DTensor operations; relying on DTensor
to synchronize rank-zero RNG state is deprecated. LocalTensorMode tests are
useful diagnosis but are not the public compiled execution path.

Typical durable artifacts are `placements.json`, `optimizer.json`,
`metrics.json`, and `optimizer.log`. Generate full optimizer state and execution
traces only when repeated exploration or timeline debugging warrants their
size.

For a full-shard or topology-inverted plan, include the confidence card from
`full-shard-planning.md` and preserve the serial and optimistic endpoint
artifacts independently.
