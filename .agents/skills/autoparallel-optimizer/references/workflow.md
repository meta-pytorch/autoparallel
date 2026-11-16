# AutoParallel Optimization Workflow

## Start from a working entry point

Reuse the model's current AutoParallel integration. Useful repository examples:

- `docs/api_walkthrough.md` for model, mesh, and input conventions;
- `examples/example_autoparallel.py` for minimal planning;
- `examples/example_llama3.py` for lowering and reordered metrics;
- `examples/notebook_explore_llama3.ipynb` and `docs/save_load.md` for offline
  exploration.

Construct the model under `with torch.device("meta"):`. Do not convert a
materialized model with `.to("meta")` or `.to_empty(device="meta")` before
capture: `nn.Module._apply` replaces each parameter independently and can
silently untie shared weights such as tied embeddings and decoders. Have
`input_fn` create global inputs on the mesh device type, not on meta;
AutoParallel calls it under its fake mode. After `apply_placement`, materialize
the parallel model with `to_empty` and initialize it or load a checkpoint.

A fake process group is useful for capture and planning, but execution evidence
needs a real one.

Capture the real training boundary when plan quality matters. If the actual
loss lies outside the captured graph, say the plan excludes it and is
provisional; include a representative loss only when that matches the intended
integration and the user authorized it.

## Capture and optimize

The basic planning operation is:

```python
with AutoParallel(model, input_fn, mesh, mp_policy=mp_policy) as autop:
    autop.add_input_constraints(input_placements)
    autop.add_output_constraints(output_placements)
    if memory_bounds is not None:
        autop.add_parameter_memory_constraint(*memory_bounds)

    placement = autop.optimize_placement(verbose=True)
    optimizer_json = autop.sharding_optimizer.get_json()
    autop.sharding_optimizer.save_placements(placements_path)
    parallel_model = autop.apply_placement(placement)
```

Boundary constraints align with the flattened input or output pytree. Use
`None` for non-tensor leaves, for example `[(Shard(0),), None]` for
`(logits, None)`. For a tensor leaf, `None` keeps the default batch `Shard(0)`;
use `UNCONSTRAINED`, imported from `autoparallel`, to let the solver choose, and
omit the whole call only when every boundary tensor is unconstrained. Each
boundary can be constrained only once per capture.

Omitting the parameter-memory call leaves replication legal. Calling it with no
arguments is different: it uses `low=0` and `high=1 / world_size`, and the
resulting budget pins every parameter to a minimum-local-bytes placement.
Bounds are normalized fractions of byte-weighted persistent parameter storage,
counting aliased storage once and excluding parameters with node constraints.
Record the achieved values from `get_json()["summary"]["parameter_storage"]`.

For searches that may grow large, pass `solver_time_limit_seconds` to
`AutoParallel`. When CBC reaches the limit without proving optimality, the
solve raises `TimeoutError` rather than returning the incumbent. The limit
covers the PuLP solve, not graph capture or ILP construction in `__enter__`,
so keep an outer wall-clock budget as well. An infeasible solve logs the
violated constraints before raising.

For a prefetch discount, follow `full-shard-planning.md`.

Repeated-subgraph clustering is optional compression of the ILP. AutoParallel
skips malformed groups atomically and lists them in the JSON clustering
summary; report them because the solve used less compression. Retry with
`repeated_subgraphs=False` only for comparison or on an older checkout that
still aborts.

`optimize_placement()` emits `autoparallel_sharding_optimizer_log` and
`autoparallel_solution` structured trace artifacts; set `TORCH_TRACE` before
the run to keep them.

## Keep one cost model through evaluation

`AutoParallel.__enter__` installs its communication topology, but `__exit__`
restores the previous global value, and compilation normally happens
afterward. Without an outer lifetime, the ILP can use the NCCL model while the
reordering pass and `estimate_graph_metrics()` silently use PyTorch's default.
The skill helper keeps one topology active across planning, compilation, and
metrics collection:

```python
import subprocess
import sys

repo_root = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], text=True
).strip()
sys.path.insert(0, f"{repo_root}/.agents/skills/autoparallel-optimizer/scripts")
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

Adapt the loss, inputs, and reordering options to the workload. Omitted
reordering options inherit `aten_autobucketing_config`. Forward and backward
must both run inside `with collector:`: backward compiles lazily, and the
helper's post-grad hook is removed when the context exits. The helper refuses
an already configured `post_grad_custom_post_pass`, records every
specialization recompile, and writes forward/backward Perfetto traces when
given `trace_dir=`.

The helper's ATen pass must be the only overlap scheduler. Keep
`overlap_scheduling=False`: Inductor otherwise schedules again after the custom
post-grad pass, so the metrics would describe an intermediate graph, and the
collector raises if it sees Inductor overlap scheduling enabled. The pass
inserts no overlap control dependencies, so later Inductor scheduling or fusion
can still move the order that the metrics describe.

Do not infer realized activation checkpointing from `enable_ac=True`. The
`torch.compile` partition replays the save decisions made inside
`apply_placement`: it ignores the backend AC pass's `PREFER_RECOMPUTE` tags and
honors only its `MUST_SAVE` tags, so the saved set may not change. Apply the
model's own checkpoint policy before capture, and verify the save/recompute
decisions in the partitioned graphs or their memory effect.

The helper detects the NCCL topology by default and records whether it fell
back to PyTorch's model. Detection returns `None` without CUDA, for an
unrecognized GPU, or for a mesh size that is not divisible by the GPU's
canonical ranks per node, such as a partial-node sub-allocation. Pass an
explicit `NCCLTopoConfig` as `nccl_topology=` in those cases, describing the
allocated communicator rather than the host's unused GPUs. A multi-node result
from the fallback model is provisional.

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

Use `explain_placement()` for a placement family and `print_costs_for_node()`
for a redistribution matrix, and quote their output when it supports the
answer. Explore before lowering unless the purpose is to compare candidates
after scheduling.

For repeated or CPU-side exploration, capture on GPU and call
`autop.sharding_optimizer.save("model.ap")`. For ATen-only graphs,
`ShardingOptimizer.load("model.ap")` can later inspect and re-solve without
model code, a live mesh, a process group, or a GPU. `get_json()`,
`print_costs_for_node()`, and the constraint/resolve/diff loop work on a loaded
optimizer; `explain_placement()` currently needs a live one. Custom ops and some
`local_map` regions may require their registration modules to be importable
before loading. Treat the artifact as trusted pickle data and keep code and
PyTorch versions aligned.

## Validate

Run compiled forward and backward with local input shapes and initialized
parameters, and check outputs and gradients for finite values. For a numerical
comparison, use the same parameters, inputs, seed, dtype, and tolerances as the
unsharded reference. Call `torch.manual_seed(seed)` on every rank before
DTensor initialization or random DTensor operations; relying on DTensor to
synchronize rank-zero RNG state is deprecated. LocalTensorMode tests help
diagnosis but are not the public compiled execution path.

Typical artifacts are `placements.json`, `optimizer.json`, `metrics.json`, and
`optimizer.log`. Save full optimizer state and execution traces only when
repeated exploration or timeline debugging warrants their size.
