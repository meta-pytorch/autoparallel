# Basic Concepts

This page introduces the minimum set of concepts you need to read AutoParallel
logs and use the API effectively.

## Mental model

AutoParallel works in four stages:

1. **Trace the model** into a joint forward/backward FX graph.
2. **Enumerate valid sharding strategies** for each graph node.
3. **Solve a global optimization problem** to choose one strategy per node.
4. **Apply the chosen placements** to produce a parallelized module.

A useful shorthand is:

```text
model -> graph -> candidate shardings -> optimized plan -> parallel module
```

## Device mesh

A `DeviceMesh` describes the logical arrangement of ranks that AutoParallel is
allowed to use.

Examples:

- 1D mesh: `(8,)`
- 2D mesh: `(dp, tp) = (4, 8)`

```python
mesh = torch.distributed.device_mesh.init_device_mesh(
    "cuda",
    (4, 8),
    mesh_dim_names=("dp", "tp"),
)
```

The mesh dimensions matter because placements are written in mesh order. On the
2D mesh above:

- `(Shard(0), Replicate())` means shard on batch over `dp`, replicate over `tp`
- `(Replicate(), Shard(1))` means replicate over `dp`, shard tensor dim 1 over `tp`

## DTensor placements

AutoParallel uses the same placement vocabulary as DTensor.

### `Replicate()`

Each rank holds the full tensor.

### `Shard(dim)`

The tensor is partitioned evenly along tensor dimension `dim` across the
corresponding mesh dimension.

Example on a `(dp, tp)` mesh:

```python
(Shard(0), Replicate())
```

means “shard tensor dim 0 across `dp`, but replicate across `tp`.”

### `Partial()`

Each rank holds a partial contribution that still needs a reduction to become a
fully valid value. This often appears around distributed matmuls and gradient
aggregation. In logs or debugging output, you may also see this written in a
more explicit form such as `P(sum)`.

## Global shape vs local shape

This is one of the most important distinctions for newcomers.

### During tracing

AutoParallel reasons about **global** tensors.

- `input_fn()` in the full API should return global-shaped tensors.
- `sample_inputs` in the simple API describe global inputs too, even when passed
  as DTensors constructed from local shards.

### During execution

The returned parallel module runs on **local** tensors per rank.

If your global batch is 32 and the input is sharded on batch across 4 DP ranks,
then each rank should receive local batch 8 at runtime.

## Input constraints and output constraints

These tell AutoParallel what sharding must hold at the graph boundary.

Typical first constraint on a 1D or `(dp, tp)` mesh:

```python
(Shard(0),)              # 1D mesh
(Shard(0), Replicate())  # 2D mesh
```

This says “shard the input batch on the first mesh dimension.”

Why constraints matter:

- They encode what data layout your training loop provides.
- They reduce ambiguity for the optimizer.
- They make logs much easier to interpret.

## Parameter memory constraint

Without a parameter memory constraint, the optimizer may decide that replicating
parameters everywhere is cheapest because it avoids communication.

In many training settings, that is not what you want. Adding a parameter memory
constraint forces the optimizer to consider parameter sharding.

With the full API:

```python
autop.add_parameter_memory_constraint(low=None, high=None)
```

With the simple API:

```python
parameter_memory_budget=(None, None)
```

In the current implementation, `low=None, high=None` means “use the default
bounds,” which effectively pushes parameter memory toward being divided across
ranks.

## What the optimizer is choosing

For each graph node, AutoParallel considers valid strategies such as:

- replicated inputs with replicated outputs
- sharded activations with replicated weights
- sharded weights with replicated activations
- partial outputs that later reduce

The optimizer is not choosing layer by layer independently. It solves for the
whole graph at once, so a strategy that looks locally expensive may still be
chosen if it reduces redistributions elsewhere.

## Redistribution

When one node produces a tensor in one placement and the next node expects a
different placement, AutoParallel inserts a redistribution.

Typical cases:

- `Shard -> Replicate`: all-gather
- `Partial -> Replicate`: all-reduce
- `Partial -> Shard`: reduce-scatter
- `Shard(dim_a) -> Shard(dim_b)`: all-to-all

These redistributions are a major part of the optimizer's cost model.

## Why AutoParallel sometimes makes surprising choices

A few common reasons:

- A placement reduces communication later in the graph.
- An intra-node communication is much cheaper than an inter-node one.
- The parameter memory constraint is missing, so replication looks cheap.
- The output constraint forces a layout that changes the best internal plan.
- The model contains operations for which only a limited set of placements is
  currently implemented.

If you want the full explanation of the objective and constraints, read
[How AutoParallel Chooses a Strategy](how_autoparallel_chooses_a_strategy.md).

## `auto_parallel(...)` vs `AutoParallel(...)`

### `auto_parallel(...)`

Use this when you already know:

- the sample inputs
- the desired output sharding
- whether you want a parameter memory budget

It is the easier entry point.

### `AutoParallel(...)`

Use this when you need to:

- set input/output constraints explicitly
- inspect verbose optimizer logs
- use `autop.sharding_optimizer` helpers directly
- experiment with constraints and re-solve

## Meta models and materialization

It is common to construct large models on the meta device before parallelizing:

```python
with torch.device("meta"):
    model = MyModel(...)
```

The returned parallel module must then be materialized on a real device:

```python
parallel_model.to_empty(device="cuda")
```

At that point, parameters still need real values. You must initialize them or
load a checkpoint before real training or inference.

## Compilation

The parallelized module runs eagerly by default. For better execution behavior,
you can compile it with the AutoParallel backend:

```python
from autoparallel import autoparallel_backend

parallel_model = torch.compile(
    parallel_model,
    backend=autoparallel_backend(),
)
```

This is optional for getting started, but common in real use.
