# AutoParallel Codebase Pipeline

This document is a code-oriented guide for new contributors. It explains the
main pipeline, the important modules, and how data moves from a user model to a
parallelized module.

AutoParallel is experimental and tightly coupled to PyTorch internals such as
FX, Dynamo export, AOTAutograd, DTensor, and Inductor. The best mental model is:

```text
user model
  -> fake/global tracing
  -> joint forward/backward FX graph
  -> per-node sharding strategy enumeration
  -> ILP optimization
  -> graph lowering with redistributions
  -> parallel nn.Module with sharded params/buffers
  -> optional torch.compile backend passes
```

## Public Entry Points

The public API is exported from `autoparallel/__init__.py`:

- `auto_parallel(...)`: simple wrapper for common usage.
- `AutoParallel(...)`: context-manager API for debugging and custom constraints.
- `autoparallel_backend(...)`: `torch.compile` backend wrapper for activation
  checkpointing and communication/compute overlap passes.
- `with_sharding_constraint(...)`: model-level constraint helper.

The main implementation lives in `autoparallel/api.py`.

## End-to-End Pipeline

### 1. User Defines Model, Mesh, and Example Inputs

Users provide:

- an `nn.Module`, often built on the `meta` device,
- a PyTorch `DeviceMesh`,
- example inputs,
- output placement constraints,
- optionally mixed precision and parameter memory constraints.

The simple API accepts real tensors or DTensors as `sample_inputs`. DTensor
inputs are important because their placements become input constraints. Regular
tensors are treated as replicated on every mesh dimension.

Relevant files:

- `autoparallel/api.py`
- `autoparallel/input_validation.py`
- `docs/api_walkthrough.md`
- `examples/example_autoparallel.py`
- `examples/example_hf.py`

### 2. Input Metadata Is Normalized

In `auto_parallel(...)`, sample inputs are converted into metadata:

- global shapes,
- dtypes,
- devices,
- input placement tuples,
- pytree structure.

This is handled by `_extract_input_info(...)` and `_make_input_fn(...)` in
`autoparallel/input_validation.py`.

The generated `input_fn()` creates fresh tensors with the same global metadata.
It is called later inside `FakeTensorMode`, so the tensors become fake tensors
instead of real allocations.

### 3. AutoParallel Context Setup

`AutoParallel.__init__` prepares the optimization environment:

- deep-copies the user model so tracing and dtype wrappers do not mutate it,
- canonicalizes and applies mixed precision wrappers if requested,
- moves meta parameters and buffers into fake tensors on the mesh device,
- stores the mesh, cost model, and dynamic-shape setting,
- optionally creates a `ShapeEnv` for symbolic shapes.

`AutoParallel.__enter__` then:

- configures the NCCL topology cost model,
- enters the `DeviceMesh` context,
- traces the model into a joint graph,
- disables Inductor comprehensive padding while AutoParallel is active,
- constructs a `ShardingOptimizer`.

Relevant files:

- `autoparallel/api.py`
- `autoparallel/tracing.py`
- `autoparallel/cast_parametrization.py`
- `autoparallel/cost_models/nccl_cost_model.py`
- `autoparallel/cost_models/collective_runtime_estimation.py`

### 4. Model Is Traced Into a Joint FX Graph

Tracing happens in `build_joint_graph(...)` in `autoparallel/api.py`.

The flow is:

1. Call `input_fn()` under `FakeTensorMode`.
2. Optionally convert fake inputs to symbolic dynamic inputs.
3. Capture a forward graph with Dynamo export.
4. Restore model state after capture.
5. Add unused params and buffers so they still appear in the parameter specs.
6. Use AOTAutograd to export a joint forward/backward graph.
7. Clean up and normalize the graph.
8. Optionally replace `view -> mm -> view` patterns with `einsum`.
9. Add alias nodes to expose more optimization opportunities.

The resulting graph is a single FX graph containing forward computation,
backward computation, parameter nodes, gradients, tangents, and outputs.
AutoParallel optimizes this joint graph rather than optimizing only the forward
path.

Relevant files:

- `autoparallel/api.py`
- `autoparallel/tracing.py`
- `autoparallel/graph_passes/graph_utils.py`
- `autoparallel/graph_passes/extract_forward.py`

## Sharding Strategy Generation

### 5. The Optimizer Builds Placement Options

`ShardingOptimizer` is implemented in `autoparallel/optimize_sharding.py`.

It first creates a concrete copy of the graph with symbolic dimensions replaced
by their hinted concrete values. The optimizer uses this concrete graph for
strategy enumeration, cost estimation, graph clustering, and ILP construction.
The original graph is kept for `apply_sharding`, which may still need symbolic
shape metadata.

For each tensor-producing node, `build_sharding_metadata()` creates an
`OpStrategy`. An `OpStrategy` is a list of possible `OpSpec` choices. Each
`OpSpec` describes:

- expected input DTensor specs,
- produced output DTensor specs,
- redistribution costs from predecessor placements.

Placeholders and parameters start with all valid placements generated by
`_create_all_options(...)`. Call-function nodes get strategies from
`get_placement_options_for_node(...)`.

Relevant files:

- `autoparallel/optimize_sharding.py`
- `autoparallel/shardings/placement_options.py`
- `autoparallel/shardings/propagation_rules.py`

### 6. Placement Rules Come From DTensor Plus AutoParallel Overrides

`autoparallel/shardings/placement_options.py` dispatches strategy generation.

For normal ops:

- if AutoParallel has a custom rule in `_op_rules`, it uses that,
- otherwise it asks PyTorch DTensor for an op strategy through helper wrappers.

AutoParallel adds custom rules in `autoparallel/shardings/propagation_rules.py`.
These rules cover cases where the default DTensor propagation is missing,
too strict, or not shaped for AutoParallel's optimizer.

Important examples:

- view and reshape-like ops,
- `operator.getitem`,
- pointwise behavior,
- tensor factory ops,
- matmul/einsum behavior,
- local-map and MoE-related higher-order ops,
- flex attention higher-order ops.

After strategies are generated, AutoParallel:

- propagates tensor metadata,
- fills missing redistribution costs,
- removes invalid shardings where tensor dimensions are too small for the mesh,
- deduplicates equivalent configurations,
- caches repeated placement-option lookups.

## Cost Model

### 7. Compute Cost

Compute cost is estimated in `autoparallel/cost_models/compute_estimation.py`.

The broad idea is:

- count FLOPs when possible,
- estimate memory read/write time,
- estimate compute time from device throughput,
- use the max of memory time and compute time,
- apply a small launch floor for tiny kernels,
- treat pure view-like shape operations as cheap or free.

The module contains hardware limit tables for several GPU families and a flop
counter extension for `einsum`.

### 8. Communication Cost

Communication cost is estimated in
`autoparallel/cost_models/collective_runtime_estimation.py`.

The key transition types are:

- `Shard -> Replicate`: all-gather,
- `Partial -> Replicate`: all-reduce,
- `Partial -> Shard`: reduce-scatter,
- `Shard(dim_a) -> Shard(dim_b)`: all-to-all,
- `Replicate -> Shard`: local narrowing, usually no collective.

By default, `AutoParallel.__enter__` detects an NCCL topology config and the
cost model dispatches to `autoparallel/cost_models/nccl_cost_model.py`. This is
important because intra-node and inter-node collectives have very different
costs.

Redistribution cost also includes penalties for non-contiguous layouts and
non-dim-0 shard reshuffling, because those cases need extra memory movement.

### 9. Transition Cost

The optimizer also adds a small sharding-transition penalty when a producer and
consumer use different placements. This is a tie-breaker that encourages
placement stability when communication and compute costs are otherwise similar.

## ILP Optimization

### 10. Decision Variables

The ILP is built in `ShardingOptimizer`.

A decision variable represents:

```text
(node, argument index, output strategy index, producer input strategy index)
```

Each variable has:

- total cost,
- compute cost,
- communication cost,
- transition cost,
- selected `OpSpec`,
- input and output DTensor specs.

For repeated subgraphs, graph clustering can link equivalent decision variables
so the ILP is smaller.

Relevant files:

- `autoparallel/optimize_sharding.py`
- `autoparallel/graph_passes/graph_clustering.py`

### 11. Default Constraints

The optimizer adds these constraints before solving:

- uniqueness: each node argument selects exactly one choice,
- same-output consistency: all tensor arguments of a multi-input op agree on
  one output strategy,
- flow consistency: producer output placement matches consumer input placement,
- invalid-cost constraints: impossible configurations cannot be selected,
- forward/backward consistency constraints,
- gradient-reduce dtype constraints.

User-facing constraints are layered on top:

- `add_input_constraints(...)`,
- `add_output_constraints(...)`,
- `add_parameter_memory_constraint(...)`,
- node constraints through optimizer helpers,
- model-embedded `with_sharding_constraint(...)`.

### 12. Solving

`get_solution(...)` sets the objective and solves the ILP with PuLP's CBC
solver. The objective minimizes total estimated runtime cost across the joint
graph:

```text
compute cost + communication cost + transition cost
```

The result is a mapping:

```text
FX node -> chosen OpSpec
```

Public debugging helpers include:

- `get_log(...)`,
- `print_costs_for_node(...)`,
- `explain_placement(...)`,
- `diff_solutions(...)`,
- `save(...)` and `load(...)`,
- `save_placements(...)` and `load_placements(...)`,
- `get_json(...)`.

## Applying the Solution

### 13. Lowering the Graph to Local Execution

`apply_placement(...)` calls `apply_sharding_to_model(...)` in
`autoparallel/apply_sharding.py`.

The important class is `ApplyShardingInterpreter`, an FX interpreter that walks
the original joint graph and inserts the behavior implied by the chosen
placements.

For each operation, it:

- looks up the producer specs and target input specs,
- redistributes local tensors when placements differ,
- handles `operator.getitem` specially for tuple outputs,
- localizes shape arguments for tensor factories and view ops,
- wraps view inputs in DTensor in static mode when DTensor should perform
  global-to-local shape conversion,
- executes the original op,
- converts DTensor outputs back to local tensors.

The output is a parallel FX graph that operates on local tensors and explicit
collective/redistribution behavior.

Relevant files:

- `autoparallel/apply_sharding.py`
- `autoparallel/shardings/ordered_sharding.py`

### 14. Parameters and Buffers Are Sharded

`_shard_params_and_buffers(...)` builds DTensor parameters and buffers from the
solved placements. It uses the original graph's named parameter and buffer
descriptors to map FQNs to FX nodes.

The returned dictionaries are:

```text
fqn -> sharded Parameter
fqn -> sharded buffer DTensor
```

`make_parallel_module(...)` then constructs the final module.

Relevant files:

- `autoparallel/apply_sharding.py`
- `autoparallel/module_construction.py`

### 15. Parallel Module Construction

`autoparallel/module_construction.py` creates a new module class that mirrors
the user's original model class.

It preserves:

- user-defined instance attributes,
- nested module structure,
- `ModuleDict`-like containers when possible,
- parameter aliases,
- buffer aliases,
- module aliases,
- orphan submodules needed by initialization code.

It also replaces the module's `forward` with the AutoParallel-generated
function and wraps `init_weights` if the model has one.

### 16. Runtime Forward

The generated `forward` in `AutoParallel.apply_placement(...)`:

1. Flattens user inputs.
2. Validates local runtime shapes and dtypes against traced expectations.
3. Reads DTensor parameters and buffers from the module.
4. Converts parameters and buffers to local tensors.
5. Boxes params, buffers, and runtime inputs into the AOTAutograd-compiled
   function.
6. Uses the joint forward/backward function when gradients are enabled.
7. Uses a forward-only extracted graph under `torch.no_grad()`.

The returned parallel module expects local per-rank tensors at runtime, not
global tensors.

### 17. Initialization and Loading

A common workflow is:

```python
with torch.device("meta"):
    model = MyModel(...)

parallel_model = auto_parallel(...)
parallel_model.to_empty(device="cuda")
parallel_model.init_weights()
```

`autoparallel/init_weights.py` makes typical single-GPU initialization code
work with sharded DTensor parameters. It intercepts parameter and buffer
assignments during `init_weights` and copies the assigned full tensor into the
existing DTensor placement.

Save/load support lives in:

- `autoparallel/serialization.py`
- `docs/save_load.md`

## Optional Compilation Pipeline

The eager parallel module can be passed to:

```python
torch.compile(parallel_model, backend=autoparallel_backend())
```

`autoparallel/compile.py` wraps Inductor and can enable:

- activation checkpointing joint pass,
- collective bucketing,
- overlap scheduling,
- insertion of overlap dependencies,
- prefetch limits.

Activation checkpointing logic is in:

- `autoparallel/graph_passes/activation_checkpointing.py`

Other graph and scheduling passes live under:

- `autoparallel/graph_passes/`
- `autoparallel/graph_passes/async_tp/`
- `autoparallel/graph_passes/autobucketing_inductor/`

## Important Supporting Areas

### Custom Ops and Constraints

`autoparallel/collectives.py` exposes sharding constraints and related
collective helpers. Model authors can use `with_sharding_constraint(...)` inside
model code to force an intermediate placement.

`autoparallel/ops.py` contains registered AutoParallel-specific operations.

### Local Map and MoE

AutoParallel has special handling for `local_map` and MoE-style communication.
Placement options for local-map higher-order ops are generated in
`placement_options.py`, while user-facing examples and explanations are in:

- `docs/hc_and_moe.md`
- `examples/example_local_map.py`
- `examples/example_dcp.py`
- `examples/native_ds3/`

### Dynamic Shapes

When `dynamic=True`, `AutoParallel` traces with symbolic dimensions. The
optimizer still works on a concretized graph, but `apply_sharding` preserves the
original symbolic graph and recreates local fake inputs with fresh symbols for
lowering. Runtime input validation allows dimensions marked dynamic to vary.

Relevant files:

- `autoparallel/api.py`
- `autoparallel/optimize_sharding.py`
- `autoparallel/apply_sharding.py`
- `autoparallel/input_validation.py`
- `tests/test_dynamic_shapes.py`

### JSON and Visualization

The optimizer can export strategy decisions to JSON with `get_json()`.

Relevant files:

- `autoparallel/export_json.py`
- `autoparallel/visualizer/build_display_from_json.py`
- `tests/test_export_json.py`

## Directory Map

```text
autoparallel/
  api.py                         public APIs and orchestration
  tracing.py                     fake tensor conversion and decomposition setup
  input_validation.py            sample input metadata and runtime checks
  optimize_sharding.py           ILP optimizer and debugging helpers
  apply_sharding.py              graph lowering and sharded param creation
  module_construction.py         final parallel module construction
  init_weights.py                DTensor-aware init_weights wrapper
  compile.py                     torch.compile backend wrapper
  collectives.py                 sharding constraints and collective helpers
  ops.py                         custom operator registrations
  serialization.py               optimizer and placement save/load
  export_json.py                 visualization/export format

autoparallel/shardings/
  placement_options.py           per-node strategy generation
  propagation_rules.py           custom DTensor propagation rules
  dtensor_sharding_helpers.py    wrappers around DTensor strategy APIs
  ordered_sharding.py            optimized redistribution ordering

autoparallel/cost_models/
  compute_estimation.py          operation runtime estimates
  collective_runtime_estimation.py redistribution cost estimates
  nccl_cost_model.py             NCCL topology-aware cost model

autoparallel/graph_passes/
  graph_utils.py                 graph cleanup and helper analysis
  graph_clustering.py            repeated-subgraph detection
  activation_checkpointing.py    recomputation/AC tagging and pass
  extract_forward.py             forward-only graph extraction
  auto_bucketing.py              bucketing helpers
  async_tp/                      async tensor-parallel passes
  autobucketing_inductor/        Inductor-oriented bucketing passes

docs/                            user and contributor documentation
examples/                        runnable examples
tests/                           behavior and regression tests
```

## How to Read the Code

For a first pass, read in this order:

1. `docs/basic_concepts.md`
2. `docs/api_walkthrough.md`
3. `autoparallel/api.py`
4. `autoparallel/optimize_sharding.py`
5. `autoparallel/shardings/placement_options.py`
6. `autoparallel/shardings/propagation_rules.py`
7. `autoparallel/apply_sharding.py`
8. `autoparallel/module_construction.py`
9. `autoparallel/compile.py`

Then use tests to understand edge cases:

- `tests/test_api.py`
- `tests/test_auto_parallel_simple.py`
- `tests/test_optimize_placement.py`
- `tests/test_propagation_rules.py`
- `tests/test_apply_sharding.py`
- `tests/test_dynamic_shapes.py`
- `tests/test_flex_attention.py`
- `tests/test_inference_path.py`

## Debugging Workflow

When investigating a model or optimizer decision:

1. Start with the full `AutoParallel` API instead of `auto_parallel(...)`.
2. Add explicit input and output constraints.
3. Add a parameter memory constraint if you expect FSDP-like sharding.
4. Call `optimize_placement(verbose=True)`.
5. Read the optimizer log for chosen placements and cost breakdowns.
6. Use `print_costs_for_node(...)` for a suspicious node.
7. Use `explain_placement(...)` to compare a target placement with the chosen
   placement.
8. Temporarily add a node constraint and compare with `diff_solutions(...)`.
9. Inspect the parallel graph emitted by structured logs or `parallel_gm`.

Common symptoms:

- Replicated parameters: missing or loose parameter memory constraint.
- Infeasible ILP: contradictory input/output/node constraints or shard dim too
  small for the mesh.
- Unexpected all-gather/all-reduce: producer and consumer placements disagree.
- Shape mismatch at runtime: passing global tensors to a module that expects
  local tensors.
- Dynamic-shape compile failure: check whether symbolic dims were concretized
  too early or local shape args were not localized.

## Contributor Notes

- Prefer existing DTensor strategy APIs before adding custom propagation rules.
- Add custom rules only when the default rule is missing or does not preserve
  the metadata AutoParallel needs.
- Keep optimizer constraints explicit; hidden state makes debugging ILP failures
  difficult.
- Add focused tests when touching strategy enumeration, cost modeling,
  constraints, or graph lowering.
- Be careful with aliases: parameters, buffers, and modules can share identity,
  and the code intentionally preserves those relationships.
- The traced graph uses global shapes; the returned module executes on local
  tensors. Many bugs come from mixing those two worlds.
