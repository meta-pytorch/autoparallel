# AutoParallel as an Agent-Facing Parallelism Planner

## Executive summary

Coding agents weaken one part of AutoParallel's original pitch: writing an
explicit FSDP or tensor-parallel recipe is becoming cheaper. For established
transformers, an agent can often adapt a TorchTitan or Megatron-style plan
competently.

That does not eliminate the harder problem. Distributed execution is a global,
hardware-dependent optimization problem over the derived forward and backward
graph. A generated recipe is not evidence that the recipe is feasible,
correct, or fast for a particular workload and cluster.

AutoParallel should therefore be repositioned as an **explainable parallelism
planner, plan evaluator, and verification backend**:

```text
                         ┌───────────────────────┐
model + workload ───────▶│                       │──────▶ executable plan
topology + objectives ──▶│ AutoParallel backend │──────▶ cost and coverage
candidate plan ─────────▶│                       │──────▶ verification evidence
                         └───────────────────────┘
```

The durable promise is not “no manual parallelism code.” It is:

> AutoParallel searches or evaluates the distributed design space and returns
> a reproducible plan together with the evidence needed to decide whether to
> trust it.

Agents become clients of that backend. They can propose model code, boundary
layouts, constraints, or complete candidate plans; AutoParallel supplies
graph-level analysis, cost estimation, consistency checks, and measured
feedback.

## Concede what agents commoditize

The README's “no manual parallelism code required” message is genuinely less
differentiated in an agent-driven workflow. For a known architecture, explicit
parallelism code is mechanical, widely exemplified, and increasingly easy to
generate.

This matters most for fixed models on fixed clusters. A maintained hand-written
recipe can be simple, predictable, and well benchmarked. AutoParallel should
not argue that agents are incapable of producing such code.

Its stronger argument is that code generation and execution planning are
different jobs:

- An agent can emit a plausible plan; AutoParallel can compare it with other
  legal plans.
- An agent can reuse a known recipe; AutoParallel can re-evaluate it for a new
  batch size, sequence length, mesh, or topology.
- An agent can repair model code; AutoParallel can expose the graph regions and
  constraints responsible for a failure.
- An agent can launch experiments; AutoParallel can reduce how many expensive
  distributed experiments are necessary.

## Why the backend remains valuable

### The optimization object is a derived graph

AutoParallel plans over the joint forward and backward graph produced through
Dynamo and AOTAutograd. Backward operations, saved tensors, gradient
reductions, and many redistribution opportunities are not explicitly present
in the original `nn.Module` source.

Those properties can sometimes be derived manually, but doing so across an
arbitrary exported graph is difficult and brittle. The compiler graph makes
them explicit and allows consistency constraints and costs to be applied
uniformly.

The sequence-parallel versus column-parallel examples in
[`adaptive_sharding.md`](adaptive_sharding.md) illustrate the benefit. The
choice depends not only on a projection in isolation, but also on gradient
placements and the layout expected by residual consumers.

### The answer is a function of the deployment

A parallelization recipe is an artifact. What users need is closer to a
function:

```text
plan = f(graph, shapes, topology, memory, objective, compiler behavior)
```

Changing batch size, sequence length, mesh dimensions, GPU generation, or
inter-node bandwidth can change the preferred strategy without changing the
model source. This remains true even if model architectures become stable.

### Distributed feedback is expensive

Kernel-generation agents benefit from a tight compile-run-profile loop.
Distributed plans are more expensive to evaluate: they require multi-process
or multi-node launches, have noisier performance, and can fail only at larger
scales.

Agents do not make this feedback free. A planner should use structural
constraints and calibrated cost models to prune the search space, then reserve
real cluster time for a small number of high-value measurements.

### The cost model is an asset, but not an oracle

AutoParallel combines analytical compute estimates, logic derived from NCCL,
measured communication data, interpolation, and extrapolation. That is a
valuable body of systems knowledge that cannot be recreated reliably by
language-model reasoning alone.

It must nevertheless be presented honestly:

- The default NCCL path has detailed collective modeling, but some hardware
  configurations remain extrapolated.
- The generic fallback AllToAll estimator contains a `5x` heuristic; this does
  not apply when the NCCL topology path handles the collective, but it matters
  when the fallback is selected.
- Blackwell contains values scaled or estimated from Hopper that still require
  direct calibration.
- Prefetch overlap is optional and currently represented as a caller-selected
  discount rather than a measured overlap prediction.
- Compute estimates use analytical efficiency assumptions that need validation
  against real kernels.

Cost errors matter both when ranking plans and when returning numbers to a
caller. They do not harmlessly cancel inside the solver: relative errors across
operators or collective types can change the selected plan.

## What the repository already provides

Most of the technical foundation exists:

| Capability | Existing implementation | Product gap |
|---|---|---|
| Joint forward/backward capture | `api.py` | no stable graph artifact contract |
| Global placement search | `optimize_sharding.py` | solver internals exposed through a stateful protocol |
| Upstream DTensor strategies | `shardings/` | fallback and coverage are not first-class results |
| Communication modeling | `cost_models/` | incomplete provenance and calibration reporting |
| Executable lowering | `apply_sharding.py` | no independently verifiable plan package |
| Runtime and memory simulation | `estimate_graph_metrics.py` | not connected to a public plan evaluator |
| Optimizer persistence | `save`, `load`, placement JSON | not a versioned public artifact |
| Counterfactual exploration | constraints, `resolve`, solution diffs | undocumented as a supported session |
| Placement explanations | logs, JSON, `explain_placement` | primarily stdout and trace artifacts |
| Numerical simulation | `LocalTensorMode` correctness tests | not available as a supported verifier |

The highest-leverage change is to connect these pieces into an API that can
score both solver-generated and caller-proposed plans.

## Product shape: an oracle and a planner

AutoParallel should expose two complementary modes.

### `score_plan`: evaluate a candidate

An agent or human may already have a plan. AutoParallel should validate and
score it rather than insist on replacing it.

```python
report = score_plan(
    model,
    workload,
    topology,
    plan=candidate,
    objectives=objectives,
)

report.valid
report.coverage
report.estimated.step_time_us
report.estimated.exposed_communication_us
report.estimated.peak_memory_bytes
report.confidence
report.assumptions
```

This is not merely a wrapper around `load_placements` and
`estimate_graph_metrics`. A credible implementation must:

1. Match the candidate to the captured graph and strategy-space version.
2. Reject placements that are not representable or legal.
3. Lower the plan into a parallel graph.
4. Validate placement and collective consistency.
5. Estimate the scheduled graph with a documented runtime estimator.
6. Account separately for parameters, activations, optimizer state, and
   temporary collective buffers where supported.
7. Report any opaque or manually costed regions.

Initially, externally supplied plans can be restricted to strategies already
representable by AutoParallel. Supporting arbitrary distributed programs is a
larger problem and should not be implied by the API.

### `plan`: search and return candidates

The planner should return the best cost-model plan plus alternatives rather
than a single unexplained answer:

```python
result = plan(model, workload, topology, objectives)

result.selected
result.alternatives
result.coverage
result.explanation
result.optimizer_session
```

The selected plan should be evaluated through the same `score_plan` path used
for caller-proposed plans. That prevents the solver from receiving privileged,
less rigorous treatment.

### Preserve the exploration session

A stateless call is the right default for agents, but the existing stateful
workflow is valuable. Adding a constraint, re-solving, removing it, diffing
solutions, and explaining a placement is already an effective
propose-evaluate-revise loop.

Expose it as a stable, addressable planning session. Do not force every
counterfactual to retrace and rebuild the ILP.

## A reproducible artifact contract

The durable interface should be based on versioned artifacts. Local Python
wrappers may accept an `nn.Module`, but a service boundary should not rely on a
mutable “model reference” that may execute arbitrary code or resolve
differently across environments.

### Planning request

```json
{
  "schema_version": 1,
  "model": {
    "artifact": "model.exported_program.pt2",
    "content_hash": "sha256:...",
    "pytorch_version": "..."
  },
  "workload": {
    "mode": "training",
    "inputs": [
      {"shape": [1024, 4096], "dtype": "bfloat16", "layout": ["S0", "R"]}
    ],
    "dynamic_dimensions": {"input.0": [0]}
  },
  "topology": {
    "mesh": [16, 8],
    "mesh_names": ["dp", "tp"],
    "profile": "h100_nvswitch_400g_v1"
  },
  "objectives": {
    "mode": "throughput",
    "parameter_memory_bytes": 40000000000,
    "activation_memory_bytes": 20000000000
  },
  "constraints": []
}
```

### Plan response

```json
{
  "schema_version": 1,
  "status": "planned",
  "plan_id": "sha256:...",
  "planner_version": "...",
  "strategy_space_version": "...",
  "cost_model_version": "...",
  "estimated": {
    "step_time_us": 0.0,
    "compute_us": 0.0,
    "communication_us": 0.0,
    "exposed_communication_us": 0.0,
    "peak_memory_bytes": 0
  },
  "coverage": {
    "direct": 0,
    "decomposed": 0,
    "replicate_only": 0,
    "opaque": 0,
    "unsupported": 0
  },
  "confidence": {
    "level": "medium",
    "reasons": []
  },
  "verification": {
    "highest_completed_tier": "structural"
  },
  "artifacts": {},
  "warnings": []
}
```

For a fixed model artifact, workload, topology, constraints, and planner
version, the result should be deterministic, cacheable, and comparable.

## Required capabilities

### 1. Calibration provenance and confidence

Every returned cost should carry provenance such as:

- `measured`
- `interpolated`
- `extrapolated`
- `analytical`
- `fallback`

Confidence should be derived from the operations and collectives on the
critical path, not from a single global label. One poorly calibrated dominant
AllToAll should lower confidence more than many insignificant analytical
pointwise costs.

The first calibration work should include:

- replacing or containing the generic fallback AllToAll heuristic;
- measuring Blackwell AllToAll directly;
- comparing compute estimates with representative generated kernels;
- modeling observed prefetch and overlap instead of assuming either zero or
  complete hiding;
- recording the exact topology and software stack used for measurements.

### 2. Structured coverage analysis

Replace `supports(model) -> bool` with an `analyze` report that classifies
every relevant region as:

- directly supported;
- supported through decomposition;
- replicate-only fallback;
- opaque or manually specified;
- unsupported or untraceable.

The report should identify trace breaks, operator and module paths, relevant
shapes, candidate counts, and whether the current memory constraints appear
feasible. Suggestions such as adding `local_map` should be presented as
possible repairs, not asserted automatically when the tool cannot infer the
intended distributed semantics.

### 3. Machine-readable failures

Expected failures should have stable codes and structured context:

- `TRACE_FAILED`
- `UNSUPPORTED_OP`
- `REPLICATE_ONLY_REGION`
- `INFEASIBLE_CONSTRAINTS`
- `MEMORY_OVER_BUDGET`
- `TOPOLOGY_PROFILE_MISSING`
- `INPUT_SHAPE_MISMATCH`
- `MESH_MISMATCH`
- `PYTORCH_VERSION_UNSUPPORTED`

Include the FX node, operator, module path, source location, shapes, and
constraint identifiers where available. Repair hints should be derived from
known facts. Diagnosing a minimal contradictory subset of ILP constraints may
require additional solver work and should not be implied by a generic
infeasibility exception.

### 4. Explanations as return values

Promote `get_json`, optimizer serialization, `explain_placement`, cost tables,
solution diffs, and the visualizer output into supported data products. An
explanation should answer:

- Why was this placement selected?
- Which alternatives were legal?
- Which constraints eliminated an alternative?
- Which graph edges dominate communication?
- Which tensors dominate memory?
- What changes under a different mesh or budget?
- Where did the estimate rely on extrapolation or fallback behavior?

### 5. Tiered verification

`verify_plan` should report distinct claims rather than a single pass/fail bit:

1. **Structural verification:** graph validity, shape propagation, legal
   placements, boundary contracts, and collective symmetry.
2. **Simulated numerical verification:** forward and backward comparison using
   real values across ranks simulated with `LocalTensorMode`.
3. **Small-cluster verification:** real collectives, forward and gradient
   parity, checkpoint round trip, repeated execution, and compilation checks.
4. **Target validation:** measured memory, throughput, compilation behavior,
   and comparison with baselines on the intended topology.

Fake tensors support the first tier, not numerical equivalence. Existing
`LocalTensorMode` correctness tests provide a foundation for the second tier,
although upstream integration issues currently prevent using the full
`apply_placement()` path directly.

Cost-model calibration gates trustworthy performance scoring; it does not
block structural or numerical verification. These workstreams can progress in
parallel.

### 6. Profile-guided refinement

Use cluster time for a bounded feedback loop:

1. Produce a small, diverse set of feasible plans.
2. Benchmark representative operations or short training steps.
3. Update a topology-specific calibration profile.
4. Re-score and, when appropriate, re-solve.
5. Preserve predicted and measured results in the plan artifact.

This lets an agent explore without turning expensive distributed execution
into unconstrained trial and error.

## Scope boundaries

### No pipeline parallelism today

The current optimizer handles intra-stage placement decisions; micro-pipelined
tensor-parallel code is not model pipeline parallelism. At frontier scale,
pipeline partitioning and scheduling are often necessary. The product should
state this limitation and eventually compose with or optimize pipeline stages.

### `local_map` is an intentional boundary with incomplete visibility

`local_map` is a legitimate compiler escape hatch for data-dependent layouts
that static DTensor placements cannot express. The problem is not that the
boundary exists; it is that the planner cannot currently evaluate everything
inside it.

Require an optional cost and memory contract for opaque regions so
`score_plan` can include them. Over time, richer layout representations and
communication models can reduce the size of these opaque regions.

### Replication fallback must never be silent

Falling back to replication may preserve correctness while destroying the
quality of a plan. Coverage reports and plan artifacts must identify every
replicate-only region, its estimated cost, and whether strict mode would reject
it.

### Compatibility is part of correctness

Because AutoParallel relies on private and evolving PyTorch APIs, every plan
must record the PyTorch build and relevant compiler configuration. The project
needs a tested compatibility matrix and narrow integration adapters that
isolate upstream churn.

## Benchmark and trust program

Agent ergonomics are secondary if the selected plans are not demonstrably
competitive. Establish a public benchmark matrix covering:

- dense decoder, encoder-decoder, FlexAttention, and MoE workloads;
- multiple parameter scales, sequence lengths, and batch sizes;
- 1D and 2D meshes, followed by pipeline and richer hybrid configurations;
- multiple GPU generations and single- and multi-node systems;
- throughput, exposed communication, peak memory, compile time, and planning
  time;
- numerical parity and distributed checkpoint compatibility;
- strong hand-written FSDP, TP, and hybrid baselines.

Report wins, ties, and losses. Classify each loss as a strategy-space gap,
cost-model error, compiler limitation, runtime overhead, or unsupported
communication pattern. Use the same measurements to calibrate the confidence
returned by `score_plan`.

## Delivery sequence

The work has parallel tracks rather than one strict dependency chain:

### Track A: inspectable planning

1. Version the request, plan, topology, and diagnostic schemas.
2. Expose existing JSON, serialization, explanation, and diff functionality.
3. Add structured coverage reporting and explicit replication fallbacks.
4. Preserve the stateful exploration session behind a stable API.

### Track B: trustworthy evaluation

1. Build the benchmark and calibration harness.
2. Attach per-cost provenance and confidence.
3. Ship `score_plan` for representable plans with clearly labeled estimates.
4. Add profile-guided candidate evaluation.

### Track C: verification

1. Ship structural verification.
2. Promote `LocalTensorMode` numerical comparison into a supported tool.
3. Add small-cluster execution and checkpoint verification.
4. Add target-topology validation artifacts.

### Track D: broader and more stable optimization

1. Add activation, optimizer-state, and temporary-buffer memory objectives.
2. Compose with or incorporate pipeline parallelism.
3. Improve context and expert-parallel modeling.
4. Add cost contracts for `local_map` and other opaque regions.
5. Isolate private PyTorch dependencies and upstream reusable functionality.

Machine-readable errors and basic analysis reports benefit existing human
users and can ship opportunistically across these tracks.

## Success criteria

Evaluate the project by outcomes, not by lines of parallelism code removed:

- A broad model corpus receives useful coverage reports even when planning
  fails.
- Unsupported, opaque, extrapolated, and replicate-only regions are never
  silent.
- Structural verification catches invalid plans before cluster launch.
- Numerical verification matches the unsharded model within declared
  tolerances.
- Estimated plan rankings correlate with measured rankings on supported
  topologies.
- Solver-generated plans are scored through the same path as external plans.
- Selected plans are competitive with strong hand-written baselines.
- Re-planning for a new topology is cheaper than authoring and validating a
  new recipe.
- Agents can diagnose common failures and explain plan tradeoffs using only
  structured outputs.
- Plan artifacts remain reproducible across the declared compatibility window.

## Strategic conclusion

Agents reduce the cost of authoring distributed code, so AutoParallel should
not build its identity around eliminating that code. Its defensible role is to
provide trustworthy search and evaluation over a large, changing,
hardware-dependent design space.

The intended division of labor is:

| Agent | AutoParallel |
|---|---|
| Interpret goals and deployment constraints | Capture and analyze the derived graph |
| Produce traceable model semantics | Enumerate legal distributed strategies |
| Propose constraints or complete plans | Search, score, and compare plans |
| Add explicit custom regions where necessary | Validate boundaries and include declared costs |
| Run selected experiments | Calibrate predictions from measurements |
| Explain decisions to the user | Return reproducible evidence and diagnostics |

The long-term moat is not code generation. It is trusted planning,
measurement, and verification beneath generated distributed programs.
