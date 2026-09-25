# AutoParallel in an Agent-Driven World

## Executive summary

Coding agents are making explicit FSDP, tensor-parallel, and checkpointing
code cheaper to write. That erodes AutoParallel's value as a convenience
layer. It does not eliminate the planning problem underneath, and it increases
the rate at which new model variants outgrow established parallelism recipes.

The durable promise is not “no one has to write parallelism code.” It is:

> AutoParallel searches or evaluates the legal strategy space, prices the
> tradeoffs against a versioned hardware model, produces an executable plan,
> and supplies evidence about whether the plan should be trusted.

```text
                         ┌───────────────────────┐
model + workload ───────▶│                       │──────▶ executable plan
topology + objectives ──▶│ AutoParallel backend │──────▶ costs and alternatives
candidate plan ─────────▶│                       │──────▶ verification evidence
                         └───────────────────────┘
```

This is a repositioning from convenience automation to **planning and
evidence**. Agents and humans can propose model code, constraints, or complete
candidate plans. AutoParallel should analyze the derived graph, search or
score the representable alternatives, validate their consistency, and return
reproducible artifacts.

The immediate product is an agent-consumable plan evaluator and planner. The
long-term moat is trusted search, measurement, and verification across a
large, changing, hardware-dependent distributed design space.

## Concede what agents commoditize

The README's “no manual parallelism code required” message is genuinely less
differentiated in an agent-driven workflow.

A TorchTitan-style plan for a standard transformer is mechanical, widely
exemplified, and increasingly easy for an agent to generate. For fixed models
on fixed clusters, a maintained hand-written recipe can be simple,
predictable, and well benchmarked. AutoParallel should not argue that agents
are incapable of producing such code.

The stronger distinction is between **authoring distributed code** and
**deciding whether a distributed plan is good**:

- An agent can emit a plausible plan; AutoParallel can compare it with other
  legal plans.
- An agent can reuse a familiar recipe; AutoParallel can re-evaluate it for a
  different shape, mesh, or topology.
- An agent can repair model code; AutoParallel can identify the graph region
  and constraint responsible for a failure.
- An agent can launch experiments; AutoParallel can reduce the number of
  expensive distributed experiments required.

Stating this concession first makes the remaining case more credible.

## Why the planner remains valuable

### Kernel agents are stack consumers

PTX, CuTe, and Triton agents work because they have a tight loop: generate,
compile, run, profile, and compare against numeric and throughput ground
truth. They do not replace compilers, assemblers, profilers, or correctness
checks; they consume them.

Distributed planning lacks the same cheap feedback loop. A candidate can
require a multi-process or multi-node launch, performance is noisier, and some
failures appear only at scale. Agents reduce the cost of producing candidates,
not the cost of evaluating every candidate on a cluster.

A planner should use graph constraints and calibrated models to prune the
space, then reserve real hardware for a small set of high-value measurements.

### The optimization object is a derived graph

AutoParallel operates on the joint forward and backward graph produced
through Dynamo and AOTAutograd. Backward operations, saved tensors, gradient
reductions, and many redistribution opportunities are not explicitly
represented in the original `nn.Module` source.

Some of these properties can be derived manually, but doing so across an
arbitrary exported graph is difficult and brittle. The compiler graph makes
them explicit and allows placement consistency, memory, and communication
costs to be handled uniformly.

The sequence-parallel versus column-parallel examples in
[`adaptive_sharding.md`](adaptive_sharding.md) illustrate the distinction.
The selected strategy depends not only on an individual projection, but also
on gradient layouts and the placement expected by residual consumers.

### The answer is a function of the deployment

A parallelism recipe is an artifact. The required behavior is closer to a
function:

```text
plan = f(graph, shapes, topology, memory, objective, compiler behavior)
```

Changing batch size, sequence length, mesh dimensions, node count, GPU
generation, or inter-node bandwidth can change the preferred plan without
changing the model source. This remains true even if architecture churn slows.

### The cost model is a systems asset, not an oracle

AutoParallel combines:

- analytical compute and memory-traffic estimates;
- algorithm and protocol logic derived from NCCL;
- measured communication data;
- interpolation within measured ranges;
- extrapolation to configurations without direct measurements;
- fallback estimators for unrecognized topologies.

That body of systems knowledge is valuable and cannot be recreated reliably
through language-model reasoning alone. It must nevertheless be exposed with
its limitations:

- The generic fallback AllToAll estimator contains a `5x` heuristic. This is
  not used when the configured NCCL path handles AllToAll, but it matters when
  the fallback path is selected.
- Some Blackwell values are estimated from Hopper and require direct
  calibration.
- Compute estimates use analytical efficiency assumptions.
- Prefetch overlap is represented as a caller-selected discount rather than a
  prediction derived from a measured or scheduled overlap model.

These inaccuracies matter inside the solver as well as at the API boundary.
Relative cost errors can change which plan is selected; they do not harmlessly
cancel. Returned numbers must identify how they were produced, and unreliable
paths must be rejectable in strict mode.

## What exists today

Support and maturity are not binary. The current foundation is substantial,
but its claims have different levels of evidence.

| Capability | Current state | Important qualification |
|---|---|---|
| Joint forward/backward capture | Core and exercised in CI | tied to evolving PyTorch internals |
| DTensor strategy enumeration | Core | coverage depends on upstream rules and local overrides |
| Global ILP | Core | optimal only within candidates and modeled costs |
| Executable lowering | Core | not packaged as an independently verifiable plan artifact |
| `torch.compile` backend | Implemented | activation checkpointing and overlap use private compiler hooks |
| NCCL cost model | Strongest coverage on H100 NVSwitch | coverage varies by collective, size, topology, and architecture |
| Numerical correctness simulation | Tested with `LocalTensorMode` | full `apply_placement()` path is blocked by upstream issues |
| Distributed checkpointing | Tested | real multi-GPU coverage is narrower than simulated coverage |
| Dynamic batches, inference, mixed precision | Tested | not equivalent to arbitrary dynamic programs |
| FlexAttention | Tested | auxiliary layouts have explicit restrictions |
| `local_map` / MoE | Supported as an explicit boundary | internal communication is not globally optimized or fully priced |
| TorchTitan integration | LLaMA integration in CI | DeepSeek/MoE integration is less mature |
| Pipeline parallelism | Not supported | micro-pipelined TP is not model pipeline parallelism |

Plan JSON, trace artifacts, optimizer serialization, re-solving, placement
explanations, and solution diffs already exist. They are fragmented across
internal methods, logs, stdout, and tracing artifacts. The public package does
not expose a coherent way to score a plan.

These capabilities should become the center of the interface.

## Claims the product must make precisely

### Say “cost-model-optimal”

The ILP returns an exact optimum over the candidate strategies and costs it is
given. It does not prove that the plan is fastest on real hardware.

Every result should identify:

- the strategy-space version;
- the cost-model and calibration-profile versions;
- the topology profile;
- the workload shapes and dynamic dimensions;
- all constraints and boundary layouts;
- replication fallbacks and opaque regions;
- whether important costs are measured, interpolated, extrapolated,
  analytical, or fallback estimates.

### Support is a taxonomy

A graph region may be:

- directly supported;
- supported through decomposition;
- representable only with replication;
- opaque or manually specified;
- unsupported or untraceable.

Replicate-only is particularly dangerous because it may yield a correct,
working, quietly slow plan. It must be visible in every response and
configurable as fatal.

### Verification claims must name their substrate

Fake tensors validate graph construction and shape propagation; they do not
establish numerical equivalence. Simulated ranks exercise real values but not
real network behavior. A small real mesh catches different failures from the
target topology.

No single `verified: true` field can represent all of these claims.

### The caller still supplies policy

The planner needs representative inputs, a topology, objectives, and some
boundary constraints. Dynamic MoE regions may still require explicit local
semantics. Agents can help provide this policy, but defaults and assumptions
must be returned rather than silently applied.

### Compatibility is part of correctness

Because AutoParallel relies on private and evolving PyTorch APIs, plan
artifacts must record the PyTorch build and relevant compiler configuration.
The project needs a tested compatibility matrix and clear failure codes when
the environment is unsupported.

## Product shape: evaluate and search

AutoParallel should expose two complementary operations that share the same
validation and scoring implementation.

### `score_plan`: evaluate a candidate

An agent or human may already have a plan. AutoParallel should evaluate it
rather than require the caller to abandon it.

```python
report = score_plan(
    artifact,
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
report.verification
report.assumptions
```

Initially, this API should accept only plans expressible through the captured
graph and AutoParallel's candidate strategy space. Supporting arbitrary
distributed Python programs is a separate problem.

Scoring a plan requires more than connecting placement JSON to graph metrics:

1. Match the plan to the model, graph, and strategy-space versions.
2. Validate that every selected placement is available and legal.
3. Reject missing, stale, or ambiguous node mappings.
4. Lower the candidate into a parallel graph.
5. Validate boundary contracts and collective consistency.
6. Estimate the scheduled graph with a documented runtime estimator.
7. Account separately for parameters, activations, optimizer state, temporary
   buffers, and opaque-region contracts where supported.
8. Return coverage, confidence, assumptions, and verification evidence.

### `plan`: search and return alternatives

```python
result = plan(artifact, workload, topology, objectives)

result.selected
result.alternatives
result.coverage
result.explanation
result.session_id
```

The planner should return the selected cost-model-optimal plan plus a small
set of diverse alternatives with estimated deltas. The solver's own plan must
pass through the same `score_plan` validation and reporting path used for
caller-supplied plans.

This prevents solver-generated plans from receiving less scrutiny than
external plans.

### Preserve a warm exploration session

The existing constraint and re-solve workflow is already agent-shaped:

```text
add constraint -> resolve -> inspect diff -> explain -> remove or revise
```

A stateless request should remain the default, but `session_id` should address
the traced graph and built optimizer so counterfactuals do not require
re-tracing.

```json
{
  "schema_version": 1,
  "session_id": "...",
  "mutations": [
    {
      "op": "add_node_constraint",
      "node_id": "...",
      "placement": ["S0", "S1"]
    }
  ],
  "return": ["score", "diff", "explanation"]
}
```

Sessions expire. A stale session returns `PLAN_SESSION_EXPIRED`, after which
the caller can replay the immutable request.

## A versioned artifact contract

Local Python wrappers may accept an `nn.Module`, but a durable service
boundary should accept an immutable exported artifact. A mutable model
reference can execute arbitrary Python and may resolve to a different graph in
a different environment.

### Request

```json
{
  "schema_version": 1,
  "model": {
    "artifact": "model.exported_program.pt2",
    "content_hash": "sha256:...",
    "pytorch_build": "...",
    "export_options_hash": "sha256:..."
  },
  "workload": {
    "mode": "training",
    "inputs": [
      {
        "shape": [1024, 4096],
        "dtype": "bfloat16",
        "layout": ["S0", "R"]
      }
    ],
    "dynamic_dimensions": {"input.0": [0]}
  },
  "topology": {
    "mesh": [16, 8],
    "mesh_names": ["dp", "tp"],
    "profile": "h100_nvswitch_400g_v1",
    "profile_hash": "sha256:..."
  },
  "objectives": {
    "mode": "throughput",
    "parameter_memory_bytes": 40000000000,
    "activation_memory_bytes": 20000000000
  },
  "constraints": [],
  "policy": {
    "replicate_only": "warn",
    "fallback_costs": "reject"
  }
}
```

### Response

```json
{
  "schema_version": 1,
  "status": "planned",
  "plan_id": "sha256:...",
  "session_id": "...",
  "provenance": {
    "planner_version": "...",
    "solver_version": "...",
    "solver_options_hash": "sha256:...",
    "strategy_space_version": "...",
    "cost_model_version": "...",
    "calibration_profile_hash": "sha256:...",
    "pytorch_build": "...",
    "compiler_config_hash": "sha256:..."
  },
  "estimated": {
    "step_time_us": 0.0,
    "compute_us": 0.0,
    "communication_us": 0.0,
    "exposed_communication_us": 0.0,
    "peak_memory_bytes": 0,
    "intervals": {
      "step_time_us": [0.0, 0.0]
    }
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
    "critical_terms": [
      {
        "node_id": "...",
        "kind": "all_to_all",
        "message_bytes": 0,
        "mesh_dimension": "tp",
        "provenance": "extrapolated",
        "calibration_range": {}
      }
    ],
    "reasons": []
  },
  "verification": {
    "highest_completed_tier": 1,
    "results": {}
  },
  "artifacts": {
    "placements": "placements.json",
    "parallel_graph": "parallel_graph.pt2",
    "explanation": "explanation.json",
    "visualization": "plan.html"
  },
  "warnings": []
}
```

For fixed artifact hashes, workload, topology, policy, solver configuration,
and implementation versions, results should be reproducible and cacheable.
Bit-for-bit determinism should be tested rather than assumed.

## Required capabilities

### 1. Fine-grained cost provenance

Every important cost term should carry one of:

- `measured`: the specific operation class and configuration was directly
  measured within the declared calibration range;
- `interpolated`: derived between measurements within that range;
- `extrapolated`: derived outside the measured range or for another
  architecture;
- `analytical`: produced by an analytical model;
- `fallback`: produced by a generic heuristic.

Provenance must include topology, message size or tensor shape, mesh
dimension, algorithm/protocol where applicable, software stack, and
calibration range. A single label such as “AllGather measured” is too coarse.

Confidence should be weighted by critical-path contribution. One dominant,
poorly calibrated AllToAll should matter more than many negligible analytical
pointwise costs.

Initial calibration priorities are:

- replace or contain the generic fallback AllToAll heuristic;
- measure Blackwell AllToAll directly;
- compare compute estimates with representative generated kernels;
- model observed prefetch and overlap rather than choosing between zero and
  complete hiding;
- record the topology and software stack for every measurement.

These improvements gate high-confidence performance claims. They do not block
shipping structured analysis, low-confidence estimates, or correctness
verification, provided unreliable paths are explicit and optionally fatal.

### 2. `analyze`: structured coverage before planning

Replace a Boolean support check with a report that remains useful when
planning cannot proceed. It should:

- attempt graph capture with the supplied workload;
- identify trace breaks and source locations;
- classify tensor-producing regions using the support taxonomy;
- report candidate counts and search-space size;
- identify replicate-only and opaque regions;
- estimate whether any representable strategy can satisfy memory constraints;
- suggest possible repairs without pretending to infer the caller's intended
  distributed semantics.

Wrapping a region in `local_map` may be a useful suggestion, but it should not
be asserted as the correct repair automatically.

### 3. Machine-readable failures

Expected failures need stable codes and structured context:

- `TRACE_FAILED`
- `UNSUPPORTED_OP`
- `REPLICATE_ONLY_REGION`
- `PLAN_NOT_REPRESENTABLE`
- `PLAN_GRAPH_MISMATCH`
- `INFEASIBLE_CONSTRAINTS`
- `MEMORY_OVER_BUDGET`
- `TOPOLOGY_PROFILE_MISSING`
- `INPUT_SHAPE_MISMATCH`
- `MESH_MISMATCH`
- `PYTORCH_VERSION_UNSUPPORTED`
- `PLAN_SESSION_EXPIRED`

Include node identifiers, operators, module paths, source locations, shapes,
and constraint identifiers where available. Human-readable messages should be
rendered from this data.

A generic infeasibility result does not identify the minimal contradictory
constraint set. Exact repair hints require additional diagnosis, such as an
infeasible-subset analysis; until that exists, the response should distinguish
known conflicts from hypotheses.

### 4. Explanations as data

Promote existing JSON, logs, optimizer serialization, solution diffs,
placement explanations, cost tables, and visualization into supported return
artifacts. An explanation should answer:

- Why was this placement selected?
- Which alternatives were legal?
- Which constraints eliminated an alternative?
- Which graph edges dominate communication?
- Which tensors dominate memory?
- What changes under another mesh, budget, or boundary layout?
- Which conclusions depend on fallback or extrapolated costs?

### 5. Tiered verification

`verify_plan` should report separate claims:

| Tier | Claim | Substrate |
|---|---|---|
| 1 — Structural | graph, shapes, placements, boundaries, and collective symmetry are valid | fake tensors and graph analysis |
| 2 — Simulated numerical | forward values and gradients match an unsharded reference | `LocalTensorMode` with real values |
| 3 — Small cluster | real collectives, parity, checkpointing, repeated execution, and compilation work | small real mesh |
| 4 — Target validation | memory, throughput, compilation behavior, and baselines are measured | intended topology |

Existing `LocalTensorMode` correctness tests provide a foundation for Tier 2.
Their current direct-joint-graph path should not be presented as equivalent to
verifying the complete public `apply_placement()` path. Upstream issues around
`ProcessGroup` deepcopy and `LocalTensor` tangents must be resolved or worked
around before making that stronger claim.

Estimated performance is never presented as measured performance. Cost-model
calibration gates trustworthy performance scoring; it does not block
structural or numerical verification. These workstreams should progress in
parallel.

### 6. Profile-guided refinement

Calibration should be an operational capability:

1. Produce a small, diverse set of feasible plans.
2. Benchmark selected collectives, kernels, or short training steps.
3. Store the measurements with their precise configuration and scope.
4. Update only the model terms justified by those measurements.
5. Re-score and, when appropriate, re-solve.
6. Preserve both predicted and observed values in the plan artifact.

An aggregate step benchmark validates the aggregate prediction; it does not
automatically convert every modeled component into a measured component. Use
`validated` for aggregate evidence and reserve `measured` for directly
measured terms.

### 7. Opaque-region cost contracts

`local_map` is an intentional boundary for data-dependent or otherwise
inexpressible distributed semantics. The planner should allow such a region to
declare or provide:

- input and output placement contracts;
- estimated or measured runtime as a function of relevant shapes;
- peak and temporary memory;
- collective classes and mesh dimensions;
- calibration provenance;
- verification hooks.

Without this contract, the region remains visible as opaque and lowers the
confidence of the complete plan. Richer layout representations may shrink the
boundary over time, but eliminating it is not a prerequisite for useful global
planning.

## Benchmarks and trust

The adoption question is empirical:

> Are AutoParallel's plans competitive with strong hand-written plans, and do
> its estimates rank alternatives correctly enough to guide search?

### Minimal viable benchmark

Ship a focused benchmark early:

- LLaMA3-8B;
- one 1D and one 2D mesh;
- one supported H100 topology;
- a strong hand-written TorchTitan FSDP+TP baseline;
- throughput, peak memory, planning time, and numerical parity;
- predicted and measured critical-path costs;
- a classification of every meaningful discrepancy.

This is not enough to establish generality. It is enough to exercise the
measurement, attribution, and reporting pipeline.

### Full matrix

Expand to:

- dense decoders, encoder-decoder models, FlexAttention, and MoE;
- multiple model scales, batch sizes, and sequence lengths;
- 1D and 2D meshes, followed by pipeline and richer hybrid configurations;
- multiple GPU generations and single- and multi-node systems;
- throughput, exposed communication, peak memory, compilation time, and
  planning time;
- numerical parity and distributed-checkpoint compatibility;
- strong hand-written FSDP, TP, and hybrid baselines.

Report wins, ties, and losses. Attribute losses to strategy-space gaps,
cost-model errors, compiler limitations, runtime overhead, or unsupported
communication patterns.

Rank correlation is the primary metric for choosing among plans, but it is not
the only metric. Absolute error and calibrated intervals matter when reporting
expected latency, exposed communication, or memory risk.

## Scope boundaries

### No pipeline parallelism today

AutoParallel currently decides intra-stage placements. Micro-pipelined tensor
parallelism is not model pipeline parallelism. At frontier scale, stage
partitioning and scheduling are often necessary; the product should state this
limitation and eventually compose with or incorporate a pipeline planner.

### `local_map` is a real, partially opaque boundary

Data-dependent routing cannot always be represented by static DTensor
placements. Keeping an explicit boundary is legitimate. The planner should
price it through a declared or measured cost contract instead of treating it
as free or fully understood.

### External plans are intentionally constrained at first

The first `score_plan` should evaluate selections from AutoParallel's known
strategy space. It should not claim to score arbitrary programs containing
unknown communication, scheduling, or rank-dependent behavior.

Support can expand through richer plan schemas and opaque-region contracts.

### Private PyTorch dependencies remain a product risk

Every artifact should record the PyTorch build and compiler configuration.
The project should publish a compatibility matrix, isolate private integration
behind narrow adapters, and upstream reusable functionality where practical.

## Delivery plan

The work should proceed in parallel tracks. Cost calibration is important, but
it should not unnecessarily block diagnostics or correctness verification.

### Track A — inspectable planning

1. Define versioned request, response, plan, topology, and diagnostic schemas.
2. Expose existing JSON, serialization, explanation, and diff functionality.
3. Add `analyze` with explicit replicate-only and opaque coverage.
4. Preserve the warm exploration session behind a stable API.
5. Add machine-readable errors and strict policy modes.

### Track B — trustworthy evaluation

1. Build the minimal benchmark and calibration harness.
2. Attach fine-grained cost provenance and intervals.
3. Ship `score_plan` for representable plans, clearly labeling unreliable
   estimates.
4. Route solver-generated plans through the same scoring path.
5. Add profile-guided candidate evaluation.

### Track C — verification

1. Ship structural verification.
2. Promote simulated numerical comparison into a supported tool.
3. Verify the full public lowering and execution path.
4. Add small-cluster execution and checkpoint verification.
5. Add target-topology validation artifacts.

### Track D — broader and more stable optimization

1. Add activation, optimizer-state, and temporary-buffer memory objectives.
2. Add cost contracts for `local_map` and other opaque regions.
3. Compose with or incorporate pipeline parallelism.
4. Improve context- and expert-parallel modeling.
5. Isolate private PyTorch dependencies and upstream reusable functionality.

## Success criteria

Evaluate outcomes rather than lines of generated parallelism code:

- A broad model corpus receives useful coverage reports even when planning
  fails.
- Unsupported, opaque, extrapolated, fallback, and replicate-only regions are
  never silent.
- Structural verification catches invalid plans before cluster launch.
- Numerical verification matches the unsharded reference within declared
  tolerances.
- Solver-generated and caller-proposed plans pass through the same validation
  and scoring path.
- Estimated rankings correlate with measurements on supported topologies.
- Absolute error and reported intervals are calibrated for supported claims.
- Selected plans are competitive with strong hand-written baselines.
- Every benchmark loss has an attributed cause.
- Re-planning for a new topology is cheaper than authoring and validating a
  new recipe.
- Agents can diagnose common failures and explain tradeoffs using structured
  outputs alone.
- Artifacts remain reproducible within the declared compatibility window.

## Division of labor

| Agent or human | AutoParallel |
|---|---|
| Interpret goals and deployment constraints | Capture and analyze the derived graph |
| Produce traceable single-device semantics | Enumerate legal distributed strategies |
| Propose constraints or representable plans | Search, score, and compare alternatives |
| Mark explicit custom regions | Validate boundaries and incorporate declared costs |
| Choose among documented tradeoffs | Return calibrated estimates and uncertainty |
| Run selected experiments | Preserve measurements and refine calibration |
| Explain decisions to the user | Return reproducible evidence and diagnostics |

## If only one product change is funded

Ship a narrow `score_plan` / `plan_only` vertical slice that:

- accepts only AutoParallel-representable plans;
- supports one explicitly calibrated H100 topology;
- makes replication and fallback costs loud or fatal;
- returns versioned provenance and critical-path confidence;
- runs structural verification;
- compares against one strong hand-written baseline.

Do not wait for every cost path or architecture to be calibrated. Limit the
supported envelope, reject requests outside it in strict mode, and expand from
measured evidence.

This changes AutoParallel from a tool that merely writes parallelism into a
backend that can evaluate it. It also closes the most dangerous failure mode:
a plan that runs correctly and is quietly slow because important regions were
replicated or poorly modeled.

## Strategic conclusion

Agents reduce the cost of authoring distributed code, so AutoParallel should
not build its identity around eliminating that code. Its defensible role is to
provide trusted planning and evidence across changing models, workloads, and
hardware.

The long-term moat is not code generation. It is the combination of:

- a legal strategy space over the derived forward/backward graph;
- calibrated cost and memory models;
- executable lowering;
- explicit uncertainty and coverage;
- correctness and performance verification;
- reproducible plan artifacts.

Agents make those capabilities more valuable because they generate more
candidates and more architecture variants. AutoParallel should be the backend
that tells them which candidates are legal, which are promising, and what has
actually been verified.
