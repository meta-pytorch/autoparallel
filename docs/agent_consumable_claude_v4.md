# AutoParallel as an Agent-Facing Planner and Plan Evaluator

## Executive summary

Coding agents are making explicit FSDP, tensor-parallel, and checkpointing code
cheaper to write. That erodes AutoParallel's value as a convenience layer. It
does not eliminate the planning problem underneath, and it increases the rate at
which new model variants outgrow established recipes.

The durable promise is not "no one has to write parallelism code." It is:

> AutoParallel searches or evaluates the legal strategy space, prices the
> tradeoffs against a versioned hardware model, produces an executable plan, and
> supplies evidence about whether that plan should be trusted.

```text
                         ┌───────────────────────┐
model + workload ───────▶│                       │──────▶ executable plan
topology + objectives ──▶│ AutoParallel backend  │──────▶ costs and alternatives
candidate plan ─────────▶│                       │──────▶ verification evidence
                         └───────────────────────┘
```

This is a repositioning from convenience automation to **planning and
evidence**. Agents and humans propose model code, constraints, or complete
candidate plans; AutoParallel analyzes the derived graph, searches or scores the
representable alternatives, validates their consistency, and returns
reproducible artifacts.

This document supersedes the five earlier drafts listed in
[the docs index](README.md).

## Concede what agents commoditize

The README's "no manual parallelism code required" is genuinely less
differentiated now. A TorchTitan-style plan for a standard transformer is
mechanical, widely exemplified, and increasingly easy to generate. For a fixed
model on a fixed cluster, a maintained hand-written recipe can be simple,
predictable, and well benchmarked. AutoParallel should not argue that agents are
incapable of producing such code.

The stronger distinction is between **authoring distributed code** and **deciding
whether a distributed plan is good**:

- An agent can emit a plausible plan; AutoParallel can compare it against the
  other legal plans and say what it costs.
- An agent can reuse a familiar recipe; AutoParallel can re-evaluate that recipe
  for a different shape, mesh, or topology.
- An agent can repair model code; AutoParallel can name the graph region and the
  constraint responsible for a failure.
- An agent can launch experiments; AutoParallel can reduce how many expensive
  distributed experiments are needed.

Stating the concession first makes the remaining case more credible.

## Why the planner remains valuable

### Kernel agents are the proof, not the counterexample

PTX, CuTe, and Triton agents work because the loop is tight — generate, compile,
run, profile, compare against numeric and throughput ground truth — and because
they *do not bypass the stack*. They do not replace compilers, assemblers,
profilers, or correctness checks. They consume them.

So the existence of capable kernel agents is evidence **for** building this
backend, not against it. A successful codegen agent is a stack consumer, and the
layer it consumes has to exist and has to be trustworthy.

Distributed planning then fails the tight-loop precondition. A candidate can
require a multi-node launch, performance is noisier, and some failures appear
only at scale. Agents reduce the cost of producing candidates, not the cost of
evaluating them. A planner should prune the space with graph constraints and
calibrated models, then reserve real hardware for a small set of high-value
measurements.

### The optimization object is a derived graph

AutoParallel plans over the joint forward and backward graph produced through
Dynamo and AOTAutograd (`api.py`). Backward operations, saved tensors, gradient
reductions, and most redistribution opportunities are not present in the
original `nn.Module` source.

The barrier is not that backward math is unknowable — the backward of a matmul
is well defined. It is that the facts that decide real cases are properties of
the graph *after* partitioning, saved-tensor selection, recompute policy, and
representation-level graph passes. Reasoning about them from `model.py` means
reconstructing AOTAutograd's output, at which point you have rebuilt the
compiler and still have no cost model.

[adaptive_sharding.md](adaptive_sharding.md) makes this concrete. LLaMA3-8B and
LLaMA3-70B, identical training configuration, diverge on `wo`: the 8B keeps it
sequence-parallel, the 70B goes column-parallel. The deciding factors are that
column-parallel yields `P(sum)S(0)` weight gradients — one reduce-scatter
dimension — where sequence-parallel yields `P(sum)P(sum)` and needs a full 2D
reduce-scatter, and that sequence-parallel `wo` emits `S(0)S(1)`, which the
residual add consumes for free.

The same document records a sharper version. Sequence-parallel strategies are
only *visible* to the solver when the `view → mm → view` pattern is fused to
preserve the sequence dimension; without that pass the decomposition folds
sequence into batch and the entire strategy family disappears. Whether a
strategy can be considered at all is a property of the compiler's intermediate
representation.

### The answer is a function of the deployment

A recipe is an artifact. The required behavior is closer to a function:

```text
plan = f(graph, shapes, topology, memory, objective, compiler behavior)
```

Batch size, sequence length, mesh dimensions, node count, GPU generation, and
inter-node bandwidth each move the preferred plan without touching model source.
This holds even if architecture churn slows.

### The cost model is a systems asset, not an oracle

`cost_models/nccl_cost_model.py` ports NCCL's `tuning.cc` algorithm and protocol
selection — Ring, Tree, CollNet, NVLS, NVLS Tree crossed with LL, LL128, Simple
— then corrects it against nccl-tests measurements on H100 NVSwitch at 1, 2, 4,
8, 16, and 32 nodes. `adaptive_sharding.md` records that the adaptive behavior
*requires* that model: the generic estimator did not price communication finely
enough to separate the candidates. That body of systems knowledge cannot be
recreated by language-model reasoning. It is measured on hardware.

It must be exposed with its limits, and vague honesty is not useful. Precisely:

**On the default path** — `cost_model="nccl"`, which calls
`set_nccl_topo_config(detect_nccl_topo_config(mesh))`:

- Blackwell is extrapolated. `_BLACKWELL_BW_SCALE = 640.0 / 320.0` scales
  bandwidth from Hopper, and `_A2A_CE_BW[BLACKWELL]` is annotated `estimated
  proportionally from bw_intra ratio; needs profiling`.
- Compute costs are analytical: FLOPs over throughput at an assumed 70%
  efficiency versus bytes over bandwidth at 70%, with a 7 µs launch floor. None
  of it has been checked against real generated kernels.
- Overlap is selected, not predicted. `apply_prefetch_discount` is a
  caller-chosen multiplier defaulting to `scale=0.0` — collectives treated as
  free — and is not applied automatically. The truth lies between free and fully
  exposed, and `estimate_graph_metrics` already computes `exposed_comm_time` by
  multi-stream simulation, so a predicted overlap is within reach.

**On the fallback path only**, where `_nccl_topo_config is None`:

- `all_to_all_cost` ends with `total_time *= 5` under a `FIXME`.
  `collective_comm_cost` routes to `nccl_all_to_all_cost` whenever the NCCL
  config is set, so this does not affect default planning. It still needs
  containing: `adaptive_sharding.md` concludes that intra-node all-to-all is
  competitive with all-gather for seq-par → head-par SDPA transitions, and a 5×
  multiplier inverts exactly that comparison. A caller who selects the fallback
  gets a different plan family for a reason that is not a property of the
  hardware.

**These errors do not cancel in the solver.** A bias uniform across all costs
would. A bias on one collective type or operator class shifts the selected plan
away from every strategy that uses it. Calibration error is a plan-selection
problem first and a reporting problem second. Returned numbers must say how they
were produced, and unreliable paths must be rejectable in strict mode.

## What exists today

Support and maturity are not binary, and that applies to this list as much as to
operators.

| Capability | State | Evidence | Qualification |
|---|---|---|---|
| Joint fwd/bwd capture | Core | CI | tied to evolving PyTorch internals |
| DTensor strategy enumeration | Core | CI | coverage depends on upstream rules and local overrides |
| Global ILP | Core | CI | optimal only within candidates and modeled costs |
| Executable lowering | Core | CI | not packaged as an independently verifiable artifact |
| `torch.compile` backend | Implemented | CI | AC and overlap use private compiler hooks |
| NCCL cost model | Strongest on H100 NVSwitch | `test_nccl_cost_model` | coverage varies by collective, size, topology, arch |
| Compute model | Analytical | — | 70% efficiency and 7 µs floor unvalidated |
| Runtime/memory simulation | Works | `test_estimate_graph_metrics` | single peak figure; no category breakdown |
| Numerical correctness | Simulated | `test_correctness` (`LocalTensorMode`) | verifies the direct joint-graph path, not public `apply_placement()` |
| Distributed checkpointing | Tested | `test_dcp_roundtrip` | real multi-GPU coverage narrower than simulated |
| Dynamic shapes, inference, mixed precision | Tested | dedicated test files | not equivalent to arbitrary dynamic programs |
| FlexAttention | Tested | `test_flex_attention` | auxiliary layouts have explicit restrictions |
| `local_map` / MoE | Explicit boundary | examples, DSV3 | internal communication not globally optimized or priced |
| TorchTitan integration | LLaMA3 in CI | `--module autoparallel.llama3` | DeepSeek/MoE integration less mature (CI disabled) |
| Pipeline parallelism | Not supported | — | micro-pipelined TP is not model PP |

Plan JSON, trace artifacts, optimizer serialization, re-solving, placement
explanations, and solution diffs all exist. They are fragmented across internal
methods, logs, stdout, and tracing artifacts. `__init__.py` exports five symbols
and none of them scores a plan. These capabilities should become the center of
the interface.

## Claims the product must make precisely

### Say "cost-model-optimal"

The ILP returns an exact optimum over the candidate strategies and costs it is
given. It does not prove the plan is fastest on real hardware. Every result
should identify the strategy-space version; the cost-model and
calibration-profile versions; the topology profile; the workload shapes and
dynamic dimensions; all constraints and boundary layouts; replication fallbacks
and opaque regions; and how each important cost was produced.

### Support is a taxonomy

A graph region may be directly supported, supported through decomposition,
representable only with replication, opaque or manually specified, or
unsupported.

**Replicate-only is the dangerous case.** It does not fail; it yields a correct,
working, quietly slow plan. It is the one failure mode a caller cannot detect
from the outcome and would not think to check. It must be visible in every
response and configurable as fatal.

### Verification claims must name their substrate

Fake tensors validate graph construction and shape propagation; they do not
establish numerical equivalence. Simulated ranks exercise real values but not
real network behavior. A small real mesh catches different failures from the
target topology. No single `verified: true` field can carry all of these claims.

### Measured, validated, and modeled are different claims

This distinction is easy to lose and expensive to lose.

- **Measured**: this specific operation class and configuration was directly
  benchmarked, inside a declared calibration range.
- **Validated**: an *aggregate* prediction was checked against an aggregate
  measurement.
- **Modeled**: produced analytically, by interpolation, by extrapolation, or by
  a fallback heuristic.

An end-to-end step benchmark that matches prediction validates the *sum*. Errors
in opposite directions cancel inside a sum, so agreement at the aggregate level
is not evidence about any individual term. Promoting per-collective entries to
`measured` on the strength of an aggregate match would make the cost model
report rising confidence while its component accuracy is unknown — the failure
mode the provenance scheme exists to prevent.

### The caller still supplies policy

The planner needs representative inputs, a topology, objectives, and some
boundary constraints. Data-dependent MoE regions may still require explicit
local semantics. Agents can help supply this policy, but defaults and
assumptions must be *returned*, not silently applied.

### Compatibility is part of correctness

AutoParallel depends on private, evolving PyTorch APIs and nightly builds. Plan
artifacts must record the PyTorch build and relevant compiler configuration, and
the project needs a tested compatibility matrix plus clear failure codes when the
environment is unsupported. A plan that is not reproducible against a declared
PyTorch version is not a plan.

## Product shape: evaluate and search

Two operations sharing one validation and scoring implementation.

### `score_plan`: evaluate a candidate

```python
report = score_plan(artifact, workload, topology,
                    plan=candidate, objectives=objectives)

report.valid
report.coverage
report.estimated.step_time_us
report.estimated.exposed_communication_us
report.estimated.memory          # by category
report.confidence
report.verification
report.assumptions
```

Initially this accepts **only plans expressible through the captured graph and
AutoParallel's candidate strategy space**. `load_placements` matches placement
strings against the enumerated strategies, so a plan outside that space cannot be
scored — it must be rejected explicitly, not silently mismatched. Scoring
arbitrary distributed Python programs is a separate problem and the API must not
imply it.

Scoring requires more than connecting placement JSON to graph metrics:

1. Match the plan to the model, graph, and strategy-space versions.
2. Validate that every selected placement is available and legal.
3. Reject missing, stale, or ambiguous node mappings.
4. Lower the candidate into a parallel graph.
5. Validate boundary contracts and collective consistency.
6. Estimate the scheduled graph with a documented runtime estimator.
7. Account separately for parameters, activations, optimizer state, temporary
   collective buffers, and opaque-region contracts. Bucketing makes temporary
   buffers real — `max_in_flight_gb` exists as a knob for exactly this.
8. Return coverage, confidence, assumptions, and verification evidence.

### `plan`: search and return alternatives

```python
result = plan(artifact, workload, topology, objectives)

result.selected
result.alternatives    # small diverse set with estimated deltas
result.coverage
result.explanation
result.session_id
```

The solver's own plan passes through the same `score_plan` validation and
reporting path as a caller-supplied plan, so solver output receives no less
scrutiny and the two sets of numbers stay comparable.

### Preserve a warm exploration session

The existing workflow is already agent-shaped:

```text
add constraint -> resolve -> inspect diff -> explain -> remove or revise
```

`resolve()` is cheap by construction — it re-solves without rebuilding the
objective. A stateless request stays the default, but `session_id` should
address the traced graph and built optimizer so counterfactuals do not retrace.

```json
{
  "schema_version": 1,
  "session_id": "...",
  "mutations": [
    {"op": "add_node_constraint", "node_id": "...", "placement": ["S0", "S1"]}
  ],
  "return": ["score", "diff", "explanation"]
}
```

Sessions expire; a stale id returns `PLAN_SESSION_EXPIRED` and the caller
replays the immutable request.

## A versioned artifact contract

Local Python wrappers may accept an `nn.Module`, but a durable service boundary
should accept an immutable exported artifact. A mutable model reference executes
arbitrary Python and may resolve to a different graph in a different environment.

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
      {"shape": [1024, 4096], "dtype": "bfloat16", "layout": ["S0", "R"]}
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
    "fallback_costs": "reject",
    "extrapolated_costs": "warn"
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
    "memory": {
      "parameters_bytes": 0,
      "activations_bytes": 0,
      "optimizer_state_bytes": 0,
      "collective_buffers_bytes": 0,
      "peak_bytes": 0
    },
    "intervals": {
      "step_time_us": [0.0, 0.0],
      "peak_bytes": [0, 0]
    }
  },
  "coverage": {
    "direct": 0, "decomposed": 0,
    "replicate_only": 0, "opaque": 0, "unsupported": 0
  },
  "confidence": {
    "level": "medium",
    "critical_terms": [
      {
        "node_id": "...",
        "kind": "all_to_all",
        "message_bytes": 0,
        "mesh_dimension": "tp",
        "n_nodes": 4,
        "algo_proto": "ring/simple",
        "basis": "extrapolated",
        "calibration_range": {},
        "share_of_critical_path": 0.31
      }
    ],
    "reasons": []
  },
  "verification": {"highest_completed_tier": 1, "results": {}},
  "artifacts": {
    "placements": "placements.json",
    "parallel_graph": "parallel_graph.pt2",
    "explanation": "explanation.json",
    "visualization": "plan.html"
  },
  "warnings": []
}
```

For fixed artifact hashes, workload, topology, policy, solver configuration, and
implementation versions, results should be reproducible and cacheable.
Bit-for-bit determinism should be **tested**, not assumed — a CBC-backed solve
does not obviously provide it across platforms.

## Required capabilities

### 1. Fine-grained cost provenance and confidence

Tag every important cost term `measured`, `interpolated`, `extrapolated`,
`analytical`, or `fallback`, with the `validated` distinction from above
available for aggregate evidence.

Provenance must be keyed by *configuration*, not by collective name. The cost
tables are indexed by algorithm, protocol, node count, ranks per node, and
message size, so a label such as "AllGather: measured" asserts far more than any
measurement supports. All-gather measured at 1 GB intra-node says nothing about
all-gather at 4 MB across 16 nodes. Carry topology, message size or tensor shape,
mesh dimension, algorithm/protocol where applicable, software stack, and
calibration range.

Weight confidence by critical-path contribution. One dominant, poorly calibrated
all-to-all matters more than a hundred negligible analytical pointwise costs.

Initial calibration priorities:

- measure Blackwell all-to-all directly (already item 1 of the TODO block in
  `nccl_cost_model.py`);
- compare compute estimates against representative generated kernels; validate
  or replace the 70% efficiency assumption and the 7 µs floor;
- model observed prefetch and overlap instead of choosing between zero and
  complete hiding;
- contain the fallback all-to-all heuristic, or route it to the NCCL path;
- record the exact topology and software stack for every measurement.

These gate high-confidence performance claims. They do **not** block structured
analysis, low-confidence estimates, or correctness verification, provided
unreliable paths are explicit and optionally fatal.

### 2. `analyze`: structured coverage before planning

Replace a Boolean support check with a report that stays useful when planning
cannot proceed. It should attempt capture; identify trace breaks and source
locations; classify tensor-producing regions using the taxonomy; report
candidate counts and search-space size; identify replicate-only and opaque
regions; and estimate whether any representable strategy can satisfy the memory
constraints.

Present repairs as possibilities. Wrapping a region in `local_map` may be a good
suggestion, but it must not be asserted as the correct fix: the tool cannot infer
the intended distributed semantics of a region it failed to analyze.

### 3. Machine-readable failures

Stable codes with structured context; human-readable messages rendered *from*
the data.

```json
{
  "error": "REPLICATE_ONLY_REGION",
  "node_id": "mm_14",
  "module_path": "layers.3.attention.wq",
  "op": "aten.mm.default",
  "source": {"file": "model.py", "line": 142},
  "shapes": [[8192, 4096], [4096, 4096]],
  "detail": "no sharded strategy available; plan falls back to replication",
  "estimated_cost_us": 812.0
}
```

Codes: `TRACE_FAILED`, `UNSUPPORTED_OP`, `REPLICATE_ONLY_REGION`,
`PLAN_NOT_REPRESENTABLE`, `PLAN_GRAPH_MISMATCH`, `INFEASIBLE_CONSTRAINTS`,
`MEMORY_OVER_BUDGET`, `TOPOLOGY_PROFILE_MISSING`, `INPUT_SHAPE_MISMATCH`,
`MESH_MISMATCH`, `PYTORCH_VERSION_UNSUPPORTED`, `PLAN_SESSION_EXPIRED`.

`PLAN_NOT_REPRESENTABLE` and `PLAN_GRAPH_MISMATCH` are the two failures a
caller-supplied plan actually hits, and they must be distinguishable: "I cannot
express your strategy" is a different conversation from "your plan was built
against a different graph."

A generic infeasibility result does not identify the minimal contradictory
constraint set. That needs an irreducible-infeasible-subset analysis, which CBC
does not provide for free. Until it exists, distinguish known conflicts from
hypotheses rather than implying a diagnosis.

`export_json.py` already extracts node, `module_path`, and source location.

### 4. Explanations as data

Promote `get_json`, `get_log`, `explain_placement`, `print_costs_for_node`,
`diff_solutions`, optimizer serialization, and
`visualizer/build_display_from_json.py` into supported return artifacts. An
explanation should answer: why this placement; which alternatives were legal and
at what cost; which constraint eliminated a requested one; which edges dominate
communication and which tensors dominate memory; what changes under another mesh
or budget; and which conclusions depend on fallback or extrapolated costs.

### 5. Tiered verification

| Tier | Claim | Substrate |
|---|---|---|
| 1 Structural | graph, shapes, placements, boundaries, collective symmetry are valid | fake tensors and graph analysis |
| 2 Simulated numerical | forward values and gradients match an unsharded reference | `LocalTensorMode`, real values |
| 3 Small cluster | real collectives, parity, checkpointing, repeat determinism, compilation | small real mesh |
| 4 Target | memory, throughput, compile behavior, baselines measured | intended topology |

`tests/test_correctness.py` is the foundation for Tier 2, but it exercises the
direct joint-graph path — that is **not** the same claim as verifying the public
`apply_placement()` path, and the two must not be conflated in a response. Its
docstring records the blockers to the stronger claim:

1. `ProcessGroup` objects from `compile_on_one_rank` are not deepcopy-safe,
   breaking `extract_forward_graph`'s deepcopy of the joint graph.
2. AOT autograd's compiled backward rejects `LocalTensor` tangents because
   `LocalTensor` does not implement `__coerce_same_metadata_as_tangent__`.

Estimated performance is never presented as measured performance. Calibration
gates trustworthy performance scoring; it does not gate structural verification,
numerical verification, or coverage reporting.

### 6. Profile-guided refinement

1. Produce a small, diverse set of feasible plans.
2. Benchmark selected collectives, kernels, or short training steps.
3. Store measurements with their precise configuration and scope.
4. Update **only the model terms justified by those measurements**.
5. Re-score, and re-solve when the calibration moves a decision.
6. Preserve predicted and observed values in the plan artifact.

Step 4 is where discipline is required. An aggregate benchmark earns
`validated` on the aggregate; only directly benchmarked configurations earn
`measured`.

### 7. Opaque-region cost contracts

`local_map` is an intentional boundary for data-dependent semantics that static
DTensor placements cannot express. The planner should let such a region declare
or supply:

- input and output placement contracts;
- estimated or measured runtime as a function of the relevant shapes;
- peak and temporary memory;
- collective classes and mesh dimensions used internally;
- calibration provenance for the above;
- verification hooks.

Without a contract the region stays visible as opaque and lowers whole-plan
confidence. Richer layout representations may shrink the boundary over time —
see `cute_sharding_design.md` — but eliminating it is not a prerequisite for
useful global planning, and it is probably not achievable for data-dependent
routing.

## Benchmarks and trust

The adoption question is empirical:

> Are AutoParallel's plans competitive with strong hand-written plans, and do its
> estimates rank alternatives correctly enough to guide search?

**Metrics.** Rank correlation is primary for *choosing among* plans. It is not
sufficient. Absolute error and calibrated intervals matter whenever a number is
reported rather than compared — expected latency, exposed communication, and
above all memory, where "does this fit in 80 GB" is a threshold question that
rank correlation cannot answer.

### Minimal viable benchmark

Ship this early. It is not enough to establish generality; it is enough to
exercise the measurement, attribution, and reporting pipeline end to end.

- LLaMA3-8B, one 1D and one 2D mesh, one supported H100 topology;
- one strong hand-written TorchTitan FSDP+TP baseline;
- throughput, peak memory, planning time, numerical parity;
- predicted and measured critical-path costs;
- a classification of every meaningful discrepancy.

### Full matrix

Dense decoders, encoder-decoder, FlexAttention, and MoE; multiple scales, batch
sizes, and sequence lengths; 1D and 2D meshes then pipeline and hybrid; at least
two GPU generations, single- and multi-node; throughput, exposed communication,
peak memory, compile time, planning time; numerical parity and DCP
compatibility; strong hand-written FSDP, TP, and hybrid baselines.

Report wins, ties, **and losses**, attributing each loss to a strategy-space gap,
cost-model error, compiler limitation, runtime overhead, or unsupported
communication pattern. That attribution turns a benchmark into a roadmap, and the
same measurements calibrate reported confidence.

## Scope boundaries

**No pipeline parallelism.** The optimizer decides intra-stage placement; the
`_pipelined_*` symbols in `graph_passes/async_tp/` are micro-pipelined TP and
unrelated. At frontier scale, stage partitioning and scheduling are not optional.
State the limitation and plan to compose with a pipeline planner rather than
absorb one soon.

**`local_map` is a real, partially opaque boundary.** Data-dependent routing
cannot always be expressed as static DTensor placements. Keeping the boundary is
legitimate; treating its interior as free or fully understood is not. Price it
through a declared or measured contract.

**External plans are intentionally constrained at first.** The first
`score_plan` evaluates selections from the known strategy space. It must not
claim to score arbitrary programs with unknown communication, scheduling, or
rank-dependent behavior. Support expands through richer plan schemas and
opaque-region contracts.

**Private PyTorch dependencies remain a product risk.** Record the build and
compiler configuration in every artifact; publish a compatibility matrix;
isolate integration behind narrow adapters; upstream what is reusable.

## Delivery: one vertical slice, then a sequence

This repository has one dominant contributor and a long tail. Parallel tracks
presume parallel people, so the plan is a single narrow slice that exercises the
whole pipeline, followed by an ordered expansion.

### Step 1 — the slice

Ship a `score_plan` / `plan` vertical that:

- accepts only AutoParallel-representable plans, rejecting others with
  `PLAN_NOT_REPRESENTABLE` or `PLAN_GRAPH_MISMATCH`;
- supports exactly one explicitly calibrated H100 topology profile, and returns
  `TOPOLOGY_PROFILE_MISSING` outside it;
- makes replication and fallback costs loud, and fatal under strict policy;
- returns versioned provenance and critical-path confidence;
- runs Tier 1 structural verification;
- is compared against one strong hand-written baseline.

Do not wait for every cost path or architecture to be calibrated. Limit the
envelope, reject outside it, and expand from measured evidence.

This is deliberately a cut through every layer — schema, coverage, scoring,
provenance, verification, benchmark — rather than a complete layer. It is the
smallest thing that proves the repositioning is real.

### Then, in order

2. **Coverage and diagnostics breadth.** Full `analyze` report; the complete
   error-code set; strict policy modes across all categories.
3. **Explanations and the warm session.** Promote the existing explanation
   surfaces to return values; version all schemas; expose `session_id`.
4. **Verification tiers 1 and 2 as products**, including the two upstream fixes
   and the public-lowering-path claim. Independent of calibration work.
5. **Provenance depth and profile-guided refinement.** Per-configuration
   provenance, `validated` versus `measured` discipline, calibration loop.
6. **Full benchmark matrix** with loss attribution.
7. **Breadth and stability.** Activation, optimizer-state, and buffer memory
   objectives; opaque-region contracts; pipeline composition; context- and
   expert-parallel modeling; compatibility matrix and narrow adapters.

## Success criteria

- A broad model corpus receives useful coverage reports even when planning fails.
- Unsupported, opaque, extrapolated, fallback, and replicate-only regions are
  never silent.
- Structural verification catches invalid plans before cluster launch.
- Numerical verification matches the unsharded reference within declared
  tolerances, and the claim names its substrate.
- Solver-generated and caller-proposed plans pass through the same validation and
  scoring path.
- Estimated rankings correlate with measurements on supported topologies.
- Absolute error and reported intervals are calibrated for the claims that are
  reported as values rather than comparisons.
- `measured`, `validated`, and `modeled` are never conflated in an artifact.
- Selected plans are competitive with strong hand-written baselines, and every
  loss has an attributed cause.
- Re-planning for a new topology is cheaper than authoring and validating a new
  recipe.
- Agents can diagnose common failures and explain tradeoffs from structured
  output alone.
- Artifacts reproduce within the declared compatibility window, and determinism
  is tested rather than assumed.

## Division of labor

| Agent or human | AutoParallel |
|---|---|
| Interpret goals and deployment constraints | Capture and analyze the derived graph |
| Produce traceable single-device semantics | Enumerate legal distributed strategies |
| Propose constraints or representable plans | Search, score, and compare alternatives |
| Mark explicit custom regions | Validate boundaries and incorporate declared costs |
| Choose among documented tradeoffs | Return calibrated estimates and uncertainty |
| Run the selected experiments | Preserve measurements and refine calibration |
| Explain decisions to the user | Return reproducible evidence and diagnostics |

## Strategic conclusion

Agents reduce the cost of authoring distributed code, so AutoParallel should not
build its identity around eliminating that code. Its defensible role is trusted
planning and evidence across changing models, workloads, and hardware.

The moat is the combination of a legal strategy space over the derived
forward/backward graph, calibrated cost and memory models, executable lowering,
explicit uncertainty and coverage, correctness and performance verification, and
reproducible artifacts.

Agents make those capabilities more valuable because they produce more candidates
and more architecture variants. AutoParallel should be the backend that tells
them which candidates are legal, which are promising, and what has actually been
verified — and, just as importantly, which of those three claims it is making.
