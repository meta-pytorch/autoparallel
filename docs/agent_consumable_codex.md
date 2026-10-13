# AutoParallel in an Agent-Driven Development World

## Executive summary

Coding agents reduce the cost of writing explicit FSDP, tensor-parallel, and
checkpointing code. That weakens AutoParallel's value as a convenience layer,
but it does not remove the underlying planning problem.

Distributed execution is a global, hardware-dependent optimization problem.
The best plan depends on the complete forward and backward graph, tensor
shapes, memory limits, collective behavior, and opportunities to overlap
communication with computation. Generating syntactically correct sharding code
does not establish that the code is correct, feasible, or fast.

AutoParallel should therefore position itself as an **explainable parallelism
planner and verification backend** for both agents and humans:

```text
model artifact + workload + topology + objectives
                         |
                         v
                AutoParallel planner
                         |
                         v
       plan + evidence + generated module + diagnostics
```

The durable product promise is not “no one has to write parallelism code.” It
is:

> AutoParallel searches the valid strategy space, predicts the tradeoffs,
> produces an executable plan, and supplies the evidence needed to trust it.

## Why agents do not eliminate the need

Agents are already effective at reproducing established recipes. Given a
standard transformer and a familiar topology, an agent can often generate a
reasonable combination of FSDP and tensor parallelism. That work will continue
to become cheaper.

The hard parts remain:

- **Global decisions.** A placement changes the legal and profitable choices
  of downstream operations. AutoParallel's ILP enforces uniqueness,
  consistency, graph-flow, and forward/backward constraints jointly.
- **Expensive feedback.** Testing one kernel can take seconds. Testing a
  distributed plan can consume a multi-node allocation and still produce noisy
  results or failures that only appear at scale.
- **Hardware dependence.** A good plan on one NVLink topology may be poor on a
  different interconnect, GPU generation, or node count.
- **Correctness.** Generated code must preserve forward values, gradients,
  aliasing, checkpoint semantics, and collective ordering across ranks.
- **Architecture variation.** Agents make it cheaper to create new model
  variants, precisely where fixed LLaMA- or GPT-specific recipes stop being
  sufficient.

An agent can author a candidate implementation. It still benefits from a
specialized system that constrains the search space and provides executable,
checkable evidence.

## What AutoParallel already provides

The repository contains most of the foundation for this role:

- Joint forward/backward graph capture through Dynamo and AOTAutograd.
- DTensor-based strategy enumeration, including reuse of upstream strategies.
- A global ILP over computation, communication, placement transitions, memory,
  and forward/backward consistency.
- NCCL-aware topology and collective cost modeling.
- Lowering from a selected plan to an executable distributed module.
- A `torch.compile` backend with activation checkpointing and
  communication/computation overlap passes.
- Dynamic batch shapes, inference, mixed precision, uneven sharding,
  FlexAttention, distributed checkpointing, and `local_map` composition.
- Structured trace artifacts, JSON export, optimizer serialization, re-solving,
  placement explanations, and solution diffs.

These capabilities are more valuable to an agent than another code template.
They should become the center of the public interface.

## Current limits that the product must expose honestly

### “Optimal” is conditional

The ILP returns an exact optimum for the candidate strategies and costs it is
given. It does not prove that the plan is the fastest plan on real hardware.
The strategy space may omit an implementation, and the cost model includes
analytical assumptions and topology-specific empirical constants.

Public results should therefore use language such as **cost-model-optimal** and
always identify:

- the strategy-space version;
- the cost-model version;
- the hardware/topology profile;
- the workload shapes used for planning;
- any replication fallbacks or opaque regions;
- whether costs are estimated, measured, or mixed.

### Support is not binary

An operation may be directly supported, supported through a decomposition,
limited to replicated placement, or hidden inside a manually managed
`local_map` region. A single `supports(model) -> bool` would conceal these
important distinctions.

### Fake execution is not full verification

Fake tensors can validate graph construction and shape propagation, but not
numerical equivalence. Fake process groups cannot establish real collective
ordering, performance, or freedom from target-scale failures.

### The user still supplies policy

The current APIs require some combination of a device mesh, representative
inputs, boundary placements, a parameter-memory budget, and optional node
constraints. Dynamic MoE communication remains a manual composition point,
and pipeline parallelism is outside the current optimizer. Agents can help
provide this policy, but AutoParallel should make every assumption explicit.

### PyTorch integration is fragile

AutoParallel relies heavily on evolving PyTorch internals and nightly builds.
That is reasonable for research, but an agent-facing backend needs a supported
compatibility matrix, reproducible environments, and clear failures when an
upstream API changes.

## Proposed agent-facing contract

The primary interface should be a single planning request with serializable
inputs and outputs. A live Python convenience wrapper can build this request,
but the durable contract should be an artifact schema rather than a sequence of
context-manager calls.

### Request

```json
{
  "model": {
    "artifact": "model.exported_program.pt2",
    "content_hash": "..."
  },
  "workload": {
    "training": true,
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

The request must identify an immutable model artifact or exported graph. A
plain “model reference” is insufficient for a reproducible or secure service
because resolving it may execute arbitrary Python and may produce different
graphs in different environments.

### Response

```json
{
  "status": "planned",
  "plan_id": "...",
  "estimated_cost": {
    "total_us": 0.0,
    "compute_us": 0.0,
    "communication_us": 0.0,
    "transition_us": 0.0,
    "peak_memory_bytes": 0
  },
  "coverage": {
    "direct": 0,
    "decomposed": 0,
    "replicate_only": 0,
    "opaque_regions": 0
  },
  "confidence": {
    "level": "medium",
    "reasons": []
  },
  "artifacts": {
    "placements": "placements.json",
    "parallel_module": "parallel_module.pt2",
    "explanation": "explanation.json",
    "visualization": "plan.html"
  },
  "warnings": []
}
```

The response should be deterministic for a fixed artifact, planner version,
cost profile, and request. Every result should be cacheable and comparable.

## Required capabilities

### 1. `analyze` preflight

Replace a Boolean support check with a structured coverage report. It should:

- attempt graph capture with the supplied workload;
- identify the first trace break and its source location;
- classify every tensor-producing region as direct, decomposed,
  replicate-only, opaque, or unsupported;
- report candidate counts and search-space size;
- suggest concrete repairs, such as wrapping a region in `local_map`;
- estimate whether any strategy can satisfy the memory budget.

This report must be useful even when planning cannot proceed.

### 2. `plan_only`

Planning should not require a live target cluster. It should accept an explicit
topology profile and return:

- the selected placements;
- estimated compute, communication, and memory costs;
- the best known baseline plans under the same constraints;
- top alternative plans and their estimated deltas;
- assumptions, fallbacks, and unsupported regions;
- a confidence score based on cost-model calibration coverage.

A fake mesh is an implementation technique, not part of the public contract.

### 3. Machine-readable failures

Every expected failure should have a stable code, structured context, and
repair suggestions. Examples include:

- `TRACE_FAILED`
- `UNSUPPORTED_OP`
- `REPLICATE_ONLY_REGION`
- `INFEASIBLE_CONSTRAINTS`
- `MEMORY_OVER_BUDGET`
- `TOPOLOGY_PROFILE_MISSING`
- `SHAPE_CONTRACT_MISMATCH`
- `PYTORCH_VERSION_UNSUPPORTED`

Errors should include the FX node, operator, module path, source location,
constraint identifiers, and relevant shapes where available. Human-readable
messages remain useful, but they should be rendered from structured data.

### 4. First-class explanations

Promote the existing logs, JSON export, `explain_placement`, solution diffs,
and serialized optimizer state into supported outputs. An explanation should
answer:

- Why was this placement selected?
- Which alternatives were legal?
- Which constraint eliminated a requested placement?
- Which edges dominate communication?
- Which parameters or activations dominate memory?
- What changes if the mesh, memory budget, or boundary layout changes?

Agents should be able to modify one constraint, re-solve, and present the
delta without tracing the model again.

### 5. Tiered verification

`verify_plan` should report separate results rather than one misleading
pass/fail bit:

1. **Structural:** graph validity, shape propagation, placement consistency,
   and collective symmetry.
2. **Simulated numerical:** forward and backward comparison across simulated
   ranks using real tensor values.
3. **Small-cluster execution:** real collectives, numerical comparison,
   checkpoint round trip, and deterministic repeated execution.
4. **Target validation:** memory measurement, compilation behavior,
   throughput, and comparison with baselines on the intended topology.

A plan should carry its highest completed verification tier. Estimated
performance must never be presented as measured performance.

### 6. Profile-guided refinement

The planner should support a bounded feedback loop:

1. Produce a small set of diverse, feasible plans.
2. Benchmark representative operations or short training steps.
3. Calibrate the cost model for the target cluster.
4. Re-solve using the calibrated costs.
5. Retain both predicted and measured results in the plan artifact.

This is a better use of expensive cluster time than unconstrained
trial-and-error generation. It also gives agents objective evidence for
accepting or rejecting a plan.

## Benchmark and trust requirements

Agent ergonomics will not matter if users do not trust the selected plans. The
project needs a public benchmark matrix covering:

- representative dense transformers, encoder-decoder models, FlexAttention,
  and MoE models;
- multiple model sizes and sequence lengths;
- 1D, 2D, and eventually pipeline/hybrid meshes;
- at least two GPU generations and both single- and multi-node systems;
- throughput, peak memory, compilation time, and planning time;
- numerical parity and checkpoint compatibility;
- comparisons with strong hand-written FSDP, TP, and hybrid baselines.

Report wins, ties, and losses. For every loss, identify whether the cause was a
missing strategy, cost-model error, compiler limitation, or runtime overhead.
This dataset should also calibrate the confidence returned by `plan_only`.

## Product priorities

### Phase 1: make planning inspectable

- Define versioned request, plan, diagnostic, and topology schemas.
- Expose plan-only execution and structured coverage analysis.
- Return existing JSON, explanation, diff, and serialization artifacts through
  a stable API.
- Make replication fallbacks visible and optionally fatal.
- Record complete provenance for every plan.

### Phase 2: make plans trustworthy

- Add structural and simulated-numerical verification.
- Establish strong manual baselines and an automated benchmark matrix.
- Add measured-versus-estimated calibration reports.
- Validate actual multi-node behavior on a small set of supported topologies.

### Phase 3: broaden the optimization problem

- Add activation and optimizer-state memory objectives.
- Incorporate pipeline parallelism and richer context/expert parallel choices.
- Improve support for data-dependent and custom operations while retaining
  explicit `local_map` boundaries where static placement abstractions do not
  apply.
- Add profile-guided candidate generation and re-optimization.

### Phase 4: stabilize the backend

- Publish a PyTorch compatibility matrix.
- Isolate private PyTorch integration behind narrow adapters.
- Upstream generally useful DTensor and Inductor functionality where possible.
- Version plan artifacts and provide migration tooling.

## Success criteria

The project should be evaluated on outcomes rather than the amount of
parallelism code it removes:

- Planning succeeds with useful diagnostics on a broad model corpus.
- Unsupported or replicate-only regions are never silent.
- Structural verification catches invalid plans before cluster launch.
- Numerical verification agrees with the unsharded model within declared
  tolerances.
- Estimated rankings correlate with measured rankings on supported hardware.
- Selected plans are competitive with strong human-written baselines.
- Re-planning for a new topology is substantially cheaper than writing and
  validating a new recipe.
- An agent can diagnose and repair common failures using only structured
  outputs.

## Strategic conclusion

If AutoParallel is positioned mainly as a way to avoid writing sharding code,
agents will erode its differentiation. If it becomes the planner, verifier,
and evidence layer beneath agent-generated distributed programs, agents make
it more valuable.

The intended division of labor is:

| Agent | AutoParallel |
|---|---|
| Interpret user goals and deployment constraints | Enumerate legal strategies |
| Produce traceable single-device model semantics | Optimize globally across the graph |
| Add explicit custom regions where necessary | Enforce placement and gradient consistency |
| Choose among documented tradeoffs | Predict and measure cost |
| Explain decisions to the user | Return reproducible plans and evidence |

The long-term moat is not code generation. It is trustworthy search over a
large, changing, hardware-dependent distributed design space.
