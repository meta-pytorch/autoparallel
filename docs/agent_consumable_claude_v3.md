# AutoParallel as an Agent-Facing Planner and Plan Evaluator

## Executive summary

Coding agents are making explicit FSDP and tensor-parallel recipes cheap to
write. That erodes one part of AutoParallel's original pitch. It does not touch
the planning problem underneath, and it raises the rate of architectural change
that makes planning necessary.

The durable promise is not "no manual parallelism code." It is:

> AutoParallel searches *or evaluates* the distributed design space and returns
> a reproducible plan together with the evidence needed to decide whether to
> trust it.

```text
model artifact + workload ──▶┌──────────────────────┐──▶ executable plan
topology + objectives ──────▶│ AutoParallel backend │──▶ cost and coverage
candidate plan ─────────────▶└──────────────────────┘──▶ verification evidence
```

The repositioning is from *automation* to *measurement*. AutoParallel is the
only component in the stack that can price a parallelization plan without
running the job. Agents become clients of that backend: they propose model
code, boundary layouts, constraints, or complete candidate plans, and
AutoParallel supplies graph-level analysis, costed comparison, consistency
checks, and measured feedback.

This document supersedes [agent_consumable.md](agent_consumable.md),
[agent_consumable_claude.md](agent_consumable_claude.md),
[agent_consumable_codex.md](agent_consumable_codex.md),
[agent_consumable_claude_v2.md](agent_consumable_claude_v2.md), and
[agent_consumable_codex_v2.md](agent_consumable_codex_v2.md).

## Concede what agents commoditize

The README's "no manual parallelism code required" is genuinely less
differentiated now. For a known architecture on a fixed cluster, explicit
parallelism code is mechanical, widely exemplified, and increasingly easy to
generate — and a maintained hand-written recipe can be simple, predictable,
and well benchmarked. AutoParallel should not argue that agents cannot produce
such code.

The stronger claim is that code generation and execution planning are
different jobs:

- An agent can emit a plausible plan; AutoParallel can compare it against the
  other legal plans and say what it costs.
- An agent can reuse a known recipe; AutoParallel can re-evaluate that recipe
  for a new batch size, sequence length, mesh, or interconnect.
- An agent can repair model code; AutoParallel can name the graph region and
  the constraint responsible for a failure.
- An agent can launch experiments; AutoParallel can reduce how many expensive
  distributed experiments are needed.

## Why the backend remains valuable

### Kernel agents are the proof, not the counterexample

PTX, CuTe, and Triton generation agents work well, and it is tempting to read
that as a preview of sharding agents. It is closer to the opposite. Those
agents succeed because the loop is tight — compile, run, profile in seconds,
with measured throughput as ground truth — and because they *do not bypass the
stack*. They propose code and still rely on the compiler, the assembler, and
the profiler to say whether it was any good.

A successful codegen agent is a stack consumer. Their success argues that the
layer underneath must exist and must be trustworthy, not that it stops being
needed.

Distributed planning then fails the tight-loop precondition. A wrong kernel
costs seconds; a wrong plan costs a multi-node allocation, produces noisy
measurements, and can fail only at scale. Agents do not make that feedback
free. The job of a planner is to prune the space with structural constraints
and calibrated costs, then spend real cluster time on a few high-value
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

[adaptive_sharding.md](adaptive_sharding.md) shows this concretely. LLaMA3-8B
and LLaMA3-70B, identical training configuration, diverge on `wo`: the 8B keeps
it sequence-parallel, the 70B goes column-parallel. The deciding factors are
that column-parallel yields `P(sum)S(0)` weight gradients — one reduce-scatter
dimension — where sequence-parallel yields `P(sum)P(sum)` and needs a full 2D
reduce-scatter, and that sequence-parallel `wo` emits `S(0)S(1)`, which the
residual add consumes for free.

The same document records a sharper version of the point: sequence-parallel
strategies are only *visible* to the solver when the `view → mm → view` pattern
is fused to preserve the sequence dimension. Without that pass, the
decomposition folds sequence into batch and the entire strategy family
disappears. Whether a strategy can be considered at all is a property of the
compiler's intermediate representation.

### The answer is a function of the deployment

A recipe is an artifact. What users need is closer to a function:

```text
plan = f(graph, shapes, topology, memory, objective, compiler behavior)
```

Batch size, sequence length, mesh shape, GPU generation, and inter-node
bandwidth each move the preferred strategy without touching model source. This
holds even if architectures stabilize.

### The cost model is an asset that cannot be inferred

`cost_models/nccl_cost_model.py` ports NCCL's `tuning.cc` algorithm and
protocol selection — Ring, Tree, CollNet, NVLS, NVLS Tree crossed with LL,
LL128, Simple — then corrects it against nccl-tests measurements on H100
NVSwitch at 1, 2, 4, 8, 16, and 32 nodes.

[adaptive_sharding.md](adaptive_sharding.md) records that the adaptive
behavior *requires* that model: the generic estimator did not price
communication finely enough to separate the candidates. Empirical tables are
not recoverable by inference. No model has them in weights; they are measured
on hardware, and keeping them current is a standing capability rather than a
one-time task.

It is an asset, not an oracle. See the next section for exactly where it is
weak.

## What exists today, and how mature

Support is not binary, and that applies to this list as much as to operators.

| Capability | Implementation | Maturity | Product gap |
|---|---|---|---|
| Joint fwd/bwd capture | `api.py` | Core, CI | no stable graph artifact contract |
| Global placement search | `optimize_sharding.py` | Core, CI | solver reachable only via a stateful protocol |
| DTensor strategy enumeration | `shardings/` | Core, on upstream strategies | coverage and fallback not first-class results |
| Communication modeling | `cost_models/` | H100 NVSwitch 1–32 nodes calibrated | no provenance or calibration reporting |
| Compute modeling | `cost_models/compute_estimation.py` | Analytical, unvalidated | 70% efficiency and 7 µs floor unchecked vs real kernels |
| Executable lowering | `apply_sharding.py` | Core, CI | no independently verifiable plan package |
| Runtime and memory simulation | `estimate_graph_metrics.py` | Works, single peak figure | not wired to a public evaluator; no memory breakdown |
| Plan persistence | `save`/`load`, placement JSON | Works | not a versioned public artifact |
| Counterfactual exploration | constraints, `resolve`, diffs | Works | undocumented as a supported session |
| Placement explanations | `get_json`, `explain_placement` | Works | stdout and trace artifacts only |
| Numerical simulation | `LocalTensorMode` tests | Single GPU | not a supported verifier; upstream blockers |
| Distributed checkpointing | DCP tests | Tested | — |
| Dynamic shapes, inference, mixed precision | dedicated tests | Tested | — |
| FlexAttention | `test_flex_attention` | Tested | — |
| `local_map` / MoE | examples, DSV3 | Examples | TorchTitan DSV3 CI currently disabled |
| TorchTitan integration | `--module autoparallel.llama3` | In CI | — |
| Pipeline parallelism | — | Not supported | — |

`__init__.py` exports five symbols and none of them scores a plan. The
highest-leverage change is connecting these pieces into an API that can score
both solver-generated and caller-proposed plans.

## What the product must say out loud

### "Optimal" means cost-model-optimal

The ILP returns an exact optimum over the candidate strategies and costs it is
given. It does not prove the plan is fastest on real hardware. Every published
result should say **cost-model-optimal** and identify the strategy-space
version, the cost-model version and calibration profile, the topology profile,
the workload shapes, any replication fallbacks or opaque regions, and whether
each cost is measured, interpolated, extrapolated, analytical, or fallback.

### Where the cost model is weak, precisely

Vague honesty is not useful. The specific weaknesses, and which path each
affects:

**On the default path** (`cost_model="nccl"`, which calls
`set_nccl_topo_config(detect_nccl_topo_config(mesh))`):

- Blackwell is extrapolated, not measured. `_BLACKWELL_BW_SCALE = 640.0 / 320.0`
  scales bandwidth from Hopper, and `_A2A_CE_BW[BLACKWELL]` carries the comment
  `estimated proportionally from bw_intra ratio; needs profiling`.
- Compute costs are analytical: FLOPs over device throughput at an assumed 70%
  efficiency, versus bytes over bandwidth at 70%, with a 7 µs launch floor.
  None of that has been checked against real generated kernels.
- Overlap is not predicted. `apply_prefetch_discount` is a caller-selected
  multiplier defaulting to `scale=0.0` — collectives treated as entirely free —
  and it is not applied automatically. The truth is between free and fully
  exposed, and `estimate_graph_metrics` already computes `exposed_comm_time` by
  multi-stream simulation, so a predicted overlap is within reach.

**On the fallback path only** (non-`nccl` cost model, where
`_nccl_topo_config is None`):

- `all_to_all_cost` in `collective_runtime_estimation.py` ends with
  `total_time *= 5` under a `FIXME`. `collective_comm_cost` routes to
  `nccl_all_to_all_cost` whenever the NCCL config is set, so this does not
  affect default planning. It still needs containing: `adaptive_sharding.md`
  concludes that intra-node all-to-all is competitive with all-gather for
  seq-par → head-par SDPA transitions, and a 5× multiplier inverts exactly that
  comparison. A caller who selects the fallback gets a different plan family
  for a reason that is not a property of the hardware.

**These errors do not cancel in the solver.** A bias uniform across all costs
would. A bias on one collective type or one operator class shifts the selected
plan away from every strategy that uses it. Calibration error is a
plan-selection problem first and a reporting problem second.

### Support is not binary, and replicate-only is the dangerous case

An operation may be directly supported, supported through decomposition,
limited to replicated placement, hidden inside a manually specified `local_map`
region, or unsupported.

**Replicate-only regions do not fail.** They produce a correct, working, slow
plan. That is the one failure mode a caller cannot detect from the outcome and
would not think to check. Every replicate-only region must appear in the
coverage report and the plan artifact with its estimated cost, and strict mode
must be able to reject it.

### Fake execution is not verification

Fake tensors validate graph construction, shape propagation, and placement
consistency. They carry no data and say nothing about numerical equivalence. A
fake process group establishes neither collective ordering nor performance nor
freedom from target-scale failure. Any pass/fail returned to a caller must name
which claim it checked.

### The caller still supplies policy

The APIs need some combination of device mesh, representative inputs, boundary
placements, a parameter-memory budget, and optional node constraints.
Data-dependent MoE communication remains a manual composition point, and
pipeline parallelism is outside the optimizer. Agents can help supply this
policy; AutoParallel should make every assumption explicit rather than
defaulting silently.

### Compatibility is part of correctness

AutoParallel depends on private, evolving PyTorch APIs and nightly builds.
Every plan artifact must record the PyTorch build and relevant compiler
configuration, and the project needs a tested compatibility matrix plus narrow
adapters isolating upstream churn. A plan that is not reproducible against a
declared PyTorch version is not a plan.

## Product shape: a planner and an evaluator

### `score_plan`: evaluate a candidate

A caller may already have a plan. Validate and score it rather than insisting
on replacing it.

```python
report = score_plan(model, workload, topology, plan=candidate,
                    objectives=objectives)

report.valid
report.coverage
report.estimated.step_time_us
report.estimated.exposed_communication_us
report.estimated.memory            # by category, see below
report.confidence
report.assumptions
```

This is **not** a wrapper around `load_placements` and
`estimate_graph_metrics`. A credible implementation must:

1. Match the candidate to the captured graph and strategy-space version.
2. Reject placements that are not representable or not legal.
3. Lower the plan into a parallel graph.
4. Validate placement and collective consistency.
5. Estimate the scheduled graph with a documented runtime estimator.
6. Account separately for parameters, activations, optimizer state, and
   temporary collective buffers. Bucketing makes the last category real —
   `max_in_flight_gb` exists as a knob for exactly this reason.
7. Report every opaque or manually costed region.

Note the bound this implies. `load_placements` matches placement strings
against the enumerated strategies, so **externally supplied plans are initially
restricted to strategies AutoParallel can already represent**. Scoring
arbitrary distributed programs is a much larger problem and the API must not
imply it.

### `plan`: search and return candidates

```python
result = plan(model, workload, topology, objectives)

result.selected
result.alternatives     # top-k with cost deltas
result.coverage
result.explanation
result.session_id
```

The selected plan is evaluated through the same `score_plan` path as a
caller-proposed plan. The solver gets no privileged, less rigorous treatment,
and the two sets of numbers stay comparable.

### The warm session

A one-shot call is the right default, but the existing stateful workflow is the
most valuable loop in the optimizer. `add_node_constraint` → `resolve()` →
`diff_solutions` is propose/evaluate/revise, and `resolve()` is cheap by
construction: it re-solves without rebuilding the objective.
`remove_constraints` and `explain_placement` complete it.

Give it a wire format so it survives a service boundary:

```json
{
  "session_id": "...",
  "mutate": [
    {"op": "add_node_constraint", "node": "mm_14", "placement": ["S0", "S1"]}
  ],
  "return": ["estimated", "diff", "explanation"]
}
```

Sessions expire; a stale id returns `PLAN_SESSION_EXPIRED` and the caller
replays the one-shot request. Do not force every counterfactual to retrace the
model and rebuild the ILP.

## The artifact contract

Local Python wrappers may accept an `nn.Module`, but a service boundary must
not rely on a mutable model reference: resolving one executes arbitrary Python
and may produce different graphs in different environments.

### Request

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
  "constraints": [],
  "policy": {"replicate_only_is_fatal": false}
}
```

### Response

```json
{
  "schema_version": 1,
  "status": "planned",
  "plan_id": "sha256:...",
  "provenance": {
    "planner_version": "...",
    "strategy_space_version": "...",
    "cost_model_version": "...",
    "calibration_profile": "h100_nvswitch_400g_v1",
    "pytorch_version": "..."
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
    }
  },
  "coverage": {
    "direct": 0, "decomposed": 0,
    "replicate_only": 0, "opaque": 0, "unsupported": 0
  },
  "confidence": {
    "level": "medium",
    "critical_path_provenance": [
      {"collective": "all_gather", "share_of_comm": 0.61, "basis": "measured"},
      {"collective": "all_to_all", "share_of_comm": 0.31, "basis": "extrapolated"}
    ],
    "reasons": []
  },
  "verification": {"highest_completed_tier": "structural"},
  "artifacts": {},
  "warnings": []
}
```

For a fixed artifact, workload, topology, constraints, and planner version, the
result is deterministic, cacheable, and comparable.

## Required capabilities

### 1. Calibration provenance and critical-path confidence

Tag every returned cost `measured`, `interpolated`, `extrapolated`,
`analytical`, or `fallback`.

Derive confidence from the operations and collectives **on the critical path,
weighted by their share of it** — not from a single global label. One poorly
calibrated dominant all-to-all should lower confidence far more than a hundred
insignificant analytical pointwise costs.

First calibration work, in priority order:

- Measure Blackwell all-to-all directly (already item 1 of the TODO block in
  `nccl_cost_model.py`).
- Compare compute estimates against representative generated kernels; validate
  or replace the 70% efficiency assumption and the 7 µs floor.
- Replace the prefetch discount with a *predicted* overlap, using the
  multi-stream simulation that `estimate_graph_metrics` already performs.
- Contain the fallback all-to-all heuristic, or route it to the NCCL path.
- Record the exact topology and software stack for every measurement.

### 2. Structured coverage analysis

Replace `supports(model) -> bool` with an `analyze` report that is useful even
when planning cannot proceed. It should attempt capture, report the first trace
break with its source location, classify every relevant region as direct,
decomposed, replicate-only, opaque, or unsupported, report candidate counts and
search-space size, and say whether the memory budget looks satisfiable at all.

Present repairs such as "wrap this region in `local_map`" as *possible* fixes.
Do not assert them: the tool cannot infer the intended distributed semantics of
a region it failed to analyze.

Generated model code trends dynamic and op-inventive while tracing assumes
mostly-static FX-traceable graphs, so this is the most frequently hit path for
an agent caller — and a human debugging a 512-GPU job wants the same report.

### 3. Machine-readable failures

Stable codes, structured context, human-readable messages rendered *from* the
data rather than instead of it.

```json
{
  "error": "REPLICATE_ONLY_REGION",
  "node": "mm_14",
  "module_path": "layers.3.attention.wq",
  "op": "aten.mm.default",
  "source": {"file": "model.py", "line": 142},
  "shapes": [[8192, 4096], [4096, 4096]],
  "detail": "no sharded strategy available; plan falls back to replication",
  "estimated_cost_us": 812.0
}
```

Codes: `TRACE_FAILED`, `UNSUPPORTED_OP`, `REPLICATE_ONLY_REGION`,
`INFEASIBLE_CONSTRAINTS`, `MEMORY_OVER_BUDGET`, `TOPOLOGY_PROFILE_MISSING`,
`INPUT_SHAPE_MISMATCH`, `MESH_MISMATCH`, `PYTORCH_VERSION_UNSUPPORTED`,
`PLAN_SESSION_EXPIRED`.

Repair hints must be derived from known facts. In particular, naming the
minimal contradictory subset of ILP constraints requires an irreducible
infeasible subset computation that CBC does not hand you for free. Until that
work is done, `INFEASIBLE_CONSTRAINTS` should report the constraints that were
added and by whom, not pretend to identify the culprit.

`export_json.py` already extracts node, `module_path`, and source location.

### 4. Explanations as return values

Promote `get_json`, `get_log`, `explain_placement`, `print_costs_for_node`,
`diff_solutions`, optimizer serialization, and
`visualizer/build_display_from_json.py` into supported data products. An
explanation should answer:

- Why was this placement selected?
- Which alternatives were legal, and what did they cost?
- Which constraint eliminated a requested placement?
- Which edges dominate communication? Which tensors dominate memory?
- What changes under a different mesh, budget, or boundary layout?
- Where did the estimate rely on extrapolation or a fallback?

### 5. Tiered verification

Report distinct claims, never one pass/fail bit. A plan carries its highest
completed tier, and **estimated performance is never presented as measured
performance**.

| Tier | Checks | Substrate |
|---|---|---|
| 1 Structural | graph validity, shape propagation, legal placements, boundary contracts, collective symmetry | fake tensors |
| 2 Simulated numerical | forward and backward parity vs unsharded reference, real values | `LocalTensorMode`, single GPU |
| 3 Small-cluster | real collectives, gradient parity, checkpoint round trip, repeat determinism, compilation | small real mesh |
| 4 Target | measured memory, throughput, compile behavior vs baselines | intended topology |

Tier 2 exists in `tests/test_correctness.py`. Its docstring records the two
upstream blockers keeping it off the `apply_placement()` path, and those are
the critical path to shipping it:

1. `ProcessGroup` objects from `compile_on_one_rank` are not deepcopy-safe,
   breaking `extract_forward_graph`'s deepcopy of the joint graph.
2. AOT autograd's compiled backward rejects `LocalTensor` tangents because
   `LocalTensor` does not implement `__coerce_same_metadata_as_tangent__`.

**Calibration gates trustworthy performance scoring. It does not gate
structural or numerical verification, or coverage reporting.** Those advance
independently.

### 6. Profile-guided refinement

Make calibration a runtime capability, or the cost model never reaches hardware
nobody has profiled:

1. Produce a small, diverse set of feasible plans.
2. Benchmark representative collectives or short training steps.
3. Update a topology-specific calibration profile.
4. Re-score, and re-solve when the calibration moves a decision.
5. Keep predicted *and* measured values in the plan artifact, promoting the
   affected entries from `extrapolated` to `measured`.

Each cluster-hour becomes a permanent improvement to the asset that
differentiates the project, instead of one more trial-and-error run.

## Scope boundaries

**No pipeline parallelism.** The optimizer decides intra-stage placement; the
`_pipelined_*` symbols in `graph_passes/async_tp/` are micro-pipelined TP and
unrelated. At frontier scale, pipeline partitioning and scheduling are not
optional. State the limitation, and plan to compose with pipeline stages rather
than to absorb them soon.

**`local_map` is a legitimate boundary with incomplete visibility.** It is the
right escape hatch for data-dependent layouts that static DTensor placements
cannot express, and eliminating it is not a realistic goal — routing is
data-dependent, and `cute_sharding_design.md` itself concedes that ragged and
uneven strategies fall outside a single affine layout. The problem is not that
the boundary exists; it is that the planner cannot see inside it. Require an
optional **cost and memory contract** for opaque regions so `score_plan` can
include them rather than trusting them, and let richer layout representations
shrink the opaque area where static layouts genuinely do apply.

**Replication fallback must never be silent.** Covered above; it belongs in
this list because it is a scope boundary the caller inherits.

## Benchmarks and trust

The question that gates adoption has never been published: **are the solver's
plans competitive with what a good engineer writes by hand?**

### Minimal viable benchmark

Ship this first, so trust work has something concrete within weeks:

- LLaMA3-8B, two mesh shapes (1D and 2D), one node count;
- one strong hand-written TorchTitan FSDP+TP baseline;
- throughput, exposed communication, peak memory, planning time, numerical
  parity;
- predicted versus measured cost for every collective on the critical path.

That last line is the important one: it is simultaneously a benchmark result
and the first entry in the calibration profile.

### Full matrix

Dense decoder, encoder-decoder, FlexAttention, and MoE; multiple scales,
sequence lengths, and batch sizes; 1D and 2D meshes then hybrids; at least two
GPU generations, single- and multi-node; throughput, exposed communication,
peak memory, compile time, planning time; numerical parity and DCP
compatibility; strong hand-written FSDP, TP, and hybrid baselines.

Report wins, ties, **and losses**. Classify each loss as a strategy-space gap,
cost-model error, compiler limitation, runtime overhead, or unsupported
communication pattern. That attribution turns a benchmark into a roadmap, and
the same measurements calibrate the confidence returned by `score_plan`.

## Delivery sequence

Sized for the team that exists rather than four parallel tracks. This repository
has one dominant contributor and a long tail; parallel tracks presume parallel
people. Ordered so each step ships something usable.

1. **Coverage and honesty.** `analyze` with the region taxonomy; replicate-only
   surfaced everywhere and rejectable in strict mode; machine-readable errors.
   No dependency on anything else, and it improves the human experience today.
2. **Explanations and the session.** Promote `get_json`, `explain_placement`,
   and `diff_solutions` to return values; version the request, plan, topology,
   and diagnostic schemas; expose the warm session with a wire format.
3. **Minimal viable benchmark plus provenance.** The LLaMA3-8B comparison
   above, and per-cost provenance tagging wired into confidence.
4. **`score_plan` for representable plans**, with clearly labeled estimates and
   the solver's own plan routed through the same path.
5. **Verification tiers 1 and 2**, including the two upstream fixes. Runs in
   parallel with 3 and 4 — it does not wait on calibration.
6. **Profile-guided refinement and the full benchmark matrix.**
7. **Breadth and stability.** Activation, optimizer-state, and buffer memory
   objectives; pipeline composition; cost contracts for opaque regions;
   compatibility matrix and narrow PyTorch adapters.

## Success criteria

Evaluate on outcomes, not on parallelism code removed:

- A broad model corpus gets useful coverage reports even when planning fails.
- Unsupported, opaque, extrapolated, and replicate-only regions are never
  silent.
- Structural verification catches invalid plans before cluster launch.
- Numerical verification matches the unsharded model within declared
  tolerances.
- Estimated rankings correlate with measured rankings on supported topologies.
  Rank correlation is the target, not absolute error.
- Solver-generated plans are scored through the same path as external plans.
- Selected plans are competitive with strong hand-written baselines, and every
  loss has an attributed cause.
- Re-planning for a new topology is cheaper than authoring and validating a new
  recipe.
- A caller can diagnose common failures and explain tradeoffs from structured
  output alone.
- Plan artifacts reproduce across the declared compatibility window.

## Division of labor

| Agent or human | AutoParallel |
|---|---|
| Interpret goals and deployment constraints | Capture and analyze the derived graph |
| Produce traceable model semantics | Enumerate legal distributed strategies |
| Propose constraints or complete plans | Search, score, and compare plans |
| Mark explicit custom regions where needed | Validate boundaries and include declared costs |
| Run the selected experiments | Calibrate predictions from measurements |
| Explain the decision to the user | Return reproducible evidence and diagnostics |

## If you only do one thing

Make replicate-only regions loud, and ship `score_plan` over representable
plans with honest provenance.

The first closes the failure mode neither an agent nor a human can detect from
the outcome: a plan that works, is correct, and is quietly slow. The second
changes what AutoParallel is *for* — from a thing that writes your parallelism
to the only thing that can price it without burning a cluster allocation.

The long-term moat is not code generation. It is trusted planning,
measurement, and verification beneath generated distributed programs — and
trust is the half that is currently missing.
