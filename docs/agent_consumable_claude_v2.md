# AutoParallel in an Agent-Driven World

## Executive summary

Coding agents are making explicit FSDP, tensor-parallel, and checkpointing
code cheap to write. That erodes AutoParallel's value as a convenience layer.
It does not touch the planning problem underneath, and it raises the rate of
architectural change that makes planning necessary.

The durable promise is not "no one has to write parallelism code." It is:

> AutoParallel searches the legal strategy space, prices the tradeoffs
> against a hardware model, produces an executable plan, and supplies the
> evidence needed to trust it.

```text
model artifact + workload + topology + objectives
                         |
                         v
                AutoParallel planner
                         |
                         v
       plan + evidence + generated module + diagnostics
```

The repositioning is from *automation* to *measurement*. AutoParallel is the
only component in the stack that can price a parallelization plan without
running the job — including plans it would not have generated itself. That
capability is what an agent needs and cannot synthesize, and it is currently
buried behind a context manager and a log.

This document supersedes [agent_consumable.md](agent_consumable.md),
[agent_consumable_claude.md](agent_consumable_claude.md), and
[agent_consumable_codex.md](agent_consumable_codex.md).

## Concede the real point first

The line on the front page of the README — "no manual parallelism code
required" — is the part of the pitch that agents genuinely erode.

A TorchTitan-style parallelization plan is a couple hundred lines of
`parallelize_module(model, {"attention.wq": ColwiseParallel(), ...})`. It is
mechanical, thoroughly exemplified in public code, and current agents write it
competently for a standard transformer. The historical argument — every new
architecture needs a hand-written plan and there are not enough experts to
write them — is weaker than it was.

Say so first. The rest of the case is much stronger than the part being
conceded, and a document that defends everything persuades no one.

## Why the planner survives

### Kernel agents are the proof, not the counterexample

PTX, CuTe, and Triton generation agents work well, and it is tempting to read
that as a preview of sharding agents. It is closer to the opposite. Those
agents succeed because the loop is tight — compile, run, profile in seconds,
with measured throughput as ground truth — and because they *do not bypass the
stack*. They propose code and still rely on the compiler, the assembler, and
the profiler to tell them whether it was any good.

A successful codegen agent is a stack consumer. Their success argues that the
layer underneath must exist and must be trustworthy, not that it stops being
needed.

Sharding then fails the tight-loop precondition in the way that matters most.
A wrong kernel costs seconds. A wrong plan costs a multi-node allocation, and
its failure modes are non-local: deadlock, OOM that appears only at scale, or
a silent throughput cliff that looks like a working job. Nothing in that loop
supplies cheap ground truth. Supplying it is the product.

### The optimization target has no source form

AutoParallel does not optimize the model. It optimizes the joint
forward+backward graph produced by `aot_export_joint_with_descriptors`
(`api.py`). The backward nodes an agent would have to reason about do not
exist in `model.py`; AOTAutograd has not emitted them yet.

This is not "an agent would do a worse job at the same problem." It is a
different problem over a different object, and
[adaptive_sharding.md](adaptive_sharding.md) contains the proof. LLaMA3-8B and
LLaMA3-70B, identical training configuration, diverge on `wo`: the 8B keeps it
sequence-parallel, the 70B goes column-parallel. The deciding factors are that
column-parallel yields `P(sum)S(0)` weight gradients — one reduce-scatter
dimension — where sequence-parallel yields `P(sum)P(sum)` and needs a full 2D
reduce-scatter, and that sequence-parallel `wo` emits `S(0)S(1)`, which the
residual add consumes for free.

Neither fact is visible from model source at any level of reasoning skill.
Both are properties of a graph that does not exist until after tracing.

### The answer is a function of the deployment, not the model

Holding the architecture fixed, the optimal plan still moves with batch size,
sequence length, mesh shape, node count, and interconnect. Batch size alone
shifts the solver between DP-dominant and TP-dominant regimes.

An agent that writes parallelism code emits an *artifact*. What is needed is a
*function*, re-evaluated per cluster and per configuration. This argument
holds even for a reader who believes architectures will stop churning.

### The cost model is measured, not inferable

`cost_models/nccl_cost_model.py` ports NCCL's `tuning.cc` algorithm and
protocol selection — Ring, Tree, CollNet, NVLS, NVLS Tree crossed with LL,
LL128, Simple — then corrects it against nccl-tests measurements on H100
NVSwitch at 1, 2, 4, 8, 16, and 32 nodes.

[adaptive_sharding.md](adaptive_sharding.md) records that the adaptive
sequence-parallel behavior *requires* that model: the generic estimator did
not price communication finely enough to separate the candidates. Empirical
tables are not recoverable by inference. No model has them in weights and no
amount of reasoning reconstructs them. They are measured on hardware, and
keeping them current is a standing capability rather than a one-time task
(see [Profile-guided refinement](#7-profile-guided-refinement)).

### Recipes go stale fastest where agents push hardest

An agent can recite the Megatron placement recipe for LLaMA. Agentic coding
produces architecture variants faster than recipes get written, which is
exactly the regime where memorized recipes fail and a graph-in, solver-out
backend earns its keep. Agents raise the rate of architectural change, and
that makes the hand-tuned plan the bottleneck rather than the model code.

## What already exists, and how mature it is

Support is not binary — and that applies to this list as much as to operators.

| Capability | Maturity | Evidence |
|---|---|---|
| Joint fwd/bwd capture (Dynamo + AOTAutograd) | Core | CI |
| DTensor strategy enumeration | Core, on upstream strategies | CI |
| Global ILP over compute/comm/transition/memory | Core | CI |
| Lowering a plan to an executable module | Core | CI |
| `torch.compile` backend (AC, bucketing, overlap) | Core | CI |
| NCCL cost model | Calibrated H100 NVSwitch, 1–32 nodes | A100 partial; B200 extrapolated |
| Numerical correctness vs unsharded reference | Single GPU via `LocalTensorMode` | `apply_placement` path blocked upstream |
| Distributed checkpointing | Tested | `test_dcp_roundtrip`, `test_dcp_ordered_sharding` |
| Dynamic batch shapes, inference, mixed precision | Tested | dedicated test files |
| FlexAttention | Tested | `test_flex_attention` |
| `local_map` / MoE composition | Examples + DSV3 | TorchTitan DSV3 CI currently disabled |
| TorchTitan integration | LLaMA3 in CI | `--module autoparallel.llama3` |
| Pipeline parallelism | Not supported | — |

Plan artifacts, JSON export, optimizer serialization, re-solving, placement
explanations, and solution diffs all exist. They are reachable only through
logs, `trace_structured` events, and stdout. `__init__.py` exports five
symbols and none of them scores a plan.

These capabilities are worth more to an agent than another code template.
They should become the center of the public interface.

## Limits the product must state out loud

### "Optimal" is conditional: say cost-model-optimal

The ILP returns an exact optimum over the candidate strategies and costs it is
given. It does not prove the plan is fastest on real hardware. The strategy
space may omit an implementation, and the cost model carries analytical
assumptions plus topology-specific empirical constants.

Every published result should say **cost-model-optimal** and identify:

- the strategy-space version;
- the cost-model version and calibration profile;
- the hardware/topology profile;
- the workload shapes used for planning;
- any replication fallbacks or opaque regions;
- whether each cost is measured, interpolated, extrapolated, or modeled.

### The cost model has known-broken paths today

This is the gap between the honesty framing above and shipping it. Three
specific defects are acceptable inside the solver, where only relative
ordering matters and consistent bias largely cancels, and are not acceptable
in an exported number:

- `all_to_all_cost` in `cost_models/collective_runtime_estimation.py` ends
  with `total_time *= 5` under a `FIXME: this is a hack, we need to spend some
  more effort on the cost model`.
- Blackwell bandwidth is ratio-scaled from Hopper
  (`_BLACKWELL_BW_SCALE = 640.0 / 320.0`) and `_A2A_CE_BW[BLACKWELL]` carries
  the comment `estimated proportionally from bw_intra ratio; needs profiling`.
- `apply_prefetch_discount` defaults to `scale=0.0` — collectives treated as
  entirely free — and is *not* applied automatically, so the default solve
  systematically over-prices sharded parameters.

A `confidence: "medium"` field does not repair a systematically wrong value;
it labels it. A caller told "medium confidence, 400 µs" behaves very
differently from one told "this path carries an unexplained ×5 correction."
Repairing these paths gates the planning API, not the other way round.

### Support is not binary, and replicate-only is the dangerous case

An operation may be directly supported, supported through a decomposition,
limited to replicated placement, hidden inside a manually managed `local_map`
region, or unsupported. A `supports(model) -> bool` conceals the distinction
that matters most.

**Replicate-only regions do not fail.** They silently produce a correct,
working, slow plan. That is the one failure mode a caller cannot detect from
the outcome and would not think to check. Replication fallbacks must be
visible in every response and configurable as fatal.

### Fake execution is not verification

Fake tensors validate graph construction, shape propagation, and placement
consistency. They carry no data, so they say nothing about numerical
equivalence. A fake process group establishes neither real collective ordering
nor performance nor freedom from target-scale failures. Any pass/fail returned
to a caller must name which of those it checked.

### The user still supplies policy

The current APIs need some combination of a device mesh, representative
inputs, boundary placements, a parameter-memory budget, and optional node
constraints. Data-dependent MoE communication remains a manual composition
point, and pipeline parallelism is outside the optimizer. Agents can help
supply this policy; AutoParallel should make every assumption explicit rather
than defaulting silently.

### PyTorch integration is fragile

AutoParallel depends heavily on evolving PyTorch internals and nightly builds.
That is fine for research. An agent-facing backend needs a compatibility
matrix, reproducible environments, and a clear, coded failure when an upstream
API moves.

## The contract

The primary interface is a planning request with serializable inputs and
outputs. A Python convenience wrapper can build the request, but the durable
contract is an artifact schema, not a sequence of context-manager calls.

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
  "constraints": [],
  "policy": {"replicate_only_is_fatal": false}
}
```

The request must identify an **immutable artifact or exported graph**. A plain
model reference is insufficient for anything reproducible or cacheable:
resolving it executes arbitrary Python and may yield different graphs in
different environments.

### Response

```json
{
  "status": "planned",
  "plan_id": "...",
  "estimated_cost": {
    "total_us": 0.0,
    "compute_us": 0.0,
    "communication_us": 0.0,
    "exposed_communication_us": 0.0,
    "transition_us": 0.0,
    "peak_memory_bytes": 0,
    "oom_risk": "low"
  },
  "coverage": {
    "direct": 0,
    "decomposed": 0,
    "replicate_only": 0,
    "opaque_regions": 0,
    "unsupported": 0
  },
  "confidence": {
    "level": "medium",
    "cost_provenance": {
      "all_gather": "measured",
      "reduce_scatter": "measured",
      "all_to_all": "modeled"
    },
    "reasons": []
  },
  "verification_tier": 1,
  "provenance": {
    "planner_version": "...",
    "strategy_space_version": "...",
    "cost_model_version": "...",
    "topology_profile": "h100_nvswitch_400g_v1"
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

Responses are deterministic for a fixed artifact, planner version, cost
profile, and request. Every result is cacheable and comparable.

### The warm session

A one-shot schema cannot express "change one constraint and re-solve without
re-tracing," which is the single most valuable loop the optimizer already
supports. `add_node_constraint` → `resolve()` → `diff_solutions` is exactly
propose/evaluate/revise, and `resolve()` is cheap by construction: it re-solves
without rebuilding the objective. `remove_constraints` and `explain_placement`
complete it.

Keep it, and give it a home in the contract: `plan_id` keys a warm planner.

```json
{
  "plan_id": "...",
  "mutate": [
    {"op": "add_node_constraint", "node": "mm_14", "placement": ["S0", "S1"]}
  ],
  "return": ["estimated_cost", "diff"]
}
```

Sessions expire; a stale `plan_id` returns `PLAN_SESSION_EXPIRED` and the
caller replays the one-shot request. Stateless is the default door, not the
only one.

## Required capabilities

### 1. Repair the exportable cost paths

**This gates capability 3 and capability 6.** Fit a real model for
`all_to_all` and delete the ×5. Profile AllToAll on Blackwell NVSwitch — it is
already item 1 of the TODO block in `nccl_cost_model.py`. Decide whether
`apply_prefetch_discount` should be on by default, and if the answer is yes,
make it so; shipping an oracle whose default configuration misprices FSDP is
worse than shipping none.

Then attach per-collective provenance — `measured`, `interpolated`,
`extrapolated`, `modeled` — to every cost the API returns. A caller who knows
a number is extrapolated can weight it; a caller handed a bare float cannot.

### 2. `analyze` preflight with a coverage taxonomy

Replace a Boolean support check with a structured coverage report that is
useful even when planning cannot proceed. It should:

- attempt graph capture with the supplied workload;
- report the first trace break and its source location;
- classify every tensor-producing region as direct, decomposed,
  replicate-only, opaque, or unsupported;
- report candidate counts and search-space size;
- suggest concrete repairs, such as wrapping a region in `local_map`;
- estimate whether any strategy can satisfy the memory budget.

Generated model code trends dynamic and op-inventive while tracing assumes
mostly-static FX-traceable graphs, so this is the most frequently hit path for
an agent caller. It is also plain good engineering: a human debugging a
512-GPU job wants the same report.

### 3. `plan` and `plan_only`

Planning must not require a live target cluster. The two halves already exist
and are not connected: `load_placements` takes a complete plan as JSON, and
`estimate_graph_metrics` returns critical-path time, compute time,
communication time, exposed communication time, and peak memory. Wire them
together, accept an explicit topology profile, and export the result.

```python
from autoparallel import plan, score_plan

report = score_plan(artifact, workload, topology, plan="placements.json")
report.estimated_cost.exposed_communication_us
report.coverage.replicate_only
report.confidence.cost_provenance
```

`score_plan` accepting a *caller-supplied* plan is what makes this useful to an
agent with its own opinions, and to a human comparing against a hand-written
TorchTitan baseline. Return alternatives too: the solver's own plan, the top-k
neighbours with cost deltas, and the assumptions behind each.

A fake mesh is an implementation technique, not part of the public contract.

### 4. Machine-readable failures

Every expected failure gets a stable code, structured context, and a repair
suggestion. Human-readable messages are rendered *from* the structured data,
not instead of it.

```json
{
  "error": "INFEASIBLE_CONSTRAINTS",
  "node": "mm_14",
  "module_path": "layers.3.attention.wq",
  "op": "aten.mm.default",
  "source": {"file": "model.py", "line": 142},
  "detail": "node constraint S(0)S(1) conflicts with parameter memory budget",
  "suggested_fix": {
    "action": "relax_constraint",
    "constraint": "memory_constraint_high"
  }
}
```

Codes: `TRACE_FAILED`, `UNSUPPORTED_OP`, `REPLICATE_ONLY_REGION`,
`INFEASIBLE_CONSTRAINTS`, `MEMORY_OVER_BUDGET`, `TOPOLOGY_PROFILE_MISSING`,
`SHAPE_CONTRACT_MISMATCH`, `PYTORCH_VERSION_UNSUPPORTED`,
`PLAN_SESSION_EXPIRED`.

`export_json.py` already extracts node, `module_path`, and source location.

### 5. Explanations as data

Promote `get_json`, `get_log`, `explain_placement`, `print_costs_for_node`,
`diff_solutions`, and `visualizer/build_display_from_json.py` from logs and
stdout into returned artifacts. An explanation should answer:

- Why was this placement selected?
- Which alternatives were legal, and what did they cost?
- Which constraint eliminated a requested placement?
- Which edges dominate communication?
- Which parameters or activations dominate memory?
- What changes if the mesh, budget, or boundary layout changes?

A caller that can explain *why* can hand the reasoning to a human. That is the
difference between a tool that is trusted and one that is second-guessed.

### 6. Tiered verification

`verify_plan` reports separate results, never one pass/fail bit. A plan
carries its highest completed tier, and **estimated performance is never
presented as measured performance**.

| Tier | Checks | Substrate |
|---|---|---|
| 1 Structural | graph validity, shape propagation, placement consistency, collective symmetry | fake tensors |
| 2 Simulated numerical | forward and backward vs unsharded reference, real values | `LocalTensorMode`, single GPU |
| 3 Small-cluster | real collectives, numerical parity, checkpoint round trip, repeat determinism | small real mesh |
| 4 Target | peak memory, compile behavior, throughput vs baselines | intended topology |

Tier 2 already exists in `tests/test_correctness.py`. Its docstring records the
two upstream blockers keeping it off the `apply_placement()` path, and those
are the critical path to shipping it as an API:

1. `ProcessGroup` objects from `compile_on_one_rank` are not deepcopy-safe,
   breaking `extract_forward_graph`'s deepcopy of the joint graph.
2. AOT autograd's compiled backward rejects `LocalTensor` tangents because
   `LocalTensor` does not implement `__coerce_same_metadata_as_tangent__`.

### 7. Profile-guided refinement

Calibration must be a runtime capability, not a maintenance chore, or the cost
model never reaches hardware nobody has profiled. Support a bounded loop:

1. Produce a small set of diverse, feasible plans.
2. Benchmark representative collectives or short training steps.
3. Calibrate the cost model for the target cluster.
4. Re-solve against calibrated costs.
5. Retain predicted *and* measured values in the plan artifact, and promote
   the affected entries from `modeled` to `measured` in `cost_provenance`.

This is a far better use of expensive cluster time than unconstrained
trial-and-error, and it converts each cluster-hour into a permanent
improvement to the asset that differentiates the project.

## Benchmarks and trust

Ergonomics will not matter if the plans are not trusted, and the question that
gates adoption has never been published: **are the solver's plans competitive
with what a good engineer writes by hand?**

### Minimal viable benchmark

Ship this before the full matrix, so Phase 2 has something concrete:

- LLaMA3-8B, two mesh shapes (1D and 2D), one node count;
- one strong hand-written TorchTitan FSDP+TP baseline;
- throughput, peak memory, planning time, numerical parity;
- predicted versus measured cost for every collective on the critical path.

### Full matrix

- dense transformers, encoder-decoder, FlexAttention, and MoE;
- multiple model sizes and sequence lengths;
- 1D, 2D, and eventually hybrid meshes;
- at least two GPU generations, single- and multi-node;
- throughput, peak memory, compile time, planning time;
- numerical parity and checkpoint compatibility;
- comparison with strong hand-written FSDP, TP, and hybrid baselines.

Report wins, ties, **and losses**. For every loss, attribute the cause: missing
strategy, cost-model error, compiler limitation, or runtime overhead. That
attribution turns a benchmark into a roadmap, and the same dataset calibrates
the confidence returned by `plan_only`.

## Scope boundaries

An agent-facing tool that overstates coverage is worse than one that declines.
Both boundaries belong in `analyze` output and in the docs.

**No pipeline parallelism.** Nothing in the repository does PP; the
`_pipelined_*` symbols in `graph_passes/async_tp/` are micro-pipelined TP and
unrelated. PP is graph partitioning and scheduling across stages, and at
frontier scale it is not optional. AutoParallel decides intra-stage
parallelism. Say so.

**`local_map` is a real boundary, not a temporary gap.** The hybrid shape has
the caller writing `local_map` regions for data-dependent MoE communication —
which means the caller chooses the expert-parallel strategy for the most
communication-sensitive part of a modern model. It is tempting to plan on
eliminating the escape hatch, and that is probably over-optimistic: routing is
data-dependent, and `cute_sharding_design.md` itself concedes that ragged and
uneven strategies fall outside a single affine layout. The realistic position
is to keep explicit boundaries and **price what is inside them** — extend
`score_plan` to cost `local_map` regions rather than trusting them, and let the
CuTe layout work shrink the hatch where static layouts genuinely do apply.

## Phases

### Phase 1 — make planning inspectable and honest

- Repair the exportable cost paths (capability 1). Gates the rest of Phase 1.
- Define versioned request, plan, diagnostic, and topology schemas.
- Ship `analyze` and `plan_only`, including `score_plan` over caller plans.
- Make replication fallbacks visible and optionally fatal.
- Return existing JSON, explanation, diff, and serialization artifacts through
  a stable API; add the warm session keyed by `plan_id`.
- Record complete provenance on every plan.

### Phase 2 — make plans trustworthy

- Tier 1 and Tier 2 verification, including the two upstream fixes.
- Minimal viable benchmark, then the full matrix.
- Measured-versus-estimated calibration reports.
- Validate real multi-node behavior on a small set of supported topologies.

### Phase 3 — broaden the optimization problem

- Activation and optimizer-state memory objectives.
- Pipeline parallelism; richer context- and expert-parallel choices.
- Cost `local_map` regions; improve data-dependent and custom op support.
- Profile-guided candidate generation and re-optimization (capability 7).

### Phase 4 — stabilize the backend

- Publish a PyTorch compatibility matrix.
- Isolate private PyTorch integration behind narrow adapters.
- Upstream generally useful DTensor and Inductor functionality.
- Version plan artifacts and provide migration tooling.

## Success criteria

Evaluate on outcomes, not on how much parallelism code was removed:

- Planning succeeds with useful diagnostics across a broad model corpus.
- Unsupported and replicate-only regions are never silent.
- Structural verification catches invalid plans before cluster launch.
- Numerical verification agrees with the unsharded model within declared
  tolerances.
- Estimated rankings correlate with measured rankings on supported hardware.
  Rank correlation is the target, not absolute error.
- Selected plans are competitive with strong human-written baselines, and
  every loss has an attributed cause.
- Re-planning for a new topology is substantially cheaper than writing and
  validating a new recipe.
- A caller can diagnose and repair common failures from structured output
  alone.

## Division of labor

| Agent or human | AutoParallel |
|---|---|
| Interpret goals and deployment constraints | Enumerate legal strategies |
| Produce traceable single-device semantics | Optimize globally over the joint graph |
| Mark explicit custom regions where needed | Enforce placement and gradient consistency |
| Choose among documented tradeoffs | Predict, then measure, cost |
| Explain the decision to the user | Return reproducible plans and evidence |

## If you only do one thing

Ship `plan_only` / `score_plan` with repaired cost paths and honest
provenance, and make replicate-only regions loud.

The first changes what AutoParallel is *for* — from a thing that writes your
parallelism to the only thing that can price it without burning a cluster
allocation. The second closes the failure mode that neither an agent nor a
human can detect from the outcome: a plan that works, is correct, and is
quietly slow.

The long-term moat is not code generation. It is trustworthy search over a
large, changing, hardware-dependent design space — and the trust half is the
half that is currently missing.
