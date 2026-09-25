# Making AutoParallel Agent-Consumable

AutoParallel was designed for a human in the loop: build a model, enter a
context manager, add constraints, read a log, iterate. The likely caller in a
year is an agent. This document argues that adapting to that is a
repositioning rather than a retrofit, and proposes a plan ordered by
dependency rather than by convenience.

It expands on [agent_consumable.md](agent_consumable.md), which covers the
same ground more briefly.

## Concede the real point first

The line on the front page of the README — "no manual parallelism code
required" — is the part of the pitch that agents genuinely erode.

Writing a TorchTitan-style parallelization plan is a couple hundred lines of
`parallelize_module(model, {"attention.wq": ColwiseParallel(), ...})`. It is
mechanical, thoroughly exemplified in public code, and current agents write
it competently for a standard transformer. The historical argument — every
new architecture needs a hand-written plan, and there are not enough experts
to write them — is weaker than it was.

Stating this up front matters. The rest of the argument is considerably
stronger than the part being conceded, and a document that defends
everything persuades no one.

## Why the backend survives anyway

### Kernel agents are the proof, not the counterexample

PTX, CuTe, and Triton generation agents work well, and it is tempting to
read that as a preview of sharding agents. It is closer to the opposite.
Those agents succeed because the loop is tight — compile, run, profile in
seconds, with measured throughput as ground truth — and because they *do not
bypass the stack*. They propose code and still rely on the compiler, the
assembler, and the profiler to tell them whether it was any good.

A successful codegen agent is a stack consumer. The lesson is that the layer
underneath has to exist and has to be trustworthy, not that it stops being
needed.

Sharding then fails the tight-loop precondition in the way that matters
most. A wrong kernel costs seconds. A wrong sharding plan costs a multi-node
launch, and its failure modes are non-local: deadlock, OOM that only appears
at scale, or a silent throughput cliff that looks like a working job.
Nothing in that loop supplies cheap ground truth. Supplying it is the
product.

### The optimization target has no source form

AutoParallel does not optimize the model; it optimizes the joint
forward+backward graph produced by `aot_export_joint_with_descriptors`
(`api.py`). The backward nodes an agent would need to reason about do not
exist in `model.py`. They have not been generated yet.

This is not "the agent will do a worse job at the same problem." It is a
different problem operating on a different object, and
[adaptive_sharding.md](adaptive_sharding.md) contains the proof. LLaMA3-8B
and LLaMA3-70B, same training configuration, diverge on `wo`: the 8B keeps
it sequence-parallel, the 70B goes column-parallel. The reasons are that
column-parallel produces `P(sum)S(0)` weight gradients — one reduce-scatter
dimension — where sequence-parallel produces `P(sum)P(sum)` and needs a full
2D reduce-scatter, and that sequence-parallel `wo` happens to emit
`S(0)S(1)`, which the residual add consumes for free.

Neither of those facts is visible from model source at any level of
reasoning skill. They are properties of a graph that AOTAutograd has not
emitted yet.

### The answer is a function of the deployment, not the model

Even holding the architecture fixed, the optimal plan moves with batch size,
sequence length, mesh shape, and interconnect. Batch size alone flips the
solver between DP-dominant and TP-dominant regimes; the 8B/70B divergence
above happens at identical config.

An agent that writes parallelism code emits an *artifact*. What is actually
needed is a *function*, re-evaluated per cluster and per configuration. This
argument holds even for a reader who thinks architectures will stop
churning.

### The cost model is measured, not reasoned to

This is the defensible asset and it is the one most often described in
passing. `cost_models/nccl_cost_model.py` ports NCCL's `tuning.cc` algorithm
and protocol selection — Ring, Tree, CollNet, NVLS, NVLS Tree crossed with
LL, LL128, Simple — and then corrects it against nccl-tests measurements on
H100 NVSwitch at 1, 2, 4, 8, 16, and 32 nodes.

[adaptive_sharding.md](adaptive_sharding.md) records that the adaptive
sequence-parallel behavior *requires* that model; the generic estimator did
not price communication finely enough to separate the candidates. Empirical
tables are not recoverable by inference. No model has them in weights, and
no amount of reasoning reconstructs them. They have to be measured on
hardware.

### Recipes go stale exactly where agents push hardest

An agent can recite the Megatron placement recipe for LLaMA. Agentic coding
produces architecture variants faster than recipes get written — which is
precisely the regime where memorized recipes fail and a graph-in,
solver-out backend earns its keep. Agents raise the rate of architectural
change, and that makes the hand-tuned plan the bottleneck rather than the
model code.

## Reposition, don't retrofit

The productive shape is hybrid: the agent proposes single-device semantics
and, where communication is data-dependent, `local_map` regions; AutoParallel
is the optimizer and the verifier underneath.

That implies an inversion of the current framing. The cost oracle is not an
agent affordance bolted onto an automation library. It is the primary
product, and the automation is one client of it. AutoParallel is the only
thing in the stack that can price a parallelization plan without running the
job — including plans the solver would never have generated on its own.

Most of the machinery already exists and is pointed the wrong way:

| Capability | Exists as | Missing |
|---|---|---|
| Plan serialization | `save_placements` / `load_placements` | not a public entry point |
| Runtime + memory estimate | `estimate_graph_metrics` → `GraphMetrics` | not exported from `__init__.py` |
| Per-node costs and alternatives | `get_json`, `get_log`, `explain_placement` | log artifacts, not return values |
| Propose / evaluate / revise loop | `add_node_constraint` → `resolve` → `diff_solutions` | undocumented as an API surface |
| Planning without CUDA | fake process group in `examples/example_hf.py` | example, not an API |

`__init__.py` exports five symbols and none of them scores a plan.

## The plan

Ordered by dependency. Item 1 gates items 2 and 7; items 4 and 5 are
independent and worth doing regardless of who is calling.

### 1. Make the cost model safe to export

**This is the prerequisite, not a cleanup task.** The cost model currently
carries approximations that are defensible inside the ILP and dangerous
across an API boundary:

- `all_to_all_cost` in `cost_models/collective_runtime_estimation.py` ends
  with `total_time *= 5` under a `FIXME: this is a hack, we need to spend
  some more effort on the cost model`.
- Blackwell bandwidth is scaled from Hopper by the `bw_intra` ratio
  (`_BLACKWELL_BW_SCALE = 640.0 / 320.0`), and `_A2A_CE_BW[BLACKWELL]` is
  annotated `estimated proportionally from bw_intra ratio; needs profiling`.
- `apply_prefetch_discount` defaults to `scale=0.0` — collectives treated as
  entirely free — and is *not* applied automatically, so the default solve
  systematically over-prices sharded parameters.

Inside the solver only relative ordering matters and a consistent bias
largely cancels. The moment a number crosses an API boundary to a caller
that cannot squint at it, a 5× fudge factor stops being an approximation and
becomes a defect with a blast radius. An agent will believe the microsecond
count.

Two things to ship:

1. Replace the `all_to_all` multiplier with a fitted model, and measure
   AllToAll on Blackwell NVSwitch (already item 1 of the TODO block in
   `nccl_cost_model.py`).
2. Attach **calibration provenance** to every cost the API returns:
   `measured`, `interpolated`, `extrapolated`, or `modeled`. A caller that
   knows a number is extrapolated can weight it; a caller handed a bare
   float cannot.

Decide separately whether `apply_prefetch_discount` should be on by default.
Shipping an oracle whose default configuration misprices FSDP is worse than
shipping no oracle.

### 2. Ship the plan oracle

The two halves exist and are not connected. `load_placements` takes a
complete plan as JSON; `estimate_graph_metrics` returns critical-path time,
compute time, communication time, exposed communication time, and peak
memory. Wire them together and export the result.

```python
from autoparallel import score_plan

report = score_plan(
    model, input_fn, mesh,
    plan="plan.json",       # or a solution dict; None scores the solver's own
    device="fake",          # planning without CUDA, per example_hf.py
)

report.step_time_us
report.exposed_comm_us
report.peak_memory_bytes
report.oom_risk             # peak vs device capacity
report.confidence           # per-collective provenance, from item 1
```

This is the single highest-leverage change in the document. It lets a caller
evaluate a plan it wrote itself, which is what makes AutoParallel useful to
an agent that has its own opinions — and it is equally useful to a human
comparing against a hand-tuned TorchTitan baseline.

### 3. Keep the stateful loop; add a stateless door

A stateless one-shot — model reference, sample shapes, mesh as plain JSON,
budget in; parallel module, plan JSON, and cost summary out — is the right
default, and `export_json.py` already builds most of the return payload.

But do not deprecate the session. `add_node_constraint` → `resolve()` →
`diff_solutions` is already a propose/evaluate/revise loop, and `resolve()`
is deliberately cheap: it re-solves without rebuilding the objective.
`remove_constraints` and `explain_placement` complete it. That is the most
agent-shaped API in the repository and it exists today; it needs
documentation and a stable name, not replacement.

Ship both. One-shot for the common case, addressable session for the case
where the caller is exploring.

### 4. Machine-readable failures with repair hints

Failures today are human-readable logs: infeasible-ILP `RuntimeError`,
`_check_forward_args` shape mismatches in `input_validation.py`, Dynamo
trace breaks. Each needs a code, the offending node with its `module_path`
(which `export_json.py` already extracts), and an actionable suggestion.

```json
{
  "error": "INFEASIBLE_CONSTRAINTS",
  "node": "mm_14",
  "module_path": "layers.3.attention.wq",
  "detail": "node constraint S(0)S(1) conflicts with parameter memory budget",
  "suggested_fix": {
    "action": "relax_constraint",
    "constraint": "memory_constraint_high"
  }
}
```

Codes worth defining: `INFEASIBLE_CONSTRAINTS`, `TRACE_FAILED`,
`UNSUPPORTED_OP`, `MEMORY_OVER_BUDGET`, `INPUT_SHAPE_MISMATCH`,
`MESH_MISMATCH`.

This is not agent-specific work. It is better engineering that a human
debugging a 512-GPU job benefits from identically.

### 5. `supports(model)` preflight

Generated model code trends dynamic and op-inventive; tracing assumes
mostly-static FX-traceable graphs. Report traceability and unsupported ops
*before* solving, with pointers to regions that should be wrapped in
`local_map`, rather than failing with a stack trace mid-pipeline.

```python
report = supports(model, input_fn, mesh)

report.traceable            # bool
report.unsupported_ops      # [{op, node, module_path}]
report.local_map_candidates # regions with data-dependent communication
report.blocking             # what must be fixed vs what degrades quality
```

Also not agent-specific, and cheap relative to its value.

### 6. Return the "why" as data

The optimizer already records per-node costs, chosen strategies, and
alternatives. `get_json` builds the structure; `explain_placement` compares a
target against the chosen placement; `print_costs_for_node` prints the
redistribution matrix; `visualizer/build_display_from_json.py` renders it.
All of it is reachable only through logs, `trace_structured` events, or
stdout.

Return per-node cost breakdowns, top-k alternatives, and the visualizer
artifact as values. A caller that can explain *why* a plan was chosen can
hand that explanation to a human, which is the difference between a tool
that is trusted and one that is second-guessed.

### 7. `verify_plan()` on a real substrate

Be precise about what is being verified. Fake tensors carry no data, so a
fake-tensor forward and backward checks shapes, placements, and
traceability — not numerics. For a call that returns pass/fail to a caller
that will act on it, that distinction is the whole point.

The real substrate already exists: `tests/test_correctness.py` compares a
parallelized model against an unsharded single-GPU reference using
`LocalTensorMode`, on one GPU. Its docstring records the two upstream
blockers keeping it off the `apply_placement()` path:

1. `ProcessGroup` objects from `compile_on_one_rank` are not deepcopy-safe,
   which breaks `extract_forward_graph`'s deepcopy of the joint graph.
2. AOT autograd's compiled backward rejects `LocalTensor` tangents because
   `LocalTensor` does not implement `__coerce_same_metadata_as_tangent__`.

Those two issues are the critical path to a credible `verify_plan()`. The
target:

```python
result = verify_plan(model, input_fn, mesh, plan)

result.placements_consistent  # fake tensor: shapes and placements typecheck
result.numerics_match         # LocalTensorMode: real data, fwd + bwd
result.max_abs_err
result.cost_vs_baseline       # from item 2
```

Report the two levels separately. "Placements are consistent" and "gradients
are correct" are different claims and a caller should be able to tell which
one it got.

## Scope boundaries to state out loud

An agent-facing tool that overstates its coverage is worse than one that
declines. Two boundaries belong in `supports()` output and in the docs.

**No pipeline parallelism.** Nothing in the repository does PP; the
`_pipelined_*` symbols in `graph_passes/async_tp/` are micro-pipelined TP and
unrelated. PP is a graph-partitioning and scheduling problem across stages,
and at frontier scale it is not optional. AutoParallel decides intra-stage
parallelism, and the pitch should say so.

**`local_map` is a concession, not a clean division of labor.** The hybrid
shape above says the agent may write `local_map` regions for MoE. That means
the agent chooses the expert-parallel strategy for the most
communication-sensitive part of a modern model, unverified — exactly the
case the rest of this document argues callers should not be trusted with.

Two honest responses, and they are not exclusive:

- Extend `score_plan()` to cover `local_map` regions, so that even a
  hand-written region gets priced rather than trusted.
- Shrink the escape hatch. `cute_sharding_design.md` proposes a
  representation general enough to express layouts DTensor cannot name —
  ordered `S(0)S(0)`, non-contiguous `cat`, XOR-swizzled zigzag. Fewer
  inexpressible layouts means fewer regions where the optimizer has to step
  aside.

## Sequencing

```
1. cost model provenance + all_to_all fit     ── gates 2 and 7
2. score_plan()                               ── the thesis
3. stateless entry point (session preserved)
4. error codes                    ┐ independent of 1-3,
5. supports() preflight           ┘ ship whenever
6. costs and alternatives as return values
7. verify_plan()                              ── needs 1, and two upstream fixes
```

Items 4 and 5 are the cheapest and benefit humans equally; there is no
reason to hold them behind the strategic work. Item 2 is the one that
changes what AutoParallel is for, and item 1 is the reason it can be
believed.
