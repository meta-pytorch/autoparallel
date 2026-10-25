# AutoParallel Optimization Workflow

For a generic request to parallelize a model or add DDP/FSDP, first follow
[parallelization-routing.md](parallelization-routing.md). This workflow covers
the AutoParallel attempt; it does not by itself decide whether the result meets
the user's bar or erase the evidence when a fallback is selected. Before
selecting AutoParallel over DDP/FSDP, also follow
[baseline-counterfactuals.md](baseline-counterfactuals.md).

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
silently untie shared weights such as tied embeddings and decoders, changing
the captured graph and parameter count. Confirm that the constructor is
meta-safe; pretrained loading, data-dependent initialization, and `.item()`
during setup may require a model-specific config-only or empty-init path.
Create example inputs outside the meta context so `input_fn` receives real
tensors. After `apply_placement`, materialize the parallel model with
`to_empty` and initialize it or load a checkpoint.

When the intended job requires checkpoint/resume, portability, or elastic
restart, treat that as an acceptance gate rather than an integration follow-up.
Record the state-checkpoint API and format, included model and optimizer state,
mesh, and restore target. Exercise a save/load round trip with the realized
sharded module and optimizer, then run at least one post-load step. A DCP round
trip may restore into a differently sharded or non-AutoParallel model, but that
evidence does not cover a different format or omitted optimizer state.

Use a real process group for execution evidence. A fake process group is useful
for capture and planning experiments, but does not show that real collectives
execute correctly.

Before capture, audit the forward with `global-spmd-modeling.md`. In particular,
code written around `batch_size_per_gpu`, local random permutations, or explicit
process-group reductions may change meaning when traced over a global logical
batch. Warn the user and propose an adaptation before proceeding when unclear.

Establish the representative workload with
[workload-representativeness.md](workload-representativeness.md). Record the
global batch for one forward/backward microstep, effective DP degree, local
batch, gradient accumulation, effective optimizer batch, material input shapes,
and their source. Do not infer global batch from world size. A minimal synthetic
input is suitable for smoke coverage but cannot establish strategy quality.
When a global microbatch is supplied, use that exact value in `input_fn` for
capture and derive the executed local input batch from the effective DP degree.
Check divisibility and verify the captured and executed input shapes before
interpreting the plan.

Use a supplied `DeviceMesh`. Otherwise follow `mesh-selection.md`: confirm the
intended GPU and node allocation rather than inferring it from visible devices,
then record the inferred shape, dimension roles, and physical-link rationale.

Use the real training boundary when plan quality matters. If the supplied model
returns logits but the actual loss is outside the captured graph, state that the
plan excludes the loss and is provisional. Include the representative loss in
the captured module only when that matches the intended integration and the user
authorized the adaptation.

Do not infer that activation checkpointing was realized from a compiler flag.
Verify changed save/recompute decisions and a corresponding graph or memory
effect. In the current two-partition AutoParallel path, a second-compilation
`enable_ac=True` pass can be neutralized by replaying the first partitioner's
save decisions. Until the first partition is checkpoint-aware, prefer the
user's model-level checkpoint policy applied before capture, and leave relative
competitiveness unavailable unless every baseline uses the same realized
policy.

When an authorized controlled benchmark begins with underfilled per-rank work,
keep the initial point and run only the bounded batch-sensitivity comparison in
`workload-representativeness.md`. Each batch point is a different execution
configuration and needs an aligned baseline. Never substitute a larger
synthetic batch for the user's actual deployment workload.

## Capture and optimize

The basic planning operation is:

```python
with AutoParallel(model, input_fn, mesh, mp_policy=mp_policy) as autop:
    autop.add_input_constraints(input_placements)
    autop.add_output_constraints(output_placements)

    if memory_bounds is not None:
        autop.add_parameter_memory_constraint(*memory_bounds)
    if prefetch_discount is not None:
        autop.sharding_optimizer.apply_prefetch_discount(
            scale=prefetch_discount
        )

    placement = autop.optimize_placement(verbose=True)
    optimizer_json = autop.sharding_optimizer.get_json()
    autop.sharding_optimizer.save_placements(placements_path)
    parallel_model = autop.apply_placement(placement)
```

Output constraints align with the flattened output pytree. Use `None` for
non-tensor leaves, for example `[(Shard(0),), None]` for `(logits, None)`.
Use `UNCONSTRAINED`, imported from `autoparallel`, for a tensor leaf whose
placement the solver should choose. `None` retains the default batch `Shard(0)`
behavior for tensor leaves, so it is not an unconstrained request. Omit the
whole constraint call only when every boundary tensor is unconstrained.

Omitting the parameter-memory call leaves replication legal. Calling it with no
arguments is different: it uses `low=0` and `high=1 / world_size`, forcing
sufficiently shardable parameters toward their minimum local fraction. Bounds
are normalized fractions of byte-weighted persistent parameter storage. Record
the achieved values from `get_json()["summary"]["parameter_storage"]`.

Apply a prefetch discount only as an explicit sensitivity assumption and always
record its scale. Repeated calls replace the objective scale without mutating
the base communication costs, so endpoint solves can reuse one optimizer. Save
each placement and JSON result before the next solve. For a full-shard plan,
follow `full-shard-planning.md` rather than tuning the scale until a desired
placement appears.

Capture the placement, `get_json()["summary"]`, exact constraints, discount,
mesh, and communication cost model before leaving the context. Save the full
optimizer with `save()` when repeated counterfactual analysis is likely.

Repeated-subgraph clustering is optional compression of the ILP, not a reason
to discard an otherwise supported graph. AutoParallel skips malformed groups
atomically and records them in the JSON clustering summary. Preserve and report
those diagnostics because the solve used less compression. Retry with
`repeated_subgraphs=False` only for comparison or an older checkout that still
aborts.

`optimize_placement()` also emits `autoparallel_sharding_optimizer_log` and
`autoparallel_solution` structured trace artifacts. Set `TORCH_TRACE` to a trace
directory before the run to retain them; use `tlparse` when a parsed trace is
useful. This is optional and does not replace explicit placement or optimizer
artifacts.

## Keep one cost model through evaluation

`AutoParallel.__enter__` installs its communication topology, but `__exit__`
restores the previous global value. Compilation normally occurs afterward.
Without an outer lifetime, the ILP can therefore use the NCCL model while the
reordering pass and `estimate_graph_metrics()` silently use PyTorch's default
model.

Use the skill session to keep the exact topology active across planning,
compilation, metrics collection, and durable reporting:

```python
import subprocess
import sys
from pathlib import Path

repo_root = Path(
    subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()
)
skill_scripts = repo_root / ".agents/skills/autoparallel-optimizer/scripts"
sys.path.insert(0, str(skill_scripts))
from report_session import AutoParallelReportSession

with AutoParallelReportSession(
    mesh,
    request=request,
    campaign=campaign,
    compiler_config={"enable_ac": False, "overlap_scheduling": False},
    source_root=repo_root,
    external_source_roots=external_source_roots,
    entry_point="training.entrypoint:main",
    launch_config={
        "launcher": "torchrun",
        "nodes": 1,
        "processes_per_node": 8,
        "world_size": 8,
    },
    mode="measure",
) as session:
    session.mark_substage("capture")
    with AutoParallel(
        model,
        input_fn,
        mesh,
        mp_policy=mp_policy,
        cost_model=session.cost_model,
    ) as autop:
        # Add the same intended constraints as the planning run.
        session.mark_substage("solve")
        placement = autop.optimize_placement(verbose=True)
        session.record_plan(autop.sharding_optimizer)
        session.mark_substage("lower")
        parallel_model = autop.apply_placement(placement)
        session.mark_lowered()

    session.mark_substage("compile")
    compiled = torch.compile(
        parallel_model,
        backend=autoparallel_backend(
            enable_ac=False,
            overlap_scheduling=False,
        ),
    )
    def step():
        parallel_model.zero_grad(set_to_none=True)
        output = compiled(*local_inputs)
        model_specific_loss(output).backward()

    session.mark_substage("measure")
    session.measure_step("target-step", step)
```

Resolve the checkout explicitly if the evaluation process starts outside its
Git worktree. Adapt the loss, inputs, activation-checkpointing setting, and
reordering options to the workload. Omitted reordering options inherit
`aten_autobucketing_config`; override them only intentionally.

The session's ATen pass is the only overlap scheduler in this workflow. Keep
`overlap_scheduling=False`: Inductor otherwise runs its scheduler after the
custom post-grad pass, so the recorded metrics would describe an intermediate
graph and the graph would be scheduled twice. The collector raises if it sees
Inductor overlap scheduling enabled.

The session detects the NCCL topology by default, passes that same object to
AutoParallel through `session.cost_model`, records whether the topology was
detected or supplied explicitly, enforces the resolved fallback policy, and
restores the prior global state. Its collector runs on every rank, but only rank
0 writes the shared report. Pass an explicit `NCCLTopoConfig` as
`nccl_topology=` when the hardware or allocation is non-standard, including a
sub-allocation whose mesh size is not divisible by the detector's canonical
GPUs per node. Describe the allocated communicator, not the host's unused GPUs.
Do not treat a multi-node fallback result as a final deployment ranking; obtain
or construct the topology profile, or report the evaluation as provisional.

Pass a stable Python entry point and a sanitized launch description rather than
copying `sys.argv`; command lines can contain credentials or sensitive input
paths. The session records the repository name, Git revision, dirty-state flag,
and hashes of the tracked diff and porcelain status. Untracked contents are not
read; only their paths contribute to the state fingerprint. Prefer clean source
revisions, or retain an authorized patch separately when exact dirty-source
replay is required.

For a suite, pass stable campaign and case IDs plus the expected case count, and
register external model repositories through `external_source_roots`. Mark the
active substage before capture, solve, lower, compile, execution, or measurement
so a failed bundle identifies the responsible operation. Set
`include_optimizer_state=True` only for an authorized trusted bundle that will
be re-solved offline; the artifact is version-coupled pickle data.
The standard capture level retains a compact optimizer replay by default. It
stores the compressed FX replay inputs and regenerates strategies and the ILP
from the recorded AutoParallel revision. Pass
`include_optimizer_replay=False` only when graph-retention policy forbids the
artifact; the resulting run cannot participate in offline optimizer replay.

The session refuses an already configured `post_grad_custom_post_pass`; compose a
model-specific pass explicitly instead of silently changing pass order. It
identifies backward graphs by `tangents_*` placeholders and treats an
all-`is_backward` graph as backward when no tangent placeholder is present.
Mixed partitioner tags can survive recomputation in a partitioned forward
graph, so they do not make the phase ambiguous by themselves. It records every
specialization. The lower-level collector can emit optional forward/backward
Perfetto traces with `trace_dir=` for a specialized workflow; keep those traces
outside the bundle or register them as artifacts. Metrics contain a compact
placeholder count by default; pass
`include_placeholder_signatures=True` only when shapes and dtypes are needed to
diagnose specialization, because full signatures are large for parameter-heavy
graphs. Forward and backward must both execute before leaving `with session:`
because backward compilation is lazy and the post-grad hook is restored when
the context exits.

### Use the lower-level collector in an isolated harness

When the request is specifically to repair a local evaluation harness and no
durable report is required, use the existing collector directly rather than
reimplementing its reordering and metric logic:

```python
from collect_reordered_metrics import ReorderedMetricsCollector

collector = ReorderedMetricsCollector(mesh)
with collector:
    with AutoParallel(
        model,
        input_fn,
        mesh,
        mp_policy=mp_policy,
        cost_model=collector.cost_model,
    ) as autop:
        placement = autop.optimize_placement(verbose=True)
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
collector.write_json(metrics_path)
```

After editing an isolated evaluation harness, run the static validator without
executing the workload:

```bash
python .agents/skills/autoparallel-optimizer/scripts/validate_evaluation_harness.py \
  path/to/evaluation_harness.py
```

It rejects a missing collector lifetime, an unshared cost model, duplicate
scheduler configuration, absent backward coverage, or non-durable metrics.
Fix every issue and include each returned `handoff_requirements` statement in
the final response.

Resolve the skill scripts directory explicitly when the harness cannot import
it normally. Keep forward and backward inside the collector context so lazy
backward compilation reaches its post-grad pass. Call these results estimated
post-reordering metrics, separate them from the ILP objective, and keep phase
memory peaks separate. The pass does not insert overlap control dependencies;
later Inductor scheduling or fusion can still move execution relative to the
recorded graph order, so these metrics are not measured runtime.

## Explore the stable joint graph

Node names are stable while solving counterfactual placements:

When a saved optimizer is currently infeasible because of a newly added
exploratory constraint, first remove exactly the names returned when it was
added and re-solve with the original output and memory constraints unchanged.
Establish that recovery before re-adding the exploratory constraint for bounded
constraint-subtraction ablations. This preserves the causal baseline.

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

Use `explain_placement()` for a placement family and
`print_costs_for_node()` for a redistribution matrix. Preserve their text when
it supports the answer. Explore before lowering unless the purpose is to
compare multiple candidates after scheduling.

For CPU-side exploration, capture on GPU and call
`autop.sharding_optimizer.save("model.ap")`. For ATen-only graphs,
`ShardingOptimizer.load("model.ap")` can later inspect and re-solve without
model code, a live mesh, a process group, or a GPU. Custom ops and some
`local_map` regions may require their registration modules to be importable
before loading. Treat the artifact as trusted pickle data and keep code and
PyTorch versions aligned.

## Validate and report

For a generic routing decision, capture once and derive only the applicable
DDP/FSDP counterfactuals. Preserve the user's existing configuration when one
exists; otherwise use a versioned standard template. Record serialized,
implementation-aware reordered, and optimistic legal-overlap latency estimates
plus calibration error. Keep memory scopes explicit and do not infer full
training-memory feasibility from the parameter constraint alone.

Use the policy classifier for the default relative verdict, then call
`record_competitiveness_assessment()` before accepting AutoParallel. A
schedule-sensitive or unavailable comparison is not an AutoParallel win. Do
not execute a real baseline merely to resolve uncertainty unless the user
explicitly authorized the benchmark cost.

For execution, materialize or load initialized parameters after
`to_empty(device="cuda")`, run compiled forward and backward with local input
shapes, and check outputs and gradients for finite values. For numerical
comparison, use the same parameters, inputs, seed, dtype, and tolerances as the
unsharded reference. Call `torch.manual_seed(seed)` on every participating rank
before DTensor initialization or random DTensor operations; relying on DTensor
to synchronize rank-zero RNG state is deprecated. LocalTensorMode tests are
useful diagnosis but are not the public compiled execution path.

For an AutoParallel-only hardware measurement, set the session to
`mode="measure"` and pass a complete iteration callable to `measure_step()`.
The helper warms up the compiled path, times repetitions with CUDA events and a
full-device completion barrier, captures peak allocated and reserved memory,
and takes the maximum observation across ranks. It rejects a new graph
specialization during the measured window, then compares mean step time and
peak allocated memory with the latest reordered forward/backward estimates.
The callable owns gradient clearing and must use one stable input specialization.
Exclude the optimizer update when directly calibrating against the reordered
graph estimate because that estimate contains only forward/backward. A separate
production-step measurement may include the update; pass
`optimizer_update=True` to `measure_step()` to record the wider boundary, and
do not interpret its ratio to the narrower graph estimate as cost-model error.
Keep the boundary identical across strategies.

For training acceptance, pair that measurement with an optimizer-memory
assessment. Record the optimizer, parameter/gradient/moment dtypes, state
sharding, observed peak, additional persistent bytes absent from the measured
boundary, projected peak, device limit, and required headroom. For a
forward/loss/backward boundary that already holds parameters and gradients,
AdamW's incremental persistent state is normally the two local moment tensors;
compute their bytes from local element counts and actual moment dtypes rather
than assuming they equal parameter bytes. Do not label the projection measured,
and do not accept a plan that only fits before the intended optimizer state is
materialized.

For production training performance, include the actual optimizer update in
the measured callable or add a versioned, held-out-calibrated optimizer-cost
envelope to every strategy. Hold optimizer implementation, foreach/fused mode,
state dtype, gradient accumulation, and update frequency fixed. If only the
forward/loss/backward boundary is comparable, report that useful partial result
but leave end-to-end competitiveness inconclusive.
Preserve transformations between backward and the update. For global-norm
gradient clipping, keep the norm definition, threshold, order, and frequency
fixed and include both local norm work and the placement-dependent global
reduction in every candidate's boundary. An unclipped measurement is not a
proxy for a clipped production step. Treat the clipped workload as a new
decision relative to an unclipped report, but keep exact clipped retries and
linked clipped baselines under the same clipped `decision_id`. Record clipping
frequency and optimizer-update frequency explicitly even when both are once per
accumulation cycle. State that those two frequencies are the same for every
candidate rather than leaving the equality implicit.
In the response, report `local_norm_work` and
`placement_dependent_global_reduction` separately; a reference to the global
norm does not make its local work explicit.
Record the trainable-parameter mask and optimizer parameter groups before
capture. After AutoParallel lowering or native DDP/FSDP wrapping, assert that
every parameter FQN retains the intended `requires_grad` value and that the
optimizer contains only the intended parameters. Also preserve exact group
membership and per-group hyperparameters such as weight decay and learning
rate; a single-group timing run is not comparable to a grouped production
optimizer. Record tied-parameter alias groups before capture and assert that
lowering or wrapping preserves them; silently untying an embedding and output
weight changes updates and can duplicate optimizer state. Treat a mismatch as
a semantic failure, not a performance result.
State the trainability gate in that order: check the per-FQN invariant after
lowering or wrapping and before accepting execution evidence.
Record mutable buffer names, shapes, dtypes, placement, and update semantics.
For BatchNorm over a sharded logical batch, verify whether AutoParallel reduces
global statistics and keeps replicated running buffers synchronized. Ordinary
DDP BatchNorm uses rank-local statistics and is not a matched baseline; use
SyncBatchNorm or an equivalent globally reduced implementation when global
statistics are the intended contract.
For memory, count persistent local parameter bytes for frozen and trainable
parameters, but gradients and optimizer state only for local trainable
parameters. For FP32 AdamW this is
`all_local_parameter_bytes + 3 * local_trainable_parameter_bytes` when the
gradient and two FP32 moments are all resident.
For gradient accumulation greater than one, the comparable boundary is a full
accumulation cycle. Measure or model the intermediate no-sync microsteps, the
final synchronized microstep, and one optimizer update separately at their
actual frequencies. Report total cycle latency and latency amortized per
microstep. A synchronized microstep cannot be multiplied by the accumulation
count, and optimizer work cannot be charged once per microstep.
Record whether every strategy actually suppresses intermediate reductions.
When compiled DDP alternates `no_sync()` and synchronized microsteps, use
`static_graph=False` unless that exact static-graph template has executed
successfully; an internal reducer-hook assertion is a template incompatibility,
not evidence that DDP itself is infeasible.
Build the optimizer envelope from paired partial-step and full-step evidence.
Use local parameter or optimizer-state bytes to represent placement-dependent
work and include a fixed dispatch term where supported. Stratify calibration by
realized kernel family, dtype policy, framework version, and hardware; do not
reuse a coefficient merely because the optimizer class has the same name.
Freeze the fitted parameters and uncertainty before measuring prospective
architectures. Record prediction errors and winner accuracy afterward without
retroactively changing the frozen prediction.

Before using the frozen envelope for a new decision, write an explicit
applicability block even when the prompt supplies the matches:

```text
calibration platform: hardware, framework, dtype policy, realized kernel family
optimizer contract: groups, update frequency, accumulation, state placement,
  dispatch
calibration scale: local parameter-byte and tensor-count ranges
```

Mark every group matched, mismatched, or unknown. Widen or reject the envelope
for a mismatch or material unknown. A modeled prediction does not mean the
current candidate executed on target; use the strongest evidence level actually
reached for that candidate.
Prefer an in-step all-rank optimizer profile from the same materialized process
over subtracting partial- and full-step medians from separate launches. Record
local parameter bytes, optimizer parameter tensor count and tensor types, since
dispatch and fragmentation can remain material at equal byte size.
When AutoParallel loses a measured comparison, also compare max-rank collective
operation counts, bucket counts, and matmul counts with the native baseline.
Aggregate communication bytes or time alone can hide launch fragmentation. If
optimizer self time is competitive but AutoParallel emits many more reductions
or matmul fragments, record that separately rather than blaming the optimizer
update. Inspect the pre/post bucket counts and dependency-blocked candidates
before proposing coalescing. Terminal gradient reductions may be compatible;
tensor-parallel activation reductions whose outputs feed later compute are not.
When dependency-bound collectives dominate, target placement selection,
local-kernel cost, or post-lowering candidate reranking instead.
For multi-dimensional meshes, also group counts by process-group identity.
All-gather-based discovery can omit process groups that contain only
all-reduces in a phase. Use one all-group discovery counterfactual when this
omission is material; require all-rank execution and independent launches
before proposing it as a default because cross-group ordering is part of the
runtime contract. Compare launch-level medians, not only the reduced collective
count. Treat a neutral or slower held-out result as evidence that broader
discovery needs per-group profitability selection rather than a global default.
Also compare collective kernel time by rank and mesh coordinate. A stable
subgroup-correlated max-rank stall indicates cross-communicator launch-order or
peer-wait skew that a single-rank graph estimate cannot represent; preserve it
as a separate finding rather than absorbing it into a global launch constant.
When the user leaves `foreach` or `fused` at its framework default, record both
the configured value and the kernel family observed for every strategy;
automatic selection can depend on parameter tensor type and placement.
Before accepting AutoParallel from a target comparison, use at least three
independent process launches for every strategy and aggregate launch-level
medians or envelopes. Also repeat close losses before treating the ranking as
stable. Warmup repetitions inside one launch establish steady state but do not
substitute for independent launches.
Also preserve max-rank first-warmup and total-warmup wall times. These include
lazy compilation and establish time-to-ready, not compiler-only latency. If the
user provides an intended step count or job duration, compare startup plus
steady-state execution for that horizon, report the break-even point, and use
the native fallback when the run ends before AutoParallel amortizes its startup.
That finite-job result is a projection unless the full job ran end to end. In a
decision card, use `executed_on_target` only when its components explicitly came
from target execution and reserve `measured` for a directly measured decision
boundary. Do not infer either provenance level from timing numbers alone. When
every strategy satisfies the acceptance bar but startup is not amortized,
record `STARTUP_NOT_AMORTIZED`; do not claim `PERFORMANCE_BAR_MISSED`.
When run length is unknown, report that the steady-state decision is conditional.
Do not reject a long-running throughput win solely because startup is slower,
or hide a short-job time-to-solution loss behind steady-state throughput.

Apply the attempt budget to sharding search as well as capture and lowering.
Pass `solver_time_limit_seconds` for CBC-backed searches that may grow large,
but do not treat it as a complete wall-clock timeout: MPS serialization,
import, and presolve can occur outside CBC's timed branch-and-bound phase. A
non-optimal incumbent after the limit is diagnostic evidence, not an
acceptable placement. Record `SHARDING_SOLVER_TIMEOUT`, preserve the report,
and route to the normal fallback when the outer budget expires.

If a partially sharded plan beats full-shard before the update but loses over
the full step, compare its local parameter-storage fraction with the full-shard
template. Record `OPTIMIZER_COST_MISSING_FROM_OBJECTIVE` when the gap tracks
the additional local parameters and optimizer work; preserve both boundaries
for future objective calibration.
Use at most one targeted parameter-memory upper-bound counterfactual near the
full-shard fraction to test that hypothesis before proposing objective work.
Record the unconstrained plan, constrained plan, and native full-shard result.
Treat the constraint as diagnostic evidence, not a production default.
When DDP fits and a sharded AutoParallel plan loses, one fully replicated
AutoParallel counterfactual can isolate search/reordering quality from the
native-wrapper gap. Compare collective and matmul counts plus the search
objective. If the replicated counterfactual buckets far more aggressively and
beats the selected plan while its objective is worse, record a ranking
inversion caused by placement-dependent bucketing; do not compensate with a
global cost multiplier. When post-lowering reordered metrics recover the
measured ordering, retain or regenerate a bounded alternative-placement
shortlist and use the reordered ranking before spending target runs on a single
serial-objective winner.

Typical durable artifacts are `placements.json`, `optimizer.json`,
`metrics.json`, and `optimizer.log`. The session automatically records plan
signals for high communication, replicated parameters, selected replication
despite sharded alternatives, and skipped clustering groups. Strategy-space
coverage is derived from the live optimizer before it is discarded. Generate
full optimizer state and execution traces only when repeated exploration or
timeline debugging warrants their size.

For a full-shard or topology-inverted plan, include the confidence card from
`full-shard-planning.md` and preserve the serial and optimistic endpoint
artifacts independently.
