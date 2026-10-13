# Workload representativeness and batch sensitivity

AutoParallel quality depends on the work represented by each captured step. A
smoke input can prove capture, lowering, and execution, but it cannot rank
AutoParallel against DDP or FSDP unless it represents the intended training
workload.

## Establish the workload contract

Before planning, record:

- the global batch for one forward/backward microstep;
- the effective data-parallel degree and local batch per rank;
- gradient-accumulation steps and the resulting effective optimizer batch;
- sequence length or other material input dimensions; and
- whether these values came from the user, the training entry point, a
  maintained benchmark, or a synthetic smoke configuration.

When the user or training configuration specifies the global batch for one
forward/backward microstep, treat it as authoritative. Construct the global
logical capture inputs with that exact batch and execute the lowered model with
`local_batch = global_batch / effective_dp_degree`. Verify divisibility before
capture; do not round, truncate, substitute world size, or choose a more
convenient batch. Write the assertion explicitly, for example
`global_batch % effective_dp_degree == 0`; an integral quotient does not leave
the contract check implicit.

If “global batch” denotes the effective optimizer batch rather than one
microstep, do not reinterpret it silently. Resolve or record the microbatch and
gradient-accumulation contract, then evaluate the actual per-microstep global
batch. Keep both values in the report.

The visible GPU count does not define any batch size. In particular, do not set
the global batch equal to the world size merely to obtain local batch one. The
effective DP degree may also be smaller than the world size when another mesh
axis is used.

Preserve the user's real microbatch even when it is small. Do not silently
increase it to make AutoParallel look better. If the workload is missing, use a
clearly labeled hypothesis for planning or ask for the training configuration
when a quality decision depends on it. A synthetic workload can support smoke
evidence, but not a production strategy selection.

## Diagnose an underfilled plan

Treat unexpected replication as a selection question before calling it a
propagation failure. The selected placement artifact records only the winning
strategy. Inspect the live optimizer or trusted full optimizer state and
separate:

- nodes with at least one sharded-output option;
- replicated selections that had a sharded-output option;
- nodes whose complete option set is replicate-only; and
- operators that emitted an implicit replication-fallback warning.

Small scalars, sequence indices, and values derived from them may be
intentionally replicated. A plan dominated by replicated selections with
sharded alternatives points to the objective, constraints, or workload rather
than missing propagation rules.

An underfilled or cost-model-sensitive regime is plausible when several signals
agree:

- material operations are pinned to the estimator's kernel-launch floor across
  both replicated and batch-sharded strategies;
- a batch-sharded input is redistributed to replicated near the model boundary;
- most material activations are selected replicated despite sharded options;
- transient memory reflects the global rather than local batch; or
- measured DDP fits and outperforms the selected plan.

The estimator reaching a launch floor does not prove the real compiled kernels
are launch-bound. Fusion and shape-dependent kernel efficiency can make the
per-operation model strategy-dependent. Prefer target measurements for close
or surprising rankings.

## Run a bounded sensitivity experiment

Batch sensitivity is appropriate for a controlled benchmark, cost-model
calibration, or user-authorized workload tuning. It is not an automatic retry
for a user's fixed training contract.

Keep model configuration, sequence length, mesh, dtype, compiler settings, and
step boundary fixed. Start with the intended workload, then try at most one or
two larger local microbatches chosen from the entry point or available memory;
do not hard-code a universal threshold. Give every point a distinct execution
configuration and compare AutoParallel with the same applicable baseline.

For each point, retain placement-family counts, measured or estimated latency,
peak memory, and the exact global and local batch. Stop when the representative
case is covered, the plan stops changing materially, or the next point would
exceed the experiment budget.

A larger-batch win is evidence that AutoParallel and its cost model are
workload-sensitive. It does not replace the decision for the original
microbatch. If AutoParallel misses the bar on the intended workload, retain the
normal DDP or FSDP fallback there and keep the sensitivity result as a separate
regression and calibration case.

## Durable findings

Use stable findings when supported by evidence:

- `SYNTHETIC_UNDERFILLED_WORKLOAD` when a smoke configuration is too weak for a
  quality decision;
- `SELECTED_REPLICATION_WITH_SHARDED_OPTIONS` when replication is primarily an
  optimizer choice rather than strategy coverage;
- `LAUNCH_FLOOR_DOMINATED_OBJECTIVE` when material alternatives share the
  estimator floor; and
- `BATCH_SENSITIVITY_OBSERVED` when a controlled batch change materially alters
  placement or relative performance.

Use `BATCH_CONTRACT_MISMATCH` as an error when the reported global microbatch,
captured global input shape, effective DP degree, and executed local input shape
do not agree.

Include the relevant counts, batch contract, representative operations, and
linked run IDs. These findings identify skill or cost-model work; they do not
change cluster defaults automatically.
