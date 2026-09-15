# Permanent AutoParallel experiment harness

This harness has one supported source and runtime stack. Users choose a model
and setting; source revisions, Python/PyTorch, the distributed backend, and
assets are resolved from checked-in locks.

## Run

```bash
python -m harness.cli run --model llama3_8b --setting 2d-32gpu
```

The command performs source checkout, runtime and asset verification,
validation, packaging, MAST dry-run, submission, CRITICAL/99 priority
verification, monitoring, retrieval, and canonical analysis. It is resumable:
rerunning the same model and setting reuses the task state and never submits a
second job when `job_id.txt` exists. Every task uses the immutable
`attempts/001` layout; legacy `attempt/` layouts and additional attempt numbers
fail closed rather than being migrated or retried automatically.

Valid settings are declared in `run_settings.toml`:

- `llama3_8b`: `2d-{8gpu,16gpu,32gpu,64gpu,128gpu}`,
  `3d-long-{2x2x4,4x2x4,8x2x4}`,
  `replanning-{seq2k-lb2,seq4k-lb2,seq8k-lb2,seq16k-lb2,seq32k-lb2,seq2k-lb4,seq2k-lb8}`,
  and `planning-scalability`;
- `muse_glimmer_30b`: `2d-{8gpu,16gpu,32gpu,64gpu}`;
- `deepseek_v3_16b`: `3d-{2x2x4,2x2x8,4x2x8}`.

`HARNESS_WORKSPACE_ROOT` may select the parent directory for task records. It
does not affect experiment inputs or source versions.

`HARNESS_CACHE_ROOT` may select an absolute shared source/runtime cache. The
cache is namespaced by the experiment-lock digest and materialized under a
process-safe file lock; task attempts, evidence, mounts, and results remain
independent.

## Locked stack

`experiment_lock.toml` is the only authority for:

- exact TorchTitan and AutoParallel commits and remotes;
- `torchtitan_conda_prod:902` and exact runtime versions;
- the legacy/default DTensor backend.

Campaigns cannot declare source revisions, conda packages, or
`parallelism.spmd_backend`. Validation rejects such overrides. The source trees
must be clean and at the exact locked commits.

`asset_lock.toml` pins the internal OilFS workspace, relative paths, file
counts, and content hashes for every model, tokenizer, replay, placement, and
dataset asset. There is no fallback to an unpinned local path or network
download.

`HARNESS_CORE.sha256` covers the lock files, active campaigns, harness,
launcher, active workloads, and measurement scripts. A code or campaign change
requires a version bump and manifest regeneration.

## Execution profiles

- `tt_main_default_v1`: native TorchTitan model/parallelizer with ordinary
  Inductor compilation.
- `gt_manual_eager_v1`: GraphTrainer with eager memory policy and full Inductor
  compilation.
- `apgt_validated_v1`: AutoParallel + GraphTrainer validated defaults.

All active settings set `TORCHINDUCTOR_CUDAGRAPHS=0`. Both GraphTrainer
profiles use full Inductor compilation and disable `cudagraph_pass`.

MainTrainer and manual GraphTrainer never import or invoke the AutoParallel
parallelizer. The AutoParallel profile is the only profile that enables
`compile.enable_autoparallel`.

All paired arms retain identical model, inputs, batch and sequence shape,
precision, activation checkpointing intent, optimizer, scheduler, physical
allocation, rank-to-GPU mapping, and measurement method. A campaign must
declare the complete trainer/parallelization/compiler stack as its variable
when profile defaults differ.

The `planning-scalability` setting is a planner-only fake-H100 cost-model
study, not GPU training-performance evidence. One eight-rank MAST allocation
is audited, rank zero runs the existing AutoParallel search-profile CLI for
the declared 1D through 4D meshes, and all ranks participate in completion
gates. Each mesh records three lazy, seeded Approx runs and three independent
LP-reference runs.

## Lower-level commands

The existing `validate`, `package`, `render-mast`, `submit`, and `analyze`
subcommands remain for diagnosis. They require explicit source and asset paths
where applicable. `run` is the supported reproduction entry point.

## Historical provenance

Original campaign files and their model-specific pins are stored unchanged in
`provenance/campaigns/`. They are evidence, not runnable active campaigns.
`UPSTREAM.md` maps each historical pin to the unified stack.

The unified stack is a new-stack retake. Historical performance numbers remain
attributed to their original source and runtime pins.

## Validation status

The initial release is labeled `source-validated, GPU-unverified`: source,
configuration, package, and local unit checks are required before publishing,
but no GPU or MAST job is launched as part of the branch construction.
