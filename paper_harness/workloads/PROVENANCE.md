# Workload adapter provenance

The active adapters target the single source/runtime stack in
`experiment_lock.toml`. Historical campaign definitions are preserved under
`provenance/campaigns/` and are not executable inputs.

- `llama3_2d/` retains the old sample-shaped replay contract. MainTrainer uses
  native LLaMA parallelization, manual GraphTrainer uses native GraphTrainer
  with eager memory policy, and only the AP arm enables AutoParallel.
- `llama3_3d/` uses separately locked DP2/4/8 replays generated directly at
  sequence length 16K by the real C4 example dataloader and tokenizer.
  MainTrainer and both GraphTrainer arms consume identical per-rank batches.
- `llama3_seqlen/` retains the fixed C4 schedule. Every replanning job creates
  an Approx 4x8/2K/LB2 canonical placement before its fresh/replay comparison.
- `muse_glimmer/` is a thin wrapper around the pinned TorchTitan Muse configs;
  manual and AP select independent compile defaults.
- `deepseek_v3/` retains the sample-shaped SDPA workload and maps the requested
  folded-EP/TP meshes through the pinned AutoParallel bridge only in the AP arm.

No active workload may select a source revision, runtime package, or SPMD
backend. Those values come only from the checked-in locks.
