# Workload adapter provenance

The active adapters target the single source/runtime stack in
`experiment_lock.toml`. Historical campaign definitions are preserved under
`provenance/campaigns/` and are not executable inputs.

- `llama3_2d/` retains the old sample-shaped replay contract. MainTrainer uses
  native LLaMA parallelization, manual GraphTrainer uses native GraphTrainer
  with eager memory policy, and only the AP arm enables AutoParallel.
- `llama3_3d/` retains the accepted DP2 x CP2 x TP2 replay and measurement
  contract. Its manual arm no longer calls an AutoParallel CP helper.
- `llama3_seqlen/` retains the fixed C4 schedule and canonical-2K placement
  replay contract.
- `muse_glimmer/` is a thin wrapper around the pinned TorchTitan Muse configs;
  manual and AP select independent compile defaults.
- `deepseek_v3/` retains the sample-shaped SDPA workload and uses the pinned
  AutoParallel DeepSeek bridge only in the AP arm.

No active workload may select a source revision, runtime package, or SPMD
backend. Those values come only from the checked-in locks.
