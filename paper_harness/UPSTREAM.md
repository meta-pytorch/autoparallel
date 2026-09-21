# Unified legacy-DTensor stack provenance

## TorchTitan

The locked source is `kaijian/deepseek-baseline-parity-20260918` in
`AlbedoWang/torchtitan` at `94e596cc`. Its history has `383cae9f` as an actual
ancestor and retains the `60517d28` AutoParallel/GraphTrainer integration.

- `383cae9f`: eager SAC treats AutoParallel collectives like other
  GraphTrainer collectives.
- `8eec9e6f`: validated AutoParallel compiler defaults.
- `0dcc68c3`: Muse Glimmer GraphTrainer/AutoParallel support.
- `c59ce51a`: timeout propagation to all multi-axis process groups.
- `58b458293`: DeepSeek V3 integration replayed onto the validated-defaults
  lineage and merged without source conflicts.
- `3f0b0475`: LLaMA 3D behavior is ported from the preserved source snapshot;
  its unavailable original commit is not claimed as an ancestor.
- `20004e05` (PR #4553): ported as `6df7bc5f` to save the matched
  AutoParallel A2A-to-linear SAC boundaries without restoring the broader
  collective policy removed by `383cae9f`.
- `26c329bd`: experiment-only child of `6df7bc5f` that adds the folded-EP/TP
  DeepSeek adapter and makes DeepSeek and Muse consume the configured
  AutoParallel solver.
- `4a297d02`: fixes GraphTrainer FSDP dependency ordering.
- `8ceafd62`: shards DeepSeek AutoParallel logits over TP.
- `94e596cc`: cherry-picks the CP bucket-plan activation from `c2116c31` onto
  `8ceafd62`; TorchTitan passes `parallel_dims.cp_enabled` to AutoParallel's
  `synchronize_world_buckets` setting, leaving the non-CP default disabled.

The 3D port includes only the approved legacy-DTensor AP configuration, CP
input-ownership seam, DP-shard/CP/TP mesh, CP-aware SDPA, DTensor output, and
placement save/load behavior. The unrelated BlockMask private-API change is
intentionally excluded.

## AutoParallel

The maintained harness remains on `kaijian/paper-submission`. Runtime source
is pinned to the commit immediately before the lock/harness-only updates.

- `b8ace2a5` is represented by replay `3408588`.
- `570bf072` is represented by the equivalent cuDNN broadcast-mask fix at
  `5102d629`.
- `4b6c31bc` is represented by the CP stack and N-D ordered-sharding replay
  ending at `6f649f8`.
- The `b6865e7c` branch is merged into the current paper branch. Conflict
  resolution retains current CP/FlexAttention behavior and restores the
  DeepSeek aliases, symbolic handling, dynamic estimator, and checkpointed
  layer initialization.
- `e63da659` enables the H100 DeepSeek cuDNN SDPA path, and `0a3f3123` aligns
  the AutoParallel DeepSeek model semantics with TorchTitan.
- `fb056bd4`, `eabe93f9`, `dab6a2db`, `b66e7405`, and `8d009717` form the
  ordered-sharding sequence: multi-boundary fallback-adjoint eligibility,
  real-chain eligibility, alias and multi-input-consumer reachability, and
  producer-keyed boundary lowering.
- `6fb00205` adds opt-in cross-rank consensus for world bucket plans.
- `684b8533` restores the real-shape shard-order coverage and adds CP
  bucket-consensus coverage. It is the AutoParallel runtime pin; the following
  harness-only commit updates this lock and provenance.

## Reproduction policy

`experiment_lock.toml` is authoritative. Active campaigns contain no source
or runtime pins and cannot override the locked `default` DTensor backend.
Original campaign files are preserved under `provenance/campaigns/`.

A moving branch name is not evidence for a result. Every report records the
exact harness content manifest, runtime source commits, lock digest, source
tree digests, runtime versions, resolved campaign, asset hashes, and command.
