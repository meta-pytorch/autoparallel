# Unified legacy-DTensor stack provenance

## TorchTitan

The maintained source branch is `kaijian/unified-383-repro` in
`AlbedoWang/torchtitan`. Its history has `383cae9f` as an actual ancestor and
retains the `60517d28` AutoParallel/GraphTrainer integration.

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
- `26c329bd1`: experiment-only child of `6df7bc5f` that restores the previously
  validated folded-EP/TP DeepSeek adapter and makes DeepSeek and Muse consume
  the serialized AutoParallel solver selection.

The 3D port includes only the approved legacy-DTensor AP configuration, CP
input-ownership seam, DP-shard/CP/TP mesh, CP-aware SDPA, DTensor output, and
placement save/load behavior. The unrelated BlockMask private-API change is
intentionally excluded.

## AutoParallel

The maintained harness is based on `kaijian/paper-submission`. Runtime source
is pinned to experiment branch commit `f167147d`, a child of the previous
runtime pin that only adds sealed-package provenance input to the existing
planner profiler.

- `b8ace2a5` is represented by replay `3408588`.
- `570bf072` is represented by the equivalent cuDNN broadcast-mask fix at
  `5102d629`.
- `4b6c31bc` is represented by the CP stack and N-D ordered-sharding replay
  ending at `6f649f8`.
- The `b6865e7c` branch is merged into the current paper branch. Conflict
  resolution retains current CP/FlexAttention behavior and restores the
  DeepSeek aliases, symbolic handling, dynamic estimator, and checkpointed
  layer initialization.

## Reproduction policy

`experiment_lock.toml` is authoritative. Active campaigns contain no source
or runtime pins and cannot override the locked `default` DTensor backend.
Original campaign files are preserved under `provenance/campaigns/`.

A moving branch name is not evidence for a result. Every report records the
exact harness content manifest, runtime source commits, lock digest, source
tree digests, runtime versions, resolved campaign, asset hashes, and command.
