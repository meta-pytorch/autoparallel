# Unified legacy-DTensor stack provenance

## TorchTitan

The locked source is `kaijian/ap-submission-4d` in `AlbedoWang/torchtitan` at
`c0fac771`, a child of `85c216d5`, a child of `3b24e620`, which is a child of
`4b46d18d` on `kaijian/deepseek-baseline-parity-20260918`. Its history has `383cae9f` as an actual
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
- `69e752e3`, `2876c9ac`, `63d411f9`, and `e8068c44` replace the exact-FQN
  AutoParallel SAC boundary rule with the fail-closed structural A2A-to-WO
  matcher and report its decisions.
- `a68fa447` is the compatibility cherry-pick of TorchTitan PR #4859. It
  reorders HSDP gradient reduction to reduce-scatter before all-reduce in the
  shared GraphTrainer pass pipeline used by manual and AutoParallel modes.
- `a25d7db3` sets `insert_overlap_deps=False` in the AutoParallel
  full-Inductor configs, so Inductor's overlap scheduling adds no control
  deps.
- `b32acd82` passes the ms-converted AutoParallel estimator to Inductor's
  overlap scheduling (`aten_distributed_optimizations.custom_runtime_estimation`),
  so it uses the same estimates, including the NCCL cost profile, as the AP
  pass.
- `7966c411` tunes Inductor's overlap-scheduling parameters for the
  AutoParallel full-Inductor compile: `compute_overlap_multipler=0.5` and
  `max_compute_pre_fetch` 10 -> 20.
- `4b46d18d` restores `max_compute_pre_fetch` to 10 and keeps
  `compute_overlap_multipler=0.5`.
- `cf091127` (child of `4b46d18d`) sets `aten_distributed_optimizations.enable_overlap_scheduling`
  to False, so Inductor runs no overlap pass (pass 2) after the AP
  reordering/bucketing pass. Experiment pin for the LLaMA 3 8B 4D CP hang
  (v20: ring attention traces per-CP-rank graphs and pass 2 orders mesh_cp
  collectives differently across CP ranks).
- `3b24e620` (child of `cf091127`)
  replaces SDPA ring-attention CP with Ulysses all-to-all in all three arms:
  `apply_cp_to_forward` (tt, graph_trainer manual) uses funcol
  `all_to_all_single` on the CP group, and the AutoParallel CP local_map body
  uses `autoparallel.collectives.all_to_all("cp")`. Every CP rank traces the
  same graph (ring attention did not, and v20-v22 apgt hung on mesh_cp order).
  Ulysses needs contiguous sequence shards, so the LLaMA 3 3D/4D settings use
  `context_parallel_load_balancer=None` and the trainer rejects default-backend
  SDPA CP with a load balancer.
- `85c216d5` (child of `3b24e620`)
  restores `enable_overlap_scheduling=True` (the `4b46d18d` setting): with
  Ulysses every CP rank traces the same graph, so the reason for `cf091127` no
  longer holds.
- `c0fac771` (child of `85c216d5`)
  builds the AutoParallel runtime estimator on the mesh AutoParallel lowered the
  model on, recorded by `AutoParallelGraph.apply_placement_for_fx_module`.
  Before, it used the TorchTitan legacy (dp_replicate, fsdp, tp) mesh, whose
  process groups differ from the AutoParallel mesh for 4D, 3D CP, and EP, so the
  estimator priced every collective on the flattened world. It is the runtime
  pin.

The 3D port includes only the approved legacy-DTensor AP configuration, CP
input-ownership seam, DP-shard/CP/TP mesh, CP-aware SDPA, DTensor output, and
placement save/load behavior. The unrelated BlockMask private-API change is
intentionally excluded.

## AutoParallel

The maintained harness remains on `kaijian/paper-submission`; the 4D line
(v19-v30) is on `kaijian/paper-submission-4d`, which merges the runtime
commits `9a16ded`..`98b20b4` and `0a551aa` at their original SHAs. Runtime
source is pinned to the commit immediately before the lock/harness-only
updates.

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
  bucket-consensus coverage.
- `048ce7c` adds the parameter-axis constraint used to require HSDP parameters
  to replicate on `dp_replicate` while leaving `fsdp` and `tp` to the solver.
- `2f7c0bb`, `481f6ea`, `8d17878`, and `af91dc2` propagate physical shard order
  through gradient producers and require lowering to match the modeled
  collectives and cost.
- `8004eb3` stages orthogonal HSDP `Partial -> Replicate` reductions before the
  remaining ordered redistribution.
- `63582207` is the compatibility cherry-pick of AutoParallel PR #535. The
  selected campaigns opt into its calibrated `h100_nvswitch_roce_400g` profile.
- `1e7dae8` keeps the outer autograd context (`grad_fn_seq_nr`) when
  AutoParallel interprets its compiled graphs, so GraphTrainer's selective
  activation remat sees the AP backward region.
- `46d03f9` is the compatibility cherry-pick of AutoParallel PR #536. It
  passes AutoParallel runtime estimates to the overlap scheduler in
  milliseconds.
- `a300c78` rejects cross-bucket cycles in the patched greedy bucketing, so
  HSDP+TP 2-hop parameter gathers no longer fail the bucket merge's
  topological sort.
- `77cfe2c` runs orthogonal HSDP `Partial -> Replicate` reductions after the
  ordered redistribution instead of before it (reduce_scatter on `fsdp`, then
  all_reduce on `dp_replicate`), and prices them last in the solver-side
  logical plan. At `dp_shard >= 3` the concrete plan then matches the logical
  plan again, so the FSDP-like storage order is no longer rejected and the
  default-order all_to_all chains disappear.
- `952e69d` prices the same orthogonal `Partial -> Replicate` reductions last
  in the solver's edge cost (`estimate_strategy_comms_cost`) whenever the edge
  also has a `Partial -> Shard` mesh dim, as lowering runs them. Before, 3D
  gradients were priced all_reduce-first, which pushed Muse Glimmer 4x8x2 to
  gather activations over `fsdp` through tp all_to_all chains.
- `cf5df05` leaves the approximate solver's `max_time_s` unbounded by default.
  Each rank solves independently, and on Muse Glimmer 4x8x2 the 60 s cap
  truncated the polish at a speed-dependent point: 3 of 64 ranks returned a
  different plan and the job hung in mismatched collectives. The sweep budgets
  (`bp_iters`, `max_sweeps`, `star_passes`) still bound the solve.
- `9a16ded` (cherry-pick of `02ea6f4`) makes view inputs contiguous
  before DTensor wrapping in static lowering, so `apply_sharding` no longer
  runs uncached full-mesh clone sharding propagation per view op (hours on the
  LLaMA 3 4D mesh).
- `7a14fa2` (cherry-pick of `b723cd2`) keeps mesh dims the parameter
  storage does not shard (the HSDP replicate dim) in place when projecting the
  storage order onto a gradient producer's inputs. Without it the 4D
  lm_head-backward tangent keeps the default order and the lowered graph
  differs per rank (step-1 deadlock).
- `62a6085` orders a parameter's storage by the mesh dims its forward
  chain releases (never-released outermost, later releases further out) when
  the earlier ordered-storage gates left it unordered, and keeps the order only
  if every param/grad chain edge then runs exactly the collectives the solver
  priced. On the LLaMA 3 4D mesh the earlier gates ordered only lm_head, so
  FSDP gathers and gradient reductions lowered to unpriced all_to_all chains.
- `98b20b4` prices the default-order redistributions an approximate
  solve selects that have no one-collective-per-mesh-dim plan by the plan
  lowering emits for them, and re-solves until no unpriced one is selected.
  On the LLaMA 3 4D 2x2x2x2 mesh, gathering the middle mesh dim of S0S0S1S0
  activations lowered to all_to_all, all_gather, all_to_all priced as one
  all_gather.
- `0a551aa` (child of `98b20b4`)
  makes the split-dim strategy seed honor `add_parameter_axis_constraint`: the
  one-dimensional seed solve of a constrained mesh dim gets the same parameter
  placement constraint as the full problem. Before, the seed sharded parameters
  on that dim (1/size memory cap), so the strategy-radius ball around the seed
  excluded the solver's best parameter storage; on the LLaMA 3 4D mesh
  feed_forward.w2 was stored S0 on tp and paid a tp all_to_all every step. It is
  the runtime pin.

## Reproduction policy

`experiment_lock.toml` is authoritative. Active campaigns contain no source
or runtime pins and cannot override the locked `default` DTensor backend.
Original campaign files are preserved under `provenance/campaigns/`.

A moving branch name is not evidence for a result. Every report records the
exact harness content manifest, runtime source commits, lock digest, source
tree digests, runtime versions, resolved campaign, asset hashes, and command.
