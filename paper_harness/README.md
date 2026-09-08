# Permanent AutoParallel experiment harness

This branch is the single harness for new TorchTitan, GraphTrainer, and AutoParallel experiments. It starts from the Muse Glimmer harness because that version already separated paired GraphTrainer runs, native TorchTitan runs, profiler-off performance phases, same-allocation trace phases, serialized-config parity, package verification, and fail-closed analysis.

The historical `native_torchtitan/` and `paired_graphtrainer_ap/` directories remain as provenance. New campaigns use the shared `harness/`, `launcher/`, and `workloads/` code.

## Source ownership

TorchTitan and AutoParallel are external inputs. The harness never checks out, patches, or writes either repository. A campaign pins each source commit and the CLI receives existing checkout paths.

Clean checkouts are required by default. A campaign may explicitly select `dirty_policy = "snapshot"`; that records the base commit, porcelain status, binary diff, untracked-file hashes, complete source-tree hash, and packaged-tree hash. This mode is intended for evaluating a change before submitting it upstream, not for silently bypassing provenance checks.

The vendored snapshot and the clean AutoParallel/TorchTitan submission stacks are recorded in [UPSTREAM.md](UPSTREAM.md). Historical campaign pins were intentionally left unchanged; a new campaign must explicitly pin the submission commits when it opts into this stack.

## Stable arm profiles

| Profile | Meaning |
| --- | --- |
| `tt_main_manual_jit_v1` | TorchTitan MainTrainer, manual parallelization, ordinary `torch.compile` |
| `gt_manual_aot_v1` | GraphTrainer `aot_fx_trace`, manual parallelization, full Inductor |
| `apgt_v1` | GraphTrainer `aot_fx_trace`, AutoParallel placement/lowering, full Inductor |
| `gt_manual_cp_legacy_v1` / `apgt_cp_legacy_v1` | Frozen final-fairness 3D CP pair |
| `ap_backend_legacy_v1` | Historical `autoparallel_backend` path only |

For the current `apgt_v1` contract, both GraphTrainer arms serialize the same compile settings except for `compile.enable_autoparallel`. The manual arm keeps GraphTrainer's joint transformer-block bucketing pass. The AutoParallel arm is selected by the source pass builder, skips that manual pass, and passes the fixed AutoParallel overlap/bucketing settings to terminal full Inductor. A source checkout that does not prove this contract is rejected; changing the contract requires a new profile version and explicit review.

The model adapters, model-specific input/output constraints, memory constraints, tracing path, and placement application are fixed by the `apgt_v1` source contract. The solver and placement solve/save/load mode remain declared experiment variables.

## AutoParallel solver settings

The companion TorchTitan submission branch accepts the following fields under `[compile]`. These are forwarded to the existing PR523 AutoParallel constructor or `optimize_placement`; the harness does not patch either source tree.

| Field | Default | AutoParallel destination |
| --- | --- | --- |
| `autoparallel_solver` | `"ilp"` | `solver` |
| `autoparallel_fast_build` | `true` | `fast_build` |
| `autoparallel_lazy_costs` | `"auto"` | `lazy_costs=None`; `"lazy"` and `"eager"` map to `True` and `False` |
| `autoparallel_strategy_radius` | `2` | `strategy_radius`; placement replay forces `0` |
| `autoparallel_optimality_check` | `false` | `optimality_check` |
| `autoparallel_approx_candidate_limit` | `128` | `approximate_options.candidate_limit` |
| `autoparallel_approx_bp_iters` | `400` | `approximate_options.bp_iters` |
| `autoparallel_approx_bp_tol` | `0.001` | `approximate_options.bp_tol` |
| `autoparallel_approx_max_sweeps` | `12` | `approximate_options.max_sweeps` |
| `autoparallel_approx_max_time_s` | `60.0` | `approximate_options.max_time_s` |
| `autoparallel_approx_star_passes` | `2` | `approximate_options.star_passes` |
| `autoparallel_approx_max_star_children` | `32` | `approximate_options.max_star_children` |
| `autoparallel_approx_group_domain_limit` | `512` | `approximate_options.group_domain_limit` |

`autoparallel_placements_save_path` and `autoparallel_placements_load_path` retain the prior placement JSON behavior and are mutually exclusive. Approximate-only options are forwarded only when `autoparallel_solver = "approx"`. The integration deliberately does not expose `dynamic`, `cost_model`, repeated-subgraph handling, graph adapters, constraints, or lowering hooks as campaign knobs.

## Campaign format

`campaign.toml` is the authored input. Validation writes a complete `resolved_campaign.json`, serialized TorchTitan config for every phase/arm, `source_lock.json`, parity reports, and package manifests.

Campaigns declare:

- source remotes, commits, and dirty policy;
- model, attention backend, dataset revision, tokenizer and input identity;
- local/global batch, sequence length, precision, optimizer, scheduler and SAC;
- DP-replicate, DP-shard, TP, CP, PP and EP settings;
- arm profiles, phase order, warmup/measurement windows and trace ranks;
- Kineto/TORCH_TRACE behavior, MAST resources, environment and artifacts;
- the variable under test and exact config/environment paths allowed to differ.

The launcher does not impose `TORCHINDUCTOR_COMPILE_THREADS`: PyTorch's runtime default is preserved unless a campaign pins the variable in `[mast.environment]`. Compile parallelism is recorded experiment state because it can materially change cold-start duration and rank skew.

Optional `[measurement.primary]` settings select an exact historical metric from structured logs or rank-0 TensorBoard. Optional `[comparison.acceptance]` settings compare that metric's relative arm gap with a declared reference and percentage-point tolerance. Interleaved campaigns may map each arm to its performance phase with `[comparison.performance_phase_by_arm]`.

Common TorchTitan config sections use their native names. Less common settings go in `[torchtitan.overrides]` as dotted paths. Unknown upstream fields fail when the exact pinned source parses the generated arguments. An arm or phase may override a common dotted path; the resulting serialized configs are compared before submission.

`training.local_batch_size` is the microbatch per data-parallel replica. TP and CP ranks share or shard that logical batch; they do not multiply global batch.

Scaling and sequence sweeps use `[[matrix.points]]`. Select one immutable point per MAST allocation with `--point`.

## Data path

New streaming campaigns use a pinned dataset/tokenizer and a deterministic index schedule. The manifest is validated before launch and all paired arms use the same schedule. The measured training loop performs no tensor hashing, file append, or added synchronization. Historical replay/cache modes remain explicit in their presets.

Campaigns may declare `data.preflight_auditor` as a `workloads.*` module. Its `audit` function runs on every rank before any training phase; one TP representative per DP rank performs expensive batch hashing and all ranks must produce a passing or explicitly skipped audit record.

The old LLaMA sequence-length and DeepSeek adapters hashed tensors and appended JSONL records from the dataloader iterator. That instrumentation is removed from the permanent adapters. Imported historical reports retain an explicit warning that their end-to-end timing included it.

## Commands

Use a Python interpreter from the exact compatible environment:

```bash
python -m harness.cli validate campaigns/muse_glimmer_30b_scaling.toml \
  --point 16gpu --mode gate \
  --torchtitan-root "$TORCHTITAN_ROOT" \
  --autoparallel-root "$AUTOPARALLEL_ROOT" \
  --asset-root muse_glimmer="$MUSE_ASSETS" \
  --asset-root muse_input_manifests="$MUSE_INPUT_MANIFESTS" \
  --asset-root c4_hf_cache="$C4_HF_CACHE" \
  --output "$TASK_ROOT/validation"

python -m harness.cli package campaigns/muse_glimmer_30b_scaling.toml \
  --point 16gpu --mode gate \
  --torchtitan-root "$TORCHTITAN_ROOT" \
  --autoparallel-root "$AUTOPARALLEL_ROOT" \
  --asset-root muse_glimmer="$MUSE_ASSETS" \
  --asset-root muse_input_manifests="$MUSE_INPUT_MANIFESTS" \
  --asset-root c4_hf_cache="$C4_HF_CACHE" \
  --attempt "$TASK_ROOT/attempts/001"

python -m harness.cli render-mast --attempt "$TASK_ROOT/attempts/001"
python -m harness.cli submit --attempt "$TASK_ROOT/attempts/001"
python -m harness.cli analyze campaigns/muse_glimmer_30b_scaling.toml \
  --point 16gpu --mode gate --attempt-root "$TASK_ROOT/attempts/001" \
  --output "$TASK_ROOT/attempts/001/analysis"
```

`--mode gate` resolves each campaign to a two-step functional phase plus a two-step trace smoke; it can validate integration but can never authorize a performance conclusion. The default `--mode formal` keeps the campaign's declared performance and trace phases.

`render-mast` performs the TorchX dry run, parses the MAST scheduler request, downloads the newly generated workspace and payload fbpkgs, verifies complete content hashes and executable bits, then re-imports the downloaded sources and re-parses every config. `submit` accepts only that audited immutable payload and an unchanged launcher. Every comparison runs all arms sequentially in one allocation, with a fresh process, rendezvous port, and compiler cache per arm. Allocation identity and rank-to-GPU mapping must remain unchanged across arms. The dry-run audit also requires `hpcClusterUuid=MastGenAICluster`; combined with the `grandteton_80g_roce` resource, this selects the H100 RoCE subtype in the GenAI MAST cluster (accounted as `MACHINE_TYPE_T20_GRAND_TETON_HBM3_ROCE_GENAI`).

Performance phases keep Kineto and TORCH_TRACE disabled. Trace phases run after performance in the same allocation and enable Kineto and TORCH_TRACE only on declared ranks; tlparse consumes the saved TORCH_TRACE directory offline.

## Canonical campaigns

- `campaigns/llama3_8b_2d_scaling.toml`: 8 through 128 GPUs, with TorchTitan TP, GraphTrainer manual, and AP+GraphTrainer arms. The GraphTrainer-manual/AP pair receives an additional strict one-toggle audit.
- `campaigns/llama3_8b_3d_legacy.toml`: final fair DP2 x CP2 x TP2 pair.
- `campaigns/llama3_8b_seqlen.toml`: 2K through 32K at 32 GPUs.
- `campaigns/muse_glimmer_30b_scaling.toml`: 16 through 128 GPUs.
- `campaigns/deepseek_v3_16b.toml`: 16/32-GPU workload; historical failed runs remain non-comparative.
- `campaigns/repro_*.toml`: fixed 32-GPU reproduction presets with explicit historical metric and relative-gap acceptance definitions.

Legacy evidence is imported read-only with `analyze`; missing or failed artifacts never become a performance conclusion.

## Local validation

```bash
VALIDATION_ROOT="$TASK_ROOT/validation/static" \
  PYTHON_BIN=/path/to/compatible/python scripts/run_static_preflight.sh
```

The initial MAST validation is functional only: the smallest existing real topology for LLaMA 2D, LLaMA 3D, Muse, and DeepSeek performs source/package/config/allocation checks, real compile/train steps, and an isolated trace smoke. It is not a latency or speedup benchmark.
