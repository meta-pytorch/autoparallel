# Enable Native `view → mm → view` in AutoParallel via DTensor Strided Sharding

## Summary

AutoParallel currently rewrites PyTorch's `view → mm → view` decomposition of `nn.Linear` into `einsum` (see `_APPLY_VIEW_MM_VIEW_PATTERN` in `autoparallel/api.py`). That workaround was introduced in AP #26/#424 because DTensor's view ops could not faithfully propagate sharding across flatten→mm→unflatten.

DTensor has since gained native support for this via `_StridedShard` placements and the `mm_single_dim_strategy` path (upstream pytorch PR #172385). AutoParallel, however, does not reach that path — it uses the legacy `register_op_strategy` mm rule and explicitly strips `_StridedShard` from its placeholder expansion.

This PR wires AP up to use the upstream single-dim mm path, enumerates `_StridedShard` variants from upstream input strategies, and fixes the `is_shard()`-miss bugs in AP's local-shape/FLOP/validity checks. Benchmarks on LLaMA3-8B confirm the change is a strict win on both solver time and solver objective; it also unblocks 32-layer configs that the einsum path cannot solve in a reasonable time.

## Headline Result

Benchmarked on LLaMA3-8B at PR #424-class config (`dim=4096, seqlen=8192, 64-rank 8×8 fake-PG mesh`, `cost_model=nccl`, single-H100, fake collectives):

| Scale | Solver time | Solver objective (NCCL cost proxy) |
|---|---|---|
| LLaMA3-8B **2-layer** | NATIVE 45.7s vs EINSUM 76.1s (**-40%**) | NATIVE 57576 vs EINSUM 57761 (**-0.32% cheaper**) |
| LLaMA3-8B **32-layer** | NATIVE **29.5 min** vs EINSUM **>4 h (timed out)** | NATIVE 520184, EINSUM unknown |

- Objectives reproducible across seeds 0 and 1 (solver is deterministic given the graph).
- EINSUM's strategy-space-per-node is ~1.5-2× larger (einsum `bsk,kn->bsn` has 4 axes vs. mm `mk,kn->mn` with 3), making ILP scaling superlinearly worse at depth.
- `_StridedShard` never appears in the chosen strategies for the LLaMA3-8B configs tested. Phase 1's `_StridedShard` enumeration is correct when dormant and ready when exercised by other workloads.

## What's Done

### 1. Route mm-family ops through the single-dim path (opt-in)

In `autoparallel/shardings/dtensor_sharding_helpers.py`:
- Added `_PREFER_SINGLE_DIM_OPS = {mm, addmm, bmm, baddbmm, _scaled_mm}`.
- Added `ENABLE_SINGLE_DIM_MM_FAMILY: bool = False` (opt-in toggle).
- `get_op_strategy` now prefers the upstream single-dim path for those ops **when the flag is True**, bypassing the legacy `op_strategy_funcs` that otherwise shadows it. Default behavior is unchanged.

To opt in, set `dtensor_sharding_helpers.ENABLE_SINGLE_DIM_MM_FAMILY = True` before constructing `AutoParallel`, or use the `enable_single_dim_mm_family` pytest fixture in new tests.

### 2. Enumerate `_StridedShard` variants in placeholder expansion

`_try_single_dim_strategy` collects candidate `split_factor`s from upstream input OpStrategies and emits `Shard(d)` plus one `_StridedShard(d, sf)` per candidate `sf` for every `_ShardingPlaceholder` slot. Previous plain-`Shard`-only behavior is preserved when no input carries `_StridedShard`.

### 3. Add `is_shard_like()` helper and fix `is_shard()`-miss bugs

`_StridedShard.is_shard()` returns `False`, which caused several AP call sites to silently treat `_StridedShard` dims as unsharded (over-counting FLOPs, wrong local shapes, keeping invalid strategies). Fixed by:

- New `is_shard_like(p)` helper in `shardings/dtensor_sharding_helpers.py`.
- Applied at:
  - `apply_sharding.py:_localize_shape_arg` — local shape was not being divided by mesh_size for `_StridedShard` dims.
  - `cost_models/compute_estimation.py:_get_sharded_shape_stride` — over-counted FLOPs for strided strategies.
  - `shardings/propagation_rules.py:remove_invalid_configs` (strategy-shape validity), LayerNorm fwd/bwd reduction-axis checks, `aten.pad` trailing-dim removal.
  - `shardings/placement_options.py` — flex_attention Q/KV dim validity adjustment.

### 4. Tests

Three new tests in `tests/test_propagation_rules.py`:
- `test_mm_strategy_enumerates_strided_shard` — `_StridedShard`-bearing input yields `_StridedShard`-bearing output with matching `split_factor`.
- `test_mm_strategy_plain_shard_still_present` — regression: plain-Shard inputs do not spuriously produce `_StridedShard` outputs.
- `test_mm_strategy_backward_grad_weight_strided` — backward mm with `_StridedShard` on both contracting-dim inputs yields strategies with Partial output.

All existing tests in `tests/test_optimize_placement.py` (11 tests) pass with both `_APPLY_VIEW_MM_VIEW_PATTERN = True` and `False`. The three new tests also pass in both configurations.

### 5. End-to-end numerical correctness

`pytorch/agent_space/numerical_check_linear3d.py` runs a small 3-D Linear model through AP with both flag values and compares forward output to a single-device reference: **max abs diff = 0.000e+00** in all pairwise comparisons.

## What's Next

1. **Review + merge this PR**, which lands the routing + `_StridedShard` enumeration behind `ENABLE_SINGLE_DIM_MM_FAMILY = False`. Zero default-behavior change.
2. **Real training throughput with `compile=True` and a real multi-rank setup**, with the flag flipped to `True`. The solver objective (NCCL-cost proxy) is already cheaper on the single-dim path; this would confirm step-time parity or improvement. Out of scope for this PR.
3. **Flip `ENABLE_SINGLE_DIM_MM_FAMILY = True`** as the default in a follow-up PR once step-time is confirmed.
4. **Flip `_APPLY_VIEW_MM_VIEW_PATTERN = False`** as the default (separate toggle, but naturally pairs with step 3 for Linear workloads).
5. **Remove `_replace_view_mm_view_with_einsum`** and its pattern matchers in `autoparallel/graph_passes/graph_utils.py` after a release with no regressions.

## Not in Scope

- Making PyTorch stop decomposing `nn.Linear` (separate upstream effort; the TODO at `autoparallel/graph_passes/graph_utils.py:247` points to it).
- `nn.Bilinear`, scaled_dot_product_attention, or other non-mm matmul paths that don't go through the view-flatten.
- `is_shard()`-miss sites in `cost_models/collective_runtime_estimation.py` (lines 128, 146, 176, 194, 235): those are gated behind upstream `redistribute_cost` which returns `inf` for any `_StridedShard`-involving transition, so the solver avoids them regardless. Worth cleaning up later for defense-in-depth.

## Known Caveats

1. **Conservative `_StridedShard` redistribute cost** (`torch/distributed/tensor/_collective_utils.py:535-536`): returns `inf` for any transition between specs where one has `shard_order=None` (true for all `_StridedShard` specs). This means the solver cannot cross-redistribute between strided and non-strided mid-graph — acceptable for the view→mm→view chain (end-to-end zero-cost match), restrictive for graphs that need it elsewhere.
2. **Strategy-space blow-up** from enumerating `_StridedShard(sf)` variants is bounded because sf is drawn only from upstream-observed split_factors. Empirically no impact on LLaMA3-8B solve times.
3. **`_StridedShard` not exercised by LLaMA3-8B at tested hyperparameters**. The solver did not choose strided strategies even when enumerated; NATIVE beats EINSUM on solver time and objective without them. The capability remains useful for workloads that do exercise it (e.g. `[Shard(batch), Shard(seq)]` input on a 2-D mesh where batch×seq sharding interleaves).

## Artifacts

Code changes:
- `autoparallel/shardings/dtensor_sharding_helpers.py` — `_PREFER_SINGLE_DIM_OPS`, `is_shard_like`, extended `_try_single_dim_strategy`, updated `get_op_strategy`.
- `autoparallel/apply_sharding.py`, `autoparallel/cost_models/compute_estimation.py`, `autoparallel/shardings/propagation_rules.py`, `autoparallel/shardings/placement_options.py` — `is_shard_like` adoption.
- `tests/test_propagation_rules.py` — three new mm-strategy tests.

Validation scripts (not part of the PR, under `pytorch/agent_space/`):
- `repro_mm_strided.py` — pre-change comparison (upstream single-dim vs. legacy `_mm_like_strategy`).
- `verify_ap_mm_strided.py` — post-change verification on synthetic schemas.
- `bench_llama3_8b.py`, `bench_llama3_8b_einsum_only.py` — full LLaMA3-8B benchmark.
- `numerical_check_linear3d.py` — end-to-end forward numerical check.
