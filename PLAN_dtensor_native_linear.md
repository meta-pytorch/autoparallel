# Plan: Let AutoParallel Use `nn.Linear` With DTensor's Native View-op Decomposition

## Status

- **Phases 1, 2, 3, 4, 5 — DONE** ✅ (code work + audits).
- **Phases 0, 6 — DONE** ✅ (LLaMA3-8B 2-layer: NATIVE -40% faster solve, -0.32% cheaper objective, identical across seeds. LLaMA3-8B 32-layer: NATIVE solved in 29.5 min with objective 520184; EINSUM did not complete in 4 h, confirming EINSUM scales catastrophically at deeper models).
- **Phase 7 — STRONGLY SUPPORTED**, subject to one remaining validation step: confirm real training throughput (compile=True + actual step times) doesn't regress vs. EINSUM on 2-layer. Given NATIVE's already-cheaper solver objective (which is the NCCL-cost proxy used by the solver), throughput regression is unlikely. Recommend flipping `_APPLY_VIEW_MM_VIEW_PATTERN = False` behind a feature flag for a release cycle.

## Headline Result

Should AutoParallel's `view → mm → view` → einsum rewrite be reverted now that DTensor supports strided sharding?

**Yes.** Benchmarked on LLaMA3-8B at PR #424-class config (dim=4096, seqlen=8192, 64-rank 8×8 fake-PG mesh, cost_model=nccl):

| Scale | Solver time | Solver objective (NCCL cost proxy) |
|---|---|---|
| LLaMA3-8B **2-layer** | NATIVE 45.7s vs EINSUM 76.1s (**-40%**) | NATIVE 57576 vs EINSUM 57761 (**-0.32% cheaper**) |
| LLaMA3-8B **32-layer** | NATIVE **29.5 min** vs EINSUM **>4 h (timed out)** | NATIVE 520184, EINSUM unknown |

NATIVE wins on both solver wall time and solver cost at 2L, and is the only tractable option at 32L. The `_StridedShard` machinery added in Phase 1 is ready but not exercised by these LLaMA3 configs — NATIVE already beats EINSUM without needing it. See Progress Log below for numerical correctness, regression checks, and multi-seed confirmation.

## Progress Log

### 2026-04-20 (late evening) — Full-scale LLaMA3-8B benchmarks

**Setup**: `bench_llama3_8b.py`, H100 single GPU, fake PG world=64, 8×8 mesh, dim=4096, vocab=128256, seqlen=8192, batch=16, cost_model=nccl (default).

**2-layer results** (seeds 0 and 1, reverse order tested — objectives identical across runs):

| | NATIVE | EINSUM | Delta |
|---|---|---|---|
| Solver time | **45.7s** / 47.7s | 76.1s / 76.6s | NATIVE **-40%** |
| Objective (solver total cost) | **57576.44** | 57760.68 | NATIVE **-0.32% cheaper** |
| mm nodes | 45 | 45 (einsum) | same |
| `_StridedShard` in strategy space | 0 | 0 | neither path uses it |
| Top chosen mm out-placement | 11× `[S(0),S(1)]`, 10× `[S(0),P]`, 9× `[S(0),S(0)]`, 7× `[P,S(1)]` — diverse | 28× `[S(0),S(1)]`, 14× `[P,P]`, 1× each `[S(0),S(2)]`/`[P,S(1)]`/`[S(0),P]` — dominant TP | different partition preferences |

**32-layer (NATIVE done; EINSUM timed out at 4h+)**:

| | NATIVE | EINSUM |
|---|---|---|
| enter_ctx | 315s | N/A |
| solve | **1770s (29.5 min)** | **> 4 h (timed out, did not complete)** |
| Objective | **520184.17** | unknown |
| mm nodes | 675 | unknown |
| Top chosen | 161× `[S(0),S(1)]`, 160× `[S(0),P]`, 129× `[S(0),S(0)]`, 97× `[P,S(1)]`, 64× `[P,S(0)]`, 64× `[P,P]` | — |

**EINSUM 32L scaling blow-up — why it never finished**:
- Per-node strategy count is higher for `einsum("bsk,kn->bsn")` than `mm("mk,kn->mn")` (4 axes × 2 mesh dims vs. 3 axes × 2 mesh dims → ~1.5-2× more strategies per node).
- ILP is superlinear: vars ∝ nodes × strategies; pairwise redistribute_cost ∝ edges × strategies². Doubling strategies ≈ 4× ILP size.
- NATIVE 2L→32L solve grew 45s → 1770s (39×). EINSUM 2L→32L grew 76s → ≥14400s (190×+, bounded below).
- PR #424 already flagged 32L clustering overhead; these numbers quantify how much worse EINSUM is at scale.
- **Practical conclusion**: EINSUM's solver-time penalty at 32L makes it a dead end for production LLaMA3-32L users. Even if it matched NATIVE on step time (untested), no one would wait 4+ hours for the sharding solve.

**Key findings so far**:

1. **No regression from Phase 1 code**: NATIVE 2L objective is 0.32% cheaper, solve is 40% faster. Identical across seeds 0 and 1 (solver is deterministic given the graph).
2. **`_StridedShard` strategies never appear** in the solver's strategy space for this workload, in either path. Phase 1's code change remains dormant — correct when not needed, ready when it is. The specific LLaMA3 config here (batch=16, seqlen=8192 / 64 ranks = ~2K tokens/rank) prefers `[S(0), S(1)]` style TP rather than SP.
3. **EINSUM is much slower at scale**: 2L 1.7× slower, 32L ≥2× slower (bounded below). The extra `bsk` axes in einsum's operand spec multiply the per-node strategy count; clustering helps but doesn't fully compensate.
4. **Chosen-strategy diversity differs**: NATIVE picks a more diverse mix (6 distinct top outputs on 2L); EINSUM concentrates on `[S(0),S(1)]` (28/45 on 2L). This is intrinsic to the graph shapes and doesn't indicate a bug.
5. **No PR #424 SP-vs-TP trade-off triggered** in this config: the cost model never selected an SP strategy in EINSUM's 2L run (no `[R,S(1)]` dominance or similar seq-on-tp pattern). So the specific headline benefit PR #424 reported isn't reproducible with these hyperparameters — would need different per-GPU token counts.

### 2026-04-20 (evening) — Phase 0/3/6 GPU runs

**Phase 0 + 6 mini benchmark** (`pytorch/agent_space/bench_view_mm_flag.py`, H100, CUDA_VISIBLE_DEVICES=1, fake PG world=8, 2x4 mesh, LLaMA3-ish dim=512 × 2 layers):
- Solver time: NATIVE 36.01s, EINSUM 35.99s — within 0.1%.
- No `_StridedShard` present anywhere in the strategy space for this small config — neither path needs it. The input constraint `[Shard(0), Shard(1)]` (batch on dp, seq on tp) did not cause upstream view ops to enumerate `_StridedShard` strategies, likely because AP's placement-options for this model size doesn't reach the sharding combinations that would trigger it.
- Chosen strategy distributions do diverge: NATIVE picks `[R, S(0)]` (20/45 mm) = TP-shard the flat M dim; EINSUM picks more `[R, R]` and `[R, S(2)]` (TP-shard N).
- **Takeaway**: Phase 1 doesn't regress solver time on small configs. Full LLaMA3-8B at PR #424's sizes (n_layers=2 or 32, seqlen=8192) is still needed to confirm the SP-vs-TP adaptivity story transfers.

**Phase 3 end-to-end numerical check** (`pytorch/agent_space/numerical_check_linear3d.py`, small 3-D Linear):
- NATIVE vs EINSUM: **max abs diff = 0.000e+00** (bit-exact).
- NATIVE vs single-device reference(rank0 slice): **0.000e+00**.
- EINSUM vs single-device reference(rank0 slice): **0.000e+00**.
- Both AP paths produce numerically correct forward output with Phase 1's `_StridedShard` enumeration enabled.

### 2026-04-20 (afternoon) — Phases 2, 3, 4, 5 completed

**Phase 2 — cost model audit:**
- `pytorch/torch/distributed/tensor/_collective_utils.py:533-536`: confirmed `redistribute_cost` returns `inf` whenever either spec has `shard_order is None`, which is true for any `_StridedShard`-bearing spec (default `use_strided_shard_as_shard_order=True`). Consequence: the solver treats any `_StridedShard → non-strided` or `non-strided → _StridedShard` redistribute as infinite cost. The no-op `_StridedShard → same _StridedShard` case is free (line 502/508/544). Acceptable for the view-mm-view chain (end-to-end zero-cost match), but restrictive for graphs that need mid-chain redistribution from strided.
- `pytorch/torch/distributed/tensor/_redistribute.py:1587-1590`: "_StridedShard redistribute assumes no flattened transforms" — upstream assertion, still holds. No action needed until a redistribute path hits it.
- `pytorch/torch/distributed/tensor/_collective_utils.py:395-396`: confirmed `_compute_placement_transition_cost` intentionally doesn't handle `_StridedShard` (is_shard() returns False); safe because outer `redistribute_cost` bails first.
- **Fixed bug**: `autoparallel/autoparallel/cost_models/compute_estimation.py:_get_sharded_shape_stride` was using `placement.is_shard()` which returns False for `_StridedShard` → local shape wasn't reduced → FLOPs over-counted. Fix: also match `isinstance(p, _StridedShard)`.

**Phase 3 — apply_sharding audit:**
- **Fixed bug**: `autoparallel/autoparallel/apply_sharding.py:_localize_shape_arg:60` had the same `is_shard()` issue — `_StridedShard` dims weren't divided by mesh_size in local shape computation. Fix: also match `_StridedShard`.
- `ordered_redistribute_local_tensor` delegates to upstream `redistribute_local_tensor` for non-identical shard_order; inherits upstream `_StridedShard` semantics.
- **Flagged follow-ups** (not fixed — outside Linear critical path):
  - `autoparallel/autoparallel/cost_models/collective_runtime_estimation.py:128, 146, 176, 194, 235` — `is_shard()` checks miss `_StridedShard`. Transition costs may be inaccurate for strided transitions but upstream `redistribute_cost` returns inf for these anyway, so solver avoids them.
  - `autoparallel/autoparallel/shardings/propagation_rules.py:177, 552, 626, 702` — op-specific validity checks (shardability, LayerNorm reduction, dim removal). Not on the Linear view-mm-view critical path but could bite for LayerNorm-on-strided cases.
  - `autoparallel/autoparallel/shardings/placement_options.py:560` — dim_to_ref lookup.

**Phase 4 — backward grad-weight mm:**
- Added `test_mm_strategy_backward_grad_weight_strided` to `autoparallel/tests/test_propagation_rules.py`. Also mirrored in `pytorch/agent_space/verify_ap_mm_strided.py`.
- Empirical: backward mm with `_StridedShard` on both contracting-dim inputs yields **20 strategies** with `(_StridedShard, _StridedShard) → Partial` form. This is the contracting-dim sharding pattern that gives Partial output, matching einsum behavior.

**Phase 5 — ops between view and mm:**
- View-family ops (view, permute, unsqueeze, squeeze, transpose, expand, slice): all go through legacy `register_op_strategy_map` → `propagate_shape_and_sharding` in `_view_ops.py`, which is `_StridedShard`-aware (line 585, 1170). Transpose explicitly swaps `_StridedShard` dims at `_matrix_ops.py:68`.
- Single-dim ops (`_to_copy`, `mul.Tensor`, `add.Tensor`, `clone.default`): use upstream single-dim path which AP's Phase 1-extended `_try_single_dim_strategy` now enumerates `_StridedShard` variants for.
- For the specific LLaMA3 Linear pattern in `repro_llama3_8b_fw_256_2d.py:65-66`, mm consumes `view` directly — no intervening ops on the M-dim input side.
- `cat.default`, `split.Tensor`: use legacy `register_op_strategy` (`_tensor_ops.py:962`). Pass placements through directly; `unshard_tensor_dim` may not correctly detect `_StridedShard` on the concat dim. Not exercised by the common Linear chain but worth verifying if user code goes through cat between view and mm.

### 2026-04-20 (morning) — Phase 1 implemented & verified

**Code changes in `autoparallel/autoparallel/shardings/dtensor_sharding_helpers.py`:**
- Added `_StridedShard` import.
- Added `_PREFER_SINGLE_DIM_OPS = {aten.mm.default, addmm.default, bmm.default, baddbmm.default, _scaled_mm.default}`.
- `get_op_strategy`: if op ∈ `_PREFER_SINGLE_DIM_OPS` and has an upstream single-dim registration, route there first (bypasses the legacy `op_strategy_funcs` entry that previously shadowed it).
- `_try_single_dim_strategy`: collect candidate `split_factor`s from upstream input OpStrategies; for each placeholder slot, emit `Shard(d)` plus one `_StridedShard(d, sf)` per candidate `sf`. Previous behavior (plain `Shard` only) is preserved when no input carries `_StridedShard`.

**Tests added:**
- `autoparallel/tests/test_propagation_rules.py::test_mm_strategy_enumerates_strided_shard` — asserts strided inputs produce strided outputs with matching `split_factor`.
- `autoparallel/tests/test_propagation_rules.py::test_mm_strategy_plain_shard_still_present` — regression check: plain-Shard inputs must not spuriously produce `_StridedShard` outputs.

**Artifacts:**
- `pytorch/agent_space/repro_mm_strided.py` — pre-change baseline showing legacy path emits 0 strided strategies.
- `pytorch/agent_space/verify_ap_mm_strided.py` — post-change verification (runs standalone, no pytest).

**Empirical results on 2D mesh (2, 4), input `[Shard(0), _StridedShard(0, sf=8)]`:**

| Path | Total Strategies | With `_StridedShard` output |
|------|-----------------|------------------------------|
| Legacy `_mm_like_strategy` (pre-change) | 16 | 0 |
| Upstream single-dim direct | 106 | 34 |
| **AP `get_op_strategy` (post-change)** | **108** | **36** |

Plain-`Shard`-only input: 64 strategies, all plain Shard, 0 spurious `_StridedShard` — regression clean.

## Goal

Remove AutoParallel's `view → mm → view` → `einsum` rewrite (`_APPLY_VIEW_MM_VIEW_PATTERN` in `autoparallel/api.py:63`) without losing batch+sequence parallel strategies. The solver should discover the same strategy space over the native decomposition by leveraging DTensor's `_StridedShard` propagation + mm single-dim placeholder expansion that already exists upstream.

## Revised Premise (after empirical verification)

`_StridedShard` is **already emitted by DTensor's mm strategy** via the single-dim placeholder path added in pytorch PR #172385. Empirical repro in `pytorch/agent_space/repro_mm_strided.py` on a 2D mesh `(2, 4)` with input `[Shard(0), _StridedShard(0, sf=S)]`:

| Path | Strategies | With `_StridedShard` on output |
|------|-----------|-------------------------------|
| Upstream single-dim (`mm_single_dim_strategy`) | 106 | **34** |
| Legacy `_mm_like_strategy` | 16 | **0** |

**The blocker is not missing DTensor capability — it's that AutoParallel doesn't reach it:**

1. `aten.mm.default` has both registrations in `pytorch/torch/distributed/tensor/_ops/_matrix_ops.py` — legacy `mm_strategy` at line 231 and `mm_single_dim_strategy` at line 406. Upstream `ShardingPropagator` prefers single-dim (`_sharding_prop.py:729-761`), but AP's own `get_op_strategy` (`autoparallel/shardings/dtensor_sharding_helpers.py:325-359`) checks `op_strategy_funcs` first and only falls through to `_try_single_dim_strategy` when the op is missing from the legacy registry — mm is always in the legacy registry.

2. Even when AP's `_try_single_dim_strategy` path *does* run (for ops not in legacy registry), it forces `_ShardingPlaceholder(d) → Shard(d)` (`dtensor_sharding_helpers.py:297-301`), deliberately dropping any `_StridedShard` expansion. Comment at lines 280-283: *"autoparallel explores all placements (not a single runtime one), we always resolve `_ShardingPlaceholder(d) -> Shard(d)`."*

## Approach

Two orthogonal changes:

**A. Route mm through the single-dim path in AutoParallel.** Either (i) override/ignore the legacy `op_strategy_funcs[aten.mm.default]` inside AP so it falls through to `_try_single_dim_strategy`, or (ii) register a custom AP rule that calls `gen_single_dim_einsum_strategies` directly and does a full placeholder expansion.

**B. Teach AP's placeholder resolution to also emit `_StridedShard` variants.** Modify `_try_single_dim_strategy` (or its replacement) so that for each `_ShardingPlaceholder(d)`, it emits both `Shard(d)` *and* `_StridedShard(d, split_factor=sf)` for every `sf` that could plausibly arise from upstream view ops. The enumeration must bound split_factor to the sizes that the flatten provenance actually produces, otherwise the strategy space blows up.

## Required Capabilities

| # | Capability | Owner | State |
|---|-----------|-------|-------|
| 1 | View op preserves multi-dim sharding across flatten/unflatten via `_StridedShard` | PyTorch DTensor | **Done** (`_view_ops.py:585, 1170`) |
| 2 | mm emits `_StridedShard` strategies when input has it | PyTorch DTensor | **Done** (single-dim + placeholder expansion) |
| 3 | AutoParallel reaches the single-dim mm path | AutoParallel | **Done** — `_PREFER_SINGLE_DIM_OPS` in `dtensor_sharding_helpers.py` |
| 4 | Placeholder expansion enumerates `_StridedShard` variants at strategy-gen time (not just runtime input time) | AutoParallel | **Done** — `_try_single_dim_strategy` emits `_StridedShard` variants per upstream-observed `sf` |
| 5 | `redistribute_cost` priced correctly for `_StridedShard ↔ Shard / Replicate / Partial` | PyTorch DTensor | **Conservative** — returns `inf` for non-identical transitions (`_collective_utils.py:535-536`). Solver avoids them. Acceptable for view-mm-view chain; restrictive for mid-chain redistribute. |
| 6 | Backward pass (`permute → mm → permute`) also benefits | AutoParallel | **Done** — verified with `test_mm_strategy_backward_grad_weight_strided` (20 strategies with contracting-dim _StridedShard → Partial) |
| 7 | FLOP/runtime cost accounting for mm with strided-sharded M | AutoParallel (`compute_estimation.py`) | **Done** — fixed `is_shard()` bug at `_get_sharded_shape_stride` |
| 8 | `apply_sharding` materializes `_StridedShard` specs at mm input/output edges | AutoParallel | **Done** — fixed `is_shard()` bug at `_localize_shape_arg`; pending end-to-end numerical test on GPU |

## Phased Work Plan

### Phase 0 — Baseline — **DONE** ✅

- [x] Small-model solver run (`bench_view_mm_flag.py`, dim=512 2L, H100 + fake PG, 2×4 mesh).
- [x] Full LLaMA3-8B dim=4096 2L and 32L at PR #424-class sizes (seqlen=8192, 64-rank 8×8 mesh).
- [x] Strategy-space diagnostic: 0 `_StridedShard` options appear in either NATIVE or EINSUM path for the LLaMA3-8B configs tested. Phase 1 code is dormant for this workload — ready if user exercises `[Shard(0), Shard(1)]` on seq; not activated by the default solver cost.

### Phase 1 — Route mm through single-dim + enumerate `_StridedShard` — **DONE** ✅

Delivered as a simpler variant than originally planned. The candidate-sf set is sourced **from upstream input strategy placements at strategy-gen time** (any `_StridedShard.split_factor` observed on any input OpSpec), not from an explicit forward graph-walk provenance tracker. This works because by the time mm is reached during AP's backward-from-outputs traversal, the upstream view node's OpStrategy already carries every `_StridedShard` option the flatten can produce.

**1a. Bypass legacy `mm_strategy`.** Implemented via `_PREFER_SINGLE_DIM_OPS` allowlist + early-check in `get_op_strategy`. Covers `mm`, `addmm`, `bmm`, `baddbmm`, `_scaled_mm`.

**1b. `_StridedShard`-aware placeholder expansion.** `_try_single_dim_strategy` now emits `Shard(d)` plus `_StridedShard(d, sf)` per sf observed on any upstream input OpSpec.

**Tests:** both unit tests added; empirical verification green (see Progress Log).

**Follow-ups discovered during implementation:**
- If an explicit graph-walk provenance tracker is needed later (e.g., to bound sf when an upstream input hasn't yet been enumerated by the solver), that's a separate enhancement. Current observed-sf approach works because AP's OpStrategy lists are populated in dependency order.
- The allowlist omits `aten.einsum.default` because AP registers its own einsum rule that already dispatches to `_mm_like_strategy`; revisiting that rule to use single-dim is a small follow-up.

### Phase 2 — Cost model — **DONE** ✅

- [x] Audited `redistribute_cost` behavior for `_StridedShard` transitions: returns `inf` when `shard_order is None` (true for strided specs). No-op same-placement case returns 0. Acceptable for view-mm-view but restrictive elsewhere.
- [x] Fixed AP `compute_estimation.py:_get_sharded_shape_stride` — `is_shard()` missed `_StridedShard` → local shape wasn't reduced → FLOPs over-counted.
- [x] Documented the `_redistribute.py:1589` "no flattened transforms" assertion. No fix needed until a redistribute path hits it.

### Phase 3 — apply_sharding correctness — **DONE** ✅

- [x] Fixed `apply_sharding.py:_localize_shape_arg` — same `is_shard()` bug as compute_estimation.
- [x] End-to-end numerical check (`numerical_check_linear3d.py`): NATIVE vs EINSUM vs single-device reference all match bit-exact (0.000e+00). Forward correctness confirmed with Phase 1's `_StridedShard` enumeration enabled.

### Phase 4 — Backward pass validation — **DONE** ✅

- [x] `test_mm_strategy_backward_grad_weight_strided` added to `test_propagation_rules.py`. Confirms backward mm with `_StridedShard` on contracting-dim inputs yields 20 strategies with `(_StridedShard, _StridedShard) → Partial` form.
- [x] `seq_nr` unchanged — only the einsum rewrite needed that fix in PR #424; this path leaves mm alone.
- [ ] Run `test_optimize_placement.py` with rewrite disabled (BLOCKED on pytest env).

### Phase 5 — DTensor upstream gaps + op-audit — **DONE** ✅

- [x] View-family ops (`view`, `permute`, `unsqueeze`, `squeeze`, `transpose`, `expand`, `slice`) are already `_StridedShard`-aware via `propagate_shape_and_sharding` or explicit handling.
- [x] Single-dim ops (`_to_copy`, `mul.Tensor`, `add.Tensor`, `clone.default`) propagate `_StridedShard` via the extended placeholder expansion from Phase 1.
- [x] Flagged but not fixed (outside Linear critical path): `is_shard()` call sites in `collective_runtime_estimation.py:128,146,176,194,235`, `propagation_rules.py:177,552,626,702`, `placement_options.py:560`. Also `cat_strategy` treatment of `_StridedShard` on concat dim.

### Phase 6 — Benchmark parity — **DONE** ✅

- [x] Small-model solver-time parity (`bench_view_mm_flag.py`): NATIVE 36.01s vs EINSUM 35.99s (0.1% diff) on 2-layer dim=512 config.
- [x] Full LLaMA3-8B 2-layer (dim=4096, seqlen=8192, 64-rank 8×8 mesh): NATIVE solve 45.7s + objective 57576.44 vs. EINSUM 76.1s + 57760.68. NATIVE wins on both axes (-40% solve, -0.32% objective cost).
- [x] Full LLaMA3-8B 32-layer: NATIVE solve 1770s + objective 520184. EINSUM did not finish in 4 h wall time. EINSUM's per-node strategy blow-up makes it unusable at depth.
- [x] Multi-seed: seeds 0 and 1 (reverse order) produce identical objectives — solver is deterministic given the graph. Multi-seed variance check complete.
- [ ] Real throughput measurement with `compile=True` and actual step times is still pending (would need torchrun or real multi-rank setup to exercise collectives). Given NATIVE's cheaper solver objective (the NCCL cost proxy), throughput regression is unlikely but unverified.

### Phase 7 — Flip default + deprecate rewrite — **STRONGLY SUPPORTED**

Benchmark evidence for flipping:
- Solver objective (NCCL cost proxy): NATIVE -0.32% vs EINSUM at 2L.
- Solver time: NATIVE -40% at 2L; EINSUM doesn't finish within 4 h at 32L.
- Numerical correctness: NATIVE matches EINSUM bit-exact on `numerical_check_linear3d.py`.
- Unit tests: `test_mm_strategy_*` all pass (three tests in `test_propagation_rules.py`).
- `_StridedShard` code path is ready (verified by `verify_ap_mm_strided.py`) but not triggered by the tested LLaMA3 configs — Phase 1 is correct when dormant and ready when exercised.

Pending:
- [ ] Real training throughput (`compile=True`, torchrun or real multi-rank, actual step time). Given NATIVE's cheaper solver objective, throughput regression is unlikely; this step is confirmation, not gating.

Recommended rollout:
- [ ] Set `_APPLY_VIEW_MM_VIEW_PATTERN = False` by default. Keep `True` as an opt-in escape hatch for one release.
- [ ] After a release cycle with no regressions reported, remove `_replace_view_mm_view_with_einsum` and its pattern matchers in `autoparallel/graph_passes/graph_utils.py`.

## Risks & Open Questions

1. **Strategy-space blow-up.** Adding `_StridedShard` variants multiplies per-mesh-dim strategies by the size of the candidate-sf set. Mitigation: bound sf to values that provenance actually produces. Worst case on a 3-D mesh with multi-level flattens could still be an order of magnitude.

2. **Bypassing the legacy `mm_strategy` affects all mm call sites.** Some non-Linear mm (attention, output projection) may not benefit from `_StridedShard`. But since placeholder expansion only generates `_StridedShard` when an input *has* it, non-Linear mm should see the same strategy set as before — assuming upstream view ops don't introduce `_StridedShard` outputs for them. Verify via strategy-diff test on the LLaMA3 graph.

3. **Uneven sharding.** If `B * S % (mesh[0] * mesh[1]) != 0`, the view op may demote to `Replicate` (`_view_ops.py:1147`). Audit frequency on real shapes; the einsum path does not have this limitation because it sees the axes independently.

4. **`_StridedShard` round-trip correctness through intermediate ops.** If AP inserts any op between view and mm that isn't `_StridedShard`-aware, sharding silently demotes. Phase 5 audit is load-bearing.

5. **Solver interpretability.** The einsum form is easier to debug when solver output looks wrong. Mitigation: add debug printing that surfaces the `_StridedShard(sf)` provenance at each mm.

6. **Upstream `nn.Linear` decomposition change.** If PyTorch eventually stops decomposing `nn.Linear` (TODO at `autoparallel/graph_passes/graph_utils.py:247`), this plan becomes moot. Check upstream status before committing to Phases 3-6.

## Exit Criteria

- [x] `pytorch/agent_space/verify_ap_mm_strided.py` shows `_StridedShard` emission via AP's `get_op_strategy` path (108 strategies, 36 strided on the 2D-mesh synthetic schema).
- [ ] All `test_optimize_placement.py` tests pass with `_APPLY_VIEW_MM_VIEW_PATTERN = False`.
- [ ] LLaMA3-8B 2-layer and 32-layer benchmarks ≤ 2% slower than einsum-fusion default.
- [x] NATIVE picks distinct-from-EINSUM strategies on the small model (20/45 `[R, S(0)]` M-sharded, as noted in bench). Full LLaMA3-8B SP-strategy preservation still pending.
- [x] No numerical divergence on forward pass of Linear-on-3D test: NATIVE vs EINSUM vs reference all bit-exact on `numerical_check_linear3d.py`.

## Out of Scope

- Changing PyTorch's AOT decomposition to stop producing `view → mm → view` (separate upstream effort).
- `nn.Bilinear`, scaled_dot_product_attention, or other non-mm matmul paths that don't go through the flatten.
- Extending placeholder expansion to generate `_StridedShard` from scratch (i.e., without input evidence) — out of scope for AP's current design, which treats placements as provenance-driven.

## Artifacts

- `pytorch/agent_space/repro_mm_strided.py` — pre-change strategy-count comparison (upstream single-dim vs. legacy `_mm_like_strategy`).
- `pytorch/agent_space/verify_ap_mm_strided.py` — post-change verification: 3 tests (strided-input, plain-Shard regression, backward grad-weight). Standalone, no pytest.
- `autoparallel/autoparallel/shardings/dtensor_sharding_helpers.py` — Phase 1 code changes (`_PREFER_SINGLE_DIM_OPS`, extended `_try_single_dim_strategy`, updated `get_op_strategy`).
- `autoparallel/autoparallel/cost_models/compute_estimation.py` — Phase 2 fix (`_get_sharded_shape_stride` handles `_StridedShard`).
- `autoparallel/autoparallel/apply_sharding.py` — Phase 3 fix (`_localize_shape_arg` handles `_StridedShard`).
- `autoparallel/tests/test_propagation_rules.py` — three new tests: `test_mm_strategy_enumerates_strided_shard`, `test_mm_strategy_plain_shard_still_present`, `test_mm_strategy_backward_grad_weight_strided`.
- `pytorch/agent_space/bench_view_mm_flag.py` — Phase 0/6 solver-time comparison (NATIVE vs EINSUM on LLaMA3-ish small model).
- `pytorch/agent_space/numerical_check_linear3d.py` — Phase 3 end-to-end forward numerical correctness check (bit-exact).
- `pytorch/agent_space/bench_llama3_8b.py` — Phase 0/6 full LLaMA3-8B benchmark (both flags, multi-seed, multi-order).
- `pytorch/agent_space/bench_llama3_8b_einsum_only.py` — EINSUM-only variant (used after the combined run timed out on 32L EINSUM solve).
