# CuTe-Based Sharding Propagation

A sharding propagation framework built on CuTe layouts, designed to replace DTensor's strategy enumeration with a more expressive representation and a simpler, recipe-based propagation engine.

## Architecture

```
placement.py          ShardedLayout: CuTe hierarchical layout + mesh_dim_map + partial
       |
propagation.py        5 primitives (Carry, Insert, Remove, Merge, Split) + generic engine
       |
op_registry.py        ATen op name -> propagation function (300+ ops)
       |
strategy.py           enumerate_shardings, enumerate_strategies, sharding hints
       |
redistribute.py       Per-mesh-dim GPU stride classification + collective planning
```

## Core Design Decisions

### 1. CuTe is for representation and redistribution, NOT propagation

CuTe provides three things:
- **Representation**: hierarchical `(shape, stride)` tuples express `(local, mesh)` factorizations per tensor dim
- **Construction**: `logical_divide(Layout(global_shape), local_sizes)` canonically splits each dim into `(local, mesh)`
- **Redistribution**: `composition(right_inverse(tgt._ld()), src._ld())` maps source coordinates to target coordinates for data movement planning

CuTe composition is **not used for propagation**. Propagation is per-dim sharding descriptor inference — closer to type inference than address-function transformation. Using composition for propagation (transpose, broadcast, etc.) was tried and rejected because it flattens the sub-dim structure needed by view operations.

### 2. Every propagation rule is a recipe of 5 primitives

Instead of per-operator propagation functions with custom logic, every operator declares a **dim recipe** — a list of DimSpec objects describing what happens to each output dim:

| Primitive | What it does | Example ops |
|-----------|-------------|-------------|
| **Carry** | Output dim inherits sharding from input dim(s) | transpose, einsum, broadcast |
| **Insert** | New replicate dim appears | unsqueeze |
| **Remove** | Dim disappears; if sharded, mark Partial | reduction, slice |
| **Merge** | Multiple dims become one | view flatten, cat |
| **Split** | One dim becomes multiple | view unflatten |

A generic engine `propagate(recipe, removed, inputs)` applies any recipe. Per-operator effort is minimal — just define the recipe (1-5 lines). The 5 primitive implementations are written and tested once.

```python
# Transpose(0, 1) on 3D tensor:
recipe = [Carry([(0, 1)]), Carry([(0, 0)]), Carry([(0, 2)])]

# Einsum "mk,kn->mn":
recipe = [Carry([(0, 0)]), Carry([(1, 1)])]
removed = [Remove((0, 1), reduce_op="sum")]

# View (4,8,16) -> (32,16):
recipe = [Merge([(0, 0), (0, 1)]), Carry([(0, 2)])]

# Cat([A,B], dim=0):
recipe = [Merge([(0, 0), (1, 0)], cross_tensor=True), Carry([(0, 1), (1, 1)])]
```

### 3. The 3-level hierarchy is necessary

`ShardedLayout` uses a 3-level hierarchical CuTe Layout:

```
Level 1: tensor dims — tuple of level-2 modes
Level 2: sub-dims (from view merge/split/cat history) — tuple of level-3 sub-dims
Level 3: (local, mesh0, mesh1, ...) — flat tuple of scalar sizes
```

Why not just 2-level `(local, mesh)` per dim?

A 2-level layout loses three things for merged dims that view split needs:
1. **Sub-dim grouping boundaries** — where one source dim ends and the next begins
2. **Per-mesh-dim strides** — the actual element-to-device mapping per mesh dim
3. **Mesh boundary positions** — where in the merged dim the mesh splits occur

The 3-level hierarchy IS the view provenance. Extracting it as separate metadata would recreate the same structure in a different form. The `_ld()` method provides a 2-level projection when needed (for redistribution composition).

**`_to_uniform` normalization**: ensures all level-3 elements are scalars. Handles S(0)S(0) nesting (`(1, (4, 2))` flattened to `(1, 4, 2)`) and cat scalar sub-dims (`64` wrapped as `(64, 1)`). Detects whether a mode is level-1 (from logical_divide) or level-2 (from view merge) to avoid collapsing sub-dim structure.

### 4. Sub-dim structure is safe from composition

No propagation rule uses CuTe composition. The sets {operations that could flatten sub-dims} and {operations that encounter sub-dims} are **disjoint**:

- **Slice/gather**: use partial evaluation, reject sharded dims — never encounter multi-sub-dim modes
- **View**: uses `view_groups` + structural sub-dim merge/split, not composition
- **All other rules**: structural mode manipulation (swap, compare, carry) — never transforms internal structure

### 5. Only two things are irreducibly non-CuTe

1. **`mesh_dim_map`**: `{tensor_dim: (mesh_dim_ids...)}` — CuTe's layout algebra can't detect when the same mesh dim is assigned to different tensor dims across inputs (a compatibility constraint, not an address-function property)

2. **`partial`**: `{mesh_dim: reduce_op}` — semantic annotation that values need reduction. CuTe can detect that elements are replicated (`gpu_stride = 0`) but can't distinguish "replicated with correct values" from "replicated with partial sums pending reduction"

### 6. Calling convention matches ATen signatures

Every op registry entry accepts the ATen op's exact arguments, with tensors replaced by ShardedLayouts. Dispatch is `fn(*args, **kwargs)`. For ops with sharding-irrelevant arguments (e.g., `roll`'s `shifts`), lambdas in the registry adapt the signature:

```python
"aten.roll.default": lambda self, shifts, dims=None: propagate_replicate_affected(self, dims or []),
"aten._softmax.default": lambda self, dim, half_to_float: propagate_replicate_affected(self, dim),
```

Core propagation functions keep clean minimal signatures. Adaptation is visible at the registration site.

### 7. Partial linearity is an op-level concern, not an engine concern

`propagate()` unconditionally inherits Partial annotations from inputs. Whether that inheritance is semantically valid depends on the op's arithmetic linearity. `check_partial_linearity(inputs, linearity)` is a standalone pre-check called by op adapters **before** `propagate()`:

| Linearity | Rule | Example ops |
|-----------|------|-------------|
| `None` (non-linear) | Reject any Partial input | relu, gelu, sigmoid, exp |
| `"unary"` | Partial passes through | neg, scalar mul, identity |
| `"additive"` | Partial + Partial → Partial (same reduce_op) | add, sub |
| `"multiplicative"` | Partial × Replicate → Partial; Partial × Partial → reject | mul, div |

`propagate_broadcast` is pure shape/sharding compatibility — no linearity parameter. This keeps the engine generic and linearity visible at the op registration site.

### 8. Redistribution uses per-mesh-dim GPU strides from hier_layout

The GPU stride per mesh dim is the fundamental binary classifier:
- `gs = 0`: data is GPU-independent (Replicate or Partial)
- `gs > 0`: data is GPU-dependent (Shard)

`_get_per_mesh_dim_gpu_stride(sharded, mesh_dim)` extracts the stride for a specific mesh dim from the 3-level `hier_layout`, NOT from `_ld()` (which fuses all mesh dims into one stride and loses the per-mesh-dim decomposition needed for S(0)S(0)).

For each mesh dim, compare source vs target GPU strides:

| src_gs | tgt_gs | Collective |
|--------|--------|------------|
| > 0 | = 0 | all_gather |
| = 0 | > 0 | local reinterpret (no communication) |
| = 0 | = 0 | no_op (or all_reduce if Partial) |
| > 0 | > 0, same | no_op |
| > 0 | > 0, different | all_to_all |

**Mesh dim ordering**: nested `logical_divide` produces innermost-first ordering in the tuple, but `mesh_dim_map` uses outermost-first (construction order). `_get_per_mesh_dim_gpu_stride` reverses the flattened mesh elements to match.

### 9. Strategy enumeration is over standard shardings; non-standard layouts emerge from propagation

`enumerate_shardings(tensor_shape, mesh_shape)` generates standard candidates (Replicate, Shard(d, mesh_dim), S(0)S(0) with both LTR and RTL orderings) via per-mesh-dim Cartesian product.

Non-standard layouts (cat non-contiguous, XorStride zigzag) are **derived**, not enumerated. They emerge from propagation when standard input shardings flow through ops like cat and view. The optimizer doesn't discover them — they appear as intermediate states and can accumulate through pointwise chains.

**Op-specific sharding hints** (`register_sharding_hint`) allow domain experts to inject non-standard candidates for specific ops (e.g., XorStride for ring attention). This is where CuTe's representational power pays off — DTensor can't express these candidates at all.

### 10. Enumeration includes all valid strategies, including suboptimal ones

`enumerate_strategies` produces every (input_shardings, output_sharding) pair where propagation succeeds. This includes strategies like `(Replicate, Shard) → Shard` where the replicate input wastes memory but the computation is correct. The optimizer (ILP solver) decides based on total cost (communication + memory). Propagation's job is correctness, not optimality.

## Representation Details

### ShardedLayout

```python
class ShardedLayout:
    hier_layout: Layout      # 3-level CuTe hierarchical layout
    mesh_dim_map: dict        # {tensor_dim: (mesh_dim_ids...)}
    partial: dict             # {mesh_dim: reduce_op}
```

**Construction**: `ShardedLayout.replicate(shape)`, `ShardedLayout.shard(shape, dim, mesh_size, mesh_dim)`, `ShardedLayout.shard_multi(shape, specs)`.

**Properties**: `global_shape`, `local_sizes`, `_ld()` (2-level projection), `get_placements()`.

### S(0)S(0) — ordered sharding

Same tensor dim sharded by two mesh dims. Application order matters:

```python
# Left-to-right: mesh0=2 first (outermost), mesh1=4 second (innermost)
shard_multi(shape, [(0, 2), (0, 4)])   # gpu stride: mesh0=64, mesh1=16

# Right-to-left: mesh1=4 first (outermost), mesh0=2 second (innermost)
shard_multi(shape, [(0, 4), (0, 2)])   # gpu stride: mesh0=32, mesh1=16
```

Different GPU strides → different element-to-device mappings → redistribution (all_to_all) needed to convert between them. `enumerate_shardings` generates both orderings.

### Cat non-contiguous layouts

Cat on a sharded dim produces a non-contiguous layout. Each device holds chunks from all source tensors, separated by gaps in the global index space:

```
cat([A, B], dim=0), both sharded on dim 0 by 2 GPUs:
GPU 0: [a0, a1, b0, b1] → global positions {0, 1, 4, 5}
GPU 1: [a2, a3, b2, b3] → global positions {2, 3, 6, 7}
```

CuTe represents this with a source-index sub-dim. DTensor can't (Shard = contiguous blocks only). Communication is deferred — pointwise ops propagate through without redistribution.

## Invariant Validation

`_validate_sharded_layout` is called after every `propagate()` invocation:

1. **Shape consistency**: positive global/local sizes, global divisible by local
2. **Mesh dim uniqueness**: no mesh_dim assigned to multiple output tensor dims
3. **Partial/shard exclusivity**: can't be both Partial and sharded on same mesh_dim
4. **mesh_dim_map completeness**: every tensor dim has an entry

Currently raises `AssertionError` (development mode). For production: should return `None` with warning.

## AutoParallel Integration

The optimizer should use `ShardedLayout` natively — not convert to DTensorSpec (lossy). Integration requires:

1. **`optimizer_bridge.py`** (not yet built): wraps `enumerate_strategies()` output for the ILP solver, computes redistribute cost matrix via `plan_redistribute()` + existing cost models
2. **Cost model**: existing `autoparallel/cost_models/` provides compute and communication cost estimation
3. **Graph application**: convert winning ShardedLayouts to DTensorSpec at the final step, or generate collective calls directly from `plan_redistribute()` output

## Operator Coverage

302 ATen ops covered by 19 direct recipe builders + 18 delegating wrappers.

### Primitive usage by direct recipe builders

| Function | Carry | Insert | Remove | Merge | Split | ATen ops |
|----------|:-----:|:------:|:------:|:-----:|:-----:|:--------:|
| `propagate_transpose` | x | | | | | 1 |
| `propagate_permute` | x | | | | | 1 |
| `propagate_unsqueeze` | x | x | | | | 1 |
| `propagate_slice` | x | | x | | | 2 |
| `propagate_gather` | x | | x | | | 1 |
| `propagate_reduction` | x | x | x | | | 24 |
| `propagate_broadcast` | x | | | | | (internal) |
| `propagate_pointwise` | x | | | | | 191 |
| `propagate_view` | x | | | x | x | 4 |
| `propagate_einsum` | x | | x | | | (internal) |
| `propagate_cat` | x | | | x | | 1 |
| `propagate_identity` | x | | | | | 16 |
| `propagate_expand` | x | x | x | | | 2 |
| `propagate_repeat` | x | x | x | | | 1 |
| `propagate_split` | x | x | x | | | 2 |
| `propagate_topk` | x | x | x | | | (via lambda) |
| `propagate_embedding` | x | x | x | | | 1 |
| `propagate_convolution` | x | | x | | | 1 |
| `propagate_replicate_affected` | x | | | | | 30 |

### Delegating wrappers

| Function | Delegates to | ATen ops |
|----------|-------------|:--------:|
| `propagate_mm` | einsum | 1 |
| `propagate_bmm` | einsum | 1 |
| `propagate_addmm` | mm + broadcast | 1 |
| `propagate_baddbmm` | bmm + broadcast | 1 |
| `propagate_dot` | einsum | 1 |
| `propagate_t` | transpose | 1 |
| `propagate_movedim` | permute | (internal) |
| `propagate_squeeze` | view | 5 |
| `propagate_flatten` | view | (internal) |
| `propagate_unflatten` | view | (internal) |
| `propagate_stack` | unsqueeze + cat | 1 |
| `propagate_unbind` | slice | 1 |
| `propagate_sort` | replicate_affected | (via lambda) |
| `propagate_argmax` | replicate_affected | (via lambda) |
| `propagate_layer_norm` | replicate_affected | (via lambda) |
| `propagate_index_select` | gather | 1 |
| `propagate_scatter` | broadcast | 4 |
| `propagate_dropout` | identity | 5 |

### Primitive coverage summary

| Primitive | Used by | Role |
|-----------|:-------:|------|
| **Carry** | 19 functions | Universal — every op uses it to pass dims through |
| **Remove** | 9 functions | Reduction, contraction, slice, expand, repeat |
| **Insert** | 7 functions | Unsqueeze, keepdim, expand, repeat, topk, embedding |
| **Merge** | 2 functions | View flatten (within-tensor), cat (cross-tensor) |
| **Split** | 1 function | View unflatten |

The vast majority of ops (261 of 302) use only Carry with optional Insert/Remove. Merge and Split are specialized for reshape and concatenation.

## Testing

202 tests covering:
- All 5 primitives in isolation and composition
- All operator recipes (transpose, permute, view, einsum, cat, etc.)
- S(0)S(0) LTR/RTL ordering, FSDP+TP end-to-end
- Redistribution planning with per-mesh-dim GPU strides
- Strategy enumeration with 2D meshes
- Partial linearity (additive, multiplicative, non-linear rejection)
- Invariant validation (mesh_dim uniqueness, partial/shard exclusivity)
- Edge cases (view→transpose→view round-trip, partial+sharded, cat incompatible dims)
