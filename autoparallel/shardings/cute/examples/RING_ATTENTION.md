# Ring Attention with CuTe Layouts

Ring attention distributes causal attention across GPUs by splitting the sequence into chunks,
assigning them to GPUs with a zigzag pattern for load balancing, and rotating KV blocks
through a ring of send/recv operations.

This document shows how the algorithm maps to our CuTe-based sharding abstractions,
including a complete working example.

## The Problem

Causal (lower-triangular) attention is inherently unbalanced: query chunk 0 attends to
1 key chunk, while query chunk 7 attends to all 8. Naive contiguous sharding gives some
GPUs much more work than others.

## Working Example

The entire ring attention schedule — zigzag assignment AND ring rotation across all steps —
is expressible as a single CuTe layout using two stride types:

```python
from autoparallel.shardings.cute._pycute import Layout, XorStride, ModStride

N_gpus = 4
N_chunks = 2 * N_gpus  # 8 chunks total, 2 per GPU
N_steps = N_gpus       # 4 ring steps

# Full ring attention layout:
# (gpu_bit0, gpu_bit1, step, zigzag_pair) -> chunk index
ring_layout = Layout(
    (2, 2, N_steps, 2),
    (ModStride(1, N_gpus), ModStride(2, N_gpus), ModStride(N_gpus - 1, N_gpus), XorStride(N_chunks - 1))
)

# Verify all steps:
for step in range(N_steps):
    print(f"Step {step}:")
    for b1 in range(2):
        for b0 in range(2):
            gpu = b0 + 2 * b1
            chunks = [ring_layout(b0, b1, step, pair) for pair in range(2)]
            print(f"  GPU {gpu}: chunks {chunks}")
```

Output:
```
Step 0:
  GPU 0: chunks [0, 7]
  GPU 1: chunks [1, 6]
  GPU 2: chunks [2, 5]
  GPU 3: chunks [3, 4]
Step 1:
  GPU 0: chunks [3, 4]
  GPU 1: chunks [0, 7]
  GPU 2: chunks [1, 6]
  GPU 3: chunks [2, 5]
Step 2:
  GPU 0: chunks [2, 5]
  GPU 1: chunks [3, 4]
  GPU 2: chunks [0, 7]
  GPU 3: chunks [1, 6]
Step 3:
  GPU 0: chunks [1, 6]
  GPU 1: chunks [2, 5]
  GPU 2: chunks [3, 4]
  GPU 3: chunks [0, 7]
```

### How the layout works

The layout has 4 modes, each with a different stride type:

```
Layout((2,        2,             4,                    2           ),
       (ModStride(1,4), ModStride(2,4), ModStride(3,4), XorStride(7)))
        └─ gpu bit 0    └─ gpu bit 1    └─ ring step    └─ zigzag pair
```

**Evaluation** (`inner_product` left-to-right):
1. `b0 * ModStride(1,4) + b1 * ModStride(2,4)` → `ModStride(gpu, 4)` (GPU index mod 4)
2. `+ step * ModStride(3,4)` → `ModStride((gpu + 3*step) % 4, 4)` = `ModStride((gpu - step) % 4, 4)` (ring rotation)
3. The ModStride resolves to an integer: `source_gpu = (gpu - step) % 4`
4. `+ pair * XorStride(7)` → `source_gpu ^ (pair * 7)` (zigzag: chunk `source_gpu` or `7 - source_gpu`)

**Key constraint**: XorStride must be the **last** mode. CuTe evaluates left-to-right,
so all ModStride modes must resolve to an integer before the XOR is applied.

### Two stride types, two algebraic roles

| Stride type | Algebra | Role | Operation |
|-------------|---------|------|-----------|
| `ModStride(v, n)` | Z/nZ (modular ring) | Cyclic rotation | `(gpu - step) % N` |
| `XorStride(v)` | GF(2)^n (bitwise XOR) | Reflection / zigzag | `source ^ 7 = 7 - source` |

These are **not interchangeable**: ModStride provides rotation (cyclic shift), XorStride
provides reflection (bit flip). Ring attention needs both — rotation for the ring steps,
reflection for the zigzag pairing.

## Zigzag Assignment (Step 0)

The zigzag pattern pairs each GPU with one "light" query chunk and one "heavy" one:

```
GPU 0: chunks {0, 7}    work: 1 + 8 = 9 attention pairs
GPU 1: chunks {1, 6}    work: 2 + 7 = 9 attention pairs
GPU 2: chunks {2, 5}    work: 3 + 6 = 9 attention pairs
GPU 3: chunks {3, 4}    work: 4 + 5 = 9 attention pairs
                                         (perfectly balanced)
```

In CuTe, step 0 alone is: `Layout((4, 2), (1, XorStride(7)))`. GPU `g` at position `s`
holds chunk `g ^ (s * 7)`, which gives `g` (for s=0) and `7 - g` (for s=1).

This is a `ShardedLayout` where the sequence dimension has an XorStride in its mesh
stride — something DTensor's `Shard(dim)` cannot represent (contiguous blocks only).

### Registering via sharding hints

The zigzag assignment is not discoverable by standard enumeration (which only generates
Replicate and contiguous Shard). It's registered as an op-specific hint:

```python
from autoparallel.shardings.cute import register_sharding_hint

def ring_attention_hint(tensor_shapes, mesh_shape):
    # Build zigzag ShardedLayout for the sequence dim
    ...

register_sharding_hint(
    "aten._scaled_dot_product_flash_attention.default",
    ring_attention_hint
)
```

## Ring Rotation (Steps 1-3)

Each GPU keeps its Q chunks fixed and rotates KV chunks through the ring:

```
Step 0 (local):    GPU 0: Q={0,7} KV={0,7}    GPU 1: Q={1,6} KV={1,6}
                   GPU 2: Q={2,5} KV={2,5}    GPU 3: Q={3,4} KV={3,4}

Step 1 (shift 1):  GPU 0: Q={0,7} KV={3,4}    GPU 1: Q={1,6} KV={0,7}
                   GPU 2: Q={2,5} KV={1,6}    GPU 3: Q={3,4} KV={2,5}

Step 2 (shift 2):  GPU 0: Q={0,7} KV={2,5}    GPU 1: Q={1,6} KV={3,4}
                   GPU 2: Q={2,5} KV={0,7}    GPU 3: Q={3,4} KV={1,6}

Step 3 (shift 3):  GPU 0: Q={0,7} KV={1,6}    GPU 1: Q={1,6} KV={2,5}
                   GPU 2: Q={2,5} KV={3,4}    GPU 3: Q={3,4} KV={0,7}
```

### Communication pattern

The shift is a circular left rotation: GPU `g` sends its KV to GPU `(g+1) % N` and
receives from GPU `(g-1) % N`. This is a point-to-point send/recv — each GPU talks
to exactly one neighbor per step.

## Bit Decomposition

The 8 chunks are addressed by 3 bits. The 4 GPUs use 2 bits (b0, b1). The layout
decomposes the index space accordingly:

```python
from autoparallel.shardings.cute._pycute import logical_divide, composition

# Decompose 8-element sequence into 3-bit address space
seq = Layout(8, 1)
seq_bits = logical_divide(seq, (Layout((2, 2, 2)),))

# Step 0 accessor: (b0, b1) = GPU, pair = zigzag
R_step0 = Layout((2, 2, 2), (1, 2, XorStride(7)))

# Compose to get element-to-device mapping
result = composition(seq_bits, R_step0)
# GPU 0: elements 0, 7  |  GPU 1: elements 1, 6
# GPU 2: elements 2, 5  |  GPU 3: elements 3, 4
```

## Deriving Collectives Between Steps

Each step's KV assignment is a slice of the full ring layout. Since each slice is a
concrete CuTe layout, we can use `plan_redistribute` to derive the collective:

```python
from autoparallel.shardings.cute import plan_redistribute

# Extract step 0 and step 1 layouts (as ShardedLayouts on the sequence dim)
# step0: Layout((4, 2), (1, XorStride(7)))
# step1: Layout((4, 2), (1, XorStride(7))) but on shifted GPU positions

collectives = plan_redistribute(step0_layout, step1_layout)
# Yields: ("ppermute", mesh_dim, {"perm": [(0, 1), (1, 2), (2, 3), (3, 0)], ...})
# Each GPU sends to exactly one neighbor — detected as a collective permutation.
```

The planner detects that the ring shift is a **ppermute** (collective permutation), not
a full all_to_all. Each source device maps to exactly one target device. The permutation
list `[(src, dst), ...]` encodes the ring shift pattern directly.

This corresponds to:
- JAX: `jax.lax.ppermute(x, axis_name, perm=[(0,1), (1,2), (2,3), (3,0)])`
- PyTorch: `torch.distributed._functional_collectives.permute_tensor`

## Causal Attention Matrix

```
         k=0  k=1  k=2  k=3  k=4  k=5  k=6  k=7
   q=0  [ x ]  .    .    .    .    .    .    .
   q=1  [ x ] [x ]  .    .    .    .    .    .
   q=2  [ x ] [x ] [x ]  .    .    .    .    .
   q=3  [ x ] [x ] [x ] [x ]  .    .    .    .
   q=4  [ x ] [x ] [x ] [x ] [x ]  .    .    .
   q=5  [ x ] [x ] [x ] [x ] [x ] [x ]  .    .
   q=6  [ x ] [x ] [x ] [x ] [x ] [x ] [x ]  .
   q=7  [ x ] [x ] [x ] [x ] [x ] [x ] [x ] [x ]
```

Work per GPU per step (count of valid q >= k pairs):

```
         Step 0   Step 1   Step 2   Step 3   Total
GPU 0:     3        2        2        2        9
GPU 1:     3        2        2        2        9
GPU 2:     3        2        2        2        9
GPU 3:     3        2        2        2        9
```

With zigzag, each GPU handles 9 attention pairs (perfectly balanced).
With contiguous sharding, GPU 0 would handle 3 pairs while GPU 3 handles 22.

## Notes

### ppermute detection

`plan_redistribute` detects that ring attention's step-to-step transition is a
**ppermute** (collective permutation), not a full all_to_all. The detection checks
if each source device's data maps to exactly one target device. For ring shifts,
this is always true — GPU `g` sends to GPU `(g+1) % N`.

The collective hierarchy from most specific to most general:
```
no_op → local_reinterpret → all_gather → reduce_scatter → all_reduce → ppermute → all_to_all
```

### XorStride must be last in mixed layouts

When combining ModStride and XorStride in a single layout, the XorStride mode must be
the last mode. CuTe's `inner_product` evaluates left-to-right: ModStride modes must
all resolve to an integer before the XOR is applied. If a ModStride follows an XorStride,
the modular reduction wraps the XOR result incorrectly.
