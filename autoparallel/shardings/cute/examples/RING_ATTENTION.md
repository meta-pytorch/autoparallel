# Ring Attention with CuTe Layouts

Ring attention distributes causal attention across GPUs by splitting the sequence into chunks,
assigning them to GPUs with a zigzag pattern for load balancing, and rotating KV blocks
through a ring of send/recv operations.

This document shows how the algorithm maps to our CuTe-based sharding abstractions.

## The Problem

Causal (lower-triangular) attention is inherently unbalanced: query chunk 0 attends to
1 key chunk, while query chunk 7 attends to all 8. Naive contiguous sharding gives some
GPUs much more work than others.

## Step 1: Zigzag Assignment via XorStride

The zigzag pattern pairs each GPU with one "light" query chunk and one "heavy" one:

```
8 chunks, 4 GPUs:

GPU 0: chunks {0, 7}    work: 1 + 8 = 9 attention pairs
GPU 1: chunks {1, 6}    work: 2 + 7 = 9 attention pairs
GPU 2: chunks {2, 5}    work: 3 + 6 = 9 attention pairs
GPU 3: chunks {3, 4}    work: 4 + 5 = 9 attention pairs
                                         (perfectly balanced)
```

In CuTe, this is:

```python
Layout((N_gpus, 2), (1, XorStride(2 * N_gpus - 1)))
```

For 4 GPUs: `Layout((4, 2), (1, XorStride(7)))`. GPU `g` at position `s` holds chunk
`g ^ (s * 7)`, which gives `g` (for s=0) and `7 - g` (for s=1).

In our framework, this is a `ShardedLayout` where the sequence dimension has an XorStride
in its mesh stride — something DTensor's `Shard(dim)` cannot represent (contiguous blocks only).

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

The optimizer evaluates this hint candidate alongside standard shardings, accepting it
only if the benefit (load-balanced attention) outweighs the cost (non-standard
redistribution to set up the zigzag).

## Step 2: Ring Rotation of KV Blocks

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

At each step, each GPU computes attention between its local Q and the current KV,
applying the causal mask (skip q < k pairs). After 4 steps, every GPU has seen all
8 key chunks.

### Communication pattern

The shift is a circular left rotation: GPU `g` sends its KV to GPU `(g+1) % N` and
receives from GPU `(g-1) % N`. This is a point-to-point send/recv — each GPU talks
to exactly one neighbor per step.

## How This Maps to Our Abstractions

### Each step is a CuTe layout

The KV assignment at each step is a concrete `ShardedLayout`. Step 0 uses the zigzag
layout. Step 1 is the same zigzag applied to the shifted ring positions. Each is a
valid CuTe layout with specific shape/stride tuples.

### Redistribution between steps via `plan_redistribute`

Since each step is a ShardedLayout, we can compute the transition:

```python
from autoparallel.shardings.cute import plan_redistribute

collective = plan_redistribute(step_0_layout, step_1_layout)
# Currently yields: all_to_all (both sharded, different GPU strides)
```

The full element mapping from `plan_redistribute_detailed` reveals the exact data
movement: each GPU sends to exactly one neighbor and receives from exactly one.

### Two separate concerns

| Concern | What | Representation | Where it lives |
|---------|------|---------------|----------------|
| Initial assignment | Which chunks on which GPU | XorStride in ShardedLayout | Sharding hint |
| Step-to-step rotation | Circular shift of KV | P2P send/recv | Runtime scheduling |

The XorStride (sharding) is the same at every step — it defines the pairing of two
chunks per GPU (chunk `g` paired with chunk `2N-1-g`). The ring rotation (scheduling)
changes which GPU holds which pair. The pairing relationship is always XOR; the position
rotates.

### The XorStride is constant across steps

```
Step 0: GPU g holds {g,       7-g}       = {g,       g ^ 7}
Step 1: GPU g holds {(g-1)%4, 7-(g-1)%4} = {(g-1)%4, ((g-1)%4) ^ 7}
Step 2: GPU g holds {(g-2)%4, 7-(g-2)%4} = {(g-2)%4, ((g-2)%4) ^ 7}
```

The second chunk is always `first_chunk XOR 7`. The ring rotation changes the first
chunk via modular arithmetic `(g-t) % N`, which is outside CuTe's affine stride algebra.

## Current Limitations

### `plan_redistribute` classifies as all_to_all

Our redistribution planner compares per-mesh-dim GPU strides between source and target
layouts. When both are sharded with different strides, it classifies the collective as
`all_to_all`. For ring attention, this is technically correct (data moves between GPUs)
but overly general — the actual pattern is a simple ring shift (P2P send/recv), which
is cheaper than a full all_to_all.

**Future optimization**: use the element mapping from `plan_redistribute_detailed` to
detect the ring structure (each GPU sends to exactly one neighbor) and generate P2P
`isend`/`irecv` pairs instead of a full all_to_all collective.

### Modular arithmetic not expressible as CuTe stride

The ring rotation `(g - t) % N` is modular, not affine. CuTe strides represent affine
functions (`coord * stride`). So while each step's layout is a valid CuTe layout, the
step-to-step transition can't be expressed as a single stride operation. This is why the
rotation is handled as explicit communication (send/recv) rather than a layout
transformation.

## Causal Attention Matrix

For reference, the full causal attention matrix (q attends to k if q >= k):

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

With zigzag, each GPU handles 9 attention pairs (perfectly balanced).
With contiguous sharding, GPU 0 would handle 3 pairs while GPU 3 handles 22.
