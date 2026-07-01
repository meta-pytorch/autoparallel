# Context Parallel Local Map Design

This note explains the AutoParallel changes that make Context Parallel (CP)
attention work through `local_map`.

## Background

`local_map` lets a model mark a region that runs on local tensors while the
surrounding graph still has DTensor placements. AutoParallel traces these
regions as `local_map_hop` boundaries so the optimizer can reason about the
region's input and output placements.

CP attention needs one placement distinction that existing AutoParallel support
did not model:

```text
K/V forward input placement: Replicate on CP
K/V input gradient placement: Partial on CP
```

This differs from MoE Expert Parallel local-map regions, where the backward
input-gradient placements can follow the forward input placements.

## Previous Unsupported Path

The unsupported path was specific to PyTorch's `local_map_hop` autograd path.
Ordinary DTensor `local_map` already supports `in_grad_placements` by calling
`to_local(grad_placements=...)`.

`local_map_hop` did two incompatible things:

1. It rejected any local-map region with `in_grad_placements`.
2. Its generated backward graph interpreted input gradients using the forward
   input placements.

The second behavior is wrong for CP K/V gradients. The backward local-map
outputs are gradients for the forward inputs, so their placements must be the
declared `in_grad_placements` when present.

AutoParallel also had an optimizer-side assertion that rejected
`in_grad_placements` metadata before strategy construction.

## Implementation

The support has three pieces.

### HOP Autograd Shim

`autoparallel.tracing._enable_local_map_in_grad_placements()` temporarily
patches the PyTorch `local_map_hop` autograd implementation while
AutoParallel is tracing or applying placements.

The shim keeps PyTorch's existing forward/backward graph construction, then
rewrites the backward local-map metadata:

```text
backward out_placements = forward in_grad_placements
```

when `in_grad_placements` is provided.

This is scoped to AutoParallel's tracing/apply-placement contexts and restores
the original PyTorch functions when the context exits.

### Placement Option Support

`get_local_map_placement_option()` no longer rejects
`in_grad_placements`. It continues to consume `in_placements` and
`out_placements` from the HOP metadata. For backward local-map HOPs, the shim
has already made `out_placements` reflect `in_grad_placements`.

This keeps CP semantics out of the generic placement-option code.

### CP Helper

`autoparallel.context_parallel` provides placement helpers for attention
regions. For the CP axis:

```text
Q forward: Shard(sequence)
Q grad:    Shard(sequence)

K/V forward: Replicate
K/V grad:    Partial
```

The helpers support 2D, 3D, and 4D meshes with named DP, CP, and TP axes.

## Why This Matches Torchtitan

Torchtitan full-DTensor CP declares the same local-map contract for attention:
Q remains sequence-sharded, K/V are gathered for the forward kernel, and K/V
gradients are partial contributions across CP ranks.

The difference is tracing mode. Torchtitan normally uses ordinary DTensor
`local_map` semantics. AutoParallel uses `local_map_hop` to preserve a graph
boundary for placement analysis, so AutoParallel needs the HOP-specific
metadata fix described above.

## Testing

The CP test is end-to-end. It runs AutoParallel tracing, optimization, and
placement application, executes the resulting graph under `LocalTensorMode`,
and compares both forward outputs and Q/K/V gradients against a non-CP SDPA
reference.

Covered mesh layouts:

```text
("dp_shard", "cp", "tp")
("dp_shard", "cp")
("dp_shard", "tp")
("dp_replicate", "dp_shard", "cp", "tp")
```
