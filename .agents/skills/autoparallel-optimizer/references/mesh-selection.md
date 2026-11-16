# Choosing a Device Mesh

AutoParallel requires a `DeviceMesh`, but users may provide only the GPU count
and physical topology. When the mesh is unspecified, make a bounded, explicitly
labeled recommendation. Do not explore meshes with more than two dimensions:
the current strategy space makes ILP solve time impractical.

## Gather the inputs

Determine the allocated GPUs and nodes, the GPUs per high-bandwidth domain
(such as one NVSwitch island), whether ranks in a fast domain are contiguous,
the global batch, the dominant shardable model dimensions (hidden, heads, KV
heads, FFN, vocabulary, experts), memory pressure, and whether the model code
names or consumes mesh axes such as `dp`, `tp`, or `ep`.

Visible GPUs establish availability, not the planned allocation. Prefer an
explicit user statement or launch configuration; if neither gives the world
size and node count, ask before planning. Do not assume eight-GPU nodes when
the supplied topology says otherwise.

## Candidate policy

Honor an explicit user mesh. Otherwise consider at most:

1. a 1D baseline `(world_size,)`, usually named `("dp",)`;
2. one topology-aligned 2D candidate `(dp, parallel)` when memory, large
   matmuls, explicit model axes, or inter-node locality give it a concrete
   purpose;
3. one smaller 2D alternative when divisibility, local batch, or memory make
   the preferred factor doubtful.

Keep latency-sensitive TP or EP communication inside the fastest domain:

```python
mesh_shape = (world_size // gpus_per_fast_domain, gpus_per_fast_domain)
mesh_dim_names = ("dp", "tp")
```

Reduce the inner factor when model dimensions or head counts do not divide it,
the model's own TP/EP implementation limits it, or the fast domain is smaller.

When all intended GPUs share one high-bandwidth domain, 2D has no link-locality
advantage; use 1D unless the model needs separate data and tensor/expert axes.
A size-one dimension such as `(1, 8)` still expands the strategy space.

Each mesh is a separate planning run. Do not enumerate every factorization. If
a 2D solve is impractically slow, report it and fall back to the best valid 1D
result rather than adding a dimension.

## Validate and report

Before capture, check that the mesh product equals the allocated world size,
the global batch divides by the DP factor into a useful local batch, the inner
factor divides the dimensions expected to shard, the rank layout maps the inner
dimension to fast links, and the NCCL topology describes the same allocation.
Compare 1D and 2D candidates only with the same topology and estimator
settings, using both ILP cost and post-reordering metrics when budget permits.

Record the mesh source (user or inferred), the allocation source, topology and
rank-layout assumptions, the selected shape and names, the rationale (memory,
divisibility, global batch, effective DP degree, local batch, link locality),
and alternatives considered. An inferred mesh is an optimization hypothesis and
a starting point, not the only correct decomposition.
