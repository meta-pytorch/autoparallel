# Choosing a Device Mesh

AutoParallel requires a `DeviceMesh`, but users may provide only the total GPU
count and physical topology. When the mesh is unspecified, make a bounded,
explicitly labeled recommendation. Do not explore meshes with more than two
dimensions: the current strategy space makes ILP solve time impractical.

## Gather the inputs

Determine:

- total allocated GPUs and nodes;
- GPUs in each high-bandwidth domain, such as one NVLink/NVSwitch island;
- rank ordering and whether ranks belonging to a fast domain are contiguous;
- global batch size and the smallest acceptable batch per data-parallel group;
- dominant hidden, attention-head, KV-head, FFN, vocabulary, or expert
  dimensions that may be sharded;
- parameter and activation memory pressure;
- whether model code names or consumes mesh axes such as `dp`, `tp`, or `ep`.

Distinguish GPUs visible to the current shell from GPUs intended for the run.
`nvidia-smi`, CUDA device count, and host inventory establish availability, not
the user's planned allocation. Prefer an explicit user statement or launch
configuration. If neither supplies world size and node count, ask before
running a plan; do not silently use every visible GPU.

Do not infer a standard eight-GPU node when the supplied topology says
otherwise. When less consequential topology details are unavailable, state the
assumption or ask when they materially change the choice.

## Candidate policy

Honor an explicit user mesh. Otherwise consider only a small set:

1. A 1D baseline `(world_size,)`. Use the semantic dimension name expected by
   the model; `("dp",)` is the common data/FSDP baseline.
2. When there is a concrete memory, compute-parallel, semantic-axis, or
   inter-node locality benefit, one preferred 2D candidate `(dp, parallel)`,
   where `dp * parallel == world_size`. Usually the inner `parallel` dimension
   is TP and is named `tp`; for an MoE integration with explicit expert
   semantics it may instead be `ep`.
3. At most one smaller 2D alternative when divisibility, local batch size, or
   memory makes the preferred candidate doubtful.

Do not add a 2D run merely because a factorization exists, and do not enumerate
every factorization. Each mesh changes the strategy space and requires a
separate planning run.

## Choose the preferred 2D factor

Keep latency-sensitive TP or EP communication inside the fastest physical
domain. With `num_nodes` nodes and `gpus_per_fast_domain` contiguous ranks per
node, the usual first candidate is:

```python
mesh_shape = (world_size // gpus_per_fast_domain, gpus_per_fast_domain)
mesh_dim_names = ("dp", "tp")
```

For a standard full-node allocation, this is commonly
`(num_nodes, gpus_per_node)`: TP stays intra-node and DP/FSDP spans nodes.
Verify the actual rank layout before relying on this interpretation.

Reduce the inner factor to a divisor of `world_size` when:

- dominant model dimensions or attention-head counts are incompatible with the
  larger factor;
- the model's explicit TP/EP implementation limits the supported degree;
- a different high-bandwidth island size is reported by the topology.

Prefer powers of two when several factors are otherwise equivalent, but do not
reject a supported non-power-of-two topology merely for convention.

## 1D versus 2D

Prefer the 1D candidate when the model fits comfortably, simultaneous data and
model parallel axes are not important, or fast solver turnaround matters most.

Prefer the topology-aligned 2D candidate when model memory or large matmuls make
tensor/expert sharding valuable, while retaining an outer data/FSDP axis. Also
prefer it when a 1D mesh would make frequent latency-sensitive collectives span
slow inter-node links.

When all intended GPUs are in one high-bandwidth domain, 2D has no link-locality
advantage over 1D. If the model fits and has no explicit need for separate data
and tensor/expert axes, select 1D and stop. A 2D shape containing a size-one
dimension, such as `(1, 8)`, can express those roles but expands the placement
representation and strategy space; use it only when the distinction has a
specific benefit.

## Validate a proposed mesh

Before capture, check:

- the product of mesh dimensions equals the allocated world size;
- global batch and explicitly batch-sharded inputs are compatible with the DP
  factor;
- the resulting local batch per DP group is valid and large enough for the
  model and representative kernels;
- important tensor dimensions and head/expert counts support the proposed inner
  factor where those strategies are expected;
- the global-SPMD model audit uses the same DP degree;
- the rank layout maps the inner dimension to the intended fast links;
- the NCCL topology profile describes the same allocation.

Then run AutoParallel independently for the chosen mesh. If budget permits,
compare the 1D baseline and preferred 2D candidate using both ILP cost and
post-reordering forward/backward metrics. Do not compare costs produced with
different topology or runtime-estimator settings.

If a 2D solve becomes impractically slow, report it and fall back to the best
valid 1D result. Do not respond by adding a third mesh dimension; changing the
sizes of two existing dimensions does not reliably reduce the strategy space.

## Report the decision

Record:

```text
source: user | inferred
intended world_size and node count
allocation source: user | launcher/config
topology: nodes, GPUs per fast domain, rank-layout assumption
selected mesh: shape and dimension names
rationale: memory, model divisibility, global batch, effective DP degree,
  local batch per DP group, and link locality
alternatives evaluated or rejected
```

An inferred mesh is an optimization hypothesis. Present it as the recommended
starting point, not as the only correct decomposition.
