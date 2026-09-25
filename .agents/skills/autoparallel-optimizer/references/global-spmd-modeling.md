# Global SPMD Modeling for AutoParallel

AutoParallel traces one program over global logical tensors and derives
per-rank execution by assigning placements. The model must therefore express
the intended global computation. Merely replacing a local batch size with
`local_batch * world_size` is insufficient when operations were defined to act
independently inside each process-local batch.

A model can trace, solve, and execute while implementing the wrong semantics.
Treat this audit as a prerequisite to interpreting its sharding plan.

## Audit the model

Look for constructs whose meaning may depend on a process-local batch or rank:

- `torch.randperm`, random sampling, sample dropping, masking, or stochastic
  routing along the batch dimension;
- indexing, gathering, scattering, sorting, or top-k over a locally flattened
  batch;
- reductions or normalization over batch elements, masked-token counts, or
  prototype assignments;
- `torch.distributed` collectives inside model semantics;
- calls to rank, world-size, or process-group APIs and Python branches on them;
- constants such as `batch_size_per_gpu`, local token counts, or local expert
  capacities embedded in tensor reshapes and normalization factors;
- data augmentation, collation, mutation, or other process-local preprocessing
  embedded in `forward()`.

For each occurrence, determine whether the intended operation is global across
all samples or independent within each data-parallel shard. Do not infer this
choice solely from the current implementation: existing distributed code may
encode local behavior only because every process previously ran the model
independently.

## Common adaptations

### Independent per-rank permutation or sample dropping

A local implementation such as `torch.randperm(local_batch)` cannot simply
become `torch.randperm(global_batch)`: that changes independent local sampling
into one permutation that mixes all ranks.

Represent the replica and local-batch structure in the global program, apply
independent permutations along the local-batch axis, gather or scatter along
that axis, and flatten only after preserving the groups. AutoParallel provides
`autoparallel.ops.permutation(..., independent=True)` for independent
per-slice permutations.

Do not hard-code the GPU count casually. Derive the replica dimension from the
declared execution configuration or make it an explicit model/input parameter,
and document whether changing the DP degree changes the intended stochastic
semantics.

### Rank-local indices and masks

Do not flatten indices from every rank into one indistinguishable vector when
they are meaningful only within a local shard. Preserve a leading replica
dimension, for example `[dp, local_count]`, and use `gather`, `take_along_dim`,
or `scatter_add` along the local dimension. Carry the same grouping through
associated weights and counts.

### Global reductions expressed as collectives

Code written for per-process execution may use `dist.all_reduce(local_sum)` to
compute a global normalization. In the global SPMD model, express this as a
normal tensor reduction over all relevant logical batch dimensions. Placement
propagation and lowering should introduce the required communication.

Do not mechanically remove a collective when it represents custom semantics
that cannot be expressed by the global tensor operation. Such a region may need
an explicit `local_map` boundary and an associated cost model.

### Process-local preprocessing

Move data loading, collation, image transforms, and other host/process-local
work outside the traced model when they are not part of the global tensor
computation. Keep the traced `forward()` tensor-oriented and make all values
that affect its result explicit inputs.

## Warning and change policy

When a likely violation is found, report:

```text
warning: SPMD_MODELING_ASSUMPTION
location: <module/function/source line>
current behavior: <what happens independently per process>
global behavior: <what tracing the enlarged logical batch would mean>
proposed adaptation: <global tensor formulation>
semantic risk: <randomness, normalization, routing, loss scaling, etc.>
```

Propose concrete model changes, but do not make a non-trivial adaptation when
the intended global semantics are ambiguous. Ask the user to confirm whether
the operation should be global or independent per DP shard.

## Validate an adaptation

At minimum:

1. Check equivalence at DP size one.
2. Construct a multi-rank reference by running the original local operation on
   each shard under controlled seeds.
3. Run the global formulation on the concatenated or explicitly grouped batch.
4. Split its outputs back into rank-local pieces and compare values and
   gradients with the reference.
5. Repeat for another global batch size to catch embedded local-batch constants.

For stochastic operations, compare the intended distribution or use explicit
random inputs/seeds; exact equality is inappropriate when the two formulations
consume randomness differently.
