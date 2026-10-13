# Realizability failure triage

Use this procedure when an ILP plan exists but placement application, lowering,
or execution fails. A feasible solve establishes only `ilp_feasible`; retain
the first failed stage and do not turn a replicated diagnostic into the selected
production plan.

## Uneven shards and fixed shapes

When a logical dimension is not divisible by its mesh axis, rank-zero's
ceiling-sized local shard cannot reconstruct the global size by multiplication.
A padded width at a fixed `view` usually means lowering lost logical global
metadata, not that the source view is invalid.

- Inspect every `DTensor.from_local()` reconstruction on the failing path. When
  `tgt_spec.tensor_meta` describes the global tensor, preserve its logical
  `shape` and `stride`, plus the existing shard order.
- Compute balanced shard extents with `q, r = divmod(logical_size, shard_count)`:
  `r` ranks have extent `q + 1` and the other ranks have extent `q`. Do not put
  the entire remainder on one final rank. For 20,005 over 8 ranks, the extents
  are five ranks of 2,501 and three ranks of 2,500.
- Check both extent classes and their rank mapping. Do not assume equal local
  extents.
- If the view is still illegal with correct logical metadata, exclude that
  specific placement during strategy enumeration rather than accepting an
  unrealizable low-cost option.
- Use a replicated lowering only to isolate the sharded path. Closure requires
  `apply_placement()` plus output shape, value, and gradient parity with an
  unsharded reference on the distinct rank-local extents.

## Parameter and buffer aliases

Build identity-based alias groups from the source module with duplicate FQNs
retained before capture loses Python object identity. Every persistent alias
group has one compatible placement decision and is counted once for storage.
Map all captured FQNs through the group's canonical descriptor, then restore
all names to the same runtime object; never fix a lookup by cloning state.

After lowering, check for every alias group:

- object identity, placement, global and local shapes, dtype, and
  `requires_grad`;
- one intended optimizer-group membership and one optimizer-state allocation;
- summed gradients and outputs for tied parameters;
- mutation, update, and checkpoint round-trip semantics for mutable observer or
  normalization buffers.

First preserve the defect as a minimized regression with one canonical FQN and
one alias FQN. Keep a negative assertion that a genuinely missing name still
fails. For mutable state, perform at least two successive updates through
different aliases and verify shared state after each; then repeat the relevant
end-to-end model path against the unpartitioned native reference, comparing
outputs and mutated training state. Merely eliminating the lookup error or
compiling once does not close the defect.

A parameter shape error or missing aliased-buffer lookup is a semantic or
placement-application failure. Keep a qualified native fallback and rerun
AutoParallel under the exact source contract after the invariant is fixed.

## Functionalized input mutations

AOTAutograd represents an input mutation as a graph output, then copies that
result back into the original runtime input outside the visible graph. A plan
can therefore solve and lower while returning a sharded mutation result to a
replicated input, causing the mutation epilogue to fail on rank-local shape or
placement.

Discover each `InputMutationAOTOutput` descriptor and pair it with its
`mutated_input` descriptor. Add a default placement-equality constraint between
the original input and mutation result, or represent and cost the required
output-side redistribution to the input placement. Do not patch `copy_`,
silently change the caller's input contract, or replicate unrelated internal
computation. Reject mutation forms that cannot be paired unambiguously.

Validate the minimized mutation and affected model across the complete compiled
call, including the out-of-graph epilogue. Compare the mutated input and normal
outputs with the unsharded reference, check gradients when differentiable, and
exercise repeated calls plus relevant alias or view mutations. A feasible plan
and successful lowering still establish only `ilp_feasible` and `lowered` until
that boundary executes and passes parity.

## Repeated-subgraph clustering

Clustering is optional ILP compression. An unclustered retry that solves proves
recoverability of the base planner, not equivalence, lowering, or performance.
Begin the retry finding with `clustering role: optional ILP compression` so the
diagnostic interpretation is not left implicit.

Validate a complete repeated-region group before mutating cluster links:
corresponding nodes need compatible targets, input arity, metadata, and option
structure. Build links in temporary state and skip a malformed group atomically
with diagnostics; do not abort planning, partially link it, or disable all
other valid groups.

Under identical inputs, constraints, and cost model, compare the expanded
feasible placement families and objective. Record skipped groups, solve time,
and ILP variable/constraint counts to quantify lost compression. Preserve the
normal downstream gates: placement application, numerical checks, target
execution, and measurement remain separate evidence.
