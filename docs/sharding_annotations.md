# Sharding Annotations and Shardy-like Propagation

By default AutoParallel hands the entire sharding decision to the ILP: every
node enumerates every valid placement and the solver picks the global optimum.
That is the right default for a fresh model, but at scale the search space is
large even though the user often already knows the high-level plan — "the
attention and MLP projections are tensor-parallel; the batch is data-parallel".

This page describes how to express that plan as a few **sharding annotations**
and have AutoParallel **propagate** them through the graph the way
[Shardy](https://github.com/openxla/shardy) does, turning the unambiguous part
of the graph into ILP constraints. This shrinks the search space and the solve
time while leaving the genuine cost tradeoffs to the solver. With a typical
tensor-parallel annotation on LLaMA-3 it reaches the *same* objective as the
full ILP on a noticeably smaller problem.

If you are new to the project, start with
[Getting Started](getting_started.md) and
[How AutoParallel Chooses a Strategy](how_autoparallel_chooses_a_strategy.md).

## The annotation API

Annotations are added on the `AutoParallel` context manager, after the input /
output constraints and before `optimize_placement`:

```python
with AutoParallel(model, input_fn, mesh) as autop:
    autop.add_parameter_memory_constraint(low=None, high=None)
    autop.add_input_constraints([(Shard(0), Replicate())])
    autop.add_output_constraints([(Shard(0), Shard(2))])

    # Annotate the tensor-parallel plan. A glob matches the weight in every
    # layer at once. Only the tp axis is pinned; the data axis is left open.
    column_parallel = (None, Shard(0))   # shard the output dim
    row_parallel = (None, Shard(1))      # shard the input dim
    for proj in ["wq", "wk", "wv"]:
        autop.annotate_parameter(f"layers.*.attention.{proj}.weight", column_parallel)
    autop.annotate_parameter("layers.*.attention.wo.weight", row_parallel)
    for proj in ["w1", "w3"]:
        autop.annotate_parameter(f"layers.*.feed_forward.{proj}.weight", column_parallel)
    autop.annotate_parameter("layers.*.feed_forward.w2.weight", row_parallel)

    autop.propagate_annotations()        # propagate + constrain
    sharding = autop.optimize_placement()
```

A placement is a tuple with one entry per mesh dimension. Each entry is a
`Placement` (`Shard(d)`, `Replicate()`, ...) or **`None`** to leave that mesh
axis *open* for propagation / the ILP to decide. Leaving the data axis open is
the common case for weights: you pin the tensor-parallel axis and let the
optimizer choose FSDP vs DDP on the data axis.

The available annotation methods are:

- `annotate_parameter(fqn, placements, priority=1)` — `fqn` is a parameter
  fully-qualified name or a glob pattern (e.g. `"layers.*.attention.wq.weight"`).
- `annotate_input(idx, placements, priority=0)` /
  `annotate_output(idx, placements, priority=0)` — graph input/output by index.
- `annotate_node(node, placements, priority=0)` — an arbitrary FX node.

`priority` controls the order annotations propagate (lower first, matching
Shardy). Activations/IO default to a higher priority than weights so that where
they compete for the same mesh axis (the data axis of a matmul) the
data-parallel sharding wins and the weight is all-gathered, rather than the
activation being resharded.

`propagate_annotations()` returns a `PropagationResult` summarizing the
reduction (`nodes_determined`, `axis_constraints`, `reduction`).

## How propagation works

Propagation mirrors the structure of Shardy's propagation, expressed over
AutoParallel's existing per-node strategy lists (which already encode each op's
sharding rule):

- **Per-mesh-axis.** A placement is propagated one mesh axis at a time. This is
  what lets a weight's tensor-parallel sharding flow through a matmul on the
  `tp` axis while the `dp` axis is resolved independently (data-parallel batch,
  with FSDP all-gathers left to the ILP). It is the analogue of Shardy
  projecting tensor shardings onto per-factor axes.

- **Reshard-free.** Along an edge a consumer is only narrowed to the placements
  it can take *without* a reshard from the producer (zero redistribution cost).
  At a genuine reshard boundary — a necessary collective such as an all-reduce
  or all-gather — no zero-cost option exists, so propagation stops there and the
  ILP decides the collective.

- **To a fixed point.** A worklist re-examines a node's neighbors whenever its
  set of candidate shardings shrinks, until nothing changes.

- **Priority rounds.** Annotations propagate in priority order; later rounds
  cannot override what an earlier round determined.

Once propagation reaches a fixed point, every mesh axis of a node whose sharding
became unambiguous is turned into a per-axis ILP constraint
(`add_node_axis_constraint`), which constrains that one axis and leaves the rest
of the node free.

### What is and isn't pinned

Propagation deliberately only pins genuine **`Shard`** placements — the
tensor-parallel structure the annotations describe, which is invariant in the
optimum. It does *not* pin:

- **`Replicate`** — pinning it would forbid the ILP from instead sharding that
  axis (for example choosing sequence parallelism on the residual stream).
- **`Partial`** — a pending reduction whose collective the ILP places; pinning
  it fixes where the reduction happens and can even be infeasible (a `Partial`
  value cannot be added to a `Replicate` residual without first reducing it).

Both are genuine cost tradeoffs, so leaving them open keeps the optimum
reachable at little cost to the reduction.

Two more correctness rules keep the constraint set feasible and faithful:

- **Parameters are sources only.** A parameter's placement is its *stored*
  sharding, which legitimately differs from the *compute* sharding a consumer
  needs by a reshard (an FSDP all-gather). Propagation never infers a
  parameter's sharding from its consumers, so an open data axis stays free for
  FSDP, and a per-axis parameter constraint still counts toward the memory
  budget on its free axes.
- **Backward pass via the pairing.** The forward/backward consistency
  constraints already tie each gradient to its forward tensor, so the
  parameter/input gradients and output tangents are left for the pairing to
  decide; the rest of the backward graph is constrained normally (and the
  forward annotations are mirrored onto the gradients to drive that).

## How a pin is applied: variable fixing vs constraints

`propagate_annotations(method=...)` (forwarded to
`ShardingOptimizer.add_node_axis_constraint`) controls how each determined axis
is committed to the ILP:

- **`"fix"` (default)** sets the upper bound of the ruled-out decision variables
  to 0, so the solver's presolve drops those columns and the problem actually
  shrinks.
- **`"constraint"`** adds an `== 1` equality row over the matching variables.
  It is removable by name, but on a large mesh adding thousands of rows without
  removing any columns can *slow* the solve.

Variable fixing is strictly better for solve time (and never worse for the
objective), which is why it is the default.

## Solver performance and the LP relaxation

`ShardingOptimizer.solve_lp_relaxation()` solves the continuous relaxation
(binaries relaxed to `[0, 1]`) and reports the objective, solve time, and how
many variables came out fractional. It exposes two facts that matter for
performance:

1. **The relaxation is integral.** On LLaMA-3 (2D and 3D meshes), with and
   without annotations, the LP relaxation comes out with *zero* fractional
   variables and an integrality gap of 0% — its optimum already *is* the integer
   optimum. So `solve_lp_relaxation(extract=True)` returns a valid optimal
   per-node strategy dict (same form as `get_solution`) while skipping
   branch-and-bound, which is several times faster than the MILP solve (e.g. on
   the 16-layer 2D model, ~10s vs ~50s; on a 2M-variable 3D problem, ~45s vs
   ~160s). This is the single biggest available speedup and is exact whenever
   the relaxation is integral (it falls back to `None` when it is not).

2. **Where annotations help the MILP.** Because the relaxation is integral,
   there is little branch-and-bound to cut, so the annotation speedup is
   scale-dependent: on a ~400k-variable problem the MILP overhead is a large
   fraction and pinning the TP structure gives ~1.7–1.8×; on a ~2M-variable
   problem the solve is dominated by the relaxation/model size itself, so the
   speedup shrinks toward ~1× even though the *search space* shrinks more (the
   extra mesh axis gives more axes to pin — e.g. −59% strategy choices on 3D vs
   −36% on 2D). The annotation speedup on the *LP* solve is correspondingly
   modest (~1.1–1.4×). The takeaway: annotations reduce the search space and
   keep the optimum exact, but for raw solve time on this (integral) problem the
   larger lever is solving the relaxation directly.

A separate, orthogonal cost is that building the ILP for a 3-axis mesh is slow:
per-node strategy enumeration grows with the number of mesh axes (it is cubic
for a 3-axis mesh, dominated by the 4D attention tensors), which is independent
of the solve and of annotations.

## Example

`examples/example_llama3_annotated.py` runs the full ILP and the
annotated+propagation path on a LLaMA-3-1B model on a 2D mesh and prints the
comparison: the annotated path reaches the same objective on a search space
reduced by roughly a third, with a correspondingly faster solve.
