# How AutoParallel Chooses a Strategy

AutoParallel formulates sharding as a cost minimization problem. Each
node in the computation graph has a finite set of valid sharding
strategies. A strategy specifies the expected input placements, the
output placement, and the redistribution costs from each predecessor's
possible output placements. The ILP solver selects one strategy per node
to minimize total cost across the entire graph simultaneously.

This document explains what goes into that cost, how constraints shape
the solution, and how to interpret and debug the optimizer's decisions.

## The three cost components

Every candidate strategy has a cost that is the sum of three components:
compute, communication, and transition.

### Compute cost

The compute cost estimates how long an operation takes on a single GPU
after sharding. In the current cost model, it is the maximum of two
estimates:

- **Compute time**: FLOPs divided by device throughput (currently
  assuming 70% efficiency). For a bf16 matmul `[M, K] × [K, N]` on an
  H100 (989 TFLOPS bf16), the compute time is
  `2·M·K·N / (989e12 × 0.7)` microseconds.

- **Memory time**: total bytes read and written, divided by device memory
  bandwidth (currently assuming 70% efficiency). For the same matmul, the
  memory cost is `(M·K + K·N + M·N) × 2 / (3.35 TB/s × 0.7)`
  microseconds.

The cost is `max(compute_time, memory_time)`, reflecting that a kernel
is limited by whichever resource is the bottleneck. Large matmuls are
compute-bound; small or pointwise ops are memory-bound.

The current cost model applies a 7 μs kernel launch floor to prevent
tiny operations from appearing "free." Without this, the solver would
place hundreds of near-zero-cost pointwise ops in exotic placements with
no penalty.

These are hardware-informed approximations designed to capture the
relative ranking of strategies, not exact kernel runtimes. View
operations (`reshape`, `expand`, `permute`) have zero compute cost —
they don't move data.

### Communication cost

When an operation's expected input placement doesn't match its
producer's output placement, a collective communication is needed to
redistribute the tensor. The communication cost estimates this
redistribution time.

By default, AutoParallel uses the **NCCL cost model**, which models
actual NCCL algorithm selection:

- **Algorithms**: Ring, Tree, CollNet Direct/Chain, NVLS, NVLS Tree
- **Protocols**: LL (low latency), LL128, Simple
- **Topology**: intra-node NVLink bandwidth vs inter-node network
  bandwidth, with per-architecture constants for A100, H100, and B200

The key practical consequence: intra-node communication (NVLink/NVSwitch)
is much cheaper than inter-node communication (network). On an H100
NVSwitch node, 8-way intra-node all-gather bandwidth is ~320 GB/s per
GPU, while inter-node bandwidth might be 25-50 GB/s. This means the
solver strongly prefers strategies that keep heavy communication within
the node.

The specific collectives and when they arise:

| Redistribution | Collective | Example |
|---|---|---|
| `Shard(d)` → `Replicate` | All-gather | Gathering a sharded weight before matmul |
| `Partial` → `Replicate` | All-reduce | Summing partial results |
| `Partial` → `Shard(d)` | Reduce-scatter | Reducing and sharding gradients |
| `Shard(d)` → `Shard(d')` | All-to-all | Switching sharding dimension |
| `Replicate` → `Shard(d)` | No collective (local narrow) | Each rank takes its shard of the replicated tensor |

Non-dim-0 sharding (`Shard(1)`, `Shard(2)`) adds a reshuffling penalty
because the data must be rearranged in memory after the collective.

### Transition cost

A small tie-breaker penalty (currently 1.0) is applied whenever a
producer and consumer disagree on placement. Without this, the solver
might flip placements between adjacent operations when costs are nearly
tied, producing a plan with unnecessary redistributions. The transition
cost encourages placement stability through the graph.

## The prefetch discount

FSDP-style parallelism shards parameters across ranks and all-gathers
them before each matmul. Naively, the solver would see the all-gather
cost and prefer to replicate parameters instead. But in practice, the
all-gather can overlap with the previous layer's compute — it's
effectively free.

`apply_prefetch_discount` models this overlap by scaling down
communication costs for edges that can be prefetched:

- **Forward**: edges where the producer is "parameter-derived" — meaning
  its value comes from a model parameter, such as a weight cast or view
  feeding a matmul. The all-gather for these can run during the previous
  layer's forward compute.
- **Backward**: edges into "terminal-derived" nodes — meaning gradient
  reduction nodes at the end of the backward graph. The reduce-scatter
  can run during the next layer's backward compute.

The default scale is 0.0 (fully overlapped), meaning these collectives
are treated as free. **This discount is not applied automatically** —
you must call it explicitly on the `ShardingOptimizer` before solving:

```python
with AutoParallel(model, input_fn, mesh) as autop:
    # ... add constraints ...
    autop.sharding_optimizer.apply_prefetch_discount(scale=0.0)
    sharding = autop.optimize_placement()
```

Without this call, the solver sees the full all-gather cost and may
prefer replicating parameters over sharding them. With it, the solver
treats FSDP parameter sharding as essentially free communication.

## How constraints shape the solution

The ILP solver minimizes cost subject to constraints. Some are internal
(ensuring a valid sharding plan), and some are user-specified.

### Internal constraints

These are always active:

- **Uniqueness**: each node selects exactly one coherent strategy across
  all of its arguments
- **Consistency**: for a given node, the strategy selected for each
  argument must agree on the same output placement
- **Flow**: producer and consumer strategies are linked through
  redistribution costs. When placements differ, the optimizer accounts
  for the required communication
- **Forward-backward consistency**: matched forward and backward nodes
  (e.g., a matmul and its corresponding gradient matmul) use the same
  placement

### User constraints

These change the solution:

**Input/output constraints** pin placements at the graph boundary — the
model's inputs and outputs. If you specify
`add_input_constraints([(Shard(0), Replicate())])`, the solver must shard
the input's batch dimension on the first mesh axis. The solver optimizes
everything in between.

**Memory constraints** force parameter sharding. Without
`add_parameter_memory_constraint()`, the solver might replicate all
parameters (avoiding communication entirely). With it, the solver must
shard parameters to fit within the memory budget, which triggers
FSDP-style all-gathers before compute. The prefetch discount makes
these all-gathers cheap, so the solver converges on an FSDP-like
strategy naturally.

**Node constraints** (`add_node_constraint`) force specific placements on
individual operations. Use this when you know the right strategy for a
particular layer and want the solver to optimize around it.

**`with_sharding_constraint`** (in the model code itself) forces a
specific intermediate placement. Unlike `add_node_constraint`, this is
embedded in the model and doesn't require access to the
`ShardingOptimizer`.

## Reading the optimizer log

After calling `optimize_placement(verbose=True)`, the log annotates
each node in the graph with its chosen strategy and cost breakdown.

Here's what a typical annotation looks like:

```
mm = torch.ops.aten.mm.default(view, wq_weight)  # placement=S(0)R, RS(1) -> S(0)S(1) cost=[(0.0, 42.3, 0), (0.0, 42.3, 0)]
```

The fields:

- **`placement=S(0)R, RS(1) -> S(0)S(1)`**: the full sharding strategy.
  The part before `->` lists the input placements (one per tensor input),
  and the part after `->` is the output placement. Here, the first input
  has placement `S(0)R` (Shard(0) on DP, Replicate on TP), the second
  input (weight) has `RS(1)` (Replicate on DP, Shard(1) on TP), and
  the output is `S(0)S(1)` (Shard(0) on DP, Shard(1) on TP).

- **`cost=[(comm, compute, transition), ...]`**: one tuple per tensor
  input argument. The `comm` component is specific to that argument
  (the cost to redistribute that input from its producer's placement).
  The `compute` component reflects the node's overall strategy cost,
  repeated in each tuple for reporting convenience. The `transition`
  component is 1 if this input required a placement change, 0 otherwise.
  The total cost of a node is the sum across all its argument tuples.

At the bottom of the log:

```
total_cost: 15234.50
  comm_cost: 3421.20
  compute_cost: 11802.30
  transition_cost: 11.00
```

This is the full objective value summed across all nodes. Compare
`comm_cost` vs `compute_cost` to understand whether the model is
compute-bound or communication-bound at this configuration.

## Debugging: a typical workflow

When the optimizer makes a surprising choice, here's a systematic way
to investigate:

1. **Solve and read the log.** Call `optimize_placement(verbose=True)`
   and scan the annotations. Look at the summary totals — is the
   solution comm-dominated or compute-dominated?

2. **Ask why a placement wasn't chosen.** If you expected a specific
   placement, use `explain_placement` to compare it against the chosen
   one:

   ```python
   autop.sharding_optimizer.explain_placement((Shard(0), Replicate()))
   ```

   This walks the graph and reports, per node, whether the target
   placement is available and how its cost compares to the chosen one.

3. **Inspect a specific node.** If a particular node's strategy looks
   wrong, examine its redistribution cost matrix:

   ```python
   node = autop.gm.graph.find_nodes(
       op="call_function", target=torch.ops.aten.mm.default
   )[0]
   autop.sharding_optimizer.print_costs_for_node(node, arg=0)
   ```

   This prints a matrix of redistribution costs from each possible
   input placement (columns) to each possible output placement (rows).

4. **Test a hypothesis.** Force the placement you think is better and
   re-solve:

   ```python
   solution_a = autop.sharding_optimizer.get_solution()
   autop.sharding_optimizer.add_node_constraint(node, (Shard(0), Shard(1)))
   solution_b = autop.sharding_optimizer.get_solution()
   autop.sharding_optimizer.diff_solutions(solution_a, solution_b)
   ```

   `diff_solutions` shows the cost impact and which nodes shifted.

### `explain_placement` output

```
Node              Shape         Chosen    Ch.Comp  Ch.Comm  Tgt.Comp  Tgt.Comm  Status
mm_1              [2048, 4096]  S(0)S(1)  42.3     0.0      84.6      0.0       CHOSEN CHEAPER
layer_norm        [2048, 4096]  S(0)S(1)  12.1     0.0      N/A       N/A       UNAVAILABLE
```

Status values:
- `CHOSEN CHEAPER`: the solver's choice has lower total cost
- `TARGET CHEAPER`: the target is locally cheaper but wasn't chosen due
  to global interactions (e.g., it forces expensive downstream
  redistributions)
- `UNAVAILABLE`: the target placement isn't a valid option for that
  operation (e.g., can't shard a dimension that's too small)
- `TIE`: costs are equal; other factors (transition penalties, global
  interactions) determined the choice

### `diff_solutions` output

```
Objective: 15234.5 -> 15891.2 (+656.7)
  compute: 11802.3 -> 11802.3 (+0.0)
  comm:    3421.2 -> 4077.9 (+656.7)
  trans:   11.0 -> 11.0 (+0.0)

Placement changes (12 nodes changed, 847 unchanged):
  S(0)R -> S(0)S(1): 8 nodes
  S(0)S(1) -> S(0)R: 4 nodes
```

This shows the cost impact of a constraint change and which nodes
shifted placement. If the objective increased, the forced placement is
more expensive than what the solver chose. If it decreased, you found
an improvement (which may indicate a cost model inaccuracy worth
investigating).

## Common patterns the solver discovers

### FSDP vs full replication

Without memory constraints, the solver replicates all parameters — no
communication needed. With `add_parameter_memory_constraint()`, it must
shard parameters to fit within the budget. Combined with the prefetch
discount (which makes the resulting all-gathers free), this naturally
produces an FSDP-like strategy where parameters are sharded at rest
and gathered before compute.

### Sequence-parallel vs column-parallel

On a 2D mesh (DP, TP), linear layers have two main TP strategies:

- **Column-parallel**: activation replicated across TP, weight sharded
  on output dimension
- **Sequence-parallel**: activation sharded across TP on the sequence
  dimension, weight replicated

The crossover depends on a single ratio: tokens per DP rank (`M`) vs
the layer's output dimension (`N`). When `M > N`, sequence-parallel
reads less data. When `M < N`, column-parallel reads less data.

This produces adaptive strategies: LLaMA3-8B at long sequences uses
sequence-parallel everywhere, while LLaMA3-70B uses a hybrid (seq-par
for attention, col-par for MLP) because the 70B MLP has a larger output
dimension. See [adaptive_sharding.md](adaptive_sharding.md) for the
full analysis.

### Batch-size sensitivity

Small batch sizes make the model compute-bound — communication costs are
small relative to compute, so the solver favors TP (which reduces
per-GPU compute). Large batch sizes make the model communication-bound —
parameter all-gathers become the bottleneck, so the solver shifts toward
pure DP (which avoids TP communication). The crossover depends on model
size, sequence length, and hardware topology.

## Further reading

- [adaptive_sharding.md](adaptive_sharding.md) — detailed analysis of
  the sequence-parallel vs column-parallel tradeoff for LLaMA3
