# Split-Dim Seed Search

Split-dim seed search builds a placement seed for a full mesh by solving one
one-dimensional sharding problem per mesh dimension. The full mesh optimizer can
then search only the Hamming ball around that seed with the existing PuLP ILP
solver, or solve the LP relaxation of the same restricted problem.

This is intended for mesh-discovery sweeps where full strategy enumeration is
too large, but a small neighborhood around a fabric-aware seed is still useful.

## Usage

```python
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel.mesh_search import build_split_dim_seed
from autoparallel.optimize_sharding import ShardingOptimizer

input_placement = (Shard(0), Replicate(), Replicate())

seed = build_split_dim_seed(
    gm,
    mesh_shape=(16, 8, 4),
    input_placements=input_placement,
    cost_model=topo_config,
    force_grad_reduce_in_higher_precision=True,
    repeated_subgraphs=True,
    one_d_cache=seed_cache,
)

opt = ShardingOptimizer(
    gm,
    mesh,
    force_grad_reduce_in_higher_precision=True,
    repeated_subgraphs=True,
    strategy_seed=seed,
    strategy_radius=2,
)

opt.add_sharded_input_constraint([input_placement])
opt.add_sharded_output_constraint([input_placement])
opt.add_parameter_memory_constraint(0.0, 1.0 / mesh.size())

solution = opt.get_solution()
lp_result = opt.solve_lp_relaxation(extract=True)
```

`strategy_seed` maps FX node names to placement tuples. `strategy_radius`
keeps placements whose Hamming distance from the seed placement is at most that
radius. Nodes not present in the seed keep their full strategy space.

## Topology

When `cost_model` is an `NCCLTopoConfig`, each one-dimensional seed solve uses
the topology of the corresponding full-mesh dimension. This preserves the
original physical node configuration while making the 1D solve see the right
fabric tier for that dimension.

For `cost_model="nccl"`, the topology is detected from the full mesh before the
per-dimension seed solves.

## Solvers

`get_solution()` solves the restricted Hamming space as an ILP.

`solve_lp_relaxation()` solves the LP relaxation of the same restricted problem
and returns the objective, status, variable fractionality, and optionally an
extracted sharding solution when the relaxation is integral.
