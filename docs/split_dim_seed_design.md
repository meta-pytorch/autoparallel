# Split-Dim Seed Design

## Goal

Provide a small, reusable mesh-discovery primitive:

- build a fabric-aware seed by solving independent 1D problems;
- restrict the full-mesh strategy space to a Hamming ball around that seed;
- solve that restricted problem with the existing ILP path or its LP relaxation.

This branch intentionally does not include TRW-S, lazy cost builds, approximate
solver changes, or experiment runners.

## Public Surface

`autoparallel.mesh_search.build_split_dim_seed(...)` returns
`{node.name: placement_tuple}` for a target mesh shape.

`ShardingOptimizer(..., strategy_seed=seed, strategy_radius=r)` applies the
Hamming-ball restriction during strategy generation, then builds the normal PuLP
problem.

`ShardingOptimizer.solve_lp_relaxation(...)` solves the continuous relaxation of
the same PuLP problem and reports objective/status diagnostics.

## Topology Handling

`NCCLTopoConfig.mesh_dim_topo_override` lets a one-dimensional seed solve reuse
the `MeshDimTopo` derived from the corresponding full-mesh dimension. The
override is accepted only for 1D dim0 and only when the override rank count
matches the 1D mesh size.

The cache key for 1D seeds includes:

- one-dimensional size and input placement;
- physical NCCL config, including original `num_nodes` and `gpus_per_node`;
- derived per-dimension `MeshDimTopo`.

That keeps same-size dimensions on different fabrics from sharing a seed.

## Strategy Filtering

Strategy generation stores the active seed and current FX node name in
`propagation_rules`. Placement generation keeps strategies whose output
placement is inside the active Hamming radius.

The placement-option cache includes mesh identity and seed information. The
optimizer resets that cache when installing and removing a seed so seeded and
unseeded builds cannot reuse each other's filtered entries.

If a node has no seed entry, its full strategy space is kept. If filtering an
operator strategy would remove every option, the original options are kept so
the build does not fail because of an over-tight seed.

## Solver Behavior

The restricted search still uses the normal optimizer lifecycle:

1. generate placement options;
2. build decision variables and costs;
3. add default and user constraints;
4. solve with PuLP.

The ILP path is `get_solution()`. The LP path relaxes existing binary variables
to continuous variables, solves, reports fractionality, and restores the
variable categories before returning.
