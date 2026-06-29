# AutoParallel Documentation

This directory contains newcomer guides, conceptual background, troubleshooting
notes, and deeper explanations of how AutoParallel chooses sharding strategies.

If you're new to the project, use the reading order below.

## Start here

- [Getting Started](getting_started.md)
- [Basic Concepts](basic_concepts.md)
- [API Walkthrough](api_walkthrough.md)

## Troubleshooting and reference

- [Troubleshooting](troubleshooting.md)
- [FAQ](faq.md)

## How AutoParallel works

- [How AutoParallel Chooses a Strategy](how_autoparallel_chooses_a_strategy.md)
- [Adaptive Sharding: Sequence-Parallel vs Column-Parallel](adaptive_sharding.md)
- [Split-Dim Seed Design](split_dim_seed_design.md)

## Advanced usage

- [Using `local_map` for MoE and Custom Communication Patterns](local_map_and_moe.md)
- [Split-Dim Seed Search](split_dim_seed.md)
- [Saving and Loading Optimizer State](save_load.md)
