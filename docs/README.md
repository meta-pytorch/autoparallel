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

## Advanced usage

- [Using `local_map` for MoE and Custom Communication Patterns](local_map_and_moe.md)
- [Saving and Loading Optimizer State](save_load.md)

## Engineering plans

- [CPU-Only Capture: Goals and Test Plan](cpu_only_capture.md)

## Strategy and positioning

- [AutoParallel as an Agent-Facing Planner and Plan Evaluator](agent_consumable_claude_v5.md) —
  current version; supersedes the drafts below
- [Making AutoParallel Agent-Consumable](agent_consumable.md) (draft)
- [Making AutoParallel Agent-Consumable, expanded](agent_consumable_claude.md) (draft)
- [AutoParallel in an Agent-Driven Development World](agent_consumable_codex.md) (draft)
- [AutoParallel in an Agent-Driven World](agent_consumable_claude_v2.md) (draft)
- [AutoParallel as an Agent-Facing Parallelism Planner](agent_consumable_codex_v2.md) (draft)
- [AutoParallel as an Agent-Facing Planner and Plan Evaluator, v3](agent_consumable_claude_v3.md) (draft)
- [AutoParallel in an Agent-Driven World, v3](agent_consumable_codex_v3.md) (draft)
- [AutoParallel as an Agent-Facing Planner and Plan Evaluator, v4](agent_consumable_claude_v4.md) (draft)
- [AutoParallel as an Agent-Facing Planner and Plan Evaluator (Codex v4)](agent_consumable_codex_v4.md) (draft)

## Agent workflow

- [AutoParallel Optimizer skill](../.agents/skills/autoparallel-optimizer/SKILL.md) —
  repository-scoped instructions for planning, inspecting, and evaluating
  sharding strategies with coding agents
