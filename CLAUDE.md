# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutoParallel is a PyTorch library for automatic model sharding and parallelization. It analyzes PyTorch models and automatically determines optimal sharding strategies for distributed training.

**WARNING**: This project is highly under development. See README.md for current PyTorch version requirements.

## Core Architecture

The library consists of several key components that work together:

- **api.py**: Main entry point with `AutoParallel` class that orchestrates the sharding process
- **optimize_sharding.py**: Contains `ShardingOptimizer` that uses PuLP (linear programming) to find optimal sharding strategies
- **apply_sharding.py**: Applies computed sharding strategies to PyTorch models using DTensor specs
- **propagation_rules.py**: Defines how tensor sharding propagates through different PyTorch operations
- **compute_estimation.py**: Estimates runtime costs for different sharding strategies
- **export_module.py**: Handles AOT (Ahead-of-Time) compilation and module export

The optimization flow: Model → FX Graph → Sharding Options → Linear Program → Optimal Strategy → Apply Sharding

## Development Commands

### Setup
```bash
# Install in development mode
uv pip install -e .
```

### Linting and Code Quality
```bash
# Install pre-commit hooks
uv pip install pre-commit

# Run all linters and formatters
pre-commit run --all-files
```

The pre-commit setup includes:
- Black (code formatting)
- flake8 (linting)
- isort (import sorting)
- mypy (type checking)

### Running Examples
```bash
# Basic autoparallel example
python examples/example_autoparallel.py

# LLaMA-3 example
python examples/example_llama3.py
```

### Testing
```bash
# Run tests (check for pytest or unittest patterns)
python -m pytest tests/
```

## Key Dependencies

- **torch**: Core PyTorch functionality and distributed tensor support
- **pulp**: Linear programming solver for optimization
- **filecheck**: Testing utilities

## Development Notes

- Requires Python ≥3.10
- Uses PyTorch's FX graph representation for model analysis
- Leverages DTensor for distributed tensor operations
- Uses linear programming (PuLP) to solve sharding optimization problems
- Includes fake tensor mode for shape inference without actual computation

# Commit messages

Don't commit unless the user explicitly asks you to.

When writing a commit message, don't make a bullet list of the individual
changes. Instead, if the PR is large, explain the order to review changes
(e.g., the logical progression), or if it's short just omit the bullet list
entirely.

Disclose that the PR was authored with Claude.

# Coding Style Guidelines

Follow these rules for all code changes in this repository:

- Minimize comments; be concise; code should be self-explanatory and self-documenting.
- Comments should be useful, for example, comments that remind the reader about
  some global context that is non-obvious and can't be inferred locally.
- Don't make trivial (1-2 LOC) helper functions that are only used once unless
  it significantly improves code readability.
- Prefer clear abstractions. State management should be explicit.
  For example, if managing state in a Python class: there should be a clear
  class definition that has all of the members: don't dynamically `setattr`
  a field on an object and then dynamically `getattr` the field on the object.
- Match existing code style and architectural patterns.
- Assume the reader has familiarity with PyTorch. They may not be the expert
  on the code that is being read, but they should have some experience in the
  area.

If uncertain, choose the simpler, more concise implementation.
