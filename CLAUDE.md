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
