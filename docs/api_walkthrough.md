# API Walkthrough

This guide walks through the full lifecycle of using AutoParallel and explains
when to use the simple helper versus the full context-manager API.

## Choose the API

Use:

- `auto_parallel(...)` when you want the shortest path to a parallelized module
- `AutoParallel(...)` when you want to add constraints, inspect logs, or debug
  specific optimizer decisions

## Workflow overview

Most usage follows this sequence:

1. Create or load a model.
2. Define a `DeviceMesh`.
3. Provide example inputs.
4. Tell AutoParallel how inputs and outputs are sharded.
5. Optionally add a parameter memory constraint.
6. Optimize placements.
7. Apply placements and materialize the returned module.
8. Initialize or load weights.
9. Run eager or compiled execution.

## Simple API: `auto_parallel(...)`

The simple API is a wrapper around the full API.

```python
from autoparallel import auto_parallel

parallel_model = auto_parallel(
    model,
    mesh,
    sample_inputs=(sample_x,),
    out_shardings=(Shard(0), Replicate()),
    mp_policy=mp_policy,
    parameter_memory_budget=(None, None),
    dynamic=False,
)
```

### Arguments

#### `model`

The `nn.Module` to parallelize. It may be constructed on the meta device.

#### `mesh`

A `DeviceMesh` that defines the distributed topology.

#### `sample_inputs`

Either:

- a pytree of sample inputs, or
- a callable returning that pytree

Leaf tensors may be:

- ordinary `torch.Tensor`: assumed replicated on all mesh dimensions
- `DTensor`: placements are extracted and used as input constraints

Using DTensor sample inputs is usually the clearest way to express input
sharding.

#### `out_shardings`

A pytree matching the model output structure. Each leaf is a placement tuple.

Examples:

```python
out_shardings=(Shard(0),)
out_shardings=(Shard(0), Replicate())
out_shardings={"logits": (Shard(0),), "loss": (Replicate(),)}
```

#### `mp_policy`

Optional `MixedPrecisionPolicy` used when building the parallelized model.

#### `parameter_memory_budget`

Optional `(low, high)` pair controlling parameter memory constraints. Passing
`(None, None)` uses the current default bounds and is often appropriate for
training workloads.

#### `dynamic`

If `True`, AutoParallel traces with symbolic dimensions so the returned module
accepts variable batch sizes more easily.

## Full API: `AutoParallel(...)`

Use the full API when you want to control the optimization process.

```python
from autoparallel import AutoParallel

with AutoParallel(model, input_fn, mesh, mp_policy=mp_policy) as autop:
    autop.add_input_constraints([(Shard(0), Replicate())])
    autop.add_output_constraints([(Shard(0), Replicate())])
    autop.add_parameter_memory_constraint(low=None, high=None)

    sharding = autop.optimize_placement(verbose=True)
    parallel_model = autop.apply_placement(sharding)
```

## Step-by-step with the full API

### 1. Define `input_fn()`

`input_fn()` is called during tracing and must return global-shaped example
inputs.

```python
def input_fn():
    global_batch = 64
    seq_len = 128
    hidden_dim = 4096
    return torch.randn(global_batch, seq_len, hidden_dim, device="cuda")
```

A common mistake is to return a per-rank local batch here. During tracing,
AutoParallel wants the global logical input.

### 2. Add input constraints

```python
autop.add_input_constraints([(Shard(0), Replicate())])
```

This says the runtime input is batch-sharded across the first mesh dimension and
replicated across the second.

### 3. Add output constraints

```python
autop.add_output_constraints([(Shard(0), Replicate())])
```

For a first experiment, constraining outputs to match inputs is often the most
predictable option.

### 4. Add parameter memory constraints

```python
autop.add_parameter_memory_constraint(low=None, high=None)
```

Without this, the optimizer may choose to replicate parameters if that lowers
communication cost.

### 5. Optimize placements

```python
sharding = autop.optimize_placement(verbose=True)
```

This solves the global optimization problem. With `verbose=True`, the optimizer
log is emitted through Python logging and structured tracing.

### 6. Apply placements

```python
parallel_model = autop.apply_placement(sharding)
```

This builds the parallelized module from the chosen strategy.

## Runtime model lifecycle

After either API returns a parallelized module, the normal sequence is:

```python
parallel_model.to_empty(device="cuda")
# initialize or load weights
out = parallel_model(local_x)
out.sum().backward()
```

A few important details:

- The returned module expects local per-rank inputs.
- If the original model was built on meta, you must initialize or load weights
  after `to_empty`.
- The module can be compiled with `torch.compile` and
  `autoparallel_backend()`.

## Running the script with multiple GPUs

AutoParallel does not launch distributed workers itself. For a real run, launch
your script with the usual PyTorch distributed launcher and make sure the world
size matches the mesh size.

Examples:

```bash
torchrun --nproc_per_node=4 my_script.py
torchrun --nproc_per_node=8 my_script.py
```

Typical mapping:

- mesh `(4,)` -> world size 4
- mesh `(2, 4)` -> world size 8

The fake-process-group examples in `examples/` are useful for quick smoke tests
on one machine, but they are not substitutes for a real multi-process launch.

## Seeing optimizer logs

`optimize_placement(verbose=True)` sends the human-readable optimizer log
through Python logging. If you want to see it in a small standalone script,
configure logging explicitly:

```python
import logging

logging.basicConfig(level=logging.INFO)
```

## Example: simple API with DTensor inputs

```python
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel import auto_parallel


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(1024, 4096, bias=False)
        self.down = nn.Linear(4096, 1024, bias=False)

    def forward(self, x):
        return self.down(torch.relu(self.up(x)))


mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))

with torch.device("meta"):
    model = ToyBlock()

sample_x = DTensor.from_local(
    torch.randn(8, 128, 1024),
    mesh,
    [Shard(0), Replicate()],
)

parallel_model = auto_parallel(
    model,
    mesh,
    sample_inputs=(sample_x,),
    out_shardings=(Shard(0), Replicate()),
    mp_policy=MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    ),
    parameter_memory_budget=(None, None),
)
```

## Example: compile after parallelization

```python
from autoparallel import autoparallel_backend

parallel_model = torch.compile(
    parallel_model,
    backend=autoparallel_backend(),
)
```

Compilation is not required for correctness, but it is the intended path for
more optimized execution.

## When to use optimizer internals

The full API exposes `autop.sharding_optimizer`, which is useful when:

- a chosen placement surprises you
- the optimization is infeasible
- you want to compare two constrained solutions
- you want to inspect redistribution costs for a specific node

Useful methods documented elsewhere include:

- `apply_prefetch_discount(...)`
- `explain_placement(...)`
- `print_costs_for_node(...)`
- `diff_solutions(...)`

See [How AutoParallel Chooses a Strategy](how_autoparallel_chooses_a_strategy.md)
for the optimizer-focused workflow.

## Which API should you start with?

Start with `auto_parallel(...)` if:

- you are validating basic support for a model
- you already know the desired input and output sharding
- you do not need verbose optimizer inspection yet

Start with `AutoParallel(...)` if:

- your first run failed
- you want to control constraints explicitly
- you want to understand or change the chosen plan
- you are doing research or development on optimizer behavior
