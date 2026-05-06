# Getting Started with AutoParallel

AutoParallel automatically chooses sharding strategies for a PyTorch model,
then applies them to produce a distributed module. Instead of manually deciding
where to use FSDP, tensor parallelism, or intermediate redistributions, you
provide a model, a device mesh, and example inputs; AutoParallel traces the
joint forward/backward graph, solves for a low-cost sharding plan, and returns a
parallelized module.

This guide is the best place to start if you are new to the project.

## What AutoParallel is good at today

AutoParallel is currently most useful when all of the following are true:

- You already have a PyTorch `nn.Module` training workload.
- You are comfortable with PyTorch distributed concepts such as `DeviceMesh`
  and DTensor placements.
- You want AutoParallel to choose among parameter sharding, activation sharding,
  and tensor parallel layouts for standard transformer-style computation.
- You are okay using experimental APIs and PyTorch nightly.

If you are trying to understand why the optimizer chose a specific strategy,
read [How AutoParallel Chooses a Strategy](how_autoparallel_chooses_a_strategy.md)
after this guide.

## Requirements

- Python 3.10+
- PyTorch nightly newer than 2.10
- CUDA GPUs for real execution

Install AutoParallel from GitHub:

```bash
pip install git+https://github.com/pytorch-labs/autoparallel.git
```

For local development:

```bash
pip install -e .
```

## The two APIs

AutoParallel exposes two ways to use the library:

- `auto_parallel(...)`: the simpler API; best for first use
- `AutoParallel(...)`: the full context-manager API; use this when you want to
  add constraints, inspect logs, or call optimizer utilities directly

Most newcomers should start with `auto_parallel(...)`, then switch to the full
API when they need more control.

## Your first successful run

The easiest way to get a feel for the project is the HuggingFace example:

```bash
pip install transformers
python examples/example_hf.py --model gpt2 --mesh 8
```

That example uses a fake process group so you can exercise the AutoParallel
pipeline on a single machine without launching a real multi-process job. It is
best thought of as a convenient single-process smoke test for tracing,
optimization, and module construction, not as a real 8-rank training launch. A
successful run ends with:

```text
Forward + backward OK
```

## Minimal example with the simple API

The `auto_parallel(...)` helper takes:

- a model
- a `DeviceMesh`
- sample inputs for tracing
- the desired output sharding

```python
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel import auto_parallel


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w2(torch.relu(self.w1(x)))


mesh = init_device_mesh("cuda", (4,), mesh_dim_names=("dp",))

with torch.device("meta"):
    model = MLP(dim=1024, hidden_dim=4096)

local_batch = 8
seq_len = 128
sample_x = DTensor.from_local(
    torch.randn(local_batch, seq_len, 1024),
    mesh,
    [Shard(0)],
)

parallel_model = auto_parallel(
    model,
    mesh,
    sample_inputs=(sample_x,),
    out_shardings=(Shard(0),),
    mp_policy=MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    ),
    parameter_memory_budget=(None, None),
)

parallel_model.to_empty(device="cuda")
# Initialize or load weights here if the model was constructed on meta.

x = torch.randn(local_batch, seq_len, 1024, device="cuda")
out = parallel_model(x)
out.sum().backward()
```

This example assumes a real 4-rank CUDA setup. In practice, that usually means
launching the script with `torchrun` on a machine with at least 4 visible GPUs.
If you only want a first smoke test on one machine without a real distributed
launch, use `examples/example_hf.py`, which sets up a fake process group for
you.

### What the example is doing

- The `DTensor` sample input tells AutoParallel that the global input is sharded
  on batch dimension 0 across the mesh.
- `out_shardings=(Shard(0),)` asks for the output to stay batch-sharded.
- `parameter_memory_budget=(None, None)` applies the default parameter memory
  constraint through the simple API. This usually matters in training settings,
  because otherwise the optimizer may prefer to replicate parameters.
- `parallel_model.to_empty(device="cuda")` materializes the returned module on
  CUDA. If you built the original model on the meta device, you must then
  initialize or load its parameters before real execution.

## Minimal example with the full API

Use the full `AutoParallel` API when you want explicit control over constraints
or verbose optimizer logs.

```python
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel import AutoParallel


class Block(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w2(torch.relu(self.w1(x)))


mesh = init_device_mesh("cuda", (4,), mesh_dim_names=("dp",))

with torch.device("meta"):
    model = Block(1024, 4096)


def input_fn():
    global_batch = 32
    seq_len = 128
    return torch.randn(global_batch, seq_len, 1024, device="cuda")


mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
)

with AutoParallel(model, input_fn, mesh, mp_policy=mp_policy) as autop:
    autop.add_input_constraints([(Shard(0),)])
    autop.add_output_constraints([(Shard(0),)])
    autop.add_parameter_memory_constraint(low=None, high=None)

    sharding = autop.optimize_placement(verbose=True)
    parallel_model = autop.apply_placement(sharding)

parallel_model.to_empty(device="cuda")
# Initialize or load weights here.
```

The important distinction is that `input_fn()` returns global-shaped tensors,
while the resulting parallel module expects each rank to receive its local shard
at runtime.

This full-API example also assumes a real distributed run. If your mesh is
`(4,)`, you typically launch 4 processes:

```bash
torchrun --nproc_per_node=4 my_script.py
```

For a 2D mesh like `(2, 4)`, the product of the mesh dimensions must match the
world size. On a single node, that would usually mean:

```bash
torchrun --nproc_per_node=8 my_script.py
```

AutoParallel does not launch processes for you; it assumes your distributed job
has already been launched and that the mesh matches the active world size.

## Logging and verbose optimizer output

When you call `optimize_placement(verbose=True)`, AutoParallel emits the
optimizer log through Python logging. If your script has not configured
logging, you may not see much output.

For ad hoc debugging, add:

```python
import logging

logging.basicConfig(level=logging.INFO)
```

Some examples in this repository use `logging.DEBUG` to show even more detail.

## Recommended first workflow

For a first experiment, use this checklist:

1. Start with a small model or one of the examples.
2. Use a 1D mesh first, usually with batch sharding on mesh dim 0.
3. Constrain both inputs and outputs explicitly.
4. Add a parameter memory constraint for training.
5. Run `optimize_placement(verbose=True)` and inspect the log.
6. Only after that, move to a 2D mesh or custom constraints.

## What to read next

- [Basic Concepts](basic_concepts.md): core ideas and terminology
- [API Walkthrough](api_walkthrough.md): end-to-end lifecycle with both APIs
- [Troubleshooting](troubleshooting.md): common errors and what they usually mean
- [FAQ](faq.md): quick answers to common questions
- [How AutoParallel Chooses a Strategy](how_autoparallel_chooses_a_strategy.md):
  deeper explanation of the optimizer
- [Using `local_map` for MoE and Custom Communication Patterns](local_map_and_moe.md):
  advanced integration for dynamic communication patterns
