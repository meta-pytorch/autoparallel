# FAQ

## Is AutoParallel stable?

No. It is explicitly experimental. Expect missing features, unsupported cases,
and API changes.

## Do I need multiple GPUs to try it?

Not necessarily for the tracing and optimization pipeline. The
`examples/example_hf.py` script uses a fake process group so you can exercise
much of the flow on a single machine.

For real execution, you still need CUDA hardware appropriate for the mesh you
want to run.

## Should I start with `auto_parallel(...)` or `AutoParallel(...)`?

Start with `auto_parallel(...)` if you want the shortest path to a working
parallelized module.

Start with `AutoParallel(...)` if you need:

- explicit input/output constraints
- verbose logs
- direct access to optimizer utilities
- iterative debugging of the chosen plan

## How do I run my script on multiple GPUs?

Use PyTorch's normal distributed launcher, typically `torchrun`, and make sure
the world size matches the product of your mesh dimensions.

Examples:

```bash
torchrun --nproc_per_node=4 my_script.py
torchrun --nproc_per_node=8 my_script.py
```

Typical mapping:

- mesh `(4,)` -> world size 4
- mesh `(2, 4)` -> world size 8

The fake-process-group examples in `examples/` are helpful for smoke tests, but
they are not real multi-process distributed launches.

## What models work best today?

Standard transformer-style models and other graphs built mostly from common
PyTorch ops are the best fit.

Advanced custom communication patterns are possible, but usually require manual
composition through `local_map`.

## What if my model has MoE or custom all-to-all communication?

Use `local_map` for the region where communication depends on runtime data.
AutoParallel can still optimize the surrounding dense parts of the model.

See [Using `local_map` for MoE and Custom Communication Patterns](local_map_and_moe.md).

## What is the difference between a global input and a local runtime input?

During tracing, AutoParallel reasons about the global logical input.

At runtime, each rank passes only its local shard to the returned parallel
module.

This is the most common source of confusion for first-time users.

## Why does the optimizer choose replication when I expected sharding?

Often because no parameter memory constraint was added. Without that, full
replication may be cheapest in the optimizer's cost model.

Use:

- `autop.add_parameter_memory_constraint(low=None, high=None)` in the full API
- `parameter_memory_budget=(None, None)` in the simple API

## Do I need to build the model on the meta device?

No, but it is often convenient for large models.

If you do build on meta, the returned parallel module must later be materialized
with `to_empty(device="cuda")` and then initialized or loaded from a checkpoint.

## Does AutoParallel compile the model automatically?

No. The returned module runs eagerly unless you explicitly compile it:

```python
parallel_model = torch.compile(
    parallel_model,
    backend=autoparallel_backend(),
)
```

## I used `verbose=True`, but I do not see the optimizer log. Why?

The verbose optimizer report is emitted through Python logging. In a standalone
script, configure logging explicitly:

```python
import logging

logging.basicConfig(level=logging.INFO)
```

## What does `sample_inputs` do in the simple API?

It provides example inputs for tracing and also communicates input sharding.

- Plain `Tensor` leaves are treated as replicated.
- `DTensor` leaves carry placement information.

## What does `out_shardings` mean?

It specifies the required sharding at the model output boundary. The structure
must match the model output structure.

## Why is PyTorch nightly required?

AutoParallel relies on evolving PyTorch export, fake tensor, AOTAutograd,
Inductor, and DTensor functionality. Stable releases may not yet contain the
APIs or behavior the project expects.

## Is this for training only, or inference too?

The project is primarily oriented around training-oriented graph analysis and
sharding decisions, but the generated module also has a forward-only path for
inference. In practice, training is the more developed mental model for the
current docs and examples.

## Where should I go after the FAQ?

- [Getting Started](getting_started.md)
- [Basic Concepts](basic_concepts.md)
- [API Walkthrough](api_walkthrough.md)
- [Troubleshooting](troubleshooting.md)
- [How AutoParallel Chooses a Strategy](how_autoparallel_chooses_a_strategy.md)
