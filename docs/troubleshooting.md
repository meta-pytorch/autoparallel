# Troubleshooting

AutoParallel is still experimental, so failures are not unusual. This guide
covers the most common failure modes and what they usually mean.

## First debugging step

If you are using the full API, start here:

```python
sharding = autop.optimize_placement(verbose=True)
```

The verbose log is the fastest way to answer:

- what placements were available
- what placements were chosen
- whether communication or compute dominated
- whether constraints made the problem infeasible

The verbose report is emitted through Python logging. If your script has not
configured logging, add:

```python
import logging

logging.basicConfig(level=logging.INFO)
```

## The optimizer cannot find a feasible solution

Typical error:

```text
The sharding optimizer could not find a feasible solution.
```

Usually this means one of the following:

- input and output constraints contradict available strategies
- the device mesh is too small for the requested sharding
- a forced placement is unsupported for some operation in the graph

What to try:

1. Remove output constraints first.
2. Relax custom node constraints.
3. Start with a 1D mesh.
4. Use only batch sharding on the input.
5. Re-run with `verbose=True` and inspect the warning log.

## The optimizer replicated all parameters

Symptom:

- the chosen plan looks mostly replicated
- you expected FSDP-style parameter sharding but do not see it

Most common cause:

- no parameter memory constraint was added

Fix:

Full API:

```python
autop.add_parameter_memory_constraint(low=None, high=None)
```

Simple API:

```python
parameter_memory_budget=(None, None)
```

If you are reading the optimizer docs closely, also note that the prefetch
discount is not applied automatically. For more detail, see
[How AutoParallel Chooses a Strategy](how_autoparallel_chooses_a_strategy.md).

## My first run failed during tracing

Common causes:

- unsupported or not-yet-handled PyTorch operators
- Python-side control flow that does not export cleanly
- inputs whose structure or types differ between tracing and execution
- custom communication performed directly in the model graph

What to try:

- Start from one of the examples and gradually move toward your model.
- Reduce the model to the smallest failing submodule.
- Replace unsupported custom communication with `local_map` if the pattern is
  intentionally manual.
- Make sure `input_fn()` or `sample_inputs` matches the true runtime input
  structure.

## Local/global shape confusion

Symptom:

- shape checks fail at runtime
- the compiled module rejects inputs that look "correct"

Common cause:

- the tracing inputs were treated as local instead of global, or vice versa

Rule of thumb:

- `input_fn()` in the full API returns global logical inputs
- runtime execution uses local per-rank inputs
- `DTensor.from_local(...)` in the simple API still describes a global input
  placement, even though you provide a local shard to build the DTensor

If in doubt, start with a global batch size that divides cleanly across the mesh
and verify the expected local batch manually.

## The runtime model exists, but outputs are garbage or unstable

Possible causes:

- the model was created on meta and never initialized
- `to_empty(device="cuda")` was called, but weights were neither loaded nor
  initialized afterward
- an uninitialized model happened to run numerically

Fix:

After `to_empty`, initialize or load the parameters before real training or
inference.

## Runtime input checking fails

The generated parallel module validates runtime inputs against the traced input
signature. Failures usually mean:

- wrong number of tensor leaves
- wrong dtype
- wrong local shape
- pytree structure changed between tracing and execution

Fixes:

- Keep the exact same argument structure.
- Keep non-tensor arguments in the same positions.
- Check whether the runtime input should be local or global.
- Recreate `sample_inputs` or `input_fn()` from the actual training step.

## `local_map` shape mismatch or placement mismatch

If you are using `local_map`, common causes include:

- returned local tensor shape does not match declared `out_placements`
- `in_placements` entries do not line up with traced input order
- non-tensor arguments are missing `None` entries in `in_placements`

What to check:

- Every output placement matches the actual local tensor shape.
- Every non-tensor argument has `None` in `in_placements`.
- If using a decorator form, inspect the traced graph order if placements seem
  shifted.

For the full MoE/custom communication workflow, see
[Using `local_map` for MoE and Custom Communication Patterns](local_map_and_moe.md).

## Hanging collectives in custom code

Most often this comes from using raw `torch.distributed` collectives inside a
`local_map` region or from inconsistent split sizes in `all_to_all`.

Use `autoparallel.collectives` wrappers instead of raw distributed calls inside
`local_map`, and verify that all split sizes match across ranks. The main
wrappers are `all_gather`, `reduce_scatter`, `all_reduce`, and `all_to_all`;
see [Using `local_map` for MoE and Custom Communication Patterns](local_map_and_moe.md)
for a concrete example.

## Version or environment issues

Typical signs:

- import-time errors
- missing internal APIs
- tracing/export failures that look unrelated to your model

Checklist:

- Use Python 3.10+
- Use a recent enough PyTorch nightly
- Re-test on the exact example scripts in `examples/`

Because AutoParallel depends on internal and evolving PyTorch machinery,
mismatched versions can fail in ways that look like user mistakes.

## How to reduce a bug report

A good bug report usually includes:

- the exact PyTorch version and commit/nightly date
- CUDA and GPU type
- mesh shape and mesh dim names
- whether you used `auto_parallel` or `AutoParallel`
- a minimal model that reproduces the issue
- the exact constraints used
- the verbose optimizer log if optimization succeeded but looked wrong

## Practical debugging sequence

When something is off, this order is usually fastest:

1. Try a smaller model or submodule.
2. Use a 1D mesh.
3. Constrain only the input batch sharding.
4. Add the parameter memory constraint.
5. Run `optimize_placement(verbose=True)`.
6. Only then add output constraints, 2D meshes, or custom placement overrides.
7. Once the script logic is sound, switch from a fake-process-group smoke test
   to a real `torchrun` launch.
