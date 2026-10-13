# Making AutoParallel Agent-Consumable

## Background: why agents don't replace AutoParallel

Kernel-generation agents (PTX / CuTe / Triton) succeed because the loop is
tight: one kernel, compile + run + profile in seconds, with numeric output
and measured throughput as ground truth. They also don't bypass the stack --
they propose code and still rely on the compiler, assembler, and profiler to
verify it.

Distributed sharding is a different problem class:

- The decision is global, not local. `optimize_sharding.py` solves one ILP
  over thousands of FX nodes under uniqueness, flow, and consistency
  constraints. Annotating ops one at a time cannot hold that joint optimum,
  and a single bad placement causes non-local failures (deadlock, OOM at
  scale, performance cliffs).
- Feedback is expensive. A wrong kernel costs seconds; a wrong sharding plan
  costs a multi-GPU launch with nondeterministic NCCL behavior. A solver plus
  hardware cost model (`cost_models/`) should prune the space before launch,
  not trial-and-error generation.
- Memorized recipes go stale fastest under agents. An agent can recite TP/FSDP
  placements for LLaMA, but agentic coding produces endless architecture
  variants -- exactly where memorized recipes break and a graph-in,
  solver-out backend earns its keep.

The productive shape is hybrid: the agent proposes single-device semantics
(and at most `local_map` regions for MoE or custom communication), and
AutoParallel acts as the verifier and optimizer backend.

## What to add

1. One stateless tool call, not a protocol. The power API today is a stateful
   context manager (`AutoParallel.__enter__` -> `add_input_constraints` ->
   `optimize_placement` -> `apply_placement` in `autoparallel/api.py`).
   `auto_parallel()` is closer but still takes live `nn.Module` and
   `DeviceMesh` objects. An agent needs
   `(model ref + sample shapes + mesh as plain JSON + budget)` ->
   `(parallel model + plan JSON + cost summary)` with no interactive steps.
   `export_json.py` already builds the plan dict -- promote it to a
   first-class return value instead of a log artifact.

2. Cheap `plan_only` mode. Agents need a fast retry loop without CUDA. The
   fake-process-group path (`examples/example_hf.py`) proves planning without
   hardware is possible; expose it as an explicit API returning estimated
   communication, compute, memory, and OOM risk in JSON.

3. Machine-readable errors with repair hints. Current failures are
   human-readable logs: infeasible-ILP `RuntimeError`,
   `_check_forward_args` shape mismatches, Dynamo trace breaks on dynamic
   code. Each needs an error code (`INFEASIBLE_CONSTRAINTS`,
   `TRACE_FAILED`, `MEMORY_OVER_BUDGET`, `UNSUPPORTED_OP`) plus the offending
   node and `module_path` (which `export_json.py` already extracts) and a
   suggested fix: relax constraint X, wrap region Y in `local_map`, or grow
   the mesh.

4. A `supports(model)` preflight. Agent-generated code tends to be dynamic
   and op-inventive, while tracing assumes mostly-static FX-traceable graphs.
   Report traceability and unsupported ops before solving, with pointers to
   regions that should be wrapped in `local_map`, instead of failing with a
   stack trace mid-pipeline.

5. Expose the "why". The optimizer already records per-node costs and
   alternatives (`get_log` / `get_json`), but they are buried in logs and
   `trace_structured` events. Return per-node cost breakdowns, top-k
   alternatives, and the visualizer output
   (`visualizer/build_display_from_json.py`) as artifacts the agent can hand
   back to the user.

6. Built-in `verify_plan()`. A single call running fake-tensor forward and
   backward equivalence plus a cost-vs-baseline comparison, returning
   pass/fail JSON -- so the agent can self-check instead of asking the human
   to run `torchrun`.
