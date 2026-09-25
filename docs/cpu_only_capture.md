# CPU-Only Capture: Goals and Test Plan

AutoParallel currently couples CUDA-target capture to planning, even though the
cost models and ILP solve are CPU work. This document defines what "CPU-only"
should mean, what has to be true for it to be safe, and the tests that establish
it.

It covers two separable pieces of work:

- **Planning without a GPU** — bounded work, but target profiles and
  device-sensitive caches must be explicit for correctness.
- **Capture without a GPU** — larger, and carries a real risk of silently
  producing a different graph.

They should be scoped and sequenced differently.

There is also a useful intermediate product: **capture once on a suitable GPU,
then plan many times on CPU**. That removes a GPU from the iterative planning
loop even if target-faithful, GPU-less capture proves impractical.

## Current decision

Prioritize **GPU capture -> CPU planning**. On the tested PyTorch nightly,
CUDA-target tracing without a visible GPU reaches a native AOTAutograd device
check even for a basic linear model. CPU-only capture is therefore deferred
until that framework boundary has a supported virtualization mechanism; it is
not on the critical path for CPU planning.

## Why this matters

Four concrete payoffs, in rough order of value:

1. **A topology test matrix.** The cost model's job is to distinguish topologies
   — 8 GPUs on NVLink versus 256 across 32 nodes, Hopper versus Blackwell. CI
   runs on `g5.4xlarge` / `g5.12xlarge`, so today that behaviour is exercised
   against one patched architecture. With the device profile as an input, a CPU
   runner can sweep (architecture × node count × mesh shape) in seconds.
2. **No silent cost-model downgrade.** Without a GPU,
   `detect_nccl_topo_config` (`nccl_cost_model.py:1364`) catches the exception
   and returns `None`, which routes planning to the PyTorch default cost model —
   the path whose `all_to_all_cost` ends in `total_time *= 5`. The only signal is
   a `logger.info`. An unrecognized GPU name falls through the same way.
3. **Evaluation without provisioning.** Someone assessing AutoParallel can see
   what plan it picks for their model before allocating hardware.
4. **Cheap automated search.** An agent can vary mesh shapes, memory bounds,
   topology assumptions, and user constraints without reserving a GPU for every
   candidate. This is more important than making the one-time capture itself
   GPU-free.

## Goals

- CI can run the full planning and cost-model test matrix on a CPU runner.
- A **declared set** of models captures on CPU and produces a semantically
  identical graph and plan to a matching GPU reference.
- `analyze` distinguishes "capture completed" from "capture fidelity was
  verified against a matching GPU reference"; success alone cannot prove that
  the captured graph is the graph that will execute.
- Contributors can iterate on the cost model, the ILP, and the graph passes
  without a GPU.

## Non-goals

- Capturing **arbitrary** user models on CPU. User code can query the device in
  ways we cannot intercept.
- Running or numerically verifying a plan on CPU. Verification tiers 2–4 need
  real hardware and stay that way.
- Replacing GPU CI. The GPU jobs remain the reference.

## The dependency layers

Device coupling in the capture path sits at four levels, and they need different
treatment.

| Layer | What it is | Reachable from Python? |
|---|---|---|
| L0 | `init_device_mesh("cuda", …)`, process group construction | Yes — fake PG plus a patched `device_count` |
| L1 | Python device queries in library and model code | Yes — this is what `_CUDA_PATCHES` partially covers |
| L2 | ATen/C++ device-property queries inside meta and fake kernels | **No** — a Python patch does not intercept `at::cuda::getCurrentDeviceProperties()` |
| L3 | Arbitrary user code doing arbitrary device things | No, and out of scope |

L0 and L1 are partially virtualized by test scaffolding, not yet solved for a
CPU-only environment. In particular, `_CUDA_PATCHES` does not patch
`torch.cuda.is_available()` or `torch.cuda.current_device()`. The former changes
the Llama and DSv3 SDPA backend list; the latter is reached by
`_get_device_from_mesh`. Those tests currently run with a real CUDA runtime, so
they do not establish CPU-only capture. L2 is the harder question this plan
exists to answer.

### Python-visible orchestration and planning coupling

Three current call sites are relevant. Two discover a GPU name for a table
lookup; the other discovers a current device index solely to construct the
logical fake-tensor device:

| Site | Call | Failure mode today |
|---|---|---|
| `shardings/placement_options.py:666` | `device_handle.current_device()` | capture setup needs an index only to build a logical device for `move_to_fake` |
| `nccl_cost_model.py:1364` | `torch.cuda.get_device_name(0)` | returns `None`, silently downgrades to the fallback cost model |
| `compute_estimation.py:260` | `torch.get_device_module().get_device_name(device)` | on a CPU target currently raises `AttributeError` because `torch.cpu` has no `get_device_name`; called per node |

CUDA calls in the async-TP implementation are a separate apply/runtime concern.
They should not be counted as capture blockers unless a trace actually reaches
them.

### "No GPU" has three meanings

Test these environments separately:

1. a CUDA-capable PyTorch build with a real GPU matching the target profile;
2. the same kind of build with no visible device, for example with
   `CUDA_VISIBLE_DEVICES` empty;
3. a CPU-only PyTorch build with no CUDA runtime or CUDA kernel registrations.

Passing (2) does not establish (3). A CUDA-enabled wheel can contain operator
schemas, fake implementations, and backend code that a CPU-only build omits.
Conversely, globally making `torch.cuda.is_available()` return true on (3) can
send unrelated framework code into CUDA initialization, so that is not a safe
general virtualization mechanism.

## Explicit target and topology inputs

The fix for Python-level coupling is to promote the information represented by
the conftest patches into supported inputs. Do not collapse two independent
concepts into one implicit "GPU profile":

- `TargetDeviceProfile`: device type, architecture/name, compute capability,
  compute and memory limits, and any target backend policy relevant to graph
  capture.
- `TopologyProfile`: GPUs per host and link/NCCL characteristics. Mesh shape and
  dimension names remain explicit inputs rather than inferred hardware facts.

A composite planning environment can contain both. The distinction matters
because graph capture should normally be reusable across node counts and mesh
shapes, while changing target architecture or attention backend may invalidate
the capture.

Pass these values directly where practical. If deep cost-model call sites make
a scoped ambient value necessary, use a nest-safe context rather than a bare
module global: concurrent analyses for H100 and B200 must not observe each
other's profile.

`AutoParallel.__enter__` does **not** currently restore all ambient state when
it fails. Its exception path closes `self.stack`, but NCCL configuration and
`torch._inductor.config.comprehensive_padding` are restored only by
`__exit__`, which Python does not call after a failed `__enter__`. Both leaks
have been reproduced. Register restoration on the `ExitStack` immediately when
each value is changed, and test failures both before and after the padding
mutation.

The profile replaces device discovery in AutoParallel code; it does not
automatically virtualize arbitrary model code. Model branches such as
`has_cuda_capability()` need either explicit model configuration or a narrowly
scoped capture adapter. A process-wide `torch.cuda` monkeypatch is unsuitable
for a public API.

When a CUDA-target planning query cannot use a real device and has no explicit
profile, fail loudly with a diagnostic such as `TARGET_PROFILE_MISSING`.
Fallback should only happen when the caller requested it explicitly, and the
chosen `cost_path` should be part of the result.

### Profile-sensitive caches

The compute-cost cache is currently unsafe for profile sweeps. Its key contains
the operator, sharded input specs, and non-tensor arguments, but not the target
device or its performance limits. In one reproduced sequence, an H100 estimate
of `198.53 us` was returned unchanged after switching to A100; clearing the
cache produced the correct fresh A100 estimate of `629.30 us`.

Include an immutable target-profile identity in every device-dependent cache
key. Clearing caches on a profile transition is an acceptable defensive
measure, but keying by profile is preferable for concurrent analyses and for
alternating between targets. Audit collective and strategy caches for the same
class of omission, and add alternating-order tests (`H100 -> A100 -> H100` and
the reverse), not just isolated-process tests.

## The equivalence criterion

This is the crux of the whole plan. **A CPU capture that succeeds but produces a
different graph is worse than one that fails**, because you would compute an
optimal plan for a program you will not execute, with no signal.

Two levels of check, both required.

### Graph identity (localizes failures)

After normalization, CPU- and GPU-captured joint graphs must be **structurally
identical**:

- same node sequence, same op targets including overloads
- same edges
- same tensor metadata: shape, dtype, stride, device
- same aliasing, mutation, and symbolic-shape constraints
- same AOTAutograd descriptors — `params_spec`, and the forward/backward node
  pairing that `get_param_and_grad_nodes` and friends rely on
- same input/output pytree specifications and training-versus-inference mode
- same user-provided activation-checkpointing metadata, and identical generated
  AC tags after running the same downstream AC pass
- same `nn_module_stack` / `module_path`, which `export_json.py` and graph
  clustering depend on

Permitted normalizations, and nothing beyond this list without a written
justification:

- `stack_trace` file paths
- absolute paths inside node metadata
- alpha-renaming of FX nodes and symbolic dimensions, provided all uses and
  constraints remain identical

Keep a second strict diff, including node names and printed graph text, as a
useful pinned-nightly diagnostic. It should not be the semantic gate: harmless
FX or symbolic-name allocation differences can occur without changing the
program.

### Plan identity (the product guarantee)

Capture on CPU, capture on GPU, plan both with the same profile and workload.
The canonicalized `placements.json` content must be identical. Byte identity is
also desirable once serialization key order and formatting are deterministic,
but it is not stronger than canonical semantic equality.

This is the check that matters to a user, and it catches anything the graph diff
misses — for instance a strategy-enumeration difference driven by metadata the
structural diff does not cover.

Graph identity and plan identity are both required. Equal plans can occur by
coincidence for different strategy spaces, while equal-looking graphs can still
produce different plans because of omitted metadata or cost inputs.

### Reporting fidelity

`analyze` cannot discover target fidelity from a CPU capture alone. Report one
of four states:

- `verified_hardware`: the semantic graph fingerprint matches an unpatched GPU
  reference whose hardware matches the capture signature;
- `profile_consistent`: it matches another profile-driven capture, but that
  reference did not run on matching hardware;
- `unverified`: capture completed, but no matching reference exists;
- `unsupported`: capture failed or a known-incompatible operation was reached.

Only `verified_hardware` supports the claim that CPU capture is target-faithful.
Compatibility is attached to a capture-signature regime, not merely a model
class: verifying one Llama shape and dtype does not verify every SDPA backend
choice that Llama can trigger.

### A trap in the reference

CI runs on A10G but `conftest` patches Python queries to report an H100. This is
potentially internally inconsistent: Python model code sees capability 9.0,
while an unpatched ATen/C++ query can still see the physical A10G. Comparing a
CPU capture against this reference proves less than it appears to.

The two-way diff needs one of:

- a reference job on hardware whose real identity matches the claimed profile
  (a real H100), or
- both sides claiming the same profile, with the comparison explicitly labeled
  `profile_consistent` rather than hardware-verified. This catches regressions
  in the Python-visible path but cannot establish that no physical A10G state
  leaked through L2.

Do the second for routine CI and the first periodically. State which one a given
result came from.

### Capture signature and invalidation

Backend selection is not a function of GPU name alone. A reusable reference or
capture artifact must record at least:

- exact PyTorch build/commit and AutoParallel revision;
- decomposition-table identity — `build_joint_graph` starts from Inductor's
  version-dependent `select_decomp_table()` and then modifies it;
- target architecture/capability, relevant CUDA/cuDNN build versions, available
  kernels, and enabled SDPA/backend policy;
- training or inference mode, grad mode, autocast and mixed-precision policy;
- input pytree, shapes or dynamic-shape constraints, dtypes, strides,
  non-tensor/control values, mask, dropout, causal/GQA settings, and other
  values that affect backend eligibility;
- graph-pass configuration, including repeated-subgraph handling and AC policy;
- custom-op library/schema versions.

Mesh topology does not normally belong in this signature unless it changes
model control flow during capture. It belongs in the downstream planning key.

The test models also contain hidden mutable capture state:
`ScaledDotProductAttention.backends` is a class-level list initialized only on
the first model construction. An H100-profile capture followed by an
A100-profile capture in the same process can therefore reuse the wrong backend
list. Remove this cache, key it by the target profile, or reset it explicitly in
the harness; otherwise a topology/architecture matrix is order-dependent.

## Phase 0 results

The layered experiment was run with PyTorch
`2.14.0.dev20260615+cu130`. On a CUDA-enabled build with no visible GPU:

- fake tensors on logical device `cuda:0` work;
- fake cuDNN, Flash, and Efficient SDPA fail in native backend checks;
- Math SDPA reaches Dynamo, but AOTAutograd fails with
  `hasPrimaryContext expects a valid device index`;
- a basic linear model fails at the same AOT boundary.

The linear result is decisive for scope: CUDA-target CPU capture is blocked at
a model-independent native AOT boundary on this nightly. Running GPT-2, Llama,
or a larger compatibility matrix would currently duplicate the same failure
rather than reveal model-specific support.

On a real H100, fused cuDNN, Flash, and Efficient attention graphs captured
successfully. Math attention produced a substantially different decomposed
graph, confirming that forcing Math is not a target-faithful workaround. The
class-level Llama backend-cache ordering bug described above was also
reproduced.

A saved optimizer loaded and re-solved successfully with no visible GPU and no
process group. This validates the narrow, fixed-mesh/fixed-cost replay path,
though not topology, architecture, or mesh sweeps. The focused existing test
suite passed (`132 passed`).

A genuinely CPU-only PyTorch wheel was not available on the test host, so that
environment remains uncharacterized. It cannot improve the already-failing
CUDA-build case unless its AOT/fake implementation avoids the native CUDA
boundary, and it may introduce additional missing-schema or registration
failures.

The near-term conclusion is therefore **GPU capture -> CPU planning**. Treat
target-faithful CPU capture as future PyTorch integration work, not as a
prerequisite for removing GPUs from planning and agent search loops.

### Native blockers and future investigation

- SDPA backend selection. Dispatching `scaled_dot_product_attention` runs
  backend choice that consults native device state for the fused backends on
  the tested nightly. Forcing `SDPBackend.MATH` is **not** an acceptable default
  workaround: the H100 experiment produced matmuls and softmax instead of the
  target fused node, changing the strategy space and AC tagging. `mm` is also
  in the AC save list, so tagging does not simply disappear; it moves to
  different nodes and granularity, which is still a different optimization
  problem.
- AOTAutograd's CUDA primary-context check. This is reached by Math SDPA and a
  basic linear graph, so fixing SDPA backend selection alone is insufficient.
- cuDNN attention gating, enabled on H100+ (commit `798b61b`).
- FlexAttention's HOP fake implementation.
- Abstract implementations for the Triton custom ops in
  `examples/native_ds3/moe_ops.py`.
- Fake-mode ownership at the Dynamo/AOT boundary. `build_joint_graph` already
  notes that its user-created `FakeTensorMode` cannot simply be passed into
  `aot_export_joint_with_descriptors`; a CPU-only test must cover this exact
  handoff, not just eager fake execution.
- Any device-capability query inside a decomposition on the capture path.

## Fallback architecture: capture once, plan many

CPU-only planning should not wait for CPU-only capture. Introduce an explicit
boundary after graph capture and before strategy enumeration/costing:

```text
model + inputs + target capture profile
             |
         GPU capture
             v
       planner artifact
             |
   mesh + topology + constraints
             v
       CPU planning / ILP
```

A planner artifact needs the normalized joint graph and the metadata required
to enumerate strategies: tensor metadata, parameter and buffer identities,
forward/backward descriptors used by planning, module paths, AC tags, aliasing,
and symbolic constraints. It should also contain its capture signature and a
semantic graph hash.

The existing `ShardingOptimizer.save()` is useful but sits later in the
pipeline. It serializes a fixed mesh, already-enumerated strategies, and already
computed decision-variable costs. It supports CPU re-solving and changing
constraints for that same setup without a process group, but it cannot drive an
architecture/topology/mesh sweep. That requires either:

- an earlier planner artifact from which `ShardingOptimizer` can be rebuilt for
  a new mesh and profile; or
- a supported re-enumeration and re-costing operation over a loaded graph.

Do not assume that an FX `GraphModule` alone is an executable capture artifact.
`apply_placement` and AOT compilation also use `params_spec`, `buffers_spec`,
input/output pytrees, forward metadata, fake inputs, and other private
`JointWithDescriptors` state. A practical split is:

- a stable, explicit **planner artifact** for CPU analysis;
- a version-locked **executable capture** for application, or a fresh capture
  before applying the chosen plan.

Because the current optimizer serializer uses `torch.load(...,
weights_only=False)`, treat its files as trusted artifacts, not as safe inputs
from arbitrary users. A new portable planner format should be versioned,
validated, and avoid pickled executable objects where possible.

## Test matrix

### Planning (no capture) — must pass on CPU, no exceptions

| Axis | Values |
|---|---|
| Architecture | A100, H100, B200, GB200, plus one unrecognized name |
| Node count | 1, 2, 4, 8, 16, 32 |
| Mesh | 1D, 2D, 3D |
| Cost model | `nccl`, explicit `NCCLTopoConfig`, explicitly requested fallback |

Assertions per cell: planning completes; `cost_path` is reported and matches
expectation; an unrecognized architecture with no explicit target profile
raises `TARGET_PROFILE_MISSING` rather than falling back silently; solved plans
are deterministic across repeated runs and independent of matrix execution
order. Warm-cache results must equal fresh-process results when alternating
target architectures.

This matrix is the main prize. None of it needs capture to be CPU-able — it can
run against stored capture artifacts. Use the artifact matching each target
architecture/backend regime, unless graph invariance across those regimes has
itself been established. Re-costing an H100/cuDNN-attention graph as A100 is not
a valid architecture sweep. Node count, link topology, and compatible mesh
shape can usually vary against one capture.

### Capture — per model in the declared set

| Model | Notable features |
|---|---|
| gpt2 (HF) | simplest path, smoke test |
| llama3 debug | SDPA, RoPE, GQA, AC |
| llama3 + flex attention | future case; the current debug model raises `NotImplementedError` for this path |
| dsv3 debug | MoE and `local_map` |

Test native DSv3 custom operators separately if the debug model does not
actually exercise `examples/native_ds3/moe_ops.py`; merely importing or listing
them as suspects is not coverage.

Checks per model:

1. capture completes on CPU under the profile;
2. graph identity against the GPU reference, per the criterion above;
3. plan identity — canonically identical `placements.json`;
4. repeat with `MixedPrecisionPolicy`, with `dynamic=True`, with the
   inference extraction/application path, and with `repeated_subgraphs=True`
   (clustering keys on `OpStrategy` stringification, so it is worth a separate
   cell).

Also vary the SDPA eligibility inputs—dtype, head dimension, sequence length,
mask/causal mode, dropout, GQA, and grad mode—using representative boundary
cases rather than taking a Cartesian product. Backend equivalence for one
attention shape does not generalize to all attention shapes.

A model that fails (1) is simply not in the declared set. A model that passes (1)
but fails (2) or (3) is a **bug to fix or a model to remove from the set** — it
must never ship as a warning.

## Success criteria

### Near-term: GPU capture, CPU planning

- The planning matrix runs green on a CPU runner in CI, and that job is required.
- Planning no longer depends on `tests/conftest.py` monkeypatching
  `torch.cuda.*`.
- CUDA-target planning without either a real device or an explicit target
  profile fails loudly.
- A GPU-produced planner artifact can be loaded and re-planned on a machine
  with no process group and no visible GPU, including re-enumeration and
  re-costing for mesh, topology, and target-profile changes whose capture
  signatures remain compatible.
- Alternating target profiles cannot reuse compute or communication costs from
  the preceding profile.
- A failed `AutoParallel.__enter__` restores NCCL configuration, Inductor
  padding configuration, and every other ambient value it changed.

### Future: target-faithful CPU capture

- `analyze` reports `verified_hardware`, `profile_consistent`, `unverified`, or
  `unsupported` CPU capture; it never turns successful tracing alone into a
  fidelity claim.
- Every model/configuration in the declared CPU set has passing graph-identity
  and plan-identity tests against a GPU reference.
- The declared set, capture signature, and reference hardware are stated in the
  docs.
- Capture-only compatibility shims, if any remain, are narrowly scoped and
  documented.

## Risks

**Silent graph divergence.** The reason plan identity is a required check and not
a nice-to-have. Mitigated by the two-way diff and by the artifact content hash —
a CPU-captured artifact planned against GPU-captured expectations must trip
`PLAN_GRAPH_MISMATCH`.

**Reference drift.** The GPU reference must be regenerated when PyTorch nightly
moves, or the diff starts failing for reasons unrelated to CPU capture. Tie the
reference to the recorded PyTorch build.

**Stale or underspecified cache keys.** SDPA eligibility depends on input and
runtime settings as well as architecture. Validate the full capture signature
on load and reject rather than reuse an artifact when any graph-affecting field
is unknown or mismatched. The compute-cost cache's missing target identity is a
known instance of this bug class, not merely a hypothetical risk.

**Ambient-state races.** Module globals, global CUDA monkeypatches, and the
class-level SDPA backend cache can leak one target profile into another. Failed
`AutoParallel.__enter__` currently also leaks NCCL and Inductor padding state.
Include nested-context, exception-cleanup, concurrent-analysis, and
randomized-order tests.

**Private-API serialization.** AutoParallel uses private Dynamo, AOTAutograd,
and Inductor APIs. Persisting their internal Python objects across nightly
updates is brittle. Keep the planner artifact schema owned by AutoParallel and
version it separately from any same-process executable capture.

**Overclaiming.** "AutoParallel runs on CPU" will be read as "any model." Publish
the tiered guarantee instead:

- planning: CPU-only always, given a profile;
- capture: CPU for the declared set, with the verification level stated;
- verification tiers 2–4 and calibration: real hardware, unavoidably.

## Sequencing

1. Add explicit target/topology profiles, include the target profile in
   device-dependent cache keys, and make ambient-state cleanup exception-safe.
2. Define the planner-artifact boundary and support strategy re-enumeration and
   re-costing for compatible mesh/profile inputs. Existing optimizer
   serialization can bootstrap fixed-profile CPU replay, but not the full
   matrix.
3. Run the full architecture/topology/mesh planning matrix on a CPU runner.
4. Make GPU capture -> CPU planning the supported product path and remove
   planning's reliance on conftest CUDA patches.
5. Pursue an upstream solution for the native AOT primary-context dependency;
   characterize a true CPU-only PyTorch build when one is available.
6. Only after the native boundary is fixed, build the semantic graph-identity
   and plan-identity harness and establish a declared CPU-capture set. Retain
   only justified, scoped capture shims.
