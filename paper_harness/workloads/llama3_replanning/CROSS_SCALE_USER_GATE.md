# Cross-scale placement replay user gate

The current campaign intentionally contains only the 4x8 sequence-length and
local-batch sweeps. Replaying the canonical 4x8 placement on 2x8 or 8x8 is
blocked on an AutoParallel API decision and is not implemented by the harness.

## Evidence

`autoparallel/serialization.py` saves both `mesh_shape` and `mesh_dim_names` in
placement format version 1 (`save_placements`, lines 403-424). Its
`load_placements` implementation rejects any shape mismatch before it matches
saved node placements to current strategies (lines 427-488). The public
documentation likewise says that the graph and mesh must match
(`docs/save_load.md`, lines 99-102).

TorchTitan's official GraphTrainer LLaMA integration calls this API directly
when `compile.autoparallel_placements_load_path` is set
(`torchtitan/experiments/graph_trainer/llama3/parallelize_autoparallel.py`,
lines 258-265). There is no mesh-remapping layer in that path. A canonical
placement saved on `[4, 8]` therefore fails on `[2, 8]` and `[8, 8]` by design.

## Proposed API for review (not implemented)

Add an opt-in keyword to the existing API:

```python
load_placements(path, *, allow_mesh_shape_change: bool = False)
```

With the default `False`, behavior and placement format v1 remain unchanged.
With `True`, loading would still require identical mesh dimension names and
order, then match every saved output and input placement string against a
strategy available on the current mesh. It would not rewrite placements or
silently substitute a strategy. Any missing node, missing strategy, changed
axis name/order, or non-divisible shard would remain a hard error. The return
type remains the existing `dict[Node, OpSpec]`.

Before enabling this in a paper experiment, the API needs unit coverage for
accepted DP-size changes and fail-closed cases, plus an end-to-end 4x8-to-2x8
and 4x8-to-8x8 correctness gate on the real LLaMA 8B workload.

Alternatives are to solve and save a separate canonical placement at every
mesh (which answers a different replanning question), or introduce a version-2
logical placement format with an explicit axis-size remapper (more expressive,
but a larger compatibility surface). Until the choice is approved, the two
cluster-scale rows must remain unmeasured rather than bypassing the v1 guard.
