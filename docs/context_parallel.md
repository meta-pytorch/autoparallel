# Context Parallel Attention

Context Parallel (CP) lets transformer attention shard the sequence dimension
across a named mesh axis while keeping the rest of the model in AutoParallel's
normal placement flow. The public entry points are exported from
`autoparallel`.

## Basic Usage

```python
import torch
from autoparallel import make_context_parallel

mesh = torch.distributed.device_mesh.init_device_mesh(
    "cuda",
    (dp, cp, tp),
    mesh_dim_names=("dp_shard", "cp", "tp"),
)

cp_sdpa = make_context_parallel(mesh, is_causal=True)
out = cp_sdpa(q, k, v)
```

For SDPA tensors shaped `(batch, heads, sequence, head_dim)`, the default
dimension arguments are:

| Argument | Default | Tensor dimension |
| --- | --- | --- |
| `batch_dim` | `0` | batch |
| `head_dim` | `1` | attention heads |
| `seq_dim` | `2` | sequence |

## Public API

`make_context_parallel(...)` builds an attention callable for CP. SDPA is the
default attention kind.

Important arguments:

- `mesh`: Device mesh with named DP, CP, and/or TP dimensions.
- `kind`: `"sdpa"` or `"flex_attention"`.
- `is_causal`: Whether SDPA uses causal masking.
- `dropout_p`: Dropout probability for SDPA.
- `scale`, `enable_gqa`: Passed through to the selected attention kind.
- `block_mask`, `score_mod`, `kernel_options`: FlexAttention options.
- `batch_dim`, `seq_dim`, `head_dim`: Tensor dimensions used for placement.

`make_context_parallel_sdpa(...)` is a compatibility alias for SDPA callers.

`context_parallel_attention_placements(...)` returns the Q/K/V and output
placements for a mesh. Use it when you need to add matching AutoParallel input
or output constraints.

## Mesh Names

Supported mesh axis names:

| Role | Names |
| --- | --- |
| DP shard | `dp`, `dp_shard`, `fsdp`, `data`, `data_parallel` |
| CP | `cp`, `context`, `context_parallel` |
| TP | `tp`, `tensor`, `tensor_parallel` |
| Extra DP axis | `dp_replicate`, `ddp` |

For 1D and 2D meshes, pass `mesh_dim_names` explicitly so AutoParallel can
identify the CP axis unambiguously.

## Notes

- CP support is intended for model code that already runs under AutoParallel.
- The Llama test model can opt in by setting `context_parallel_mesh` on
  `TransformerModelArgs`.
- Q, K, V, and attention output use matching placement tuples at the
  AutoParallel boundary.
- FlexAttention CP supports `block_mask`, `scale`, and `enable_gqa`.
- FlexAttention `score_mod` is not supported when a CP axis is present.
- SDPA dropout is not supported when a CP axis is present.
