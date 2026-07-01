# Context Parallel Attention

Context Parallel (CP) shards the sequence dimension across a mesh axis. Each
rank computes attention for its local query slice while using full-sequence
keys and values.

AutoParallel provides helpers for wrapping local attention kernels with the
placements needed for CP:

```python
import torch

from autoparallel.context_parallel import make_context_parallel_sdpa

mesh = torch.distributed.device_mesh.init_device_mesh(
    "cuda",
    (dp_size, cp_size, tp_size),
    mesh_dim_names=("dp_shard", "cp", "tp"),
)

cp_sdpa = make_context_parallel_sdpa(mesh, is_causal=True)
out = cp_sdpa(q, k, v)
```

The SDPA helper expects tensors in `(B, H, S, D)` layout by default:

- `B`: batch
- `H`: attention heads
- `S`: sequence
- `D`: head dimension

The default placements are:

| Mesh axis | Q placement | K/V placement | Output placement |
|---|---|---|---|
| `dp_shard` | `Shard(B)` | `Shard(B)` | `Shard(B)` |
| `cp` | `Shard(S)` | `Replicate()` | `Shard(S)` |
| `tp` | `Shard(H)` | `Shard(H)` | `Shard(H)` |

For Torchtitan-style `(B, L, N, H)` tensors, use:

```python
from autoparallel.context_parallel import context_parallel_local_map


@context_parallel_local_map(mesh=mesh, batch_dim=0, seq_dim=1, head_dim=2)
def inner_attention(q, k, v):
    ...
```

## Supported Meshes

The helpers recognize these mesh axis names:

- DP shard: `dp`, `dp_shard`, `fsdp`, `data`, `data_parallel`
- DP replicate: `dp_replicate`, `ddp`
- CP: `cp`, `context`, `context_parallel`
- TP: `tp`, `tensor`, `tensor_parallel`

Common layouts:

```text
("dp_shard", "cp")
("dp_shard", "tp")
("dp_shard", "cp", "tp")
("dp_replicate", "dp_shard", "cp", "tp")
```

For 1D and 2D meshes, pass `mesh_dim_names`. A 2D mesh can mean either
`("dp_shard", "cp")` or `("dp_shard", "tp")`, and the CP helper needs the
names to select the correct placements.

## Gradient Placements

CP attention has asymmetric gradient placements on the CP axis:

| Tensor | Forward CP placement | Gradient CP placement |
|---|---|---|
| Q | `Shard(sequence)` | `Shard(sequence)` |
| K | `Replicate()` | `Partial()` |
| V | `Replicate()` | `Partial()` |

K and V are replicated in forward because each query shard attends to the full
key/value sequence. In backward, each CP rank computes only the gradient
contribution from its local query shard. The full K/V gradients are the sum of
those per-rank contributions, so the local gradients are `Partial()`.

The CP helpers pass these gradient placements through `local_map` so
AutoParallel can insert the required reductions when gradients leave the local
attention region.
