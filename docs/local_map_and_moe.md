# Using `local_map` for MoE and Custom Communication Patterns

AutoParallel automatically shards models by analyzing the computation graph
and finding optimal placements for every tensor. But some operations —
particularly in Mixture-of-Experts (MoE) models — involve communication
patterns that depend on runtime data. These can't be expressed through
DTensor's placement primitives (`Shard`, `Replicate`, `Partial`), which
describe static relationships between global and local tensor shapes.

`local_map` is the composition mechanism for these cases. It lets you define
a region where you manage communication manually, while AutoParallel
optimizes everything outside it.

## Prerequisites

This document assumes a 2D mesh with data-parallel and expert-parallel
dimensions:

```python
import torch
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.distributed.tensor.placement_types import Shard, Replicate, Partial
from autoparallel.api import AutoParallel
from autoparallel.collectives import local_map, all_to_all, axis_size

mesh = torch.distributed.device_mesh.init_device_mesh(
    "cuda",
    (dp_size, ep_size),
    mesh_dim_names=("dp", "ep"),
)
```

A few things to keep in mind throughout:

- **Placement tuples follow mesh dimension order.** `(Shard(0), Replicate())`
  means shard on dim 0 of "dp", replicate on "ep".
- **Inside `local_map`, tensors are local shards.** They've already been
  redistributed to match `in_placements` — you work with plain
  `torch.Tensor`s, not DTensors.
- **Use `autoparallel.collectives` for communication inside `local_map`**,
  not raw `torch.distributed` ops. The wrappers in `autoparallel.collectives`
  have correct `torch.autograd.Function` backward implementations so
  gradients flow through collectives.

## When DTensor placements are sufficient for MoE

If routing is perfectly balanced — every token is routed to the same
number of experts (fixed top-k) *and* every expert receives exactly the
same number of tokens — then the all-to-all has equal split sizes, the
permutation is a static reshape, and the expert computation is a batched
matmul with a fixed batch dimension. All of this is expressible through
DTensor placements, and AutoParallel can optimize it automatically without
`local_map`.

In practice, most MoE models don't enforce perfectly balanced routing.
They use auxiliary losses or bias terms to *encourage* balance, but allow
imbalance because forcing uniform routing would hurt model quality — the
whole point of routing is that different tokens should go to different
experts based on content. The rest of this document covers the general
case where routing is data-dependent and imbalanced.

## Why DTensor placements can't describe dynamic MoE routing

DTensor's `Shard(dim)` means "divide this dimension evenly across ranks" —
analogous to `torch.chunk`. Given the global shape and the mesh, you can
compute the local shape on any rank without running the model.

In an MoE layer with dynamic routing, a router sends each token to its
top-k experts. The number of tokens each expert receives depends on the
routing decisions, which are computed at runtime. After the all-to-all
that dispatches tokens to experts, the local tensor on each rank contains
a data-dependent number of tokens. No static placement annotation can
describe this distribution.

## What `local_map` does

`local_map` wraps a function and declares what DTensor placements the
inputs and outputs have. Inside the wrapped function, you work with
regular (local) tensors and handle communication yourself. The optimizer
treats the `local_map` region as an opaque operation with known input/output
placements, and optimizes everything around it.

```python
from autoparallel.collectives import local_map

result = local_map(
    my_function,
    # What placements the outputs will have (one tuple per output)
    out_placements=((Shard(0), Shard(0)),),
    # What placements the inputs should have (one tuple per input)
    in_placements=(
        (Shard(0), Shard(0)),       # tokens: sharded on batch across both DP and EP
        (Replicate(), Shard(0)),    # expert weights: replicated on DP, one expert per EP rank
    ),
    redistribute_inputs=True,       # Automatically redistribute inputs to match in_placements
    device_mesh=mesh,
)(tokens, expert_weights)
```

Key points:
- Each placement tuple has one entry per mesh dimension
- `redistribute_inputs=True` means AutoParallel will insert the necessary
  collectives to get inputs into the declared placements before entering
  the region
- Inside the function, tensors are plain local tensors — no DTensor wrapper
- The function must return tensors whose shapes are consistent with the
  declared `out_placements`

## What belongs inside `local_map` for MoE

The entire MoE dispatch block — from routing through expert computation
to combining results — must live inside a single `local_map` region.
This isn't just the communication primitives; it's everything in between
too, because the intermediate tensors have distributions that only your
code understands.

Concretely, an MoE dispatch involves:

1. **Routing**: compute expert assignments via top-k on router scores
2. **Token permutation**: sort tokens by assigned expert
3. **All-to-all (dispatch)**: send tokens to the rank that owns each expert
4. **Expert computation**: run local experts on received tokens
5. **All-to-all (combine)**: send results back to originating ranks
6. **Token un-permutation**: restore original token order
7. **Score weighting**: combine expert outputs weighted by router scores

All 7 steps must be inside the `local_map`. After step 3, the tensor's
first dimension means "tokens assigned to my experts, from all ranks" —
a distribution that depends on routing decisions. Steps 1-2 produce the
permutation that steps 3, 5, and 6 depend on. There's no point where you
can draw a line and say "above here is auto-shardable, below is manual."

The reason `local_map` works cleanly here is that the combine all-to-all
(step 5) reverses the dispatch all-to-all (step 3), restoring the original
token distribution. The output of the entire block has the same static
placement as the input — the dynamic routing is fully contained within
the region.

## Minimal MoE example

Here's a simplified MoE layer showing the pattern. The dense layers
(gate projection, shared expert) are auto-sharded by AutoParallel.
Only the expert dispatch block uses `local_map`.

```python
import torch
from torch import nn
from autoparallel.collectives import local_map, all_to_all, axis_size
from torch.distributed.tensor.placement_types import Shard, Replicate


def moe_dispatch(
    x: torch.Tensor,              # (batch * seq, dim)
    gate_scores: torch.Tensor,    # (batch * seq, num_experts)
    expert_w1: torch.Tensor,      # (num_local_experts, dim, ffn_dim)
    expert_w2: torch.Tensor,      # (num_local_experts, ffn_dim, dim)
    top_k: int,
    num_experts: int,
    axis_name: str,
) -> torch.Tensor:
    """MoE dispatch: route tokens to experts, compute, and combine results.

    Everything in this function runs on local tensors — no DTensor.
    Communication is via explicit collective calls.
    """
    ep_size = axis_size(axis_name)
    n_tokens = x.shape[0]
    dim = x.shape[-1]

    # 1. Route: pick top-k experts per token
    top_scores, expert_indices = torch.topk(gate_scores, k=top_k, dim=-1)
    top_scores = torch.softmax(top_scores, dim=-1)

    # 2. Count tokens per expert and permute
    num_tokens_per_expert = torch.histc(
        expert_indices.flatten().float(), bins=num_experts, min=0, max=num_experts
    ).int()

    sorted_indices = torch.argsort(expert_indices.flatten(), stable=True)
    token_indices = sorted_indices // top_k
    score_indices = sorted_indices

    routed_input = x[token_indices]
    routed_scores = top_scores.flatten()[score_indices]

    # 3. All-to-all dispatch: send tokens to expert-owning ranks
    with torch.no_grad():
        recv_counts = all_to_all(num_tokens_per_expert, None, None, axis_name)
        send_splits = (
            num_tokens_per_expert.view(ep_size, -1).sum(dim=1).cpu().tolist()
        )
        recv_splits = (
            recv_counts.view(ep_size, -1).sum(dim=1).cpu().tolist()
        )

    routed_input = all_to_all(routed_input, recv_splits, send_splits, axis_name)

    # 4. Expert computation (grouped matmul)
    experts_per_rank = num_experts // ep_size
    recv_counts_local = recv_counts.view(ep_size, experts_per_rank).sum(dim=0)
    offsets = torch.cumsum(recv_counts_local, dim=0, dtype=torch.int32)
    h = torch.nn.functional.silu(
        torch._grouped_mm(routed_input, expert_w1.transpose(-2, -1), offs=offsets)
    )
    routed_output = torch._grouped_mm(h, expert_w2.transpose(-2, -1), offs=offsets)

    # 5. All-to-all combine: send results back
    routed_output = all_to_all(routed_output, send_splits, recv_splits, axis_name)

    # 6-7. Un-permute and score-weight
    out = torch.zeros(n_tokens, dim, dtype=x.dtype, device=x.device)
    out.scatter_add_(0, token_indices.unsqueeze(-1).expand_as(routed_output),
                     routed_output * routed_scores.unsqueeze(-1))
    return out
```

## Wiring `local_map` into the model

The model's forward method calls into the `local_map`-wrapped dispatch.
Everything outside the dispatch — the router's gate projection, the shared
expert FFN, residual connections — is a standard `nn.Module` operation
that AutoParallel shards automatically.

```python
class MoELayer(nn.Module):
    def __init__(self, dim, ffn_dim, num_experts, top_k, mesh):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.top_k = top_k
        self.num_experts = num_experts
        self.mesh = mesh

        # Expert weights: each EP rank holds num_experts // ep_size experts
        ep_size = mesh.size(mesh.mesh_dim_names.index("ep"))
        local_experts = num_experts // ep_size
        self.expert_w1 = nn.Parameter(torch.empty(local_experts, dim, ffn_dim))
        self.expert_w2 = nn.Parameter(torch.empty(local_experts, ffn_dim, dim))

    def forward(self, x):
        bs, slen, dim = x.shape
        x_flat = x.view(-1, dim)

        # Gate projection — auto-sharded by AutoParallel
        gate_scores = self.gate(x_flat)

        # Expert dispatch — wrapped in local_map
        #
        # Placement contract:
        # - x_flat, gate_scores: (Shard(0), Shard(0)) — batch-sharded on DP and EP
        # - expert weights: (Replicate(), Shard(0)) — each EP rank has its local experts
        # - output: (Shard(0), Shard(0)) — same as input
        out = local_map(
            moe_dispatch,
            out_placements=((Shard(0), Shard(0)),),
            in_placements=(
                (Shard(0), Shard(0)),       # x_flat
                (Shard(0), Shard(0)),       # gate_scores
                (Replicate(), Shard(0)),    # expert_w1
                (Replicate(), Shard(0)),    # expert_w2
                None,                       # top_k (non-tensor)
                None,                       # num_experts (non-tensor)
                None,                       # axis_name (non-tensor)
            ),
            redistribute_inputs=True,
            device_mesh=self.mesh,
        )(x_flat, gate_scores, self.expert_w1, self.expert_w2,
          self.top_k, self.num_experts, "ep")

        return out.view(bs, slen, dim)
```

## How AutoParallel sees this

From AutoParallel's perspective, the `local_map` call is a single node
in the FX graph with declared input and output placements. The optimizer:

1. Knows the inputs must be redistributed to match `in_placements`
   (e.g., if `x_flat` arrives as `(Shard(0), Replicate())`, an all-gather
   or redistribute will be inserted on the EP dimension)
2. Knows the output has placement `(Shard(0), Shard(0))`
3. Optimizes everything else — the gate projection, any shared expert,
   residual connections, the rest of the transformer — jointly via ILP

The MoE dispatch is a fixed cost in the optimization: AutoParallel doesn't
try to optimize the communication inside `local_map`, but it does account
for the redistribution cost to get inputs into the required placements.

## EP `local_map` design space

The 2D `(dp, ep)` example above is the smallest useful EP `local_map`
contract, not the only one. Higher-rank meshes do not imply a single
placement policy per rank. A 3D, 4D, or 5D mesh can support multiple EP
variants depending on which axes are used for token sharding, expert
ownership, expert compute, and parameter storage.

The EP `local_map` boundary should keep the same semantic shape across those
variants:

1. Inputs enter the region already redistributed to the declared dense and
   routing placements.
2. The local body reorders tokens, dispatches them across the EP axis, runs
   local expert compute, combines outputs across the EP axis, and restores
   the original token order.
3. The output leaves the region with a static DTensor placement, often as a
   `Partial` value that is reduced or redistributed at the MoE boundary.

The axes in a larger mesh have distinct roles:

| Axis | Role inside or around EP `local_map` |
|---|---|
| `dp` | Logical batch or token sharding axis. |
| `dp_replicate` | Data-parallel replica axis; usually replicated at the local_map boundary. |
| `dp_shard`, `fsdp`, `efsdp` | Parameter storage/FSDP axes; usually not the token-exchange axis. |
| `cp` | Context or sequence sharding axis. |
| `tp` | Dense tensor-parallel or sequence-parallel axis; can also be an expert-compute axis if the expert kernels support it. |
| `ep` | Expert ownership and token all-to-all axis. |
| `pp` | Pipeline axis; it should stay outside an intra-stage `local_map` boundary. |

This gives several useful mesh-rank variants:

| Mesh axes | Variant | Status |
|---|---|---|
| `(dp, tp, ep)` | Logical DP + TP/SP + EP. `dp` shards batch or tokens, `tp` shards sequence or hidden state, and `ep` owns experts. | Near-term design target. |
| `(dp, cp, ep)` | DP + CP + EP without TP. `cp` shards sequence/context, and `ep` owns experts. | Near-term design target. |
| `(dp_replicate, efsdp, ep)` | Sparse EP view. `dp_replicate` and `efsdp` are replica/storage axes, while `ep` shards the expert dimension. | Canonical sparse EP view. |
| `(dp, cp, tp, ep)` | Full logical dense + EP view. `dp` shards batch, `cp` and `tp` can jointly shard sequence, and `ep` owns experts. | Best general reasoning model. |
| `(dp_replicate, dp_shard, tp, ep)` | Explicit data replicate/shard split without CP. | Useful when storage axes must remain visible. |
| `(dp_replicate, efsdp, tp, ep)` | Sparse EP plus visible expert-compute TP axis. | Future gated; requires an explicit expert-kernel layout. |
| `(dp_replicate, dp_shard, cp, tp, ep)` | Complete logical dense/sparse view. Dense activations use the non-EP axes, while sparse experts are usually represented by folding into `efsdp`. | Future design target. |

For structured activations shaped `(batch, seq, hidden)`, sequence sharding is
usually expressed as `Shard(1)`. For flattened MoE dispatch inputs shaped
`(tokens, hidden)`, batch and sequence sharding both appear as token-dimension
`Shard(0)`. Keep this distinction explicit when translating a high-level
parallel-dims design into concrete `in_placements` and `out_placements`.

Expert tensor parallelism is a separate design axis from expert parallelism.
EP decides which rank owns each expert and exchanges tokens with all-to-all.
Expert tensor parallelism shards the expert MLP computation itself. Adding a
separate `etp` axis, or treating `tp` as expert tensor parallel inside the EP
region, requires a matching expert weight layout, grouped-GEMM contract, and
combine rule. It should not be modeled as a placement-only change.

## Helpers in `autoparallel.collectives`

AutoParallel provides collective wrappers that work inside `local_map`
regions and have correct backward pass implementations:

| Function | Forward | Backward |
|---|---|---|
| `all_gather(x, gather_dim, axis_name)` | All-gather on `gather_dim` | Reduce-scatter |
| `reduce_scatter(x, scatter_dim, axis_name)` | Reduce-scatter on `scatter_dim` | All-gather |
| `all_reduce(x, axis_name)` | Sum all-reduce | Sum all-reduce |
| `all_to_all(x, out_splits, in_splits, axis_name)` | All-to-all | All-to-all (reversed splits) |
| `axis_size(axis_name)` | Returns mesh size for named dim | — |

These use `axis_name` (a mesh dimension name like `"ep"` or `"dp"`) to
resolve the process group from the current mesh context. The
`torch.autograd.Function` wrappers ensure gradients flow correctly through
the collectives during backward.

## `flex_local_map`: give the solver placement alternatives

A plain `local_map` fixes one input/output placement contract, so the solver's
only freedom is the redistribution cost to reach it. `flex_local_map` instead
declares several *semantically equivalent* contracts for the same region (e.g.
MoE `DP->EP` vs `DP+TP->EP+ETP`) and lets the solver choose:

```python
from autoparallel import flex_local_map

# Built OUTSIDE forward (e.g. in __init__), with an explicit device_mesh.
self.moe = flex_local_map(
    moe_body,
    alternatives=[
        {
            "name": "dp_ep",
            "in_placements": ...,
            "out_placements": ...,
        },
        {
            "name": "dp_tp_ep_etp",
            "fn": moe_body_with_tp,
            "in_placements": ...,
            "out_placements": ...,
            "cost_hint": 1.0,
        },
    ],
    device_mesh=mesh,
    redistribute_inputs=True,
)
```

Each alternative is a dict with required `in_placements` and `out_placements`.
`fn` defaults to the first argument, while `name` and the non-negative `cost_hint`
are optional. The hint is added to the solver cost; it is not a measured runtime.
AutoParallel emits one solver strategy per alternative.

Usage constraint: apply `flex_local_map` **outside** `forward` and pass an explicit
`device_mesh`, then call the stored callable inside `forward`. The alternatives are
carried on a tuple-subclass that Dynamo only preserves when it is a pre-existing
(sourced) object; constructing the wrapper inside `forward` traces it as empty.
`in_grad_placements` is reserved for parity with `local_map`; passing a non-`None`
value currently raises `NotImplementedError`.

`optimize_placement()` first selects an alternative while constraining the forward and
backward nodes to the same index. This is the only ILP solve. It then captures the
selected `fn` through the normal `local_map` Dynamo/AOT path and replaces that callsite's
forward body, saved activations, and backward body in place. After finalization the node
is an ordinary single-strategy `local_map` and follows the existing apply path. Only the
selected function is captured. Alternative functions must preserve the same user-visible
tensor contract and differentiable outputs; their internal saved activations may differ.

## `with_sharding_constraint` for simpler cases

If you don't need custom communication but want to force a specific
intermediate placement, use `with_sharding_constraint` instead of
`local_map`. This is analogous to JAX's `with_sharding_constraint`:

```python
from autoparallel.collectives import with_sharding_constraint

# Force activations to be sequence-sharded on the TP dimension
x = with_sharding_constraint(x, (Shard(0), Shard(1)))
```

This inserts a redistribute if the current placement doesn't match,
and constrains the optimizer to maintain this placement. Use it when
the operation is expressible through DTensor but you want to override
the optimizer's choice.

## Putting it all together

Here's how the `MoELayer` from above is used end-to-end with
AutoParallel. The model contains both dense layers (auto-sharded) and
the MoE dispatch (manual via `local_map`):

```python
from autoparallel.api import auto_parallel

class MyModel(nn.Module):
    def __init__(self, dim, ffn_dim, num_experts, top_k, mesh):
        super().__init__()
        self.embed = nn.Linear(dim, dim, bias=False)
        self.moe = MoELayer(dim, ffn_dim, num_experts, top_k, mesh)
        self.head = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        x = self.embed(x)
        x = x + self.moe(x)
        return self.head(x)

# Create model on meta device
with torch.device("meta"):
    model = MyModel(dim=512, ffn_dim=2048, num_experts=64, top_k=2, mesh=mesh)

batch_size = 8 * dp_size * ep_size
seq_len = 1024

sample_input = DTensor.from_local(
    torch.randn(batch_size // (dp_size * ep_size), seq_len, 512),
    mesh,
    (Shard(0), Shard(0)),
)

parallel_model = auto_parallel(
    model,
    mesh,
    sample_inputs=(sample_input,),
    out_shardings=(Shard(0), Shard(0)),
)

parallel_model.to_empty(device="cuda")
parallel_model.init_weights()

# Run forward + backward with local batch
local_batch = batch_size // (dp_size * ep_size)
x = torch.randn(local_batch, seq_len, 512, device="cuda")
out = parallel_model(x)
out.sum().backward()
```

AutoParallel optimizes the sharding of `embed`, `head`, and the gate
projection inside `MoELayer`. The `local_map` region is treated as a
single op with fixed input/output placements.

## Common pitfalls

**Output shape must match `out_placements`.** The local tensor returned
by your `local_map` function must have a shape consistent with the
declared output placements and the global shape. If you return a tensor
with an unexpected local shape, tracing or lowering will fail with a
shape mismatch error.

**Use `autoparallel.collectives`, not raw `torch.distributed`.** The
wrappers in `autoparallel.collectives` (`all_to_all`, `all_gather`, etc.)
have `torch.autograd.Function` backward implementations. If you call
`torch.distributed.all_to_all_single` directly, gradients won't flow
through the collective and backward will silently produce wrong results.

**All-to-all split sizes must be globally consistent.** Each rank's
`output_split_sizes` must match the corresponding `input_split_sizes` on
the sending ranks. If they don't agree, the collective will hang or
produce garbage. When computing splits from `num_tokens_per_expert`,
make sure the all-to-all that exchanges token counts (step 3 in the
example) completes before you use the received counts.

**`num_experts` must be divisible by `ep_size`.** The examples above
assume each EP rank owns `num_experts // ep_size` experts. If this
doesn't divide evenly, you'll need to handle uneven expert assignment
(some ranks own more experts than others), which complicates the split
size computation and grouped matmul offsets.

**Non-tensor arguments need `None` in `in_placements`.** For non-tensor
arguments (ints, strings, etc.), set the corresponding `in_placements`
entry to `None`. Missing or mismatched entries will cause placement
validation errors.

**`in_placements` maps to traced input order, not Python argument order.**
`in_placements` is a 1:1 mapping to the flattened inputs of the wrapped
function as seen by the tracer. When using `local_map` as a decorator
(rather than calling it inline), Dynamo may reorder captured variables
(closed-over tensors) before explicit arguments during tracing. If
placements seem misaligned, check the actual argument order in the
traced graph.

## Summary

| Scenario | Mechanism |
|---|---|
| Standard ops (linear, attention, pointwise) | Automatic — AutoParallel handles it |
| Force a specific intermediate sharding | `with_sharding_constraint` |
| Data-dependent communication (MoE routing, custom all-to-all patterns) | `local_map` |

The key insight: `local_map` isn't a workaround — it's the designed
composition point between automatic and manual parallelism. AutoParallel
optimizes the dense parts of your model globally, and you handle the
parts where the communication pattern is determined by runtime data.
