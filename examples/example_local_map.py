# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import functools

import torch
from torch import nn
from torch.distributed._tensor.experimental import local_map
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.utils.checkpoint import create_selective_checkpoint_contexts

from autoparallel.api import AutoParallel

world_size = 256

fake_store = FakeStore()
torch.distributed.init_process_group(
    "fake", store=fake_store, rank=0, world_size=world_size
)
mesh = torch.distributed.device_mesh.init_device_mesh(
    "cuda",
    (world_size // 32, 8, 4),
    mesh_dim_names=(
        "dp",
        "tp",
        "cp",
    ),
)
assert mesh.ndim == 3, "Please also update local_map"


def policy_fn(ctx, op, *args, **kwargs):
    if (
        op == torch.ops.aten._scaled_dot_product_flash_attention.default
        or op == torch.ops.aten._scaled_dot_product_efficient_attention.default
    ):
        # NOTE: we can't save nondeterministic_seeded ops, the run with rng wrapper is not traceable yet
        return torch.utils.checkpoint.CheckpointPolicy.PREFER_SAVE
    return torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE


context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)


@local_map(
    out_placements=((Replicate(), Replicate(), Replicate()),),
    in_placements=(
        (Replicate(), Replicate(), Replicate()),
        (Replicate(), Replicate(), Replicate()),
    ),
    redistribute_inputs=True,
    in_grad_placements=None,
    device_mesh=mesh,
)
def replicate_linear(w, x):
    return torch.matmul(x, w.t())


@local_map(
    out_placements=(
        (Shard(0), Shard(0), Replicate()),
        None,
    ),
    in_placements=(
        (Shard(0), Shard(0), Replicate()),
        None,
    ),
    redistribute_inputs=True,
    in_grad_placements=None,
    device_mesh=mesh,
)
def sharded_pointwise(x, scalar):
    return x + scalar, scalar


@local_map(
    out_placements=((Shard(0), Shard(1), Shard(2)),),
    in_placements=(
        (Shard(0), Shard(1), Shard(2)),
        (Shard(0), Shard(1), Shard(2)),
        (Shard(0), Shard(1), Shard(2)),
    ),
    redistribute_inputs=True,
    in_grad_placements=None,
    device_mesh=mesh,
)
def context_parallel_attention(query, key, value):
    out = nn.functional.scaled_dot_product_attention(
        query=query, key=key, value=value, is_causal=False
    )
    return out


class Block(nn.Module):
    def __init__(self, nheads, dim1, dim2):
        super().__init__()
        self.nheads = nheads
        bias = False
        self.wq = nn.Linear(dim1, dim1, bias=bias)
        self.wk = nn.Linear(dim1, dim1, bias=bias)
        self.wv = nn.Linear(dim1, dim1, bias=bias)
        self.wo = nn.Linear(dim1, dim1, bias=bias)
        self.w1 = nn.Linear(dim1, dim2, bias=bias)
        self.w2 = nn.Linear(dim2, dim1, bias=bias)

    def init_weights(self):
        for lin in [self.wq, self.wk, self.wv, self.wo, self.w1, self.w2]:
            torch.nn.init.normal_(lin.weight)
            if lin.bias is not None:
                torch.nn.init.normal_(lin.bias)

    def _compute_attention(self, x):
        boosted_weight, scalar = sharded_pointwise(self.wq.weight, 10)
        q = replicate_linear(boosted_weight, x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
        k = k.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
        v = v.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)

        o = context_parallel_attention(q, k, v)
        o = o.permute(0, 2, 1, 3).flatten(-2)

        o = self.wo(o)
        return o

    def forward(self, x):
        o = torch.utils.checkpoint.checkpoint(
            self._compute_attention, x, use_reentrant=False, context_fn=context_fn
        )

        o0 = o + x

        o = self.w1(o0)
        o = torch.nn.functional.relu(o)
        o = self.w2(o)

        o = o0 + o

        return o


bs = 8 * mesh.shape[0]
seq_len = 256
nheads = 48
dim1 = 6144
dim2 = dim1 * 4


def input_fn():
    print(f"global input shape: {(bs, seq_len, dim1)}")
    return torch.rand(bs, seq_len, dim1, device="cuda")


# parallelize the model
with torch.device("meta"):
    model = Block(nheads, dim1, dim2)

# MP policy causing some deepcopy issues
# mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
# mp_policy = None

with AutoParallel(model, input_fn, mesh, mp_policy, compile=True) as autop:
    assert any(n.meta.get("nn_module_stack") for n in autop.gm.graph.nodes)
    assert any(n.meta.get("fwd_nn_module_stack") for n in autop.gm.graph.nodes)
    autop.add_parameter_memory_constraint(low=None, high=None)

    x_sharding = (Shard(0),) + (Replicate(),) * (mesh.ndim - 1)

    autop.add_input_constraints([x_sharding])
    autop.add_output_constraints([x_sharding])

    sharding_placement = autop.optimize_placement()

    # AutoParallel produces a module with meta-DTensor parameters that need to be initialized
    parallel_mod = autop.apply_placement(sharding_placement)

parallel_mod.to_empty(device="cuda")
parallel_mod.init_weights()

# now let's run it
x = (torch.rand(bs // mesh.shape[0], seq_len, dim1, device="cuda"),)
out = parallel_mod(*x)
out.backward(torch.randn_like(out))

# Validate
seqs = set()
for n in autop.gm.graph.nodes:
    if "checkpoint" in n.meta.get(
        "stack_trace", ""
    ):  # placeholders don't have stack trace
        is_bwd = n.meta.get("partitioner_tag", "") == "is_backward"
        if not is_bwd:
            if "getitem" in str(n.target):
                # getitem nodes are tagged same as their parent
                expected = policy_fn(None, n.args[0].target, (), ())
            else:
                expected = policy_fn(None, n.target, (), ())
            actual = n.meta.get("recompute")
            # NOTE: this assert only supports policy_fns on op alone
            assert actual == expected
            seqs.add(n.meta["seq_nr"])
        else:
            # fwd counterpart should have already populated seqs
            assert n.meta["seq_nr"] in seqs

mm_nodes = autop.gm.graph.find_nodes(
    op="call_function", target=torch.ops.aten.mm.default
)

print("All good!")
