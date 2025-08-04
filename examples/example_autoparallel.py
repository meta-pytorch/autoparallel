# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel.api import AutoParallel


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
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
        k = k.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
        v = v.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)

        o = nn.functional.scaled_dot_product_attention(q, k, v)
        o = o.permute(0, 2, 1, 3).flatten(-2)

        o = self.wo(o)
        return o

    def forward(self, x):
        o = torch.utils.checkpoint.checkpoint(
            self._compute_attention, x, use_reentrant=False
        )

        o0 = o + x

        o = self.w1(o0)
        o = torch.nn.functional.relu(o)
        o = self.w2(o)

        o = o0 + o

        return o


world_size = 256

fake_store = FakeStore()
torch.distributed.init_process_group(
    "fake", store=fake_store, rank=0, world_size=world_size
)

use_1d_mesh = False

if use_1d_mesh:
    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda", (world_size,), mesh_dim_names=("dp",)
    )
else:
    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda",
        (world_size // 8, 8),
        mesh_dim_names=(
            "dp",
            "tp",
        ),
    )

bs = 8 * mesh.shape[0]
seq_len = 256
nheads = 48
dim1 = 6144
dim2 = dim1 * 4


def input_fn():
    return torch.rand(bs, seq_len, dim1, device="cuda")


# parallelize the model
with torch.device("meta"):
    model = Block(nheads, dim1, dim2)

mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
# mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
# mp_policy = None

with AutoParallel(model, input_fn, mesh, mp_policy) as autop:
    autop.add_parameter_memory_constraint(low=None, high=None)

    x_sharding = (Shard(0),) + (Replicate(),) * (mesh.ndim - 1)

    autop.add_input_constraints([x_sharding])
    autop.add_output_constraints([x_sharding])

    sharding_placement = autop.optimize_placement()

    # AutoParallel produces a module with meta-DTensor parameters that need to be initialized
    parallel_mod = autop.apply_placement(sharding_placement)

parallel_mod.to_empty(device="cuda")
parallel_mod.init_weights()

parallel_mod.compile(fullgraph=True)

# now let's run it
x = (torch.rand(bs // mesh.shape[0], seq_len, dim1, device="cuda"),)
out = parallel_mod(*x)
out.backward(torch.randn_like(out))

print("All good!")

mm_nodes = autop.gm.graph.find_nodes(
    op="call_function", target=torch.ops.aten.mm.default
)

# assert (
#     mm_nodes[0].meta.get("recompute")
#     == torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE
# )

# TODO: change this assert once we fix AC
assert mm_nodes[0].meta.get("recompute") is None
