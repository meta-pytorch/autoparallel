# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.tensor.placement_types import Shard
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel._testing.models.dsv3 import (
    DeepSeekV3Model,
    DeepSeekV3ModelArgs,
    MoEArgs,
)
from autoparallel.api import AutoParallel

# must symbolically evaluate to run on 32 dp ranks
# world_size = 2048
fake_evaluate = False

world_size = 256

fake_store = FakeStore()
torch.distributed.init_process_group(
    "fake", store=fake_store, rank=0, world_size=world_size
)
# mesh = torch.distributed.device_mesh.init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
mesh = torch.distributed.device_mesh.init_device_mesh(
    "cuda",
    (world_size // 64, 64),
    mesh_dim_names=(
        "dp",
        "ep",
    ),
)

device = torch.device("cuda")

bs = 4 * mesh.shape[0] * mesh.shape[1]
seq_len = 1024

config = DeepSeekV3ModelArgs(
    vocab_size=102400,
    max_seq_len=seq_len,
    dim=2048,
    inter_dim=10944,
    moe_inter_dim=1408,
    n_layers=1,  # 27,
    n_dense_layers=0,  # 1,
    n_heads=16,
    moe_args=MoEArgs(
        num_experts=64,
        num_shared_experts=2,
        top_k=6,
        score_func="softmax",
        route_norm=False,
        score_before_experts=False,
        mesh=mesh,
    ),
    q_lora_rank=0,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    mscale=0.70,
    use_flex_attn=False,
    attn_mask_type="causal",
)

# parallelize the model
with torch.device("meta"):
    model = DeepSeekV3Model(config).bfloat16()


def input_fn():
    return torch.randint(
        0,
        config.vocab_size,
        (bs, seq_len),
        device=device,
    )


with AutoParallel(model, input_fn, mesh, dynamic=True) as autop:
    autop.add_parameter_memory_constraint(low=None, high=None)

    # x_sharding = (Shard(0), Replicate())
    x_sharding = (Shard(0), Shard(0))

    autop.add_input_constraints([x_sharding])
    autop.add_output_constraints([x_sharding])

    sharding_placement = autop.optimize_placement()
    pp_mod = autop.apply_placement_pp(sharding_placement)

pp_mod.to_empty(device="cuda")
# run weight init on our sharded DTensor params
# TODO: plumb init_std through
# pp_mod.init_weights(
#     init_std=0.02, buffer_device="cuda"
# )  # maybe not correct value
pp_mod.init_weights(buffer_device="cuda")
x = (
    torch.randint(
        0,
        config.vocab_size,
        (bs // mesh.shape[0] // mesh.shape[1], seq_len),
        device=torch.device("cuda"),
    ),
)

# Symbolically evaluate in case you want to test running a graph bigger than your gpu

mode: nullcontext[None] | FakeTensorMode = nullcontext()
if fake_evaluate:
    mode = FakeTensorMode(
        allow_non_fake_inputs=True,
        shape_env=ShapeEnv(),
    )

with mode:
    # # now let's run it
    outputs = pp_mod(*x)
    assert len(outputs) == 1
    output = outputs[0]
    output.backward(torch.randn_like(output))


print("All good!")
