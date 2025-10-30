# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext

import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.tensor.placement_types import Shard
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel._testing.models.dsv3 import (
    precompute_freqs_cis,
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
    n_layers=8,  # 27,
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

with torch.device("meta"):
    model = DeepSeekV3Model(config).bfloat16()
    embed, layers, norm, output = list(model.children())
    layers = [nn.ModuleDict({k: v}) for k, v in layers.items()]
    assert len(layers) == 8

class PipelineStage(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.register_buffer(
            "freqs_cis", precompute_freqs_cis(config), persistent=False
        )

    def forward(self, h):
        # intermediate stages only have layers
        for layer in self.layers.values():
            # h = (bs=1024, seq=1024, hidden=2048)
            h = layer(h, self.freqs_cis)
        return h

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(buffer_device=buffer_device)

class FirstPipelineStage(PipelineStage):
    def __init__(self, embed, layers):
        super().__init__(layers)
        self.tok_embeddings = embed

    def forward(self, tokens):
        # torch.Size([1024, 1024])
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        # torch.Size([1024, 1024, 2048])
        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)
        return h

class LastPipelineStage(PipelineStage):
    def __init__(self, layers, norm, output):
        super().__init__(layers)
        self.norm = norm
        self.output = output

    def forward(self, h):
        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)
        h = self.norm(h) if self.norm is not None else h
        output = self.output(h) if self.output is not None else h
        return output

def input_fn():
    return torch.randint(
        0,
        config.vocab_size,
        (bs, seq_len),
        device=device,
    )

def input_fn_after_first_stage():
    return torch.randn(
        (bs, seq_len, config.dim),
        device=device,
        dtype=torch.bfloat16,
    )

def runtime_input_fn():
    return (
        torch.randint(
            0,
            config.vocab_size,
            (bs // mesh.shape[0] // mesh.shape[1], seq_len),
            device=device,
        ),
    )

def runtime_input_fn_after_first_stage():
    return (
        torch.randn(
            (bs // mesh.shape[0] // mesh.shape[1], seq_len, config.dim),
            device=device,
            dtype=torch.bfloat16,
        ),
    )

with torch.device("meta"):
    stage0 = FirstPipelineStage(embed, layers[0])
    stage1 = PipelineStage(layers[1])
    stage2 = PipelineStage(layers[2])
    stage3 = PipelineStage(layers[3])
    stage4 = PipelineStage(layers[4])
    stage5 = PipelineStage(layers[5])
    stage6 = PipelineStage(layers[6])
    stage7 = LastPipelineStage(layers[7], norm, output)

####################
# Stage 0
# model = stage0

# Stage 1-7
model = stage7
input_fn = input_fn_after_first_stage
runtime_input_fn = runtime_input_fn_after_first_stage
####################

with AutoParallel(model, input_fn, mesh, dynamic=True) as autop:
    autop.add_parameter_memory_constraint(low=None, high=None)

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
x = runtime_input_fn()

# Symbolically evaluate in case you want to test running a graph bigger than your gpu

with (
    FakeTensorMode(
        allow_non_fake_inputs=True,
        shape_env=ShapeEnv(),
    )
    if fake_evaluate
    else nullcontext()
):
    # # now let's run it
    output = pp_mod(*x)
    assert not isinstance(output, tuple)
    output.backward(torch.randn_like(output))


print("All good!")
