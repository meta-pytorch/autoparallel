# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard

from autoparallel.api import auto_parallel


def test_init(device_mesh_1d):
    dim = 128

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            self.register_buffer("buf", torch.empty(dim))

        def forward(self, x):
            return self.linear(x) + self.buf

        def init_weights(self):
            self.linear.weight = torch.nn.Parameter(torch.ones(dim, dim) * 9.0)
            with torch.no_grad():
                self.linear.bias.fill_(98.6)
            self.buf = torch.arange(dim)

    with torch.device("meta"):
        model = Model(dim)

    batch_size = 512
    local_batch_size = batch_size // device_mesh_1d.size()
    x = DTensor.from_local(
        torch.rand(local_batch_size, dim, device="cuda"),
        device_mesh_1d,
        [Shard(0)],
    )
    parallel_mod = auto_parallel(
        model,
        device_mesh_1d,
        sample_inputs=(x,),
        out_shardings=(Shard(0),),
        compile=False,
    )
    parallel_mod.to_empty(device="cuda")
    parallel_mod.init_weights()
    assert torch.equal(
        parallel_mod.get_parameter("linear.weight").full_tensor(),
        torch.full((dim, dim), 9.0, device="cuda"),
    )
    assert torch.equal(
        parallel_mod.get_parameter("linear.bias").full_tensor(),
        torch.full((dim,), 98.6, device="cuda"),
    )
    assert torch.equal(
        parallel_mod.get_buffer("buf").full_tensor(), torch.arange(dim, device="cuda")
    )


def test_init_inplace_data(device_mesh_1d):
    """Test that init_weights using self.weight.data[:] = value works correctly."""
    dim = 128

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            self.register_buffer("buf", torch.empty(dim))

        def forward(self, x):
            return self.linear(x) + self.buf

        def init_weights(self):
            self.linear.weight.data[:] = torch.ones(dim, dim) * 9.0
            self.linear.bias.data[:] = torch.full((dim,), 98.6)
            self.buf.data[:] = torch.arange(dim, dtype=torch.float32)

    with torch.device("meta"):
        model = Model(dim)

    batch_size = 512
    local_batch_size = batch_size // device_mesh_1d.size()
    x = DTensor.from_local(
        torch.rand(local_batch_size, dim, device="cuda"),
        device_mesh_1d,
        [Shard(0)],
    )
    parallel_mod = auto_parallel(
        model,
        device_mesh_1d,
        sample_inputs=(x,),
        out_shardings=(Shard(0),),
        compile=False,
    )
    parallel_mod.to_empty(device="cuda")
    parallel_mod.init_weights()
    assert torch.equal(
        parallel_mod.get_parameter("linear.weight").full_tensor(),
        torch.full((dim, dim), 9.0, device="cuda"),
    )
    assert torch.equal(
        parallel_mod.get_parameter("linear.bias").full_tensor(),
        torch.full((dim,), 98.6, device="cuda"),
    )
    assert torch.equal(
        parallel_mod.get_buffer("buf").full_tensor(),
        torch.arange(dim, dtype=torch.float32, device="cuda"),
    )


def test_init_aliased_buffers(device_mesh_1d):
    """Test that init_weights works when a submodule buffer aliases a top-level buffer.

    This mirrors the torchtitan Decoder pattern where rope.cache and freqs_cis
    are the same tensor. named_buffers(remove_duplicate=True) deduplicates them,
    so only freqs_cis ends up on the parallel model. The init_weights hook must
    still correctly propagate values set via the aliased buffer (rope.cache).
    """
    dim = 128

    class RoPE(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.register_buffer("cache", torch.zeros(dim), persistent=False)

        def forward(self, x):
            return x + self.cache

        def init_weights(self):
            self.cache = torch.arange(dim).float()

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            self.rope = RoPE(dim)
            self.register_buffer("freqs_cis", self.rope.cache, persistent=False)

        def forward(self, x):
            return self.linear(x) + self.freqs_cis

        def init_weights(self):
            with torch.no_grad():
                self.linear.weight.fill_(1.0)
                self.linear.bias.fill_(0.0)
            self.rope.init_weights()
            self.freqs_cis = self.rope.cache

    with torch.device("meta"):
        model = Model(dim)

    assert model.freqs_cis is model.rope.cache

    batch_size = 512
    local_batch_size = batch_size // device_mesh_1d.size()
    x = DTensor.from_local(
        torch.rand(local_batch_size, dim, device="cuda"),
        device_mesh_1d,
        [Shard(0)],
    )
    parallel_mod = auto_parallel(
        model,
        device_mesh_1d,
        sample_inputs=(x,),
        out_shardings=(Shard(0),),
        compile=False,
    )

    parallel_mod.to_empty(device="cuda")
    parallel_mod.init_weights()

    expected = torch.arange(dim).float().cuda()
    assert torch.equal(parallel_mod.get_buffer("freqs_cis").full_tensor(), expected)


def test_init_aliased_parameters(device_mesh_1d):
    """Test that init_weights works when a parameter is registered under two FQNs.

    This mirrors weight tying in LLMs where embed.weight and lm_head.weight
    are the same parameter. named_parameters() deduplicates them, so the alias
    FQN is missing from the parallel model. The init_weights hook must not
    crash on the missing alias.
    """
    dim = 128

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.embed = nn.Linear(dim, dim, bias=False)
            # Weight tying: lm_head.weight aliases embed.weight.
            # named_parameters() yields embed.weight first (canonical),
            # lm_head.weight is the alias. Forward only uses embed.
            self.lm_head = nn.Linear(dim, dim, bias=False)
            self.lm_head.weight = self.embed.weight

        def forward(self, x):
            return self.embed(x)

        def init_weights(self):
            with torch.no_grad():
                self.embed.weight.fill_(1.0)

    with torch.device("meta"):
        model = Model(dim)

    assert model.lm_head.weight is model.embed.weight

    batch_size = 512
    local_batch_size = batch_size // device_mesh_1d.size()
    x = DTensor.from_local(
        torch.rand(local_batch_size, dim, device="cuda"),
        device_mesh_1d,
        [Shard(0)],
    )
    parallel_mod = auto_parallel(
        model,
        device_mesh_1d,
        sample_inputs=(x,),
        out_shardings=(Shard(0),),
        compile=False,
    )

    parallel_mod.to_empty(device="cuda")
    parallel_mod.init_weights()

    expected = torch.ones(dim, dim, device="cuda")
    assert torch.equal(
        parallel_mod.get_parameter("embed.weight").full_tensor(), expected
    )


def test_aliased_buffers_both_used_in_forward(device_mesh_1d):
    """Test that aliased buffers work when both aliases are accessed in forward.

    When move_to_fake preserves aliasing, Dynamo sees both accesses as the same
    tensor and deduplicates the graph input. The compiled graph and the forward
    method must agree on the number of primals.
    """
    dim = 128

    class RoPE(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.register_buffer("cache", torch.zeros(dim), persistent=False)

        def forward(self, x):
            return x + self.cache

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            self.rope = RoPE(dim)
            self.register_buffer("freqs_cis", self.rope.cache, persistent=False)

        def forward(self, x):
            # Both aliases are accessed in the forward pass
            return self.linear(x) + self.freqs_cis + self.rope.cache

        def init_weights(self):
            with torch.no_grad():
                self.linear.weight.fill_(1.0)
                self.linear.bias.fill_(0.0)
            self.rope.cache = torch.arange(dim).float()
            self.freqs_cis = self.rope.cache

    with torch.device("meta"):
        model = Model(dim)

    assert model.freqs_cis is model.rope.cache

    batch_size = 512
    local_batch_size = batch_size // device_mesh_1d.size()
    x = DTensor.from_local(
        torch.rand(local_batch_size, dim, device="cuda"),
        device_mesh_1d,
        [Shard(0)],
    )
    parallel_mod = auto_parallel(
        model,
        device_mesh_1d,
        sample_inputs=(x,),
        out_shardings=(Shard(0),),
        compile=False,
    )

    parallel_mod.to_empty(device="cuda")
    parallel_mod.init_weights()

    # The key assertion is that tracing + running succeeds with both aliases
    # used in forward (no "too many values to unpack" error).
    inp = torch.rand(local_batch_size, dim, device="cuda")
    out = parallel_mod(inp)
    assert out.shape[-1] == dim


def test_aliased_parameters_both_used_in_forward(device_mesh_1d):
    """Test that aliased parameters work when both aliases are accessed in forward.

    This mirrors weight tying where embed.weight and lm_head.weight point to the
    same parameter and both are used during forward.
    """
    dim = 128

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.embed = nn.Linear(dim, dim, bias=False)
            self.lm_head = nn.Linear(dim, dim, bias=False)
            self.lm_head.weight = self.embed.weight

        def forward(self, x):
            # Both aliases are used in forward
            return self.embed(x) + self.lm_head(x)

        def init_weights(self):
            with torch.no_grad():
                self.embed.weight.fill_(1.0)

    with torch.device("meta"):
        model = Model(dim)

    assert model.lm_head.weight is model.embed.weight

    batch_size = 512
    local_batch_size = batch_size // device_mesh_1d.size()
    x = DTensor.from_local(
        torch.rand(local_batch_size, dim, device="cuda"),
        device_mesh_1d,
        [Shard(0)],
    )
    parallel_mod = auto_parallel(
        model,
        device_mesh_1d,
        sample_inputs=(x,),
        out_shardings=(Shard(0),),
        compile=False,
    )

    parallel_mod.to_empty(device="cuda")
    parallel_mod.init_weights()

    # The key assertion is that tracing + running succeeds with both aliases
    # used in forward (no "too many values to unpack" error).
    inp = torch.rand(local_batch_size, dim, device="cuda")
    out = parallel_mod(inp)
    assert out.shape[-1] == dim


def test_init_load_state_dict(device_mesh_1d):
    """Test that init_weights can use load_state_dict to initialize parameters."""
    dim = 128

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            self.register_buffer("buf", torch.empty(dim))

        def forward(self, x):
            return self.linear(x) + self.buf

        def init_weights(self):
            state = {
                "linear.weight": torch.ones(dim, dim) * 3.0,
                "linear.bias": torch.full((dim,), 7.0),
                "buf": torch.arange(dim, dtype=torch.float32),
            }
            self.load_state_dict(state)

    with torch.device("meta"):
        model = Model(dim)

    batch_size = 512
    local_batch_size = batch_size // device_mesh_1d.size()
    x = DTensor.from_local(
        torch.rand(local_batch_size, dim, device="cuda"),
        device_mesh_1d,
        [Shard(0)],
    )
    parallel_mod = auto_parallel(
        model,
        device_mesh_1d,
        sample_inputs=(x,),
        out_shardings=(Shard(0),),
        compile=False,
    )
    parallel_mod.to_empty(device="cuda")
    parallel_mod.init_weights()
    assert torch.equal(
        parallel_mod.get_parameter("linear.weight").full_tensor(),
        torch.full((dim, dim), 3.0, device="cuda"),
    )
    assert torch.equal(
        parallel_mod.get_parameter("linear.bias").full_tensor(),
        torch.full((dim,), 7.0, device="cuda"),
    )
    assert torch.equal(
        parallel_mod.get_buffer("buf").full_tensor(),
        torch.arange(dim, dtype=torch.float32, device="cuda"),
    )


def test_init_submodule_init_weights(device_mesh_1d):
    """Test that init_weights can call submodule init_weights methods."""
    dim = 128

    class MLP(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim)
            self.fc2 = nn.Linear(dim, dim)

        def forward(self, x):
            return self.fc2(self.fc1(x))

        def init_weights(self):
            with torch.no_grad():
                self.fc1.weight.fill_(1.0)
                self.fc1.bias.fill_(0.0)
                self.fc2.weight.fill_(2.0)
                self.fc2.bias.fill_(0.5)

    class Attention(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.qkv = nn.Linear(dim, dim * 3, bias=False)
            self.proj = nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            return self.proj(self.qkv(x)[:, :dim])

        def init_weights(self):
            nn.init.ones_(self.qkv.weight)
            nn.init.ones_(self.proj.weight)

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.mlp = MLP(dim)
            self.attn = Attention(dim)

        def forward(self, x):
            return self.mlp(x) + self.attn(x)

        def init_weights(self):
            self.mlp.init_weights()
            self.attn.init_weights()

    with torch.device("meta"):
        model = Model(dim)

    batch_size = 512
    local_batch_size = batch_size // device_mesh_1d.size()
    x = DTensor.from_local(
        torch.rand(local_batch_size, dim, device="cuda"),
        device_mesh_1d,
        [Shard(0)],
    )
    parallel_mod = auto_parallel(
        model,
        device_mesh_1d,
        sample_inputs=(x,),
        out_shardings=(Shard(0),),
        compile=False,
    )
    parallel_mod.to_empty(device="cuda")
    parallel_mod.init_weights()
    assert torch.equal(
        parallel_mod.get_parameter("mlp.fc1.weight").full_tensor(),
        torch.ones(dim, dim, device="cuda"),
    )
    assert torch.equal(
        parallel_mod.get_parameter("mlp.fc2.bias").full_tensor(),
        torch.full((dim,), 0.5, device="cuda"),
    )
    assert torch.equal(
        parallel_mod.get_parameter("attn.qkv.weight").full_tensor(),
        torch.ones(dim * 3, dim, device="cuda"),
    )
    assert torch.equal(
        parallel_mod.get_parameter("attn.proj.weight").full_tensor(),
        torch.ones(dim, dim, device="cuda"),
    )


def test_init_access_submodule_params(device_mesh_1d):
    """Test that init_weights can iterate over submodule parameters."""
    dim = 128

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear1 = nn.Linear(dim, dim)
            self.linear2 = nn.Linear(dim, dim)

        def forward(self, x):
            return self.linear2(self.linear1(x))

        def init_weights(self):
            for name, param in self.named_parameters():
                if "weight" in name:
                    with torch.no_grad():
                        param.fill_(1.0)
                elif "bias" in name:
                    with torch.no_grad():
                        param.fill_(0.0)

    with torch.device("meta"):
        model = Model(dim)

    batch_size = 512
    local_batch_size = batch_size // device_mesh_1d.size()
    x = DTensor.from_local(
        torch.rand(local_batch_size, dim, device="cuda"),
        device_mesh_1d,
        [Shard(0)],
    )
    parallel_mod = auto_parallel(
        model,
        device_mesh_1d,
        sample_inputs=(x,),
        out_shardings=(Shard(0),),
        compile=False,
    )
    parallel_mod.to_empty(device="cuda")
    parallel_mod.init_weights()
    assert torch.equal(
        parallel_mod.get_parameter("linear1.weight").full_tensor(),
        torch.ones(dim, dim, device="cuda"),
    )
    assert torch.equal(
        parallel_mod.get_parameter("linear2.weight").full_tensor(),
        torch.ones(dim, dim, device="cuda"),
    )
    assert torch.equal(
        parallel_mod.get_parameter("linear1.bias").full_tensor(),
        torch.zeros(dim, device="cuda"),
    )
    assert torch.equal(
        parallel_mod.get_parameter("linear2.bias").full_tensor(),
        torch.zeros(dim, device="cuda"),
    )


def test_init_submodule_load_state_dict(device_mesh_1d):
    """Test that init_weights can call load_state_dict on a submodule."""
    dim = 128

    class SubModule(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            self.register_buffer("scale", torch.empty(dim))

        def forward(self, x):
            return self.linear(x) * self.scale

        def init_weights(self):
            state = {
                "linear.weight": torch.eye(dim),
                "linear.bias": torch.zeros(dim),
                "scale": torch.ones(dim) * 5.0,
            }
            self.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.sub = SubModule(dim)
            self.out = nn.Linear(dim, dim)

        def forward(self, x):
            return self.out(self.sub(x))

        def init_weights(self):
            self.sub.init_weights()
            with torch.no_grad():
                self.out.weight.fill_(1.0)
                self.out.bias.fill_(0.0)

    with torch.device("meta"):
        model = Model(dim)

    batch_size = 512
    local_batch_size = batch_size // device_mesh_1d.size()
    x = DTensor.from_local(
        torch.rand(local_batch_size, dim, device="cuda"),
        device_mesh_1d,
        [Shard(0)],
    )
    parallel_mod = auto_parallel(
        model,
        device_mesh_1d,
        sample_inputs=(x,),
        out_shardings=(Shard(0),),
        compile=False,
    )
    parallel_mod.to_empty(device="cuda")
    parallel_mod.init_weights()
    assert torch.equal(
        parallel_mod.get_parameter("sub.linear.weight").full_tensor(),
        torch.eye(dim, device="cuda"),
    )
    assert torch.equal(
        parallel_mod.get_parameter("sub.linear.bias").full_tensor(),
        torch.zeros(dim, device="cuda"),
    )
    assert torch.equal(
        parallel_mod.get_buffer("sub.scale").full_tensor(),
        torch.full((dim,), 5.0, device="cuda"),
    )
    assert torch.equal(
        parallel_mod.get_parameter("out.weight").full_tensor(),
        torch.ones(dim, dim, device="cuda"),
    )


def test_init_optional_submodule(device_mesh_1d):
    """Test that init_weights can check for optional (None) submodules.

    Mirrors the torchtitan Decoder pattern where rope may or may not be present.
    When rope is None, the parallel model must still have the attribute so the
    None check doesn't raise AttributeError.
    """
    dim = 128

    class Model(nn.Module):
        def __init__(self, dim, use_rope=False):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            self.rope = nn.Linear(dim, dim) if use_rope else None

        def forward(self, x):
            return self.linear(x)

        def init_weights(self):
            with torch.no_grad():
                self.linear.weight.fill_(1.0)
                self.linear.bias.fill_(0.0)
            if self.rope is not None:
                with torch.no_grad():
                    self.rope.weight.fill_(2.0)

    with torch.device("meta"):
        model = Model(dim, use_rope=False)

    batch_size = 512
    local_batch_size = batch_size // device_mesh_1d.size()
    x = DTensor.from_local(
        torch.rand(local_batch_size, dim, device="cuda"),
        device_mesh_1d,
        [Shard(0)],
    )
    parallel_mod = auto_parallel(
        model,
        device_mesh_1d,
        sample_inputs=(x,),
        out_shardings=(Shard(0),),
        compile=False,
    )
    parallel_mod.to_empty(device="cuda")
    parallel_mod.init_weights()

    assert parallel_mod.rope is None
    assert torch.equal(
        parallel_mod.get_parameter("linear.weight").full_tensor(),
        torch.ones(dim, dim, device="cuda"),
    )


def test_init_data_assign_raises(device_mesh_1d):
    """Test that `buf.data = value` raises an error during init_weights.

    `tensor.data = value` is a C++ storage swap that bypasses __torch_dispatch__,
    so DTensor silently loses the assignment. The init_weights context must detect
    this and raise a clear error directing users to use `self.<name> = value` or
    `self.<name>.data[:] = value` instead.
    """
    dim = 128

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            self.register_buffer("buf", torch.empty(dim))

        def forward(self, x):
            return self.linear(x) + self.buf

        def init_weights(self):
            self.buf.data = torch.arange(dim, dtype=torch.float32)

    with torch.device("meta"):
        model = Model(dim)

    batch_size = 512
    local_batch_size = batch_size // device_mesh_1d.size()
    x = DTensor.from_local(
        torch.rand(local_batch_size, dim, device="cuda"),
        device_mesh_1d,
        [Shard(0)],
    )
    parallel_mod = auto_parallel(
        model,
        device_mesh_1d,
        sample_inputs=(x,),
        out_shardings=(Shard(0),),
        compile=False,
    )
    parallel_mod.to_empty(device="cuda")
    with pytest.raises(RuntimeError, match=r"Cannot use `.data = ...` on a DTensor"):
        parallel_mod.init_weights()
