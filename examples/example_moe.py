import torch
from torch import nn
from torch.nn import functional as F


class FFN(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super().__init__()
        self.w1 = nn.Linear(in_channels, inter_channels)
        self.w2 = nn.Linear(inter_channels, in_channels)
        self.w3 = nn.Linear(in_channels, inter_channels)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MOE(nn.Module):
    def __init__(self, in_channels, inter_channels, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            FFN(in_channels, inter_channels) for _ in range(num_experts)
        )

    def forward(self, x):
        assert x.ndim == 3
        shape = x.shape
        x = x.flatten(0, 1)
        indices = torch.randint(
            0, self.num_experts, (x.shape[0],), dtype=torch.int64, device=x.device
        )
        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            idx = torch.where(indices == i)
            output[idx] += expert(x[idx])
        return output.reshape(shape)


class BatchLinear(nn.Module):
    def __init__(self, in_channels, out_channels, num_experts):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_experts, out_channels, in_channels))
        self.bias = nn.Parameter(torch.randn(num_experts, out_channels))

    def forward(self, x):
        assert x.ndim == 3
        return x @ self.weight.transpose(-2, -1) + self.bias[:, None, :]


class BatchFFN(nn.Module):
    def __init__(self, in_channels, inter_channels, num_experts):
        super().__init__()
        self.w1 = BatchLinear(in_channels, inter_channels, num_experts)
        self.w2 = BatchLinear(inter_channels, in_channels, num_experts)
        self.w3 = BatchLinear(in_channels, inter_channels, num_experts)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MOEBatched(nn.Module):
    def __init__(self, in_channels, inter_channels, num_experts):
        super().__init__()
        self.num_experts = num_experts
        # TODO: need to fix case with bias, as the parameter memory constraint is not satisfied
        # because too many GPUs for the number of experts
        self.router = nn.Linear(in_channels, num_experts, bias=False)
        self.experts = BatchFFN(in_channels, inter_channels, num_experts)
        self.top_k = 4

    def init_weights(self):
        pass

    def forward(self, x):
        assert x.ndim == 3

        # route tokens to experts
        scores = self.router(x)

        # select topk experts following some criteria
        dim = -1
        scores = F.softmax(scores, dim=dim)
        # TODO: this is wrong, we need to do a sinkhorn here to ensure that the tokens are evenly distributed
        top_scores, selected_experts_indices = torch.topk(scores, k=self.top_k, dim=dim)
        idxs = selected_experts_indices.flatten(-2, -1).argsort(dim=-1, stable=True)
        top_scores = top_scores.flatten(-2, -1).gather(-1, idxs)
        idxs = idxs // self.top_k

        # route tokens for each expert
        xs = x.gather(-2, idxs[:, :, None].expand(-1, -1, x.shape[-1]))
        xs = xs.unflatten(1, (self.num_experts, -1))
        tokens_per_expert = xs.shape[2]
        xs = xs.permute(1, 0, 2, 3).flatten(1, 2)
        out = self.experts(xs)

        out = out.unflatten(1, (-1, tokens_per_expert))
        out = out.permute(1, 0, 2, 3).flatten(1, 2)

        out = out * top_scores[:, :, None]

        # TODO: add shared expert
        res = torch.zeros_like(x)
        res = res.scatter_add(-2, idxs[:, :, None].expand(-1, -1, x.shape[-1]), out)
        return res


class MOEBatchedDebug(nn.Module):
    def __init__(self, in_channels, inter_channels, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.experts = BatchFFN(in_channels, inter_channels, num_experts)
        self.top_k = 4

    def init_weights(self):
        pass

    def forward(self, x):
        assert x.ndim == 3
        shape = x.shape
        xs = (
            x.unflatten(1, (self.num_experts, -1))
            .permute(1, 0, 2, 3)
            .repeat(1, 1, self.top_k, 1)
            .flatten(1, 2)
        )
        out = self.experts(xs)
        out = out.unflatten(1, (-1, self.top_k)).sum(2)
        return out.reshape(shape)


"""

in_channels = 64
inter_channels = 128
num_experts = 8

bs = 8
seqlen = 64

x = torch.rand(bs, seqlen, in_channels).cuda()

m = MOE(in_channels, inter_channels, num_experts).cuda()
m2 = MOEBatched(in_channels, inter_channels, num_experts).cuda()

o = m(x)
o = m2(x)
"""

from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel.api import AutoParallel

world_size = 2048

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


in_channels = 7168
inter_channels = 2048
num_experts = 128

bs = 8 * mesh.shape[0] * mesh.shape[1]
seqlen = 2048 * 2


def input_fn():
    return torch.rand(bs, seqlen, in_channels).cuda()


def model_fn():
    return MOEBatched(in_channels, inter_channels, num_experts)
    # return MOEBatchedDebug(in_channels, inter_channels, num_experts)


with torch.device("meta"):
    model = model_fn()

with AutoParallel(model, input_fn, mesh) as autop:
    autop.add_parameter_memory_constraint(low=None, high=None)

    from torch.distributed.tensor.placement_types import Replicate, Shard

    x_sharding = (Shard(0), Shard(0))

    autop.add_input_constraints([x_sharding])
    autop.add_output_constraints([x_sharding])

    import time

    t = time.time()
    sharding_placement = autop.optimize_placement()
    print(f"Took {time.time() - t:.2f} s")
    parallel_mod = autop.apply_placement(sharding_placement)

# run weight init on our sharded DTensor params
parallel_mod.to_empty(device="cuda")
parallel_mod.init_weights()

# now let's run it
x = (
    torch.randn(
        (bs // mesh.shape[0] // mesh.shape[1], seqlen, in_channels),
        device=torch.device("cuda"),
    ),
)
out = parallel_mod(*x)
out.backward(torch.randn_like(out))
print("All good!")
