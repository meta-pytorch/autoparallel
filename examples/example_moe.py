import torch
from torch import nn
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.nn import functional as F
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel.api import AutoParallel
from autoparallel.propagation_rules import register_opschema_rule


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


def _init_approximate_solution(scores, top_k, variables):
    top_idxs = scores.topk(top_k, dim=-1).indices.tolist()
    for token in variables.keys():
        for expert in variables[token].keys():
            token_id = int(token.split("_")[1])
            expert_id = int(expert.split("_")[1])
            initial_value = 1 if expert_id in top_idxs[token_id] else 0
            variables[token][expert].setInitialValue(initial_value)


def _assign_tokens_per_expert_2d(scores, top_k, init_sol=True, time_limit=1.0):
    import pulp

    num_per_expert = scores.shape[0] // scores.shape[1] * top_k
    prob = pulp.LpProblem("TokenExpertAssignment", pulp.LpMaximize)
    experts = ["expert_{}".format(i) for i in range(scores.shape[1])]
    tokens = ["token_{}".format(i) for i in range(scores.shape[0])]
    scores_dict = pulp.makeDict([tokens, experts], scores.tolist(), 0)
    variables = pulp.LpVariable.dicts("var", (tokens, experts), cat=pulp.LpBinary)
    for token in tokens:
        prob += pulp.lpSum([variables[token][expert] for expert in experts]) == top_k
    for expert in experts:
        prob += (
            pulp.lpSum([variables[token][expert] for token in tokens]) == num_per_expert
        )

    if init_sol:
        _init_approximate_solution(scores, top_k, variables)
    prob += pulp.lpSum(
        [
            variables[token][expert] * scores_dict[token][expert]
            for token in tokens
            for expert in experts
        ]
    )
    verbose = False
    solver = pulp.PULP_CBC_CMD(msg=verbose, warmStart=init_sol, timeLimit=time_limit)
    prob.solve(solver)
    res = [[variables[token][expert].value() for expert in experts] for token in tokens]
    res = [[i for i, v in enumerate(r) if v == 1] for r in res]
    return torch.tensor(res, dtype=torch.int32, device=scores.device)


@torch.library.custom_op("autoparallel::assign_tokens_to_experts", mutates_args=())
def assign_tokens_to_experts(scores: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    MILP formulation of the token assignment problem, guarantees that each token
    is assigned to exactly top_k experts, and every expert is assigned to exactly
    the same number of tokens.

    NOTE: This performs a GPU->CPU transfer! Need to implement a working version of
    Sinkhorn-Knopp on GPU to avoid this.

    NOTE: The MILP solver is *slow* and can take a long time to converge.
    """
    shape = scores.shape[:-1]
    scores_flat = scores.flatten(0, -3)
    res = []
    for score in scores_flat:
        assert score.ndim == 2, f"score must be 2D, got {score.shape}"
        res.append(_assign_tokens_per_expert_2d(score, top_k))
    return torch.stack(res, dim=0).reshape(shape + (top_k,))


@assign_tokens_to_experts.register_fake
def _(scores, top_k):
    return torch.empty(
        tuple(scores.shape[:-1]) + (top_k,), device=scores.device, dtype=torch.int32
    )


@register_opschema_rule(torch.ops.autoparallel.assign_tokens_to_experts.default)
def _(mesh, op_schema):
    from torch.distributed.tensor._ops.utils import expand_to_full_mesh_op_strategy

    mat1_strategy = op_schema.args_schema[0]

    assert len(mat1_strategy.shape) == 3

    single_mesh_dim_strategies = []

    # placement list stores placements of [output, mat1]
    # first we always have replicate all for inputs and output
    single_mesh_dim_strategies.append([Replicate(), Replicate()])
    single_mesh_dim_strategies.append([Shard(0), Shard(0)])

    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=1
    )


# scores = torch.rand(1, 8192, 128, device="cuda")
# for i in range(0, 32):
#     k = 8192 // 128 * 4
#     scores[:, i * k: (i + i) * k, i * 4 : (i + 1) * 4] += 1
# r = assign_tokens_to_experts(scores.cpu(), 4)
# r = _assign_tokens_per_expert_2d(scores[0], 4)
# r = torch.ops.autoparallel.assign_tokens_to_experts(scores, 4)
# from IPython import embed; embed(); exit()


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
        # top_scores, selected_experts_indices = torch.topk(scores, k=self.top_k, dim=dim)
        selected_experts_indices = torch.ops.autoparallel.assign_tokens_to_experts(
            scores, self.top_k
        )
        top_scores = scores.gather(dim, selected_experts_indices)
        idxs = selected_experts_indices.flatten(-2, -1).argsort(dim=-1, stable=True)
        top_scores = top_scores.flatten(-2, -1).gather(-1, idxs)
        idxs = idxs // self.top_k

        # route tokens for each expert
        xs = x.gather(-2, idxs[:, :, None].expand(-1, -1, x.shape[-1]))
        # this assumes the experts are balanced
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
seqlen = 2048 * 2 * 2


def input_fn():
    return torch.rand(bs, seqlen, in_channels).cuda()


def model_fn():
    return MOEBatched(in_channels, inter_channels, num_experts)
    # return MOEBatchedDebug(in_channels, inter_channels, num_experts)


with torch.device("meta"):
    model = model_fn()

with AutoParallel(model, input_fn, mesh) as autop:
    autop.add_parameter_memory_constraint(low=None, high=None)

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
