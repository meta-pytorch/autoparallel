# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Literal

import moe_ops

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class MoEArgs:
    num_experts: int = 8
    num_shared_experts: int = 1

    # router
    score_func: Literal["softmax", "sigmoid"] = "sigmoid"
    route_norm: bool = False
    route_scale: float = 1.0
    score_before_experts: bool = True

    # token-choice
    top_k: int = 1
    use_grouped_mm: bool = True  # grouped mm or for-loop for the experts computation
    load_balance_coeff: float | None = 1e-3


# can be used as dense FFN layer or shared experts in MoE layers
class FeedForward(nn.Module):
    """
    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(dim, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(hidden_dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3
        # shape (ob, ib*slen, dim)
        h = F.silu(torch.ops.autoparallel.batched_mm(x, self.w1.transpose(-2, -1)))
        h = h * torch.ops.autoparallel.batched_mm(x, self.w3.transpose(-2, -1))
        out = torch.ops.autoparallel.batched_mm(h, self.w2.transpose(-2, -1))
        return out

    def init_weights(self, init_std: float = 0.02):
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=init_std)


class GroupedExperts(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        assert x.dim() == 3
        assert num_tokens_per_expert.dim() == 2
        offsets = torch.cumsum(num_tokens_per_expert, dim=1, dtype=torch.int32)
        # grouped mm between a 3D tensor and a 3D tensor and 2D offsets
        h = F.silu(
            torch.ops.autoparallel.batched_grouped_mm(
                x.bfloat16(), self.w1.bfloat16().transpose(-2, -1), offs=offsets
            )
        )
        h = h * torch.ops.autoparallel.batched_grouped_mm(
            x.bfloat16(), self.w3.bfloat16().transpose(-2, -1), offs=offsets
        )
        out = torch.ops.autoparallel.batched_grouped_mm(
            h, self.w2.bfloat16().transpose(-2, -1), offs=offsets
        ).type_as(x)
        return out

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=init_std)


class TokenChoiceTopKRouter(nn.Module):
    """This class implements token-choice routing. In token-choice top-K routing, each token is
        routed to top K experts based on the router scores.

    Args:
        dim (int): Dimension of input tokens.
        num_experts (int): Number of experts in each moe layer.
        top_k (int): Number of experts each token will be routed to in token-choice routing.
        score_func (Literal["softmax", "sigmoid"]): Whether to use sigmoid or softmax for router scores.
        route_norm (bool): Whether to normalize the routing scores when using sigmoid.
        route_scale (float): Scaling factor applied to the routing scores.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        score_func: Literal["softmax", "sigmoid"],
        route_norm: bool,
        route_scale: float,
    ):
        super().__init__()
        self.gate = nn.Parameter(torch.empty(dim, num_experts))
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale

    def forward(
        self, x: torch.Tensor, expert_bias: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(ob, ib*slen, dim)``.
            expert_bias (torch.Tensor | None, optional): Optional bias tensor for experts with shape ``(num_experts,)``.
                Used for load balancing. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - top_scores (torch.Tensor):
                    Routing scores for selected experts with shape ``(ob, ib*slen, top_k)``.
                - selected_experts_indices (torch.Tensor):
                    Expert indices selected for each token with shape ``(ob, ib*slen, top_k)``.
                - num_tokens_per_expert (torch.Tensor):
                    Number of tokens assigned to each expert with shape ``(ob, num_experts,)``.
        """
        assert x.dim() == 3
        if expert_bias is not None:
            assert expert_bias.dim() == 1
        # scores shape (ob, ib*slen, num_experts)
        scores = torch.ops.autoparallel.batched_mm(x, self.gate.transpose(-2, -1))

        # By default, sigmoid or softmax is performed in float32 to avoid loss explosion
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores.to(torch.float32))
        elif self.score_func == "softmax":
            scores = F.softmax(scores.to(torch.float32), dim=2)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_function}")

        # top scores shape (ob, ib*slen, top_k)
        # selected_experts_indices shape (ob, ib*slen, top_k)
        # NOTE: The expert_bias is only used for routing. The gating value
        #       top_scores is still derived from the original scores.
        if expert_bias is not None:
            _, selected_experts_indices = torch.topk(
                scores + expert_bias, k=self.top_k, dim=2
            )
            top_scores = scores.gather(dim=2, index=selected_experts_indices)
        else:
            top_scores, selected_experts_indices = torch.topk(
                scores, k=self.top_k, dim=2
            )

        if self.score_func == "sigmoid" and self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator
        top_scores = top_scores * self.route_scale

        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        # num_tokens_per_expert has shape (ob, num_experts,)
        num_tokens_per_expert = torch.ops.autoparallel.batched_histc(
            selected_experts_indices.reshape(selected_experts_indices.shape[0], -1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        return top_scores, selected_experts_indices, num_tokens_per_expert

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.gate, mean=0.0, std=init_std)


class MoE(nn.Module):
    def __init__(self, moe_args: MoEArgs, dim: int, hidden_dim: int):
        super().__init__()

        num_experts = moe_args.num_experts
        self.experts = GroupedExperts(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
        )
        self.top_k = moe_args.top_k
        self.router = TokenChoiceTopKRouter(
            dim=dim,
            num_experts=num_experts,
            top_k=moe_args.top_k,
            score_func=moe_args.score_func,
            route_norm=moe_args.route_norm,
            route_scale=moe_args.route_scale,
        )
        self.shared_experts = (
            FeedForward(dim=dim, hidden_dim=hidden_dim * moe_args.num_shared_experts)
            if moe_args.num_shared_experts > 0
            else None
        )
        self.score_before_experts = moe_args.score_before_experts

        # define fields for auxiliary-loss-free load balancing (https://arxiv.org/abs/2408.15664)
        # NOTE: tokens_per_expert is accumulated in the model forward pass.
        #       expert_bias is updated outside the model in an optimzer step pre hook
        #       to work with gradient accumulation.
        self.load_balance_coeff = moe_args.load_balance_coeff
        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff > 0.0
            self.register_buffer(
                "expert_bias",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,
            )
            self.register_buffer(
                "tokens_per_expert",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.expert_bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(ob, ib, slen, dim)``.

        Returns:
            out (torch.Tensor): Output tensor with shape ``(ob, ib, slen, dim)``.
        """
        ob, ib, slen, dim = x.shape
        # x shape (ob, ib*slen, dim)
        x = x.reshape(ob, ib * slen, dim)

        # top_scores shape (ob, ib*slen, top_k,)
        # selected_experts_indices shape (ob, ib*slen, top_k,)
        # num_tokens_per_expert shape (ob, num_experts,)
        (
            top_scores,
            selected_experts_indices,
            num_tokens_per_expert,
        ) = self.router(x, self.expert_bias)

        # tokens_per_expert will be used to update the expert bias for load balancing.
        # TODO: Activation Checkpointing has the side effect of double counting tokens_per_expert --
        #       first in the forward pass, and then in the backward pass. However, this has no
        #       effect on the expert bias update thanks to the torch.sign() operator.
        if self.load_balance_coeff is not None:
            with torch.no_grad():
                self.tokens_per_expert.add_(torch.sum(num_tokens_per_expert, dim=0))
        # routed_input shape (ob, padded_length, dim) - padded and permuted
        # top_scores_experts_sorted shape (ob, ib*slen*top_k,)
        # token_indices_experts_sorted shape (ob, ib*slen*top_k,)
        # batch_permuted_indices shape (ob, padded_length) - permutation indices
        # routed_input_shape_before_permute list[torch.Size] - original shapes before permutation
        (
            padded_permuted_routed_input,
            tokens_per_expert,
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            permuted_indices,
            routed_input_shape_before_permute,
            input_splits,
            output_splits,
        ) = torch.ops.autoparallel.token_dispatch(
            x,
            top_scores,
            selected_experts_indices,
            num_tokens_per_expert,
            self.router.num_experts,
            self.top_k,
            self.score_before_experts,
            None,  # mesh
        )

        # shape (ob, padded_length, dim) - experts process padded/permuted input
        padded_permuted_routed_output = self.experts(
            padded_permuted_routed_input, tokens_per_expert
        )

        # shared expert
        if self.shared_experts is not None:
            out = self.shared_experts(x)
        else:
            out = torch.zeros_like(x)

        out = torch.ops.autoparallel.token_combine(
            out,
            padded_permuted_routed_output,
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            permuted_indices,
            routed_input_shape_before_permute,
            self.score_before_experts,
            input_splits,
            output_splits,
            None,  # mesh
        )
        out = out.reshape(ob, ib, slen, dim)
        return out

    def init_weights(
        self,
        init_std: float,
        buffer_device: torch.device,
    ):
        self.experts.init_weights(init_std)
        self.router.init_weights(init_std)
        if self.shared_experts is not None:
            self.shared_experts.init_weights(init_std)

        if self.load_balance_coeff is not None:
            with torch.device(buffer_device):
                self.expert_bias = torch.zeros(
                    self.experts.num_experts, dtype=torch.float32
                )
                self.tokens_per_expert = torch.zeros(
                    self.experts.num_experts, dtype=torch.float32
                )


if __name__ == "__main__":

    from autoparallel.api import AutoParallel
    from torch.distributed.fsdp import MixedPrecisionPolicy
    from torch.distributed.tensor.placement_types import Replicate, Shard
    from torch.testing._internal.distributed.fake_pg import FakeStore

    # Model configuration
    world_size = 256
    fake_store = FakeStore()
    torch.distributed.init_process_group(
        "fake", store=fake_store, rank=0, world_size=world_size
    )
    ep_degree = 32
    dp_degree = world_size // ep_degree
    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda",
        (dp_degree, ep_degree),
        mesh_dim_names=(
            "dp",
            "ep",
        ),
    )

    bs = 8 * world_size
    ob = dp_degree
    ib = bs // ob
    seq_len = 256
    dim = 512
    hidden_dim = 2048
    num_experts = 128
    top_k = 1

    def input_fn():
        print(f"global input shape: {(ob, ib, seq_len, dim)}")
        return torch.rand(ob, ib, seq_len, dim, device="cuda")

    # parallelize the model
    with torch.device("meta"):
        moe_args = MoEArgs(
            num_experts=num_experts,
            num_shared_experts=1,
            score_func="softmax",
            route_norm=False,
            route_scale=1.0,
            score_before_experts=True,
            top_k=top_k,
            use_grouped_mm=True,
            load_balance_coeff=1e-3,
        )
        model = MoE(
            moe_args=moe_args,
            dim=dim,
            hidden_dim=hidden_dim,
        )

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )

    with AutoParallel(model, input_fn, mesh, mp_policy, compile=True) as autop:
        assert any(n.meta.get("nn_module_stack") for n in autop.gm.graph.nodes)
        assert any(n.meta.get("fwd_nn_module_stack") for n in autop.gm.graph.nodes)
        autop.add_parameter_memory_constraint(low=None, high=None)

        x_sharding = (
            Shard(0),
            Shard(1),
        ) + (
            Replicate(),
        ) * (mesh.ndim - 2)

        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([x_sharding])

        sharding_placement = autop.optimize_placement()

        # AutoParallel produces a module with meta-DTensor parameters that need to be initialized
        parallel_mod = autop.apply_placement(sharding_placement)

    parallel_mod.to_empty(device="cuda")
    parallel_mod.init_weights(init_std=0.02, buffer_device="cuda")

    # now let's run it
    x = (torch.rand(bs // mesh.shape[0], 1, seq_len, dim, device="cuda"),)
    out = parallel_mod(*x)
    out.backward(torch.randn_like(out))

    print("All good!")
