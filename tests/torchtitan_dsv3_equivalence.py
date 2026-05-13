# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""Compare AutoParallel's local_map DSv3 model with TorchTitan's DSv3 model.

This is a 4-GPU distributed integration check. It verifies that both models can
load the same full state exactly, then compares one forward/backward step.
"""

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel._testing.models.dsv3 import (
    DeepSeekV3Model as AutoParallelDeepSeekV3Model,
)
from autoparallel._testing.models.dsv3 import make_dsv3_config
from autoparallel.api import AutoParallel

WORLD_SIZE = 4
LOCAL_BATCH_SIZE = 2
SEQ_LEN = 1024
SEED = 123

DSV3_DIM = 256
DSV3_VOCAB_SIZE = 2048
DSV3_ROPE_DIM = 64
DSV3_NUM_LAYERS = 4
DSV3_NUM_DENSE_LAYERS = 0
DSV3_NUM_EXPERTS = 4
DSV3_NUM_SHARED_EXPERTS = 2
DSV3_ROUTER_TOP_K = 2
DSV3_DENSE_HIDDEN_DIM = 1024
DSV3_MOE_HIDDEN_DIM = 256

AP_DP_DEGREE = 2
AP_EP_DEGREE = 2
TT_DP_SHARD_DEGREE = 4
TT_EP_DEGREE = 2

NUMERICS_RTOL = 5e-4
NUMERICS_ATOL = 1e-4


def _import_torchtitan():
    try:
        import torchtitan  # noqa: F401
    except ModuleNotFoundError:
        candidates = (
            # AutoParallel CI runs this script from inside a cloned TorchTitan
            # checkout.
            Path.cwd(),
            # Local development often keeps TorchTitan next to AutoParallel.
            Path(__file__).resolve().parents[2] / "torchtitan",
        )
        for candidate in candidates:
            if (candidate / "torchtitan").exists():
                sys.path.insert(0, str(candidate))
                break
        import torchtitan  # noqa: F401


def _make_autoparallel_config(seq_len: int):
    return make_dsv3_config(
        num_experts=DSV3_NUM_EXPERTS,
        top_k=DSV3_ROUTER_TOP_K,
        n_layers=DSV3_NUM_LAYERS,
        n_dense_layers=DSV3_NUM_DENSE_LAYERS,
        max_seq_len=seq_len,
    )


def _build_torchtitan_config(seq_len: int):
    _import_torchtitan()

    from torchtitan.models.deepseek_v3 import (
        _EMBEDDING_INIT,
        _NORM_INIT,
        DeepSeekV3Model,
        Embedding,
        Linear,
        RMSNorm,
        RoPE,
        _build_dsv3_layers,
        _output_linear_init,
    )

    layers = _build_dsv3_layers(
        n_layers=DSV3_NUM_LAYERS,
        n_dense_layers=DSV3_NUM_DENSE_LAYERS,
        dim=DSV3_DIM,
        n_heads=16,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=DSV3_ROPE_DIM,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=DSV3_DENSE_HIDDEN_DIM,
        moe_hidden_dim=DSV3_MOE_HIDDEN_DIM,
        num_experts=DSV3_NUM_EXPERTS,
        num_shared_experts=DSV3_NUM_SHARED_EXPERTS,
        router_top_k=DSV3_ROUTER_TOP_K,
        router_score_func="softmax",
        score_before_experts=False,
        attn_backend="sdpa",
        moe_comm_backend="standard",
        non_blocking_capacity_factor=None,
    )
    for layer_config in layers:
        layer_config.attention.rope_max_seq_len = seq_len

    return DeepSeekV3Model.Config(
        vocab_size=DSV3_VOCAB_SIZE,
        dim=DSV3_DIM,
        tok_embeddings=Embedding.Config(
            num_embeddings=DSV3_VOCAB_SIZE,
            embedding_dim=DSV3_DIM,
            param_init=_EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=DSV3_DIM, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=DSV3_DIM,
            out_features=DSV3_VOCAB_SIZE,
            param_init=_output_linear_init(DSV3_DIM),
        ),
        rope=RoPE.Config(
            dim=DSV3_ROPE_DIM,
            max_seq_len=seq_len,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
        layers=layers,
    )


def _init_distributed() -> torch.device:
    if "WORLD_SIZE" not in os.environ:
        raise RuntimeError(f"run with torchrun --nproc-per-node {WORLD_SIZE}")
    if int(os.environ["WORLD_SIZE"]) != WORLD_SIZE:
        raise RuntimeError(f"expected exactly {WORLD_SIZE} ranks")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    return device


def _make_seed_state(seq_len: int):
    config = _make_autoparallel_config(seq_len)
    model = AutoParallelDeepSeekV3Model(config, compute_dtype=torch.bfloat16)
    model.init_weights(buffer_device=torch.device("cpu"), seed=SEED)
    return {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }


def _copy_full_tensor(target: torch.Tensor, source: torch.Tensor, device: torch.device):
    source = source.to(device=device, dtype=target.dtype)
    if isinstance(target, DTensor):
        replicated_source = DTensor.from_local(
            source,
            device_mesh=target.device_mesh,
            placements=(Replicate(),) * target.device_mesh.ndim,
        )
        source = replicated_source.redistribute(
            target.device_mesh,
            target.placements,
        )
    target.copy_(source)


def _load_full_state(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
    device: torch.device,
):
    targets: dict[str, torch.Tensor] = {}
    targets.update(model.named_parameters())
    targets.update(model.named_buffers())
    with torch.no_grad():
        for name, source in state.items():
            assert name in targets, f"missing state target {name}"
            _copy_full_tensor(targets[name], source, device)


def _loss_and_grad_norm(model, tokens: torch.Tensor, labels: torch.Tensor):
    logits = model(tokens)
    if isinstance(logits, DTensor):
        logits = logits.to_local()
    assert not torch.any(torch.isnan(logits)), "forward produced NaNs"

    local_loss_sum = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1).float(),
        labels.flatten(0, 1),
        reduction="sum",
    )
    global_loss_sum = local_loss_sum.detach().clone()
    dist.all_reduce(global_loss_sum, op=dist.ReduceOp.SUM)
    global_tokens = torch.tensor(
        labels.numel() * dist.get_world_size(),
        device=labels.device,
        dtype=torch.float32,
    )
    loss = local_loss_sum / global_tokens
    loss.backward()

    local_grad_sq = torch.zeros((), device=labels.device)
    for parameter in model.parameters():
        grad = parameter.grad
        if grad is None:
            continue
        if isinstance(grad, DTensor):
            grad = grad.to_local()
        local_grad_sq = local_grad_sq + grad.float().pow(2).sum()
    dist.all_reduce(local_grad_sq, op=dist.ReduceOp.SUM)
    return global_loss_sum / global_tokens, local_grad_sq.sqrt()


def _state_for_compare(model):
    state = {}
    for name, tensor in model.state_dict().items():
        if isinstance(tensor, DTensor):
            tensor = tensor.full_tensor()
        state[name] = tensor.detach().cpu()
    return state


def _make_inputs(device: torch.device, local_batch_size: int, seq_len: int):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank == 0:
        full_tokens = torch.randint(
            0,
            DSV3_VOCAB_SIZE,
            (local_batch_size * world_size, seq_len),
            device=device,
        )
    else:
        full_tokens = torch.empty(
            (local_batch_size * world_size, seq_len),
            dtype=torch.long,
            device=device,
        )
    dist.broadcast(full_tokens, src=0)
    return full_tokens.chunk(world_size, dim=0)[rank].contiguous()


def _run_autoparallel(
    device: torch.device,
    tokens: torch.Tensor,
    seed_state: dict[str, torch.Tensor],
):
    seq_len = tokens.shape[1]
    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda",
        (AP_DP_DEGREE, AP_EP_DEGREE),
        mesh_dim_names=("dp", "ep"),
    )
    config = _make_autoparallel_config(seq_len)
    global_batch_size = tokens.shape[0] * dist.get_world_size()

    with torch.device("meta"):
        model = AutoParallelDeepSeekV3Model(
            config,
            mesh=mesh,
            compute_dtype=torch.bfloat16,
        )

    def input_fn():
        return torch.randint(
            0,
            config.vocab_size,
            (global_batch_size, seq_len),
            device=device,
        )

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    with AutoParallel(
        model, input_fn, mesh, mp_policy=mp_policy, dynamic=True
    ) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)
        x_sharding = (Shard(0), Shard(0))
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([x_sharding])
        parallel_model = autop.apply_placement(
            autop.optimize_placement(verbose=False),
            decompose_after_sharding=False,
        )

    parallel_model.to_empty(device=device)
    _load_full_state(parallel_model, seed_state, device)
    loss, grad_norm = _loss_and_grad_norm(parallel_model, tokens, tokens)
    return loss, grad_norm, _state_for_compare(parallel_model)


def _run_torchtitan(
    device: torch.device,
    tokens: torch.Tensor,
    seed_state: dict[str, torch.Tensor],
):
    _import_torchtitan()

    from torchtitan.config import (
        ActivationCheckpointConfig,
        CompileConfig,
        ParallelismConfig,
        TrainingConfig,
    )
    from torchtitan.distributed import ParallelDims
    from torchtitan.models.deepseek_v3 import DeepSeekV3Model
    from torchtitan.models.deepseek_v3.parallelize import parallelize_deepseekv3

    seq_len = tokens.shape[1]
    parallel_dims = ParallelDims(
        dp_replicate=1,
        dp_shard=TT_DP_SHARD_DEGREE,
        cp=1,
        tp=1,
        pp=1,
        ep=TT_EP_DEGREE,
        world_size=WORLD_SIZE,
    )
    training = TrainingConfig(
        local_batch_size=tokens.shape[0],
        seq_len=seq_len,
        mixed_precision_param="bfloat16",
        mixed_precision_reduce="float32",
    )
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=TT_DP_SHARD_DEGREE,
        expert_parallel_degree=TT_EP_DEGREE,
        disable_loss_parallel=True,
    )
    compile_config = CompileConfig(enable=False)
    ac_config = ActivationCheckpointConfig(mode="none")
    config = _build_torchtitan_config(seq_len)
    config.update_from_config(
        trainer_config=type(
            "TrainerConfig",
            (),
            {
                "training": training,
                "parallelism": parallelism,
                "debug": type("DebugConfig", (), {"moe_force_load_balance": False})(),
            },
        )()
    )

    with torch.device("meta"):
        model = DeepSeekV3Model(config)
    parallelize_deepseekv3(
        model,
        parallel_dims=parallel_dims,
        training=training,
        parallelism=parallelism,
        compile_config=compile_config,
        ac_config=ac_config,
        dump_folder="/tmp",
    )
    model.to_empty(device=device)
    _load_full_state(model, seed_state, device)
    loss, grad_norm = _loss_and_grad_norm(model, tokens, tokens)
    return loss, grad_norm, _state_for_compare(model)


def main():
    device = _init_distributed()
    try:
        torch.manual_seed(SEED)
        tokens = _make_inputs(
            device,
            local_batch_size=LOCAL_BATCH_SIZE,
            seq_len=SEQ_LEN,
        )
        seed_state = _make_seed_state(seq_len=tokens.shape[1])

        ap_loss, ap_grad_norm, ap_state = _run_autoparallel(
            device,
            tokens,
            seed_state,
        )
        tt_loss, tt_grad_norm, tt_state = _run_torchtitan(
            device,
            tokens,
            seed_state,
        )

        if dist.get_rank() == 0:
            for name in sorted(set(ap_state) | set(tt_state)):
                if name not in ap_state or name not in tt_state:
                    raise AssertionError(
                        f"state missing {name}: "
                        f"ap={name in ap_state}, tt={name in tt_state}"
                    )
                if not torch.equal(ap_state[name], tt_state[name]):
                    diff = (ap_state[name].float() - tt_state[name].float()).abs().max()
                    raise AssertionError(
                        f"state differs for {name} with max diff {diff.item()}"
                    )

        # The full states above must match exactly. The optimized AP graph may
        # still choose sharded matmuls/reductions that differ from TorchTitan's
        # FSDP-only arithmetic order, so loss and grad norm are close checks.
        torch.testing.assert_close(
            ap_loss,
            tt_loss,
            rtol=NUMERICS_RTOL,
            atol=NUMERICS_ATOL,
        )
        torch.testing.assert_close(
            ap_grad_norm,
            tt_grad_norm,
            rtol=NUMERICS_RTOL,
            atol=NUMERICS_ATOL,
        )
        if dist.get_rank() == 0:
            print(
                "AutoParallel and TorchTitan DSv3 numerics are close: "
                f"loss={ap_loss.item():.6f}, grad_norm={ap_grad_norm.item():.6f}"
            )
    finally:
        dist.barrier(device_ids=[device.index])
        torch.cuda.synchronize(device)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
