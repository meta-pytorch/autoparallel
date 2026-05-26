# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import time

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn_func
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Shard

from autoparallel._testing.models.qwen3 import (
    Qwen3ModelArgs,
    Transformer,
    qwen3_30b_a3b_args,
    qwen3_235b_a22b_args,
    qwen3_moe_debug_args,
)
from autoparallel.api import AutoParallel
from autoparallel.compile import autoparallel_backend


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a real Qwen3 MoE AutoParallel training sanity check."
    )
    parser.add_argument(
        "--flavor",
        choices=("moe-tiny", "moe-debug", "30b-a3b", "235b-a22b"),
        default="30b-a3b",
        help="Qwen3 MoE model size. Defaults to the real Qwen3-30B-A3B model.",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=4,
        help="Global batch size across data-parallel ranks.",
    )
    parser.add_argument(
        "--microbatch-size",
        type=int,
        default=1,
        help="Per-rank input microbatch size before EP all-gather inside the model.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=8192,
        help="Sequence length. Defaults to 8192 for the 4xH100 sanity run.",
    )
    parser.add_argument(
        "--dp-degree",
        type=int,
        default=2,
        help="Data-parallel mesh degree.",
    )
    parser.add_argument(
        "--ep-degree",
        type=int,
        default=2,
        help="Expert-parallel mesh degree.",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=30,
        help="Number of optimizer steps.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--optimizer",
        choices=("adamw", "sgd", "none"),
        default="adamw",
        help="Optimizer to use after backward. Use sgd/none for large-model memory smoke runs.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping max norm.",
    )
    parser.add_argument(
        "--loss-chunk-size",
        type=int,
        default=512,
        help=(
            "Sequence chunk size for vocab-parallel cross entropy. "
            "Keeps the 8192-token real-model run from materializing full-size "
            "float logits and exp buffers at once."
        ),
    )
    parser.add_argument(
        "--skip-loss-improvement-check",
        action="store_true",
        help="Only require finite forward/backward/optimizer steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for model initialization and synthetic data generation.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the placed module with the AutoParallel backend before training.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full AutoParallel optimizer log.",
    )
    return parser.parse_args()


def make_model_args(flavor: str, seq_len: int | None) -> Qwen3ModelArgs:
    if flavor == "moe-tiny":
        max_seq_len = 512 if seq_len is None else seq_len
        return Qwen3ModelArgs(
            dim=64,
            n_layers=1,
            n_heads=4,
            n_kv_heads=2,
            head_dim=16,
            hidden_dim=128,
            vocab_size=128,
            max_seq_len=max_seq_len,
            moe_enabled=True,
            moe_hidden_dim=32,
            num_experts=8,
            top_k=2,
            route_norm=True,
            score_before_experts=False,
            moe_axis_name="ep",
        )
    overrides = {"moe_axis_name": "ep"}
    if seq_len is not None:
        overrides["max_seq_len"] = seq_len
    if flavor == "moe-debug":
        return qwen3_moe_debug_args(**overrides)
    if flavor == "30b-a3b":
        return qwen3_30b_a3b_args(**overrides)
    if flavor == "235b-a22b":
        return qwen3_235b_a22b_args(**overrides)
    raise ValueError(f"Unknown Qwen3 MoE flavor: {flavor}")


def init_distributed(args):
    if "WORLD_SIZE" not in os.environ or "LOCAL_RANK" not in os.environ:
        raise RuntimeError(
            "Run this example with torchrun, e.g. "
            "torchrun --standalone --nproc-per-node 4 "
            "examples/example_sanity_check_qwen3_moe.py"
        )

    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    expected_world_size = args.dp_degree * args.ep_degree
    if world_size != expected_world_size:
        raise ValueError(
            f"WORLD_SIZE ({world_size}) must equal dp-degree * ep-degree "
            f"({args.dp_degree} * {args.ep_degree} = {expected_world_size})."
        )
    if args.global_batch_size % args.dp_degree != 0:
        raise ValueError(
            f"global-batch-size ({args.global_batch_size}) must be divisible by "
            f"dp-degree ({args.dp_degree})."
        )

    local_dp_batch_size = args.global_batch_size // args.dp_degree
    local_dp_microbatch = args.microbatch_size * args.ep_degree
    if local_dp_batch_size % local_dp_microbatch != 0:
        raise ValueError(
            f"local DP batch size ({local_dp_batch_size}) must be divisible by "
            f"microbatch-size * ep-degree "
            f"({args.microbatch_size} * {args.ep_degree} = {local_dp_microbatch})."
        )

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda",
        (args.dp_degree, args.ep_degree),
        mesh_dim_names=("dp", "ep"),
    )
    return device, mesh


def make_local_tokens(args, mesh, device, vocab_size: int) -> torch.Tensor:
    coordinate = mesh.get_coordinate()
    if coordinate is None:
        raise RuntimeError("DeviceMesh coordinate is unavailable on this rank.")
    dp_rank, _ep_rank = coordinate
    local_dp_batch_size = args.global_batch_size // args.dp_degree

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    tokens = torch.randint(
        0,
        vocab_size,
        (args.global_batch_size, args.seq_len + 1),
        generator=generator,
        dtype=torch.long,
    )

    start = dp_rank * local_dp_batch_size
    stop = start + local_dp_batch_size
    return tokens[start:stop].to(device, non_blocking=True)


def vocab_parallel_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    vocab_size: int,
    vocab_group,
    vocab_rank: int,
    vocab_degree: int,
    global_token_count: int,
) -> torch.Tensor:
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} is incompatible with "
            f"labels shape {tuple(labels.shape)}."
        )

    local_vocab_size = logits.shape[-1]
    vocab_start = vocab_rank * local_vocab_size
    vocab_stop = vocab_start + local_vocab_size
    if vocab_rank == vocab_degree - 1:
        vocab_stop = vocab_size

    logits = logits.float()
    local_max = logits.amax(dim=-1)
    with torch.no_grad():
        global_max = local_max.detach().clone()
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=vocab_group)

    shifted_logits = logits - global_max.unsqueeze(-1)
    local_exp_sum = shifted_logits.exp().sum(dim=-1)
    global_exp_sum = dist_nn_func.all_reduce(
        local_exp_sum,
        op=dist.ReduceOp.SUM,
        group=vocab_group,
    )

    target_mask = (labels >= vocab_start) & (labels < vocab_stop)
    local_target = torch.zeros_like(labels, dtype=torch.long)
    local_target[target_mask] = labels[target_mask] - vocab_start
    local_target_logits = logits.gather(-1, local_target.unsqueeze(-1)).squeeze(-1)
    local_target_logits = local_target_logits * target_mask.to(logits.dtype)
    target_logits = dist_nn_func.all_reduce(
        local_target_logits,
        op=dist.ReduceOp.SUM,
        group=vocab_group,
    )

    loss_sum = (global_exp_sum.log() + global_max - target_logits).sum()
    return loss_sum / (global_token_count * vocab_degree)


def chunk_ranges(size: int, chunk_size: int):
    if chunk_size <= 0:
        yield 0, size
        return
    for start in range(0, size, chunk_size):
        yield start, min(start + chunk_size, size)


def print_rank0(message: str) -> None:
    if dist.get_rank() == 0:
        print(message, flush=True)


def print_cuda_memory(stage: str, device: torch.device) -> None:
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    max_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    print_rank0(
        f"{stage}: cuda allocated={allocated:.2f}GiB "
        f"reserved={reserved:.2f}GiB max_reserved={max_reserved:.2f}GiB"
    )


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG)

    device, mesh = init_distributed(args)
    ep_group = mesh.get_group("ep")
    ep_rank = mesh.get_local_rank("ep")
    local_dp_batch_size = args.global_batch_size // args.dp_degree
    local_dp_microbatch = args.microbatch_size * args.ep_degree
    gradient_accumulation_steps = local_dp_batch_size // local_dp_microbatch

    torch.manual_seed(args.seed)
    model_args = make_model_args(args.flavor, args.seq_len)
    if args.seq_len is None:
        args.seq_len = model_args.max_seq_len
    if model_args.num_experts % args.ep_degree != 0:
        raise ValueError(
            f"num_experts ({model_args.num_experts}) must be divisible by "
            f"ep-degree ({args.ep_degree})."
        )
    trace_global_batch_size = args.microbatch_size * args.dp_degree * args.ep_degree

    with torch.device("meta"):
        model = Transformer(model_args, mesh=mesh, moe_axis_name="ep")

    def input_fn():
        return torch.randint(
            0,
            model_args.vocab_size,
            (trace_global_batch_size, args.seq_len),
            device=device,
        )

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    print_rank0(
        f"Qwen3 {args.flavor} sanity check: "
        f"mesh=(dp={args.dp_degree}, ep={args.ep_degree}), "
        f"global_batch={args.global_batch_size}, "
        f"local_dp_batch={local_dp_batch_size}, "
        f"per_rank_microbatch={args.microbatch_size}, "
        f"local_dp_microbatch={local_dp_microbatch}, "
        f"grad_accum={gradient_accumulation_steps}, "
        f"trace_global_batch={trace_global_batch_size}, "
        f"seq_len={args.seq_len}, "
        f"loss_chunk_size={args.loss_chunk_size}, "
        f"optimizer={args.optimizer}"
    )

    t0 = time.time()
    with AutoParallel(
        model,
        input_fn,
        mesh,
        mp_policy,
        dynamic=True,
        repeated_subgraphs=True,
    ) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)
        autop.add_input_constraints([(Shard(0), Shard(0))])
        autop.add_output_constraints([(Shard(0), Shard(2))])
        sharding_placement = autop.optimize_placement(verbose=args.verbose)
        parallel_mod = autop.apply_placement(sharding_placement)

    print_rank0(f"Tracing + optimization took {time.time() - t0:.1f}s")
    print_cuda_memory("after AutoParallel", device)

    parallel_mod.to_empty(device=device)
    print_cuda_memory("after to_empty", device)
    parallel_mod.init_weights(buffer_device=device, seed=args.seed)  # type: ignore[operator]
    print_cuda_memory("after init_weights", device)

    if args.compile:
        parallel_mod = torch.compile(parallel_mod, backend=autoparallel_backend())

    batch = make_local_tokens(args, mesh, device, model_args.vocab_size)
    inputs = batch[:, :-1].contiguous()
    labels = batch[:, 1:].contiguous()

    ep_coordinate = mesh.get_coordinate()[1]
    input_microbatches = []
    label_microbatches = []
    for start in range(0, local_dp_batch_size, local_dp_microbatch):
        stop = start + local_dp_microbatch
        input_block = inputs[start:stop]
        input_start = ep_coordinate * args.microbatch_size
        input_stop = input_start + args.microbatch_size
        input_microbatches.append(input_block[input_start:input_stop].contiguous())
        label_microbatches.append(labels[start:stop].contiguous())

    global_token_count = args.global_batch_size * args.seq_len
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(parallel_mod.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(parallel_mod.parameters(), lr=args.lr)
    else:
        optimizer = None

    try:
        losses: list[float] = []
        for step in range(args.train_steps):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            else:
                parallel_mod.zero_grad(set_to_none=True)
            step_loss = torch.zeros((), device=device)
            for micro_inputs, micro_labels in zip(
                input_microbatches, label_microbatches
            ):
                logits = parallel_mod(micro_inputs)

                seq_ranges = list(chunk_ranges(logits.shape[1], args.loss_chunk_size))
                for chunk_idx, (seq_start, seq_stop) in enumerate(seq_ranges):
                    logits_chunk = logits[:, seq_start:seq_stop]
                    labels_chunk = micro_labels[:, seq_start:seq_stop]
                    loss = vocab_parallel_cross_entropy(
                        logits_chunk,
                        labels_chunk,
                        vocab_size=model_args.vocab_size,
                        vocab_group=ep_group,
                        vocab_rank=ep_rank,
                        vocab_degree=args.ep_degree,
                        global_token_count=global_token_count,
                    )
                    if torch.any(torch.isnan(loss)):
                        raise RuntimeError("Found NaNs in Qwen3 MoE training loss.")

                    retain_graph = chunk_idx != len(seq_ranges) - 1
                    loss.backward(retain_graph=retain_graph)
                    step_loss = step_loss + loss.detach()

            torch.nn.utils.clip_grad_norm_(
                parallel_mod.parameters(), args.max_grad_norm
            )
            if optimizer is not None:
                optimizer.step()

            with torch.no_grad():
                logged_loss = step_loss.clone()
                dist.all_reduce(logged_loss, op=dist.ReduceOp.SUM)
                loss_value = float(logged_loss.item())
            losses.append(loss_value)
            print_rank0(f"step={step:03d} loss={loss_value:.6f}")
            print_cuda_memory(f"after step {step:03d}", device)

        if (
            not args.skip_loss_improvement_check
            and len(losses) > 1
            and losses[-1] >= losses[0]
        ):
            raise RuntimeError(
                f"Qwen3 MoE training loss did not improve: "
                f"initial={losses[0]:.6f}, final={losses[-1]:.6f}"
            )

        if len(losses) > 1:
            print_rank0(
                f"Loss improved: initial={losses[0]:.6f}, final={losses[-1]:.6f}"
            )
        dist.barrier(device_ids=[device.index])
        torch.cuda.synchronize(device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
