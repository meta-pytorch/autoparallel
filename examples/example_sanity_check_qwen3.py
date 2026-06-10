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
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel._testing.models.qwen3 import Transformer, qwen3_8b_args
from autoparallel.api import AutoParallel
from autoparallel.compile import autoparallel_backend


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a real Qwen3 8B AutoParallel training sanity check."
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=16,
        help="Global batch size across data-parallel ranks.",
    )
    parser.add_argument(
        "--microbatch-size",
        type=int,
        default=1,
        help="Per-DP-rank microbatch size for gradient accumulation.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=4096,
        help="Sequence length. Defaults to Qwen3 8B's max sequence length.",
    )
    parser.add_argument(
        "--dp-degree",
        type=int,
        default=2,
        help="Data-parallel mesh degree.",
    )
    parser.add_argument(
        "--tp-degree",
        type=int,
        default=2,
        help="Tensor-parallel mesh degree.",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=20,
        help="Number of optimizer steps.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping max norm.",
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


def init_distributed(args):
    if "WORLD_SIZE" not in os.environ or "LOCAL_RANK" not in os.environ:
        raise RuntimeError(
            "Run this example with torchrun, e.g. "
            "torchrun --standalone --nproc-per-node 4 "
            "examples/example_sanity_check_qwen3.py"
        )

    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    expected_world_size = args.dp_degree * args.tp_degree
    if world_size != expected_world_size:
        raise ValueError(
            f"WORLD_SIZE ({world_size}) must equal dp-degree * tp-degree "
            f"({args.dp_degree} * {args.tp_degree} = {expected_world_size})."
        )
    if args.global_batch_size % args.dp_degree != 0:
        raise ValueError(
            f"global-batch-size ({args.global_batch_size}) must be divisible by "
            f"dp-degree ({args.dp_degree})."
        )
    local_batch_size = args.global_batch_size // args.dp_degree
    if local_batch_size % args.microbatch_size != 0:
        raise ValueError(
            f"local batch size ({local_batch_size}) must be divisible by "
            f"microbatch-size ({args.microbatch_size})."
        )

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda",
        (args.dp_degree, args.tp_degree),
        mesh_dim_names=("dp", "tp"),
    )
    return device, mesh


def make_local_tokens(args, mesh, device, vocab_size: int) -> torch.Tensor:
    coordinate = mesh.get_coordinate()
    if coordinate is None:
        raise RuntimeError("DeviceMesh coordinate is unavailable on this rank.")
    dp_rank, _tp_rank = coordinate
    local_batch_size = args.global_batch_size // args.dp_degree

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    tokens = torch.randint(
        0,
        vocab_size,
        (args.global_batch_size, args.seq_len + 1),
        generator=generator,
        dtype=torch.long,
    )

    start = dp_rank * local_batch_size
    stop = start + local_batch_size
    return tokens[start:stop].to(device, non_blocking=True)


def vocab_parallel_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    vocab_size: int,
    tp_group,
    tp_rank: int,
    tp_degree: int,
    global_token_count: int,
) -> torch.Tensor:
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} is incompatible with "
            f"labels shape {tuple(labels.shape)}."
        )

    local_vocab_size = logits.shape[-1]
    vocab_start = tp_rank * local_vocab_size
    vocab_stop = vocab_start + local_vocab_size
    if tp_rank == tp_degree - 1:
        vocab_stop = vocab_size

    logits = logits.float()
    local_max = logits.amax(dim=-1)
    with torch.no_grad():
        global_max = local_max.detach().clone()
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tp_group)

    shifted_logits = logits - global_max.unsqueeze(-1)
    local_exp_sum = shifted_logits.exp().sum(dim=-1)
    global_exp_sum = dist_nn_func.all_reduce(
        local_exp_sum,
        op=dist.ReduceOp.SUM,
        group=tp_group,
    )

    target_mask = (labels >= vocab_start) & (labels < vocab_stop)
    local_target = torch.zeros_like(labels, dtype=torch.long)
    local_target[target_mask] = labels[target_mask] - vocab_start
    local_target_logits = logits.gather(-1, local_target.unsqueeze(-1)).squeeze(-1)
    local_target_logits = local_target_logits * target_mask.to(logits.dtype)
    target_logits = dist_nn_func.all_reduce(
        local_target_logits,
        op=dist.ReduceOp.SUM,
        group=tp_group,
    )

    loss_sum = (global_exp_sum.log() + global_max - target_logits).sum()
    return loss_sum / (global_token_count * tp_degree)


def print_rank0(message: str) -> None:
    if dist.get_rank() == 0:
        print(message, flush=True)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG)

    device, mesh = init_distributed(args)
    tp_group = mesh.get_group("tp")
    tp_rank = mesh.get_local_rank("tp")
    local_batch_size = args.global_batch_size // args.dp_degree
    gradient_accumulation_steps = local_batch_size // args.microbatch_size

    torch.manual_seed(args.seed)
    model_args = qwen3_8b_args(max_seq_len=args.seq_len)
    trace_global_batch_size = args.microbatch_size * args.dp_degree

    with torch.device("meta"):
        model = Transformer(model_args)

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
        "Qwen3 8B sanity check: "
        f"mesh=(dp={args.dp_degree}, tp={args.tp_degree}), "
        f"global_batch={args.global_batch_size}, "
        f"local_batch={local_batch_size}, "
        f"microbatch={args.microbatch_size}, "
        f"grad_accum={gradient_accumulation_steps}, "
        f"trace_global_batch={trace_global_batch_size}, "
        f"seq_len={args.seq_len}"
    )

    t0 = time.time()
    with AutoParallel(
        model,
        input_fn,
        mesh,
        mp_policy,
        repeated_subgraphs=True,
    ) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)
        autop.add_input_constraints([(Shard(0), Replicate())])
        autop.add_output_constraints([(Shard(0), Shard(2))])
        sharding_placement = autop.optimize_placement(verbose=args.verbose)
        parallel_mod = autop.apply_placement(sharding_placement)

    print_rank0(f"Tracing + optimization took {time.time() - t0:.1f}s")

    parallel_mod.to_empty(device=device)
    parallel_mod.init_weights(buffer_device=device, seed=args.seed)  # type: ignore[operator]

    if args.compile:
        parallel_mod = torch.compile(parallel_mod, backend=autoparallel_backend())

    batch = make_local_tokens(args, mesh, device, model_args.vocab_size)
    inputs = batch[:, :-1].contiguous()
    labels = batch[:, 1:].contiguous()
    input_microbatches = inputs.split(args.microbatch_size, dim=0)
    label_microbatches = labels.split(args.microbatch_size, dim=0)
    global_token_count = args.global_batch_size * args.seq_len
    optimizer = torch.optim.AdamW(parallel_mod.parameters(), lr=args.lr)

    try:
        losses: list[float] = []
        for step in range(args.train_steps):
            optimizer.zero_grad(set_to_none=True)
            step_loss = torch.zeros((), device=device)
            for micro_inputs, micro_labels in zip(
                input_microbatches, label_microbatches
            ):
                logits = parallel_mod(micro_inputs)
                if torch.any(torch.isnan(logits)):
                    raise RuntimeError("Found NaNs in Qwen3 forward output.")

                loss = vocab_parallel_cross_entropy(
                    logits,
                    micro_labels,
                    vocab_size=model_args.vocab_size,
                    tp_group=tp_group,
                    tp_rank=tp_rank,
                    tp_degree=args.tp_degree,
                    global_token_count=global_token_count,
                )
                if torch.any(torch.isnan(loss)):
                    raise RuntimeError("Found NaNs in Qwen3 training loss.")

                loss.backward()
                step_loss = step_loss + loss.detach()

            torch.nn.utils.clip_grad_norm_(
                parallel_mod.parameters(), args.max_grad_norm
            )
            optimizer.step()

            with torch.no_grad():
                logged_loss = step_loss.clone()
                dist.all_reduce(logged_loss, op=dist.ReduceOp.SUM)
                loss_value = float(logged_loss.item())
            losses.append(loss_value)
            print_rank0(f"step={step:03d} loss={loss_value:.6f}")

        if losses[-1] >= losses[0]:
            raise RuntimeError(
                f"Qwen3 training loss did not improve: initial={losses[0]:.6f}, "
                f"final={losses[-1]:.6f}"
            )

        print_rank0(f"Loss improved: initial={losses[0]:.6f}, final={losses[-1]:.6f}")
        dist.barrier(device_ids=[device.index])
        torch.cuda.synchronize(device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
