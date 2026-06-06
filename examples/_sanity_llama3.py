"""Real LLaMA3 AutoParallel training sanity check on a 2D or 3D mesh.

Traces the model, picks a sharding strategy with the approximate (TRW-S) solver,
applies it as DTensor, and trains a fixed random batch for a few steps on real
GPUs. Pass: the loss curve goes down. Adapted from example_sanity_check_qwen3.py.

The batch is data-parallel on the `dp` axis only; any other axes (`cp`, `tp`)
are model-sharding axes (the solver shards params/activations over them). Logits
are vocab-parallel on `tp` and replicated on `cp`, so the loss is reduced over
the world and normalized by global_token_count * (world_size // dp_degree).

Run: torchrun --standalone --nproc-per-node N examples/_sanity_llama3.py --mesh 2,2,8 --model 8b
"""
import argparse
import logging
import os
import time

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn_func
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel._testing.models.llama3 import Transformer, TransformerModelArgs
from autoparallel.api import AutoParallel

_CFG = {
    "1b": dict(dim=2048, n_layers=16, n_heads=32, n_kv_heads=8, ffn_dim_multiplier=1.5, multiple_of=256),
    "8b": dict(dim=4096, n_layers=32, n_heads=32, n_kv_heads=8, ffn_dim_multiplier=1.3, multiple_of=1024),
}
_NAMES = {2: ("dp", "tp"), 3: ("dp", "cp", "tp")}


def parse_args():
    p = argparse.ArgumentParser(description="LLaMA3 AutoParallel training sanity check.")
    p.add_argument("--model", type=str, default="1b", choices=list(_CFG))
    p.add_argument("--mesh", type=str, default="2,2", help="comma-separated mesh dims")
    p.add_argument("--global-batch-size", type=int, default=8)
    p.add_argument("--microbatch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--train-steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--solver", type=str, default="approx")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def init_distributed(args):
    if "WORLD_SIZE" not in os.environ or "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Run with torchrun --standalone --nproc-per-node N ...")
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dims = tuple(int(x) for x in args.mesh.split(","))
    prod = 1
    for d in dims:
        prod *= d
    if prod != world_size:
        raise ValueError(f"WORLD_SIZE {world_size} != prod(mesh) {prod}")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda", dims, mesh_dim_names=_NAMES[len(dims)]
    )
    return device, mesh


def placement_for(name, *, is_output):
    if name == "dp":
        return Shard(0)
    if name == "tp" and is_output:
        return Shard(2)
    return Replicate()


def make_local_tokens(args, mesh, device, vocab_size):
    names = mesh.mesh_dim_names
    dp_rank = mesh.get_coordinate()[names.index("dp")]
    dp_degree = mesh["dp"].size()
    local_batch_size = args.global_batch_size // dp_degree
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed)
    tokens = torch.randint(
        0, vocab_size, (args.global_batch_size, args.seq_len + 1),
        generator=gen, dtype=torch.long,
    )
    start = dp_rank * local_batch_size
    return tokens[start:start + local_batch_size].to(device, non_blocking=True)


def vocab_parallel_cross_entropy(logits, labels, *, vocab_size, tp_group, tp_rank,
                                 tp_degree, normalizer):
    local_vocab_size = logits.shape[-1]
    vocab_start = tp_rank * local_vocab_size
    vocab_stop = vocab_size if tp_rank == tp_degree - 1 else vocab_start + local_vocab_size
    logits = logits.float()
    local_max = logits.amax(dim=-1)
    with torch.no_grad():
        global_max = local_max.detach().clone()
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tp_group)
    shifted = logits - global_max.unsqueeze(-1)
    global_exp_sum = dist_nn_func.all_reduce(
        shifted.exp().sum(dim=-1), op=dist.ReduceOp.SUM, group=tp_group)
    mask = (labels >= vocab_start) & (labels < vocab_stop)
    local_target = torch.zeros_like(labels, dtype=torch.long)
    local_target[mask] = labels[mask] - vocab_start
    local_target_logits = logits.gather(-1, local_target.unsqueeze(-1)).squeeze(-1)
    local_target_logits = local_target_logits * mask.to(logits.dtype)
    target_logits = dist_nn_func.all_reduce(
        local_target_logits, op=dist.ReduceOp.SUM, group=tp_group)
    loss_sum = (global_exp_sum.log() + global_max - target_logits).sum()
    return loss_sum / normalizer


def print_rank0(msg):
    if dist.get_rank() == 0:
        print(msg, flush=True)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    device, mesh = init_distributed(args)
    names = mesh.mesh_dim_names
    world_size = dist.get_world_size()
    tp_group = mesh.get_group("tp")
    tp_rank = mesh.get_local_rank("tp")
    tp_degree = mesh["tp"].size()
    dp_degree = mesh["dp"].size()
    local_batch_size = args.global_batch_size // dp_degree
    grad_accum = local_batch_size // args.microbatch_size
    # logits are distinct only across dp (cp/tp replicate the per-token loss),
    # so the world all-reduce over-counts by world_size // dp_degree.
    normalizer = args.global_batch_size * args.seq_len * (world_size // dp_degree)

    torch.manual_seed(args.seed)
    model_args = TransformerModelArgs(
        rope_theta=500000, vocab_size=128256, max_seq_len=args.seq_len, **_CFG[args.model],
    )
    trace_global_batch = args.microbatch_size * dp_degree

    with torch.device("meta"):
        model = Transformer(model_args)

    def input_fn():
        return torch.randint(0, model_args.vocab_size,
                             (trace_global_batch, args.seq_len), device=device)

    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    x_sharding = tuple(placement_for(n, is_output=False) for n in names)
    out_sharding = tuple(placement_for(n, is_output=True) for n in names)
    print_rank0(f"LLaMA3-{args.model} sanity: mesh={tuple(mesh.shape)}{names} "
                f"solver={args.solver} in={x_sharding} out={out_sharding} "
                f"global_batch={args.global_batch_size} microbatch={args.microbatch_size} "
                f"grad_accum={grad_accum} seq_len={args.seq_len} steps={args.train_steps} lr={args.lr}")

    t0 = time.time()
    with AutoParallel(model, input_fn, mesh, mp_policy, repeated_subgraphs=True,
                      solver=args.solver) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([out_sharding])
        sharding_placement = autop.optimize_placement(verbose=args.verbose)
        parallel_mod = autop.apply_placement(sharding_placement)
    print_rank0(f"trace+optimize+apply took {time.time() - t0:.1f}s")

    parallel_mod.to_empty(device=device)
    parallel_mod.init_weights(buffer_device=device)

    batch = make_local_tokens(args, mesh, device, model_args.vocab_size)
    inputs = batch[:, :-1].contiguous()
    labels = batch[:, 1:].contiguous()
    input_mbs = inputs.split(args.microbatch_size, dim=0)
    label_mbs = labels.split(args.microbatch_size, dim=0)
    optimizer = torch.optim.AdamW(parallel_mod.parameters(), lr=args.lr)

    try:
        losses = []
        step_times = []
        for step in range(args.train_steps):
            torch.cuda.synchronize(device)
            t_step = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            step_loss = torch.zeros((), device=device)
            for mi, ml in zip(input_mbs, label_mbs):
                logits = parallel_mod(mi)
                loss = vocab_parallel_cross_entropy(
                    logits, ml, vocab_size=model_args.vocab_size, tp_group=tp_group,
                    tp_rank=tp_rank, tp_degree=tp_degree, normalizer=normalizer)
                loss.backward()
                step_loss = step_loss + loss.detach()
            torch.nn.utils.clip_grad_norm_(parallel_mod.parameters(), args.max_grad_norm)
            optimizer.step()
            torch.cuda.synchronize(device)
            step_times.append(time.perf_counter() - t_step)
            with torch.no_grad():
                logged = step_loss.clone()
                dist.all_reduce(logged, op=dist.ReduceOp.SUM)
            losses.append(float(logged.item()))
            print_rank0(f"step={step:03d} loss={losses[-1]:.6f} step_time={1000*step_times[-1]:.0f}ms")

        warmup = min(3, max(0, len(step_times) - 2))
        steady = sorted(step_times[warmup:])
        if steady:
            mean_ms = 1000 * sum(steady) / len(steady)
            print_rank0(f"[latency] solver={args.solver} per-step (excl {warmup} warmup, "
                        f"{len(steady)} steps): mean={mean_ms:.0f}ms "
                        f"median={1000*steady[len(steady)//2]:.0f}ms min={1000*steady[0]:.0f}ms")
        print_rank0(f"\nloss curve: {[round(x, 4) for x in losses]}")
        verdict = "PASS" if losses[-1] < losses[0] else "FAIL"
        print_rank0(f"SANITY {verdict}: loss {losses[0]:.4f} -> {losses[-1]:.4f}")
        dist.barrier(device_ids=[device.index])
        torch.cuda.synchronize(device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
