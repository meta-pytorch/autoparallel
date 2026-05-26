# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import time

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel._testing.models.qwen3 import (
    Qwen3ModelArgs,
    Transformer,
    qwen3_235b_a22b_args,
    qwen3_30b_a3b_args,
    qwen3_8b_args,
    qwen3_debug_args,
    qwen3_moe_debug_args,
)
from autoparallel.api import AutoParallel
from autoparallel.compile import autoparallel_backend


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trace, optimize, and smoke-test dense Qwen3 with AutoParallel."
    )
    parser.add_argument(
        "--flavor",
        choices=("tiny", "moe-tiny", "debug", "8b", "moe-debug", "30b-a3b", "235b-a22b"),
        default="tiny",
        help="Qwen3 model size to instantiate. Defaults to tiny for faster runs.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Sequence length. Defaults to 8 for tiny, 512 for debug, and 4096 for 8b.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=64,
        help="Fake process-group world size.",
    )
    parser.add_argument(
        "--tp-degree",
        type=int,
        default=8,
        help="Second mesh degree. Used as TP for dense flavors and EP for MoE flavors.",
    )
    parser.add_argument(
        "--local-batch-size",
        type=int,
        default=2,
        help="Per-DP-rank batch size used for the runtime smoke pass.",
    )
    parser.add_argument(
        "--save-optimizer",
        type=str,
        default=None,
        help="Optional path for the serialized sharding optimizer state.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the placed module with the AutoParallel backend before running.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Only run tracing, optimization, and placement application.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full AutoParallel optimizer log.",
    )
    return parser.parse_args()


def make_model_args(flavor: str, seq_len: int):
    if flavor == "tiny":
        return Qwen3ModelArgs(
            dim=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            head_dim=16,
            hidden_dim=128,
            vocab_size=128,
            max_seq_len=seq_len,
        )
    if flavor == "moe-tiny":
        return Qwen3ModelArgs(
            dim=64,
            n_layers=1,
            n_heads=4,
            n_kv_heads=2,
            head_dim=16,
            hidden_dim=128,
            vocab_size=128,
            max_seq_len=seq_len,
            moe_enabled=True,
            moe_hidden_dim=32,
            num_experts=8,
            top_k=2,
            route_norm=True,
            score_before_experts=False,
        )
    if flavor == "debug":
        return qwen3_debug_args(max_seq_len=seq_len)
    if flavor == "8b":
        return qwen3_8b_args(max_seq_len=seq_len)
    if flavor == "moe-debug":
        return qwen3_moe_debug_args(max_seq_len=seq_len)
    if flavor == "30b-a3b":
        return qwen3_30b_a3b_args(max_seq_len=seq_len)
    if flavor == "235b-a22b":
        return qwen3_235b_a22b_args(max_seq_len=seq_len)
    raise ValueError(f"Unknown Qwen3 flavor: {flavor}")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG)

    seq_len = args.seq_len
    if seq_len is None:
        seq_len = {
            "tiny": 8,
            "moe-tiny": 8,
            "debug": 512,
            "8b": 4096,
            "moe-debug": 512,
            "30b-a3b": 4096,
            "235b-a22b": 4096,
        }[args.flavor]
    if args.world_size % args.tp_degree != 0:
        raise ValueError(
            f"world-size ({args.world_size}) must be divisible by "
            f"tp-degree ({args.tp_degree})."
        )

    if not torch.distributed.is_initialized():
        fake_store = FakeStore()
        torch.distributed.init_process_group(
            "fake",
            store=fake_store,
            rank=0,
            world_size=args.world_size,
        )

    model_args = make_model_args(args.flavor, seq_len)
    mesh_dim_names = ("dp", "ep") if model_args.moe_enabled else ("dp", "tp")
    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda",
        (args.world_size // args.tp_degree, args.tp_degree),
        mesh_dim_names=mesh_dim_names,
    )
    device = torch.device("cuda")

    global_batch_size = args.local_batch_size * mesh.shape[0]
    if model_args.moe_enabled:
        global_batch_size *= mesh.shape[1]

    with torch.device("meta"):
        model = Transformer(
            model_args,
            mesh=mesh if model_args.moe_enabled else None,
            moe_axis_name=mesh.mesh_dim_names[1],
        )

    def input_fn():
        return torch.randint(
            0,
            model_args.vocab_size,
            (global_batch_size, seq_len),
            device=device,
        )

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    t0 = time.time()
    with AutoParallel(
        model,
        input_fn,
        mesh,
        mp_policy,
        dynamic=model_args.moe_enabled,
        repeated_subgraphs=True,
    ) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)

        x_sharding = (Shard(0), Shard(0)) if model_args.moe_enabled else (Shard(0), Replicate())
        out_sharding = (Shard(0), Shard(2))
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([out_sharding])

        sharding_placement = autop.optimize_placement(verbose=args.verbose)
        print(f"Tracing + optimization took {time.time() - t0:.1f}s")

        if args.save_optimizer is not None:
            autop.sharding_optimizer.save(args.save_optimizer)
            autop.sharding_optimizer.save_placements(
                f"{args.save_optimizer}.placements.json"
            )

        parallel_mod = autop.apply_placement(sharding_placement)

    if args.skip_run:
        print("Placement applied successfully.")
        return

    parallel_mod.to_empty(device=device)
    parallel_mod.init_weights(buffer_device=device)  # type: ignore[operator]

    if args.compile:
        parallel_mod = torch.compile(parallel_mod, backend=autoparallel_backend())

    tokens = torch.randint(
        0,
        model_args.vocab_size,
        (args.local_batch_size, seq_len),
        device=device,
    )
    out = parallel_mod(tokens)
    if torch.any(torch.isnan(out)):
        raise RuntimeError("Found NaNs in Qwen3 forward output.")
    out.backward(torch.randn_like(out))
    print("All good!")


if __name__ == "__main__":
    main()
