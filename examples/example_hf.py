# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
AutoParallel with HuggingFace causal LM models.

Supports 1D and 2D device meshes with configurable shape and dim names.

Usage:
    # 1D mesh (8 devices)
    python examples/example_hf.py --model gpt2 --mesh 8

    # 2D mesh (2 x 4)
    python examples/example_hf.py --model gpt2 --mesh 2,4

    # 2D mesh with custom dim names
    python examples/example_hf.py --model gpt2 --mesh 2,4 --mesh-dim-names replicate,shard
"""

import argparse
import logging
import math
import time

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore
from transformers import AutoConfig, AutoModelForCausalLM

from autoparallel.api import AutoParallel

logging.basicConfig(level=logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser(description="AutoParallel with HuggingFace models")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="HuggingFace model name or path (e.g. gpt2, meta-llama/Llama-3-8B)",
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default="8",
        help="Comma-separated mesh shape (e.g. '8' for 1D, '2,4' for 2D)",
    )
    parser.add_argument(
        "--mesh-dim-names",
        type=str,
        default=None,
        help="Comma-separated mesh dim names (default: dim0,dim1,...)",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()

    mesh_shape = tuple(int(s) for s in args.mesh.split(","))
    ndim = len(mesh_shape)
    world_size = math.prod(mesh_shape)

    if args.mesh_dim_names is not None:
        dim_names = tuple(args.mesh_dim_names.split(","))
        assert (
            len(dim_names) == ndim
        ), f"mesh-dim-names has {len(dim_names)} names but mesh has {ndim} dims"
    else:
        dim_names = tuple(f"dim{i}" for i in range(ndim))

    # --- Fake distributed environment ---
    fake_store = FakeStore()
    torch.distributed.init_process_group(
        "fake", store=fake_store, rank=0, world_size=world_size
    )

    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda", mesh_shape, mesh_dim_names=dim_names
    )

    # Shard input batch dim along the first mesh dim only; replicate on the rest.
    x_sharding = (Shard(0),) + (Replicate(),) * (ndim - 1)

    print(f"Model:      {args.model}")
    print(f"Mesh:       {mesh_shape}, dim_names={dim_names}")
    print(f"Batch size: {args.batch_size} (global), Seq len: {args.seq_len}")
    print()

    # --- Load HF config and create model on meta device ---
    config = AutoConfig.from_pretrained(args.model)
    # KV-cache is an inference-time optimization (caching keys/values for
    # autoregressive generation). It's unnecessary during training and its
    # management logic introduces graph breaks that Dynamo can't trace through.
    config.use_cache = False
    vocab_size = config.vocab_size

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Config:     {type(config).__name__}, {num_params / 1e6:.1f}M params")
    print()

    # --- input_fn returns global-shaped tensors ---
    batch_size = args.batch_size
    seq_len = args.seq_len

    def input_fn():
        return torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")

    # --- AutoParallel pipeline ---
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )

    t0 = time.time()
    with AutoParallel(
        model, input_fn, mesh, mp_policy, compile=True, repeated_subgraphs=True
    ) as autop:
        autop.add_input_constraints([x_sharding])
        autop.add_parameter_memory_constraint(low=None, high=None)

        sharding_placement = autop.optimize_placement(verbose=True)
        parallel_mod = autop.apply_placement(sharding_placement)

    print(f"\nAutoParallel pipeline completed in {time.time() - t0:.2f}s")

    # --- Forward + backward ---
    parallel_mod.to_empty(device="cuda")

    local_batch = batch_size // mesh_shape[0]
    x = torch.randint(0, vocab_size, (local_batch, seq_len), device="cuda")
    out = parallel_mod(x)

    if isinstance(out, torch.Tensor):
        loss = out.sum()
    elif isinstance(out, (tuple, list)):
        loss = out[0].sum()
    else:
        loss = next(
            v.sum() for v in out.__dict__.values() if isinstance(v, torch.Tensor)
        )
    loss.backward()

    print("Forward + backward OK")


if __name__ == "__main__":
    main()
