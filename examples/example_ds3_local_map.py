# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Shard
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel._testing.models.dsv3 import DeepSeekV3Model, make_dsv3_config
from autoparallel.api import AutoParallel
from autoparallel.shardings.placement_options import NumericsLogger

_DEFAULT_DTENSOR_RNG_SEED = 0


def _seed_dtensor_rng(rng_seed: Optional[int]) -> None:
    torch.manual_seed(_DEFAULT_DTENSOR_RNG_SEED if rng_seed is None else rng_seed)


def run_test(fake_evaluate: bool, rng_seed: Optional[int], logs_dir: str):
    # Match TorchTitan's DeepSeek V3 debug model shape. This example is a
    # regression guard for placement/clustering issues that only appear at the
    # larger debug shape used by TorchTitan GraphTrainer.
    seq_len = 2048
    if fake_evaluate:
        world_size = 256

        fake_store = FakeStore()
        torch.distributed.init_process_group(
            "fake", store=fake_store, rank=0, world_size=world_size
        )
        local_rank = torch.distributed.get_rank()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        _seed_dtensor_rng(rng_seed)
        mesh = torch.distributed.device_mesh.init_device_mesh(
            "cuda",
            (world_size // 64, 64),
            mesh_dim_names=("dp", "ep"),
        )

        config = make_dsv3_config(num_experts=64, max_seq_len=seq_len)
    else:
        dp_degree = 2
        ep_degree = 2
        world_size = dp_degree * ep_degree

        assert (
            "WORLD_SIZE" in os.environ
        ), f"run with torchrun --standalone --nproc-per-node {world_size}"
        assert (
            int(os.getenv("WORLD_SIZE")) == world_size
        ), f"Need at least {world_size} GPUs for real evaluation"
        local_rank = int(os.getenv("LOCAL_RANK"))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        _seed_dtensor_rng(rng_seed)
        torch.distributed.init_process_group(backend="nccl", device_id=device)
        mesh = torch.distributed.device_mesh.init_device_mesh(
            "cuda",
            (dp_degree, ep_degree),
            mesh_dim_names=("dp", "ep"),
        )

        config = make_dsv3_config(max_seq_len=seq_len)

    local_batch_size = 8
    global_batch_size = local_batch_size * mesh.shape[0] * mesh.shape[1]

    with torch.device("meta"):
        model = DeepSeekV3Model(
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

    numerics_logger = None
    if rng_seed is not None:
        numerics_logger = NumericsLogger(logs_dir)
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

        sharding_placement = autop.optimize_placement(verbose=False)
        parallel_mod = autop.apply_placement(sharding_placement)

    parallel_mod.to_empty(device=device)
    parallel_mod.init_weights(buffer_device=device, seed=rng_seed)
    if rng_seed is not None:
        numerics_logger.log_model_weights(parallel_mod)
        torch.manual_seed(rng_seed)

    n_microbatches = 16
    full_batch = torch.randint(
        0,
        config.vocab_size,
        (local_batch_size * n_microbatches, seq_len),
        device=device,
    )
    microbatches = torch.split(full_batch, local_batch_size, dim=0)
    assert len(microbatches) == n_microbatches
    if rng_seed:
        numerics_logger.log_diff(
            full_batch.to(torch.float32), prefix="full batch input"
        )

    with torch.autograd.set_multithreading_enabled(False):
        if fake_evaluate:
            shape_env = ShapeEnv()
            with FakeTensorMode(
                allow_non_fake_inputs=True,
                shape_env=shape_env,
            ):
                for x in microbatches:
                    out = parallel_mod(x)
                    out.backward(torch.ones_like(out))
        else:
            for i, x in enumerate(microbatches):
                assert x.shape[0] == local_batch_size
                out = parallel_mod(x)
                assert not torch.any(torch.isnan(out)), "Found NaNs in forward output"
                out.backward(torch.ones_like(out))
                if rng_seed is not None:
                    numerics_logger.log_diff(out, prefix=f"mb{i} fwd out")

            if rng_seed is not None:
                for k, v in parallel_mod.named_parameters():
                    numerics_logger.log_diff(v.grad, prefix=f"grad {k}")

    print("All good!")

    if torch.distributed.is_initialized():
        if torch.distributed.get_backend() == torch.distributed.Backend.NCCL:
            torch.distributed.barrier(device_ids=[local_rank])
        else:
            torch.distributed.barrier()
        torch.cuda.synchronize(device)
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run DeepSeek V3 local_map example")
    parser.add_argument(
        "--fake-evaluate",
        action="store_true",
        default=False,
        help="Use fake evaluation mode with FakeTensorMode (default: False)",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=None,
        help="Use a specific rng seed and deterministic algorithms for run-to-run invariance (default: None).",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="out/",
        help="Directory to store logs (default: ./out/).",
    )
    args = parser.parse_args()

    if args.rng_seed is not None:
        torch.use_deterministic_algorithms(True)

    run_test(
        fake_evaluate=args.fake_evaluate, rng_seed=args.rng_seed, logs_dir=args.logs_dir
    )
