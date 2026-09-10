from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.experiments.graph_trainer.autoparallel_api import (
    AutoParallelGraph,
    AutoParallelModelOutput,
    autoparallel_constructor_kwargs,
    autoparallel_optimize_kwargs,
)
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.configs import validate_autoparallel_config
from torchtitan.experiments.graph_trainer.llama3.parallelize_autoparallel import (
    _apply_autoparallel_context_parallel_attention,
    _build_autoparallel_mesh,
)
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import device_type


def parallelize_batched_autoparallel_llama(
    model,
    *,
    parallel_dims,
    training,
    parallelism,
    compile_config,
    ac_config,
    dump_folder,
):
    """Apply the historical AutoParallel placement contract to one `[B, S]` batch."""
    validate_autoparallel_config(compile_config)
    if parallel_dims.pp_enabled:
        raise ValueError("AutoParallel Llama3 does not support PP yet")

    cp_enabled = parallel_dims.cp_enabled
    if not cp_enabled and parallel_dims.dp_replicate_enabled:
        raise ValueError("AutoParallel Llama3 does not support DDP without 3D AP")
    dense_mesh = _build_autoparallel_mesh(parallel_dims)
    explicit_cp_axis = "cp" in (dense_mesh.mesh_dim_names or ())
    if cp_enabled:
        _apply_autoparallel_context_parallel_attention(model, dense_mesh)

    local_batch_size, remainder = divmod(
        training.num_tokens_per_microbatch_per_dp_rank,
        training.max_context_length,
    )
    if remainder or local_batch_size <= 0:
        raise ValueError(
            "AutoParallel Llama3 requires an integral fixed-length local batch"
        )
    dp_degree = parallel_dims.dp_replicate * parallel_dims.dp_shard
    distributed_microbatch_tokens = (
        training.num_tokens_per_microbatch_per_dp_rank * dp_degree
    )
    if training.num_tokens_per_train_step not in (
        -1,
        distributed_microbatch_tokens,
    ):
        raise ValueError("AutoParallel Llama3 paper workloads require GA=1")
    placement_batch_size = local_batch_size * dp_degree

    def input_fn():
        tokens = torch.randint(
            0,
            model.config.vocab_size,
            (placement_batch_size, training.max_context_length),
            device=torch.device(device_type),
        )
        positions = torch.arange(
            training.max_context_length,
            dtype=torch.int64,
            device=torch.device(device_type),
        ).repeat(placement_batch_size, 1)
        return tokens, positions

    mp_policy = MixedPrecisionPolicy(
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        cast_forward_inputs=False,
    )
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        parallelism.fsdp_reshard_after_forward,
        parallel_dims.pp_enabled,
    )
    possible_input_shardings = (
        {
            "dp_replicate": Shard(0),
            "dp_shard": Shard(0),
            "fsdp": Replicate(),
            "cp": Shard(1),
            "tp": Replicate(),
        }
        if cp_enabled
        else {
            "dp_replicate": Shard(0),
            "fsdp": Shard(0),
            "tp": Replicate(),
        }
    )
    unsupported_axes = [
        name
        for name in dense_mesh.mesh_dim_names
        if name not in possible_input_shardings
    ]
    if unsupported_axes:
        raise ValueError(f"Unsupported AutoParallel mesh axes: {unsupported_axes}")
    input_sharding = tuple(
        possible_input_shardings[name] for name in dense_mesh.mesh_dim_names
    )
    output_sharding = (
        tuple(
            Shard(2)
            if name == "tp"
            else Shard(1)
            if name == "cp"
            else Shard(0)
            for name in dense_mesh.mesh_dim_names
        )
        if explicit_cp_axis or not cp_enabled
        else input_sharding
    )

    with AutoParallelGraph(
        model,
        input_fn,
        dense_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
        **autoparallel_constructor_kwargs(compile_config),
    ) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)
        autop.add_input_constraints([input_sharding, input_sharding])
        autop.add_output_constraints([output_sharding])
        if compile_config.autoparallel_placements_load_path:
            placement = autop.sharding_optimizer.load_placements(
                compile_config.autoparallel_placements_load_path
            )
            logger.info(
                "Loaded AutoParallel placements from %s",
                compile_config.autoparallel_placements_load_path,
            )
        else:
            started = time.time()
            placement = autop.optimize_placement(
                **autoparallel_optimize_kwargs(compile_config)
            )
            logger.info("AutoParallelGraph took %.2f seconds", time.time() - started)
            if compile_config.autoparallel_placements_save_path:
                save_path = Path(compile_config.autoparallel_placements_save_path)
                if (
                    not torch.distributed.is_initialized()
                    or torch.distributed.get_rank() == 0
                ):
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    autop.sharding_optimizer.save_placements(save_path)
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()

        model_output = (
            AutoParallelModelOutput(
                output_mesh=parallel_dims.get_mesh("tp"),
                output_placements=(Shard(2),),
                sharded_output_axis=2,
            )
            if parallel_dims.tp_enabled and (explicit_cp_axis or not cp_enabled)
            else None
        )
        parallel_model = autop.apply_placement_for_fx_module(
            placement,
            compile_config=compile_config,
            model_output=model_output,
            manages_context_parallel_input=not explicit_cp_axis,
        )

    return apply_compile(
        parallel_model,
        compile_config=compile_config,
        parallelism=parallelism,
        parallel_dims=parallel_dims,
        dump_folder=dump_folder,
    )
