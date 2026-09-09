from __future__ import annotations

import os
from dataclasses import replace

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import default_adamw
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.llama3 import model_registry
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.llama3.config_registry import llama3_8b

from .replay_data import FixedC4ReplayDataLoader, case_name


def _positive_int_env(name: str) -> int:
    value = int(os.environ[name])
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _base_config():
    world_size = _positive_int_env("BENCHMARK_WORLD_SIZE")
    dp_degree = _positive_int_env("BENCHMARK_DP_DEGREE")
    tp_degree = _positive_int_env("BENCHMARK_TP_DEGREE")
    local_batch_size = _positive_int_env("BENCHMARK_LOCAL_BATCH_SIZE")
    seq_len = _positive_int_env("BENCHMARK_SEQ_LEN")
    global_batch_size = _positive_int_env("BENCHMARK_GLOBAL_BATCH_SIZE")
    if (world_size, dp_degree, tp_degree) != (32, 4, 8):
        raise ValueError(
            "Replanning placement replay is currently validated only on the "
            f"canonical 4x8 mesh, got {world_size=}, {dp_degree=}, {tp_degree=}"
        )
    if global_batch_size != local_batch_size * dp_degree:
        raise ValueError(
            "Replanning uses one microbatch per optimizer step: "
            f"{global_batch_size} != {local_batch_size} * {dp_degree}"
        )
    selected_case = os.environ["BENCHMARK_REPLAY_CASE"]
    if selected_case != case_name(seq_len, local_batch_size):
        raise ValueError(
            f"BENCHMARK_REPLAY_CASE={selected_case!r} does not match "
            f"{case_name(seq_len, local_batch_size)!r}"
        )

    config = llama3_8b()
    config.model_spec = model_registry("8B", attn_backend="sdpa")
    config.hf_assets_path = os.environ["LLAMA_TOKENIZER_DIR"]
    config.dataloader = FixedC4ReplayDataLoader.Config()
    config.loss = CrossEntropyLoss.Config(
        global_vocab_size=decoder_vocab_size(config.model_spec)
    )
    config.optimizer = default_adamw(lr=3e-4)
    config.lr_scheduler = LRSchedulersContainer.Config(warmup_steps=200)
    config.training = replace(
        config.training,
        local_batch_size=local_batch_size,
        global_batch_size=global_batch_size,
        seq_len=seq_len,
        steps=25,
        dtype="float32",
        mixed_precision_param="bfloat16",
        mixed_precision_reduce="float32",
        max_norm=1.0,
    )
    config.parallelism = replace(
        config.parallelism,
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=dp_degree,
        tensor_parallel_degree=tp_degree,
        enable_sequence_parallel=True,
        context_parallel_degree=1,
        pipeline_parallel_degree=1,
        expert_parallel_degree=1,
        fsdp_reshard_after_forward="default",
    )
    config.activation_checkpoint = SelectiveAC.Config()
    config.metrics = replace(
        config.metrics,
        log_freq=5,
        enable_tensorboard=True,
        save_for_all_ranks=True,
        enable_wandb=False,
        disable_color_printing=True,
    )
    config.profiler = replace(
        config.profiler,
        enable_profiling=False,
        profile_freq=6,
        profiler_warmup=0,
        profiler_active=3,
        profiler_repeat=1,
        enable_memory_snapshot=False,
    )
    config.validator = replace(config.validator, enable=False)
    config.debug = replace(
        config.debug,
        seed=42,
        deterministic=False,
        deterministic_warn_only=False,
        enable_structured_logging=True,
        print_config=False,
        save_config_file="config.json",
    )
    config.checkpoint = replace(
        config.checkpoint,
        enable=False,
        load_only=False,
        initial_load_path=None,
        initial_load_model_only=False,
        create_seed_checkpoint=False,
    )
    config.comm = replace(config.comm, init_timeout_seconds=1200)
    return config


def autoparallel_graphtrainer_replanning_8b():
    """Official GraphTrainer LLaMA model spec with the fixed AP integration."""
    config = to_graph_trainer_config(_base_config(), model_registry)
    config.profiler = replace(config.profiler, trace_post_processor=None)
    config.compile = GraphTrainerCompileConfig(
        enable=True,
        components=["model", "loss"],
        mode="aot_fx_trace",
        memory_policy="eager",
        inductor_compilation="full",
        numerics_changing_optim=False,
        disable_passes=["cudagraph_pass"],
        enable_fsdp_ag_rs_overlap=False,
        enable_fsdp_dense_region_overlap=False,
        enable_autoparallel=True,
        autoparallel_solver="ilp",
    )
    return config
