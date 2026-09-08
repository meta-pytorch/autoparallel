from __future__ import annotations

import os
from dataclasses import replace

import torch
from torch.nn.attention import SDPBackend

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import default_adamw
from torchtitan.components.validate import Validator
from torchtitan.config import CompileConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.graph_trainer.common_utils import (
    build_decoder_config_for_backend,
)
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.llama3 import (
    model_registry as graph_llama3_model_registry,
)
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.common.attention import ScaledDotProductAttention
from torchtitan.models.llama3 import (
    llama3_configs,
    model_registry as native_llama3_model_registry,
)
from torchtitan.models.llama3.config_registry import llama3_8b

from .buffered_metrics import BufferedMetricsProcessor
from .fairness_trainer import to_fairness_graph_trainer_config
from .io_utils import bool_env, int_env, required_env
from .replay_data import EmptyDataLoader, ReplayDataLoader


def _parallelize_manual_with_autoparallel_cp(
    model,
    *,
    parallel_dims,
    training,
    parallelism,
    compile_config,
    ac_config,
    dump_folder,
):
    """Run TorchTitan manual TP/FSDP with AutoParallel's existing CP wrapper."""
    from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
    from torchtitan.experiments.graph_trainer.common_utils import apply_simple_fsdp
    from torchtitan.experiments.graph_trainer.compile import apply_compile
    from torchtitan.experiments.graph_trainer.llama3.parallelize import annotate_llama
    from torchtitan.experiments.graph_trainer.llama3.parallelize_autoparallel import (
        _apply_autoparallel_context_parallel_attention,
        _build_autoparallel_mesh,
    )

    del ac_config
    assert training.seq_len % parallel_dims.seq_len_divisor == 0

    if parallel_dims.cp_enabled:
        _apply_autoparallel_context_parallel_attention(
            model, _build_autoparallel_mesh(parallel_dims)
        )

    annotate_llama(model)
    if parallel_dims.tp_enabled:
        model.parallelize(parallel_dims)
        maybe_enable_async_tp(
            parallelism, compile_config, parallel_dims.get_mesh("tp")
        )

    model = apply_simple_fsdp(
        model,
        parallel_dims=parallel_dims,
        training=training,
    )
    return apply_compile(
        model,
        compile_config=compile_config,
        parallelism=parallelism,
        parallel_dims=parallel_dims,
        dump_folder=dump_folder,
    )


def _native_sdpa_model_spec():
    # Pin the shared AutoParallel CP body in both arms to Flash SDPA only.
    ScaledDotProductAttention.sdpa_backends = [SDPBackend.FLASH_ATTENTION]
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    native = native_llama3_model_registry("8B")
    sdpa_model = build_decoder_config_for_backend(llama3_configs["8B"], "sdpa")
    # The AutoParallel CP wrapper is itself a full-mesh local_map.  Do not wrap
    # it in TorchTitan's inner-attention local_map when the manual TP/FSDP path
    # calls model.parallelize().
    for layer in sdpa_model.layers:
        layer.attention.inner_attention.sharding_config = None
    return replace(native, model=sdpa_model)


def _base_config():
    world_size = int_env("BENCHMARK_WORLD_SIZE")
    dp_degree = int_env("BENCHMARK_DP_DEGREE")
    cp_degree = int_env("BENCHMARK_CP_DEGREE")
    tp_degree = int_env("BENCHMARK_TP_DEGREE")
    local_batch_size = int_env("BENCHMARK_LOCAL_BATCH_SIZE")
    seq_len = int_env("BENCHMARK_SEQ_LEN")
    total_steps = int_env("BENCHMARK_TOTAL_STEPS")
    log_freq = int_env("BENCHMARK_LOG_FREQ")
    if dp_degree * cp_degree * tp_degree != world_size:
        raise ValueError(
            f"DP-shard({dp_degree}) * CP({cp_degree}) * TP({tp_degree}) "
            f"must equal world size {world_size}"
        )

    config = llama3_8b()
    config.model_spec = _native_sdpa_model_spec()
    config.hf_assets_path = required_env("LLAMA_TOKENIZER_DIR")
    config.dataloader = ReplayDataLoader.Config()
    config.loss = CrossEntropyLoss.Config(
        global_vocab_size=decoder_vocab_size(config.model_spec)
    )
    config.optimizer = default_adamw(lr=3e-4)
    config.lr_scheduler = LRSchedulersContainer.Config(warmup_steps=200)
    config.training = replace(
        config.training,
        local_batch_size=local_batch_size,
        global_batch_size=local_batch_size * dp_degree,
        seq_len=seq_len,
        steps=total_steps,
        dtype="float32",
        mixed_precision_param="bfloat16",
        mixed_precision_reduce="float32",
        max_norm=1.0,
        gc_freq=50,
        gc_debug=False,
    )
    config.parallelism = replace(
        config.parallelism,
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=dp_degree,
        tensor_parallel_degree=tp_degree,
        enable_sequence_parallel=True,
        context_parallel_degree=cp_degree,
        context_parallel_load_balancer="headtail",
        pipeline_parallel_degree=1,
        expert_parallel_degree=1,
        enable_async_tensor_parallel=False,
    )
    config.activation_checkpoint = SelectiveAC.Config()
    config.metrics = BufferedMetricsProcessor.Config(
        log_freq=log_freq,
        enable_tensorboard=False,
        enable_wandb=False,
        save_for_all_ranks=True,
        disable_color_printing=True,
    )
    profile_enabled = bool_env("BENCHMARK_ENABLE_KINETO") and int(
        os.environ.get("RANK", "0")
    ) == int(os.environ.get("BENCHMARK_PROFILE_RANK", "0"))
    memory_enabled = bool_env("BENCHMARK_ENABLE_MEMORY_SNAPSHOT")
    profile_active = int_env("BENCHMARK_PROFILE_ACTIVE", minimum=1)
    profile_warmup = int(os.environ.get("BENCHMARK_PROFILE_WARMUP", "0"))
    config.profiler = replace(
        config.profiler,
        enable_profiling=profile_enabled,
        profile_freq=int_env("BENCHMARK_PROFILE_FREQ"),
        profiler_warmup=profile_warmup,
        profiler_active=profile_active,
        profiler_repeat=1,
        profiler_skip_first=0,
        enable_memory_snapshot=memory_enabled,
        memory_snapshot_freq=max(1, total_steps),
    )
    config.validator = Validator.Config(enable=False)
    config.debug = replace(
        config.debug,
        seed=42,
        deterministic=True,
        deterministic_warn_only=False,
        detect_anomaly=False,
        print_config=False,
        save_config_file="config.json",
        enable_structured_logging=bool_env("BENCHMARK_STRUCTURED_LOGGING"),
    )
    config.comm = replace(
        config.comm,
        init_timeout_seconds=1200,
        train_timeout_seconds=1200,
    )

    phase = required_env("BENCHMARK_PHASE")
    if phase in {"correctness", "steady", "profile"}:
        seed_path = required_env("BENCHMARK_SEED_CHECKPOINT")
        config.checkpoint = CheckpointManager.Config(
            enable=True,
            folder="checkpoint",
            interval=total_steps if phase == "correctness" else total_steps + 1,
            initial_load_path=seed_path,
            initial_load_model_only=True,
            initial_load_in_hf=True,
            last_save_model_only=True,
            last_save_in_hf=True,
            export_dtype="bfloat16",
            async_mode="disabled",
            load_only=phase != "correctness",
        )
    else:
        config.checkpoint = replace(
            config.checkpoint,
            enable=False,
            load_only=False,
            initial_load_path=None,
            initial_load_model_only=False,
            create_seed_checkpoint=False,
        )
    return config


def seed_checkpoint():
    config = _base_config()
    config.parallelism = replace(
        config.parallelism,
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=1,
        tensor_parallel_degree=1,
        context_parallel_degree=1,
        pipeline_parallel_degree=1,
        expert_parallel_degree=1,
    )
    config.training = replace(
        config.training,
        local_batch_size=1,
        global_batch_size=1,
        steps=1,
    )
    config.dataloader = EmptyDataLoader.Config()
    config.compile = CompileConfig(enable=False)
    config.metrics = BufferedMetricsProcessor.Config(log_freq=1)
    config.profiler = replace(
        config.profiler,
        enable_profiling=False,
        enable_memory_snapshot=False,
    )
    config.checkpoint = CheckpointManager.Config(
        enable=True,
        folder="checkpoint",
        interval=1,
        create_seed_checkpoint=True,
        last_save_model_only=True,
        last_save_in_hf=True,
        export_dtype="bfloat16",
        async_mode="disabled",
    )
    return config


def _graph_config(*, enable_autoparallel: bool):
    config = to_graph_trainer_config(_base_config(), graph_llama3_model_registry)
    config = to_fairness_graph_trainer_config(config)
    if not enable_autoparallel:
        config.model_spec = replace(
            config.model_spec,
            parallelize_fn=_parallelize_manual_with_autoparallel_cp,
        )
    config.profiler = replace(config.profiler, trace_post_processor=None)
    placement_mode = os.environ.get("BENCHMARK_AP_PLACEMENTS_MODE", "")
    placement_path = os.environ.get("BENCHMARK_AP_PLACEMENTS_PATH", "")
    if placement_mode not in {"", "save", "load"}:
        raise ValueError(
            "BENCHMARK_AP_PLACEMENTS_MODE must be empty, 'save', or 'load'"
        )
    if bool(placement_mode) != bool(placement_path):
        raise ValueError(
            "BENCHMARK_AP_PLACEMENTS_MODE and BENCHMARK_AP_PLACEMENTS_PATH "
            "must be set together"
        )
    config.compile = GraphTrainerCompileConfig(
        enable=True,
        components=["model", "loss"],
        mode="aot_fx_trace",
        memory_policy="eager",
        inductor_compilation="full",
        numerics_changing_optim=False,
        disable_passes=[
            "cudagraph_pass",
            "joint_transformer_block_bucketing_reordering_pass",
        ],
        enable_fsdp_ag_rs_overlap=False,
        enable_fsdp_dense_region_overlap=False,
        enable_autoparallel=enable_autoparallel,
        autoparallel_solver="approx",
        autoparallel_placements_save_path=(
            placement_path
            if enable_autoparallel and placement_mode == "save"
            else ""
        ),
        autoparallel_placements_load_path=(
            placement_path
            if enable_autoparallel and placement_mode == "load"
            else ""
        ),
    )
    return config


def graph_manual():
    return _graph_config(enable_autoparallel=False)


def graph_autoparallel():
    return _graph_config(enable_autoparallel=True)
