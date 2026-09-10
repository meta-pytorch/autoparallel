from __future__ import annotations

from dataclasses import replace

from torchtitan.experiments.graph_trainer.muse_glimmer.config_registry import (
    graph_trainer_muse_glimmer_30b_sdpa_c4_4x2 as _graphtrainer_manual,
)
from torchtitan.experiments.graph_trainer.muse_glimmer.config_registry import (
    graph_trainer_muse_glimmer_30b_sdpa_c4_autoparallel_4x2 as _graphtrainer_ap,
)
from torchtitan.experiments.graph_trainer.muse_glimmer.config_registry import (
    muse_glimmer_30b_sdpa_c4_torchtitan_4x2 as _torchtitan,
)

from workloads.parameter_state import register_post_load_parameter_audit


def _with_post_load_audit(config):
    config.model_spec = replace(
        config.model_spec,
        post_optimizer_build_fn=register_post_load_parameter_audit,
    )
    return config


def muse_glimmer_30b_sdpa_c4_torchtitan_4x2():
    return _with_post_load_audit(_torchtitan())


def graph_trainer_muse_glimmer_30b_sdpa_c4_4x2():
    return _with_post_load_audit(_graphtrainer_manual())


def graph_trainer_muse_glimmer_30b_sdpa_c4_autoparallel_4x2():
    return _with_post_load_audit(_graphtrainer_ap())


def muse_glimmer_30b_sdpa_seed_checkpoint():
    config = _torchtitan()
    config.training = replace(
        config.training,
        local_batch_size=1,
        global_batch_size=1,
        steps=1,
    )
    config.parallelism = replace(
        config.parallelism,
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=1,
        tensor_parallel_degree=1,
        context_parallel_degree=1,
        pipeline_parallel_degree=1,
        expert_parallel_degree=1,
    )
    config.checkpoint = replace(
        config.checkpoint,
        enable=True,
        folder="checkpoint",
        create_seed_checkpoint=True,
        last_save_model_only=True,
        last_save_in_hf=False,
        async_mode="disabled",
        load_only=False,
    )
    config.profiler = replace(
        config.profiler,
        enable_profiling=False,
        enable_memory_snapshot=False,
    )
    return config
