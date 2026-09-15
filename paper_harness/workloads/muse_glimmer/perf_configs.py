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
    config.profiler = replace(config.profiler, trace_post_processor=None)
    return config


def muse_glimmer_30b_sdpa_c4_torchtitan_4x2():
    return _with_post_load_audit(_torchtitan())


def graph_trainer_muse_glimmer_30b_sdpa_c4_4x2():
    config = _with_post_load_audit(_graphtrainer_manual())
    config.compile = replace(
        config.compile,
        inductor_compilation="full",
        disable_passes=["cudagraph_pass"],
    )
    return config


def graph_trainer_muse_glimmer_30b_sdpa_c4_autoparallel_4x2():
    config = _with_post_load_audit(_graphtrainer_ap())
    config.compile = replace(
        config.compile,
        inductor_compilation="full",
        disable_passes=["cudagraph_pass"],
    )
    return config
