"""LLaMA 3 8B FSDP x Ulysses CP x TP arms for the 3D campaigns.

These are the 2D arms at ``BENCHMARK_SEQ_LEN`` with TorchTitan's Ulysses varlen
context parallelism over ``BENCHMARK_CP_DEGREE`` ranks. Each DP rank replays
its own pre-sharded C4 samples from ``BENCHMARK_REPLAY_ROOT``.
"""

from __future__ import annotations

from dataclasses import replace

from torchtitan.config.transform.apply import apply_transforms
from torchtitan.config.transform.context_parallel import ContextParallelTransform
from torchtitan.models.common.cp_attention import UlyssesCPVarlenInnerAttention
from workloads.llama3_2d.perf_configs import (
    compiled_loss_config,
    graph_trainer_config,
    llama3_8b_config,
)

from .io_utils import int_env


def _ulysses_config():
    config = llama3_8b_config(seq_len=int_env("BENCHMARK_SEQ_LEN"))
    config = replace(
        config,
        metrics=replace(config.metrics, enable_tensorboard=False),
        debug=replace(config.debug, deterministic=True),
        comm=replace(config.comm, train_timeout_seconds=1200),
    )
    # The CP degree and the CP inner attention must change together, so set
    # the fields in place and let ``apply_transforms`` validate both.
    config.parallelism.context_parallel_degree = int_env("BENCHMARK_CP_DEGREE")
    config.parallelism.context_parallel_load_balancer = None
    return apply_transforms(
        config,
        [ContextParallelTransform(inner_attention=UlyssesCPVarlenInnerAttention)],
    )


def torchtitan_eager_8b():
    return _ulysses_config()


def torchtitan_compiled_loss_8b():
    return compiled_loss_config(_ulysses_config())


def graphtrainer_manual_8b():
    return graph_trainer_config(_ulysses_config(), enable_autoparallel=False)


def autoparallel_graphtrainer_8b():
    return graph_trainer_config(_ulysses_config(), enable_autoparallel=True)
