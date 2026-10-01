"""LLaMA 3 8B arms shared by the 2D, HSDP, and 3D campaigns.

Every arm starts from TorchTitan's ``llama3_8b`` recipe with varlen attention
and replays the same pre-tokenized C4 samples, ``LOCAL_BATCH_SIZE`` per
microbatch. Campaign settings choose the mesh, token batch, checkpoint, and
phase knobs.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor
from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.config import CompileConfig
from torchtitan.config.transform.base import convert_config_type
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.llama3 import GraphTrainerLlama3Model
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.llama3 import build_model_config, Llama3Model
from torchtitan_recipes.models.llama3 import llama3_8b
from workloads.llama3_3d.replay_data import ReplayDataLoader
from workloads.parameter_state import register_post_load_parameter_audit

LOCAL_BATCH_SIZE = 2
SEQ_LEN = 8192
AUTOPARALLEL_CONFIGURATION = "autoparallel_graphtrainer"


def _write_inductor_path_audit() -> None:
    """Assert and record that only AutoParallel loads its bucketing hooks."""
    import torch._inductor.fx_passes.bucketing as bucketing_mod
    import torch._inductor.fx_passes.fsdp as fsdp_mod

    functions = {
        "greedy_bucket_collective_by_mb": (
            f"{bucketing_mod.greedy_bucket_collective_by_mb.__module__}."
            f"{bucketing_mod.greedy_bucket_collective_by_mb.__qualname__}"
        ),
        "identify_fsdp_groups": (
            f"{fsdp_mod.identify_fsdp_groups.__module__}."
            f"{fsdp_mod.identify_fsdp_groups.__qualname__}"
        ),
    }
    patch_active = all("_patch_fsdp_bucketing" in value for value in functions.values())
    configuration = os.environ["BENCHMARK_CONFIGURATION"]
    expected_patch_active = configuration == AUTOPARALLEL_CONFIGURATION
    if patch_active != expected_patch_active:
        raise RuntimeError(
            "Unexpected AutoParallel bucketing hook state for "
            f"{configuration}: {functions}"
        )
    module_loaded = "autoparallel.graph_passes.auto_bucketing" in sys.modules
    if module_loaded != expected_patch_active:
        raise RuntimeError(
            f"Unexpected AutoParallel bucketing module state for {configuration}: "
            f"loaded={module_loaded}"
        )
    # AutoParallel passes its post-grad scheduler per compile, never globally.
    custom_post_pass = torch._inductor.config.post_grad_custom_post_pass
    if custom_post_pass is not None:
        raise RuntimeError(
            f"Unexpected global post-grad scheduler for {configuration}: "
            f"{custom_post_pass!r}"
        )

    audit_dir = Path(os.environ["MODULE_ISOLATION_AUDIT_DIR"])
    audit_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "rank": int(os.environ["RANK"]),
        "configuration": configuration,
        "autoparallel_bucketing_hook_active": patch_active,
        "autoparallel_bucketing_module_loaded": module_loaded,
        "functions": functions,
        "post_grad_custom_post_pass_present": False,
    }
    output = audit_dir / f"rank_{int(os.environ['RANK']):02d}.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def _write_placement_audit(model_parts) -> None:
    """Record AutoParallel parameter placements; each must replicate on dp_replicate.

    This is the evidence that the HSDP arms share one parameter layout rule,
    rather than a plan that happens to fit the memory budget.
    """
    parameters = {}
    violations = []
    for model_part in model_parts:
        for name, parameter in model_part.named_parameters():
            if not isinstance(parameter, DTensor):
                raise RuntimeError(f"AutoParallel parameter {name} is not a DTensor")
            names = parameter.device_mesh.mesh_dim_names
            parameters[name] = {
                "shape": list(parameter.shape),
                "mesh_shape": list(parameter.device_mesh.shape),
                "mesh_dim_names": list(names),
                "placements": [str(placement) for placement in parameter.placements],
            }
            if (
                "dp_replicate" not in names
                or not parameter.placements[names.index("dp_replicate")].is_replicate()
            ):
                violations.append(name)

    audit = {
        "rank": int(os.environ["RANK"]),
        "parameter_count": len(parameters),
        "parameter_placements": parameters,
        "replicate_axis_violations": violations,
    }
    audit_dir = Path(os.environ["PLACEMENT_AUDIT_DIR"])
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / f"rank_{int(os.environ['RANK']):03d}.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    if violations:
        raise RuntimeError(
            f"{len(violations)} parameters are not replicated on the "
            f"'dp_replicate' mesh axis: {violations[:8]}"
        )


def _register_audits(optimizers, model_parts, parallelism_context) -> None:
    _write_inductor_path_audit()
    if (
        os.environ["BENCHMARK_CONFIGURATION"] == AUTOPARALLEL_CONFIGURATION
        and parallelism_context.dp_replicate > 1
    ):
        _write_placement_audit(model_parts)
    register_post_load_parameter_audit(optimizers, model_parts, parallelism_context)


class HarnessLlama3Model(Llama3Model):
    """``Llama3Model`` that writes the harness audits once optimizers exist."""

    @dataclass(kw_only=True, slots=True)
    class Config(Llama3Model.Config):
        pass

    @classmethod
    def _register_optimizer_hooks(cls, optimizers, model_parts, parallelism_context):
        super()._register_optimizer_hooks(optimizers, model_parts, parallelism_context)
        _register_audits(optimizers, model_parts, parallelism_context)


class HarnessGraphTrainerLlama3Model(GraphTrainerLlama3Model):
    """``GraphTrainerLlama3Model`` that writes the harness audits."""

    @dataclass(kw_only=True, slots=True)
    class Config(GraphTrainerLlama3Model.Config):
        pass

    @classmethod
    def _register_optimizer_hooks(cls, optimizers, model_parts, parallelism_context):
        super()._register_optimizer_hooks(optimizers, model_parts, parallelism_context)
        _register_audits(optimizers, model_parts, parallelism_context)


def llama3_8b_config(*, seq_len: int):
    """The ``llama3_8b`` recipe with varlen attention and the C4 replay."""
    config = llama3_8b(seq_len=seq_len)
    model = convert_config_type(
        build_model_config("8B", seq_len=seq_len, attn_backend="varlen"),
        HarnessLlama3Model,
    )
    return replace(
        config,
        model=model,
        loss=CrossEntropyLoss.Config(global_vocab_size=decoder_vocab_size(model)),
        hf_assets_path=os.environ["LLAMA_TOKENIZER_DIR"],
        dataloader=ReplayDataLoader.Config(max_num_documents=LOCAL_BATCH_SIZE),
        training=replace(
            config.training,
            num_tokens_per_microbatch_per_dp_rank=LOCAL_BATCH_SIZE * seq_len,
            steps=28,
            disable_cuda_graphs=True,
        ),
        metrics=replace(
            config.metrics,
            log_freq=5,
            enable_tensorboard=True,
            save_for_all_ranks=True,
            enable_wandb=False,
            disable_color_printing=True,
        ),
        profiler=replace(
            config.profiler,
            enable_profiling=True,
            profile_freq=28,
            profiler_warmup=0,
            profiler_active=3,
            profiler_repeat=1,
            enable_memory_snapshot=False,
        ),
        debug=replace(
            config.debug,
            seed=42,
            deterministic=False,
            deterministic_warn_only=False,
            enable_structured_logging=True,
            print_config=False,
            save_config_file="config.json",
        ),
        comm=replace(config.comm, init_timeout_seconds=1200),
        checkpointer=CheckpointManager.Config(load_only=True),
    )


def compiled_loss_config(config):
    """Compile only the loss, with Inductor."""
    return replace(
        config, compile=CompileConfig(components=["loss"], backend="inductor")
    )


def graph_trainer_config(config, *, enable_autoparallel: bool):
    """Move a ``llama3_8b_config`` onto GraphTrainer with full Inductor."""
    config = to_graph_trainer_config(config, HarnessGraphTrainerLlama3Model.Config)
    return replace(
        config,
        compile=GraphTrainerCompileConfig(
            memory_policy="eager",
            inductor_compilation="full",
            disable_passes=["cuda_graph_pass"],
            enable_autoparallel=enable_autoparallel,
        ),
    )


def torchtitan_eager_8b():
    return llama3_8b_config(seq_len=SEQ_LEN)


def torchtitan_compiled_loss_8b():
    return compiled_loss_config(llama3_8b_config(seq_len=SEQ_LEN))


def graphtrainer_manual_8b():
    return graph_trainer_config(
        llama3_8b_config(seq_len=SEQ_LEN), enable_autoparallel=False
    )


def autoparallel_graphtrainer_8b():
    return graph_trainer_config(
        llama3_8b_config(seq_len=SEQ_LEN), enable_autoparallel=True
    )


EXPERIMENT_CONFIGS = (
    "torchtitan_eager_8b",
    "torchtitan_compiled_loss_8b",
    "graphtrainer_manual_8b",
    "autoparallel_graphtrainer_8b",
)
