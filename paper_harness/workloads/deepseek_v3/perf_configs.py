from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import torch
from datasets import Features, Value, load_dataset
from torch.distributed.tensor import DTensor
from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.optimizer import (
    default_adamw,
    register_moe_load_balancing_hook,
)
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import CompileConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.deepseek_v3 import (
    model_registry as graph_model_registry,
)
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.hf_datasets.text_datasets import (
    DATASETS,
    HuggingFaceTextDataLoader,
    HuggingFaceTextDataset,
)
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.deepseek_v3.config_registry import deepseek_v3_16b

C4_REPO = "allenai/c4"
C4_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
C4_TRAIN_SHARDS = 1024
C4_SHARD_METADATA = {
    0: (
        319308785,
        "8ef8d75b0e045dec4aa5123a671b4564466b0707086a7ed1ba8721626dfffbc9",
    ),
    1: (
        318039285,
        "b945059cd1a343cabe311881b7840a6f0363f570e745a0eff0e687e266f6b55d",
    ),
    2: (
        319748667,
        "2967dc7e587ced6ecb9ba617ad2d4c44901467969de5bf5b0f5a9e5b70555d75",
    ),
    3: (
        318564193,
        "b79d9abef5741578929be0d59db9ca652a8276207ef18a944b7a5f11fef5beb6",
    ),
}
DATASET_NAME = "c4_deepseek16b_pinned_audited"

MODEL_FLAVOR = os.environ.get("BENCHMARK_MODEL", "16B")
if MODEL_FLAVOR != "16B":
    raise ValueError(f"Unsupported BENCHMARK_MODEL={MODEL_FLAVOR!r}")

WORLD_SIZE = int(os.environ["BENCHMARK_WORLD_SIZE"])
EP_DEGREE = int(os.environ["BENCHMARK_EP_DEGREE"])
TP_DEGREE = int(os.environ.get("BENCHMARK_TP_DEGREE", "1"))
if WORLD_SIZE % TP_DEGREE:
    raise ValueError(
        f"Tensor parallel degree must divide world size: {WORLD_SIZE=} {TP_DEGREE=}"
    )
if WORLD_SIZE % EP_DEGREE or EP_DEGREE % TP_DEGREE:
    raise ValueError(
        "Expert parallelism must contain TP and divide the world: "
        f"{WORLD_SIZE=} {EP_DEGREE=} {TP_DEGREE=}"
    )
DP_DEGREE = WORLD_SIZE // TP_DEGREE
EFSDP_DEGREE = WORLD_SIZE // EP_DEGREE

LOCAL_BATCH_SIZE = int(os.environ.get("BENCHMARK_LOCAL_BATCH_SIZE", "4"))
if LOCAL_BATCH_SIZE < 1:
    raise ValueError(f"Local batch size must be positive: {LOCAL_BATCH_SIZE}")
SEQ_LEN = 4096
GLOBAL_BATCH_SIZE = LOCAL_BATCH_SIZE * DP_DEGREE


def _write_inductor_path_audit() -> None:
    """Assert and record the configuration-specific Inductor hook state."""
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
    custom_post_pass = torch._inductor.config.post_grad_custom_post_pass
    configuration = os.environ["BENCHMARK_CONFIGURATION"]
    expected_patch_active = configuration in {
        "autoparallel_graphtrainer",
        "autoparallel_backend_example_scheduling",
    }
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
    expected_process_custom_pass = (
        configuration == "autoparallel_backend_example_scheduling"
    )
    if (custom_post_pass is not None) != expected_process_custom_pass:
        raise RuntimeError(
            f"Unexpected custom post-grad scheduler for {configuration}: "
            f"present={custom_post_pass is not None}"
        )
    if custom_post_pass is not None and not callable(custom_post_pass):
        raise RuntimeError("Configured post-grad scheduler is not callable")

    audit_dir = Path(os.environ["MODULE_ISOLATION_AUDIT_DIR"])
    audit_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "rank": int(os.environ["RANK"]),
        "configuration": configuration,
        "autoparallel_bucketing_hook_active": patch_active,
        "autoparallel_bucketing_module_loaded": module_loaded,
        "functions": functions,
        "post_grad_custom_post_pass_present": custom_post_pass is not None,
        "post_grad_custom_post_pass_callable": (
            None
            if custom_post_pass is None
            else f"{custom_post_pass.__module__}.{custom_post_pass.__qualname__}"
        ),
    }
    output = audit_dir / f"rank_{int(os.environ['RANK']):02d}.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def _normalize_parameter_name(name: str) -> str:
    prefixes = ("model._orig_mod.", "_orig_mod.", "model.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name.removeprefix(prefix)
                changed = True
    return (
        name.replace(
            ".moe.experts.w1",
            ".moe.routed_experts.inner_experts.w1_EFD",
        )
        .replace(
            ".moe.experts.w2",
            ".moe.routed_experts.inner_experts.w2_EDF",
        )
        .replace(
            ".moe.experts.w3",
            ".moe.routed_experts.inner_experts.w3_EFD",
        )
    )


def _audit_model_parameters(_optimizers, model_parts, _parallel_dims):
    """Record shard-local initialized weights without changing training state."""
    _write_inductor_path_audit()
    audit_dir = os.environ.get("PARAMETER_AUDIT_DIR")
    if audit_dir is None:
        return

    records = []
    with torch.no_grad():
        for part_index, model_part in enumerate(model_parts):
            for name, parameter in model_part.named_parameters():
                local = (
                    parameter.to_local()
                    if isinstance(parameter, DTensor)
                    else parameter
                )
                flat = local.detach().reshape(-1)
                sample = flat[: min(flat.numel(), 8192)].contiguous()
                sample_bytes = sample.view(torch.uint8).cpu().numpy().tobytes()
                float_local = local.detach().float()
                records.append(
                    {
                        "part": part_index,
                        "name": _normalize_parameter_name(name),
                        "raw_name": name,
                        "global_shape": list(parameter.shape),
                        "local_shape": list(local.shape),
                        "dtype": str(parameter.dtype),
                        "sample_sha256": hashlib.sha256(sample_bytes).hexdigest(),
                        "sum": float(float_local.sum().item()),
                        "square_sum": float(float_local.square().sum().item()),
                    }
                )

    output_dir = Path(audit_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"rank_{int(os.environ['RANK']):02d}.json"
    output.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")


def _audit_and_register_moe_hook(optimizers, model_parts, parallel_dims):
    _audit_model_parameters(optimizers, model_parts, parallel_dims)
    register_moe_load_balancing_hook(optimizers, model_parts, parallel_dims)


def _load_c4_shard(dataset_path: str, *, shard_index: int):
    if dataset_path != C4_REPO:
        raise ValueError(f"Expected C4 path {C4_REPO!r}, got {dataset_path!r}")
    if not 0 <= shard_index < C4_TRAIN_SHARDS:
        raise ValueError(f"Invalid C4 train shard index {shard_index}")
    if shard_index not in C4_SHARD_METADATA:
        raise ValueError(f"Offline C4 shard {shard_index} was not packaged")
    filename = f"en/c4-train.{shard_index:05d}-of-{C4_TRAIN_SHARDS:05d}.json.gz"
    expected_size, expected_sha256 = C4_SHARD_METADATA[shard_index]
    local_file = Path(os.environ["C4_SHARD_ROOT"]) / filename
    if not local_file.is_file():
        raise FileNotFoundError(f"Required offline C4 shard is missing: {local_file}")
    if local_file.stat().st_size != expected_size:
        raise RuntimeError(
            f"Offline C4 shard size mismatch for {local_file}: "
            f"{local_file.stat().st_size} != {expected_size}"
        )
    digest = hashlib.sha256()
    with local_file.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise RuntimeError(
            f"Offline C4 shard SHA-256 mismatch for {local_file}: "
            f"{digest.hexdigest()} != {expected_sha256}"
        )
    return load_dataset(
        "json",
        data_files={"train": [str(local_file)]},
        features=Features(
            {
                "text": Value("string"),
                "timestamp": Value("string"),
                "url": Value("string"),
            }
        ),
        split="train",
        streaming=True,
    )


DATASETS[DATASET_NAME] = DatasetConfig(
    path=C4_REPO,
    loader=partial(_load_c4_shard, shard_index=0),
    sample_processor=lambda sample: sample["text"],
)


class PinnedC4Dataset(HuggingFaceTextDataset):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: BaseTokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:
        if dp_world_size != DP_DEGREE:
            raise ValueError(
                f"Expected C4 dp_world_size={DP_DEGREE}, got {dp_world_size}"
            )
        rank_dataset_name = f"{DATASET_NAME}_dp{dp_rank}"
        shard_index = dp_rank % len(C4_SHARD_METADATA)
        DATASETS[rank_dataset_name] = DatasetConfig(
            path=C4_REPO,
            loader=partial(_load_c4_shard, shard_index=shard_index),
            sample_processor=lambda sample: sample["text"],
        )
        super().__init__(
            dataset_name=rank_dataset_name,
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            seq_len=seq_len,
            dp_rank=0,
            dp_world_size=1,
            infinite=infinite,
        )
        self.dataset_name = DATASET_NAME


class AuditedC4DataLoader(ParallelAwareDataloader):
    @dataclass(kw_only=True, slots=True)
    class Config(HuggingFaceTextDataLoader.Config):
        pass

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        seq_len: int,
        local_batch_size: int,
        snapshot_every_n_steps: int | None = 1,
        **kwargs,
    ) -> None:
        del kwargs
        dataset = PinnedC4Dataset(
            dataset_name=config.dataset,
            dataset_path=config.dataset_path,
            tokenizer=tokenizer,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=config.infinite,
        )
        super().__init__(
            dataset,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            num_workers=config.num_workers,
            persistent_workers=config.persistent_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
            snapshot_every_n_steps=snapshot_every_n_steps,
            batch_size=local_batch_size,
        )

    def __iter__(self):
        # Input identity is fixed and audited by the campaign index manifest.
        # Keep the timed iterator free of tensor hashing and filesystem writes.
        replay_size = int(os.environ.get("FIXED_REPLAY_BATCHES", "10"))
        if replay_size < 1:
            raise ValueError("FIXED_REPLAY_BATCHES must be positive")
        source = super().__iter__()
        replay_batches = [next(source) for _ in range(replay_size)]
        batch_index = 0
        while True:
            input_dict, labels = replay_batches[batch_index % replay_size]
            batch_index += 1
            yield dict(input_dict), labels


def _base_config():
    config = deepseek_v3_16b()
    config.comm = replace(
        config.comm,
        init_timeout_seconds=1200,
        train_timeout_seconds=1200,
    )
    config.model_spec = graph_model_registry(MODEL_FLAVOR, attn_backend="sdpa")
    config.model_spec = replace(
        config.model_spec,
        post_optimizer_build_fn=_audit_and_register_moe_hook,
    )
    config.hf_assets_path = os.environ["DEEPSEEK_TOKENIZER_DIR"]
    config.loss = CrossEntropyLoss.Config(
        global_vocab_size=decoder_vocab_size(config.model_spec)
    )
    config.dataloader = AuditedC4DataLoader.Config(dataset=DATASET_NAME)
    config.optimizer = default_adamw(lr=2.2e-4)
    config.training = replace(
        config.training,
        local_batch_size=LOCAL_BATCH_SIZE,
        global_batch_size=GLOBAL_BATCH_SIZE,
        seq_len=SEQ_LEN,
        steps=28,
        dtype="float32",
        mixed_precision_param="bfloat16",
        mixed_precision_reduce="float32",
        max_norm=1.0,
    )
    config.parallelism = replace(
        config.parallelism,
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=DP_DEGREE,
        tensor_parallel_degree=TP_DEGREE,
        enable_sequence_parallel=TP_DEGREE > 1,
        context_parallel_degree=1,
        pipeline_parallel_degree=1,
        expert_parallel_degree=EP_DEGREE,
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
        enable_profiling=int(os.environ.get("RANK", "0")) == 0,
        profile_freq=28,
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
    return config


def autoparallel_graphtrainer_16b():
    config = to_graph_trainer_config(_base_config(), graph_model_registry)
    config.model_spec = replace(
        config.model_spec,
        post_optimizer_build_fn=_audit_and_register_moe_hook,
    )
    config.profiler = replace(config.profiler, trace_post_processor=None)
    config.compile = GraphTrainerCompileConfig(
        enable=True,
        components=["model", "loss"],
        inductor_compilation="full",
        disable_passes=["cudagraph_pass"],
        enable_autoparallel=True,
    )
    return config


def graphtrainer_manual_16b():
    config = autoparallel_graphtrainer_16b()
    config.compile = replace(config.compile, enable_autoparallel=False)
    return config


def torchtitan_baseline_16b():
    if "autoparallel.graph_passes.auto_bucketing" in sys.modules:
        raise RuntimeError("Native TorchTitan baseline was polluted by AutoParallel")
    from torchtitan.models.deepseek_v3.parallelize import parallelize_deepseekv3

    config = _base_config()
    config.model_spec = replace(
        config.model_spec,
        name="torchtitan/native_compile/deepseek_v3",
        parallelize_fn=parallelize_deepseekv3,
    )
    config.compile = CompileConfig(
        enable=True,
        components=["model", "loss"],
        backend="inductor",
    )
    return config


EXPERIMENT_CONFIGS = (
    "torchtitan_baseline_16b",
    "graphtrainer_manual_16b",
    "autoparallel_graphtrainer_16b",
)
