from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch.nn.attention import SDPBackend
from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data.loader import BaseDataLoader
from torchtitan.components.loss import CrossEntropyLoss, IGNORE_INDEX
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
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
from torchtitan.models.common.attention import ScaledDotProductAttention
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.llama3 import llama3_configs
from torchtitan.models.llama3 import model_registry as native_llama3_model_registry
from torchtitan.models.llama3.config_registry import llama3_8b

from workloads.parameter_state import register_post_load_parameter_audit

SOURCE_SEQUENCE_LENGTH = 8192
TARGET_SEQUENCE_LENGTH = 16384
REPLAY_SLOTS = 10
REPLAY_CAPACITY = 256
SUPPORTED_MESHES = {(2, 2, 4), (4, 2, 4), (8, 2, 4)}


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _batch_sha256(input_dict: dict[str, torch.Tensor], labels: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for name, tensor in [*sorted(input_dict.items()), ("labels", labels)]:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _validate_replay_payload(payload: dict[str, Any]) -> None:
    expected_shape = (REPLAY_SLOTS, REPLAY_CAPACITY, SOURCE_SEQUENCE_LENGTH)
    for name in ("input", "positions", "labels"):
        tensor = payload.get(name)
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"replay tensor {name!r} is missing")
        if tuple(tensor.shape) != expected_shape or tensor.dtype != torch.int64:
            raise RuntimeError(
                f"replay tensor {name!r} must be int64{expected_shape}, got "
                f"{tensor.dtype}{tuple(tensor.shape)}"
            )


def load_replay_payload(
    *, verify_file_hash: bool = False
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    replay_path = Path(_required_env("REPLAY_TENSORS_PATH"))
    manifest_path = Path(_required_env("REPLAY_MANIFEST_PATH"))
    manifest = json.loads(manifest_path.read_text())
    replay = manifest["replay_file"]
    if replay_path.stat().st_size != replay["size"]:
        raise RuntimeError("replay file size differs from its manifest")
    if verify_file_hash and _file_sha256(replay_path) != replay["sha256"]:
        raise RuntimeError("replay file SHA-256 differs from its manifest")
    payload = torch.load(replay_path, map_location="cpu", mmap=True, weights_only=True)
    _validate_replay_payload(payload)
    return payload, manifest, replay_path


def build_16k_batch(
    payload: dict[str, Any],
    *,
    slot: int,
    dp_rank: int,
    local_batch_size: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, list[list[int]]]:
    source_indices: list[list[int]] = []
    input_rows = []
    label_rows = []
    for local_index in range(local_batch_size):
        logical_index = dp_rank * local_batch_size + local_index
        pair = [2 * logical_index, 2 * logical_index + 1]
        if pair[-1] >= REPLAY_CAPACITY:
            raise RuntimeError(
                f"16K replay requires raw sample {pair[-1]}, but capacity is "
                f"{REPLAY_CAPACITY}"
            )
        if payload["labels"][slot, pair[0], -1] != payload["input"][slot, pair[1], 0]:
            raise RuntimeError(
                f"source replay is not continuous across sample pair {pair}"
            )
        source_indices.append(pair)
        input_rows.append(torch.cat([payload["input"][slot, index] for index in pair]))
        label_rows.append(torch.cat([payload["labels"][slot, index] for index in pair]))
    inputs = torch.stack(input_rows)
    labels = torch.stack(label_rows)
    positions = torch.arange(TARGET_SEQUENCE_LENGTH, dtype=torch.int64).repeat(
        local_batch_size, 1
    )
    return {"input": inputs, "positions": positions}, labels, source_indices


class FixedReplay16KDataLoader(BaseDataLoader):
    """Preload deterministic 16K batches; timed iteration is memory-only."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        dataset: str = "c4_topology_invariant_16k_single_causal_sequence"

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer,
        max_context_length: int,
        num_tokens_per_batch: int,
        **kwargs,
    ) -> None:
        del config, tokenizer, kwargs
        local_batch_size, remainder = divmod(
            num_tokens_per_batch, max_context_length
        )
        if remainder or local_batch_size <= 0:
            raise ValueError("Replay token batch is not rectangular")
        seq_len = max_context_length
        if seq_len != TARGET_SEQUENCE_LENGTH:
            raise ValueError(
                f"Expected sequence length {TARGET_SEQUENCE_LENGTH}, got {seq_len}"
            )
        if (dp_world_size, 2, 4) not in SUPPORTED_MESHES:
            raise ValueError(f"Unsupported DP degree {dp_world_size}")
        if local_batch_size != 2:
            raise ValueError(f"Expected local batch size 2, got {local_batch_size}")

        payload, manifest, replay_path = load_replay_payload()
        self._batches = []
        batch_records = []
        for slot in range(REPLAY_SLOTS):
            input_dict, labels, source_indices = build_16k_batch(
                payload,
                slot=slot,
                dp_rank=dp_rank,
                local_batch_size=local_batch_size,
            )
            batch_sha256 = _batch_sha256(input_dict, labels)
            input_dict["num_valid_tokens"] = int((labels != IGNORE_INDEX).sum())
            self._batches.append((input_dict, labels))
            batch_records.append(
                {
                    "slot": slot,
                    "source_indices": source_indices,
                    "sha256": batch_sha256,
                }
            )
        self._index = 0

        output = (
            Path(_required_env("BENCHMARK_OUTPUT_DIR"))
            / "replay_loader"
            / f"rank_{int(os.environ['RANK']):05d}.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "schema_version": 1,
            "rank": int(os.environ["RANK"]),
            "dp_rank": dp_rank,
            "dp_world_size": dp_world_size,
            "source_replay": str(replay_path),
            "source_replay_sha256": manifest["replay_file"]["sha256"],
            "source_sequence_length": SOURCE_SEQUENCE_LENGTH,
            "target_sequence_length": TARGET_SEQUENCE_LENGTH,
            "position_contract": "monotonic_0_to_16383_single_causal_sequence",
            "batch_records": batch_records,
            "iteration_contract": "memory_index_and_shallow_input_dict_copy_only",
        }
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    def __iter__(self) -> Iterator[dict[str, torch.Tensor | int]]:
        while True:
            input_dict, labels = self._batches[self._index]
            self._index = (self._index + 1) % len(self._batches)
            batch = dict(input_dict)
            batch["labels"] = labels
            batch["num_valid_tokens"] = int((labels != IGNORE_INDEX).sum())
            yield batch

    def state_dict(self) -> dict[str, Any]:
        return {"index": self._index}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        index = int(state_dict.get("index", 0))
        if not 0 <= index < len(self._batches):
            raise ValueError(f"Invalid replay index {index}")
        self._index = index


def _flash_sdpa_model_spec():
    ScaledDotProductAttention.sdpa_backends = [SDPBackend.FLASH_ATTENTION]
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    native = native_llama3_model_registry("8B")
    sdpa_model = build_decoder_config_for_backend(llama3_configs["8B"], "sdpa")
    for layer in sdpa_model.layers:
        layer.attention.inner_attention.sharding_config = None
    return replace(
        native,
        name="llama3/flash_sdpa_3d_current",
        model=sdpa_model,
    )


def _base_config():
    world_size = int(_required_env("BENCHMARK_WORLD_SIZE"))
    dp_degree = int(_required_env("BENCHMARK_DP_DEGREE"))
    cp_degree = int(_required_env("BENCHMARK_CP_DEGREE"))
    tp_degree = int(_required_env("BENCHMARK_TP_DEGREE"))
    local_batch_size = int(_required_env("BENCHMARK_LOCAL_BATCH_SIZE"))
    seq_len = int(_required_env("BENCHMARK_SEQ_LEN"))
    if (dp_degree, cp_degree, tp_degree) not in SUPPORTED_MESHES:
        raise ValueError(
            f"Unsupported 3D mesh {(dp_degree, cp_degree, tp_degree)}; "
            f"expected one of {sorted(SUPPORTED_MESHES)}"
        )
    if dp_degree * cp_degree * tp_degree != world_size:
        raise ValueError(
            f"DP-shard({dp_degree}) * CP({cp_degree}) * TP({tp_degree}) "
            f"must equal world size {world_size}"
        )
    if local_batch_size != 2 or seq_len != TARGET_SEQUENCE_LENGTH:
        raise ValueError(
            f"Expected local batch 2 and sequence length {TARGET_SEQUENCE_LENGTH}, "
            f"got {local_batch_size} and {seq_len}"
        )

    config = llama3_8b()
    config.model_spec = replace(
        _flash_sdpa_model_spec(),
        post_optimizer_build_fn=register_post_load_parameter_audit,
    )
    config.hf_assets_path = _required_env("LLAMA_TOKENIZER_DIR")
    config.dataloader = FixedReplay16KDataLoader.Config()
    config.loss = CrossEntropyLoss.Config(
        global_vocab_size=decoder_vocab_size(config.model_spec)
    )
    config.optimizer = default_adamw(lr=3e-4)
    config.lr_scheduler = LRSchedulersContainer.Config(warmup_steps=200)
    config.training = replace(
        config.training,
        num_tokens_per_microbatch_per_dp_rank=local_batch_size * seq_len,
        num_tokens_per_train_step=local_batch_size * dp_degree * seq_len,
        max_context_length=seq_len,
        steps=25,
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
        profile_freq=25,
        profiler_warmup=0,
        profiler_active=3,
        profiler_repeat=1,
        profiler_skip_first=0,
        enable_memory_snapshot=False,
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
        enable_structured_logging=True,
    )
    config.comm = replace(
        config.comm,
        init_timeout_seconds=1200,
        train_timeout_seconds=1200,
    )
    config.checkpoint = CheckpointManager.Config(
        enable=True,
        folder="checkpoint",
        interval=config.training.steps + 1,
        initial_load_path=_required_env("BENCHMARK_SEED_CHECKPOINT"),
        initial_load_model_only=True,
        initial_load_in_hf=True,
        last_save_model_only=True,
        last_save_in_hf=True,
        export_dtype="bfloat16",
        async_mode="disabled",
        load_only=True,
    )
    return config


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
    from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
    from torchtitan.experiments.graph_trainer.common_utils import apply_simple_fsdp
    from torchtitan.experiments.graph_trainer.compile import apply_compile
    from torchtitan.experiments.graph_trainer.llama3.parallelize import annotate_llama
    from torchtitan.experiments.graph_trainer.llama3.parallelize_autoparallel import (
        _apply_autoparallel_context_parallel_attention,
        _build_autoparallel_mesh,
    )

    del ac_config
    if training.max_context_length % parallel_dims.seq_len_divisor:
        raise ValueError("Sequence length is incompatible with the 3D mesh")
    _apply_autoparallel_context_parallel_attention(
        model, _build_autoparallel_mesh(parallel_dims)
    )
    annotate_llama(model)
    model.parallelize(parallel_dims)
    maybe_enable_async_tp(parallelism, compile_config, parallel_dims.get_mesh("tp"))
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


def _parallelize_fixed_3d(model, *, compile_config, **kwargs):
    if compile_config.enable_autoparallel:
        from workloads.llama3_batched_autoparallel import (
            parallelize_batched_autoparallel_llama,
        )

        return parallelize_batched_autoparallel_llama(
            model,
            compile_config=compile_config,
            **kwargs,
        )
    return _parallelize_manual_with_autoparallel_cp(
        model,
        compile_config=compile_config,
        **kwargs,
    )


def _graph_config(*, enable_autoparallel: bool):
    config = to_graph_trainer_config(_base_config(), graph_llama3_model_registry)
    config.model_spec = replace(
        config.model_spec,
        name="graphtrainer/llama3_3d_current",
        parallelize_fn=_parallelize_fixed_3d,
    )
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
        enable_autoparallel=enable_autoparallel,
        autoparallel_solver="approx",
    )
    return config


def torchtitan_main_manual_jit_8b():
    config = _base_config()
    config.compile = CompileConfig(
        enable=True,
        components=["model", "loss"],
        backend="inductor",
    )
    return config


def graphtrainer_manual_full_inductor_8b():
    return _graph_config(enable_autoparallel=False)


def autoparallel_graphtrainer_full_inductor_8b():
    return _graph_config(enable_autoparallel=True)
