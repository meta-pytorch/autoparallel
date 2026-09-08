from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from dataclasses import fields, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch
from huggingface_hub import hf_hub_download

import autoparallel
import torchtitan
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.graph_trainer.muse_glimmer.config_registry import (
    MUSE_GLIMMER_C4_DATASET,
    graph_trainer_muse_glimmer_30b_sdpa_c4_4x2,
    graph_trainer_muse_glimmer_30b_sdpa_c4_autoparallel_4x2,
)
from torchtitan.experiments.graph_trainer.muse_glimmer.sdpa import (
    MuseGlimmerPackedDocumentSDPA,
)
from torchtitan.experiments.graph_trainer.passes import compile_time_passes


TOKENIZER_HASHES = {
    "tokenizer.json": "c9dbee66967b58f31a7c27f723c3760da3526ccd0427578e8905b0abb0031c4d",
    "tokenizer_config.json": "781e6c74f571642c71202167b67d9255b28cc439bdda1582ff31346182f5a9c5",
}
C4_MANIFEST_HASH = "3d226ac8afb18463cc80ab8a194d015f0a347199e93f9978882fdf6193996f22"
OFFLINE_C4_MANIFEST_HASH = (
    "5dbbab3e7ece4f96ab3158a2cb4fb4112997c628995924627c3c1eb3065d5fb0"
)
C4_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
TORCHTITAN_REVISION = "c59ce51a6fc4f2340d320fe914a7c2049b747c8f"
INPUT_MANIFEST_HASHES = {
    2: "4fda638c406cb949694e6007b404f026b1c37a33b94511aefdf2e78b14520f71",
}
AUTOPARALLEL_FIX_HASHES = {
    "autoparallel/shardings/dtensor_sharding_helpers.py": (
        "ff66549a82ebfb158b4c479e429a9a21710f5fae28432c89e94d2390cfba8bbc"
    ),
}
GRAPH_TRAINER_SHARED_HASHES = {
    "torchtitan/experiments/graph_trainer/configs.py": (
        "ce59b39c4112220d520754c2d01aa8762256f39552d88e0b572fb537cd78edf6"
    ),
    "torchtitan/experiments/graph_trainer/passes.py": (
        "600b0e24216c9c256955fc2b2b079b3679cb64c020dd0026a007644061f71274"
    ),
    "torchtitan/experiments/graph_trainer/fsdp_passes.py": (
        "de73f223323eab6117763c76b2460897cc1c8d65951c362e1eb81766e5458018"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
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


def _same_dataclass_fields(left, right, *, exclude: set[str] | None = None) -> None:
    excluded = exclude or set()
    for field in fields(left):
        if field.name in excluded:
            continue
        if getattr(left, field.name) != getattr(right, field.name):
            raise RuntimeError(f"config mismatch: {field.name}")


def _validate_offline_c4(rank: int, tp_degree: int) -> dict[str, object]:
    cache_root = Path(os.environ["HF_HUB_CACHE"]).resolve()
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("HF_HUB_OFFLINE must be 1")
    manifest_path = cache_root / "offline_manifest.json"
    if _sha256(manifest_path) != OFFLINE_C4_MANIFEST_HASH:
        raise RuntimeError("unexpected offline C4 manifest hash")
    manifest = json.loads(manifest_path.read_text())
    expected_paths = [
        f"en/c4-train.{shard:05d}-of-01024.json.gz" for shard in range(16)
    ]
    if (
        manifest.get("repo_id") != "allenai/c4"
        or manifest.get("repo_type") != "dataset"
        or manifest.get("revision") != C4_REVISION
        or [record.get("path") for record in manifest.get("files", [])]
        != expected_paths
        or manifest.get("total_bytes") != 5_101_921_073
    ):
        raise RuntimeError("unexpected offline C4 manifest contents")
    ref = cache_root / "datasets--allenai--c4" / "refs" / "main"
    if ref.read_text().strip() != C4_REVISION:
        raise RuntimeError("offline C4 main ref is not pinned")

    dp_rank = rank // tp_degree
    record = manifest["files"][dp_rank % len(manifest["files"])]
    local_file = Path(
        hf_hub_download(
            repo_id="allenai/c4",
            filename=record["path"],
            repo_type="dataset",
            revision=C4_REVISION,
            cache_dir=cache_root,
            local_files_only=True,
        )
    ).resolve()
    if local_file.stat().st_size != record["size"]:
        raise RuntimeError("offline C4 shard size mismatch")
    hash_verified = rank % tp_degree == 0 and dp_rank < len(manifest["files"])
    if hash_verified and _sha256(local_file) != record["sha256"]:
        raise RuntimeError("offline C4 shard hash mismatch")
    return {
        "cache_root": str(cache_root),
        "manifest_sha256": OFFLINE_C4_MANIFEST_HASH,
        "revision": C4_REVISION,
        "dp_rank": dp_rank,
        "audited_shard": record,
        "resolved_path": str(local_file),
        "sha256_verified": hash_verified,
    }


def _validate_inputs(
    config,
    *,
    runner_root: Path,
    fsdp_degree: int,
    dp_rank: int,
    tp_rank: int,
) -> dict[str, object]:
    local_batch_size = config.training.local_batch_size
    manifest_path = runner_root / (
        f"c4_input_manifest_lb{local_batch_size}_s4096_dp{fsdp_degree}.jsonl"
    )
    expected_manifest_hash = INPUT_MANIFEST_HASHES[local_batch_size]
    if _sha256(manifest_path) != expected_manifest_hash:
        raise RuntimeError(f"unexpected input manifest hash: {manifest_path.name}")
    manifest_records = [
        json.loads(line) for line in manifest_path.read_text().splitlines()
    ]
    expected_records = [
        record for record in manifest_records if record["dp_rank"] == dp_rank
    ]
    if len(expected_records) != 25 or [r["step"] for r in expected_records] != list(
        range(1, 26)
    ):
        raise RuntimeError(f"unexpected input manifest records for dp rank {dp_rank}")
    if tp_rank != 0:
        return {
            "manifest": manifest_path.name,
            "manifest_sha256": expected_manifest_hash,
            "dp_rank": dp_rank,
            "local_batch_size": local_batch_size,
            "verified_steps": 0,
            "verified_by_tp_rank": 0,
        }

    tokenizer = config.tokenizer.build(
        tokenizer_path=str(runner_root / "assets" / "muse_glimmer")
    )
    dataloader = config.dataloader.build(
        dp_world_size=fsdp_degree,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
        seq_len=config.training.seq_len,
        local_batch_size=config.training.local_batch_size,
        snapshot_every_n_steps=None,
    )
    iterator = iter(dataloader)
    observed = []
    for step in range(1, 26):
        input_dict, labels = next(iterator)
        observed.append(
            {
                "dp_rank": dp_rank,
                "step": step,
                "sha256": _batch_sha256(input_dict, labels),
            }
        )
    if observed != expected_records:
        raise RuntimeError(f"runtime C4 input hashes differ for dp rank {dp_rank}")
    return {
        "manifest": manifest_path.name,
        "manifest_sha256": expected_manifest_hash,
        "dp_rank": dp_rank,
        "local_batch_size": local_batch_size,
        "verified_steps": len(observed),
        "verified_by_tp_rank": 0,
    }


def _pass_name(pass_fn) -> str:
    return getattr(getattr(pass_fn, "func", pass_fn), "__name__", repr(pass_fn))


def _validate_outer_sac_pass_prefix(manual, treatment) -> dict[str, object]:
    traced_result = SimpleNamespace(state_fqns=[])
    pass_names = {
        name: [
            _pass_name(pass_fn)
            for pass_fn in compile_time_passes(
                traced_result,
                config,
                include_inductor=False,
            )
        ]
        for name, config in (("manual", manual), ("autoparallel", treatment))
    }
    expected_sac_prefix = [
        "eliminate_dead_code_pass",
        "canonicalize_graph_pass",
        "deduplicate_fsdp_unshard_chains_pass",
        "tag_with_memory_policy_pass",
        "apply_cpu_offload_pass",
        "selective_activation_remat_pass",
    ]
    for name, names in pass_names.items():
        if names[: len(expected_sac_prefix)] != expected_sac_prefix:
            raise RuntimeError(f"{name} SAC pass prefix differs: {names}")
    return {
        "status": "shared_outer_prefix_verified",
        "memory_policy": "eager",
        "shared_sac_prefix": expected_sac_prefix,
        "pre_inductor_passes": pass_names,
        "effective_graph_parity_claimed": False,
    }


def main() -> None:
    runner_root = Path(os.environ["EXPECTED_RUNNER_ROOT"]).resolve()
    output = Path(os.environ["RUN_ROOT"]) / "runtime_preflight"
    output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["EXPECTED_WORLD_SIZE"])
    node_count = int(os.environ["EXPECTED_NODE_COUNT"])
    fsdp_degree = int(os.environ["EXPECTED_FSDP_DEGREE"])
    tp_degree = int(os.environ["EXPECTED_TP_DEGREE"])
    if world_size != node_count * 8 or world_size != fsdp_degree * tp_degree:
        raise RuntimeError("inconsistent scaling mesh environment")
    dp_rank = rank // tp_degree
    tp_rank = rank % tp_degree
    offline_c4 = _validate_offline_c4(rank, tp_degree)

    expected_compile = {
        "backend": "aot_eager",
        "memory_policy": "eager",
        "pass_pipeline": "default",
        "inductor_compilation": "full",
        "numerics_changing_optim": False,
        "enable_fsdp_ag_rs_overlap": False,
        "enable_fsdp_dense_region_overlap": False,
        "disable_passes": ["cudagraph_pass"],
        "use_autoparallel_defaults": True,
    }
    configs_by_batch = {}
    for local_batch_size in (2,):
        configs = [
            graph_trainer_muse_glimmer_30b_sdpa_c4_4x2(),
            graph_trainer_muse_glimmer_30b_sdpa_c4_autoparallel_4x2(),
        ]
        for config in configs:
            config.training = replace(
                config.training,
                local_batch_size=local_batch_size,
                global_batch_size=-1,
                seq_len=4096,
            )
            config.parallelism = replace(
                config.parallelism,
                data_parallel_shard_degree=fsdp_degree,
                tensor_parallel_degree=tp_degree,
            )
        manual, treatment = configs
        for config in configs:
            if not isinstance(config.loss, CrossEntropyLoss.Config):
                raise RuntimeError("expected standard cross-entropy loss")
            if (
                config.training.local_batch_size,
                config.training.global_batch_size,
            ) != (local_batch_size, -1):
                raise RuntimeError("unexpected batch sizes")
            if (
                config.training.seq_len,
                config.parallelism.data_parallel_shard_degree,
            ) != (4096, fsdp_degree):
                raise RuntimeError("unexpected sequence length or FSDP degree")
            if config.parallelism.tensor_parallel_degree != tp_degree:
                raise RuntimeError("unexpected TP degree")
            if config.dataloader.dataset != MUSE_GLIMMER_C4_DATASET:
                raise RuntimeError("expected pinned offline C4 dataloader")
            attention_types = {
                type(layer.attention.inner_attention)
                for layer in config.model_spec.model.layers
            }
            if attention_types != {MuseGlimmerPackedDocumentSDPA.Config}:
                raise RuntimeError(
                    f"expected packed-document SDPA, got {attention_types}"
                )
        for field_name in (
            "training",
            "parallelism",
            "dataloader",
            "optimizer",
            "lr_scheduler",
            "metrics",
            "loss",
        ):
            if getattr(manual, field_name) != getattr(treatment, field_name):
                raise RuntimeError(f"manual/treatment mismatch: {field_name}")
        if not all(
            isinstance(config.activation_checkpoint, SelectiveAC.Config)
            for config in configs
        ):
            raise RuntimeError("both arms must use SelectiveAC")
        if manual.profiler != treatment.profiler:
            raise RuntimeError("manual/treatment profiler mismatch")
        _same_dataclass_fields(
            manual.compile,
            treatment.compile,
            exclude={"enable_autoparallel"},
        )
        if (
            manual.compile.enable_autoparallel
            or not treatment.compile.enable_autoparallel
        ):
            raise RuntimeError("AutoParallel is not the sole compile toggle")
        for name, config in (("manual", manual), ("treatment", treatment)):
            observed = {
                field: getattr(config.compile, field) for field in expected_compile
            }
            if observed != expected_compile:
                raise RuntimeError(
                    f"{name} does not use validated AP GraphTrainer defaults: {observed}"
                )
        configs_by_batch[local_batch_size] = configs

    manual, treatment = configs_by_batch[2]
    outer_sac_pass_audit = _validate_outer_sac_pass_prefix(manual, treatment)

    with torch.device("meta"):
        model = manual.model_spec.model.build()
    if (model.config.dim, len(model.config.layers), model.config.vocab_size) != (
        6656,
        52,
        202048,
    ):
        raise RuntimeError("unexpected Muse Glimmer 30B model shape")

    module_paths = {
        "autoparallel": Path(autoparallel.__file__).resolve(),
        "torchtitan": Path(torchtitan.__file__).resolve(),
    }
    expected_roots = {
        "autoparallel": Path(os.environ["EXPECTED_AP_ROOT"]).resolve(),
        "torchtitan": Path(os.environ["EXPECTED_TT_ROOT"]).resolve(),
    }
    for name, path in module_paths.items():
        if not path.is_relative_to(expected_roots[name]):
            raise RuntimeError(f"unexpected {name} import path: {path}")

    shared_hashes = {}
    for relative_path, expected_hash in GRAPH_TRAINER_SHARED_HASHES.items():
        path = expected_roots["torchtitan"] / relative_path
        observed_hash = _sha256(path)
        if observed_hash != expected_hash:
            raise RuntimeError(
                f"shared GraphTrainer source differs from {TORCHTITAN_REVISION}: "
                f"{relative_path} {observed_hash}"
            )
        shared_hashes[relative_path] = observed_hash
    autoparallel_fix_hashes = {}
    for relative_path, expected_hash in AUTOPARALLEL_FIX_HASHES.items():
        path = expected_roots["autoparallel"] / relative_path
        observed_hash = _sha256(path)
        if observed_hash != expected_hash:
            raise RuntimeError(
                f"AutoParallel SDPA fix source hash differs: {relative_path} "
                f"{observed_hash}"
            )
        autoparallel_fix_hashes[relative_path] = observed_hash

    tokenizer_dir = runner_root / "assets" / "muse_glimmer"
    for name, expected_hash in TOKENIZER_HASHES.items():
        if _sha256(tokenizer_dir / name) != expected_hash:
            raise RuntimeError(f"unexpected tokenizer hash: {name}")
    if (
        _sha256(runner_root / "c4_train_1588ec454efa1a09f29cd18ddd04fe05fc8653a2.json")
        != C4_MANIFEST_HASH
    ):
        raise RuntimeError("unexpected C4 manifest hash")
    input_audits = {
        "per_gpu_bs1": _validate_inputs(
            configs_by_batch[2][0],
            runner_root=runner_root,
            fsdp_degree=fsdp_degree,
            dp_rank=dp_rank,
            tp_rank=tp_rank,
        )
    }

    handlers = [
        value.strip()
        for value in os.environ.get("TITAN_STRUCT_LOGGER_HANDLERS", "").split(",")
        if value.strip()
    ]
    expected_handler = (
        "torchtitan.observability.structured_logger.jsonl_handler."
        "register_jsonl_handler"
    )
    if handlers != [expected_handler]:
        raise RuntimeError(f"unexpected structured logger handlers: {handlers}")
    for handler in handlers:
        module_name, factory_name = handler.rsplit(".", 1)
        if not callable(getattr(importlib.import_module(module_name), factory_name)):
            raise RuntimeError(f"structured logger factory is not callable: {handler}")

    payload = {
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rank": rank,
        "python": sys.version,
        "torch": torch.__version__,
        "torch_commit": torch.version.git_version,
        "cuda": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(),
        "modules": {name: str(path) for name, path in module_paths.items()},
        "model": {"dim": 6656, "layers": 52, "vocab_size": 202048},
        "mesh": {"fsdp": fsdp_degree, "tp": tp_degree},
        "batch_cases": {
            "per_gpu_bs1": {
                "local_batch_size_per_dp_rank": 2,
                "global_batch_size": 2 * fsdp_degree,
                "per_physical_gpu_effective_batch_size": 1,
                "sequence_length": 4096,
                "tokens_per_step": 2 * fsdp_degree * 4096,
            },
        },
        "world_size": world_size,
        "node_count": node_count,
        "torchtitan_revision": TORCHTITAN_REVISION,
        "graph_trainer_shared_hashes": shared_hashes,
        "autoparallel_sdpa_fix_hashes": autoparallel_fix_hashes,
        "graph_trainer_compile": expected_compile,
        "outer_sac_pass_audit": outer_sac_pass_audit,
        "attention": "packed_document_sdpa",
        "c4_revision": C4_REVISION,
        "offline_c4": offline_c4,
        "input_audits": input_audits,
        "tokenizer_hashes": TOKENIZER_HASHES,
        "c4_manifest_hash": C4_MANIFEST_HASH,
    }
    (output / f"rank_{rank:03d}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
