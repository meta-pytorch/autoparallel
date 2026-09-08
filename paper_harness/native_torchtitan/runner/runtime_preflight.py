from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

import torchtitan
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.graph_trainer.muse_glimmer.config_registry import (
    MUSE_GLIMMER_C4_DATASET,
    muse_glimmer_30b_sdpa_c4_torchtitan_4x2,
)
from torchtitan.experiments.graph_trainer.muse_glimmer.sdpa import (
    MuseGlimmerPackedDocumentSDPA,
)
from torchtitan.experiments.graph_trainer.muse_glimmer.trainer import (
    MuseGlimmerPackedSDPATrainer,
)


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
INPUT_MANIFEST_HASH = "4fda638c406cb949694e6007b404f026b1c37a33b94511aefdf2e78b14520f71"
NATIVE_SOURCE_HASHES = {
    "torchtitan/models/muse_glimmer/parallelize.py": (
        "350e841227ee18f17a15a1d666aa7bce9c2540423f8449af455c0e6d87e3ec8c"
    ),
    "torchtitan/distributed/compile.py": (
        "02362dcf6cd4ec63ad4a16d110c8c6e6a3ac1e258108fdc684e5fcfdd566d289"
    ),
    "torchtitan/distributed/fsdp.py": (
        "0a0c2d99897ad94ec687416db61d1e7906679c08482a0798e5691dbd70e9cbf7"
    ),
    "torchtitan/experiments/graph_trainer/muse_glimmer/config_registry.py": (
        "1af988838ef154ea6cb6697b60037e3e03b95a33f5ec60defa5e9aecd3b3a1eb"
    ),
    "torchtitan/experiments/graph_trainer/muse_glimmer/trainer.py": (
        "1aa37297aeaf97acc09a9482d885174e6e7a1098f0f31716d0f0658187c4693a"
    ),
    "torchtitan/experiments/graph_trainer/muse_glimmer/sdpa.py": (
        "9d474ebabc8b470e97e4c6cbc3b85a3cd3d57b2d52895955920988d9e5b973f9"
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
    manifest_path = runner_root / "c4_input_manifest_lb2_s4096_dp8.jsonl"
    if _sha256(manifest_path) != INPUT_MANIFEST_HASH:
        raise RuntimeError("unexpected input manifest hash")
    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    expected = [record for record in records if record["dp_rank"] == dp_rank]
    if len(expected) != 25 or [record["step"] for record in expected] != list(
        range(1, 26)
    ):
        raise RuntimeError(f"unexpected input records for DP rank {dp_rank}")
    if tp_rank != 0:
        return {
            "manifest": manifest_path.name,
            "manifest_sha256": INPUT_MANIFEST_HASH,
            "dp_rank": dp_rank,
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
    if observed != expected:
        raise RuntimeError(f"runtime C4 input hashes differ for DP rank {dp_rank}")
    return {
        "manifest": manifest_path.name,
        "manifest_sha256": INPUT_MANIFEST_HASH,
        "dp_rank": dp_rank,
        "verified_steps": len(observed),
        "verified_by_tp_rank": 0,
    }


def main() -> None:
    runner_root = Path(os.environ["EXPECTED_RUNNER_ROOT"]).resolve()
    tt_root = Path(os.environ["EXPECTED_TT_ROOT"]).resolve()
    output = Path(os.environ["RUN_ROOT"]) / "runtime_preflight"
    output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["EXPECTED_WORLD_SIZE"])
    node_count = int(os.environ["EXPECTED_NODE_COUNT"])
    fsdp_degree = int(os.environ["EXPECTED_FSDP_DEGREE"])
    tp_degree = int(os.environ["EXPECTED_TP_DEGREE"])
    if (world_size, node_count, fsdp_degree, tp_degree) != (16, 2, 8, 2):
        raise RuntimeError("unexpected native benchmark mesh")
    dp_rank = rank // tp_degree
    tp_rank = rank % tp_degree

    config = muse_glimmer_30b_sdpa_c4_torchtitan_4x2()
    if not isinstance(config, MuseGlimmerPackedSDPATrainer.Config):
        raise RuntimeError("config must use the native TorchTitan Trainer")
    config.training = replace(
        config.training,
        local_batch_size=2,
        global_batch_size=-1,
        seq_len=4096,
    )
    config.parallelism = replace(
        config.parallelism,
        data_parallel_shard_degree=fsdp_degree,
        tensor_parallel_degree=tp_degree,
    )
    config.compile = replace(config.compile, enable=True, backend="inductor")

    if not isinstance(config.loss, CrossEntropyLoss.Config):
        raise RuntimeError("expected standard cross-entropy loss")
    if not isinstance(config.activation_checkpoint, SelectiveAC.Config):
        raise RuntimeError("expected native SelectiveAC")
    if config.dataloader.dataset != MUSE_GLIMMER_C4_DATASET:
        raise RuntimeError("expected pinned offline C4 dataloader")
    attention_types = {
        type(layer.attention.inner_attention)
        for layer in config.model_spec.model.layers
    }
    if attention_types != {MuseGlimmerPackedDocumentSDPA.Config}:
        raise RuntimeError(f"expected packed-document SDPA, got {attention_types}")
    expected_compile = {
        "enable": True,
        "enable_async_tensor_parallel": False,
        "components": ["model", "loss"],
        "backend": "inductor",
    }
    observed_compile = {
        field: getattr(config.compile, field) for field in expected_compile
    }
    if observed_compile != expected_compile:
        raise RuntimeError(f"unexpected native compile config: {observed_compile}")
    parallelize_fn = config.model_spec.parallelize_fn
    if (
        parallelize_fn.__module__ != "torchtitan.models.muse_glimmer.parallelize"
        or parallelize_fn.__name__ != "parallelize_muse_glimmer"
    ):
        raise RuntimeError(f"unexpected native parallelize function: {parallelize_fn}")

    with torch.device("meta"):
        model = config.model_spec.model.build()
    if (model.config.dim, len(model.config.layers), model.config.vocab_size) != (
        6656,
        52,
        202048,
    ):
        raise RuntimeError("unexpected Muse Glimmer 30B model shape")

    torchtitan_path = Path(torchtitan.__file__).resolve()
    if not torchtitan_path.is_relative_to(tt_root):
        raise RuntimeError(f"unexpected TorchTitan import path: {torchtitan_path}")
    source_hashes = {}
    for relative_path, expected_hash in NATIVE_SOURCE_HASHES.items():
        path = tt_root / relative_path
        observed_hash = _sha256(path)
        if observed_hash != expected_hash:
            raise RuntimeError(f"unexpected native source hash: {relative_path}")
        source_hashes[relative_path] = observed_hash

    tokenizer_dir = runner_root / "assets" / "muse_glimmer"
    for name, expected_hash in TOKENIZER_HASHES.items():
        if _sha256(tokenizer_dir / name) != expected_hash:
            raise RuntimeError(f"unexpected tokenizer hash: {name}")
    c4_manifest_path = (
        runner_root / "c4_train_1588ec454efa1a09f29cd18ddd04fe05fc8653a2.json"
    )
    if _sha256(c4_manifest_path) != C4_MANIFEST_HASH:
        raise RuntimeError("unexpected C4 manifest hash")

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
        "torchtitan_module": str(torchtitan_path),
        "torchtitan_revision": TORCHTITAN_REVISION,
        "native_source_hashes": source_hashes,
        "trainer_backend": "native_torchtitan",
        "parallelize_fn": f"{parallelize_fn.__module__}.{parallelize_fn.__name__}",
        "native_compile": observed_compile,
        "activation_checkpoint": "SelectiveAC",
        "attention": "packed_document_sdpa",
        "model": {"dim": 6656, "layers": 52, "vocab_size": 202048},
        "mesh": {"fsdp": fsdp_degree, "tp": tp_degree},
        "batch": {
            "local_batch_size_per_dp_rank": 2,
            "global_batch_size": 16,
            "per_physical_gpu_effective_batch_size": 1,
            "sequence_length": 4096,
            "tokens_per_step": 65536,
        },
        "world_size": world_size,
        "node_count": node_count,
        "c4_revision": C4_REVISION,
        "offline_c4": _validate_offline_c4(rank, tp_degree),
        "input_audit": _validate_inputs(
            config,
            runner_root=runner_root,
            fsdp_degree=fsdp_degree,
            dp_rank=dp_rank,
            tp_rank=tp_rank,
        ),
        "tokenizer_hashes": TOKENIZER_HASHES,
        "c4_manifest_hash": C4_MANIFEST_HASH,
    }
    (output / f"rank_{rank:03d}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
