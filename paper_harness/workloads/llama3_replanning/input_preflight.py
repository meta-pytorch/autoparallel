from __future__ import annotations

import json
from pathlib import Path

import torch

from .replay_data import (
    C4_REPO,
    C4_REVISION,
    C4_SHARD_FILE,
    C4_SHARD_SHA256,
    C4_SHARD_SIZE,
    file_sha256,
    tensor_sha256,
    tree_sha256,
)


def audit(
    *,
    payload: Path,
    resolved: dict,
    output: Path,
    environment: dict[str, str],
    rank: int,
) -> dict:
    del payload, output
    if rank != 0:
        return {"status": "skipped", "reason": "rank 0 validates shared assets"}

    replay_root = Path(environment["BENCHMARK_REPLAY_ROOT"])
    manifest_path = replay_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("dataset") != C4_REPO
        or manifest.get("revision") != C4_REVISION
        or manifest.get("c4_file") != C4_SHARD_FILE
        or manifest.get("c4_file_size") != C4_SHARD_SIZE
        or manifest.get("c4_file_sha256") != C4_SHARD_SHA256
    ):
        raise RuntimeError("Replay manifest provenance does not match pinned C4")
    tokenizer_hash = tree_sha256(Path(environment["LLAMA_TOKENIZER_DIR"]))
    if tokenizer_hash != manifest.get("tokenizer_tree_sha256"):
        raise RuntimeError("Tokenizer tree differs from the replay manifest")

    selected_case = environment["BENCHMARK_REPLAY_CASE"]
    entry = manifest.get("cases", {}).get(selected_case)
    if not isinstance(entry, dict):
        raise RuntimeError(f"Replay manifest has no case {selected_case!r}")
    expected = {
        "seq_len": resolved["training"]["seq_len"],
        "local_batch_size": resolved["training"]["local_batch_size"],
        "global_batch_size": resolved["training"]["global_batch_size"],
    }
    mismatches = {
        key: {"expected": value, "actual": entry.get(key)}
        for key, value in expected.items()
        if entry.get(key) != value
    }
    if (
        manifest.get("dp_degree")
        != resolved["parallelism"]["data_parallel_shard_degree"]
    ):
        mismatches["dp_degree"] = {
            "expected": resolved["parallelism"]["data_parallel_shard_degree"],
            "actual": manifest.get("dp_degree"),
        }
    if mismatches:
        raise RuntimeError(f"Replay metadata mismatch: {mismatches}")

    replay_path = replay_root / entry["file"]
    replay_hash = file_sha256(replay_path)
    if replay_path.stat().st_size != entry["size"] or replay_hash != entry["sha256"]:
        raise RuntimeError(f"Replay file differs from manifest: {replay_path}")
    tensors = torch.load(replay_path, map_location="cpu", mmap=True, weights_only=True)
    observed_hashes = {}
    expected_shape = tuple(entry["shape"])
    runtime_shape = (
        manifest.get("slots"),
        expected["global_batch_size"],
        expected["seq_len"],
    )
    if expected_shape != runtime_shape:
        raise RuntimeError(
            f"Replay shape {expected_shape} does not match runtime {runtime_shape}"
        )
    for name in ("input", "positions", "labels"):
        tensor = tensors.get(name)
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.int64
            or tuple(tensor.shape) != expected_shape
        ):
            raise RuntimeError(f"Replay tensor {name!r} has an invalid shape or dtype")
        observed_hashes[name] = tensor_sha256(tensor)
    if observed_hashes != entry["tensor_sha256"]:
        raise RuntimeError("Replay tensor hashes differ from the manifest")

    canonical = None
    placement_text = environment.get("CANONICAL_PLACEMENT_PATH", "")
    expected_placement_hash = environment.get("EXPECTED_CANONICAL_SHA256", "")
    if bool(placement_text) != bool(expected_placement_hash):
        raise RuntimeError(
            "CANONICAL_PLACEMENT_PATH and EXPECTED_CANONICAL_SHA256 must be "
            "set together"
        )
    if placement_text:
        placement_path = Path(placement_text)
        placement_hash = file_sha256(placement_path)
        if placement_hash != expected_placement_hash:
            raise RuntimeError(
                "Canonical placement hash mismatch: "
                f"{placement_hash} != {expected_placement_hash}"
            )
        placement = json.loads(placement_path.read_text())
        if (
            placement.get("version") != 1
            or placement.get("mesh_shape") != [4, 8]
            or placement.get("mesh_dim_names") != ["fsdp", "tp"]
            or not placement.get("placements")
        ):
            raise RuntimeError("Canonical placement is not a nonempty 4x8 fsdp/tp plan")
        canonical = {
            "path": str(placement_path),
            "sha256": placement_hash,
            "placement_nodes": len(placement["placements"]),
        }

    return {
        "status": "passed",
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "tokenizer_tree_sha256": tokenizer_hash,
        "case": selected_case,
        "replay": {
            "path": str(replay_path),
            "sha256": replay_hash,
            "shape": list(expected_shape),
            "tensor_sha256": observed_hashes,
        },
        "canonical_placement": canonical,
    }
