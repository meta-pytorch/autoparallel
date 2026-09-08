from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


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
        return {"status": "skipped", "reason": "rank 0 validates the shared replay"}
    manifest_path = Path(environment["REPLAY_MANIFEST_PATH"])
    replay_path = Path(environment["REPLAY_TENSORS_PATH"])
    manifest = json.loads(manifest_path.read_text())
    replay = manifest["replay_file"]
    if replay_path.stat().st_size != replay["size"] or _sha256(replay_path) != replay["sha256"]:
        raise RuntimeError("replay file differs from its manifest")
    tensors = torch.load(replay_path, map_location="cpu", mmap=True, weights_only=True)
    expected_shape = tuple(manifest["shape"])
    observed = {}
    for name in ("input", "positions", "labels"):
        tensor = tensors.get(name)
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"replay tensor {name!r} is missing")
        if tuple(tensor.shape) != expected_shape or tensor.dtype != torch.int64:
            raise RuntimeError(f"replay tensor {name!r} has the wrong shape or dtype")
        observed[name] = _tensor_sha256(tensor)
    if observed != manifest["tensor_sha256"]:
        raise RuntimeError("replay tensor hashes differ from the manifest")
    parallelism = resolved["parallelism"]
    return {
        "status": "passed",
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "replay_sha256": replay["sha256"],
        "tensor_sha256": observed,
        "shape": list(expected_shape),
        "logical_mapping": {
            "dp_degree": parallelism["data_parallel_shard_degree"],
            "gradient_accumulation_steps": resolved["training"][
                "gradient_accumulation_steps"
            ],
        },
    }
