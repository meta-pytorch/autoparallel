from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from torchtitan.components.dataloader import BaseDataLoader


C4_REPO = "allenai/c4"
C4_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
C4_SHARD_FILE = "en/c4-train.00000-of-01024.json.gz"
C4_SHARD_SIZE = 319308785
C4_SHARD_SHA256 = "8ef8d75b0e045dec4aa5123a671b4564466b0707086a7ed1ba8721626dfffbc9"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(file_sha256(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def case_name(seq_len: int, local_batch_size: int) -> str:
    return f"s{seq_len}_lb{local_batch_size}"


class FixedC4ReplayDataLoader(BaseDataLoader):
    """Serve pre-tokenized C4 batches without I/O in the measured iterator."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        dataset: str = "llama3_replanning_fixed_c4"

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer,
        seq_len: int,
        local_batch_size: int,
        **kwargs,
    ) -> None:
        del config, tokenizer, kwargs
        replay_root = Path(os.environ["BENCHMARK_REPLAY_ROOT"])
        selected_case = os.environ["BENCHMARK_REPLAY_CASE"]
        expected_case = case_name(seq_len, local_batch_size)
        if selected_case != expected_case:
            raise RuntimeError(
                f"Replay case {selected_case!r} does not match runtime "
                f"{expected_case!r}"
            )

        manifest = json.loads((replay_root / "manifest.json").read_text())
        if manifest.get("schema_version") != 1:
            raise RuntimeError("Unsupported replanning replay manifest version")
        if manifest.get("dp_degree") != dp_world_size:
            raise RuntimeError(
                f"Replay DP degree {manifest.get('dp_degree')} != {dp_world_size}"
            )
        entry = manifest.get("cases", {}).get(selected_case)
        if not isinstance(entry, dict):
            raise RuntimeError(f"Replay manifest has no case {selected_case!r}")
        if (
            entry.get("seq_len") != seq_len
            or entry.get("local_batch_size") != local_batch_size
        ):
            raise RuntimeError(f"Replay case metadata mismatch for {selected_case!r}")

        replay_path = replay_root / entry["file"]
        if replay_path.stat().st_size != entry["size"]:
            raise RuntimeError(f"Replay file size mismatch: {replay_path}")
        tensors = torch.load(
            replay_path, map_location="cpu", mmap=True, weights_only=True
        )
        expected_shape = tuple(entry["shape"])
        runtime_shape = (
            manifest.get("slots"),
            dp_world_size * local_batch_size,
            seq_len,
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
                raise RuntimeError(
                    f"Replay tensor {name!r} does not match {expected_shape} int64"
                )

        first = dp_rank * local_batch_size
        last = first + local_batch_size
        if last > expected_shape[1]:
            raise RuntimeError(
                f"DP rank {dp_rank} local slice [{first}:{last}] exceeds replay batch"
            )
        self._inputs = tensors["input"][:, first:last]
        self._positions = tensors["positions"][:, first:last]
        self._labels = tensors["labels"][:, first:last]
        self._index = 0

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        while True:
            slot = self._index % self._inputs.shape[0]
            self._index += 1
            yield {
                "input": self._inputs[slot],
                "positions": self._positions[slot],
            }, self._labels[slot]

    def state_dict(self) -> dict[str, Any]:
        return {"index": self._index}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        index = int(state_dict.get("index", 0))
        if index < 0:
            raise ValueError(f"Invalid replay index {index}")
        self._index = index
