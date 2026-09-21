from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torchtitan.components.dataloader import BaseDataLoader

from .io_utils import atomic_write_json, file_sha256, required_env


def hash_batch(input_dict: dict[str, torch.Tensor], labels: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for name, tensor in [*sorted(input_dict.items()), ("labels", labels)]:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class ReplayDataLoader(BaseDataLoader):
    """Load a DP-rank replay once; iteration performs no disk or network I/O."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        dataset: str = "materialized_c4_replay"

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
        started = time.perf_counter()
        shared_replay_path = os.environ.get("REPLAY_TENSORS_PATH")
        if shared_replay_path:
            replay_path = Path(shared_replay_path)
            manifest_path = Path(required_env("REPLAY_MANIFEST_PATH"))
            manifest = json.loads(manifest_path.read_text())
            replay = manifest["replay_file"]
            actual_file_hash = file_sha256(replay_path)
            if (
                replay_path.stat().st_size != replay["size"]
                or actual_file_hash != replay["sha256"]
            ):
                raise RuntimeError("Shared replay file differs from its manifest")
            payload = torch.load(
                replay_path, map_location="cpu", mmap=True, weights_only=True
            )
            expected_shape = tuple(manifest["shape"])
            if expected_shape[1] < dp_world_size * local_batch_size:
                raise RuntimeError(
                    "Shared replay does not contain enough samples for "
                    f"{dp_world_size=} and {local_batch_size=}"
                )
            for name in ("input", "positions", "labels"):
                tensor = payload.get(name)
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tuple(tensor.shape) != expected_shape
                    or tensor.dtype != torch.int64
                ):
                    raise RuntimeError(
                        f"Shared replay tensor {name!r} has the wrong shape or dtype"
                    )
            if expected_shape[2] != seq_len:
                raise RuntimeError(
                    f"Shared replay sequence length {expected_shape[2]} != {seq_len}"
                )
            first = dp_rank * local_batch_size
            last = first + local_batch_size
            self._shared_payload = payload
            self._shared_slice = slice(first, last)
            self._shared_slots = expected_shape[0]
            self._batch_hashes = [
                hash_batch(
                    {
                        "input": payload["input"][slot, first:last],
                        "positions": payload["positions"][slot, first:last],
                    },
                    payload["labels"][slot, first:last],
                )
                for slot in range(self._shared_slots)
            ]
            self._batches = None
            self._index = 0
            audit_dir = Path(required_env("BENCHMARK_OUTPUT_DIR")) / "replay_loader"
            atomic_write_json(
                audit_dir / f"rank_{int(os.environ['RANK']):05d}.json",
                {
                    "rank": int(os.environ["RANK"]),
                    "dp_rank": dp_rank,
                    "dp_world_size": dp_world_size,
                    "replay_path": str(replay_path),
                    "replay_file_sha256": actual_file_hash,
                    "batch_hashes": self._batch_hashes,
                    "batch_count": self._shared_slots,
                    "sample_slice": [first, last],
                    "initialization_s": time.perf_counter() - started,
                    "iteration_contract": "memory_slice_and_dict_construction_only",
                },
            )
            return

        replay_root = Path(required_env("BENCHMARK_REPLAY_ROOT"))
        replay_path = replay_root / f"dp_rank_{dp_rank:05d}.pt"
        manifest_path = replay_root / "manifest.json"
        if not replay_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                f"Replay is incomplete: {replay_path} and {manifest_path} are required"
            )

        manifest = json.loads(manifest_path.read_text())
        entry = manifest["replays"].get(str(dp_rank))
        if entry is None:
            raise RuntimeError(f"Replay manifest has no DP rank {dp_rank}")
        actual_file_hash = file_sha256(replay_path)
        if actual_file_hash != entry["file_sha256"]:
            raise RuntimeError(
                f"Replay file hash mismatch for DP rank {dp_rank}: "
                f"expected {entry['file_sha256']}, got {actual_file_hash}"
            )

        payload = torch.load(replay_path, map_location="cpu", weights_only=True)
        expected_metadata = {
            "dp_world_size": dp_world_size,
            "dp_rank": dp_rank,
            "seq_len": seq_len,
            "local_batch_size": local_batch_size,
        }
        for key, expected in expected_metadata.items():
            actual = payload["metadata"].get(key)
            if actual != expected:
                raise RuntimeError(
                    f"Replay metadata mismatch for {key}: expected {expected}, "
                    f"got {actual}"
                )

        self._batches = payload["batches"]
        self._batch_hashes = payload["batch_hashes"]
        if not self._batches or len(self._batches) != len(self._batch_hashes):
            raise RuntimeError(
                "Replay must contain matching, non-empty batches and hashes"
            )
        for index, ((input_dict, labels), expected_hash) in enumerate(
            zip(self._batches, self._batch_hashes, strict=True)
        ):
            actual_hash = hash_batch(input_dict, labels)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"Replay batch {index} hash mismatch: "
                    f"expected {expected_hash}, got {actual_hash}"
                )
        self._index = 0

        audit_dir = Path(required_env("BENCHMARK_OUTPUT_DIR")) / "replay_loader"
        atomic_write_json(
            audit_dir / f"rank_{int(os.environ['RANK']):05d}.json",
            {
                "rank": int(os.environ["RANK"]),
                "dp_rank": dp_rank,
                "dp_world_size": dp_world_size,
                "replay_path": str(replay_path),
                "replay_file_sha256": actual_file_hash,
                "batch_hashes": self._batch_hashes,
                "batch_count": len(self._batches),
                "initialization_s": time.perf_counter() - started,
                "iteration_contract": "memory_index_and_shallow_input_dict_copy_only",
            },
        )

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        if self._batches is None:
            while True:
                slot = self._index % self._shared_slots
                self._index += 1
                yield (
                    {
                        "input": self._shared_payload["input"][
                            slot, self._shared_slice
                        ],
                        "positions": self._shared_payload["positions"][
                            slot, self._shared_slice
                        ],
                    },
                    self._shared_payload["labels"][slot, self._shared_slice],
                )
            return
        while self._index < len(self._batches):
            input_dict, labels = self._batches[self._index]
            self._index += 1
            yield dict(input_dict), labels

    def state_dict(self) -> dict[str, Any]:
        return {"index": self._index}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        index = int(state_dict.get("index", 0))
        limit = None if self._batches is None else len(self._batches)
        if index < 0 or (limit is not None and index > limit):
            raise ValueError(f"Invalid replay index {index}")
        self._index = index
