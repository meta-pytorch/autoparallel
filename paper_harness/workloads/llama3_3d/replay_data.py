from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import grain.python as grain
import torch
from torchtitan.components.data.collators import TextCollator
from torchtitan.components.data.dataset import TextSequence
from torchtitan.components.data.loader import BaseDataLoader
from torchtitan.components.data.types import (
    DatasetBuildContext,
    TokenizedTrainingMicrobatch,
)

from .io_utils import atomic_write_json, file_sha256, int_env, required_env


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
    """Replay pre-tokenized C4 samples; iteration performs no disk or network I/O.

    A microbatch packs ``max_num_documents`` whole replay samples. Positions
    restart at every sample, so varlen attention is causal within each sample.

    ``REPLAY_TENSORS_PATH`` selects the shared ``[slots, samples, seq]`` replay:
    accumulation step ``a`` of DP rank ``r`` reads samples starting at
    ``(a * dp_world_size + r) * max_num_documents`` of the current slot, and the
    slot advances after each optimizer step, so the global batch does not
    depend on the mesh. Otherwise ``BENCHMARK_REPLAY_ROOT`` holds one finite
    replay per DP rank.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        pass

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer,
        max_context_length: int,
        num_tokens_per_microbatch: int,
        **kwargs,
    ) -> None:
        del kwargs
        started = time.perf_counter()
        local_batch_size = config.max_num_documents
        if (
            local_batch_size is None
            or num_tokens_per_microbatch != local_batch_size * max_context_length
        ):
            raise ValueError(
                "Replay microbatches pack whole samples: num_tokens_per_microbatch "
                f"({num_tokens_per_microbatch}) must equal max_num_documents "
                f"({local_batch_size}) * max_context_length ({max_context_length})"
            )
        self.max_num_documents = local_batch_size
        self._collate = TextCollator.Config().build(
            context=DatasetBuildContext(
                tokenizer=tokenizer,
                max_context_length=max_context_length,
                num_tokens_per_microbatch=num_tokens_per_microbatch,
                read_options=grain.ReadOptions(),
                max_num_documents=local_batch_size,
            )
        )
        self._index = 0

        shared_replay_path = os.environ.get("REPLAY_TENSORS_PATH")
        if shared_replay_path:
            replay_path = Path(shared_replay_path)
            replay_file_sha256 = None
            payload = torch.load(
                replay_path, map_location="cpu", mmap=True, weights_only=True
            )
            for name in ("input", "positions", "labels"):
                tensor = payload.get(name)
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.dim() != 3
                    or tensor.shape != payload["input"].shape
                    or tensor.dtype != torch.int64
                ):
                    raise RuntimeError(
                        f"Shared replay tensor {name!r} has the wrong shape or dtype"
                    )
            slots, samples, seq_len = payload["input"].shape
            global_batch_size = int_env("BENCHMARK_GLOBAL_BATCH_SIZE")
            if global_batch_size % (local_batch_size * dp_world_size):
                raise ValueError(
                    f"BENCHMARK_GLOBAL_BATCH_SIZE={global_batch_size} is not a "
                    f"multiple of {local_batch_size=} * {dp_world_size=}"
                )
            if seq_len != max_context_length or global_batch_size > samples:
                raise RuntimeError(
                    f"Shared replay {tuple(payload['input'].shape)} cannot serve "
                    f"{max_context_length=} and {global_batch_size=}"
                )
            accumulation_steps = global_batch_size // (local_batch_size * dp_world_size)
            self._batches = None
            self._payload = payload
            self._slices = []
            for slot in range(slots):
                for accumulation in range(accumulation_steps):
                    first = (accumulation * dp_world_size + dp_rank) * local_batch_size
                    self._slices.append((slot, slice(first, first + local_batch_size)))
            raw_batches = [self._raw(index) for index in range(len(self._slices))]
            batch_hashes = [hash_batch(*batch) for batch in raw_batches]
        else:
            replay_root = Path(required_env("BENCHMARK_REPLAY_ROOT"))
            replay_path = replay_root / f"dp_rank_{dp_rank:05d}.pt"
            manifest = json.loads((replay_root / "manifest.json").read_text())
            entry = manifest["replays"].get(str(dp_rank))
            if entry is None:
                raise RuntimeError(f"Replay manifest has no DP rank {dp_rank}")
            replay_file_sha256 = file_sha256(replay_path)
            if replay_file_sha256 != entry["file_sha256"]:
                raise RuntimeError(
                    f"Replay file hash mismatch for DP rank {dp_rank}: "
                    f"expected {entry['file_sha256']}, got {replay_file_sha256}"
                )
            payload = torch.load(replay_path, map_location="cpu", weights_only=True)
            expected_metadata = {
                "dp_world_size": dp_world_size,
                "dp_rank": dp_rank,
                "seq_len": max_context_length,
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
            batch_hashes = payload["batch_hashes"]
            if not self._batches or len(self._batches) != len(batch_hashes):
                raise RuntimeError(
                    "Replay must contain matching, non-empty batches and hashes"
                )
            for index, (batch, expected_hash) in enumerate(
                zip(self._batches, batch_hashes, strict=True)
            ):
                actual_hash = hash_batch(*batch)
                if actual_hash != expected_hash:
                    raise RuntimeError(
                        f"Replay batch {index} hash mismatch: "
                        f"expected {expected_hash}, got {actual_hash}"
                    )

        atomic_write_json(
            Path(required_env("BENCHMARK_OUTPUT_DIR"))
            / "replay_loader"
            / f"rank_{int(os.environ['RANK']):05d}.json",
            {
                "rank": int(os.environ["RANK"]),
                "dp_rank": dp_rank,
                "dp_world_size": dp_world_size,
                "replay_path": str(replay_path),
                "replay_file_sha256": replay_file_sha256,
                "batch_hashes": batch_hashes,
                "batch_count": len(batch_hashes),
                "initialization_s": time.perf_counter() - started,
            },
        )

    def _raw(self, index: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if self._batches is not None:
            return self._batches[index]
        slot, samples = self._slices[index % len(self._slices)]
        return (
            {
                "input": self._payload["input"][slot, samples],
                "positions": self._payload["positions"][slot, samples],
            },
            self._payload["labels"][slot, samples],
        )

    def __iter__(self) -> Iterator[TokenizedTrainingMicrobatch]:
        while self._batches is None or self._index < len(self._batches):
            input_dict, labels = self._raw(self._index)
            self._index += 1
            yield self._collate(
                [
                    TextSequence(input_ids=tokens.numpy(), labels=targets.numpy())
                    for tokens, targets in zip(
                        input_dict["input"], labels, strict=True
                    )
                ]
            )

    def state_dict(self) -> dict[str, Any]:
        return {"index": self._index}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        index = int(state_dict.get("index", 0))
        limit = None if self._batches is None else len(self._batches)
        if index < 0 or (limit is not None and index > limit):
            raise ValueError(f"Invalid replay index {index}")
        self._index = index
