from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from torchtitan.components.metrics import BaseLogger, MetricsProcessor

from .io_utils import atomic_write_json, canonical_json_hash, required_env


class BufferedMetricLogger(BaseLogger):
    """Buffer stock TorchTitan metrics and perform one write during close()."""

    def __init__(self, *, dump_folder: str, config_dict: dict[str, Any] | None) -> None:
        self._dump_folder = Path(dump_folder)
        self._config_dict = config_dict
        self._records: list[dict[str, Any]] = []
        self._created_ns = time.time_ns()

    def log(self, metrics: dict[str, Any], step: int) -> None:
        self._records.append({"step": step, "metrics": dict(metrics)})

    def close(self) -> None:
        distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        rank = (
            torch.distributed.get_rank()
            if distributed
            else int(os.environ.get("RANK", 0))
        )
        world_size = (
            torch.distributed.get_world_size()
            if distributed
            else int(os.environ.get("WORLD_SIZE", 1))
        )
        counters: dict[str, dict[str, int]] = {}
        try:
            from torch._dynamo.utils import counters as dynamo_counters

            counters = {
                str(group): {str(key): int(value) for key, value in values.items()}
                for group, values in dynamo_counters.items()
            }
        except Exception:
            counters = {}

        atomic_write_json(
            self._dump_folder / "benchmark_metrics" / f"rank_{rank:05d}.json",
            {
                "schema_version": 1,
                "arm": required_env("BENCHMARK_ARM"),
                "phase": required_env("BENCHMARK_PHASE"),
                "rank": rank,
                "world_size": world_size,
                "created_ns": self._created_ns,
                "closed_ns": time.time_ns(),
                "source_lock_sha256": required_env("BENCHMARK_SOURCE_LOCK_SHA256"),
                "replay_manifest_sha256": required_env(
                    "BENCHMARK_REPLAY_MANIFEST_SHA256"
                ),
                "allocation_fingerprint": required_env(
                    "BENCHMARK_ALLOCATION_FINGERPRINT"
                ),
                "config_sha256": canonical_json_hash(self._config_dict),
                "records": self._records,
                "torch_dynamo_counters": counters,
            },
        )


class BufferedMetricsProcessor(MetricsProcessor):
    @dataclass(kw_only=True, slots=True)
    class Config(MetricsProcessor.Config):
        pass

    def _build_metric_logger(
        self,
        *,
        config: Config,
        parallel_dims,
        dump_folder: str,
        pp_schedule: str,
        ft_enable: bool = False,
        ft_replica_id: int = 0,
        config_dict: dict[str, Any] | None = None,
        tag: str | None = None,
    ) -> BaseLogger:
        del config, parallel_dims, pp_schedule, ft_enable, ft_replica_id, tag
        return BufferedMetricLogger(
            dump_folder=dump_folder,
            config_dict=config_dict,
        )
