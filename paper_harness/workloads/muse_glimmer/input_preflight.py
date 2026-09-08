from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch

from torchtitan.experiments.graph_trainer.muse_glimmer.config_registry import (
    graph_trainer_muse_glimmer_30b_sdpa_c4_4x2,
)


def _batch_sha256(input_dict: dict[str, torch.Tensor], labels: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for name, tensor in [*sorted(input_dict.items()), ("labels", labels)]:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _manifest_path(payload: Path, template: str) -> Path:
    if Path(template).is_absolute():
        return Path(template)
    prefix = "{asset:"
    start = template.index(prefix)
    end = template.index("}", start)
    asset = template[start + len(prefix) : end]
    return Path(template[:start] + str(payload / "assets" / asset) + template[end + 1 :])


def audit(
    *,
    payload: Path,
    resolved: dict,
    output: Path,
    environment: dict[str, str],
    rank: int,
) -> dict:
    del output, environment
    parallelism = resolved["parallelism"]
    tp_degree = int(parallelism["tensor_parallel_degree"])
    fsdp_degree = int(parallelism["data_parallel_shard_degree"])
    dp_rank, tp_rank = divmod(rank, tp_degree)
    manifest_path = _manifest_path(payload, resolved["data"]["identity_manifest"])
    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    expected = [record for record in records if record["dp_rank"] == dp_rank]
    if [record["step"] for record in expected] != list(range(1, 26)):
        raise RuntimeError(f"input manifest is incomplete for DP rank {dp_rank}")
    if tp_rank != 0:
        return {
            "status": "skipped",
            "dp_rank": dp_rank,
            "reason": "TP rank 0 validates this DP rank",
        }

    config = graph_trainer_muse_glimmer_30b_sdpa_c4_4x2()
    config.training = replace(
        config.training,
        local_batch_size=int(resolved["training"]["local_batch_size"]),
        global_batch_size=int(resolved["training"]["global_batch_size"]),
        seq_len=int(resolved["training"]["seq_len"]),
    )
    config.parallelism = replace(
        config.parallelism,
        data_parallel_shard_degree=fsdp_degree,
        tensor_parallel_degree=tp_degree,
    )
    tokenizer = config.tokenizer.build(
        tokenizer_path=str(payload / "assets/muse_glimmer")
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
            {"dp_rank": dp_rank, "step": step, "sha256": _batch_sha256(input_dict, labels)}
        )
    if observed != expected:
        raise RuntimeError(f"runtime inputs differ for DP rank {dp_rank}")
    return {
        "status": "passed",
        "dp_rank": dp_rank,
        "verified_steps": len(observed),
        "manifest": str(manifest_path),
    }
