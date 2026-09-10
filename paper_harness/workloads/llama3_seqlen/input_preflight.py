from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from . import perf_configs


def _batch_sha256(input_dict: dict[str, torch.Tensor], labels: torch.Tensor) -> str:
    digest = hashlib.sha256()
    tensors = [
        (name, tensor)
        for name, tensor in sorted(input_dict.items())
        if isinstance(tensor, torch.Tensor)
    ]
    for name, tensor in [*tensors, ("labels", labels)]:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _manifest_path(payload: Path, template: str) -> Path:
    if Path(template).is_absolute():
        return Path(template)
    return Path(template.replace("{harness}", str(payload / "harness_repo")))


def audit(
    *,
    payload: Path,
    resolved: dict,
    output: Path,
    environment: dict[str, str],
    rank: int,
) -> dict:
    del output, environment
    tp_degree = int(resolved["parallelism"]["tensor_parallel_degree"])
    dp_rank, tp_rank = divmod(rank, tp_degree)
    manifest_path = _manifest_path(payload, resolved["data"]["identity_manifest"])
    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    expected = [record for record in records if record["dp_rank"] == dp_rank]
    if [record["batch_index"] for record in expected] != list(range(1, 11)):
        raise RuntimeError(f"input manifest is incomplete for DP rank {dp_rank}")
    if tp_rank != 0:
        return {
            "status": "skipped",
            "dp_rank": dp_rank,
            "reason": "TP rank 0 validates this DP rank",
        }

    config = perf_configs.autoparallel_graphtrainer_seqlen_8b()
    tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)
    dataloader = config.dataloader.build(
        dp_world_size=perf_configs.DP_DEGREE,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
        max_context_length=config.training.max_context_length,
        num_tokens_per_batch=(
            config.training.num_tokens_per_microbatch_per_dp_rank
        ),
    )
    iterator = iter(dataloader)
    observed = []
    for batch_index in range(1, 11):
        input_dict, labels = next(iterator)
        if input_dict.get("num_valid_tokens") != labels.numel():
            raise RuntimeError("latest Trainer valid-token metadata is incorrect")
        observed.append(
            {
                "batch_index": batch_index,
                "dp_rank": dp_rank,
                "input_shape": list(input_dict["input"].shape),
                "positions_shape": list(input_dict["positions"].shape),
                "labels_shape": list(labels.shape),
                "sha256": _batch_sha256(input_dict, labels),
            }
        )
    if observed != expected:
        raise RuntimeError(f"runtime inputs differ for DP rank {dp_rank}")
    return {
        "status": "passed",
        "dp_rank": dp_rank,
        "verified_batches": len(observed),
        "manifest": str(manifest_path),
    }
