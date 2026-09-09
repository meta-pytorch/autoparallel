from __future__ import annotations

from pathlib import Path

import torch

from .perf_configs import (
    REPLAY_SLOTS,
    TARGET_SEQUENCE_LENGTH,
    _batch_sha256,
    _file_sha256,
    build_16k_batch,
    load_replay_payload,
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
    parallelism = resolved["parallelism"]
    dp_degree = int(parallelism["data_parallel_shard_degree"])
    cp_degree = int(parallelism["context_parallel_degree"])
    tp_degree = int(parallelism["tensor_parallel_degree"])
    ranks_per_dp = cp_degree * tp_degree
    dp_rank, inner_rank = divmod(rank, ranks_per_dp)
    if inner_rank:
        return {
            "status": "skipped",
            "dp_rank": dp_rank,
            "reason": "CP0/TP0 validates the shared logical batch",
        }

    replay_payload, manifest, replay_path = load_replay_payload(
        verify_file_hash=dp_rank == 0
    )
    local_batch_size = int(resolved["training"]["local_batch_size"])
    records = []
    expected_positions = torch.arange(TARGET_SEQUENCE_LENGTH, dtype=torch.int64)
    for slot in range(REPLAY_SLOTS):
        input_dict, labels, source_indices = build_16k_batch(
            replay_payload,
            slot=slot,
            dp_rank=dp_rank,
            local_batch_size=local_batch_size,
        )
        expected_shape = (local_batch_size, TARGET_SEQUENCE_LENGTH)
        if any(
            tuple(tensor.shape) != expected_shape or tensor.dtype != torch.int64
            for tensor in (*input_dict.values(), labels)
        ):
            raise RuntimeError(f"Invalid transformed 16K batch at slot {slot}")
        if not torch.equal(
            input_dict["positions"], expected_positions.repeat(local_batch_size, 1)
        ):
            raise RuntimeError(f"Positions are not monotonic at slot {slot}")
        records.append(
            {
                "slot": slot,
                "source_indices": source_indices,
                "sha256": _batch_sha256(input_dict, labels),
            }
        )

    return {
        "status": "passed",
        "dp_rank": dp_rank,
        "dp_world_size": dp_degree,
        "source_manifest_sha256": _file_sha256(
            Path(environment["REPLAY_MANIFEST_PATH"])
        ),
        "source_replay_sha256": manifest["replay_file"]["sha256"],
        "source_replay": str(replay_path),
        "source_replay_hash_verified": dp_rank == 0,
        "target_sequence_length": TARGET_SEQUENCE_LENGTH,
        "position_contract": "monotonic_0_to_16383_single_causal_sequence",
        "batch_records": records,
    }
