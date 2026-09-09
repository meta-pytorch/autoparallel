from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
from pathlib import Path

import torch
from datasets import Features, Value, load_dataset
from huggingface_hub import hf_hub_download

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.hf_datasets.text_datasets import DATASETS, HuggingFaceTextDataset

from .replay_data import (
    C4_REPO,
    C4_REVISION,
    C4_SHARD_FILE,
    C4_SHARD_SHA256,
    C4_SHARD_SIZE,
    case_name,
    file_sha256,
    tensor_sha256,
    tree_sha256,
)


DEFAULT_CASES = (
    (2048, 2),
    (4096, 2),
    (8192, 2),
    (16384, 2),
    (2048, 4),
    (2048, 8),
)


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _case(value: str) -> tuple[int, int]:
    try:
        seq_len, local_batch_size = (int(item) for item in value.split(":"))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            "case must be SEQ_LEN:LOCAL_BATCH_SIZE"
        ) from error
    if seq_len <= 0 or local_batch_size <= 0:
        raise argparse.ArgumentTypeError("case values must be positive")
    return seq_len, local_batch_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize deterministic real-C4 inputs for LLaMA replanning"
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--tokenizer-dir", required=True, type=Path)
    parser.add_argument("--dp-degree", default=4, type=_positive)
    parser.add_argument("--slots", default=10, type=_positive)
    parser.add_argument(
        "--case",
        action="append",
        type=_case,
        dest="cases",
        help="SEQ_LEN:LOCAL_BATCH_SIZE; repeat for multiple cases",
    )
    return parser.parse_args()


def _load_local_json(path: str):
    return load_dataset(
        "json",
        data_files={"train": [path]},
        features=Features(
            {
                "text": Value("string"),
                "timestamp": Value("string"),
                "url": Value("string"),
            }
        ),
        split="train",
        streaming=True,
    )


def _materialize_case(
    *,
    seq_len: int,
    local_batch_size: int,
    dp_degree: int,
    slots: int,
    raw_c4_path: Path,
    tokenizer,
    output_root: Path,
) -> dict:
    name = case_name(seq_len, local_batch_size)
    dataset_name = f"llama3_replanning_{name}"
    DATASETS[dataset_name] = DatasetConfig(
        path=str(raw_c4_path),
        loader=_load_local_json,
        sample_processor=lambda sample: sample["text"],
    )
    dataset = HuggingFaceTextDataset(
        dataset_name=dataset_name,
        dataset_path=str(raw_c4_path),
        tokenizer=tokenizer,
        seq_len=seq_len,
        dp_rank=0,
        dp_world_size=1,
        infinite=False,
    )
    iterator = iter(dataset)
    global_batch_size = local_batch_size * dp_degree
    values: dict[str, list[torch.Tensor]] = {
        "input": [],
        "positions": [],
        "labels": [],
    }
    for _ in range(slots * global_batch_size):
        input_dict, labels = next(iterator)
        values["input"].append(input_dict["input"])
        values["positions"].append(input_dict["positions"])
        values["labels"].append(labels)
    tensors = {
        name: torch.stack(items).reshape(slots, global_batch_size, seq_len)
        for name, items in values.items()
    }
    output = output_root / f"{name}.pt"
    torch.save(tensors, output)
    return {
        "file": output.name,
        "size": output.stat().st_size,
        "sha256": file_sha256(output),
        "shape": [slots, global_batch_size, seq_len],
        "seq_len": seq_len,
        "local_batch_size": local_batch_size,
        "global_batch_size": global_batch_size,
        "tensor_sha256": {
            tensor_name: tensor_sha256(tensor)
            for tensor_name, tensor in tensors.items()
        },
    }


def _requested_manifest(args: argparse.Namespace) -> dict:
    cases = args.cases or list(DEFAULT_CASES)
    if len(set(cases)) != len(cases):
        raise ValueError("Replay cases must be unique")
    return {
        "schema_version": 1,
        "dataset": C4_REPO,
        "revision": C4_REVISION,
        "split": "train",
        "c4_file": C4_SHARD_FILE,
        "c4_file_size": C4_SHARD_SIZE,
        "c4_file_sha256": C4_SHARD_SHA256,
        "dp_degree": args.dp_degree,
        "slots": args.slots,
        "tokenizer_tree_sha256": tree_sha256(args.tokenizer_dir),
        "requested_cases": [
            {
                "name": case_name(seq_len, local_batch_size),
                "seq_len": seq_len,
                "local_batch_size": local_batch_size,
            }
            for seq_len, local_batch_size in cases
        ],
    }


def _existing_matches(output_root: Path, requested: dict) -> bool:
    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if any(manifest.get(key) != value for key, value in requested.items()):
        return False
    cases = manifest.get("cases", {})
    if set(cases) != {item["name"] for item in requested["requested_cases"]}:
        return False
    return all(
        (output_root / entry["file"]).is_file()
        and (output_root / entry["file"]).stat().st_size == entry["size"]
        and file_sha256(output_root / entry["file"]) == entry["sha256"]
        for entry in cases.values()
    )


def main() -> None:
    args = parse_args()
    if not args.tokenizer_dir.is_dir():
        raise FileNotFoundError(args.tokenizer_dir)
    requested = _requested_manifest(args)
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.output_root.parent / f".{args.output_root.name}.lock"
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if _existing_matches(args.output_root, requested):
            print(f"Reusing validated replay asset at {args.output_root}")
            return
        if args.output_root.exists():
            raise RuntimeError(
                f"Existing replay asset is incompatible: {args.output_root}; "
                "choose a new output directory"
            )

        staging = args.output_root.with_name(
            f".{args.output_root.name}.staging.{os.getpid()}"
        )
        staging.mkdir()
        try:
            raw_c4_path = Path(
                hf_hub_download(
                    repo_id=C4_REPO,
                    filename=requested["c4_file"],
                    repo_type="dataset",
                    revision=C4_REVISION,
                )
            )
            if (
                raw_c4_path.stat().st_size != C4_SHARD_SIZE
                or file_sha256(raw_c4_path) != C4_SHARD_SHA256
            ):
                raise RuntimeError("Downloaded C4 shard differs from the pinned source")
            tokenizer = HuggingFaceTokenizer.Config().build(
                tokenizer_path=str(args.tokenizer_dir)
            )
            cases = {
                item["name"]: _materialize_case(
                    seq_len=item["seq_len"],
                    local_batch_size=item["local_batch_size"],
                    dp_degree=args.dp_degree,
                    slots=args.slots,
                    raw_c4_path=raw_c4_path,
                    tokenizer=tokenizer,
                    output_root=staging,
                )
                for item in requested["requested_cases"]
            }
            manifest = {
                **requested,
                "cases": cases,
            }
            (staging / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            staging.replace(args.output_root)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        print(f"Prepared replay asset at {args.output_root}")


if __name__ == "__main__":
    main()
