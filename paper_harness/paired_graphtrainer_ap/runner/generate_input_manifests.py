from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from datasets import Features, Value, load_dataset
from huggingface_hub import hf_hub_download

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.hf_datasets.text_datasets import DATASETS, HuggingFaceTextDataLoader


C4_REPO = "allenai/c4"
C4_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
C4_TRAIN_SHARDS = 1024
C4_STAGED_SHARDS = 16
DATASET_NAME = "muse_glimmer_c4_pinned_offline"
CACHE_ROOT: Path | None = None


def load_pinned_offline_c4(dataset_path: str):
    if dataset_path != C4_REPO:
        raise ValueError(f"Expected C4 path {C4_REPO!r}, got {dataset_path!r}")
    local_files = [
        hf_hub_download(
            repo_id=C4_REPO,
            filename=(f"en/c4-train.{shard:05d}-of-{C4_TRAIN_SHARDS:05d}.json.gz"),
            repo_type="dataset",
            revision=C4_REVISION,
            local_files_only=True,
            cache_dir=CACHE_ROOT,
        )
        for shard in range(C4_STAGED_SHARDS)
    ]
    return load_dataset(
        "json",
        data_files={"train": local_files},
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


def batch_sha256(input_dict: dict[str, torch.Tensor], labels: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for name, tensor in [*sorted(input_dict.items()), ("labels", labels)]:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=25)
    args = parser.parse_args()

    os.environ["HF_HUB_CACHE"] = str(args.cache_root.resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    global CACHE_ROOT
    CACHE_ROOT = args.cache_root.resolve()
    DATASETS[DATASET_NAME] = DatasetConfig(
        path=C4_REPO,
        loader=load_pinned_offline_c4,
        sample_processor=lambda sample: sample["text"],
    )
    tokenizer = HuggingFaceTokenizer.Config().build(
        tokenizer_path=str(args.tokenizer_dir.resolve())
    )
    dataloader_config = HuggingFaceTextDataLoader.Config(dataset=DATASET_NAME)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dp_world_size = 8
    for local_batch_size in (2,):
        output = args.output_dir / (
            f"c4_input_manifest_lb{local_batch_size}_s4096_dp{dp_world_size}.jsonl"
        )
        with output.open("w") as stream:
            for dp_rank in range(dp_world_size):
                dataloader = dataloader_config.build(
                    dp_world_size=dp_world_size,
                    dp_rank=dp_rank,
                    tokenizer=tokenizer,
                    seq_len=4096,
                    local_batch_size=local_batch_size,
                    snapshot_every_n_steps=None,
                )
                iterator = iter(dataloader)
                for step in range(1, args.steps + 1):
                    input_dict, labels = next(iterator)
                    record = {
                        "dp_rank": dp_rank,
                        "step": step,
                        "sha256": batch_sha256(input_dict, labels),
                    }
                    stream.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
