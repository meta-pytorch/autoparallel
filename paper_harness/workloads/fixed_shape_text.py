from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, cast

import torch
from datasets import Dataset
from datasets.distributed import split_dataset_by_node

from torchtitan.components.data.loader import BaseDataLoader
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.tools.logging import logger


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    path: str
    loader: Callable[[str], Any]
    sample_processor: Callable[[dict[str, Any]], str]


class FixedShapeTextDataset:
    """Historical TorchTitan text packing with one independent row per sequence."""

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: BaseTokenizer,
        seq_len: int,
        dp_rank: int,
        dp_world_size: int,
        infinite: bool,
        datasets: dict[str, DatasetConfig],
    ) -> None:
        dataset_name = dataset_name.lower()
        if dataset_name not in datasets:
            raise ValueError(
                f"Dataset {dataset_name} is not supported. "
                f"Supported datasets are: {list(datasets)}"
            )
        config = datasets[dataset_name]
        path = dataset_path or config.path
        data = config.loader(path)
        self.dataset_name = dataset_name
        self._original_data = split_dataset_by_node(data, dp_rank, dp_world_size)
        self._data = self._original_data
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self._text_processor = config.sample_processor
        self._sample_idx = 0
        self._epoch = 0
        self._inputs_buffer: list[int] = []
        self._labels_buffer: list[int] = []
        self._positions_buffer: list[int] = []

    def _get_data_iter(self):
        if isinstance(self._data, Dataset):
            return iter([]) if self._sample_idx == len(self._data) else iter(
                self._data.skip(self._sample_idx)
            )
        return iter(self._data)

    @staticmethod
    def _normalize_positions(positions: list[int]) -> list[int]:
        offset = positions[0]
        if offset > 0:
            for index, position in enumerate(positions):
                if position == 0:
                    break
                positions[index] = position - offset
        return positions

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                tokens = self._tokenizer.encode(
                    self._text_processor(sample), add_bos=True, add_eos=True
                )
                self._inputs_buffer.extend(tokens[:-1])
                self._labels_buffer.extend(tokens[1:])
                self._positions_buffer.extend(range(len(tokens) - 1))
                self._sample_idx += 1
                while len(self._inputs_buffer) >= self.seq_len:
                    inputs = torch.LongTensor(self._inputs_buffer[: self.seq_len])
                    labels = torch.LongTensor(self._labels_buffer[: self.seq_len])
                    positions = torch.LongTensor(
                        self._normalize_positions(
                            self._positions_buffer[: self.seq_len]
                        )
                    )
                    self._inputs_buffer = self._inputs_buffer[self.seq_len :]
                    self._labels_buffer = self._labels_buffer[self.seq_len :]
                    self._positions_buffer = self._positions_buffer[self.seq_len :]
                    yield {"input": inputs, "positions": positions}, labels
            if not self.infinite:
                logger.warning("Dataset %s has run out of data", self.dataset_name)
                return
            self.reloop()

    def reloop(self) -> None:
        self._sample_idx = 0
        self._epoch += 1
        if isinstance(self._data, Dataset):
            self._data = cast(Dataset, self._original_data.shuffle(seed=42 + self._epoch))
        elif hasattr(self._data, "set_epoch") and hasattr(self._data, "epoch"):
            self._data.set_epoch(self._data.epoch + 1)

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "inputs_buffer": self._inputs_buffer,
            "labels_buffer": self._labels_buffer,
            "positions_buffer": self._positions_buffer,
        }
        if isinstance(self._data, Dataset):
            state["sample_idx"] = self._sample_idx
            state["epoch"] = self._epoch
        else:
            state["data"] = self._data.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._inputs_buffer = state["inputs_buffer"]
        self._labels_buffer = state["labels_buffer"]
        self._positions_buffer = state["positions_buffer"]
        if isinstance(self._data, Dataset):
            self._sample_idx = state["sample_idx"]
            self._epoch = state.get("epoch", 0)
            if self._epoch > 0:
                self._data = cast(
                    Dataset, self._original_data.shuffle(seed=42 + self._epoch)
                )
        else:
            data_state = state["data"]
            self._data.set_epoch(data_state.get("epoch", 0))
            self._data.load_state_dict(data_state)


class FixedShapeTextDataLoader(BaseDataLoader):
    """Batch historical fixed-length rows while satisfying latest Trainer metadata."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        dataset: str = ""
        dataset_path: str | None = None
        infinite: bool = True
        num_workers: int = 0
        persistent_workers: bool = False
        pin_memory: bool = False
        prefetch_factor: int | None = None

    def __init__(
        self,
        config: Config,
        *,
        dataset: FixedShapeTextDataset,
        dp_world_size: int,
        dp_rank: int,
        max_context_length: int,
        num_tokens_per_batch: int,
    ) -> None:
        if any(
            (
                config.num_workers,
                config.persistent_workers,
                config.pin_memory,
                config.prefetch_factor is not None,
            )
        ):
            raise ValueError(
                "fixed-shape historical input order requires the default "
                "single-process, unprefetched dataloader settings"
            )
        local_batch_size, remainder = divmod(
            num_tokens_per_batch, max_context_length
        )
        if remainder or local_batch_size <= 0:
            raise ValueError(
                "num_tokens_per_batch must be a positive multiple of "
                "max_context_length"
            )
        self.max_num_documents = config.max_num_documents
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.local_batch_size = local_batch_size
        self._dataset = dataset
        self._source = iter(dataset)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor | int]]:
        while True:
            samples = [next(self._source) for _ in range(self.local_batch_size)]
            input_dict = {
                name: torch.stack([sample[0][name] for sample in samples])
                for name in samples[0][0]
            }
            labels = torch.stack([sample[1] for sample in samples])
            input_dict["labels"] = labels
            input_dict["num_valid_tokens"] = int((labels != IGNORE_INDEX).sum())
            yield input_dict

    def state_dict(self) -> dict[str, Any]:
        return {
            "dp_world_size": self.dp_world_size,
            f"dp_rank_{self.dp_rank}": self._dataset.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not state:
            return
        if state["dp_world_size"] != self.dp_world_size:
            raise ValueError("dataloader DP degree changed across checkpoint restore")
        self._dataset.load_state_dict(state[f"dp_rank_{self.dp_rank}"])
        self._source = iter(self._dataset)
