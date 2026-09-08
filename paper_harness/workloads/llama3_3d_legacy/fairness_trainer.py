from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch

from torchtitan.experiments.graph_trainer.trainer import GraphTrainer

from .io_utils import atomic_write_json, required_env


def _tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": digest.hexdigest(),
    }


class FairnessGraphTrainer(GraphTrainer):
    """GraphTrainer that records effective model inputs during correctness runs."""

    @dataclass(kw_only=True, slots=True)
    class Config(GraphTrainer.Config):
        pass

    def post_dataloading_process(self, input_dict, labels):
        inputs, labels, extra_kwargs = super().post_dataloading_process(
            input_dict, labels
        )
        if required_env("BENCHMARK_PHASE") == "correctness":
            tensors = {"input": inputs, "labels": labels}
            tensors.update(
                {
                    name: value
                    for name, value in extra_kwargs.items()
                    if isinstance(value, torch.Tensor)
                }
            )
            rank = torch.distributed.get_rank()
            output = (
                Path(required_env("BENCHMARK_OUTPUT_DIR"))
                / "effective_inputs"
                / f"rank_{rank:05d}_step_{self.step:05d}.json"
            )
            atomic_write_json(
                output,
                {
                    "schema_version": 1,
                    "rank": rank,
                    "step": self.step,
                    "tensors": {
                        name: _tensor_record(value)
                        for name, value in sorted(tensors.items())
                    },
                },
            )
        return inputs, labels, extra_kwargs


def to_fairness_graph_trainer_config(
    config: GraphTrainer.Config,
) -> FairnessGraphTrainer.Config:
    return FairnessGraphTrainer.Config(
        **{field.name: getattr(config, field.name) for field in fields(config)}
    )
