from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, cast

import torch

from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols import BaseModel

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

    def forward_backward_step(
        self,
        *,
        input_dict,
        labels,
        global_valid_tokens,
    ):
        if self.parallel_dims.pp_enabled or self.config.compile.mode != "aot_fx_trace":
            raise RuntimeError(
                "FairnessGraphTrainer requires non-PP aot_fx_trace execution"
            )
        assert isinstance(input_dict, dict)
        assert isinstance(labels, torch.Tensor)
        assert len(self.model_parts) == 1
        model = self.model_parts[0]
        with sl.log_trace_span("preprocess_inputs"):
            inputs, labels, extra_kwargs = cast(BaseModel, model).preprocess_inputs(
                {**input_dict, "labels": labels},
                parallel_dims=self.parallel_dims,
                parallelism=self.config.parallelism,
                max_num_documents=self.dataloader.max_num_documents,
                max_context_length=self.config.training.max_context_length,
                apply_context_parallel=(
                    not self._autoparallel_manages_context_parallel_input()
                ),
            )
            self.ntokens_seen += labels.numel()
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
        params = [
            parameter
            for _, parameter in model.named_parameters(remove_duplicate=False)
            if parameter.requires_grad
        ]
        return self._make_fx_forward_backward_step(
            model,
            inputs,
            labels,
            global_valid_tokens,
            params,
            extra_kwargs,
        )


def to_fairness_graph_trainer_config(
    config: GraphTrainer.Config,
) -> FairnessGraphTrainer.Config:
    return FairnessGraphTrainer.Config(
        **{field.name: getattr(config, field.name) for field in fields(config)}
    )
