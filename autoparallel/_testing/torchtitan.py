# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""AutoParallel integration recipe for TorchTitan's Python config loader."""

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.llama3.model import GraphTrainerLlama3Model
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan_recipes.tests.models.llama3 import llama3_debugmodel


def autoparallel_llama3_debugmodel() -> GraphTrainer.Config:
    """Configure the Llama debug model with AutoParallel and four-way TP."""
    config = to_graph_trainer_config(
        llama3_debugmodel(
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1, tensor_parallel_degree=4
            )
        ),
        GraphTrainerLlama3Model.Config,
    )
    config.compile = GraphTrainerCompileConfig(enable_autoparallel=True)
    return config
