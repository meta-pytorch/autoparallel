# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""AutoParallel integration recipe for TorchTitan's Python config loader."""

from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan_recipes.tests.graph_trainer.llama3 import (
    graph_trainer_llama3_debugmodel_varlen_cross_entropy_loss,
)


def autoparallel_llama3_debugmodel() -> GraphTrainer.Config:
    """Configure the Llama debug model with AutoParallel and four-way TP."""
    config = graph_trainer_llama3_debugmodel_varlen_cross_entropy_loss()
    config.compile.enable_autoparallel = True
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.tensor_parallel_degree = 4
    return config
