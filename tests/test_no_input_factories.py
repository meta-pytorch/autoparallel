# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
from conftest import apply_cuda_patches
from torch import nn
from torch.distributed.tensor.placement_types import Replicate

from autoparallel.api import AutoParallel


class RandomFactory(nn.Module):
    def forward(self):
        return torch.randint(2, 10, (4, 8), device="cuda")


@apply_cuda_patches
def test_no_input_random_factory_lowers_and_executes(device_mesh_1d):
    with torch.device("meta"):
        model = RandomFactory()

    with AutoParallel(
        model, lambda: (), device_mesh_1d, repeated_subgraphs=False
    ) as autop:
        autop.add_input_constraints([])
        autop.add_output_constraints([(Replicate(),)])
        solution = autop.optimize_placement()
        parallel_model = autop.apply_placement(solution)

    output = parallel_model()
    assert output.shape == (4, 8)
    assert output.dtype == torch.int64
    assert torch.all((output >= 2) & (output < 10))
