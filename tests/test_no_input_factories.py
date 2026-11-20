# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from conftest import apply_cuda_patches
from torch import nn
from torch.distributed.tensor.placement_types import Replicate, Shard

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


class UniformScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        scale = torch.empty(1, device=x.device).uniform_(-1.0, 1.0).exp()
        return self.linear(x * scale)


class TransposedUniformNoise(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        bs, dim = x.shape
        noise = torch.empty(dim, bs, device=x.device).t().uniform_(-0.1, 0.1)
        # Flattening is only a valid view if uniform_ kept the transposed layout
        noise = noise.t().view(-1).view(dim, bs).t()
        return self.linear(x + noise)


class EmptyPermutedUniform(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        noise = torch.empty(2, 0, 4, device=x.device).permute(1, 2, 0)
        noise = noise.uniform_(-1.0, 2.0)
        # The dtype view needs a unit last-dim stride, i.e. the permuted layout
        y = noise.permute(2, 0, 1).view(torch.float64)
        return self.linear(x) + y.sum()


@apply_cuda_patches
@pytest.mark.parametrize(
    "model_cls", [UniformScale, TransposedUniformNoise, EmptyPermutedUniform]
)
def test_in_place_uniform_lowers_and_executes(device_mesh_1d, model_cls):
    dim = 64
    batch_size = 512
    local_bs = batch_size // device_mesh_1d.size()

    with torch.device("meta"):
        model = model_cls(dim)

    with AutoParallel(
        model, lambda: torch.randn(batch_size, dim, device="cuda"), device_mesh_1d
    ) as autop:
        autop.add_input_constraints([(Shard(0),)])
        autop.add_output_constraints([(Shard(0),)])
        solution = autop.optimize_placement()
        parallel_mod = autop.apply_placement(solution)
        targets = {n.target for n in autop.gm.graph.nodes}

    assert torch.ops.aten.as_strided.default not in targets

    parallel_mod.to_empty(device="cuda")
    out = parallel_mod(torch.randn(local_bs, dim, device="cuda"))
    assert out.shape == (local_bs, dim)
