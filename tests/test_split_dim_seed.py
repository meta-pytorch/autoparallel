# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math

import pulp
import pytest
import torch
from conftest import apply_cuda_patches
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel.api import AutoParallel
from autoparallel.cost_models.nccl_cost_model import h100_topo_config
from autoparallel.mesh_search import build_split_dim_seed
from autoparallel.optimize_sharding import ShardingOptimizer

pytestmark = [
    pytest.mark.filterwarnings("ignore:Constructing LpVariable.*:DeprecationWarning"),
    pytest.mark.filterwarnings(
        "ignore:Using LpProblem.constraints.*:DeprecationWarning"
    ),
]


class TinyMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = torch.nn.Linear(16, 32)
        self.out_proj = torch.nn.Linear(32, 16)

    def forward(self, x):
        return self.out_proj(torch.relu(self.in_proj(x)))


def _input_fn():
    return torch.randn(8, 16, device="cuda", requires_grad=True)


@apply_cuda_patches
def test_split_dim_seed_hamming_space_solves_with_ilp_and_lp():
    config = h100_topo_config(num_nodes=2, gpus_per_node=4)
    with unset_fake_temporarily():
        mesh = init_device_mesh(
            "cuda",
            (2, 2, 2),
            mesh_dim_names=("dp", "mid", "inner"),
        )

    with torch.device("meta"):
        model = TinyMLP()

    input_placement = (Shard(0), Replicate(), Replicate())
    one_d_cache = {}

    with AutoParallel(
        model,
        _input_fn,
        mesh,
        cost_model=config,
        repeated_subgraphs=False,
    ) as autop:
        seed = build_split_dim_seed(
            autop.gm,
            tuple(mesh.shape),
            input_placement,
            cost_model=config,
            repeated_subgraphs=False,
            one_d_cache=one_d_cache,
        )

        opt = ShardingOptimizer(
            autop.gm,
            mesh,
            repeated_subgraphs=False,
            strategy_seed=seed,
            strategy_radius=2,
        )
        opt.add_sharded_input_constraint([input_placement])
        opt.add_sharded_output_constraint([input_placement])
        opt.add_parameter_memory_constraint(0.0, 1.0)

        lp_result = opt.solve_lp_relaxation(extract=True)
        assert lp_result["status"] == "Optimal"
        assert math.isfinite(lp_result["objective"])

        solution = opt.get_solution(verbose=False)
        assert solution
        assert pulp.LpStatus[opt.prob.status] == "Optimal"
        assert math.isfinite(pulp.value(opt.prob.objective))
