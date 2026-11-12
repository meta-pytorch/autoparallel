# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math

import pulp
import pytest
import torch
from conftest import apply_cuda_patches
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel._testing.models.llama3 import Transformer, TransformerModelArgs
from autoparallel.api import AutoParallel


def _fake_dp4_tp4_mesh():
    return torch.distributed.device_mesh.init_device_mesh(
        "cuda",
        (4, 4),
        mesh_dim_names=("dp", "tp"),
    )


def _llama3_example_autop(device_mesh):
    vocab_size = 128
    seq_len = 16
    batch_size = 2 * device_mesh.shape[0]
    model_args = TransformerModelArgs(
        dim=64,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=vocab_size,
        multiple_of=32,
        rope_theta=500000,
        max_seq_len=seq_len,
    )
    with torch.device("meta"):
        model = Transformer(model_args)

    def input_fn():
        return torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    return AutoParallel(
        model,
        input_fn,
        device_mesh,
        mp_policy,
        repeated_subgraphs=True,
    )


@apply_cuda_patches
@pytest.mark.filterwarnings("ignore:Constructing LpVariable")
@pytest.mark.filterwarnings("ignore:Using LpProblem.constraints")
def test_lp_relaxation_certifies_llama3_example_search():
    mesh = _fake_dp4_tp4_mesh()
    with _llama3_example_autop(mesh) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)
        x_sharding = (Shard(0), Replicate())
        out_sharding = (Shard(0), Shard(2))
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([out_sharding])

        opt = autop.sharding_optimizer

        binary_vars = list(opt.pulp_variables.values())
        assert binary_vars
        assert all(var.cat == pulp.LpInteger for var in binary_vars)
        assert all(var.lowBound == 0 and var.upBound == 1 for var in binary_vars)

        continuous_vars = opt._create_pulp_variables(pulp.LpContinuous)
        assert continuous_vars
        assert all(var.cat == pulp.LpContinuous for var in continuous_vars.values())
        assert all(
            var.lowBound == 0 and var.upBound == 1 for var in continuous_vars.values()
        )

        lower_bound = opt.get_lower_bound()
        assert lower_bound.status == "Optimal"
        assert math.isfinite(lower_bound.objective)
        assert lower_bound.objective >= 0

        assert not hasattr(opt, "selected_keys")
        assert opt.prob.objective is None
        assert all(var.cat == pulp.LpInteger for var in opt.pulp_variables.values())

        solution = opt.get_solution()
        feasible_cost = pulp.value(opt.prob.objective)
        certificate_gap = (
            feasible_cost - lower_bound.objective
        ) / lower_bound.objective
        assert solution
        assert lower_bound.objective <= feasible_cost + 1e-5
        assert certificate_gap >= -1e-8
        assert math.isfinite(certificate_gap)
