# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import json

import torch
from conftest import apply_cuda_patches
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel._testing.models.dsv3 import DeepSeekV3Model, make_dsv3_config
from autoparallel._testing.models.llama3 import Transformer, TransformerModelArgs
from autoparallel.api import AutoParallel


def _assert_replay_matches(opt, replay_path):
    expected = opt.get_json()
    json.dumps(expected)
    opt.save_replay(replay_path)

    loaded = type(opt).load_replay(replay_path)
    loaded.get_solution()
    actual = loaded.get_json()
    json.dumps(actual)

    assert len(loaded.nodes) == len(opt.nodes)
    assert len(loaded.decision_vars) == len(opt.decision_vars)
    assert len(loaded.cluster_links) == len(opt.cluster_links)
    assert actual["summary"] == expected["summary"]
    assert {node["name"]: node.get("placement") for node in actual["nodes"]} == {
        node["name"]: node.get("placement") for node in expected["nodes"]
    }


@apply_cuda_patches
def test_llama3_optimizer_replay(device_mesh_2d, tmp_path):
    vocab_size = 2048
    seq_len = 128
    batch_size = 2 * device_mesh_2d.shape[0]
    model_args = TransformerModelArgs(
        dim=256,
        n_layers=2,
        n_heads=16,
        n_kv_heads=4,
        vocab_size=vocab_size,
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
    with AutoParallel(
        model,
        input_fn,
        device_mesh_2d,
        mp_policy=mp_policy,
        repeated_subgraphs=True,
    ) as autop:
        autop.add_input_constraints([(Shard(0), Replicate())])
        autop.add_output_constraints([(Shard(0), Shard(2))])
        autop.optimize_placement(verbose=False)
        _assert_replay_matches(
            autop.sharding_optimizer,
            tmp_path / "llama3-optimizer-replay.pt.gz",
        )


@apply_cuda_patches
def test_deepseek_v3_optimizer_replay(device_mesh_2d, tmp_path):
    seq_len = 128
    config = make_dsv3_config(
        n_layers=1,
        n_dense_layers=0,
        max_seq_len=seq_len,
    )
    with torch.device("meta"):
        model = DeepSeekV3Model(
            config,
            mesh=device_mesh_2d,
            compute_dtype=torch.bfloat16,
        )
    for module in model.modules():
        if hasattr(module, "axis_name"):
            module.axis_name = device_mesh_2d.mesh_dim_names[1]

    global_batch_size = device_mesh_2d.size()

    def input_fn():
        return torch.randint(
            0,
            config.vocab_size,
            (global_batch_size, seq_len),
            device="cuda",
        )

    with AutoParallel(
        model,
        input_fn,
        device_mesh_2d,
        dynamic=True,
        repeated_subgraphs=True,
    ) as autop:
        autop.add_input_constraints([(Shard(0), Shard(0))])
        autop.add_output_constraints([(Shard(0), Shard(0))])
        autop.optimize_placement(verbose=False)
        _assert_replay_matches(
            autop.sharding_optimizer,
            tmp_path / "deepseek-v3-optimizer-replay.pt.gz",
        )
