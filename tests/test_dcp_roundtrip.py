# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
End-to-end DCP round-trip test for AutoParallel.

Trains an AP-sharded model for a few steps, saves a DCP checkpoint, loads it
into a fresh non-sharded model, and verifies the post-load loss curve matches
the AP loss curve. This exercises the AP-to-non-AP resharding path that DCP
performs at load time on _StridedShard parameters.

Requires real multi-GPU (uses MultiProcessTestCase via DTensorTestBase). Skips
explicitly on hosts with fewer than 4 GPUs.
"""

import tempfile

import pytest
import torch
from torch import nn
from torch.distributed.checkpoint import load as dcp_load
from torch.distributed.checkpoint import save as dcp_save
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from autoparallel.api import AutoParallel
from autoparallel.compile import autoparallel_backend

_NHEADS = 4
_DIM1 = 64
_DIM2 = _DIM1 * 4
_SEQ = 32


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.nheads = _NHEADS
        self.wq = nn.Linear(_DIM1, _DIM1, bias=False)
        self.wk = nn.Linear(_DIM1, _DIM1, bias=False)
        self.wv = nn.Linear(_DIM1, _DIM1, bias=False)
        self.wo = nn.Linear(_DIM1, _DIM1, bias=False)
        self.w1 = nn.Linear(_DIM1, _DIM2, bias=False)
        self.w2 = nn.Linear(_DIM2, _DIM1, bias=False)

    def init_weights(self):
        for lin in [self.wq, self.wk, self.wv, self.wo, self.w1, self.w2]:
            torch.nn.init.normal_(lin.weight)

    def forward(self, x):
        q = self.wq(x).unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
        k = self.wk(x).unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
        v = self.wv(x).unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
        o = nn.functional.scaled_dot_product_attention(q, k, v)
        o = o.permute(0, 2, 1, 3).flatten(-2)
        o = self.wo(o)
        o0 = o + x
        o = self.w2(nn.functional.relu(self.w1(o0)))
        return o0 + o


def _train_step(model, optimizer, input_data, mesh, in_shard, out_shard):
    optimizer.zero_grad()
    local_in = distribute_tensor(input_data, mesh, in_shard).to_local()
    output = model(local_in)
    output_full = DTensor.from_local(output, mesh, out_shard).full_tensor()
    loss = output_full.mean().abs() * 1e-3
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.detach()


def _plain_train_step(model, optimizer, input_data):
    optimizer.zero_grad()
    output = model(input_data)
    loss = output.mean().abs() * 1e-3
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.detach()


class TestDCPRoundTrip(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @pytest.mark.skipif(
        torch.cuda.device_count() < 4,
        reason="DCP round-trip test requires at least 4 GPUs",
    )
    @with_comms
    def test_ap_to_non_ap_resharding(self):
        torch.manual_seed(21)

        mesh = init_device_mesh(
            self.device_type,
            (self.world_size // 2, 2),
            mesh_dim_names=("dp", "tp"),
        )
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.float32, reduce_dtype=torch.float32
        )

        bs = 4 * mesh.shape[0]
        test_input = torch.rand(bs, _SEQ, _DIM1, device=self.device_type)

        with torch.device("meta"):
            model = _Block()

        def input_fn():
            return test_input

        with AutoParallel(model, input_fn, mesh, mp_policy) as autop:
            autop.add_parameter_memory_constraint(low=None, high=None)
            x_sharding = (Shard(0),) + (Replicate(),) * (mesh.ndim - 1)
            autop.add_input_constraints([x_sharding])
            autop.add_output_constraints([x_sharding])
            sharding_placement = autop.optimize_placement()
            parallel_mod = autop.apply_placement(sharding_placement)

        parallel_mod.to_empty(device=self.device_type)
        parallel_mod.init_weights()
        parallel_mod = torch.compile(parallel_mod, backend=autoparallel_backend())

        optimizer = torch.optim.Adam(parallel_mod.parameters(), lr=2e-5)

        n_pre_ckpt = 5
        n_post_ckpt = 5
        ap_loss = []

        # All ranks must share one checkpoint directory; rank 0 creates it
        # and broadcasts the path to the others.
        if self.rank == 0:
            tmp_dir = tempfile.mkdtemp()
            path_bytes = tmp_dir.encode("utf-8").ljust(512, b"\0")
        else:
            path_bytes = b"\0" * 512
        path_tensor = torch.frombuffer(bytearray(path_bytes), dtype=torch.uint8).to(
            self.device_type
        )
        torch.distributed.broadcast(path_tensor, src=0)
        tmp_dir = bytes(path_tensor.cpu().tolist()).rstrip(b"\0").decode("utf-8")

        try:
            for i in range(n_pre_ckpt + n_post_ckpt):
                loss = _train_step(
                    parallel_mod,
                    optimizer,
                    test_input,
                    mesh,
                    x_sharding,
                    x_sharding,
                )
                if i == n_pre_ckpt - 1:
                    model_sd, optim_sd = get_state_dict(parallel_mod, optimizer)
                    dcp_save(
                        {"model": model_sd, "optimizer": optim_sd},
                        checkpoint_id=tmp_dir,
                    )
                if i >= n_pre_ckpt:
                    ap_loss.append(loss)

            # Load checkpoint into a fresh, non-AP model, run the same
            # post-checkpoint steps, and compare loss curves.
            new_model = _Block().to(self.device_type)
            for p in new_model.parameters():
                p.data.zero_()
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=2e-5)

            msd = get_model_state_dict(new_model)
            osd = get_optimizer_state_dict(new_model, new_optimizer)
            state_dict_to_load = {"model": msd, "optimizer": osd}
            dcp_load(state_dict_to_load, checkpoint_id=tmp_dir)
            set_state_dict(
                new_model,
                new_optimizer,
                model_state_dict=state_dict_to_load["model"],
                optim_state_dict=state_dict_to_load["optimizer"],
            )

            non_ap_loss = []
            for _ in range(n_post_ckpt):
                loss = _plain_train_step(new_model, new_optimizer, test_input)
                non_ap_loss.append(loss)

            for i, (a, b) in enumerate(zip(ap_loss, non_ap_loss)):
                torch.testing.assert_close(
                    a,
                    b,
                    rtol=1e-2,
                    atol=1e-3,
                    msg=f"DCP round-trip loss mismatch at post-ckpt step {i}: "
                    f"AP={a.item()} non-AP={b.item()}",
                )
        finally:
            torch.distributed.barrier()
            if self.rank == 0:
                import shutil

                shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    run_tests()
