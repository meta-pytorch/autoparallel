# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
import tempfile

import torch
import torch.distributed.checkpoint
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._api import distribute_tensor
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel.api import AutoParallel


def master_print(*args, **kwargs):
    if torch.distributed.get_rank() == 0:
        print(*args, **kwargs)


def rank_print(*args, **kwargs):
    print(f"rank: {torch.distributed.get_rank()}", *args, **kwargs)


mp_policy = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)

seq_len = 256
nheads = 48
dim1 = 6144
dim2 = dim1 * 4


# copy the module from example/example_autoparallel.py
class Block(nn.Module):
    def __init__(self, nheads, dim1, dim2):
        super().__init__()
        self.nheads = nheads
        bias = False
        self.wq = nn.Linear(dim1, dim1, bias=bias)
        self.wk = nn.Linear(dim1, dim1, bias=bias)
        self.wv = nn.Linear(dim1, dim1, bias=bias)
        self.wo = nn.Linear(dim1, dim1, bias=bias)
        self.w1 = nn.Linear(dim1, dim2, bias=bias)
        self.w2 = nn.Linear(dim2, dim1, bias=bias)

    def init_weights(self):
        for lin in [self.wq, self.wk, self.wv, self.wo, self.w1, self.w2]:
            torch.nn.init.normal_(lin.weight)
            if lin.bias is not None:
                torch.nn.init.normal_(lin.bias)

    def _compute_attention(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
        k = k.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
        v = v.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)

        o = nn.functional.scaled_dot_product_attention(q, k, v)
        o = o.permute(0, 2, 1, 3).flatten(-2)

        o = self.wo(o)
        return o

    def forward(self, x):
        o = self._compute_attention(x)
        o0 = o + x

        o = self.w1(o0)
        o = torch.nn.functional.relu(o)
        o = self.w2(o)

        o = o0 + o

        return o


def setup_fake_process_group_if_needed(fake_world_size):
    """Set up a fake process group if one is not already initialized."""
    if not torch.distributed.is_initialized():
        fake_store = FakeStore()
        torch.distributed.init_process_group(
            "fake", store=fake_store, rank=0, world_size=fake_world_size
        )
        return True
    return False


# run with fake process group
def prepare_autoparallel_model(fake_world_size):
    assert setup_fake_process_group_if_needed(fake_world_size)

    try:
        mesh = torch.distributed.device_mesh.init_device_mesh(
            "cuda",
            (fake_world_size // 8, 8),
            mesh_dim_names=(
                "dp",
                "tp",
            ),
        )
        bs = 8 * mesh.shape[0]
        with torch.device("meta"):
            model = Block(nheads, dim1, dim2)

        def input_fn():
            print(f"global input shape: {(bs, seq_len, dim1)}")
            return torch.rand(bs, seq_len, dim1, device="cuda")

        with AutoParallel(model, input_fn, mesh, mp_policy, compile=True) as autop:
            assert any(n.meta.get("nn_module_stack") for n in autop.gm.graph.nodes)
            assert any(n.meta.get("fwd_nn_module_stack") for n in autop.gm.graph.nodes)
            autop.add_parameter_memory_constraint(low=None, high=None)

            x_sharding = (Shard(0),) + (Replicate(),) * (mesh.ndim - 1)

            autop.add_input_constraints([x_sharding])
            autop.add_output_constraints([x_sharding])

            sharding_placement = autop.optimize_placement()
            autop.apply_placement(sharding_placement)

    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    # extract out sharding placement into a dict that is picklable so that we
    # can reuse the model placement from the solver in multiprocess
    sharding_map = {}
    for key, val in sharding_placement.items():
        fqn = key.name
        sharding_map[fqn] = val

    return model, sharding_map


# use multiprocess to run the model with sharding placements from AP solver
def multiple_process_run(rank, world_size, tmp_dir, model, sharding_map):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    torch.cuda.set_device(rank)
    try:
        torch.distributed.init_process_group(
            backend="nccl", rank=rank, world_size=world_size
        )
        mesh = torch.distributed.device_mesh.init_device_mesh(
            "cuda",
            (world_size // 2, 2),
            mesh_dim_names=("dp", "tp"),
        )
        bs = 8 * mesh.shape[0]

        # checkpoint files from all ranks must be saved under the same folder
        rank_print(f"use tmp folder {tmp_dir=} to save checkpoint and metadata file")

        torch.manual_seed(21)

        test_input = torch.rand(bs, seq_len, dim1, device="cuda")

        def input_fn():
            return test_input

        new_model = copy.deepcopy(model)
        new_model = new_model.to_empty(device="cuda")
        for param in new_model.parameters():
            param.data.zero_()
        new_optimizer = torch.optim.Adam(new_model.parameters())

        with AutoParallel(model, input_fn, mesh, mp_policy, compile=True) as autop:
            assert any(n.meta.get("nn_module_stack") for n in autop.gm.graph.nodes)
            assert any(n.meta.get("fwd_nn_module_stack") for n in autop.gm.graph.nodes)
            autop.add_parameter_memory_constraint(low=None, high=None)

            x_sharding = (Shard(0),) + (Replicate(),) * (mesh.ndim - 1)

            autop.add_input_constraints([x_sharding])
            autop.add_output_constraints([x_sharding])

            # reconstruct sharding_placement based on sharding_map instead
            # sharding_placement = autop.optimize_placement()
            sharding_placement = {}
            for node in autop.gm.graph.nodes:
                fqn = node.name
                if fqn in sharding_map:
                    # Adjust sharding_map[fqn] to use a (2,2) mesh configuration
                    adjusted_value = sharding_map[fqn]
                    for input_spec in adjusted_value.input_specs:
                        input_spec.mesh = mesh
                    if isinstance(adjusted_value.output_specs, tuple):
                        for output_spec in adjusted_value.output_specs:
                            if output_spec is None:
                                continue
                            output_spec.mesh = mesh
                    else:
                        adjusted_value.output_specs.mesh = mesh
                    sharding_placement[node] = adjusted_value

            parallel_mod = autop.apply_placement(sharding_placement)

        parallel_mod.to_empty(device="cuda")
        parallel_mod.init_weights()

        # Use smaller learning rate and gradient clipping for numerical stability
        optimizer = torch.optim.Adam(parallel_mod.parameters(), lr=2e-5)

        def train_step(
            model, optimizer, input_data, mesh, in_shard=None, out_shard=None
        ):
            optimizer.zero_grad()
            if in_shard:
                input_data = distribute_tensor(input_data, mesh, in_shard).to_local()
            output = model(input_data)
            if out_shard:
                output_d = DTensor.from_local(output, mesh, out_shard)
                output = output_d.full_tensor()
            # use mean instead of sum to avoid huge loss values, and add loss scaling
            loss = torch.abs(output.mean()) * 1e-3  # scale down the loss for stability
            torch.distributed.barrier()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            return loss

        # let original AP model train for 20 steps, save checkpoint at step 10
        ap_loss = []
        for i in range(20):
            loss = train_step(
                parallel_mod, optimizer, test_input, mesh, x_sharding, x_sharding
            )
            if i % 1 == 0:
                master_print(f"step: {i}, {loss=}")
            if i == 10:
                # save checkpoint
                master_print(f"save checkpoint at step: {i}")
                model_state_dict, optim_state_dict = get_state_dict(
                    parallel_mod,
                    optimizer,
                )
                state_dict_to_save = {
                    "model": model_state_dict,
                    "optimizer": optim_state_dict,
                }
                torch.distributed.checkpoint.save(
                    state_dict_to_save, checkpoint_id=tmp_dir
                )
            if i >= 11:
                ap_loss.append(loss)

        # resume from step 10 checkpoint and run on a new model without sharding
        msd = get_model_state_dict(new_model)
        osd = get_optimizer_state_dict(new_model, new_optimizer)
        state_dict_to_load = {"model": msd, "optimizer": osd}
        torch.distributed.checkpoint.load(state_dict_to_load, checkpoint_id=tmp_dir)
        set_state_dict(
            new_model,
            new_optimizer,
            model_state_dict=state_dict_to_load["model"],
            optim_state_dict=state_dict_to_load["optimizer"],
        )
        non_ap_loss = []
        for i in range(11, 20):
            loss = train_step(new_model, new_optimizer, test_input, mesh)
            if i % 1 == 0:
                master_print(f"(after load from ckp) step: {i}, {loss=}")
            if i >= 11:
                non_ap_loss.append(loss)
        # TODO(zpcore): enable the following assertion check once we address
        # comment https://github.com/pytorch/pytorch/pull/165197#issuecomment-3429728972

        # assert all( torch.allclose(i, j, rtol=1e-2) for i, j in zip(ap_loss,
        # non_ap_loss) ), "DCP loss curve mismatch when load state dict from AP
        # model to non AP model"
        allclose_result = all(
            torch.allclose(i, j, rtol=1e-2) for i, j in zip(ap_loss, non_ap_loss)
        )
        if not allclose_result:
            master_print(
                "DCP loss curve mismatch when load state dict from AP model to non AP model"
            )
    except Exception as e:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        raise e

    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def main():
    fake_world_size = 256
    # single process to generate sharding placement
    model, sharding_map = prepare_autoparallel_model(fake_world_size)
    world_size = 4
    tmp_dir = tempfile.mkdtemp()
    # multiprocess reuse generated sharding placement and get real output
    mp.spawn(
        multiple_process_run,
        args=(world_size, tmp_dir, model, sharding_map),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
