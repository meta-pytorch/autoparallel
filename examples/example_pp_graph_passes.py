# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext
from typing import Callable

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.distributed.tensor.placement_types import Shard
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel._testing.models.dsv3 import (
    DeepSeekV3Model,
    DeepSeekV3ModelArgs,
    MoEArgs,
)
from autoparallel.api import AutoParallelPP
from autoparallel.graph_pp_runner import (
    GraphCallables,
    GraphMeta,
    _run_dI_bw_module,
    _run_dW_bw_module,
    _run_full_bw_module,
    _run_fw_module,
    _run_reduce_grad_module,
    _run_unshard_module,
)


def _get_pp_module_and_graphs(
    model: torch.nn.Module,
    mesh: DeviceMesh,
    tracing_input_fn: Callable,
    graph_passes: list[str] = [],
) -> tuple[torch.nn.Module, GraphCallables, GraphMeta]:

    with AutoParallelPP(
        model,
        tracing_input_fn,
        mesh,
        dynamic=True,
        compile=False,
        reshard_after_forward=False,
    ) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)

        # x_sharding = (Shard(0), Replicate())
        x_sharding = (Shard(0), Shard(0))

        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([x_sharding])

        sharding_placement = autop.optimize_placement()
        res = autop.apply_placement_pp(
            sharding_placement=sharding_placement,
            graph_passes=graph_passes,
        )
        pp_mod = autop.parallel_model
    graph_callables = res["graph_callables"]
    graph_modules = GraphCallables(
        fw=graph_callables["fw"],
        full_bw=graph_callables["full_bw"],
        bw_dI=graph_callables["bw_dI"],
        bw_dW=graph_callables["bw_dW"],
        unshard=graph_callables["unshard"],
        reduce_grad=graph_callables["reduce_grad"],
    )
    graph_meta = res["graph_meta"]
    graph_meta = GraphMeta(
        num_mutate_inputs=graph_meta["num_mutate_inputs"],
        num_user_outputs=graph_meta["num_user_outputs"],
        num_symints_saved_for_bw=graph_meta["num_symints_saved_for_bw"],
        num_params=graph_meta["num_params"],
        num_buffers=graph_meta["num_buffers"],
        num_input_grads=graph_meta["num_input_grads"],
    )

    pp_mod.to_empty(device="cuda")
    pp_mod.init_weights(buffer_device="cuda")
    return pp_mod, graph_modules, graph_meta


# graph_passes=["split_dI_dW", "split_fsdp_collectives"],


def _get_fw_inputs(
    pp_mod: torch.nn.Module, eval_input_fn: Callable
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    x: list[torch.Tensor] = [
        eval_input_fn(),
    ]
    sharded_params = [
        v.to_local() if isinstance(v, DTensor) else v
        for k, v in dict(pp_mod.named_parameters(remove_duplicate=False)).items()
    ]
    buffers = [
        v.to_local() if isinstance(v, DTensor) else v
        for k, v in dict(pp_mod.named_buffers(remove_duplicate=False)).items()
    ]
    return [sharded_params, buffers, x]


# Symbolically evaluate in case you want to test running a graph bigger than your gpu


def test_graph_partition(
    model: torch.nn.Module,
    mesh: DeviceMesh,
    tracing_input_fn: Callable,
    eval_input_fn: Callable,
    fake_evaluate: bool = True,
):

    pp_mod, graph_modules, graph_meta = _get_pp_module_and_graphs(
        model, mesh, tracing_input_fn
    )
    sharded_params, buffers, x = _get_fw_inputs(pp_mod, eval_input_fn)
    with (
        FakeTensorMode(
            allow_non_fake_inputs=True,
            shape_env=ShapeEnv(),
        )
        if fake_evaluate
        else nullcontext()
    ):
        # # now let's run it
        with torch.no_grad():
            fw_args = [*sharded_params, *buffers, *x]
            output, saved_intermediates = _run_fw_module(
                graph_modules.fw, graph_meta, fw_args
            )
            tangents = [torch.randn_like(output)]
            tensors_for_backward, non_tensors_for_backward = saved_intermediates

            bw_args = [
                *non_tensors_for_backward,
                *tensors_for_backward,
                *tangents,
            ]
            del (
                tensors_for_backward,
                non_tensors_for_backward,
                tangents,
                saved_intermediates,
            )

            input_grads, param_buffer_grads = _run_full_bw_module(
                graph_modules.full_bw, graph_meta, bw_args
            )

    print("All good!")


def test_split_fsdp_collectives(
    model: torch.nn.Module,
    mesh: DeviceMesh,
    tracing_input_fn: Callable,
    eval_input_fn: Callable,
    fake_evaluate: bool = True,
):

    pp_mod, graph_modules, graph_meta = _get_pp_module_and_graphs(
        model, mesh, tracing_input_fn, graph_passes=["split_fsdp_collectives"]
    )
    sharded_params, buffers, x = _get_fw_inputs(pp_mod, eval_input_fn)
    with (
        FakeTensorMode(
            allow_non_fake_inputs=True,
            shape_env=ShapeEnv(),
        )
        if fake_evaluate
        else nullcontext()
    ):
        # # now let's run it
        with torch.no_grad():
            unshard_args = list(sharded_params)
            assert graph_modules.unshard is not None
            unsharded_params = _run_unshard_module(
                graph_modules.unshard, graph_meta, unshard_args
            )
            fw_args = [*unsharded_params, *buffers, *x]
            output, saved_intermediates = _run_fw_module(
                graph_modules.fw, graph_meta, fw_args
            )
            tangents = [torch.randn_like(output)]
            tensors_for_backward, non_tensors_for_backward = saved_intermediates

            bw_args = [
                *non_tensors_for_backward,
                *tensors_for_backward,
                *tangents,
            ]
            del (
                tensors_for_backward,
                non_tensors_for_backward,
                tangents,
                saved_intermediates,
            )
            input_grads, unsharded_param_buffer_grads = _run_full_bw_module(
                graph_modules.full_bw, graph_meta, bw_args
            )
            unsharded_grads = list(unsharded_param_buffer_grads[: len(sharded_params)])
            del unsharded_param_buffer_grads, input_grads
            assert graph_modules.reduce_grad is not None
            sharded_grads = _run_reduce_grad_module(
                graph_modules.reduce_grad, graph_meta, unsharded_grads
            )
            assert len(sharded_grads) == len(sharded_params)

    print("All good!")


def test_split_dI_dW(
    model: torch.nn.Module,
    mesh: DeviceMesh,
    tracing_input_fn: Callable,
    eval_input_fn: Callable,
    fake_evaluate: bool = True,
):

    pp_mod, graph_modules, graph_meta = _get_pp_module_and_graphs(
        model, mesh, tracing_input_fn, graph_passes=["split_dI_dW"]
    )
    sharded_params, buffers, x = _get_fw_inputs(pp_mod, eval_input_fn)
    with (
        FakeTensorMode(
            allow_non_fake_inputs=True,
            shape_env=ShapeEnv(),
        )
        if fake_evaluate
        else nullcontext()
    ):
        # # now let's run it
        with torch.no_grad():
            fw_args = [*sharded_params, *buffers, *x]
            output, saved_intermediates = _run_fw_module(
                graph_modules.fw, graph_meta, fw_args
            )
            tangents = [torch.randn_like(output)]
            tensors_for_backward, non_tensors_for_backward = saved_intermediates

            bw_args = [
                *non_tensors_for_backward,
                *tensors_for_backward,
                *tangents,
            ]
            del (
                tensors_for_backward,
                non_tensors_for_backward,
                tangents,
                saved_intermediates,
            )
            assert graph_modules.bw_dI is not None
            input_grads, activations_for_backward = _run_dI_bw_module(
                graph_modules.bw_dI, graph_meta, bw_args
            )
            dw_args = list(activations_for_backward)
            del activations_for_backward
            assert graph_modules.bw_dW is not None
            sharded_param_buffer_grads = _run_dW_bw_module(
                graph_modules.bw_dW, graph_meta, dw_args
            )
            assert len(sharded_param_buffer_grads) == (
                len(sharded_params) + len(buffers)
            )

    print("All good!")


def test_combined(
    model: torch.nn.Module,
    mesh: DeviceMesh,
    tracing_input_fn: Callable,
    eval_input_fn: Callable,
    fake_evaluate: bool = True,
):

    pp_mod, graph_modules, graph_meta = _get_pp_module_and_graphs(
        model,
        mesh,
        tracing_input_fn,
        graph_passes=["split_fsdp_collectives", "split_dI_dW"],
    )
    sharded_params, buffers, x = _get_fw_inputs(pp_mod, eval_input_fn)
    with (
        FakeTensorMode(
            allow_non_fake_inputs=True,
            shape_env=ShapeEnv(),
        )
        if fake_evaluate
        else nullcontext()
    ):
        # # now let's run it
        with torch.no_grad():
            unshard_args = list(sharded_params)
            assert graph_modules.unshard is not None
            unsharded_params = _run_unshard_module(
                graph_modules.unshard, graph_meta, unshard_args
            )
            fw_args = [*unsharded_params, *buffers, *x]
            output, saved_intermediates = _run_fw_module(
                graph_modules.fw, graph_meta, fw_args
            )
            tangents = [torch.randn_like(output)]
            tensors_for_backward, non_tensors_for_backward = saved_intermediates

            bw_args = [
                *non_tensors_for_backward,
                *tensors_for_backward,
                *tangents,
            ]
            del (
                tensors_for_backward,
                non_tensors_for_backward,
                tangents,
                saved_intermediates,
            )
            assert graph_modules.bw_dI is not None
            input_grads, activations_for_backward = _run_dI_bw_module(
                graph_modules.bw_dI, graph_meta, bw_args
            )
            dw_args = list(activations_for_backward)
            del activations_for_backward
            assert graph_modules.bw_dW is not None
            unsharded_param_buffer_grads = _run_dW_bw_module(
                graph_modules.bw_dW, graph_meta, dw_args
            )
            unsharded_grads = list(unsharded_param_buffer_grads[: len(sharded_params)])
            del unsharded_param_buffer_grads, input_grads
            assert graph_modules.reduce_grad is not None
            sharded_grads = _run_reduce_grad_module(
                graph_modules.reduce_grad, graph_meta, unsharded_grads
            )
            assert len(sharded_grads) == len(sharded_params)

    print("All good!")


if __name__ == "__main__":
    # must symbolically evaluate to run on 32 dp ranks
    # world_size = 2048
    fake_evaluate = True

    world_size = 256

    fake_store = FakeStore()
    torch.distributed.init_process_group(
        "fake", store=fake_store, rank=0, world_size=world_size
    )
    # mesh = torch.distributed.device_mesh.init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda",
        (world_size // 64, 64),
        mesh_dim_names=(
            "dp",
            "ep",
        ),
    )

    device = torch.device("cuda")

    bs = 4 * mesh.shape[0] * mesh.shape[1]
    seq_len = 1024

    config = DeepSeekV3ModelArgs(
        vocab_size=102400,
        max_seq_len=seq_len,
        dim=2048,
        inter_dim=10944,
        moe_inter_dim=1408,
        n_layers=1,  # 27,
        n_dense_layers=0,  # 1,
        n_heads=16,
        moe_args=MoEArgs(
            num_experts=64,
            num_shared_experts=2,
            top_k=6,
            score_func="softmax",
            route_norm=False,
            score_before_experts=False,
            mesh=mesh,
        ),
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mscale=0.70,
        use_flex_attn=False,
        attn_mask_type="causal",
    )

    # parallelize the model
    with torch.device("meta"):
        model = DeepSeekV3Model(config).bfloat16()
        model.tok_embeddings = None  # type: ignore[assignment]

    def tracing_input_fn() -> torch.Tensor:
        return torch.randn(
            (bs, seq_len, config.dim),
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )

    def eval_input_fn() -> torch.Tensor:
        return torch.randn(
            (bs // mesh.shape[0] // mesh.shape[1], seq_len, config.dim),
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )

    test_graph_partition(model, mesh, tracing_input_fn, eval_input_fn, fake_evaluate)
    test_split_fsdp_collectives(
        model, mesh, tracing_input_fn, eval_input_fn, fake_evaluate
    )
    test_split_dI_dW(model, mesh, tracing_input_fn, eval_input_fn, fake_evaluate)
    test_combined(model, mesh, tracing_input_fn, eval_input_fn, fake_evaluate)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
