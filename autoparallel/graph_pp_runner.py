# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Callable, Optional, Union, cast

import torch
import torch.fx as fx
from torch.distributed.pipelining.schedules import (
    _Action,
    _PipelineContext,
    _PipelineScheduleRuntime,
    _wait_batch_p2p,
)
from torch.distributed.pipelining.stage import (
    PipelineStage,
    _normalize_model_output_as_tuple,
)
from torch.distributed.tensor import DTensor


@dataclass
class GraphCallables:
    fw: fx.GraphModule
    full_bw: fx.GraphModule
    bw_dI: Optional[fx.GraphModule] = None
    bw_dW: Optional[fx.GraphModule] = None
    unshard: Optional[fx.GraphModule] = None
    reduce_grad: Optional[fx.GraphModule] = None


@dataclass
class GraphMeta:
    num_mutate_inputs: int
    num_user_outputs: int
    num_symints_saved_for_bw: int
    num_weight_buffer_grads: int


class GraphPipelineStage(PipelineStage):
    def __init__(
        self,
        submodule: torch.nn.Module,
        graph_callables: GraphCallables,
        graph_meta: GraphMeta,
        stage_index: int,
        num_stages: int,
        device: torch.device,
        input_args: Optional[Union[torch.Tensor, tuple[torch.Tensor, ...]]] = None,
        output_args: Optional[Union[torch.Tensor, tuple[torch.Tensor, ...]]] = None,
        group: Optional[torch.distributed.ProcessGroup] = None,
        dw_builder: Optional[Callable[[], Callable[..., None]]] = None,
    ):
        super().__init__(
            submodule=submodule,
            stage_index=stage_index,
            num_stages=num_stages,
            device=device,
            input_args=input_args,
            output_args=output_args,
            group=group,
            dw_builder=dw_builder,
        )
        self.graph_callables = graph_callables
        self.graph_meta = graph_meta
        self.state: dict[str, list[Any]] = {
            "sharded_params": [],
            "unsharded_params": [],
            "buffers": [],
            "sharded_grads": [],
            "unsharded_grads": [],
        }


def _run_fw_module(
    fw_module: fx.GraphModule, graph_meta: GraphMeta, fw_args: list[Any]
) -> tuple[Any, tuple[Any, Any]]:
    assert len([n for n in fw_module.graph.nodes if n.op == "placeholder"]) == len(
        fw_args
    ), f"Mismatched number of inputs to fwd, {len([n for n in fw_module.graph.nodes if n.op == 'placeholder'])}, {len(fw_args)}"
    fw_outputs = torch.fx.Interpreter(fw_module).boxed_run(fw_args)
    num_inner_fwd_outputs = graph_meta.num_mutate_inputs + graph_meta.num_user_outputs
    saved_intermediates = fw_outputs[num_inner_fwd_outputs:]
    num_tensors_for_backward = (
        len(saved_intermediates) - graph_meta.num_symints_saved_for_bw
    )
    tensors_for_backward = saved_intermediates[:num_tensors_for_backward]
    non_tensors_for_backward = saved_intermediates[num_tensors_for_backward:]
    save_for_backward = (tensors_for_backward, non_tensors_for_backward)
    user_outputs = fw_outputs[graph_meta.num_mutate_inputs : num_inner_fwd_outputs]
    if len(user_outputs) == 1:
        user_outputs = user_outputs[0]
    return user_outputs, save_for_backward


def _run_full_bw_module(
    bw_module: fx.GraphModule, graph_meta: GraphMeta, bw_args
) -> tuple[Any, list[Any]]:
    assert len([n for n in bw_module.graph.nodes if n.op == "placeholder"]) == len(
        bw_args
    ), "Mismatched number of inputs to bwd"
    bw_outputs = torch.fx.Interpreter(bw_module).boxed_run(bw_args)
    param_buffer_grads = bw_outputs[: graph_meta.num_weight_buffer_grads]
    input_grads = bw_outputs[graph_meta.num_weight_buffer_grads :]
    return input_grads, param_buffer_grads


def _run_forward_microbatch(stage: GraphPipelineStage, *args) -> tuple[Any, Any]:
    fw_args = [
        *stage.state["unsharded_params"],
        *stage.state["buffers"],
        *args,
    ]
    user_outputs, saved_intermediates = _run_fw_module(
        stage.graph_callables.fw, stage.graph_meta, fw_args
    )
    return (user_outputs, saved_intermediates)


def _run_backward_microbatch(
    backward_stage: GraphPipelineStage, bwd_kwargs: dict[str, Any]
):
    tangents = bwd_kwargs["tangents"]
    saved_intermediates = bwd_kwargs["saved_intermediates"]
    tensors_for_backward, non_tensors_for_backward = saved_intermediates

    bw_args = [
        *non_tensors_for_backward,
        *tensors_for_backward,
        *tangents,
    ]
    input_grads, param_buffer_grads = _run_full_bw_module(
        backward_stage.graph_callables.full_bw, backward_stage.graph_meta, bw_args
    )

    unsharded_grads = backward_stage.state["unsharded_grads"]
    grads_to_accumulate = param_buffer_grads[
        : len(backward_stage.state["sharded_params"])
    ]
    assert len(unsharded_grads) == len(grads_to_accumulate)
    for unsharded_grad, grad_to_accumulate in zip(unsharded_grads, grads_to_accumulate):
        if grad_to_accumulate is not None:
            if unsharded_grad is None:
                unsharded_grad = grad_to_accumulate
            else:
                unsharded_grad += grad_to_accumulate
    return input_grads


def stage_forward(
    action: _Action,
    ctx: _PipelineContext,
) -> None:
    schedule = ctx.schedule_ref
    assert isinstance(schedule, _PipelineScheduleRuntime)
    stage_index_to_stage: dict[int, GraphPipelineStage] = {
        stage.stage_index: cast(GraphPipelineStage, stage) for stage in schedule._stages
    }
    stage = stage_index_to_stage[action.stage_index]
    stage_index = stage.stage_index

    mb_index = action.microbatch_index
    assert mb_index is not None
    fwd_recv_ops = schedule.fwd_recv_ops
    arg_mbs = ctx.arg_mbs
    kwarg_mbs = ctx.kwarg_mbs

    is_next_stage_on_this_rank = stage_index + 1 in stage_index_to_stage
    is_prev_stage_on_this_rank = stage_index - 1 in stage_index_to_stage

    if (
        not stage.is_first
        # no recv op expected for V-schedule special case (see [Note: V-schedule special case])
        and not is_prev_stage_on_this_rank
    ):
        assert (
            stage_index,
            mb_index,
        ) in fwd_recv_ops, f"Computing {action=} before receiving input"

        _wait_batch_p2p(fwd_recv_ops.pop((stage_index, mb_index)))

    args = arg_mbs[mb_index]  # type: ignore[index]
    kwargs = kwarg_mbs[mb_index]  # type: ignore[index]
    assert not kwargs  # TODO: if kwargs can always be ignored, maybe remove?

    if stage.is_first:
        # First stage doesn't need to receive anything
        composite_args = args
    else:
        # Receive activations for this chunk
        # Activations only come in args form
        composite_args = stage._retrieve_recv_activations(mb_index)

    # stage._validate_fwd_input(args, kwargs) Maybe need to validate composite args?

    output, saved_intermediates = _run_forward_microbatch(stage, *composite_args)

    # See [Note: pipeline model output type]
    output_tuple = _normalize_model_output_as_tuple(output)

    # Prepare for final output merge or reduction
    # Output chunks is only used for the last stage since we only merge the output of the last stage
    if stage.is_last:
        stage.output_chunks.append(output)

    stage.fwd_cache[mb_index] = (
        output_tuple,  # stage_output
        saved_intermediates,  # saved_intermediates
    )

    #  stage._validate_fwd_outputs(output_tuple)

    schedule._maybe_compute_loss(stage, output, ctx.target_mbs, mb_index)

    # SEND/RECV op are avoided for special case with 2 adjacent stages on same rank
    # see [Note: V-schedule special case]
    if is_next_stage_on_this_rank:
        stage_index_to_stage[stage_index + 1].set_local_fwd_input(output, mb_index)


def stage_full_backward(
    action: _Action,
    ctx: _PipelineContext,
) -> None:
    schedule = ctx.schedule_ref
    assert isinstance(schedule, _PipelineScheduleRuntime)
    stage_index_to_stage: dict[int, GraphPipelineStage] = {
        stage.stage_index: cast(GraphPipelineStage, stage) for stage in schedule._stages
    }

    backward_stage_index = action.stage_index
    backward_stage = stage_index_to_stage[backward_stage_index]
    backward_mb_index = action.microbatch_index
    assert backward_mb_index is not None
    bwd_recv_ops = schedule.bwd_recv_ops
    is_next_stage_on_this_rank = backward_stage.stage_index + 1 in stage_index_to_stage
    is_prev_stage_on_this_rank = backward_stage.stage_index - 1 in stage_index_to_stage

    if (
        not backward_stage.is_last
        # no recv op expected for V-schedule special case (see [Note: V-schedule special case])
        and not is_next_stage_on_this_rank
    ):
        assert (
            backward_stage_index,
            backward_mb_index,
        ) in bwd_recv_ops, f"Attempted to run compute {action=} before receiving input"
        _wait_batch_p2p(bwd_recv_ops.pop((backward_stage_index, backward_mb_index)))

    loss = schedule._maybe_get_loss(backward_stage, backward_mb_index)
    schedule.backward_counter[backward_stage_index] += 1
    last_backward = (
        schedule.backward_counter[backward_stage_index] == schedule._n_microbatches
    )
    grad_scale_factor = schedule._n_microbatches if schedule.scale_grads else 1

    if not backward_stage.has_backward:
        return
    (
        stage_output,
        saved_intermediates,
    ) = backward_stage.fwd_cache.pop(backward_mb_index)

    # Compute backward
    if backward_stage.is_last:
        # Last stage computes gradients from loss and has no gradients from
        # next stage
        # TODO(sanketpurandare)
        # HACK till we have loss function, we populate the tangents here manually
        bwd_kwargs = {
            "stage_output": loss,
            "tangents": [torch.randn_like(stage_output)],
            "saved_intermediates": saved_intermediates,
        }
    else:
        # Otherwise, receive gradients from next stage
        output_grads = backward_stage._retrieve_recv_grads(backward_mb_index)
        # If an input to the pipeline requires gradient,
        # `torch.autograd.backward` will accumulate the gradient into the
        # `.grad` field of such input
        bwd_kwargs = {
            "stage_output": stage_output,
            "tangents": output_grads,
            "saved_intermediates": saved_intermediates,
        }

    input_grads = _run_backward_microbatch(backward_stage, bwd_kwargs)

    backward_stage.bwd_cache[backward_mb_index] = input_grads

    # skipping detach logic

    if last_backward:
        backward_stage.scale_grads(grad_scale_factor)
    # SEND/RECV op are avoided for special case with 2 adjacent stages on same rank
    # see [Note: V-schedule special case]
    if is_prev_stage_on_this_rank:
        stage_index_to_stage[backward_stage_index - 1].set_local_bwd_input(
            backward_stage.get_local_bwd_output(backward_mb_index),
            backward_mb_index,
        )


def stage_unshard(
    action: _Action,
    ctx: _PipelineContext,
) -> None:
    schedule = ctx.schedule_ref
    assert isinstance(schedule, _PipelineScheduleRuntime)
    stage_index_to_stage: dict[int, GraphPipelineStage] = {
        stage.stage_index: cast(GraphPipelineStage, stage) for stage in schedule._stages
    }
    stage = stage_index_to_stage[action.stage_index]
    if stage.graph_callables.unshard is None:
        stage.state["unsharded_params"] = stage.state["sharded_params"]
    # TODO (sanketpurandare): Add the fw_fsdp_all_gather graph call here


def stage_reshard(
    action: _Action,
    ctx: _PipelineContext,
):
    schedule = ctx.schedule_ref
    assert isinstance(schedule, _PipelineScheduleRuntime)
    stage_index_to_stage: dict[int, GraphPipelineStage] = {
        stage.stage_index: cast(GraphPipelineStage, stage) for stage in schedule._stages
    }
    stage = stage_index_to_stage[action.stage_index]
    stage.state["unsharded_params"].clear()


def stage_reduce_grad(
    action: _Action,
    ctx: _PipelineContext,
) -> None:
    schedule = ctx.schedule_ref
    assert isinstance(schedule, _PipelineScheduleRuntime)
    stage_index_to_stage: dict[int, GraphPipelineStage] = {
        stage.stage_index: cast(GraphPipelineStage, stage) for stage in schedule._stages
    }
    stage = stage_index_to_stage[action.stage_index]
    if stage.graph_callables.reduce_grad is None:
        stage.state["sharded_grads"] = stage.state["unsharded_grads"]


class GraphPPRunner:
    def __init__(
        self,
        schedule: _PipelineScheduleRuntime,
    ):
        self.schedule = schedule

    def _populate_stage_states(self, stage: GraphPipelineStage) -> None:
        sharded_params = [
            v.to_local() if isinstance(v, DTensor) else v
            for k, v in dict(
                stage.submod.named_parameters(remove_duplicate=False)
            ).items()
        ]
        buffers = [
            v.to_local() if isinstance(v, DTensor) else v
            for k, v in dict(stage.submod.named_buffers(remove_duplicate=False)).items()
        ]
        stage.state["sharded_params"] = sharded_params
        stage.state["buffers"] = buffers
        stage.state["unsharded_grads"] = [None] * len(sharded_params)
        # TODO (sanketpurandare)
        # pipeline schedule runtime does not allow us to register a custom function
        # for UNSHARD/RESHARD/REDUCE_GRAD action types yet
        # HACK remove this once we support this
        if stage.graph_callables.unshard is None:
            stage.state["unsharded_params"] = stage.state["sharded_params"]

    def _accumulate_stage_grads_and_clear_states(
        self, stage: GraphPipelineStage
    ) -> None:
        # TODO (sanketpurandare)
        # We don't have a REDUCE_GRAD action yet in the ScheduleIR yet
        # HACK remove this once Ivan's PR lands
        if stage.graph_callables.reduce_grad is None:
            stage.state["sharded_grads"] = stage.state["unsharded_grads"]
        grads = stage.state["sharded_grads"]
        params = list(stage.submod.parameters())
        for param, grad in zip(params, grads):
            if param.requires_grad and grad is not None:
                assert isinstance(grad, torch.Tensor)
                if isinstance(param, DTensor):
                    param_spec = param._spec
                    _grad = DTensor.from_local(
                        grad,
                        device_mesh=param_spec.device_mesh,
                        placements=param_spec.placements,
                        shape=param_spec.shape,
                        stride=param_spec.stride,
                    )
                else:
                    _grad = grad  # type: ignore[assignment]
                if param.grad is None:
                    param.grad = _grad
                else:
                    param.grad += _grad
        stage.state.clear()

    def step(self, *args, **kwargs) -> None:

        for stage in self.schedule._stages:
            assert isinstance(stage, GraphPipelineStage)
            self._populate_stage_states(stage)

        self.schedule.step(*args, **kwargs)

        for stage in self.schedule._stages:
            assert isinstance(stage, GraphPipelineStage)
            self._accumulate_stage_grads_and_clear_states(stage)
