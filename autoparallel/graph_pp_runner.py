import itertools
import re
from dataclasses import dataclass
from typing import Any, Callable, cast, Optional, Union

import torch
import torch.fx as fx
from torch.distributed.pipelining.schedules import (
    _Action,
    _PipelineContext,
    _PipelineScheduleRuntime,
    _wait_batch_p2p,
)
from torch.distributed.pipelining.stage import (
    _normalize_model_output_as_tuple,
    _PipelineStage,
)
from torch.distributed.tensor import DTensor


@dataclass
class GraphCallables:
    forward: fx.GraphModule
    backward: fx.GraphModule
    backward_inputs: Optional[fx.GraphModule] = None
    backward_weights: Optional[fx.GraphModule] = None
    unshard: Optional[fx.GraphModule] = None
    reduce_grad: Optional[fx.GraphModule] = None


@dataclass
class GraphMeta:
    num_mutate_inputs: int
    num_user_outputs: int
    num_symints_saved_for_bw: int


class GraphPipelineStage(_PipelineStage):
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


def _run_forward_microbatch(stage: GraphPipelineStage, *args) -> tuple[Any, Any]:
    stage_mod = stage.submod
    stage_graphs = stage.graph_callables
    stage_graph_meta = stage.graph_meta
    params_and_buffers = [
        v.to_local() if isinstance(v, DTensor) else v
        for k, v in itertools.chain(
            dict(stage_mod.named_parameters(remove_duplicate=False)).items(),
            dict(stage_mod.named_buffers(remove_duplicate=False)).items(),
        )
    ]
    boxed_args = [
        *params_and_buffers,
        *args,
    ]
    del params_and_buffers
    fw_module = stage_graphs.forward
    fw_outputs = torch.fx.Interpreter(fw_module).boxed_run(boxed_args)
    num_inner_fwd_outputs = (
        stage_graph_meta.num_mutate_inputs + stage_graph_meta.num_user_outputs
    )
    saved_intermediates = fw_outputs[num_inner_fwd_outputs:]
    user_outputs = fw_outputs[
        stage_graph_meta.num_mutate_inputs : num_inner_fwd_outputs
    ]
    if len(user_outputs) == 1:
        user_outputs = user_outputs[0]
    return (user_outputs, saved_intermediates)


def _run_backward_microbatch(
    backward_stage: GraphPipelineStage, bwd_kwargs: dict[str, Any]
):

    stage_mod = backward_stage.submod
    stage_graphs = backward_stage.graph_callables
    stage_graph_meta = backward_stage.graph_meta

    tangents = bwd_kwargs["tangents"]
    saved_intermediates = bwd_kwargs["saved_intermediates"]
    bw_module = stage_graphs.backward

    num_tensors_for_backward = (
        len(saved_intermediates) - stage_graph_meta.num_symints_saved_for_bw
    )
    tensors_for_backward = saved_intermediates[:num_tensors_for_backward]
    non_tensors_for_backward = saved_intermediates[num_tensors_for_backward:]

    bw_args = [
        *non_tensors_for_backward,
        *tensors_for_backward,
        *tangents,
    ]
    assert len([n for n in bw_module.graph.nodes if n.op == "placeholder"]) == len(
        bw_args
    ), "Mismatched number of inputs to bwd"
    bw_outputs = torch.fx.Interpreter(bw_module).boxed_run(bw_args)
    result = bw_outputs
    input_grads = result[: stage_graph_meta.num_user_outputs]
    weight_grads = result[stage_graph_meta.num_user_outputs :]

    # TODO(sanketpurandare) perform gradient accumulation here with the stage_mod weights

    return input_grads


def run_forward_graph(
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

    if stage.is_first:
        # First stage doesn't need to receive anything
        composite_args = args
    else:
        # Receive activations for this chunk
        # Activations only come in args form
        composite_args = stage._retrieve_recv_activations(mb_index)

        composite_kwargs = kwargs or {}

    # stage._validate_fwd_input(args, kwargs) Maybe need to validate composite args?

    output, saved_intermediates = _run_forward_microbatch(stage, *composite_args)

    # See what a pipeline stage is supposed to do in forward

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


def run_backward_graph(
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
        bwd_kwargs = {
            "stage_output": loss,
            "tangents": [],
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
