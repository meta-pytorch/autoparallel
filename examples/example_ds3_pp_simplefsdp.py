# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging
import copy
import os
from contextlib import nullcontext
from typing import Callable

import torch
import torch.distributed._tools.fake_collectives
import torch.distributed as dist
import torch.nn as nn
from torch._logging import trace_structured
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.pipelining.schedules import (
    FORWARD,
    FULL_BACKWARD,
    PipelineScheduleMulti,
    _PipelineSchedule,
    _PipelineScheduleRuntime,
    get_schedule_class,
)
from torch.distributed.pipelining.stage import PipelineStage
from torch.distributed.tensor.placement_types import Shard
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel._testing.models.dsv3 import (
    DeepSeekV3Model,
    DeepSeekV3ModelArgs,
    DeepSeekV3Stage0,
    DeepSeekV3StageI,
    DeepSeekV3StageN,
    MoEArgs,
)
from autoparallel._testing.models.llama3 import Transformer, TransformerModelArgs
from autoparallel.api import AutoParallelPP
from autoparallel.graph_pp_runner import (
    GraphCallables,
    GraphMeta,
    GraphPipelineStage,
    GraphPPRunner,
    stage_forward,
    stage_full_backward,
)
from torchtitan.experiments.simple_fsdp.simple_fsdp import data_parallel as simple_fsdp_data_parallel
from autoparallel.api import move_to_fake
from torchtitan.models.llama3.infra.parallelize import apply_tp
from autoparallel.cast_parametrization import apply_dtype_cast, canonicalize_mp, set_dtype_cast
from autoparallel.api import _export, enable_local_map_wrapping
from contextlib import ExitStack
from torch._functorch.aot_autograd import aot_export_joint_with_descriptors

from torch.distributed.tensor import DTensor, Replicate
from torch.utils._pytree import register_pytree_node, tree_map
from torch._functorch._aot_autograd.fx_utils import (
    get_named_buffer_nodes,
    get_named_param_nodes,
)
from torch._subclasses.fake_tensor import unset_fake_temporarily

logger = logging.getLogger(__name__)


def build_pipeline_schedule(
    stages: list[PipelineStage],
    loss_fn: Callable,
    pipeline_parallel_schedule: str,
    microbatch_size: int,
    local_batch_size: int,
    pipeline_parallel_degree: int,
) -> _PipelineSchedule:
    """Builds a pipeline schedule for the given configuration and stages."""
    schedule_class = get_schedule_class(pipeline_parallel_schedule)

    looped_schedule = issubclass(schedule_class, PipelineScheduleMulti)
    assert looped_schedule, "Only looped schedules are supported"
    # validate that the batch size is divisible by the microbatch_size otherwise we'll hang or error during training
    if local_batch_size % microbatch_size != 0:
        raise ValueError(
            f"Batch size {local_batch_size} must be divisible by {microbatch_size=}. "
        )
    n_microbatches = local_batch_size // microbatch_size
    # We expect that the number of local stages (`len(stages)`) is the same across all pp ranks
    num_total_stages = pipeline_parallel_degree * len(stages)
    if n_microbatches < num_total_stages:
        logger.warning(
            f"Number of microbatches ({n_microbatches}) is less than the total number "
            f"of stages ({num_total_stages}) which may result in a bubble in the pipeline."
        )

    schedule = schedule_class(
        stages if looped_schedule else stages[0],
        n_microbatches=n_microbatches,
        loss_fn=loss_fn,
    )
    logger.info(
        f"Using pipeline schedule {pipeline_parallel_schedule} "
        f"with {n_microbatches} microbatches and {num_total_stages} stages."
    )
    return schedule


def run_test(fake_evaluate: bool = True):
    if not fake_evaluate:
        pp_degree = 2
        dp_mod_ep_degree = 2
        ep_degree = 2
    else:
        pp_degree = 4
        dp_mod_ep_degree = 4
        ep_degree = 64

    dp_degree = dp_mod_ep_degree * ep_degree
    world_size = pp_degree * dp_mod_ep_degree * ep_degree

    # Initialize process group based on evaluation mode
    if fake_evaluate:
        assert (
            "WORLD_SIZE" in os.environ
        ), "run with torchrun --standalone --nproc-per-node 4"
        assert (
            int(os.getenv("WORLD_SIZE")) == pp_degree
        ), "world_size must be 4, for fake evaluation"
        rank = int(os.getenv("RANK"))
        device = torch.device(f"cuda:{rank}")
        fake_store = FakeStore()
        torch.distributed.init_process_group(
            "fake",
            store=fake_store,
            rank=rank * dp_degree,  # global rank is pp_rank * spmd_size
            world_size=world_size,
        )
        pp_rank = rank
    else:
        assert (
            "WORLD_SIZE" in os.environ
        ), "run with torchrun --standalone --nproc-per-node 8"
        assert (
            int(os.getenv("WORLD_SIZE")) == world_size
        ), "Need at least 8 GPUs for real evaluation"
        local_rank = int(os.getenv("LOCAL_RANK"))
        device = torch.device(f"cuda:{local_rank}")
        torch.distributed.init_process_group(backend="nccl")

    # Initialize device mesh (common for both modes)
    world_mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda",
        (pp_degree, dp_mod_ep_degree, ep_degree),
        mesh_dim_names=(
            "pp",
            "dp_mod_ep",
            "ep",
        ),
    )

    # Set pp_rank based on evaluation mode
    if not fake_evaluate:
        pp_rank = world_mesh["pp"].get_local_rank()

    stages_per_rank = 2
    total_pp_stages = pp_degree * stages_per_rank

    # This is the spmd mesh to be used for tracing
    mesh = world_mesh[("dp_mod_ep", "ep")]

    global_batch_size = 32 * dp_degree
    # Batch size that will be supplied to the schedule and will be broken down into microbatches
    local_batch_size = global_batch_size // dp_degree
    n_microbatches = 16
    # Batch size with which the spmd graphs will actually be executed
    microbatch_size = local_batch_size // n_microbatches
    assert (
        microbatch_size >= 1
    ), f"invalid config {local_batch_size=}, {n_microbatches=}"
    # Batch size to be used for spmd tracing
    spmd_batch_size = microbatch_size * dp_degree
    print(f"XXX dp_degree:{dp_degree}")
    print(f"XXX global_batch_size:{global_batch_size}")
    print(f"XXX spmd_batch_size:{spmd_batch_size}")
    print(f"XXX local_batch_size:{local_batch_size}")
    print(f"XXX microbatch_size:{microbatch_size}")

    seq_len = 1024

    if fake_evaluate:
        # eval mode fake
        config = DeepSeekV3ModelArgs(
            vocab_size=102400,
            max_seq_len=seq_len,
            dim=2048,
            inter_dim=10944,
            moe_inter_dim=1408,
            n_layers=8,  # 27,
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
        def _model_llama3():
            vocab_size = 128256
            model_args = TransformerModelArgs(
                dim=2048,
                n_layers=4,
                n_heads=32,
                n_kv_heads=8,
                ffn_dim_multiplier=1.3,
                multiple_of=1024,
                rope_theta=500000,
                vocab_size=vocab_size,
                max_seq_len=seq_len,
            )
            m = Transformer(model_args)
            return m
    else:
        # eval mode real
        print(f"XXX EVAL_MODE_REAL")
        vocab_size = 2048
        dim = 256
        n_layers = 4
        config = DeepSeekV3ModelArgs(
            vocab_size=2048,
            max_seq_len=seq_len,
            dim=256,
            inter_dim=1024,
            moe_inter_dim=256,
            n_layers=4,
            n_dense_layers=0,  # 1,
            n_heads=16,
            moe_args=MoEArgs(
                num_experts=4,
                num_shared_experts=2,
                top_k=2,
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
        )
        def _model_llama3():
            model_args = TransformerModelArgs(
                dim=dim,
                n_layers=n_layers,
                n_heads=32,
                n_kv_heads=8,
                ffn_dim_multiplier=1.3,
                multiple_of=1024,
                rope_theta=500000,
                vocab_size=vocab_size,
                max_seq_len=seq_len,
            )
            m = Transformer(model_args)
            return m

    with torch.device("meta"):
        # model = DeepSeekV3Model(config).bfloat16()
        model = _model_llama3().bfloat16()
        embed, layers, norm, output = list(model.children())
        items = list(layers.items())
        print(f"XXX CONFIG.N_LAYERS:{config.n_layers}")
        print(f"XXX len(items):{len(items)}")
        assert len(items) == config.n_layers
        n_layers_per_rank = len(items) // total_pp_stages
        layers = [
            nn.ModuleDict(items[i : i + n_layers_per_rank])
            for i in range(0, len(items), n_layers_per_rank)
        ]
        assert len(layers) == total_pp_stages
        for lst in layers:
            assert len(lst) * len(layers) == config.n_layers

    def tracing_input_fn():
        print(f"XXX TRACING_INPUT_FN SEQ_LEN:{seq_len}")
        print(f"XXX TRACING_INPUT_FN VOCAB_SIZE:{config.vocab_size}")
        ret = torch.randint(
            0,
            config.vocab_size,
            (spmd_batch_size, seq_len),
            device=device,
        )
        print(f"XXX TRACING_INPUT_FN:{ret.shape}")
        return ret

    def tracing_input_fn_after_first_stage():
        return torch.randn(
            (spmd_batch_size, seq_len, config.dim),
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )

    def runtime_input_fn():
        return torch.randint(
            0,
            config.vocab_size,
            (local_batch_size, seq_len),
            device=device,
        )

    def shape_inference_input_fn():
        return torch.randint(
            0,
            config.vocab_size,
            (microbatch_size, seq_len),
            device="meta",
        )

    def shape_inference_input_fn_after_first_stage():
        return torch.randn(
            (microbatch_size, seq_len, config.dim),
            device="meta",
            dtype=torch.bfloat16,
            requires_grad=True,
        )

    def shape_inference_output_fn_last_stage():
        return torch.randn(
            (microbatch_size, seq_len, config.vocab_size),
            device="meta",
            dtype=torch.bfloat16,
            requires_grad=True,
        )

    # Apply SimpleFSDP to the full model
    # model = data_parallel(
    #     model,
    #     mesh[("dp_mod_ep")],
    # )


    # Step 1. Construct the logical pipeline stages
    fake_mode = FakeTensorMode(allow_non_fake_inputs = True, static_shapes=True)
    fake_mode.shape_env = None # ShapeEnv()
    fake_mode.static_shapes = False

    # with fake_mode:
    with torch.device("meta"):
        virtual_pp_stages = [DeepSeekV3Stage0(embed, layers[0], config)]
        for i in range(1, total_pp_stages - 1):
            virtual_pp_stages.append(DeepSeekV3StageI(layers[i], config))
        virtual_pp_stages.append(
            DeepSeekV3StageN(layers[total_pp_stages - 1], norm, output, config)
        )
    # Step 2. Assign each logical stage(s) to pp ranks for Interleaved1F1B schedule
    pp_rank_to_stage_indices: dict[int, list[int]] = {
        rank: [rank + i * pp_degree for i in range(stages_per_rank)]
        for rank in range(pp_degree)
    }
    assert len(pp_rank_to_stage_indices) == pp_degree
    for stages in pp_rank_to_stage_indices.values():
        assert len(stages) * pp_degree == len(virtual_pp_stages)
    stage_indices_current_pp_rank = pp_rank_to_stage_indices[pp_rank]
    stage_mods: dict[int, torch.nn.Module] = {}
    stage_graphs: dict[int, GraphCallables] = {}
    stage_graph_metas: dict[int, GraphMeta] = {}
    # Step 3. Apply AutoParallel to each logical stage assigned to this pp rank
    use_cache = fake_evaluate
    root_cache = "tmp"
    os.makedirs(root_cache, exist_ok=True)
    from autoparallel.api import AutoParallelPPModule

    simple_fsdp_world_mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda",
        (2, 4),
        mesh_dim_names=("dp", "tp"),
    )
    simple_fsdp_dp_mesh = simple_fsdp_world_mesh["dp"]
    simple_fsdp_tp_mesh = simple_fsdp_world_mesh["tp"]
    for stage_idx in stage_indices_current_pp_rank:
        r = dist.get_rank()
        print(f"XXX[{r}] STAGE_BEGIN {stage_idx}/{len(stage_indices_current_pp_rank)}")
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": f"begin_tracing_stage_{stage_idx}",
                "encoding": "string",
            },
            payload_fn=lambda: "placeholder text",
        )
        stage_mod = virtual_pp_stages[stage_idx]
        eval_mode = "fake" if fake_evaluate else "real"
        if stage_idx == 0:
            input_fn = tracing_input_fn
        else:
            input_fn = tracing_input_fn_after_first_stage
        with fake_mode:
            inputs = input_fn()
            if not isinstance(inputs, tuple):
                inputs = (inputs,)
        init_weights_model = move_to_fake(copy.deepcopy(stage_mod), fake_mode, device)
        for n, p in init_weights_model.named_parameters():
            print(f"XXX MTF[{r}] PARAM[{n}] -> {p.device} {p.shape}")

        with fake_mode:
            stage_mod = move_to_fake(stage_mod, fake_mode, device)
            # stage_mod(*inputs)

            # APPLY TP
            # apply_tp(
            #     stage_mod,
            #     simple_fsdp_tp_mesh,
            #     loss_parallel=False,
            #     enable_float8_tensorwise_tp=False,
            # )
            # -------------------------

            # APPLY SIMPLE FSDP
            simple_fsdp_mode = "fully_shard"
            # simple_fsdp_param_sharding = None
            # shard_dim = 0
            # if simple_fsdp_mode == "replicate":
            #     simple_fsdp_param_sharding = (Replicate(),)
            # elif simple_fsdp_mode == "fully_shard":
            #     simple_fsdp_param_sharding = (Shard(shard_dim),)
            # elif simple_fsdp_mode == "hybrid_shard":
            #     # replicate inter-host, fully shard intra-host
            #     simple_fsdp_param_sharding = (Replicate(), Shard(shard_dim))
            #     assert (
            #         device_mesh.ndim == 2
            #     ), "hybrid sharded data parallel requires 2D DeviceMesh"
            # else:
            #     raise ValueError(f"Unsupported mode {mode}")
            stage_mod = simple_fsdp_data_parallel(stage_mod, simple_fsdp_dp_mesh, mode=simple_fsdp_mode)
            # for n, p in stage_mod.named_parameters():
            #     print(f"XXX[{r}] STAGE_MOD_POST_SFSDP.P[{n}] -> {type(p)} {p.shape} {p.device}")
            # -------------------------

        def parallelize_inputs(mesh, args):
            def to_dtensor(t):
                if isinstance(t, torch.Tensor):
                    return DTensor.from_local(t, mesh, [Replicate()])
                return t
            return tree_map(to_dtensor, args)
        # inputs = parallelize_inputs(simple_fsdp_mesh, inputs)

        device = torch.device(f"cuda:{torch.cuda.current_device()}")

        print(f"XXX[{r}] device:{device}")
        for n, b in stage_mod.named_buffers():
            print(f"XXX[{r}] PRE_TRACE STAGE_MOD.B[{n}] -> {type(b)} {b.device}")
        #
        #
        #
        for inp in inputs:
            print(f"XXX D_INPUT {inp.shape} {inp.dtype} {type(inp)}")
        # VERIFY RUN OF THE STAGE
        # with torch.distributed.tensor.experimental.implicit_replication():

        # torch._subclasses.fake_tensor._log_fake_tensor_stack = True
        #out = stage_mod(*inputs)
        #print(f"XXX out:{out}")
        #out.sum().backward()

        # AUTOP STYLE GRAPH CAPTURE
        # stack = ExitStack()
        # with enable_local_map_wrapping(), torch._dynamo.utils._disable_saved_tensors_hooks_during_tracing():
        #     torch_ir_with_fqn = _export(stage_mod, inputs)
        #     print(f"XXX TORCH_IR_WITH_FQN:{torch_ir_with_fqn}")
        #     print("XXX TORCH_IR_WITH_FQN END")
        #     joint_with_descriptors = aot_export_joint_with_descriptors(
        #         stack,
        #         torch_ir_with_fqn,
        #         inputs,
        #         decompositions=torch._inductor.decomposition.select_decomp_table(),
        #     )
        # stack.close()
        # gm = joint_with_descriptors.graph_module
        # print(f"XXX[{r}] GM.joint:{gm.print_readable(False)}")


        # COMPILER TOOLKIT GRAPH CAPTURE
        from torchtitan.experiments.compiler_toolkit.graph_utils import joint_graph_builder, export_joint
        num_params = len(list(stage_mod.parameters()))
        num_buffers = len(list(stage_mod.buffers()))
        print(f"XXX NUM_PARAMS:{num_params}")
        print(f"XXX NUM_BUFFERS:{num_buffers}")
        jd, tracing_context = export_joint(stage_mod, inputs)
        gm = jd.graph_module
        print(f"XXX[{r}] STAGE_END:{stage_idx}/{len(stage_indices_current_pp_rank)} GM.joint:{gm.print_readable(False)}")
        # DO NOT RESHARD AFTER FORWARD
        reshard_after_forward = False
        ac_stage_size_in_GiB = "auto"
        from autoparallel.activation_checkpointing import ac_joint_pass
        ac_joint_pass(gm.graph, ac_stage_size_in_GiB, reshard_after_forward)

        from autoparallel._passes.graph_partition import partition_joint_with_descriptors
        (
            fw_module,
            bw_module,
            num_params_buffers,
            num_user_outputs,
            num_mutate_inputs,
            num_fw_outs_saved_for_bw,
            num_symints_saved_for_bw,
            _indices_of_inps_to_detach,
            adjusted_flat_args,
        ) = partition_joint_with_descriptors(jd)
        assert num_params_buffers == (
            num_params + num_buffers
        ), f"num_params_buffers: {num_params_buffers}, num_params: {num_params}, num_buffers: {num_buffers}"
        print(
            f"num_params_buffers: {num_params_buffers}\n"
            f"num_user_outputs: {num_user_outputs}\n"
            f"num_mutate_inputs: {num_mutate_inputs}\n"
            f"num_fw_outs_saved_for_bw: {num_fw_outs_saved_for_bw}\n"
            f"num_symints_saved_for_bw: {num_symints_saved_for_bw}"
        )

        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "autoparallel_pp_fwd_graph",
                "encoding": "string",
            },
            payload_fn=lambda: fw_module.print_readable(
                print_output=False, include_stride=True, include_device=True
            ),
        )
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "autoparallel_pp_bwd_graph",
                "encoding": "string",
            },
            payload_fn=lambda: bw_module.print_readable(
                print_output=False, include_stride=True, include_device=True
            ),
        )

        unshard_module: Optional[torch.fx.GraphModule] = None
        reduce_grad_module: Optional[torch.fx.GraphModule] = None
        graph_passes = ["split_fsdp_collectives", "split_dI_dW"]
        if "split_fsdp_collectives" in graph_passes:
            assert (
                not reshard_after_forward
            ), "reshard_after_forward should be False to disable FSDP all_gather in the backward pass"
            from autoparallel._passes.split_fsdp_collectives import (
                split_fsdp_prefetch,
                split_fsdp_reduce_scatters_epilogue,
            )

            unshard_module, fw_module = split_fsdp_prefetch(fw_module, num_params)
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "autoparallel_pp_unshard_graph",
                    "encoding": "string",
                },
                payload_fn=lambda: unshard_module.print_readable(
                    print_output=False, include_stride=True, include_device=True
                ),
            )
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "autoparallel_pp_fwd_no_fsdp_graph",
                    "encoding": "string",
                },
                payload_fn=lambda: fw_module.print_readable(
                    print_output=False, include_stride=True, include_device=True
                ),
            )
            bw_module, reduce_grad_module = split_fsdp_reduce_scatters_epilogue(
                bw_module, num_params
            )
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "autoparallel_pp_bwd_no_fsdp_graph",
                    "encoding": "string",
                },
                payload_fn=lambda: bw_module.print_readable(
                    print_output=False, include_stride=True, include_device=True
                ),
            )
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "autoparallel_pp_reduce_grad_graph",
                    "encoding": "string",
                },
                payload_fn=lambda: reduce_grad_module.print_readable(
                    print_output=False, include_stride=True, include_device=True
                ),
            )

        bw_dI_module: Optional[torch.fx.GraphModule] = None
        bw_dW_module: Optional[torch.fx.GraphModule] = None
        num_input_grads = 0
        if "split_dI_dW" in graph_passes:
            from autoparallel._passes.split_di_dw_graph import split_di_dw_graph

            bw_dI_module, bw_dW_module, num_input_grads = split_di_dw_graph(
                bw_module,
                num_weight_gradients=num_params_buffers,
            )
            print(f"XXX[{r} NUM_INPUT_GRADS={num_input_grads}")
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "autoparallel_pp_bw_dI_graph",
                    "encoding": "string",
                },
                payload_fn=lambda: bw_dI_module.print_readable(
                    print_output=False, include_stride=True, include_device=True
                ),
            )
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "autoparallel_pp_bw_dW_graph",
                    "encoding": "string",
                },
                payload_fn=lambda: bw_dW_module.print_readable(
                    print_output=False, include_stride=True, include_device=True
                ),
            )
            # TODO:
            # if all(
            #     x is None
            #     for x in bw_dI_module.graph.find_nodes(op="output")[0].args[0][
            #         :num_input_grads
            #     ]
            # ):
            #     raise RuntimeError(
            #         "attempted to run split dI/dW pass on a graph that has no input gradients"
            #     )

        graph_meta: dict[str, int] = {
            "num_mutate_inputs": num_mutate_inputs,
            "num_user_outputs": num_user_outputs,
            "num_symints_saved_for_bw": num_symints_saved_for_bw,
            "num_params": num_params,
            "num_buffers": num_buffers,
            "num_input_grads": num_input_grads,
        }

        graph_modules: dict[str, Optional[torch.fx.GraphModule]] = {
            "fw": fw_module,
            "full_bw": bw_module,
            "bw_dI": bw_dI_module,
            "bw_dW": bw_dW_module,
            "unshard": unshard_module,
            "reduce_grad": reduce_grad_module,
        }

        # TODO:
        sharded_param_dict = {}
        sharded_buffer_dict = {}

        for fqn, tensor in stage_mod.named_parameters():
            ctx = unset_fake_temporarily
            with ctx():
                t = torch.randn(tensor.shape, dtype=tensor.dtype, device="meta")
                sharded_param_dict[fqn] = nn.Parameter(t)

        for fqn, tensor in stage_mod.named_buffers():
            ctx = unset_fake_temporarily
            with ctx():
                t = torch.randn(tensor.shape, dtype=tensor.dtype, device="meta")
                sharded_buffer_dict[fqn] = tensor

        cache = {
            "graph_callables": graph_modules,
            "graph_meta": graph_meta,
            "sharded_param_dict": sharded_param_dict,
            "sharded_buffer_dict": sharded_buffer_dict,
        }
        print(f"XXX[{r}] SPLITS OK!")
        graph_callables = cache["graph_callables"]
        graph_meta = cache["graph_meta"]
        # TODO:
        pp_mod = AutoParallelPPModule(
            cache["sharded_param_dict"],
            cache["sharded_buffer_dict"],
            init_weights_model,
        )

        torch.manual_seed(pp_rank)
        pp_mod.to_empty(device=device)
        # TODO:
        # pp_mod.init_weights(buffer_device=device)

        # Store each stage's information in stage_mods, stage_graphs, and stage_graph_metas
        stage_mods[stage_idx] = pp_mod
        stage_graphs[stage_idx] = GraphCallables(
            fw=graph_callables["fw"],
            full_bw=graph_callables["full_bw"],
            bw_dI=graph_callables["bw_dI"],
            bw_dW=graph_callables["bw_dW"],
            unshard=graph_callables["unshard"],
            reduce_grad=graph_callables["reduce_grad"],
        )
        stage_graph_metas[stage_idx] = GraphMeta(
            num_mutate_inputs=graph_meta["num_mutate_inputs"],
            num_user_outputs=graph_meta["num_user_outputs"],
            num_symints_saved_for_bw=graph_meta["num_symints_saved_for_bw"],
            num_params=graph_meta["num_params"],
            num_buffers=graph_meta["num_buffers"],
            num_input_grads=graph_meta["num_input_grads"],
        )
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": f"end_tracing_stage_{stage_idx}",
                "encoding": "string",
            },
            payload_fn=lambda: "placeholder text",
        )
# 
#     # Two stages per pp rank
#     assert (
#         len(stage_indices_current_pp_rank)
#         == len(stage_mods)
#         == len(stage_graphs)
#         == len(stage_graph_metas)
#     )
# 
#     # run weight init on our sharded DTensor params
# 
#     stages = []
#     # Step 4. Construct pipeline stages for this pp_rank using the stage modules, graphs and metadata
#     for pp_stage_idx, pp_stage_mod in stage_mods.items():
#         stage = GraphPipelineStage(
#             pp_stage_mod,
#             stage_graphs[pp_stage_idx],
#             stage_graph_metas[pp_stage_idx],
#             stage_index=pp_stage_idx,
#             num_stages=len(virtual_pp_stages),
#             device=device,
#             input_args=(
#                 shape_inference_input_fn()
#                 if pp_stage_idx == 0
#                 else shape_inference_input_fn_after_first_stage()
#             ),
#             output_args=(
#                 shape_inference_output_fn_last_stage()
#                 if pp_stage_idx == (len(virtual_pp_stages) - 1)
#                 else shape_inference_input_fn_after_first_stage()
#             ),
#             group=world_mesh.get_group("pp"),
#         )
#         stages.append(stage)
#     # Step 5. Construct the pipeline runner using the pipeline stages for this pp_rank
#     schedule = build_pipeline_schedule(
#         stages=stages,
#         loss_fn=None,
#         pipeline_parallel_schedule="Interleaved1F1B",
#         microbatch_size=microbatch_size,
#         local_batch_size=local_batch_size,
#         pipeline_parallel_degree=pp_degree,
#     )
#     assert isinstance(schedule, _PipelineScheduleRuntime)
#     # Step 6. Override the pipeline runner's action implementations
#     schedule.register_custom_function(FORWARD, stage_forward)
#     schedule.register_custom_function(FULL_BACKWARD, stage_full_backward)
# 
#     # Step 7. Register the schedule with the graph runner
# 
#     graph_pp_runner = GraphPPRunner(schedule)
# 
#     # Step 8. Run the whole pipeline once using the graph runner
#     with (
#         FakeTensorMode(
#             allow_non_fake_inputs=True,
#             shape_env=ShapeEnv(),
#         )
#         if fake_evaluate
#         else nullcontext()
#     ):
#         with torch.no_grad():
#             if pp_rank == 0:
#                 x = runtime_input_fn()
#                 graph_pp_runner.step(x)
#             else:
#                 graph_pp_runner.step()

    print("All good!")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.cuda.synchronize()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run DeepSeek V3 pipeline parallel example"
    )
    parser.add_argument(
        "--fake-evaluate",
        action="store_true",
        default=False,
        help="Use fake evaluation mode with FakeTensorMode (default: False)",
    )
    args = parser.parse_args()

    run_test(fake_evaluate=args.fake_evaluate)
