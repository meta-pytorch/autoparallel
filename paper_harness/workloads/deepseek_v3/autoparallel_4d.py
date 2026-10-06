"""AutoParallel arm for the DeepSeek V3 dp_replicate x dp_shard x tp (EP) comparison.

TorchTitan's ``parallelize_autoparallel_deepseekv3`` rejects dp_replicate. The
MainTrainer and manual GraphTrainer arms already build that mesh from
``data_parallel_replicate_degree``, so only AutoParallel needs a workload-local
parallelizer. It mirrors TorchTitan's DeepSeek V3 path on the aligned MoE mesh
and adds one constraint: every parameter is ``Replicate()`` on the replicate
axis, so the arm keeps the HSDP layout of the two manual arms.
"""

from __future__ import annotations

import time

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import device_type
from workloads.llama3_3d.autoparallel_4d import (
    REPLICATE_AXIS,
    _write_placement_audit,
)

DATA_PARALLEL_AXES = {"dp_replicate", "dp_shard_mod_ep", "dp_shard_in_ep"}


def parallelize_autoparallel_4d_deepseekv3(
    model,
    *,
    parallel_dims,
    training,
    parallelism,
    compile_config,
    ac_config,
    dump_folder,
):
    from torchtitan.experiments.graph_trainer.autoparallel_api import (
        AutoParallelGraph,
        AutoParallelModelOutput,
    )
    from torchtitan.experiments.graph_trainer.compile import apply_compile
    from torchtitan.experiments.graph_trainer.configs import (
        validate_autoparallel_config,
    )
    from torchtitan.experiments.graph_trainer.deepseek_v3.parallelize_autoparallel import (
        _load_autoparallel_dsv3_dependency,
        _preserve_moe_attributes,
        _set_torchtitan_fields,
    )

    from autoparallel import ForwardInputs

    del ac_config
    validate_autoparallel_config(compile_config)
    for name in ("pp", "cp"):
        if getattr(parallel_dims, f"{name}_enabled"):
            raise ValueError(f"DeepSeek V3 4D AutoParallel does not support {name}")
    if not parallel_dims.dp_replicate_enabled:
        raise ValueError("DeepSeek V3 4D AutoParallel requires dp_replicate")

    param_dtype = TORCH_DTYPE_MAP[training.mixed_precision_param]
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        cast_forward_inputs=False,
    )
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        parallelism.fsdp_reshard_after_forward,
        parallel_dims.pp_enabled,
    )
    (
        APDeepSeekV3Model,
        annotate_deepseekv3_for_graph_trainer,
        build_moe_mesh,
    ) = _load_autoparallel_dsv3_dependency()

    # dp_replicate, dp_shard_mod_ep, dp_shard_in_ep, tp in row-major order: the
    # same rank coordinates as TorchTitan's dp_replicate/fsdp/tp world mesh.
    ap_mesh, moe_roles = build_moe_mesh(
        dp_replicate=parallel_dims.dp_replicate,
        dp_shard=parallel_dims.dp_shard,
        cp=parallel_dims.cp,
        tp=parallel_dims.tp,
        ep=parallel_dims.ep,
        device_type=device_type,
    )
    with torch.device("meta"):
        ap_model = APDeepSeekV3Model(
            model.config,
            mesh=ap_mesh,
            roles=moe_roles,
            compute_dtype=param_dtype,
        )

    def input_fn():
        global_batch_size = training.global_batch_size
        if global_batch_size < 0:
            dp_degree = parallel_dims.dp_replicate * parallel_dims.dp_shard
            global_batch_size = training.local_batch_size * dp_degree
        tokens = torch.randint(
            0,
            ap_model.model_args.vocab_size,
            (global_batch_size, training.seq_len),
            device=torch.device(device_type),
        )
        positions = torch.arange(
            training.seq_len,
            dtype=torch.int64,
            device=torch.device(device_type),
        ).repeat(global_batch_size, 1)
        return ForwardInputs(args=(tokens,), kwargs={"positions": positions})

    x_sharding = tuple(
        Shard(0) if name in DATA_PARALLEL_AXES else Replicate()
        for name in ap_mesh.mesh_dim_names
    )
    output_sharding = tuple(
        Shard(0)
        if name in DATA_PARALLEL_AXES
        else Shard(2)
        if name == "tp"
        else Replicate()
        for name in ap_mesh.mesh_dim_names
    )

    autop = AutoParallelGraph(
        ap_model,
        input_fn,
        ap_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
        solver=compile_config.autoparallel_solver,
    )
    annotate_deepseekv3_for_graph_trainer(autop.model)

    trace_start = time.perf_counter()
    with autop:
        trace_seconds = time.perf_counter() - trace_start
        autop.add_parameter_memory_constraint(low=None, high=None)
        autop.add_input_constraints([x_sharding, x_sharding])
        autop.add_output_constraints([output_sharding])
        autop.add_parameter_axis_constraint(REPLICATE_AXIS, Replicate())

        solve_start = time.perf_counter()
        sharding_placement = autop.optimize_placement()
        solve_seconds = time.perf_counter() - solve_start
        logger.info("AutoParallelGraph took %.2f seconds", solve_seconds)

        _write_placement_audit(
            autop,
            sharding_placement,
            mesh=ap_mesh,
            solve_seconds=solve_seconds,
            trace_seconds=trace_seconds,
        )

        model_output = (
            AutoParallelModelOutput(
                output_mesh=parallel_dims.get_mesh("tp"),
                output_placements=(Shard(2),),
                sharded_output_axis=2,
            )
            if parallel_dims.tp_enabled
            else None
        )
        parallel_mod = autop.apply_placement_for_fx_module(
            sharding_placement,
            compile_config=compile_config,
            model_output=model_output,
        )

    _set_torchtitan_fields(parallel_mod)
    _preserve_moe_attributes(ap_model, parallel_mod)

    return apply_compile(
        parallel_mod,
        compile_config=compile_config,
        parallelism=parallelism,
        parallel_dims=parallel_dims,
        dump_folder=dump_folder,
    )
