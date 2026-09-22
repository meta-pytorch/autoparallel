from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from autoparallel import ForwardInputs
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.experiments.graph_trainer.autoparallel_api import (
    AutoParallelGraph,
    AutoParallelModelOutput,
)
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.configs import validate_autoparallel_config
from torchtitan.experiments.graph_trainer.muse_glimmer.sdpa import (
    build_packed_document_attention_masks,
)
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import device_type

MESH_AXIS_NAMES = ("dp_replicate", "fsdp", "tp")
REPLICATE_AXIS = "dp_replicate"


def _write_placement_audit(autop, sharding_placement, *, mesh) -> None:
    from torch._functorch._aot_autograd.fx_utils import get_param_nodes

    replicate_axis = mesh.mesh_dim_names.index(REPLICATE_AXIS)
    parameters = {}
    violations = []
    for node in get_param_nodes(autop.gm.graph):
        spec = sharding_placement[node].output_specs
        placements = tuple(spec.placements)
        parameters[node.name] = {
            "shape": list(spec.tensor_meta.shape),
            "placements": [str(placement) for placement in placements],
        }
        if not placements[replicate_axis].is_replicate():
            violations.append(node.name)

    report = {
        "rank": int(os.environ["RANK"]),
        "mesh_shape": [int(value) for value in mesh.shape],
        "mesh_dim_names": list(mesh.mesh_dim_names),
        "replicate_axis": replicate_axis,
        "parameter_count": len(parameters),
        "parameter_placements": parameters,
        "replicate_axis_violations": violations,
    }
    audit_dir = Path(os.environ["PLACEMENT_AUDIT_DIR"])
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / f"rank_{int(os.environ['RANK']):03d}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    if violations:
        raise RuntimeError(
            f"{len(violations)} parameters are not replicated on "
            f"{REPLICATE_AXIS!r}: {violations[:8]}"
        )


def parallelize_autoparallel_hsdp_muse_glimmer(
    model,
    *,
    parallel_dims,
    training,
    parallelism,
    compile_config,
    ac_config,
    dump_folder,
):
    del ac_config
    validate_autoparallel_config(compile_config)
    if parallel_dims.cp_enabled or parallel_dims.pp_enabled:
        raise ValueError("Muse Glimmer HSDP AutoParallel does not support CP or PP")
    if not (
        parallel_dims.dp_replicate_enabled
        and parallel_dims.dp_shard_enabled
        and parallel_dims.tp_enabled
    ):
        raise ValueError(
            "Muse Glimmer HSDP AutoParallel requires dp_replicate, dp_shard, and tp"
        )

    dense_mesh = parallel_dims.get_mesh(list(MESH_AXIS_NAMES))
    if tuple(dense_mesh.mesh_dim_names) != MESH_AXIS_NAMES:
        raise ValueError(
            f"Unexpected AutoParallel mesh axes: {dense_mesh.mesh_dim_names}"
        )
    window_sizes = {
        layer.attention.window_size
        for layer in model.config.layers
        if layer.attention.window_size is not None
    }

    def input_fn():
        global_batch_size = training.global_batch_size
        if global_batch_size < 0:
            dp_degree = parallel_dims.dp_replicate * parallel_dims.dp_shard
            global_batch_size = training.local_batch_size * dp_degree
        tokens = torch.randint(
            0,
            model.config.vocab_size,
            (global_batch_size, training.seq_len),
            device=torch.device(device_type),
        )
        positions = torch.arange(
            training.seq_len,
            dtype=torch.int64,
            device=torch.device(device_type),
        ).repeat(global_batch_size, 1)
        attention_masks = build_packed_document_attention_masks(
            positions,
            window_sizes,
        )
        return ForwardInputs(
            args=(tokens,),
            kwargs={"positions": positions, "attention_masks": attention_masks},
        )

    mp_policy = MixedPrecisionPolicy(
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        cast_forward_inputs=False,
    )
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        parallelism.fsdp_reshard_after_forward,
        parallel_dims.pp_enabled,
    )
    input_sharding = (Shard(0), Shard(0), Replicate())
    output_sharding = (Shard(0), Shard(0), Shard(2))

    with AutoParallelGraph(
        model,
        input_fn,
        dense_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
        repeated_subgraphs=True,
        solver=compile_config.autoparallel_solver,
    ) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)
        autop.add_parameter_axis_constraint(REPLICATE_AXIS, Replicate())
        autop.add_input_constraints([input_sharding] * (2 + 1 + len(window_sizes)))
        autop.add_output_constraints([output_sharding])

        started = time.perf_counter()
        sharding_placement = autop.optimize_placement(verbose=False)
        logger.info(
            "AutoParallelGraph placement search took %.2f seconds",
            time.perf_counter() - started,
        )
        _write_placement_audit(autop, sharding_placement, mesh=dense_mesh)
        parallel_model = autop.apply_placement_for_fx_module(
            sharding_placement,
            compile_config=compile_config,
            model_output=AutoParallelModelOutput(
                output_mesh=parallel_dims.get_mesh("tp"),
                output_placements=(Shard(2),),
                sharded_output_axis=2,
            ),
        )

    return apply_compile(
        parallel_model,
        compile_config=compile_config,
        parallelism=parallelism,
        parallel_dims=parallel_dims,
        dump_folder=dump_folder,
    )
