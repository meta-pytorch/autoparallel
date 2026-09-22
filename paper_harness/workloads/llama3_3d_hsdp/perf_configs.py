"""AutoParallel arm for the LLaMA 3 8B dp_replicate x fsdp x tp comparison.

The MainTrainer and manual GraphTrainer arms of this campaign use
``workloads.llama3_2d.perf_configs`` unchanged; both already build an HSDP mesh
from ``BENCHMARK_DP_REPLICATE_DEGREE``. Only AutoParallel needs a workload-local
parallelizer, because it must be told that the replicate axis is a replicate
axis: without that constraint the solver is free to shard parameters across it
and the arm would no longer be comparable to the two manual arms.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import device_type
from workloads.llama3_2d.perf_configs import autoparallel_graphtrainer_8b

MESH_AXIS_NAMES = ("dp_replicate", "fsdp", "tp")
REPLICATE_AXIS = "dp_replicate"


def _dense_mesh(parallel_dims):
    """The dense AutoParallel mesh, matching TorchTitan's non-CP axis choice."""
    mesh = parallel_dims.get_mesh(list(MESH_AXIS_NAMES))
    if tuple(mesh.mesh_dim_names) != MESH_AXIS_NAMES:
        raise ValueError(f"Unexpected AutoParallel mesh axes: {mesh.mesh_dim_names}")
    return mesh


def _write_placement_audit(
    autop, sharding_placement, *, mesh, solve_seconds, trace_seconds
) -> None:
    """Record every parameter placement and fail if the axis rule did not hold.

    This is the acceptance evidence that AutoParallel actually produced the HSDP
    parameter layout, rather than a plan that happens to fit the memory budget.
    """
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

    audit = {
        "rank": int(os.environ["RANK"]),
        "mesh_shape": [int(value) for value in mesh.shape],
        "mesh_dim_names": list(mesh.mesh_dim_names),
        "replicate_axis": replicate_axis,
        "trace_seconds": trace_seconds,
        "solve_seconds": solve_seconds,
        "parameter_count": len(parameters),
        "parameter_placements": parameters,
        "replicate_axis_violations": violations,
    }
    audit_dir = Path(os.environ["PLACEMENT_AUDIT_DIR"])
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / f"rank_{int(os.environ['RANK']):03d}.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    if violations:
        raise RuntimeError(
            f"{len(violations)} parameters are not replicated on the "
            f"{REPLICATE_AXIS!r} mesh axis: {violations[:8]}"
        )


def parallelize_autoparallel_hsdp_llama(
    model,
    *,
    parallel_dims,
    training,
    parallelism,
    compile_config,
    ac_config,
    dump_folder,
):
    """AutoParallel on a dp_replicate x fsdp x tp mesh with HSDP parameters.

    Mirrors TorchTitan's ``parallelize_autoparallel_llama`` non-CP path and adds
    one constraint: every parameter is ``Replicate()`` on the replicate axis.
    The fsdp and tp placements stay free, so the solver still chooses them under
    the same parameter memory budget the other AutoParallel arms use.
    """
    from torchtitan.experiments.graph_trainer.autoparallel_api import (
        AutoParallelGraph,
        AutoParallelModelOutput,
    )
    from torchtitan.experiments.graph_trainer.compile import apply_compile
    from torchtitan.experiments.graph_trainer.configs import (
        validate_autoparallel_config,
    )

    from autoparallel import ForwardInputs

    del ac_config
    validate_autoparallel_config(compile_config)
    for name in ("cp", "pp", "ep"):
        if getattr(parallel_dims, f"{name}_enabled"):
            raise ValueError(f"LLaMA 3 HSDP AutoParallel does not support {name}")
    if not (
        parallel_dims.dp_replicate_enabled
        and parallel_dims.dp_shard_enabled
        and parallel_dims.tp_enabled
    ):
        raise ValueError(
            "LLaMA 3 HSDP AutoParallel requires dp_replicate, dp_shard, and tp"
        )

    dense_mesh = _dense_mesh(parallel_dims)
    vocab_size = model.config.vocab_size

    def input_fn():
        global_batch_size = training.global_batch_size
        if global_batch_size < 0:
            dp_degree = parallel_dims.dp_replicate * parallel_dims.dp_shard
            global_batch_size = training.local_batch_size * dp_degree
        tokens = torch.randint(
            0,
            vocab_size,
            (global_batch_size, training.seq_len),
            device=torch.device(device_type),
        )
        positions = torch.arange(
            training.seq_len,
            dtype=torch.int64,
            device=torch.device(device_type),
        ).repeat(global_batch_size, 1)
        return ForwardInputs(args=(tokens,), kwargs={"positions": positions})

    mp_policy = MixedPrecisionPolicy(
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        cast_forward_inputs=False,
    )
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        parallelism.fsdp_reshard_after_forward,
        parallel_dims.pp_enabled,
    )
    input_sharding_by_axis = {
        "dp_replicate": Shard(0),
        "fsdp": Shard(0),
        "tp": Replicate(),
    }
    x_sharding = tuple(
        input_sharding_by_axis[name] for name in dense_mesh.mesh_dim_names
    )
    output_sharding = tuple(
        Shard(2) if name == "tp" else Shard(0) for name in dense_mesh.mesh_dim_names
    )

    trace_start = time.perf_counter()
    with AutoParallelGraph(
        model,
        input_fn,
        dense_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
        solver=compile_config.autoparallel_solver,
        # Matches the fresh-solve radius TorchTitan uses for a 3D AutoParallel
        # mesh; the campaign gate proves the constrained problem is feasible
        # within it for this model.
        strategy_radius=2,
    ) as autop:
        trace_seconds = time.perf_counter() - trace_start
        autop.add_parameter_memory_constraint(low=None, high=None)
        autop.add_input_constraints([x_sharding, x_sharding])
        autop.add_output_constraints([output_sharding])
        autop.add_parameter_axis_constraint(REPLICATE_AXIS, Replicate())

        solve_start = time.perf_counter()
        sharding_placement = autop.optimize_placement(verbose=False)
        solve_seconds = time.perf_counter() - solve_start
        logger.info("AutoParallelGraph took %.2f seconds", solve_seconds)

        _write_placement_audit(
            autop,
            sharding_placement,
            mesh=dense_mesh,
            solve_seconds=solve_seconds,
            trace_seconds=trace_seconds,
        )

        parallel_mod = autop.apply_placement_for_fx_module(
            sharding_placement,
            compile_config=compile_config,
            model_output=AutoParallelModelOutput(
                output_mesh=parallel_dims.get_mesh("tp"),
                output_placements=(Shard(2),),
                sharded_output_axis=2,
            ),
        )

    return apply_compile(
        parallel_mod,
        compile_config=compile_config,
        parallelism=parallelism,
        parallel_dims=parallel_dims,
        dump_folder=dump_folder,
    )


def autoparallel_graphtrainer_hsdp_8b():
    # Reuse the 2D AutoParallel arm so the two configurations stay byte-identical
    # apart from the model spec, which the campaign declares as the variable.
    config = autoparallel_graphtrainer_8b()
    config.model_spec = replace(
        config.model_spec,
        name="autoparallel_graphtrainer/hsdp_3d/llama3",
        parallelize_fn=parallelize_autoparallel_hsdp_llama,
    )
    return config


EXPERIMENT_CONFIGS = ("autoparallel_graphtrainer_hsdp_8b",)
