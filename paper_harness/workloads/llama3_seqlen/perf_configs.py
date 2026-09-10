from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import torch
from datasets import Features, Value, load_dataset
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard

from torchtitan.config import CompileConfig, TORCH_DTYPE_MAP
from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import default_adamw
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.llama3 import model_registry
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.hf_datasets.text_datasets import (
    DATASETS,
    HuggingFaceTextDataLoader,
    HuggingFaceTextDataset,
)
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.llama3.config_registry import llama3_8b
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import device_type

from workloads.parameter_state import register_post_load_parameter_audit


C4_REPO = "allenai/c4"
C4_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
C4_TRAIN_SHARDS = 1024
C4_SHARD_METADATA = {
    0: (
        319308785,
        "8ef8d75b0e045dec4aa5123a671b4564466b0707086a7ed1ba8721626dfffbc9",
    ),
    1: (
        318039285,
        "b945059cd1a343cabe311881b7840a6f0363f570e745a0eff0e687e266f6b55d",
    ),
    2: (
        319748667,
        "2967dc7e587ced6ecb9ba617ad2d4c44901467969de5bf5b0f5a9e5b70555d75",
    ),
    3: (
        318564193,
        "b79d9abef5741578929be0d59db9ca652a8276207ef18a944b7a5f11fef5beb6",
    ),
    4: (
        318579884,
        "cd9f98eac2bc6062f55d9a36bd744cc924a78ea2fd998830e0034e4456f5d014",
    ),
    5: (
        318003681,
        "8ac5907a54dbc7ab9c14624448c7c3f6afed33af9d0a855f1eae955e62e255b9",
    ),
    6: (
        318495137,
        "8fd9b9a4b74c9414466b245ebda7db041e7bd8603971de51b5db782bd758aac7",
    ),
    7: (
        318417273,
        "41dd377a1ba6b72eab0260c39c626fe45ab6b649d42d57b311d3ba21a0337cd0",
    ),
    8: (
        318131845,
        "64da652c235f089a0b52f6db5883ef5f1e9c31edc4c950332b34dd12439c99a5",
    ),
    9: (
        318185592,
        "807a548efbb10153c9eff0df5733a97a1b51ab1743242530de1b02a8ea17ace7",
    ),
    10: (
        319045292,
        "3bd0f6f664069c3bd964ce48ceae60ba47b55b54745a4b00c207bdb3a1926b17",
    ),
    11: (
        319686980,
        "5baa0c010083459ba58e34b4e93bb758caa878f7db6fba0528921329fa1a6cc5",
    ),
    12: (
        320119088,
        "fdee7442c06856e2c4b7665cc51978e9011b5e0a2112c30dd15bc9e53818842d",
    ),
    13: (
        319474856,
        "a4ab3b24087781c3577945492525696e182ffd7ca5265b958f49803a02867ecf",
    ),
    14: (
        319693210,
        "62215b2451e71b117018ef73570c944aff890624b384c538950b64c37f184c49",
    ),
    15: (
        318427305,
        "9893c9f413a1223e7b535527829bcd6df3219929fb1abf8f2a114dd8f6ea0919",
    ),
}
DATASET_NAME = "c4_llama8b_pinned_audited"

MODEL_FLAVOR = os.environ.get("BENCHMARK_MODEL", "8B")
if MODEL_FLAVOR != "8B":
    raise ValueError(f"Unsupported BENCHMARK_MODEL={MODEL_FLAVOR!r}")

CANONICAL_SEQUENCE_LENGTH = 2048
SUPPORTED_SEQUENCE_LENGTHS = (2048, 4096, 8192, 16384, 32768)

WORLD_SIZE = int(os.environ["BENCHMARK_WORLD_SIZE"])
TP_DEGREE = int(os.environ["BENCHMARK_TP_DEGREE"])
if (WORLD_SIZE, TP_DEGREE) != (32, 8):
    raise ValueError(f"Unsupported mesh: {WORLD_SIZE=} {TP_DEGREE=}")
DP_DEGREE = WORLD_SIZE // TP_DEGREE

LOCAL_BATCH_SIZE = 2
SEQ_LEN = int(os.environ["BENCHMARK_SEQ_LEN"])
if SEQ_LEN not in SUPPORTED_SEQUENCE_LENGTHS:
    raise ValueError(f"Unsupported sequence length: {SEQ_LEN}")
GLOBAL_BATCH_SIZE = LOCAL_BATCH_SIZE * DP_DEGREE


def _write_inductor_path_audit() -> None:
    """Assert and record the configuration-specific Inductor hook state."""
    import torch._inductor.fx_passes.bucketing as bucketing_mod
    import torch._inductor.fx_passes.fsdp as fsdp_mod

    functions = {
        "greedy_bucket_collective_by_mb": (
            f"{bucketing_mod.greedy_bucket_collective_by_mb.__module__}."
            f"{bucketing_mod.greedy_bucket_collective_by_mb.__qualname__}"
        ),
        "identify_fsdp_groups": (
            f"{fsdp_mod.identify_fsdp_groups.__module__}."
            f"{fsdp_mod.identify_fsdp_groups.__qualname__}"
        ),
    }
    patch_active = all("_patch_fsdp_bucketing" in value for value in functions.values())
    custom_post_pass = torch._inductor.config.post_grad_custom_post_pass
    configuration = os.environ["BENCHMARK_CONFIGURATION"]
    expected_patch_active = configuration != "torchtitan_baseline"
    if patch_active != expected_patch_active:
        raise RuntimeError(
            "Unexpected AutoParallel bucketing hook state for "
            f"{configuration}: {functions}"
        )
    module_loaded = "autoparallel.graph_passes.auto_bucketing" in sys.modules
    if module_loaded != expected_patch_active:
        raise RuntimeError(
            f"Unexpected AutoParallel bucketing module state for {configuration}: "
            f"loaded={module_loaded}"
        )
    expected_process_custom_pass = (
        configuration == "autoparallel_backend_example_scheduling"
    )
    if (custom_post_pass is not None) != expected_process_custom_pass:
        raise RuntimeError(
            f"Unexpected custom post-grad scheduler for {configuration}: "
            f"present={custom_post_pass is not None}"
        )

    audit_dir = Path(os.environ["MODULE_ISOLATION_AUDIT_DIR"])
    audit_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "rank": int(os.environ["RANK"]),
        "configuration": configuration,
        "autoparallel_bucketing_hook_active": patch_active,
        "autoparallel_bucketing_module_loaded": module_loaded,
        "functions": functions,
        "post_grad_custom_post_pass_present": custom_post_pass is not None,
        "post_grad_custom_post_pass_callable": (
            None
            if custom_post_pass is None
            else f"{custom_post_pass.__module__}.{custom_post_pass.__qualname__}"
        ),
    }
    output = audit_dir / f"rank_{int(os.environ['RANK']):02d}.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


class AutoParallelBackendOutputAdapter(torch.nn.Module):
    """Restore the TP-sharded model output to the DTensor loss boundary."""

    def __init__(self, model, model_output):
        super().__init__()
        self.model = model
        self.model_output = model_output

    def forward(self, *args, **kwargs):
        from torchtitan.experiments.graph_trainer.autoparallel_api import (
            _wrap_autoparallel_output,
        )

        return _wrap_autoparallel_output(self.model(*args, **kwargs), self.model_output)

    def init_weights(self, *args, **kwargs):
        return self.model.init_weights(*args, **kwargs)


def _register_post_load_audits(optimizers, model_parts, parallel_dims):
    _write_inductor_path_audit()
    register_post_load_parameter_audit(optimizers, model_parts, parallel_dims)


def parallelize_autoparallel_backend_llama(
    model,
    *,
    parallel_dims,
    training,
    parallelism,
    compile_config,
    ac_config,
    dump_folder,
):
    """Use the standard AutoParallel apply_placement + backend path."""
    from autoparallel import AutoParallel, ForwardInputs
    from autoparallel.compile import autoparallel_backend
    from autoparallel.cost_models.collective_runtime_estimation import (
        set_nccl_topo_config,
    )
    from autoparallel.cost_models.nccl_cost_model import detect_nccl_topo_config
    from autoparallel.graph_passes.auto_bucketing import (
        aten_autobucketing_config,
        aten_autobucketing_reordering_pass,
    )
    from autoparallel.graph_passes.debug_helpers import (
        make_custom_runtime_estimation,
    )
    from autoparallel.graph_passes.estimate_graph_metrics import (
        estimate_graph_metrics,
    )
    from torchtitan.experiments.graph_trainer.autoparallel_api import (
        AutoParallelModelOutput,
    )

    del compile_config
    if parallel_dims.dp_replicate_enabled:
        raise ValueError("AutoParallel Llama3 does not support DDP yet")
    if parallel_dims.cp_enabled:
        raise ValueError("AutoParallel Llama3 does not support CP yet")
    if parallel_dims.pp_enabled:
        raise ValueError("AutoParallel Llama3 does not support PP yet")

    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(model)

    mesh_axis_names = [
        name
        for name in ("dp_replicate", "fsdp", "tp")
        if parallel_dims.get_optional_mesh(name) is not None
    ]
    mesh = parallel_dims.get_mesh(mesh_axis_names)

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
    if reshard_after_forward is not True:
        raise RuntimeError(
            "This experiment requires reshard_after_forward=True in every arm"
        )
    input_sharding_by_axis = {
        "dp_replicate": Shard(0),
        "fsdp": Shard(0),
        "tp": Replicate(),
    }
    input_sharding = tuple(input_sharding_by_axis[name] for name in mesh.mesh_dim_names)
    output_sharding = tuple(
        Shard(2) if name == "tp" else Shard(0) for name in mesh.mesh_dim_names
    )

    with AutoParallel(
        model,
        input_fn,
        mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
    ) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)
        autop.add_input_constraints([input_sharding, input_sharding])
        autop.add_output_constraints([output_sharding])
        start = time.time()
        placement = autop.optimize_placement(verbose=False)
        logger.info(
            "AutoParallel backend placement took %.2f seconds", time.time() - start
        )
        parallel_model = autop.apply_placement(placement)

    import autoparallel.api as autoparallel_api

    hook_expectation = os.environ["AP_COLLECTIVE_MUST_SAVE_EXPECTATION"]
    hook_present = hasattr(
        autoparallel_api, "_save_autoparallel_collectives_for_first_partition"
    )
    hook_audit = getattr(autoparallel_api, "_ap_collective_must_save_audit", None)
    if hook_expectation == "on":
        if not hook_present or not isinstance(hook_audit, dict):
            raise RuntimeError("Expected AP collective MUST_SAVE hook audit is absent")
        if hook_audit.get("collective_total", 0) <= 0:
            raise RuntimeError(f"AP collective MUST_SAVE hook matched nothing: {hook_audit}")
    elif hook_expectation == "off":
        if hook_present or hook_audit is not None:
            raise RuntimeError("AP collective MUST_SAVE hook is present in the off arm")
    else:
        raise RuntimeError(f"Unknown AP hook expectation: {hook_expectation!r}")
    hook_report = {
        "expectation": hook_expectation,
        "hook_present": hook_present,
        "hook_audit": hook_audit,
        "reshard_after_forward": reshard_after_forward,
        "rank": int(os.environ["RANK"]),
    }
    hook_audit_dir = Path(os.environ["AP_COLLECTIVE_HOOK_AUDIT_DIR"])
    hook_audit_dir.mkdir(parents=True, exist_ok=True)
    (hook_audit_dir / f"rank_{int(os.environ['RANK']):02d}.json").write_text(
        json.dumps(hook_report, indent=2, sort_keys=True) + "\n"
    )

    set_nccl_topo_config(detect_nccl_topo_config(mesh))
    custom_runtime_estimation = make_custom_runtime_estimation(mesh)
    autobucketing_config = aten_autobucketing_config()
    autobucketing_config.custom_runtime_estimation = custom_runtime_estimation
    autobucketing_config.save_trace = False
    autobucketing_pass = partial(
        aten_autobucketing_reordering_pass,
        configs=autobucketing_config,
    )

    def post_grad_pass(graph):
        new_gm = autobucketing_pass(graph)
        logger.info(
            "AutoParallel post-grad graph metrics: %s",
            estimate_graph_metrics(new_gm, custom_runtime_estimation),
        )
        return new_gm

    torch._inductor.config.reorder_for_peak_memory = False
    torch._inductor.config.reorder_for_compute_comm_overlap = False
    torch._inductor.config.post_grad_custom_post_pass = post_grad_pass

    parallel_model = torch.compile(
        parallel_model,
        backend=autoparallel_backend(
            enable_ac=False,
            overlap_scheduling=True,
        ),
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
    return AutoParallelBackendOutputAdapter(parallel_model, model_output)


def _placement_payload(solution) -> dict[str, dict[str, object]]:
    from torch.distributed.tensor._op_schema import _pretty_print_spec

    return {
        node.name: {
            "output": _pretty_print_spec(strategy.output_specs),
            "inputs": [_pretty_print_spec(spec) for spec in strategy.input_specs],
        }
        for node, strategy in solution.items()
    }


def _placement_digest(payload: dict[str, dict[str, object]]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def parallelize_autoparallel_graphtrainer_seqlen_llama(
    model,
    *,
    parallel_dims,
    training,
    parallelism,
    compile_config,
    ac_config,
    dump_folder,
):
    """Solve at the target sequence length or replay the canonical 2K plan."""
    from autoparallel import ForwardInputs
    from torchtitan.experiments.graph_trainer.autoparallel_api import (
        AutoParallelGraph,
        AutoParallelModelOutput,
    )
    from torchtitan.experiments.graph_trainer.compile import apply_compile
    from torchtitan.experiments.graph_trainer.configs import (
        validate_autoparallel_config,
    )

    del ac_config
    validate_autoparallel_config(compile_config)
    if parallel_dims.dp_replicate_enabled:
        raise ValueError("AutoParallel Llama3 does not support DDP")
    if parallel_dims.cp_enabled:
        raise ValueError("AutoParallel Llama3 does not support CP")
    if parallel_dims.pp_enabled:
        raise ValueError("AutoParallel Llama3 does not support PP")

    dense_names = [
        name
        for name in ("dp_replicate", "fsdp", "tp")
        if parallel_dims.get_optional_mesh(name) is not None
    ]
    dense_mesh = parallel_dims.get_mesh(dense_names)
    mesh_shape = tuple(int(value) for value in dense_mesh.shape)
    if mesh_shape != (DP_DEGREE, TP_DEGREE):
        raise ValueError(f"Unexpected AutoParallel mesh: {mesh_shape}")

    strategy_mode = os.environ["AP_STRATEGY_MODE"]
    if strategy_mode not in ("fresh", "replay_2k"):
        raise ValueError(f"Unsupported AP_STRATEGY_MODE={strategy_mode!r}")
    canonical_path = Path(os.environ["AP_CANONICAL_PLACEMENT"])
    expected_canonical_sha256 = os.environ.get("EXPECTED_CANONICAL_SHA256", "")
    canonical_file_sha256 = None
    canonical_payload = None
    if canonical_path.is_file():
        canonical_file_sha256 = hashlib.sha256(canonical_path.read_bytes()).hexdigest()
        if canonical_file_sha256 != expected_canonical_sha256:
            raise RuntimeError(
                "Canonical placement file hash mismatch: "
                f"{canonical_file_sha256} != {expected_canonical_sha256}"
            )
        canonical_manifest = json.loads(canonical_path.read_text())
        if canonical_manifest.get("version") != 1:
            raise RuntimeError(f"Unexpected placement version: {canonical_path}")
        if tuple(canonical_manifest.get("mesh_shape", ())) != mesh_shape:
            raise RuntimeError(f"Canonical placement mesh mismatch: {canonical_path}")
        if canonical_manifest.get("mesh_dim_names") != list(dense_mesh.mesh_dim_names):
            raise RuntimeError(f"Canonical placement axis mismatch: {canonical_path}")
        canonical_payload = canonical_manifest.get("placements")
        if not isinstance(canonical_payload, dict):
            raise RuntimeError(f"Canonical placement payload is missing: {canonical_path}")
    elif strategy_mode == "replay_2k":
        raise FileNotFoundError(f"Canonical placement is missing: {canonical_path}")

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
    input_sharding = tuple(
        input_sharding_by_axis[name] for name in dense_mesh.mesh_dim_names
    )
    output_sharding = tuple(
        Shard(2) if name == "tp" else Shard(0)
        for name in dense_mesh.mesh_dim_names
    )

    audit_dir = Path(os.environ["PLACEMENT_AUDIT_DIR"])
    audit_dir.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["RANK"])
    trace_start = time.perf_counter()
    autop_context = AutoParallelGraph(
        model,
        input_fn,
        dense_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
    )
    with autop_context as autop:
        trace_seconds = time.perf_counter() - trace_start
        constraint_start = time.perf_counter()
        autop.add_parameter_memory_constraint(low=None, high=None)
        autop.add_input_constraints([input_sharding, input_sharding])
        autop.add_output_constraints([output_sharding])
        constraint_seconds = time.perf_counter() - constraint_start

        operation_start = time.perf_counter()
        if strategy_mode == "fresh":
            sharding_placement = autop.optimize_placement(verbose=False)
            solver_invoked = True
            solution_path = audit_dir / f"solution_rank_{rank:03d}.json"
            autop.sharding_optimizer.save_placements(solution_path)
            solution_payload = json.loads(solution_path.read_text())["placements"]
        else:
            sharding_placement = autop.sharding_optimizer.load_placements(canonical_path)
            solver_invoked = False
            solution_path = canonical_path
            solution_payload = _placement_payload(sharding_placement)
            if solution_payload != canonical_payload:
                raise RuntimeError("Loaded solution differs from the canonical 2K plan")
        operation_seconds = time.perf_counter() - operation_start
        solution_digest = _placement_digest(solution_payload)

        concrete_solution = autop.sharding_optimizer._to_concrete_solution(
            sharding_placement
        )
        solution_cost = autop.sharding_optimizer._compute_solution_cost(
            concrete_solution
        )
        canonical_cost = None
        placement_diff = None
        if strategy_mode == "fresh" and canonical_payload is not None:
            canonical_solution = autop.sharding_optimizer.load_placements(canonical_path)
            loaded_canonical_payload = _placement_payload(canonical_solution)
            if loaded_canonical_payload != canonical_payload:
                raise RuntimeError("Target graph did not load the canonical 2K plan exactly")
            canonical_cost = autop.sharding_optimizer._compute_solution_cost(
                autop.sharding_optimizer._to_concrete_solution(canonical_solution)
            )
            if set(solution_payload) != set(canonical_payload):
                raise RuntimeError("Fresh and canonical solutions have different node sets")
            changed_nodes = [
                name
                for name in solution_payload
                if solution_payload[name] != canonical_payload[name]
            ]
            placement_diff = {
                "changed_nodes": len(changed_nodes),
                "output_changed_nodes": sum(
                    solution_payload[name]["output"]
                    != canonical_payload[name]["output"]
                    for name in changed_nodes
                ),
                "input_changed_nodes": sum(
                    solution_payload[name]["inputs"]
                    != canonical_payload[name]["inputs"]
                    for name in changed_nodes
                ),
                "changed_node_names": changed_nodes,
            }

        logger.info(
            "AutoParallel strategy mode=%s took %.2f seconds "
            "(%d nodes, digest=%s)",
            strategy_mode,
            operation_seconds,
            len(solution_payload),
            solution_digest,
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
        apply_start = time.perf_counter()
        parallel_mod = autop.apply_placement_for_fx_module(
            sharding_placement,
            compile_config=compile_config,
            model_output=model_output,
        )
        apply_seconds = time.perf_counter() - apply_start

    audit = {
        "rank": rank,
        "mode": strategy_mode,
        "solver_invoked": solver_invoked,
        "canonical_sequence_length": CANONICAL_SEQUENCE_LENGTH,
        "target_sequence_length": training.seq_len,
        "mesh_shape": list(mesh_shape),
        "mesh_dim_names": list(dense_mesh.mesh_dim_names),
        "placement_file": str(solution_path),
        "canonical_file": str(canonical_path),
        "canonical_file_sha256": canonical_file_sha256,
        "placement_digest": solution_digest,
        "canonical_placement_digest": (
            _placement_digest(canonical_payload) if canonical_payload is not None else None
        ),
        "placed_nodes": len(solution_payload),
        "trace_seconds": trace_seconds,
        "constraint_seconds": constraint_seconds,
        "solve_or_load_seconds": operation_seconds,
        "apply_placement_seconds": apply_seconds,
        "solution_cost": solution_cost,
        "canonical_cost_on_target_graph": canonical_cost,
        "placement_diff_from_canonical": placement_diff,
    }
    compile_setup_start = time.perf_counter()
    compiled = apply_compile(
        parallel_mod,
        compile_config=compile_config,
        parallelism=parallelism,
        parallel_dims=parallel_dims,
        dump_folder=dump_folder,
    )
    audit["apply_compile_setup_seconds"] = time.perf_counter() - compile_setup_start
    (audit_dir / f"rank_{rank:03d}.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    return compiled


def _load_c4_shard(dataset_path: str, *, shard_index: int):
    if dataset_path != C4_REPO:
        raise ValueError(f"Expected C4 path {C4_REPO!r}, got {dataset_path!r}")
    if not 0 <= shard_index < C4_TRAIN_SHARDS:
        raise ValueError(f"Invalid C4 train shard index {shard_index}")
    if shard_index not in C4_SHARD_METADATA:
        raise ValueError(f"Offline C4 shard {shard_index} was not packaged")
    filename = f"en/c4-train.{shard_index:05d}-of-{C4_TRAIN_SHARDS:05d}.json.gz"
    expected_size, expected_sha256 = C4_SHARD_METADATA[shard_index]
    local_file = Path(os.environ["C4_SHARD_ROOT"]) / filename
    if not local_file.is_file():
        raise FileNotFoundError(f"Required offline C4 shard is missing: {local_file}")
    if local_file.stat().st_size != expected_size:
        raise RuntimeError(
            f"Offline C4 shard size mismatch for {local_file}: "
            f"{local_file.stat().st_size} != {expected_size}"
        )
    digest = hashlib.sha256()
    with local_file.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise RuntimeError(
            f"Offline C4 shard SHA-256 mismatch for {local_file}: "
            f"{digest.hexdigest()} != {expected_sha256}"
        )
    return load_dataset(
        "json",
        data_files={"train": [str(local_file)]},
        features=Features(
            {
                "text": Value("string"),
                "timestamp": Value("string"),
                "url": Value("string"),
            }
        ),
        split="train",
        streaming=True,
    )


DATASETS[DATASET_NAME] = DatasetConfig(
    path=C4_REPO,
    loader=partial(_load_c4_shard, shard_index=0),
    sample_processor=lambda sample: sample["text"],
)


class PinnedC4Dataset(HuggingFaceTextDataset):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: BaseTokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:
        if dp_world_size != DP_DEGREE:
            raise ValueError(
                f"Expected C4 dp_world_size={DP_DEGREE}, got {dp_world_size}"
            )
        rank_dataset_name = f"{DATASET_NAME}_dp{dp_rank}"
        DATASETS[rank_dataset_name] = DatasetConfig(
            path=C4_REPO,
            loader=partial(_load_c4_shard, shard_index=dp_rank),
            sample_processor=lambda sample: sample["text"],
        )
        super().__init__(
            dataset_name=rank_dataset_name,
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            seq_len=seq_len,
            dp_rank=0,
            dp_world_size=1,
            infinite=infinite,
        )
        self.dataset_name = DATASET_NAME


class AuditedC4DataLoader(ParallelAwareDataloader):
    @dataclass(kw_only=True, slots=True)
    class Config(HuggingFaceTextDataLoader.Config):
        pass

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        seq_len: int,
        local_batch_size: int,
        snapshot_every_n_steps: int | None = 1,
        **kwargs,
    ) -> None:
        del kwargs
        dataset = PinnedC4Dataset(
            dataset_name=config.dataset,
            dataset_path=config.dataset_path,
            tokenizer=tokenizer,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=config.infinite,
        )
        super().__init__(
            dataset,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            num_workers=config.num_workers,
            persistent_workers=config.persistent_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
            snapshot_every_n_steps=snapshot_every_n_steps,
            batch_size=local_batch_size,
        )

    def __iter__(self):
        # Input identity is fixed and audited by the campaign index manifest.
        # Keep the timed iterator free of tensor hashing and filesystem writes.
        replay_size = int(os.environ.get("FIXED_REPLAY_BATCHES", "10"))
        if replay_size < 1:
            raise ValueError("FIXED_REPLAY_BATCHES must be positive")
        source = super().__iter__()
        replay_batches = [next(source) for _ in range(replay_size)]
        batch_index = 0
        while True:
            input_dict, labels = replay_batches[batch_index % replay_size]
            batch_index += 1
            yield dict(input_dict), labels


def _base_config():
    config = llama3_8b()
    config.comm = replace(config.comm, init_timeout_seconds=1200)
    config.model_spec = model_registry(MODEL_FLAVOR, attn_backend="sdpa")
    config.model_spec = replace(
        config.model_spec,
        post_optimizer_build_fn=_register_post_load_audits,
    )
    config.hf_assets_path = os.environ["LLAMA_TOKENIZER_DIR"]
    config.loss = CrossEntropyLoss.Config(
        global_vocab_size=decoder_vocab_size(config.model_spec)
    )
    config.dataloader = AuditedC4DataLoader.Config(dataset=DATASET_NAME)
    config.optimizer = default_adamw(lr=3e-4)
    config.lr_scheduler = LRSchedulersContainer.Config(warmup_steps=200)
    config.training = replace(
        config.training,
        local_batch_size=LOCAL_BATCH_SIZE,
        global_batch_size=GLOBAL_BATCH_SIZE,
        seq_len=SEQ_LEN,
        steps=28,
        dtype="float32",
        mixed_precision_param="bfloat16",
        mixed_precision_reduce="float32",
        max_norm=1.0,
    )
    config.parallelism = replace(
        config.parallelism,
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=DP_DEGREE,
        tensor_parallel_degree=TP_DEGREE,
        enable_sequence_parallel=True,
        context_parallel_degree=1,
        pipeline_parallel_degree=1,
        expert_parallel_degree=1,
        fsdp_reshard_after_forward="default",
    )
    config.activation_checkpoint = SelectiveAC.Config()
    config.metrics = replace(
        config.metrics,
        log_freq=5,
        enable_tensorboard=True,
        save_for_all_ranks=True,
        enable_wandb=False,
        disable_color_printing=True,
    )
    config.profiler = replace(
        config.profiler,
        enable_profiling=True,
        profile_freq=28,
        profiler_warmup=0,
        profiler_active=3,
        profiler_repeat=1,
        enable_memory_snapshot=False,
    )
    config.validator = replace(config.validator, enable=False)
    config.debug = replace(
        config.debug,
        seed=42,
        deterministic=False,
        deterministic_warn_only=False,
        enable_structured_logging=True,
        print_config=False,
        save_config_file="config.json",
    )
    config.checkpoint = replace(
        config.checkpoint,
        enable=False,
        load_only=False,
        initial_load_path=None,
        initial_load_model_only=False,
        create_seed_checkpoint=False,
    )
    return config


def _graph_config():
    config = to_graph_trainer_config(_base_config(), model_registry)
    # ``to_graph_trainer_config`` installs a cudagraph-only trace annotator.
    # Cudagraphs are disabled in this experiment, so keep the profiler config
    # identical to the backend configuration instead of serializing a no-op callback.
    config.profiler = replace(config.profiler, trace_post_processor=None)
    config.compile = GraphTrainerCompileConfig(
        enable=True,
        components=["model", "loss"],
        enable_autoparallel=True,
    )
    return config


def _backend_config():
    config = _base_config()
    name = "autoparallel_backend/example_scheduling/llama3"
    config.model_spec = replace(
        config.model_spec,
        name=name,
        parallelize_fn=parallelize_autoparallel_backend_llama,
    )
    config.compile = CompileConfig(enable=False)
    return config


def autoparallel_backend_example_scheduling_8b():
    return _backend_config()


def autoparallel_graphtrainer_full_inductor_8b():
    return _graph_config()


def autoparallel_graphtrainer_seqlen_8b():
    config = _graph_config()
    config.model_spec = replace(
        config.model_spec,
        name="autoparallel_graphtrainer/seqlen_fresh_vs_2k/llama3",
        parallelize_fn=parallelize_autoparallel_graphtrainer_seqlen_llama,
    )
    return config


def torchtitan_baseline_8b():
    if "autoparallel.graph_passes.auto_bucketing" in sys.modules:
        raise RuntimeError("Native TorchTitan baseline was polluted by AutoParallel")
    from torchtitan.models.llama3.parallelize import parallelize_llama

    config = _base_config()
    config.model_spec = replace(
        config.model_spec,
        name="torchtitan/native_compile/llama3",
        parallelize_fn=parallelize_llama,
    )
    config.compile = CompileConfig(
        enable=True,
        components=["model", "loss"],
        backend="inductor",
    )
    return config
