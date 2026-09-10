from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.utils.data import IterableDataset

from torchtitan.config import CompileConfig, TORCH_DTYPE_MAP
from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import default_adamw
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.llama3 import model_registry
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
DATASET_NAME = "c4_llama8b_topology_invariant_replay"

MODEL_FLAVOR = os.environ.get("BENCHMARK_MODEL", "8B")
if MODEL_FLAVOR != "8B":
    raise ValueError(f"Unsupported BENCHMARK_MODEL={MODEL_FLAVOR!r}")

WORLD_SIZE = int(os.environ["BENCHMARK_WORLD_SIZE"])
TP_DEGREE = int(os.environ["BENCHMARK_TP_DEGREE"])
if (WORLD_SIZE, TP_DEGREE) not in {
    (8, 4),
    (16, 8),
    (32, 8),
    (64, 8),
    (128, 8),
    (8, 1),
    (16, 1),
    (32, 1),
    (64, 1),
    (128, 1),
}:
    raise ValueError(f"Unsupported mesh: {WORLD_SIZE=} {TP_DEGREE=}")
DP_DEGREE = WORLD_SIZE // TP_DEGREE

LOCAL_BATCH_SIZE = 2
SEQ_LEN = 8192
GLOBAL_BATCH_SIZE = int(os.environ["BENCHMARK_GLOBAL_BATCH_SIZE"])
if GLOBAL_BATCH_SIZE % (LOCAL_BATCH_SIZE * DP_DEGREE) != 0:
    raise ValueError(
        "Global batch must equal local batch per DP rank * DP degree * "
        "an integer gradient accumulation count"
    )
GRADIENT_ACCUMULATION_STEPS = GLOBAL_BATCH_SIZE // (
    LOCAL_BATCH_SIZE * DP_DEGREE
)
PLACEMENT_GLOBAL_BATCH_SIZE = LOCAL_BATCH_SIZE * DP_DEGREE
REPLAY_SLOTS = 10


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
    expected_patch_active = configuration in {
        "graphtrainer_manual",
        "autoparallel_backend_example_scheduling",
        "autoparallel_graphtrainer",
        "autoparallel_graphtrainer_current",
    }
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
        dp_degree = parallel_dims.dp_replicate * parallel_dims.dp_shard
        global_batch_size = training.local_batch_size * dp_degree
        if global_batch_size != PLACEMENT_GLOBAL_BATCH_SIZE:
            raise RuntimeError(
                "AutoParallel placement input must represent one microbatch "
                f"across DP ranks: {global_batch_size} != "
                f"{PLACEMENT_GLOBAL_BATCH_SIZE}"
            )
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


def parallelize_graphtrainer_autoparallel_llama(
    model,
    *,
    parallel_dims,
    training,
    **kwargs,
):
    """Keep the AP placement example at one microbatch across DP ranks."""
    from torchtitan.experiments.graph_trainer.llama3.parallelize_autoparallel import (
        parallelize_autoparallel_llama,
    )

    dp_degree = parallel_dims.dp_replicate * parallel_dims.dp_shard
    placement_global_batch_size = training.local_batch_size * dp_degree
    if placement_global_batch_size != PLACEMENT_GLOBAL_BATCH_SIZE:
        raise RuntimeError(
            "GraphTrainer AutoParallel placement batch mismatch: "
            f"{placement_global_batch_size} != {PLACEMENT_GLOBAL_BATCH_SIZE}"
        )
    return parallelize_autoparallel_llama(
        model,
        parallel_dims=parallel_dims,
        training=replace(
            training,
            global_batch_size=placement_global_batch_size,
        ),
        **kwargs,
    )


class TopologyInvariantReplayDataset(IterableDataset):
    def __init__(
        self,
        replay_path: Path,
        *,
        dp_rank: int,
        dp_world_size: int,
        local_batch_size: int,
    ) -> None:
        if dp_world_size != DP_DEGREE:
            raise ValueError(
                f"Expected replay dp_world_size={DP_DEGREE}, got {dp_world_size}"
            )
        if local_batch_size != LOCAL_BATCH_SIZE:
            raise ValueError(
                f"Expected local batch {LOCAL_BATCH_SIZE}, got {local_batch_size}"
            )
        payload = torch.load(replay_path, map_location="cpu", mmap=True, weights_only=True)
        expected_shape = (REPLAY_SLOTS, 256, SEQ_LEN)
        for name in ("input", "positions", "labels"):
            tensor = payload.get(name)
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"Replay tensor {name!r} has invalid shape: "
                    f"{None if tensor is None else tuple(tensor.shape)}"
                )
            if tensor.dtype != torch.int64:
                raise ValueError(f"Replay tensor {name!r} must be int64")
        self.payload = payload
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.local_batch_size = local_batch_size

    def __iter__(self):
        slot = 0
        accumulation_index = 0
        while True:
            first = (
                accumulation_index * self.dp_world_size + self.dp_rank
            ) * self.local_batch_size
            for sample_index in range(first, first + self.local_batch_size):
                yield (
                    {
                        "input": self.payload["input"][slot, sample_index],
                        "positions": self.payload["positions"][slot, sample_index],
                    },
                    self.payload["labels"][slot, sample_index],
                )
            accumulation_index += 1
            if accumulation_index == GRADIENT_ACCUMULATION_STEPS:
                accumulation_index = 0
                slot = (slot + 1) % REPLAY_SLOTS


class AuditedReplayDataLoader(ParallelAwareDataloader):
    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        pass

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer,
        seq_len: int,
        local_batch_size: int,
        snapshot_every_n_steps: int | None = 1,
        **kwargs,
    ) -> None:
        del kwargs, tokenizer
        if seq_len != SEQ_LEN:
            raise ValueError(f"Expected sequence length {SEQ_LEN}, got {seq_len}")
        dataset = TopologyInvariantReplayDataset(
            Path(os.environ["REPLAY_TENSORS_PATH"]),
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            local_batch_size=local_batch_size,
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
        source = super().__iter__()
        while True:
            input_dict, labels = next(source)
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
    config.dataloader = AuditedReplayDataLoader.Config(dataset=DATASET_NAME)
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


def _graph_config(
    *,
    inductor_compilation: str,
    enable_remat: bool = True,
    enable_autoparallel: bool = True,
    pass_pipeline: str = "default",
    use_historical_ap_batch_adapter: bool = True,
):
    config = to_graph_trainer_config(_base_config(), model_registry)
    if use_historical_ap_batch_adapter:
        config.model_spec = replace(
            config.model_spec,
            parallelize_fn=parallelize_graphtrainer_autoparallel_llama,
        )
    # ``to_graph_trainer_config`` installs a cudagraph-only trace annotator.
    # Cudagraphs are disabled in this experiment, so keep the profiler config
    # identical to the backend configuration instead of serializing a no-op callback.
    config.profiler = replace(config.profiler, trace_post_processor=None)
    disabled_passes = ["cudagraph_pass"]
    if not enable_remat:
        disabled_passes.extend(
            ["tag_with_memory_policy_pass", "selective_activation_remat_pass"]
        )
    config.compile = GraphTrainerCompileConfig(
        enable=True,
        components=["model", "loss"],
        mode="aot_fx_trace",
        memory_policy="eager",
        inductor_compilation=inductor_compilation,
        numerics_changing_optim=False,
        disable_passes=disabled_passes,
        enable_fsdp_ag_rs_overlap=False,
        enable_fsdp_dense_region_overlap=False,
        enable_autoparallel=enable_autoparallel,
        pass_pipeline=pass_pipeline,
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
    return _graph_config(inductor_compilation="full")


def graphtrainer_manual_full_inductor_8b():
    return _graph_config(inductor_compilation="full", enable_autoparallel=False)


def autoparallel_graphtrainer_full_inductor_current_8b():
    return _graph_config(
        inductor_compilation="full",
        use_historical_ap_batch_adapter=False,
    )


def graphtrainer_manual_full_inductor_current_8b():
    return _graph_config(
        inductor_compilation="full",
        enable_autoparallel=False,
        use_historical_ap_batch_adapter=False,
    )


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


EXPERIMENT_CONFIGS = (
    "torchtitan_baseline_8b",
    "graphtrainer_manual_full_inductor_8b",
    "graphtrainer_manual_full_inductor_current_8b",
    "autoparallel_backend_example_scheduling_8b",
    "autoparallel_graphtrainer_full_inductor_8b",
    "autoparallel_graphtrainer_full_inductor_current_8b",
)
