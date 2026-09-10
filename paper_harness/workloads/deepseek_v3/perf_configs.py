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
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard

from torchtitan.config import CompileConfig, TORCH_DTYPE_MAP
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.optimizer import (
    default_adamw,
    LRSchedulersContainer,
    register_moe_load_balancing_hook,
)
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.deepseek_v3 import (
    model_registry as graph_model_registry,
)
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.deepseek_v3.config_registry import deepseek_v3_16b
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import device_type

from workloads.fixed_shape_text import (
    DatasetConfig,
    FixedShapeTextDataLoader,
    FixedShapeTextDataset,
)


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
}
DATASET_NAME = "c4_deepseek16b_pinned_audited"
DATASETS: dict[str, DatasetConfig] = {}

MODEL_FLAVOR = os.environ.get("BENCHMARK_MODEL", "16B")
if MODEL_FLAVOR != "16B":
    raise ValueError(f"Unsupported BENCHMARK_MODEL={MODEL_FLAVOR!r}")

WORLD_SIZE = int(os.environ["BENCHMARK_WORLD_SIZE"])
EP_DEGREE = int(os.environ["BENCHMARK_EP_DEGREE"])
if EP_DEGREE != 8 or WORLD_SIZE not in (16, 32):
    raise ValueError(f"Expected world size 16/32 with EP8, got {WORLD_SIZE=}, {EP_DEGREE=}")
DP_DEGREE = WORLD_SIZE
EFSDP_DEGREE = WORLD_SIZE // EP_DEGREE

LOCAL_BATCH_SIZE = 4
SEQ_LEN = 4096
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


def _normalize_parameter_name(name: str) -> str:
    prefixes = ("model._orig_mod.", "_orig_mod.", "model.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name.removeprefix(prefix)
                changed = True
    return (
        name.replace(
            ".moe.experts.w1",
            ".moe.routed_experts.inner_experts.w1_EFD",
        )
        .replace(
            ".moe.experts.w2",
            ".moe.routed_experts.inner_experts.w2_EDF",
        )
        .replace(
            ".moe.experts.w3",
            ".moe.routed_experts.inner_experts.w3_EFD",
        )
    )


def _audit_model_parameters(_optimizers, model_parts, _parallel_dims):
    """Record shard-local initialized weights without changing training state."""
    _write_inductor_path_audit()
    audit_dir = os.environ.get("PARAMETER_AUDIT_DIR")
    if audit_dir is None:
        return

    records = []
    with torch.no_grad():
        for part_index, model_part in enumerate(model_parts):
            for name, parameter in model_part.named_parameters():
                local = (
                    parameter.to_local()
                    if isinstance(parameter, DTensor)
                    else parameter
                )
                flat = local.detach().reshape(-1)
                sample = flat[: min(flat.numel(), 8192)].contiguous()
                sample_bytes = sample.view(torch.uint8).cpu().numpy().tobytes()
                float_local = local.detach().float()
                records.append(
                    {
                        "part": part_index,
                        "name": _normalize_parameter_name(name),
                        "raw_name": name,
                        "global_shape": list(parameter.shape),
                        "local_shape": list(local.shape),
                        "dtype": str(parameter.dtype),
                        "sample_sha256": hashlib.sha256(sample_bytes).hexdigest(),
                        "sum": float(float_local.sum().item()),
                        "square_sum": float(float_local.square().sum().item()),
                    }
                )

    output_dir = Path(audit_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"rank_{int(os.environ['RANK']):02d}.json"
    output.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")


def _audit_and_register_moe_hook(optimizers, model_parts, parallel_dims):
    _audit_model_parameters(optimizers, model_parts, parallel_dims)
    register_moe_load_balancing_hook(optimizers, model_parts, parallel_dims)


def parallelize_autoparallel_backend_deepseek(
    model,
    *,
    parallel_dims,
    training,
    parallelism,
    compile_config,
    ac_config,
    dump_folder,
):
    """Run the DeepSeek 16B local-map model through the AP backend path."""
    from autoparallel import AutoParallel, ForwardInputs
    from autoparallel._testing.models.dsv3 import DeepSeekV3Model
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
    del compile_config
    if parallel_dims.dp_replicate_enabled:
        raise ValueError("AutoParallel DeepSeek V3 does not support DDP yet")
    if parallel_dims.cp_enabled:
        raise ValueError("AutoParallel DeepSeek V3 does not support CP yet")
    if parallel_dims.pp_enabled:
        raise ValueError("AutoParallel DeepSeek V3 does not support PP yet")
    if parallel_dims.tp_enabled:
        raise ValueError("AutoParallel DeepSeek V3 does not support TP yet")

    sparse_mesh = parallel_dims.get_mesh(["efsdp", "ep"])
    if sparse_mesh.mesh_dim_names != ("efsdp", "ep"):
        raise RuntimeError(f"Unexpected sparse mesh axes: {sparse_mesh.mesh_dim_names}")

    model_config = model.config
    with torch.device("meta"):
        ap_model = DeepSeekV3Model(
            model_config,
            mesh=sparse_mesh,
            compute_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        )
    del model

    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(ap_model)

    def input_fn():
        dp_degree = parallel_dims.dp_replicate * parallel_dims.dp_shard
        global_tokens = training.num_tokens_per_train_step
        if global_tokens < 0:
            global_tokens = (
                training.num_tokens_per_microbatch_per_dp_rank * dp_degree
            )
        global_batch_size, remainder = divmod(
            global_tokens, training.max_context_length
        )
        if remainder:
            raise RuntimeError("AutoParallel placement token batch is not rectangular")
        tokens = torch.randint(
            0,
            ap_model.model_args.vocab_size,
            (global_batch_size, training.max_context_length),
            device=torch.device(device_type),
        )
        positions = torch.arange(
            training.max_context_length,
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
    input_sharding = (Shard(0), Shard(0))

    with AutoParallel(
        ap_model,
        input_fn,
        sparse_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
        dynamic=True,
    ) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)
        autop.add_input_constraints([input_sharding, input_sharding])
        autop.add_output_constraints([input_sharding])
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

    set_nccl_topo_config(detect_nccl_topo_config(sparse_mesh))
    custom_runtime_estimation = make_custom_runtime_estimation(sparse_mesh)
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
    return parallel_model


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


class PinnedC4Dataset(FixedShapeTextDataset):
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
        shard_index = dp_rank % len(C4_SHARD_METADATA)
        DATASETS[rank_dataset_name] = DatasetConfig(
            path=C4_REPO,
            loader=partial(_load_c4_shard, shard_index=shard_index),
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
            datasets=DATASETS,
        )
        self.dataset_name = DATASET_NAME


class AuditedC4DataLoader(FixedShapeTextDataLoader):
    @dataclass(kw_only=True, slots=True)
    class Config(FixedShapeTextDataLoader.Config):
        pass

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        max_context_length: int,
        num_tokens_per_batch: int,
        **kwargs,
    ) -> None:
        del kwargs
        local_batch_size, remainder = divmod(
            num_tokens_per_batch, max_context_length
        )
        if remainder:
            raise ValueError("C4 token batch is not rectangular")
        dataset = PinnedC4Dataset(
            dataset_name=config.dataset,
            dataset_path=config.dataset_path,
            tokenizer=tokenizer,
            seq_len=max_context_length,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=config.infinite,
        )
        super().__init__(
            config,
            dataset=dataset,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            max_context_length=max_context_length,
            num_tokens_per_batch=num_tokens_per_batch,
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
    config = deepseek_v3_16b()
    config.comm = replace(
        config.comm,
        init_timeout_seconds=1200,
        train_timeout_seconds=1200,
    )
    config.model_spec = graph_model_registry(MODEL_FLAVOR, attn_backend="sdpa")
    config.model_spec = replace(
        config.model_spec,
        post_optimizer_build_fn=_audit_and_register_moe_hook,
    )
    config.hf_assets_path = os.environ["DEEPSEEK_TOKENIZER_DIR"]
    config.loss = CrossEntropyLoss.Config(
        global_vocab_size=decoder_vocab_size(config.model_spec)
    )
    config.dataloader = AuditedC4DataLoader.Config(dataset=DATASET_NAME)
    config.optimizer = default_adamw(lr=2.2e-4)
    config.training = replace(
        config.training,
        num_tokens_per_microbatch_per_dp_rank=LOCAL_BATCH_SIZE * SEQ_LEN,
        num_tokens_per_train_step=GLOBAL_BATCH_SIZE * SEQ_LEN,
        max_context_length=SEQ_LEN,
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
        tensor_parallel_degree=1,
        enable_sequence_parallel=False,
        context_parallel_degree=1,
        pipeline_parallel_degree=1,
        expert_parallel_degree=EP_DEGREE,
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
        enable_profiling=int(os.environ.get("RANK", "0")) == 0,
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
):
    config = to_graph_trainer_config(_base_config(), graph_model_registry)
    if enable_autoparallel:
        def reject_flat_autoparallel_input(*args, **kwargs):
            del args, kwargs
            raise RuntimeError(
                "Latest TorchTitan DeepSeek AutoParallel traces a flat token stream; "
                "the paper workload requires the historical independent [B, S] "
                "microbatch and is intentionally fail-closed"
            )

    config.model_spec = replace(
        config.model_spec,
        post_optimizer_build_fn=_audit_and_register_moe_hook,
        parallelize_fn=(
            reject_flat_autoparallel_input
            if enable_autoparallel
            else config.model_spec.parallelize_fn
        ),
    )
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
    name = "autoparallel_backend/example_scheduling/deepseek_v3"
    config.model_spec = replace(
        config.model_spec,
        name=name,
        parallelize_fn=parallelize_autoparallel_backend_deepseek,
    )
    config.compile = CompileConfig(enable=False)
    return config


def autoparallel_backend_example_scheduling_16b():
    return _backend_config()


def autoparallel_graphtrainer_full_inductor_16b():
    return _graph_config(inductor_compilation="full")


def torchtitan_baseline_16b():
    if "autoparallel.graph_passes.auto_bucketing" in sys.modules:
        raise RuntimeError("Native TorchTitan baseline was polluted by AutoParallel")
    from torchtitan.models.deepseek_v3.parallelize import parallelize_deepseekv3

    config = _base_config()
    config.model_spec = replace(
        config.model_spec,
        name="torchtitan/native_compile/deepseek_v3",
        parallelize_fn=parallelize_deepseekv3,
    )
    config.compile = CompileConfig(
        enable=True,
        components=["model", "loss"],
        backend="inductor",
    )
    return config


EXPERIMENT_CONFIGS = (
    "torchtitan_baseline_16b",
    "autoparallel_backend_example_scheduling_16b",
    "autoparallel_graphtrainer_full_inductor_16b",
)
