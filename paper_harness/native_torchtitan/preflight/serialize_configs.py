from __future__ import annotations

import argparse
import json
from pathlib import Path

from torchtitan.config import ConfigManager


CONFIG_NAME = "muse_glimmer_30b_sdpa_c4_torchtitan_4x2"


def parse_config(
    *,
    phase: str,
    assets: Path,
    dump_folder: str,
) -> dict:
    phase_args = {
        "gate": [
            "--training.steps",
            "2",
            "--profiler.no-enable-profiling",
            "--metrics.no-enable-tensorboard",
            "--metrics.log-freq",
            "1",
        ],
        "performance": [
            "--training.steps",
            "25",
            "--profiler.no-enable-profiling",
            "--metrics.enable-tensorboard",
            "--metrics.log-freq",
            "5",
        ],
        "trace": [
            "--training.steps",
            "6",
            "--profiler.enable-profiling",
            "--profiler.profile-freq",
            "6",
            "--profiler.profiler-warmup",
            "0",
            "--profiler.profiler-active",
            "3",
            "--profiler.profiler-repeat",
            "1",
            "--profiler.no-enable-memory-snapshot",
            "--metrics.enable-tensorboard",
            "--metrics.log-freq",
            "1",
        ],
    }[phase]
    config = ConfigManager().parse_args(
        [
            "--module",
            "graph_trainer.muse_glimmer",
            "--config",
            CONFIG_NAME,
            "--hf-assets-path",
            str(assets),
            "--dump-folder",
            dump_folder,
            "--debug.save-config-file",
            "config.json",
            "--checkpoint.no-enable",
            "--validator.no-enable",
            "--debug.seed",
            "42",
            "--comm.init-timeout-seconds",
            "1800",
            "--compile.enable",
            "--compile.backend",
            "inductor",
            "--training.local-batch-size",
            "2",
            "--training.global-batch-size",
            "-1",
            "--training.seq-len",
            "4096",
            "--parallelism.data-parallel-shard-degree",
            "8",
            "--parallelism.tensor-parallel-degree",
            "2",
            *phase_args,
        ]
    )
    return config.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    assets = args.runner_root.resolve() / "assets" / "muse_glimmer"
    summary = {}
    for phase in ("gate", "performance", "trace"):
        config = parse_config(
            phase=phase,
            assets=assets,
            dump_folder=f"/mnt/wsfuse/outputs/JOB/native_{phase}",
        )
        compile_config = config["compile"]
        if compile_config != {
            "enable": True,
            "enable_async_tensor_parallel": False,
            "components": ["model", "loss"],
            "backend": "inductor",
        }:
            raise RuntimeError(f"unexpected native compile config: {compile_config}")
        if config["training"]["local_batch_size"] != 2:
            raise RuntimeError("unexpected local batch size")
        if config["training"]["global_batch_size"] != -1:
            raise RuntimeError("global batch must be derived")
        if config["training"]["seq_len"] != 4096:
            raise RuntimeError("unexpected sequence length")
        if config["parallelism"]["data_parallel_shard_degree"] != 8:
            raise RuntimeError("unexpected FSDP degree")
        if config["parallelism"]["tensor_parallel_degree"] != 2:
            raise RuntimeError("unexpected TP degree")
        path = output_dir / f"world16.native.{phase}.json"
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        summary[phase] = {
            "config": str(path),
            "steps": config["training"]["steps"],
            "profiler_enabled": config["profiler"]["enable_profiling"],
            "tensorboard_enabled": config["metrics"]["enable_tensorboard"],
        }

    report = {
        "status": "passed",
        "config_name": CONFIG_NAME,
        "trainer_backend": "native_torchtitan",
        "world_size": 16,
        "fsdp_degree": 8,
        "tensor_parallel_degree": 2,
        "local_batch_size_per_dp_rank": 2,
        "global_batch_size": 16,
        "sequence_length": 4096,
        "phases": summary,
    }
    (output_dir / "validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
