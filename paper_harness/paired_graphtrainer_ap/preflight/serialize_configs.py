from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from torchtitan.config import ConfigManager


CONFIGS = {
    "manual": "graph_trainer_muse_glimmer_30b_sdpa_c4_4x2",
    "autoparallel": "graph_trainer_muse_glimmer_30b_sdpa_c4_autoparallel_4x2",
}


def parse_config(
    *,
    config_name: str,
    fsdp_degree: int,
    local_batch_size: int,
    assets: Path,
    dump_folder: str,
) -> dict:
    config = ConfigManager().parse_args(
        [
            "--module",
            "graph_trainer.muse_glimmer",
            "--config",
            config_name,
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
            "--training.local-batch-size",
            str(local_batch_size),
            "--training.global-batch-size",
            "-1",
            "--training.seq-len",
            "4096",
            "--parallelism.data-parallel-shard-degree",
            str(fsdp_degree),
            "--parallelism.tensor-parallel-degree",
            "2",
            "--training.steps",
            "25",
            "--profiler.no-enable-profiling",
            "--metrics.enable-tensorboard",
            "--metrics.log-freq",
            "5",
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
    parity = {}
    world_size = 16
    fsdp_degree = 8
    for case_name, local_batch_size in (("per_gpu_bs1", 2),):
        configs = {
            arm: parse_config(
                config_name=config_name,
                fsdp_degree=fsdp_degree,
                local_batch_size=local_batch_size,
                assets=assets,
                dump_folder=f"/mnt/wsfuse/outputs/JOB/{case_name}_s4096_world16",
            )
            for arm, config_name in CONFIGS.items()
        }
        for arm, config in configs.items():
            path = output_dir / f"{case_name}.{arm}.performance.json"
            path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

        normalized = copy.deepcopy(configs)
        toggles = {
            arm: config["compile"].pop("enable_autoparallel")
            for arm, config in normalized.items()
        }
        if normalized["manual"] != normalized["autoparallel"]:
            raise RuntimeError(f"{case_name} configs differ beyond AP toggle")
        if toggles != {"manual": False, "autoparallel": True}:
            raise RuntimeError(f"{case_name} unexpected AP toggles: {toggles}")
        parity[case_name] = {
            "status": "passed",
            "world_size": world_size,
            "fsdp_degree": fsdp_degree,
            "tensor_parallel_degree": 2,
            "local_batch_size_per_dp_rank": local_batch_size,
            "global_batch_size": local_batch_size * fsdp_degree,
            "per_physical_gpu_effective_batch_size": local_batch_size / 2,
            "autoparallel_toggle": toggles,
        }

    (output_dir / "parity.json").write_text(
        json.dumps(parity, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(parity, sort_keys=True))


if __name__ == "__main__":
    main()
