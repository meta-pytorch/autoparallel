#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import os
import socket
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

FORMAL_MESSAGE_BYTES = (
    4 << 10,
    16 << 10,
    64 << 10,
    256 << 10,
    1 << 20,
    4 << 20,
    8 << 20,
    12 << 20,
    16 << 20,
    28 << 20,
    32 << 20,
    56 << 20,
    64 << 20,
    104 << 20,
    112 << 20,
    128 << 20,
    256 << 20,
    512 << 20,
    1 << 30,
)
GATE_MESSAGE_BYTES = (4 << 10, 1 << 20, 28 << 20, 128 << 20)
WORKLOAD_MESSAGE_BYTES = (28 << 20, 104 << 20, 128 << 20)

MODE_COLLECTIVES = {
    "auto": ("all_gather", "reduce_scatter", "all_reduce", "all_to_all"),
    "ring_ll": ("all_gather", "reduce_scatter", "all_reduce"),
    "ring_ll128": ("all_gather", "reduce_scatter", "all_reduce"),
    "ring_simple": ("all_gather", "reduce_scatter", "all_reduce"),
    "tree_ll": ("all_reduce",),
    "tree_ll128": ("all_reduce",),
    "tree_simple": ("all_reduce",),
    "pat_simple": ("all_gather", "reduce_scatter"),
    "nvlstree_simple": ("all_reduce",),
}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round(fraction * (len(ordered) - 1))]


def group_ranks(nodes: int, ppn: int, lane: int = 0) -> list[int]:
    return [
        node * 8 + local_rank
        for node in range(nodes)
        for local_rank in range(lane, lane + ppn)
    ]


def make_isolated_group(nodes: int, ppn: int) -> tuple[Any, list[int]]:
    ranks = group_ranks(nodes, ppn)
    return (
        dist.new_group(
            ranks,
            backend="nccl",
            group_desc=f"h100_roce400_isolated_n{nodes}_ppn{ppn}",
        ),
        ranks,
    )


def make_concurrent_group(nodes: int, ppn: int) -> tuple[Any, list[int]]:
    rank = dist.get_rank()
    selected: tuple[Any, list[int]] | None = None
    for lane in range(0, 8, ppn):
        ranks = group_ranks(nodes, ppn, lane)
        group = dist.new_group(
            ranks,
            backend="nccl",
            group_desc=f"h100_roce400_concurrent_n{nodes}_ppn{ppn}_lane{lane}",
        )
        if rank in ranks:
            selected = group, ranks
    if selected is None:
        raise RuntimeError(f"rank {rank} was not assigned to a concurrent group")
    return selected


def allocate(
    collective: str, n_bytes: int, group_size: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor | None]:
    elements = n_bytes // torch.empty((), dtype=dtype).element_size()
    if elements < group_size or elements % group_size:
        raise ValueError(f"{n_bytes=} is incompatible with {dtype=} and {group_size=}")
    if collective == "all_gather":
        return (
            torch.ones(elements // group_size, device="cuda", dtype=dtype),
            torch.empty(elements, device="cuda", dtype=dtype),
        )
    if collective == "reduce_scatter":
        return (
            torch.ones(elements, device="cuda", dtype=dtype),
            torch.empty(elements // group_size, device="cuda", dtype=dtype),
        )
    tensor = torch.ones(elements, device="cuda", dtype=dtype)
    return tensor, torch.empty_like(tensor) if collective == "all_to_all" else None


def run_collective(
    collective: str,
    input_tensor: torch.Tensor,
    output: torch.Tensor | None,
    group: Any,
) -> None:
    if collective == "all_gather":
        assert output is not None
        dist.all_gather_into_tensor(output, input_tensor, group=group)
    elif collective == "reduce_scatter":
        assert output is not None
        dist.reduce_scatter_tensor(output, input_tensor, group=group)
    elif collective == "all_reduce":
        dist.all_reduce(input_tensor, group=group)
    else:
        assert output is not None
        dist.all_to_all_single(output, input_tensor, group=group)


def measure(
    collective: str,
    n_bytes: int,
    group: Any,
    group_size: int,
    dtype: torch.dtype,
    gate: bool,
) -> list[float]:
    input_tensor, output = allocate(collective, n_bytes, group_size, dtype)
    warmups = 3 if gate else 8
    iterations = 5 if gate else (12 if n_bytes <= (128 << 20) else 6)
    for index in range(warmups):
        if collective == "all_reduce":
            input_tensor.fill_(1)
        run_collective(collective, input_tensor, output, group)
        if index == 0:
            result = input_tensor if collective == "all_reduce" else output
            assert result is not None
            expected = (
                group_size if collective in ("all_reduce", "reduce_scatter") else 1
            )
            if not torch.all(result == expected):
                raise RuntimeError(f"incorrect {collective} result")
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        if collective == "all_reduce":
            input_tensor.fill_(1)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_collective(collective, input_tensor, output, group)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return samples


def summarize(samples: list[float]) -> dict[str, float | list[float]]:
    return {
        "samples_us": [round(value, 3) for value in samples],
        "min_us": min(samples),
        "p50_us": statistics.median(samples),
        "p90_us": percentile(samples, 0.9),
        "max_us": max(samples),
    }


def append_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")


def benchmark_case(
    *,
    mode: str,
    execution: str,
    nodes: int,
    ppn: int,
    collective: str,
    n_bytes: int,
    dtype: torch.dtype,
    gate: bool,
    output_path: Path,
    group: dist.ProcessGroup,
    ranks: list[int],
    active: bool,
) -> None:
    rank = dist.get_rank()
    dist.barrier()
    local = None
    if active:
        local = summarize(measure(collective, n_bytes, group, len(ranks), dtype, gate))
    gathered: list[dict[str, Any] | None] | None = (
        [None] * dist.get_world_size() if rank == 0 else None
    )
    dist.gather_object(local, gathered, dst=0)
    if rank == 0:
        assert gathered is not None
        active_results = [item for item in gathered if item is not None]
        record = {
            "mode": mode,
            "execution": execution,
            "nodes": nodes,
            "ppn": ppn,
            "ranks": nodes * ppn,
            "collective": collective,
            "dtype": str(dtype),
            "n_bytes": n_bytes,
            "rank_results": active_results,
            "rank_p50_max_us": max(item["p50_us"] for item in active_results),
            "rank_p50_median_us": statistics.median(
                item["p50_us"] for item in active_results
            ),
        }
        append_record(output_path, record)
        print("COST_MODEL_SAMPLE " + json.dumps(record, sort_keys=True), flush=True)
    dist.barrier()


def main() -> None:
    mode = os.environ["CALIBRATION_MODE"]
    gate = os.environ.get("CALIBRATION_GATE") == "1"
    if mode not in MODE_COLLECTIVES:
        raise ValueError(f"unknown calibration mode: {mode}")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size == 8:
        store_path = f"/tmp/{os.environ['JOB_ID']}.{mode}.gloo"
        dist.init_process_group(
            "gloo",
            init_method=f"file://{store_path}",
            rank=int(os.environ["RANK"]),
            world_size=world_size,
        )
    else:
        dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    nodes = world_size // 8
    output_root = Path(os.environ["DUMP_DIR"])
    if rank == 0:
        output_root.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": 1,
            "profile": "h100_nvswitch_roce_400g",
            "job_id": os.environ.get("JOB_ID"),
            "replicate": int(os.environ["CALIBRATION_REPLICATE"]),
            "gate": gate,
            "mode": mode,
            "hostname": socket.gethostname(),
            "nodes": nodes,
            "world_size": world_size,
            "gpu": torch.cuda.get_device_name(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "nccl_version": torch.cuda.nccl.version(),
            "nccl_algo": os.environ.get("NCCL_ALGO"),
            "nccl_proto": os.environ.get("NCCL_PROTO"),
            "started_unix": time.time(),
        }
        (output_root / f"metadata.{mode}.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
    dist.barrier()

    topologies: tuple[tuple[int, int], ...]
    if nodes == 1:
        topologies = ((1, 2), (1, 4), (1, 8))
    else:
        topologies = tuple((nodes, ppn) for ppn in (1, 2, 4, 8))
    sizes = GATE_MESSAGE_BYTES if gate else FORMAL_MESSAGE_BYTES
    output_path = output_root / f"measurements.{mode}.jsonl"

    for topology_nodes, ppn in topologies:
        if mode == "nvlstree_simple" and (topology_nodes == 1 or ppn < 4):
            continue
        execution_modes = (
            ("isolated", "concurrent") if mode == "auto" else ("isolated",)
        )
        for execution in execution_modes:
            if execution == "isolated":
                group, ranks = make_isolated_group(topology_nodes, ppn)
                active = rank in ranks
            else:
                group, ranks = make_concurrent_group(topology_nodes, ppn)
                active = True
            for collective in MODE_COLLECTIVES[mode]:
                for n_bytes in sizes:
                    benchmark_case(
                        mode=mode,
                        execution=execution,
                        nodes=topology_nodes,
                        ppn=ppn,
                        collective=collective,
                        n_bytes=n_bytes,
                        dtype=torch.float32,
                        gate=gate,
                        output_path=output_path,
                        group=group,
                        ranks=ranks,
                        active=active,
                    )
                if mode == "auto" and collective != "all_to_all":
                    for n_bytes in WORKLOAD_MESSAGE_BYTES:
                        benchmark_case(
                            mode=mode,
                            execution=execution,
                            nodes=topology_nodes,
                            ppn=ppn,
                            collective=collective,
                            n_bytes=n_bytes,
                            dtype=torch.bfloat16,
                            gate=gate,
                            output_path=output_path,
                            group=group,
                            ranks=ranks,
                            active=active,
                        )

    if rank == 0:
        (output_root / f"SUCCESS.{mode}").write_text("passed\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
