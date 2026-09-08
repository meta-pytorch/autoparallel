from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path

import torch


def _datacenter(host: str) -> str:
    labels = host.rstrip(".").split(".")
    if labels[-2:] == ["facebook", "com"]:
        labels = labels[:-2]
    if len(labels) < 2:
        raise RuntimeError(f"cannot determine datacenter from {host!r}")
    return labels[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--nproc-per-node", type=int, required=True)
    parser.add_argument("--locality", required=True)
    parser.add_argument("--gpu-substring", required=True)
    args = parser.parse_args()

    hosts = [
        host.strip()
        for host in os.environ["MAST_HPC_TASK_GROUP_HOSTNAMES"].split(",")
        if host.strip()
    ]
    if len(hosts) != args.nodes or len(set(hosts)) != args.nodes:
        raise RuntimeError(f"expected {args.nodes} unique hosts, got {hosts!r}")
    expected_dc = args.locality.rsplit(";", 1)[-1]
    if {_datacenter(host) for host in hosts} != {expected_dc}:
        raise RuntimeError(f"unexpected allocation locality: {hosts!r}")
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    if (world_size, local_world_size) != (args.world_size, args.nproc_per_node):
        raise RuntimeError(
            f"unexpected world/local sizes {world_size}/{local_world_size}; "
            f"expected {args.world_size}/{args.nproc_per_node}"
        )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    properties = torch.cuda.get_device_properties(local_rank)
    if args.gpu_substring not in properties.name:
        raise RuntimeError(
            f"expected GPU containing {args.gpu_substring!r}, got {properties.name!r}"
        )
    payload = {
        "phase": args.phase,
        "arm": args.arm,
        "hostname": socket.gethostname(),
        "hosts": hosts,
        "datacenter": expected_dc,
        "job_id": os.environ.get("JOB_ID"),
        "rank": int(os.environ["RANK"]),
        "local_rank": local_rank,
        "world_size": world_size,
        "local_world_size": local_world_size,
        "gpu_ordinal": local_rank,
        "gpu_name": properties.name,
        "gpu_total_memory_bytes": properties.total_memory,
        "gpu_compute_capability": [properties.major, properties.minor],
        "gpu_pci_bus_id": getattr(properties, "pci_bus_id", None),
        "gpu_uuid": str(getattr(properties, "uuid", "")),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_network_id": os.environ.get("DEVICE_NETWORK_ID"),
        "device_backend_network_topology": os.environ.get(
            "DEVICE_BACKEND_NETWORK_TOPOLOGY"
        ),
    }
    output = args.run_root / "allocation" / args.phase / args.arm
    output.mkdir(parents=True, exist_ok=True)
    output_path = output / f"rank_{payload['rank']:03d}.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    identity_fields = (
        "hostname",
        "rank",
        "local_rank",
        "world_size",
        "local_world_size",
        "gpu_ordinal",
        "gpu_name",
        "gpu_pci_bus_id",
        "gpu_uuid",
        "cuda_visible_devices",
        "device_network_id",
        "device_backend_network_topology",
    )
    reference_dir = args.run_root / "runtime/allocation_reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    reference_path = reference_dir / output_path.name
    if reference_path.exists():
        reference = json.loads(reference_path.read_text())
        differences = {
            field: {"reference": reference.get(field), "current": payload.get(field)}
            for field in identity_fields
            if reference.get(field) != payload.get(field)
        }
        if differences:
            raise RuntimeError(
                "allocation or rank-to-GPU mapping changed between arms: "
                f"{json.dumps(differences, sort_keys=True)}"
            )
    else:
        reference_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
