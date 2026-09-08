from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path

import torch


def datacenter(host: str) -> str:
    labels = host.rstrip(".").split(".")
    if labels[-2:] == ["facebook", "com"]:
        labels = labels[:-2]
    if len(labels) < 2:
        raise RuntimeError(f"cannot determine datacenter from {host!r}")
    return labels[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase")
    args = parser.parse_args()

    expected_nodes = int(os.environ["EXPECTED_NODE_COUNT"])
    expected_world_size = int(os.environ["EXPECTED_WORLD_SIZE"])
    expected_local_world_size = int(os.environ["EXPECTED_LOCAL_WORLD_SIZE"])
    hosts = [
        host.strip()
        for host in os.environ["MAST_HPC_TASK_GROUP_HOSTNAMES"].split(",")
        if host.strip()
    ]
    if len(hosts) != expected_nodes or len(set(hosts)) != expected_nodes:
        raise RuntimeError(
            f"expected {expected_nodes} unique hosts, got {hosts!r}"
        )
    if {datacenter(host) for host in hosts} != {os.environ["REQUIRED_PCI_DOMAIN"]}:
        raise RuntimeError(f"unexpected allocation locality: {hosts!r}")

    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    if (world_size, local_world_size) != (
        expected_world_size,
        expected_local_world_size,
    ):
        raise RuntimeError(
            "unexpected world/local sizes: "
            f"{world_size}/{local_world_size}, expected "
            f"{expected_world_size}/{expected_local_world_size}"
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    properties = torch.cuda.get_device_properties(local_rank)
    if "H100" not in properties.name:
        raise RuntimeError(f"expected H100, got {properties.name!r}")

    payload = {
        "phase": args.phase,
        "hostname": socket.gethostname(),
        "hosts": hosts,
        "datacenter": os.environ["REQUIRED_PCI_DOMAIN"],
        "job_id": os.environ.get("JOB_ID"),
        "rank": int(os.environ["RANK"]),
        "local_rank": local_rank,
        "world_size": world_size,
        "local_world_size": local_world_size,
        "gpu_ordinal": local_rank,
        "gpu_name": properties.name,
        "gpu_total_memory_bytes": properties.total_memory,
        "gpu_compute_capability": [properties.major, properties.minor],
        "gpu_pci_bus_id": properties.pci_bus_id,
        "gpu_uuid": str(properties.uuid),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_network_id": os.environ.get("DEVICE_NETWORK_ID"),
        "device_backend_network_topology": os.environ.get(
            "DEVICE_BACKEND_NETWORK_TOPOLOGY"
        ),
    }
    output = Path(os.environ["RUN_ROOT"]) / "allocation" / args.phase
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
    reference_dir = Path(os.environ["RUN_ROOT"]) / "runtime" / "allocation_reference"
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
                "allocation or rank-to-GPU mapping changed between phases: "
                f"{json.dumps(differences, sort_keys=True)}"
            )
    else:
        reference_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
