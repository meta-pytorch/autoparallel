import importlib.util
from pathlib import Path

import pytest

from autoparallel.cost_models.nccl_cost_model import GpuArch, NCCLTopoConfig

SCRIPT = (
    Path(__file__).parents[1]
    / ".agents/skills/autoparallel-optimizer/scripts/modeled_baselines.py"
)
SPEC = importlib.util.spec_from_file_location("modeled_baselines", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
modeled_baselines = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(modeled_baselines)


def plan(forward_compute=10.0, backward_compute=20.0, local_bytes=400):
    return {
        "nodes": [
            {
                "phase": "forward",
                "compute_cost": forward_compute,
                "transition_cost": 2.0,
                "inputs": [{"comm_cost": 3.0}],
            },
            {
                "phase": "backward",
                "compute_cost": backward_compute,
                "transition_cost": 5.0,
                "inputs": [{"comm_cost": 7.0}],
            },
        ],
        "summary": {"parameter_storage": {"local_bytes": local_bytes}},
    }


def topology():
    return NCCLTopoConfig(
        arch=GpuArch.HOPPER,
        num_nodes=1,
        gpus_per_node=8,
        bw_intra=320.0,
        bw_inter=50.0,
        num_channels=16,
        has_nvswitch=True,
        has_collnet=False,
        net_latency=0.0,
    )


def test_pack_buckets_keeps_large_tensor_whole():
    assert modeled_baselines.pack_buckets([4, 7, 3], 10) == [4, 10]
    assert modeled_baselines.pack_buckets([11, 2], 10) == [11, 2]


def test_overlap_envelope_orders_schedule_bounds():
    envelope = modeled_baselines.overlap_envelope(
        {
            "forward": {"compute_us": 10, "communication_us": 4},
            "backward": {"compute_us": 20, "communication_us": 30},
        },
        expected_overlap_fraction=0.5,
        schedule_model="test",
    )
    assert envelope["optimistic"] == 40
    assert envelope["expected"] == 52
    assert envelope["pessimistic"] == 64


def test_ddp_replaces_proxy_communication_with_bucketed_allreduce():
    estimate = modeled_baselines.ddp_estimate(
        plan(),
        [16 * 1024 * 1024, 16 * 1024 * 1024, 4 * 1024 * 1024],
        topology(),
        (8,),
        bucket_capacity_bytes=25 * 1024 * 1024,
    )
    assert estimate["communication"]["bucket_sizes_bytes"] == [
        20 * 1024 * 1024,
        16 * 1024 * 1024,
    ]
    assert estimate["phase_costs"]["forward"]["communication_us"] == 0
    assert estimate["phase_costs"]["backward"]["communication_us"] > 0
    assert (
        estimate["memory"]["modeled_local_training_state_bytes"] == 4 * 36 * 1024 * 1024
    )


def test_fsdp_models_two_allgathers_and_one_reduce_scatter_per_unit():
    estimate = modeled_baselines.fsdp_estimate(
        plan(),
        [32 * 1024 * 1024, 8 * 1024 * 1024],
        topology(),
        (8,),
    )
    communication = estimate["communication"]
    assert communication["wrap_unit_count"] == 2
    assert communication["forward_allgather_us"] == pytest.approx(
        communication["backward_allgather_us"]
    )
    assert communication["backward_reduce_scatter_us"] > 0
    assert estimate["memory"]["modeled_local_training_state_bytes"] == 20 * 1024 * 1024
