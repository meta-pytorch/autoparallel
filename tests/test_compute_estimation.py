# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import OpSpec
from torch.distributed.tensor.placement_types import Replicate, Shard

import autoparallel.cost_models.compute_estimation as compute_estimation
from autoparallel.cost_models.compute_estimation import (
    DEVICE_LIMITS,
    _get_device_limit,
    estimate_strategy_runtime_cost,
    override_cost_model_device_name,
)


@pytest.mark.parametrize(
    "device_name,expected",
    [
        ("NVIDIA H100 80GB HBM3", "H100"),
        ("NVIDIA H200", "H200"),
        ("NVIDIA GH200 480GB", "GH200"),
        ("NVIDIA H800", "H800"),
        ("NVIDIA B200", "B200"),
        ("NVIDIA GB200", "B200"),
        ("NVIDIA B300 SXM6 AC", "B300"),
        ("NVIDIA GB300", "GB300"),
        ("NVIDIA L40S", "L40S"),
        ("NVIDIA PG509-210", "A100"),
    ],
)
def test_device_name_resolves_to_limit(device_name, expected):
    with override_cost_model_device_name(device_name):
        assert _get_device_limit().name == expected


def test_unknown_device_raises():
    with override_cost_model_device_name("NVIDIA RTX 4090"):
        with pytest.raises(ValueError, match="Unsupported device"):
            _get_device_limit()


def test_h100_supports_fp8_compute_estimation():
    h100 = next(limit for limit in DEVICE_LIMITS if limit.name == "H100")

    assert h100.gemm_tflops[torch.float8_e4m3fn] == 1979
    assert h100.gemm_tflops[torch.float8_e5m2] == 1979


def test_b200_supports_fp8_compute_estimation():
    b200 = next(limit for limit in DEVICE_LIMITS if limit.name == "B200")

    assert b200.gemm_tflops[torch.float8_e4m3fn] == 4500
    assert b200.gemm_tflops[torch.float8_e5m2] == 4500


def test_estimate_fp8_mm(monkeypatch):
    h100 = next(limit for limit in DEVICE_LIMITS if limit.name == "H100")
    monkeypatch.setattr(compute_estimation, "_get_device_limit", lambda: h100)

    graph = torch.fx.Graph()
    lhs = graph.placeholder("lhs")
    lhs.meta["val"] = torch.empty(128, 256, dtype=torch.float8_e4m3fn, device="meta")
    rhs = graph.placeholder("rhs")
    rhs.meta["val"] = torch.empty(256, 64, dtype=torch.float8_e4m3fn, device="meta")
    mm = graph.call_function(torch.ops.aten.mm.default, (lhs, rhs))

    assert estimate_strategy_runtime_cost(mm, None) > 0


def test_sharded_factory_op_is_priced_at_local_size(monkeypatch):
    h100 = next(limit for limit in DEVICE_LIMITS if limit.name == "H100")
    monkeypatch.setattr(compute_estimation, "_get_device_limit", lambda: h100)

    def full_node(shape):
        graph = torch.fx.Graph()
        node = graph.call_function(
            torch.ops.aten.full.default,
            (shape, 0.0),
            {"dtype": torch.bfloat16, "device": torch.device("meta")},
        )
        node.meta["val"] = torch.empty(shape, dtype=torch.bfloat16, device="meta")
        return node

    global_shape = [65536, 4096]
    mesh = DeviceMesh("cpu", list(range(64)), _init_backend=False)
    tensor_meta = TensorMeta(
        torch.Size(global_shape), (global_shape[1], 1), torch.bfloat16
    )

    def cost(placement):
        spec = DTensorSpec(mesh, (placement,), tensor_meta=tensor_meta)
        return estimate_strategy_runtime_cost(
            full_node(global_shape), OpSpec(output_specs=spec, input_specs=[spec])
        )

    assert cost(Replicate()) == estimate_strategy_runtime_cost(
        full_node(global_shape), None
    )
    assert cost(Shard(0)) == estimate_strategy_runtime_cost(
        full_node([1024, 4096]), None
    )
    assert cost(Shard(0)) < cost(Replicate())


def _metadata_only_cost_nodes():
    shape = [4096, 16384]
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty(shape, dtype=torch.bfloat16, device="meta")

    def call(target, args, kwargs=None):
        node = graph.call_function(target, args, kwargs or {})
        node.meta["val"] = torch.empty(shape, dtype=torch.bfloat16, device="meta")
        return node

    factory_kwargs = {"dtype": torch.bfloat16, "device": torch.device("meta")}
    unsafe_view = call(torch.ops.aten._unsafe_view.default, (x, shape))
    empties = [
        call(torch.ops.aten.empty.memory_format, (shape,), factory_kwargs),
        call(
            torch.ops.aten.empty_strided.default, (shape, [shape[1], 1]), factory_kwargs
        ),
        call(torch.ops.aten.empty_like.default, (x,)),
    ]
    clone = call(torch.ops.aten.clone.default, (x,))
    full = call(torch.ops.aten.full.default, (shape, 0.0), factory_kwargs)
    return unsafe_view, empties, [clone, full]


def test_metadata_only_ops_are_free(monkeypatch):
    h100 = next(limit for limit in DEVICE_LIMITS if limit.name == "H100")
    monkeypatch.setattr(compute_estimation, "_get_device_limit", lambda: h100)
    unsafe_view, empties, memory_ops = _metadata_only_cost_nodes()

    for node in [unsafe_view, *empties]:
        assert estimate_strategy_runtime_cost(node, None) == 0, node.target
    for node in memory_ops:
        assert estimate_strategy_runtime_cost(node, None) > 0, node.target


def test_empty_factories_are_priced_when_deterministic_fill_is_enabled(monkeypatch):
    h100 = next(limit for limit in DEVICE_LIMITS if limit.name == "H100")
    monkeypatch.setattr(compute_estimation, "_get_device_limit", lambda: h100)
    monkeypatch.setattr(torch.utils.deterministic, "fill_uninitialized_memory", True)
    unsafe_view, empties, _ = _metadata_only_cost_nodes()

    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        assert estimate_strategy_runtime_cost(unsafe_view, None) == 0
        for node in empties:
            assert estimate_strategy_runtime_cost(node, None) > 0, node.target
    finally:
        torch.use_deterministic_algorithms(previous)

    monkeypatch.setattr(torch.utils.deterministic, "fill_uninitialized_memory", False)
    torch.use_deterministic_algorithms(True)
    try:
        for node in empties:
            assert estimate_strategy_runtime_cost(node, None) == 0, node.target
    finally:
        torch.use_deterministic_algorithms(previous)
