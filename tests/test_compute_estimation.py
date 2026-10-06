# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch

import autoparallel.cost_models.compute_estimation as compute_estimation
from autoparallel.cost_models.compute_estimation import (
    DEVICE_LIMITS,
    estimate_strategy_runtime_cost,
)


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
