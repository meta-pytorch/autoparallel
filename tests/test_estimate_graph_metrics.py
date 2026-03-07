# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import nn
from torch.distributed.tensor.placement_types import Shard
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel.api import AutoParallel
from autoparallel.graph_passes.debug_helpers import (
    _is_communication_node,
    make_custom_runtime_estimation,
)
from autoparallel.graph_passes.estimate_graph_metrics import estimate_graph_metrics


@pytest.fixture(scope="module", autouse=True)
def init_pg():
    if torch.distributed.is_initialized():
        return
    fake_store = FakeStore()
    torch.distributed.init_process_group(
        "fake", store=fake_store, rank=0, world_size=256
    )


@pytest.fixture(scope="module")
def device_mesh():
    world_size = torch.distributed.get_world_size()
    return torch.distributed.device_mesh.init_device_mesh(
        "cuda", (world_size,), mesh_dim_names=("dp",)
    )


def _constant_estimator(dur):
    """Return an estimator that gives `dur` for every call_function node, 0 otherwise."""

    def estimator(node):
        return dur if node.op == "call_function" else 0

    return estimator


def test_compute_only_graph():
    def f(a, b):
        return torch.mm(a, b)

    gm = make_fx(f, tracing_mode="fake")(torch.randn(4, 8), torch.randn(8, 16))
    metrics = estimate_graph_metrics(gm, _constant_estimator(100))

    assert metrics.compute_time == 100.0
    assert metrics.communication_time == 0
    assert metrics.exposed_comm_time == 0
    assert metrics.total_time > metrics.compute_time
    assert metrics.peak_memory >= 0


def test_compute_time_accumulates():
    def f(a, b, c):
        x = torch.mm(a, b)
        x = torch.relu(x)
        return torch.mm(x, c)

    gm = make_fx(f, tracing_mode="fake")(
        torch.randn(4, 8), torch.randn(8, 16), torch.randn(16, 32)
    )

    num_call_function = sum(1 for n in gm.graph.nodes if n.op == "call_function")
    assert num_call_function == 3

    metrics = estimate_graph_metrics(gm, _constant_estimator(50))
    assert metrics.compute_time == 50 * 3


def test_deterministic():
    def f(a, b):
        return torch.mm(a, b)

    gm = make_fx(f, tracing_mode="fake")(torch.randn(4, 8), torch.randn(8, 16))
    estimator = _constant_estimator(100)

    m1 = estimate_graph_metrics(gm, estimator)
    m2 = estimate_graph_metrics(gm, estimator)
    assert m1 == m2


def test_with_autoparallel(device_mesh):
    dim = 128

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(dim, dim)

        def forward(self, x):
            return self.linear(x)

    def input_fn():
        return (torch.rand(512, dim, device="cuda"),)

    with torch.device("meta"):
        model = Model()

    with AutoParallel(model, input_fn, device_mesh) as autop:
        autop.add_input_constraints([(Shard(0),)])
        autop.add_output_constraints([(Shard(0),)])
        sharding_placement = autop.optimize_placement()
        _ = autop.apply_placement(sharding_placement)

    estimator = make_custom_runtime_estimation(device_mesh)
    metrics = estimate_graph_metrics(autop.parallel_gm, estimator)

    assert metrics.total_time > 0
    assert metrics.compute_time >= 0
    assert metrics.peak_memory > 0
    assert metrics.exposed_comm_time <= metrics.communication_time
    assert metrics.exposed_comm_time >= 0

    has_comm = any(_is_communication_node(n) for n in autop.parallel_gm.graph.nodes)
    if has_comm:
        assert metrics.communication_time > 0
