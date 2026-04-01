# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.fx.experimental.proxy_tensor import make_fx

from autoparallel.api import AutoParallel
from autoparallel.graph_passes.debug_helpers import (
    _is_communication_node,
    make_custom_runtime_estimation,
)
from autoparallel.graph_passes.estimate_graph_metrics import estimate_graph_metrics


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


def test_with_autoparallel(device_mesh_1d):
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

    with AutoParallel(model, input_fn, device_mesh_1d) as autop:
        autop.add_input_constraints([(Shard(0),)])
        autop.add_output_constraints([(Shard(0),)])
        sharding_placement = autop.optimize_placement()
        _ = autop.apply_placement(sharding_placement)

    estimator = make_custom_runtime_estimation(device_mesh_1d)
    metrics = estimate_graph_metrics(autop.parallel_gm, estimator)

    assert metrics.total_time > 0
    assert metrics.compute_time >= 0
    assert metrics.peak_memory > 0
    assert metrics.exposed_comm_time <= metrics.communication_time
    assert metrics.exposed_comm_time >= 0

    has_comm = any(_is_communication_node(n) for n in autop.parallel_gm.graph.nodes)
    if has_comm:
        assert metrics.communication_time > 0


def _run_autoparallel(dim, device_mesh_1d, *, model_cls, requires_grad=True):
    """Run AutoParallel on a model and return the graph metrics."""
    with torch.device("meta"):
        model = model_cls()

    if not requires_grad:
        for param in model.parameters():
            param.requires_grad = False

    def input_fn():
        return torch.rand(512, dim, device="cuda")

    with AutoParallel(model, input_fn, device_mesh_1d) as autop:
        autop.add_input_constraints([(Shard(0),)])
        autop.add_output_constraints([(Shard(0),)])
        sharding_placement = autop.optimize_placement()
        _ = autop.apply_placement(sharding_placement)

    estimator = make_custom_runtime_estimation(device_mesh_1d)
    return estimate_graph_metrics(autop.parallel_gm, estimator), autop.parallel_gm


def test_training_vs_inference_metrics(device_mesh_1d):
    """Training mode (with backward) should produce higher metrics than inference."""
    dim = 128

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            return self.linear(x)

    train_metrics, _ = _run_autoparallel(
        dim, device_mesh_1d, model_cls=Model, requires_grad=True
    )
    infer_metrics, _ = _run_autoparallel(
        dim, device_mesh_1d, model_cls=Model, requires_grad=False
    )

    assert train_metrics.total_time > infer_metrics.total_time
    assert train_metrics.compute_time > infer_metrics.compute_time


def test_multi_layer_model(device_mesh_1d):
    """A deeper model should have more compute and communication than a single layer."""
    dim = 128

    class SingleLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(dim, dim)

        def forward(self, x):
            return self.linear(x)

    class ThreeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(dim, dim)
            self.linear2 = nn.Linear(dim, dim)
            self.linear3 = nn.Linear(dim, dim)

        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            return self.linear3(x)

    single_metrics, _ = _run_autoparallel(dim, device_mesh_1d, model_cls=SingleLayer)
    triple_metrics, triple_gm = _run_autoparallel(
        dim, device_mesh_1d, model_cls=ThreeLayer
    )

    assert triple_metrics.compute_time > single_metrics.compute_time
    assert triple_metrics.peak_memory > 0

    num_comm = sum(1 for n in triple_gm.graph.nodes if _is_communication_node(n))
    assert num_comm > 0
    assert triple_metrics.communication_time > 0


def test_inference_mode(device_mesh_1d):
    """Forward-only graph should have no tangent placeholders and valid metrics."""
    dim = 128

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            return self.linear(x)

    metrics, gm = _run_autoparallel(
        dim, device_mesh_1d, model_cls=Model, requires_grad=False
    )

    assert metrics.total_time > 0
    assert metrics.compute_time > 0
    assert metrics.peak_memory >= 0

    # No backward pass: only weight + input placeholders, no tangents
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    assert len(placeholders) == 2


def test_guaranteed_communication(device_mesh_1d):
    """Training mode with sharded data must produce gradient collectives."""
    dim = 128

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(dim, dim)

        def forward(self, x):
            return self.linear(x)

    metrics, gm = _run_autoparallel(dim, device_mesh_1d, model_cls=Model)

    num_comm = sum(1 for n in gm.graph.nodes if _is_communication_node(n))
    assert num_comm > 0, "Training with sharded data should require collectives"
    assert metrics.communication_time > 0
    assert metrics.exposed_comm_time >= 0
    assert metrics.exposed_comm_time <= metrics.communication_time


def test_comparison_fsdp_tp_vs_ddp(device_mesh_2d):
    """FSDP+TP vs DDP on the same model should produce different metrics."""
    dim = 128

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(dim, dim * 4, bias=False)
            self.linear2 = nn.Linear(dim * 4, dim, bias=False)

        def forward(self, x):
            return self.linear2(self.linear1(x))

    placement = (Shard(0), Replicate())

    def get_metrics(high_mem):
        with torch.device("meta"):
            model = Model()

        def input_fn():
            return torch.rand(512, dim, device="cuda")

        with AutoParallel(model, input_fn, device_mesh_2d) as autop:
            autop.add_input_constraints([placement])
            autop.add_output_constraints([placement])
            autop.add_parameter_memory_constraint(low=0, high=high_mem)
            sharding_placement = autop.optimize_placement()
            _ = autop.apply_placement(sharding_placement)

        estimator = make_custom_runtime_estimation(device_mesh_2d)
        return estimate_graph_metrics(autop.parallel_gm, estimator)

    # high_mem=None → forces parameter sharding (FSDP+TP)
    # high_mem=1.0  → allows replicated parameters (DDP)
    metrics_fsdp_tp = get_metrics(high_mem=None)
    metrics_ddp = get_metrics(high_mem=1.0)

    assert metrics_fsdp_tp.total_time > 0
    assert metrics_ddp.total_time > 0
    assert metrics_fsdp_tp != metrics_ddp
