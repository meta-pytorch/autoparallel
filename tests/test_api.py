# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.fx.traceback as fx_traceback
from torch import nn
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel.api import AutoParallel


@pytest.fixture(scope="module", autouse=True)
def init_pg():
    world_size = 256
    fake_store = FakeStore()
    if torch.distributed.is_initialized():
        return
    torch.distributed.init_process_group(
        "fake", store=fake_store, rank=0, world_size=world_size
    )


@pytest.fixture(scope="module")
def device_mesh_1d():
    world_size = torch.distributed.get_world_size()
    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda", (world_size,), mesh_dim_names=("dp",)
    )
    return mesh


def test_from_meta_model(device_mesh_1d):
    class Model(nn.Module):
        def __init__(self, dim1):
            super().__init__()
            self.linear = nn.Linear(dim1, dim1)
            self.register_buffer("buf", torch.rand(1))

        def forward(self, x, y):
            return y + 2, self.linear(x) * self.buf, x + y + self.buf

    dim = 128
    with torch.device("meta"):
        model = Model(dim)

    def input_fn():
        b = 32
        inputs = (
            torch.rand(b, dim, device="cuda"),
            torch.rand(b, 1, device="cuda"),
        )
        return inputs

    auto_p = AutoParallel(
        model,
        input_fn,
        device_mesh_1d,
    )
    assert isinstance(
        auto_p.model.get_parameter("linear.weight"), torch._subclasses.FakeTensor
    )
    assert isinstance(auto_p.model.get_buffer("buf"), torch._subclasses.FakeTensor)


def test_init(device_mesh_1d):
    dim = 128

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            self.register_buffer("buf", torch.empty(dim))

        def forward(self, x):
            return self.linear(x) + self.buf

        def init_weights(self):
            self.linear.weight = torch.nn.Parameter(torch.ones(dim, dim) * 9.0)
            with torch.no_grad():
                self.linear.bias.fill_(98.6)
            self.buf = torch.arange(dim)

    def input_fn():
        b = 512
        inputs = (torch.rand(b, dim, device="cuda"),)
        return inputs

    with torch.device("meta"):
        model = Model(dim)
    with AutoParallel(
        model,
        input_fn,
        device_mesh_1d,
    ) as autop:
        x_sharding = (Shard(0),)
        autop.add_input_constraints([x_sharding])
        sharding_placement = autop.optimize_placement()

        # AutoParallel produces a module with meta-DTensor parameters that need to be initialized
        parallel_mod = autop.apply_placement(sharding_placement)
    parallel_mod.to_empty(device="cuda")
    parallel_mod.init_weights()
    assert torch.equal(
        parallel_mod.get_parameter("linear.weight").full_tensor(),
        torch.full((dim, dim), 9.0, device="cuda"),
    )
    assert torch.equal(
        parallel_mod.get_parameter("linear.bias").full_tensor(),
        torch.full((dim,), 98.6, device="cuda"),
    )
    assert torch.equal(
        parallel_mod.get_buffer("buf").full_tensor(), torch.arange(dim, device="cuda")
    )


def test_fx_graph_annotate(device_mesh_1d):
    dim = 128

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.a = nn.Linear(dim, dim, bias=False)
            self.b = nn.Linear(dim, dim, bias=False)
            self.c = nn.Linear(dim, dim, bias=False)
            self.d = nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            with fx_traceback.annotate({"outer": 0}):
                with fx_traceback.annotate({"inner": 0}):
                    a = self.a(x)
                with fx_traceback.annotate({"inner": 1}):
                    b = self.b(a)
                with fx_traceback.annotate({"inner": 2}):
                    c = self.c(b)
                with fx_traceback.annotate({"inner": 3}):
                    d = self.d(c)
            return d

    def input_fn():
        b = 512
        inputs = (torch.rand(b, dim, device="cuda"),)
        return inputs

    with torch.device("meta"):
        model = Model(dim)

    with fx_traceback.preserve_node_meta(), AutoParallel(
        model,
        input_fn,
        device_mesh_1d,
    ) as autop:
        x_sharding = (Shard(0),)
        autop.add_input_constraints([x_sharding])
        sharding_placement = autop.optimize_placement()

        # AutoParallel produces a module with meta-DTensor parameters that need to be initialized
        _ = autop.apply_placement(sharding_placement)

    graph = autop.parallel_gm.graph

    # 4 linear -> 4 mm ops
    fw_seen_annotations = set()
    bw_seen_annotations = set()
    for mm in [n for n in graph.nodes if "mm" in n.name]:
        assert mm.meta["custom"]["outer"] == 0
        assert "inner" in mm.meta["custom"]
        if mm.meta.get("partitioner_tag", "") == "is_backward":
            bw_seen_annotations.add(mm.meta["custom"]["inner"])
        else:
            fw_seen_annotations.add(mm.meta["custom"]["inner"])
    assert fw_seen_annotations == bw_seen_annotations == {0, 1, 2, 3}

    for ph in graph.find_nodes(op="placeholder"):
        assert (
            "custom" not in ph.meta
        ), "Placeholders didn't have have custom metadata before"
    for out in graph.find_nodes(op="output"):
        assert (
            "custom" not in out.meta
        ), "Output didn't have have custom metadata before"

    # NOTE: The tests below are just to prevent semantics from changing silently.
    # Currently, custom metadata is not set for:
    # - graph inputs
    # - graph outputs
    # - collectives/waits added by AP
    for node in graph.nodes:
        if node.meta.get("custom", None) is None:
            assert (
                node.op == "placeholder"
                or node.op == "output"
                or node.target.namespace == "_c10d_functional"
            )


def test_fx_graph_annotate_overlap_pass(device_mesh_1d):
    class DummyOp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, scalar):
            ctx.save_for_backward(x)
            return x + scalar

        @staticmethod
        def backward(ctx, grad_out):
            return grad_out, None

    def mock_fw_compute(x):
        with fx_traceback.annotate({"compute": 0}):
            return DummyOp.apply(x, 10)

    def mock_bw_comm(x):
        with fx_traceback.annotate({"comm": 0}):
            return DummyOp.apply(x, 20)

    def mock_bw_compute(x):
        return DummyOp.apply(x, 30)

    class Model(nn.Module):
        def forward(self, fw_in, bw_in):
            fw_out = mock_fw_compute(fw_in)
            # bw_in blocks bw_out
            bw_in = mock_bw_comm(bw_in)
            bw_out = mock_bw_compute(bw_in)
            return fw_out, bw_out

    def input_fn():
        inputs = (torch.rand(2, 128, device="cuda", requires_grad=True),)
        grad_ins = (torch.rand(2, 128, device="cuda"),)
        return (
            *inputs,
            *grad_ins,
        )

    with torch.device("meta"):
        model = Model()

    with fx_traceback.preserve_node_meta(), AutoParallel(
        model,
        input_fn,
        device_mesh_1d,
    ) as autop:
        autop.add_input_constraints(
            [
                (Replicate(),),
                (Replicate(),),
            ]
        )
        autop.add_output_constraints(
            [
                (Replicate(),),
                (Replicate(),),
            ]
        )
        sharding_placement = autop.optimize_placement()

        # AutoParallel produces a module with meta-DTensor parameters that need to be initialized
        _ = autop.apply_placement(sharding_placement)

    graph = autop.parallel_gm.graph

    # At this point, the graph looks like:
    # graph():
    #     %primals_1 : [num_users=1] = placeholder[target=primals_1]
    #     %primals_2 : [num_users=1] = placeholder[target=primals_2]
    #     %tangents_1 : [num_users=1] = placeholder[target=tangents_1]
    #     %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_1, 10), kwargs = {})
    #     %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_2, 20), kwargs = {})
    #     %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, 30), kwargs = {})
    #     return ((add, add_2), (tangents_1, None))

    compute_nodes = {
        n for n in graph.nodes if n.meta.get("custom", {}).get("compute", None) == 0
    }
    comm_nodes = [
        n for n in graph.nodes if n.meta.get("custom", {}).get("comm", None) == 0
    ]
    assert len(compute_nodes) == 1
    assert len(comm_nodes) == 1

    # move comm nodes before compute nodes
    first_compute_node = None
    for n in graph.nodes:
        if n in compute_nodes:
            first_compute_node = n
            break

    assert first_compute_node is not None
    for node in reversed(comm_nodes):
        first_compute_node.prepend(node)

    # After pass, add_1 (comm) should be before add (compute)
    node_names = [n.name for n in graph.nodes]
    assert node_names.index("add_1") == node_names.index("add") - 1

    # The graph looks like:
    # graph():
    #     %primals_1 : [num_users=1] = placeholder[target=primals_1]
    #     %primals_2 : [num_users=1] = placeholder[target=primals_2]
    #     %tangents_1 : [num_users=1] = placeholder[target=tangents_1]
    #     %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_2, 20), kwargs = {})
    #     %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_1, 10), kwargs = {})
    #     %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, 30), kwargs = {})
    #     return ((add, add_2), (tangents_1, None))
