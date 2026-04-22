# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard

from autoparallel.api import auto_parallel


class LinearModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.linear(x)


class TwoOutputModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.a = nn.Linear(dim, dim, bias=False)
        self.b = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.a(x), self.b(x)


def _auto_parallel_with_internals(model, mesh, sample_inputs, out_shardings):
    """Like auto_parallel but also returns the AutoParallel instance."""
    from autoparallel.api import (
        AutoParallel,
        _extract_input_info,
        _flatten_out_shardings,
        _make_input_fn,
    )

    shapes, dtypes, input_placements, treespec, devices = _extract_input_info(
        sample_inputs, mesh
    )
    output_placements = _flatten_out_shardings(out_shardings)
    input_fn = _make_input_fn(shapes, dtypes, treespec, devices)

    with AutoParallel(model, input_fn, mesh) as autop:
        autop.add_input_constraints(input_placements)
        autop.add_output_constraints(output_placements)
        sharding_placement = autop.optimize_placement(verbose=False)
        parallel_model = autop.apply_placement(sharding_placement)

    return parallel_model, autop


def _make_dtensor_input(dim, mesh):
    local_bs = max(1, 32 // mesh.size())
    return DTensor.from_local(
        torch.rand(local_bs, dim, device="cuda"), mesh, [Shard(0)]
    )


# ---------------------------------------------------------------------------
# extract_forward_graph unit tests
# ---------------------------------------------------------------------------


def test_extract_forward_removes_tangent_placeholders(device_mesh_1d):
    """Forward-only graph has no tangent placeholders."""
    from autoparallel.graph_passes.extract_forward import extract_forward_graph

    dim = 128
    with torch.device("meta"):
        model = LinearModel(dim)

    x = _make_dtensor_input(dim, device_mesh_1d)
    _, autop = _auto_parallel_with_internals(model, device_mesh_1d, (x,), (Shard(0),))

    joint_gm = autop.parallel_gm
    fw_metadata = autop.joint_with_descriptors._aot_state.fw_metadata
    num_fwd_outputs = fw_metadata.num_forward_returns

    # Sanity: joint graph should have tangent placeholders
    joint_tangents = [
        n
        for n in joint_gm.graph.nodes
        if n.op == "placeholder" and "tangents" in n.target
    ]
    assert len(joint_tangents) > 0

    fwd_gm = extract_forward_graph(joint_gm, num_fwd_outputs)

    fwd_tangents = [
        n
        for n in fwd_gm.graph.nodes
        if n.op == "placeholder" and "tangents" in n.target
    ]
    assert len(fwd_tangents) == 0


def test_extract_forward_output_count(device_mesh_1d):
    """Forward-only graph output has exactly num_fwd_outputs entries."""
    from autoparallel.graph_passes.extract_forward import extract_forward_graph

    dim = 128
    with torch.device("meta"):
        model = TwoOutputModel(dim)

    x = _make_dtensor_input(dim, device_mesh_1d)
    _, autop = _auto_parallel_with_internals(
        model, device_mesh_1d, (x,), ((Shard(0),), (Shard(0),))
    )

    joint_gm = autop.parallel_gm
    fw_metadata = autop.joint_with_descriptors._aot_state.fw_metadata
    num_fwd_outputs = fw_metadata.num_forward_returns

    fwd_gm = extract_forward_graph(joint_gm, num_fwd_outputs)

    output_node = next(n for n in fwd_gm.graph.nodes if n.op == "output")
    fwd_outputs = output_node.args[0]
    assert len(fwd_outputs) == num_fwd_outputs


def test_extract_forward_fewer_nodes(device_mesh_1d):
    """Forward-only graph should have strictly fewer nodes than joint graph."""
    from autoparallel.graph_passes.extract_forward import extract_forward_graph

    dim = 128
    with torch.device("meta"):
        model = LinearModel(dim)

    x = _make_dtensor_input(dim, device_mesh_1d)
    _, autop = _auto_parallel_with_internals(model, device_mesh_1d, (x,), (Shard(0),))

    joint_gm = autop.parallel_gm
    fw_metadata = autop.joint_with_descriptors._aot_state.fw_metadata
    num_fwd_outputs = fw_metadata.num_forward_returns

    fwd_gm = extract_forward_graph(joint_gm, num_fwd_outputs)

    joint_count = sum(1 for _ in joint_gm.graph.nodes)
    fwd_count = sum(1 for _ in fwd_gm.graph.nodes)
    assert fwd_count < joint_count, (
        f"Forward graph ({fwd_count} nodes) should be smaller than "
        f"joint graph ({joint_count} nodes)"
    )


def test_extract_forward_does_not_mutate_original(device_mesh_1d):
    """extract_forward_graph should not modify the original joint graph."""
    from autoparallel.graph_passes.extract_forward import extract_forward_graph

    dim = 128
    with torch.device("meta"):
        model = LinearModel(dim)

    x = _make_dtensor_input(dim, device_mesh_1d)
    _, autop = _auto_parallel_with_internals(model, device_mesh_1d, (x,), (Shard(0),))

    joint_gm = autop.parallel_gm
    fw_metadata = autop.joint_with_descriptors._aot_state.fw_metadata
    num_fwd_outputs = fw_metadata.num_forward_returns

    original_node_count = sum(1 for _ in joint_gm.graph.nodes)
    _ = extract_forward_graph(joint_gm, num_fwd_outputs)
    assert sum(1 for _ in joint_gm.graph.nodes) == original_node_count


# ---------------------------------------------------------------------------
# Inference calling convention tests
# ---------------------------------------------------------------------------


def test_inference_fn_produces_output(device_mesh_1d):
    """Forward under no_grad should use the inference path and produce output."""
    dim = 128
    with torch.device("meta"):
        model = LinearModel(dim)

    x = _make_dtensor_input(dim, device_mesh_1d)
    parallel_model = auto_parallel(
        model,
        device_mesh_1d,
        sample_inputs=(x,),
        out_shardings=(Shard(0),),
    )
    parallel_model.to_empty(device="cuda")
    nn.init.ones_(parallel_model.linear.weight)

    with torch.no_grad():
        out = parallel_model(x.to_local())

    assert out is not None
    assert isinstance(out, torch.Tensor), f"Expected Tensor, got {type(out)}"


def test_training_and_inference_same_output(device_mesh_1d):
    """Training and inference paths should produce identical forward outputs."""
    dim = 128
    with torch.device("meta"):
        model = LinearModel(dim)

    x = _make_dtensor_input(dim, device_mesh_1d)
    parallel_model = auto_parallel(
        model,
        device_mesh_1d,
        sample_inputs=(x,),
        out_shardings=(Shard(0),),
    )
    parallel_model.to_empty(device="cuda")
    nn.init.ones_(parallel_model.linear.weight)

    local_x = x.to_local()

    train_out = parallel_model(local_x)
    with torch.no_grad():
        infer_out = parallel_model(local_x)

    assert type(train_out) is type(
        infer_out
    ), f"Output types differ: train={type(train_out)}, infer={type(infer_out)}"
    assert torch.equal(train_out, infer_out)


def test_inference_no_grad_on_output(device_mesh_1d):
    """Inference path output should not require grad."""
    dim = 128
    with torch.device("meta"):
        model = LinearModel(dim)

    x = _make_dtensor_input(dim, device_mesh_1d)
    parallel_model = auto_parallel(
        model,
        device_mesh_1d,
        sample_inputs=(x,),
        out_shardings=(Shard(0),),
    )
    parallel_model.to_empty(device="cuda")
    nn.init.ones_(parallel_model.linear.weight)

    with torch.no_grad():
        out = parallel_model(x.to_local())

    assert isinstance(out, torch.Tensor), f"Expected Tensor, got {type(out)}"
    assert not out.requires_grad


def test_multi_output_inference(device_mesh_1d):
    """Inference path works with multiple outputs and pytree unflattening."""
    dim = 128
    with torch.device("meta"):
        model = TwoOutputModel(dim)

    x = _make_dtensor_input(dim, device_mesh_1d)
    parallel_model = auto_parallel(
        model,
        device_mesh_1d,
        sample_inputs=(x,),
        out_shardings=((Shard(0),), (Shard(0),)),
    )
    parallel_model.to_empty(device="cuda")
    nn.init.ones_(parallel_model.a.weight)
    nn.init.ones_(parallel_model.b.weight)

    local_x = x.to_local()

    train_out = parallel_model(local_x)
    assert isinstance(train_out, tuple)
    assert len(train_out) == 2

    with torch.no_grad():
        infer_out = parallel_model(local_x)

    assert isinstance(infer_out, tuple), f"Expected tuple, got {type(infer_out)}"
    assert len(infer_out) == 2
    assert torch.equal(train_out[0], infer_out[0])
    assert torch.equal(train_out[1], infer_out[1])
