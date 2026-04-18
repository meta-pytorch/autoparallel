# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel.api import auto_parallel
from autoparallel.input_validation import (
    _check_forward_args,
    _compute_expected_inputs,
    _extract_input_info,
    _make_input_fn,
)


def test_check_forward_args():
    """Unit test: _check_forward_args catches shape/dtype/type mismatches."""
    expected = [torch.empty(2, 128, device="meta")]

    # Correct inputs
    _check_forward_args((torch.rand(2, 128),), expected)

    # Wrong shape
    with pytest.raises(ValueError, match="shape"):
        _check_forward_args((torch.rand(4, 128),), expected)

    # Wrong dtype
    with pytest.raises(ValueError, match="dtype"):
        _check_forward_args((torch.rand(2, 128, dtype=torch.float16),), expected)

    # Wrong number of arguments
    with pytest.raises(ValueError, match="expected 1"):
        _check_forward_args((torch.rand(2, 128), torch.rand(2, 128)), expected)

    # Non-tensor when tensor expected
    with pytest.raises(TypeError, match="Tensor"):
        _check_forward_args((42,), expected)


def test_check_forward_args_non_tensor_value():
    """Unit test: _check_forward_args validates non-tensor input equality."""
    expected = [42]

    _check_forward_args((42,), expected)

    with pytest.raises(ValueError, match="value"):
        _check_forward_args((99,), expected)


def test_compute_expected_inputs(device_mesh_1d):
    """Test local shape computation from global shape + Shard placement."""
    mesh = device_mesh_1d
    world_size = mesh.size()

    traced = [torch.empty(512, 128, device="meta")]
    placements = [(Shard(0),)]

    result = _compute_expected_inputs(traced, placements, mesh)
    assert len(result) == 1
    assert result[0].shape == (512 // world_size, 128)
    assert result[0].dtype == torch.float32


def test_compute_expected_inputs_replicated(device_mesh_1d):
    """Test that Replicate placement leaves shape unchanged."""
    mesh = device_mesh_1d

    traced = [torch.empty(512, 128, device="meta")]
    placements = [(Replicate(),)]

    result = _compute_expected_inputs(traced, placements, mesh)
    assert len(result) == 1
    assert result[0].shape == (512, 128)


def test_forward_input_validation_integration(device_mesh_1d):
    """Integration test: AutoParallelModule.forward rejects invalid inputs."""
    dim = 128
    batch_size = 512
    local_batch_size = batch_size // device_mesh_1d.size()

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)

        def forward(self, x):
            return self.linear(x)

    with torch.device("meta"):
        model = Model(dim)

    x = DTensor.from_local(
        torch.rand(local_batch_size, dim, device="cuda"),
        device_mesh_1d,
        [Shard(0)],
    )
    parallel_mod = auto_parallel(
        model,
        device_mesh_1d,
        sample_inputs=(x,),
        out_shardings=(Shard(0),),
        compile=False,
    )

    with pytest.raises(ValueError, match="shape"):
        parallel_mod(torch.rand(local_batch_size + 1, dim, device="cuda"))

    with pytest.raises(ValueError, match="dtype"):
        parallel_mod(
            torch.rand(local_batch_size, dim, device="cuda", dtype=torch.float16)
        )

    with pytest.raises(ValueError, match="expected 1"):
        parallel_mod(
            torch.rand(local_batch_size, dim, device="cuda"),
            torch.rand(local_batch_size, dim, device="cuda"),
        )

    with pytest.raises(TypeError, match="Tensor"):
        parallel_mod(42)


def test_compute_expected_inputs_dict_pytree(device_mesh_1d):
    """Test that dict pytree inputs are flattened before applying sharding."""
    mesh = device_mesh_1d
    world_size = mesh.size()

    # Simulate traced_inputs containing a dict (as produced by build_joint_graph
    # when the model takes a dict argument)
    traced = [
        {
            "x": torch.empty(512, 128, device="meta"),
            "y": torch.empty(512, 64, device="meta"),
        }
    ]
    constraints = [(Shard(0),), (Shard(0),)]

    result = _compute_expected_inputs(traced, constraints, mesh)
    assert len(result) == 2
    assert result[0].shape == (512 // world_size, 128)
    assert result[1].shape == (512 // world_size, 64)


def test_extract_input_info_dict(device_mesh_1d):
    """Test that _extract_input_info handles dict inputs."""
    mesh = device_mesh_1d
    sample = {
        "a": torch.rand(4, 8),
        "b": torch.rand(4, 16),
    }
    shapes, dtypes, placements, treespec, devices = _extract_input_info(sample, mesh)
    assert len(shapes) == 2
    assert shapes[0] == (4, 8)
    assert shapes[1] == (4, 16)
    assert len(placements) == 2
    assert len(devices) == 2
    assert all(d == torch.device("cpu") for d in devices)


def test_make_input_fn_dict_roundtrip(device_mesh_1d):
    """Test that _make_input_fn reconstructs dict structure from treespec."""
    mesh = device_mesh_1d
    sample = {
        "a": torch.rand(4, 8),
        "b": torch.rand(4, 16),
    }
    shapes, dtypes, _, treespec, devices = _extract_input_info(sample, mesh)
    input_fn = _make_input_fn(shapes, dtypes, treespec, devices)
    result = input_fn()
    # Dict is wrapped in a tuple
    assert isinstance(result, tuple) and len(result) == 1
    assert isinstance(result[0], dict)
    assert set(result[0].keys()) == {"a", "b"}
    assert result[0]["a"].shape == (4, 8)
    assert result[0]["b"].shape == (4, 16)
    # Verify per-tensor devices are preserved
    assert result[0]["a"].device == torch.device("cpu")
    assert result[0]["b"].device == torch.device("cpu")


def test_dict_input_integration(device_mesh_1d):
    """Integration test: auto_parallel with dict inputs."""
    dim = 128
    batch_size = 512
    local_batch_size = batch_size // device_mesh_1d.size()

    class DictModel(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)

        def forward(self, inputs):
            return self.linear(inputs["x"])

    with torch.device("meta"):
        model = DictModel(dim)

    x = DTensor.from_local(
        torch.rand(local_batch_size, dim, device="cuda"),
        device_mesh_1d,
        [Shard(0)],
    )
    sample_inputs = {"x": x}
    parallel_mod = auto_parallel(
        model,
        device_mesh_1d,
        sample_inputs=sample_inputs,
        out_shardings=(Shard(0),),
        compile=False,
    )

    # Should work with correct dict input
    out = parallel_mod({"x": torch.rand(local_batch_size, dim, device="cuda")})
    assert out.shape == (local_batch_size, dim)

    # Should reject wrong shape
    with pytest.raises(ValueError, match="shape"):
        parallel_mod({"x": torch.rand(local_batch_size + 1, dim, device="cuda")})
