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
from autoparallel.input_validation import _check_forward_args, _compute_expected_inputs


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
    """Test local shape computation from global shape + Shard constraint."""
    mesh = device_mesh_1d
    world_size = mesh.size()

    traced = [torch.empty(512, 128, device="meta")]
    constraints = [(Shard(0),)]

    result = _compute_expected_inputs(traced, constraints, mesh)
    assert len(result) == 1
    assert result[0].shape == (512 // world_size, 128)
    assert result[0].dtype == torch.float32


def test_compute_expected_inputs_default_constraints(device_mesh_1d):
    """Test that None constraints default to Shard(0)."""
    mesh = device_mesh_1d
    world_size = mesh.size()

    traced = [torch.empty(512, 128, device="meta")]

    result = _compute_expected_inputs(traced, None, mesh)
    assert len(result) == 1
    assert result[0].shape == (512 // world_size, 128)


def test_compute_expected_inputs_replicated(device_mesh_1d):
    """Test that Replicate constraint leaves shape unchanged."""
    mesh = device_mesh_1d

    traced = [torch.empty(512, 128, device="meta")]
    constraints = [(Replicate(),)]

    result = _compute_expected_inputs(traced, constraints, mesh)
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
