# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import copy
import inspect

import pytest
import torch
from conftest import apply_cuda_patches
from torch import nn
from torch._functorch._aot_autograd.fx_utils import get_param_and_grad_nodes
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Shard

from autoparallel.api import AutoParallel
from autoparallel.dtype_cast import make_dtype_cast_forward
from autoparallel.input_validation import ForwardInputs


class TypeDispatchModel(nn.Module):
    def __init__(self, root_parameter):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Identity())
        self.register_buffer("offset", torch.randn(4), persistent=False)
        self.register_parameter(
            "gain", nn.Parameter(torch.randn(4)) if root_parameter else None
        )

    def forward(self, x, *, labels):
        assert type(self) is TypeDispatchModel
        assert type(self.layers) is nn.Sequential
        metadata = {nn.Linear: self.offset, nn.Identity: labels}
        y = self.layers(x) + metadata[type(self.layers[0])].to(x.dtype)
        if self.gain is not None:
            y = y * self.gain
        return {
            "values": y,
            "labels": metadata[type(self.layers[1])],
            "positive": y > 0,
        }


@pytest.mark.parametrize("root_parameter", [False, True])
def test_cast_preserves_types_outputs_and_gradients(root_parameter):
    model = TypeDispatchModel(root_parameter)
    reference = copy.deepcopy(model)
    parameters = dict(model.named_parameters())
    types = {name: type(mod) for name, mod in model.named_modules()}
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, output_dtype=torch.float32
    )
    forward = make_dtype_cast_forward(model, policy)
    assert inspect.signature(forward) == inspect.signature(model.forward)

    x = torch.randn(3, 4, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    labels = torch.arange(3)
    actual = forward(x, labels=labels)
    y = nn.functional.linear(
        ref_x.to(torch.bfloat16), reference.layers[0].weight.to(torch.bfloat16)
    ) + reference.offset.to(torch.bfloat16)
    if reference.gain is not None:
        y = y * reference.gain.to(torch.bfloat16)
    expected = {"values": y.float(), "labels": labels, "positive": y > 0}
    torch.testing.assert_close(actual, expected)
    actual["values"].sum().backward()
    expected["values"].sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad)
    for name, param in model.named_parameters():
        assert param is parameters[name]
        assert param.dtype == torch.float32
        assert param.grad.dtype == torch.float32
        torch.testing.assert_close(param.grad, reference.get_parameter(name).grad)
    assert {name: type(mod) for name, mod in model.named_modules()} == types
    assert actual["labels"] is labels
    assert actual["positive"].dtype == torch.bool
    assert "offset" not in model.state_dict()


@pytest.mark.parametrize("cast_forward_inputs", [False, True])
def test_parameter_free_root_casts_nested_inputs_and_outputs(cast_forward_inputs):
    class Model(nn.Module):
        def forward(self, inputs, *, extra):
            assert type(self) is Model
            expected_dtype = torch.bfloat16 if cast_forward_inputs else torch.float32
            assert inputs[0].dtype == expected_dtype
            assert extra["value"].dtype == expected_dtype
            return inputs[0] + extra["value"], inputs[1], extra["mask"]

    model = Model()
    forward = make_dtype_cast_forward(
        model,
        MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            output_dtype=torch.float64,
            cast_forward_inputs=cast_forward_inputs,
        ),
    )
    labels = torch.arange(3)
    mask = labels > 0
    values, actual_labels, actual_mask = forward(
        (torch.ones(3), labels), extra={"value": torch.ones(3), "mask": mask}
    )
    torch.testing.assert_close(values, torch.full((3,), 2.0, dtype=torch.float64))
    assert actual_labels is labels
    assert actual_mask is mask
    assert type(model) is Model


@pytest.mark.parametrize("param_dtype", [None, torch.float32])
def test_cast_accepts_unchanged_parameter_dtype(param_dtype):
    model = nn.Linear(4, 4)
    x = torch.randn(2, 4, requires_grad=True)
    forward = make_dtype_cast_forward(
        model, MixedPrecisionPolicy(param_dtype=param_dtype, output_dtype=torch.float64)
    )
    actual = forward(x)
    torch.testing.assert_close(actual, model(x).double())
    actual.sum().backward()
    assert model.weight.grad.dtype == torch.float32


def test_cast_restores_tied_parameters_after_exception():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.left = nn.Linear(4, 4, bias=False)
            self.right = nn.Linear(4, 4, bias=False)
            self.right.weight = self.left.weight

        def forward(self, x):
            assert type(self.left) is nn.Linear
            assert self.left.weight is self.right.weight
            assert self.left.weight.dtype == torch.bfloat16
            raise ValueError("forward failed")

    model = Model()
    weight = model.left.weight
    forward = make_dtype_cast_forward(
        model, MixedPrecisionPolicy(param_dtype=torch.bfloat16)
    )
    for _ in range(2):
        with pytest.raises(ValueError, match="forward failed"):
            forward(torch.randn(2, 4))
        assert model.left.weight is weight
        assert model.right.weight is weight
        assert weight.dtype == torch.float32
    replica = copy.deepcopy(model)
    assert type(replica) is Model
    assert replica.left.weight is replica.right.weight
    assert replica.left.weight is not weight


def test_cast_preserves_aliased_buffer_updates_and_training_modes():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4, bias=False)
            self.register_buffer("counter", torch.zeros((), dtype=torch.int64))
            self.observer = nn.Module()
            self.observer.register_buffer("counter", self.counter, persistent=False)

        def forward(self, x):
            assert type(self.observer) is nn.Module
            assert self.counter is self.observer.counter
            self.counter.add_(1)
            return self.linear(x) + self.counter

    model = Model()
    model.linear.eval()
    reference = copy.deepcopy(model)
    modes = {name: mod.training for name, mod in model.named_modules()}
    keys = list(model.state_dict())
    counter = model.counter
    forward = make_dtype_cast_forward(
        model, MixedPrecisionPolicy(param_dtype=torch.bfloat16)
    )
    for step in (1, 2):
        x = torch.randn(2, 4)
        expected = (
            nn.functional.linear(
                x.to(torch.bfloat16), reference.linear.weight.to(torch.bfloat16)
            )
            + step
        )
        torch.testing.assert_close(forward(x), expected)
        assert model.counter.item() == step
        assert model.counter is counter
        assert model.observer.counter is counter
    assert list(model.state_dict()) == keys
    assert {name: mod.training for name, mod in model.named_modules()} == modes


@apply_cuda_patches
@pytest.mark.parametrize("root_parameter", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
def test_type_dispatch_survives_capture_and_sharding(
    device_mesh_1d, root_parameter, dynamic
):
    with torch.device("meta"):
        model = TypeDispatchModel(root_parameter)
    state_keys = list(model.state_dict())

    def inputs():
        return ForwardInputs(
            args=(torch.randn(512, 4, device="cuda", requires_grad=True),),
            kwargs={"labels": torch.arange(512, device="cuda")},
        )

    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
    )
    with AutoParallel(
        model, inputs, device_mesh_1d, policy, dynamic=dynamic, repeated_subgraphs=False
    ) as autop:
        assert type(autop.model) is TypeDispatchModel
        assert type(autop.model.layers[0]) is nn.Linear
        assert type(autop.model.layers[1]) is nn.Identity
        assert list(autop.model.state_dict()) == state_keys
        attributed_nodes = [
            node
            for node in autop.gm.graph.nodes
            if node.meta.get("nn_module_stack") or node.meta.get("fwd_nn_module_stack")
        ]
        assert attributed_nodes
        for node in attributed_nodes:
            stack = node.meta.get("nn_module_stack") or node.meta["fwd_nn_module_stack"]
            for path, cls in stack.values():
                assert cls is type(autop.model.get_submodule(path))
        pairs = get_param_and_grad_nodes(autop.gm.graph)
        assert {desc.target for desc in pairs} == set(dict(model.named_parameters()))
        for param, grad in pairs.values():
            assert param.meta["val"].dtype == torch.float32
            assert grad.meta["val"].dtype == torch.float32
        casts = autop.gm.graph.find_nodes(
            op="call_function", target=torch.ops.autoparallel.dtype_cast.default
        )
        assert sum(node.args[1] == torch.bfloat16 for node in casts) == len(pairs)
        assert sum(node.args[1] == torch.float32 for node in casts) == len(pairs)
        autop.add_input_constraints([(Shard(0),), (Shard(0),)])
        parallel = autop.apply_placement(autop.optimize_placement())

    parallel.to_empty(device="cuda")
    with torch.no_grad():
        for param in parallel.parameters():
            param.fill_(1)
        parallel.offset.zero_()
    for batch_size in (2, 3) if dynamic else (2,):
        x = torch.ones(batch_size, 4, device="cuda", requires_grad=True)
        labels = torch.arange(batch_size, device="cuda")
        output = parallel(x, labels=labels)
        torch.testing.assert_close(output["values"], torch.full_like(x, 4))
        assert output["values"].dtype == torch.float32
        assert output["labels"].dtype == torch.int64
        assert output["positive"].dtype == torch.bool
        output["values"].sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 4))
    assert type(parallel.layers[0]) is nn.Linear
    assert type(parallel.layers[1]) is nn.Identity


@apply_cuda_patches
def test_cast_capture_preserves_tied_frozen_and_unused_state(device_mesh_1d):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.left = nn.Linear(4, 4, bias=False)
            self.right = nn.Linear(4, 4, bias=False)
            self.right.weight = self.left.weight
            self.frozen = nn.Linear(4, 4, bias=False)
            self.frozen.weight.requires_grad_(False)
            self.unused = nn.Linear(4, 4, bias=False)
            self.register_buffer("offset", torch.ones(4), persistent=False)
            self.register_buffer("cached_offset", self.offset, persistent=False)

        def forward(self, x):
            assert type(self) is Model
            assert type(self.left) is nn.Linear
            assert self.left.weight is self.right.weight
            return (
                self.left(x)
                + self.right(x)
                + self.frozen(x)
                + self.offset
                + self.cached_offset
            )

    with torch.device("meta"):
        model = Model()

    def inputs():
        return torch.randn(512, 4, device="cuda")

    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
    with AutoParallel(
        model, inputs, device_mesh_1d, policy, repeated_subgraphs=False
    ) as autop:
        assert autop.model.left.weight is autop.model.right.weight
        assert autop.model.offset is autop.model.cached_offset
        pairs = get_param_and_grad_nodes(autop.gm.graph)
        assert {desc.target for desc in pairs} == {
            "left.weight",
            "frozen.weight",
            "unused.weight",
        }
        assert set(autop.joint_with_descriptors.buffers_spec) == {
            "offset",
            "cached_offset",
        }
        for desc, (param, grad) in pairs.items():
            assert param.meta["val"].dtype == torch.float32
            assert (grad is not None) == (desc.target == "left.weight")
        casts = autop.gm.graph.find_nodes(
            op="call_function", target=torch.ops.autoparallel.dtype_cast.default
        )
        assert sum(node.args[1] == torch.bfloat16 for node in casts) == 2


@apply_cuda_patches
def test_cast_capture_preserves_repeated_module_attribution(device_mesh_1d):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(4, 4, bias=False)])

        def forward(self, x):
            return self.layers[0](self.layers[0](x))

    with torch.device("meta"):
        model = Model()

    def inputs():
        return torch.randn(512, 4, device="cuda")

    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
    with AutoParallel(
        model, inputs, device_mesh_1d, policy, repeated_subgraphs=False
    ) as autop:
        forward_mms = [
            node
            for node in autop.gm.graph.find_nodes(
                op="call_function", target=torch.ops.aten.mm.default
            )
            if node.meta.get("nn_module_stack")
        ]
        assert len(forward_mms) == 2
        stacks = [node.meta["nn_module_stack"] for node in forward_mms]
        assert stacks[0].keys() != stacks[1].keys()
        assert all(
            list(stack.values())[-1] == ("layers.0", nn.Linear) for stack in stacks
        )
