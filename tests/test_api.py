# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.fx.traceback as fx_traceback
from torch import nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel.api import AutoParallel, auto_parallel


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
        autop.add_output_constraints([x_sharding])
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


def test_inference_mode_compilation(device_mesh_1d):
    """Test that inference mode (no gradients) works with compile=True.

    This test verifies the fix for the bug where updated_flat_args was incorrectly
    formatted as a tuple for inference mode, causing compilation to fail.

    Regression test for: updated_flat_args should be a list for inference mode,
    not a tuple of (primals, tangents).
    """
    dim = 128

    class SimpleLinear(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias=False)

        def forward(self, x):
            return self.linear(x)

    def input_fn():
        batch_size = 256
        return torch.rand(batch_size, dim, device="cuda")

    with torch.device("meta"):
        model = SimpleLinear(dim, dim * 2)

    # Set model to inference mode (no gradients)
    for param in model.parameters():
        param.requires_grad = False

    # Test with compile=True - this should succeed with the fix
    with AutoParallel(model, input_fn, device_mesh_1d, None, compile=True) as autop:
        autop.add_parameter_memory_constraint(low=None, high=device_mesh_1d.ndim)

        # R -> S(0)
        autop.add_input_constraints([(Replicate(),)])
        autop.add_output_constraints([(Shard(0),)])

        sharding_placement = autop.optimize_placement()
        parallel_mod = autop.apply_placement(sharding_placement)

        # Verify the model was created
        assert parallel_mod is not None
        assert hasattr(autop, "parallel_gm")

        # Verify graph has expected structure (forward-only, no backward pass)
        placeholders = [
            n for n in autop.parallel_gm.graph.nodes if n.op == "placeholder"
        ]
        # Should only have 2 placeholders: weight and input (no tangents for inference)
        assert len(placeholders) == 2


def test_moduledict_preservation(device_mesh_1d):
    """Test that nn.ModuleDict structure is preserved during _assign_attr."""
    dim = 128

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            # Create a ModuleDict to test preservation
            self.layers = nn.ModuleDict(
                {
                    "layer1": nn.Linear(dim, dim),
                    "layer2": nn.Linear(dim, dim),
                }
            )

        def forward(self, x):
            x = self.layers["layer1"](x)
            x = self.layers["layer2"](x)
            return x

    with torch.device("meta"):
        model = Model(dim)

    # Verify original model has ModuleDict
    assert isinstance(model.layers, nn.ModuleDict)

    batch_size = 512
    local_batch_size = batch_size // device_mesh_1d.size()
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

    # Verify that the parallel_mod preserves the ModuleDict structure
    assert isinstance(
        parallel_mod.layers, nn.ModuleDict
    ), f"Expected nn.ModuleDict but got {type(parallel_mod.layers)}"

    # Verify that the ModuleDict contains the expected layers
    assert "layer1" in parallel_mod.layers
    assert "layer2" in parallel_mod.layers
    assert isinstance(parallel_mod.layers["layer1"], nn.Module)
    assert isinstance(parallel_mod.layers["layer2"], nn.Module)

    # Verify parameters are accessible through the ModuleDict structure
    assert hasattr(parallel_mod.layers["layer1"], "weight")
    assert hasattr(parallel_mod.layers["layer2"], "weight")


def test_enter_failure_cleans_up_fake_mode(device_mesh_1d):
    """FakeTensorMode pushed during build_model_graph is unwound if __enter__ fails.

    build_model_graph() pushes FakeTensorMode onto self.stack via
    aot_export_joint_with_descriptors. If something after build_model_graph()
    raises (e.g. ShardingOptimizer), Python never calls __exit__ because
    __enter__ didn't succeed, so __enter__ must unwind self.stack explicitly.
    Without cleanup, the leaked FakeTensorMode causes "Mixing fake modes NYI"
    on subsequent usage.
    """
    from unittest.mock import patch

    dim = 128

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)

        def forward(self, x):
            return self.linear(x)

    def input_fn():
        return (torch.rand(32, dim, device="cuda"),)

    with torch.device("meta"):
        model1 = Model(dim)
    auto_p = AutoParallel(model1, input_fn, device_mesh_1d)

    # Make ShardingOptimizer raise to simulate __enter__ failing
    # after build_model_graph() has already pushed FakeTensorMode.
    with patch(
        "autoparallel.api.ShardingOptimizer", side_effect=RuntimeError("injected")
    ):
        with pytest.raises(RuntimeError, match="injected"):
            auto_p.__enter__()

    # If FakeTensorMode leaked, this would fail with "Mixing fake modes NYI"
    # during copy.deepcopy inside AutoParallel.__init__.
    with torch.device("meta"):
        model2 = Model(dim)
    AutoParallel(model2, input_fn, device_mesh_1d)


def test_unused_parameters_captured(device_mesh_1d):
    """Unused parameters should appear on the parallel model."""
    dim = 128

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.used_linear = nn.Linear(dim, dim)
            self.unused_linear = nn.Linear(dim, dim)

        def forward(self, x):
            return self.used_linear(x)

    with torch.device("meta"):
        model = Model(dim)

    batch_size = 512
    local_batch_size = batch_size // device_mesh_1d.size()
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

    param_names = {name for name, _ in parallel_mod.named_parameters()}
    assert "used_linear.weight" in param_names
    assert "used_linear.bias" in param_names
    assert (
        "unused_linear.weight" in param_names
    ), f"unused_linear.weight not found in parallel model params: {param_names}"
    assert (
        "unused_linear.bias" in param_names
    ), f"unused_linear.bias not found in parallel model params: {param_names}"


def test_aliased_submodule(device_mesh_1d):
    """Test that aliased submodules (two module attrs pointing to the same object) work.

    This mirrors the DINOVid pattern where model.model_ema = model.teacher,
    causing named_parameters(remove_duplicate=False) to yield the same tensor
    under two FQNs. move_to_fake must not assert on the second occurrence
    because the tensor was already replaced with a fake on the first pass.
    The module alias must also be re-established on the parallel model so that
    model_ema and teacher remain the same object.
    """
    dim = 128

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.student = nn.Linear(dim, dim)
            self.teacher = nn.Linear(dim, dim)
            # Alias: model_ema IS teacher (same object, two registered names)
            self.model_ema = self.teacher

        def forward(self, x):
            return self.student(x) + self.teacher(x)

        def init_weights(self):
            nn.init.ones_(self.student.weight)
            nn.init.zeros_(self.student.bias)
            nn.init.ones_(self.teacher.weight)
            nn.init.zeros_(self.teacher.bias)

    with torch.device("meta"):
        model = Model(dim)

    assert model.model_ema is model.teacher

    batch_size = 512
    local_batch_size = batch_size // device_mesh_1d.size()
    x = DTensor.from_local(
        torch.rand(local_batch_size, dim, device="cuda"),
        device_mesh_1d,
        [Shard(0)],
    )
    # The key assertion is that auto_parallel succeeds without crashing
    # on the aliased submodule in move_to_fake.
    parallel_mod = auto_parallel(
        model,
        device_mesh_1d,
        sample_inputs=(x,),
        out_shardings=(Shard(0),),
        compile=False,
    )

    # Module alias should be re-established on the parallel model
    assert parallel_mod.model_ema is parallel_mod.teacher

    parallel_mod.to_empty(device="cuda")
    parallel_mod.init_weights()

    expected = torch.ones(dim, dim, device="cuda")
    assert torch.equal(
        parallel_mod.get_parameter("student.weight").full_tensor(), expected
    )
    assert torch.equal(
        parallel_mod.get_parameter("teacher.weight").full_tensor(), expected
    )


def test_parallel_model_isinstance(device_mesh_1d):
    """AutoParallelModule should be an instance of the user's model class."""
    dim = 128

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)

        def forward(self, x):
            return self.linear(x)

    with torch.device("meta"):
        model = Model(dim)

    batch_size = 512
    local_batch_size = batch_size // device_mesh_1d.size()
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
    assert isinstance(parallel_mod, Model)


def test_user_method_accessible(device_mesh_1d):
    """User-defined methods and instance attributes should be available on parallel model."""
    dim = 128

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            self.dim = dim
            self.model_name = "test_model"

        def forward(self, x):
            return self.linear(x)

        def get_num_params(self):
            return sum(p.numel() for p in self.parameters())

    with torch.device("meta"):
        model = Model(dim)

    batch_size = 512
    local_batch_size = batch_size // device_mesh_1d.size()
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
    assert hasattr(parallel_mod, "get_num_params")
    assert parallel_mod.get_num_params() > 0
    assert parallel_mod.dim == dim
    assert parallel_mod.model_name == "test_model"


def test_user_ema_update(device_mesh_1d):
    """User-defined EMA update method should work on the parallel model."""
    dim = 128

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim, bias=False)
            self.ema_linear = nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            return self.linear(x)

        def init_weights(self):
            with torch.no_grad():
                self.linear.weight.fill_(2.0)
                self.ema_linear.weight.fill_(0.0)

        @torch.no_grad()
        def update_ema(self, decay=0.9):
            for p, ema_p in zip(self.linear.parameters(), self.ema_linear.parameters()):
                ema_p.lerp_(p, 1 - decay)

    with torch.device("meta"):
        model = Model(dim)

    batch_size = 512
    local_batch_size = batch_size // device_mesh_1d.size()
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
    parallel_mod.to_empty(device="cuda")
    parallel_mod.init_weights()

    assert torch.equal(
        parallel_mod.get_parameter("ema_linear.weight").full_tensor(),
        torch.zeros(dim, dim, device="cuda"),
    )

    parallel_mod.update_ema(decay=0.9)

    # EMA should have moved 10% toward the main weight (2.0)
    expected = torch.full((dim, dim), 0.2, device="cuda")
    assert torch.allclose(
        parallel_mod.get_parameter("ema_linear.weight").full_tensor(),
        expected,
    )


def test_user_reset_buffers(device_mesh_1d):
    """User-defined method that resets buffers should work on the parallel model."""
    dim = 128

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            self.register_buffer("step_count", torch.zeros(1))

        def forward(self, x):
            return self.linear(x) + self.step_count

        def init_weights(self):
            with torch.no_grad():
                self.linear.weight.fill_(1.0)
                self.linear.bias.fill_(0.0)
                self.step_count.fill_(42.0)

        def reset_step_count(self):
            self.step_count.zero_()

    with torch.device("meta"):
        model = Model(dim)

    batch_size = 512
    local_batch_size = batch_size // device_mesh_1d.size()
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
    parallel_mod.to_empty(device="cuda")
    parallel_mod.init_weights()

    assert parallel_mod.get_buffer("step_count").full_tensor().item() == 42.0
    parallel_mod.reset_step_count()
    assert parallel_mod.get_buffer("step_count").full_tensor().item() == 0.0


def test_user_classmethod_and_property(device_mesh_1d):
    """Classmethods and properties defined on the user model should be accessible."""
    dim = 128

    class Model(nn.Module):
        _registry = []

        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            self._hidden_dim = dim * 4

        def forward(self, x):
            return self.linear(x)

        @classmethod
        def from_config(cls, config_dim):
            return cls(config_dim)

        @property
        def hidden_dim(self):
            return self._hidden_dim

    with torch.device("meta"):
        model = Model(dim)

    batch_size = 512
    local_batch_size = batch_size // device_mesh_1d.size()
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
    assert parallel_mod.hidden_dim == dim * 4
    assert hasattr(type(parallel_mod), "from_config")
    assert type(parallel_mod)._registry is Model._registry


def test_inherited_user_model(device_mesh_1d):
    """AutoParallelModule should inherit from the leaf class, getting the full MRO."""
    dim = 128

    class BaseModel(nn.Module):
        def get_num_params(self):
            return sum(p.numel() for p in self.parameters())

    class Model(BaseModel):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)

        def forward(self, x):
            return self.linear(x)

    with torch.device("meta"):
        model = Model(dim)

    batch_size = 512
    local_batch_size = batch_size // device_mesh_1d.size()
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
    assert isinstance(parallel_mod, Model)
    assert isinstance(parallel_mod, BaseModel)
    assert parallel_mod.get_num_params() > 0


# Tests for build_compile_fn and schedule_overlap_bucketing_from_inductor_configs


def test_overlap_bucketing_called_when_compile_false(device_mesh_1d):
    """Test that schedule_overlap_bucketing_from_inductor_configs is called when compile=False.

    This verifies the fix that ensures overlap scheduling passes are run
    even when not using the full inductor compilation pipeline.
    """
    from unittest.mock import patch

    dim = 128

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)

        def forward(self, x):
            return self.linear(x)

    def input_fn():
        b = 512
        return (torch.rand(b, dim, device="cuda"),)

    with torch.device("meta"):
        model = Model(dim)

    # Track if schedule_overlap_bucketing_from_inductor_configs was called
    overlap_bucketing_called = []

    original_schedule_overlap_bucketing = None
    try:
        from torch._inductor.fx_passes.overlap_scheduling import (
            schedule_overlap_bucketing_from_inductor_configs,
        )

        original_schedule_overlap_bucketing = (
            schedule_overlap_bucketing_from_inductor_configs
        )
    except ImportError:
        pytest.skip(
            "schedule_overlap_bucketing_from_inductor_configs not available in this PyTorch version"
        )

    def tracking_overlap_bucketing(fx_g):
        overlap_bucketing_called.append(fx_g)
        # Call the original function
        return original_schedule_overlap_bucketing(fx_g)

    with patch(
        "torch._inductor.fx_passes.overlap_scheduling.schedule_overlap_bucketing_from_inductor_configs",
        side_effect=tracking_overlap_bucketing,
    ):
        with AutoParallel(
            model,
            input_fn,
            device_mesh_1d,
            compile=False,  # This should use build_compile_fn
        ) as autop:
            autop.add_input_constraints([(Shard(0),)])
            sharding_placement = autop.optimize_placement()
            _ = autop.apply_placement(sharding_placement)

    # Verify schedule_overlap_bucketing_from_inductor_configs was called
    assert (
        len(overlap_bucketing_called) > 0
    ), "schedule_overlap_bucketing_from_inductor_configs should be called when compile=False"

    # Verify a valid graph module was passed
    for fx_g in overlap_bucketing_called:
        assert isinstance(fx_g, torch.fx.GraphModule)


def test_recursive_post_grad_passes_not_called_when_compile_true(device_mesh_1d):
    """Test that build_compile_fn is NOT used when compile=True.

    When compile=True, the full inductor pipeline (compile_fx_inner) should
    be used instead of build_compile_fn.
    """
    dim = 128

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)

        def forward(self, x):
            return self.linear(x)

    def input_fn():
        b = 512
        return (torch.rand(b, dim, device="cuda"),)

    with torch.device("meta"):
        model = Model(dim)

    from torch._inductor.compile_fx import compile_fx_inner

    # When compile=True, the compiler_fn should be compile_fx_inner
    autop = AutoParallel(
        model,
        input_fn,
        device_mesh_1d,
        compile=True,
    )

    assert autop.compiler_fn is compile_fx_inner


def test_compile_false_end_to_end(device_mesh_1d):
    """Test that compile=False produces a working model with post-grad passes applied.

    This is an end-to-end test that verifies the model can be created and
    potentially executed (in fake mode).
    """
    dim = 128

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            self.register_buffer("scale", torch.ones(dim))

        def forward(self, x):
            return self.linear(x) * self.scale

        def init_weights(self):
            # TODO: can't use eye_ because of https://github.com/pytorch/pytorch/issues/173357
            # nn.init.eye_(self.linear.weight)
            nn.init.ones_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    def input_fn():
        b = 512
        return (torch.rand(b, dim, device="cuda"),)

    with torch.device("meta"):
        model = Model(dim)

    with AutoParallel(
        model,
        input_fn,
        device_mesh_1d,
        compile=False,
    ) as autop:
        autop.add_input_constraints([(Shard(0),)])
        sharding_placement = autop.optimize_placement()
        parallel_mod = autop.apply_placement(sharding_placement)

    # Verify model structure
    assert parallel_mod is not None
    assert hasattr(parallel_mod, "linear")

    # Verify parameters are DTensors
    from torch.distributed.tensor import DTensor

    for name, param in parallel_mod.named_parameters():
        assert isinstance(param, DTensor), f"Parameter {name} should be DTensor"

    # Initialize and verify
    parallel_mod.to_empty(device="cuda")
    parallel_mod.init_weights()

    # Verify weights were initialized correctly
    weight = parallel_mod.get_parameter("linear.weight").full_tensor()
    # TODO: can't use eye_ because of https://github.com/pytorch/pytorch/issues/173357
    # assert torch.allclose(weight, torch.eye(dim, device="cuda"))
    assert torch.allclose(weight, torch.ones(dim, dim, device="cuda"))
