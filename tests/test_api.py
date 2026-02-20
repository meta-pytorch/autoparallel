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


def test_init_inplace_data(device_mesh_1d):
    """Test that init_weights using self.weight.data[:] = value works correctly."""
    dim = 128

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            self.register_buffer("buf", torch.empty(dim))

        def forward(self, x):
            return self.linear(x) + self.buf

        def init_weights(self):
            self.linear.weight.data[:] = torch.ones(dim, dim) * 9.0
            self.linear.bias.data[:] = torch.full((dim,), 98.6)
            self.buf.data[:] = torch.arange(dim, dtype=torch.float32)

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
        parallel_mod.get_buffer("buf").full_tensor(),
        torch.arange(dim, dtype=torch.float32, device="cuda"),
    )


def test_init_aliased_buffers(device_mesh_1d):
    """Test that init_weights works when a submodule buffer aliases a top-level buffer.

    This mirrors the torchtitan Decoder pattern where rope.cache and freqs_cis
    are the same tensor. named_buffers(remove_duplicate=True) deduplicates them,
    so only freqs_cis ends up on the parallel model. The init_weights hook must
    still correctly propagate values set via the aliased buffer (rope.cache).
    """
    dim = 128

    class RoPE(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.register_buffer("cache", torch.zeros(dim), persistent=False)

        def forward(self, x):
            return x + self.cache

        def init_weights(self):
            self.cache = torch.arange(dim).float()

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            self.rope = RoPE(dim)
            self.register_buffer("freqs_cis", self.rope.cache, persistent=False)

        def forward(self, x):
            return self.linear(x) + self.freqs_cis

        def init_weights(self):
            with torch.no_grad():
                self.linear.weight.fill_(1.0)
                self.linear.bias.fill_(0.0)
            self.rope.init_weights()
            self.freqs_cis = self.rope.cache

    def input_fn():
        b = 512
        inputs = (torch.rand(b, dim, device="cuda"),)
        return inputs

    with torch.device("meta"):
        model = Model(dim)

    assert model.freqs_cis is model.rope.cache

    with AutoParallel(
        model,
        input_fn,
        device_mesh_1d,
    ) as autop:
        x_sharding = (Shard(0),)
        autop.add_input_constraints([x_sharding])
        sharding_placement = autop.optimize_placement()
        parallel_mod = autop.apply_placement(sharding_placement)

    parallel_mod.to_empty(device="cuda")
    parallel_mod.init_weights()

    expected = torch.arange(dim).float().cuda()
    assert torch.equal(parallel_mod.get_buffer("freqs_cis").full_tensor(), expected)


def test_init_aliased_parameters(device_mesh_1d):
    """Test that init_weights works when a parameter is registered under two FQNs.

    This mirrors weight tying in LLMs where embed.weight and lm_head.weight
    are the same parameter. named_parameters() deduplicates them, so the alias
    FQN is missing from the parallel model. The init_weights hook must not
    crash on the missing alias.
    """
    dim = 128

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.embed = nn.Linear(dim, dim, bias=False)
            # Weight tying: lm_head.weight aliases embed.weight.
            # named_parameters() yields embed.weight first (canonical),
            # lm_head.weight is the alias. Forward only uses embed.
            self.lm_head = nn.Linear(dim, dim, bias=False)
            self.lm_head.weight = self.embed.weight

        def forward(self, x):
            return self.embed(x)

        def init_weights(self):
            with torch.no_grad():
                self.embed.weight.fill_(1.0)

    def input_fn():
        b = 512
        inputs = (torch.rand(b, dim, device="cuda"),)
        return inputs

    with torch.device("meta"):
        model = Model(dim)

    assert model.lm_head.weight is model.embed.weight

    with AutoParallel(
        model,
        input_fn,
        device_mesh_1d,
    ) as autop:
        x_sharding = (Shard(0),)
        autop.add_input_constraints([x_sharding])
        sharding_placement = autop.optimize_placement()
        parallel_mod = autop.apply_placement(sharding_placement)

    parallel_mod.to_empty(device="cuda")
    parallel_mod.init_weights()

    expected = torch.ones(dim, dim, device="cuda")
    assert torch.equal(
        parallel_mod.get_parameter("embed.weight").full_tensor(), expected
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

    def input_fn():
        b = 512
        inputs = (torch.rand(b, dim, device="cuda"),)
        return inputs

    with torch.device("meta"):
        model = Model(dim)

    # Verify original model has ModuleDict
    assert isinstance(model.layers, nn.ModuleDict)

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


# Tests for the simplified auto_parallel API


def test_auto_parallel_basic(device_mesh_1d):
    """Test basic auto_parallel usage with DTensor input."""
    from torch.distributed.tensor import DTensor

    from autoparallel import auto_parallel

    dim = 128
    batch_size = 512
    local_batch_size = batch_size // device_mesh_1d.size()

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)

        def forward(self, x):
            return self.linear(x)

        def init_weights(self):
            nn.init.ones_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    with torch.device("meta"):
        model = Model(dim)

    # Create DTensor input with sharding
    x = DTensor.from_local(
        torch.rand(local_batch_size, dim, device="cuda"),
        device_mesh_1d,
        [Shard(0)],
    )

    parallel_model = auto_parallel(
        model,
        device_mesh_1d,
        sample_inputs=(x,),
        out_shardings=(Shard(0),),
        compile=False,
    )

    # Verify model was created
    assert parallel_model is not None
    assert hasattr(parallel_model, "linear")

    # Initialize and verify
    parallel_model.to_empty(device="cuda")
    parallel_model.init_weights()

    assert torch.equal(
        parallel_model.get_parameter("linear.weight").full_tensor(),
        torch.ones(dim, dim, device="cuda"),
    )


def test_auto_parallel_tuple_inputs(device_mesh_1d):
    """Test auto_parallel with multiple DTensor inputs as tuple."""
    from torch.distributed.tensor import DTensor

    from autoparallel import auto_parallel

    dim = 128
    batch_size = 512
    local_batch_size = batch_size // device_mesh_1d.size()

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear1 = nn.Linear(dim, dim)
            self.linear2 = nn.Linear(dim, dim)

        def forward(self, x, y):
            return self.linear1(x) + self.linear2(y)

    with torch.device("meta"):
        model = Model(dim)

    # Create DTensor inputs
    x = DTensor.from_local(
        torch.rand(local_batch_size, dim, device="cuda"),
        device_mesh_1d,
        [Shard(0)],
    )
    y = DTensor.from_local(
        torch.rand(local_batch_size, dim, device="cuda"),
        device_mesh_1d,
        [Shard(0)],
    )

    parallel_model = auto_parallel(
        model,
        device_mesh_1d,
        sample_inputs=(x, y),
        out_shardings=(Shard(0),),
        compile=False,
    )

    assert parallel_model is not None


def test_auto_parallel_multiple_outputs(device_mesh_1d):
    """Test auto_parallel with multiple outputs and pytree out_shardings."""
    from torch.distributed.tensor import DTensor

    from autoparallel import auto_parallel

    dim = 128
    batch_size = 512
    local_batch_size = batch_size // device_mesh_1d.size()

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear1 = nn.Linear(dim, dim, bias=True)
            self.linear2 = nn.Linear(dim, dim, bias=True)

        def forward(self, x):
            return self.linear1(x), self.linear2(x)

    with torch.device("meta"):
        model = Model(dim)

    x = DTensor.from_local(
        torch.rand(local_batch_size, dim, device="cuda"),
        device_mesh_1d,
        [Shard(0)],
    )

    # Pytree out_shardings matching tuple output
    parallel_model = auto_parallel(
        model,
        device_mesh_1d,
        sample_inputs=(x,),
        out_shardings=((Shard(0),), (Shard(0),)),
        compile=False,
    )

    assert parallel_model is not None


def test_auto_parallel_replicated_input(device_mesh_1d):
    """Test auto_parallel with regular tensor (assumed Replicate)."""
    from autoparallel import auto_parallel

    dim = 128
    batch_size = 512

    class Model(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)

        def forward(self, x):
            return self.linear(x)

    with torch.device("meta"):
        model = Model(dim)

    # Regular tensor - will be assumed Replicate
    # Output is sharded so the optimizer can find a valid solution
    x = torch.rand(batch_size, dim, device="cuda")

    parallel_model = auto_parallel(
        model,
        device_mesh_1d,
        sample_inputs=(x,),
        out_shardings=(Shard(0),),  # Shard output for valid solution
        compile=False,
    )

    assert parallel_model is not None


def test_auto_parallel_callable_inputs(device_mesh_1d):
    """Test auto_parallel with callable sample_inputs."""
    from torch.distributed.tensor import DTensor

    from autoparallel import auto_parallel

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

    def sample_inputs():
        return (
            DTensor.from_local(
                torch.rand(local_batch_size, dim, device="cuda"),
                device_mesh_1d,
                [Shard(0)],
            ),
        )

    parallel_model = auto_parallel(
        model,
        device_mesh_1d,
        sample_inputs=sample_inputs,
        out_shardings=(Shard(0),),
        compile=False,
    )

    assert parallel_model is not None


def test_auto_parallel_with_mp_policy(device_mesh_1d):
    """Test auto_parallel with mixed precision policy."""
    from torch.distributed.fsdp import MixedPrecisionPolicy
    from torch.distributed.tensor import DTensor

    from autoparallel import auto_parallel

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

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )

    parallel_model = auto_parallel(
        model,
        device_mesh_1d,
        sample_inputs=(x,),
        out_shardings=(Shard(0),),
        mp_policy=mp_policy,
        compile=False,
    )

    assert parallel_model is not None
