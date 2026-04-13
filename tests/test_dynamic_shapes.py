# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from conftest import apply_cuda_patches
from torch import nn
from torch._functorch._aot_autograd.fx_utils import get_param_nodes
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel.api import AutoParallel


class FFN(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.linear1 = nn.Linear(dim1, dim2, bias=False)
        self.linear2 = nn.Linear(dim2, dim1, bias=False)

    def forward(self, x):
        return self.linear2(self.linear1(x))


class TransformerBlock(nn.Module):
    def __init__(self, nheads, dim1, dim2):
        super().__init__()
        self.nheads = nheads
        self.wq = nn.Linear(dim1, dim1, bias=False)
        self.wk = nn.Linear(dim1, dim1, bias=False)
        self.wv = nn.Linear(dim1, dim1, bias=False)
        self.wo = nn.Linear(dim1, dim1, bias=False)
        self.w1 = nn.Linear(dim1, dim2, bias=False)
        self.w2 = nn.Linear(dim2, dim1, bias=False)

    def forward(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
        k = k.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
        v = v.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)

        o = nn.functional.scaled_dot_product_attention(q, k, v)
        o = o.permute(0, 2, 1, 3).flatten(-2)
        o = self.wo(o)

        o0 = o + x
        o = self.w1(o0)
        o = torch.nn.functional.relu(o)
        o = self.w2(o)
        return o0 + o


# ============================================================================
# Unit tests — self-contained, no mesh/GPU required
# ============================================================================


class TestCheckForwardArgs:
    """Tests for _check_forward_args with dynamic shapes."""

    def test_concrete_match(self):
        from autoparallel.input_validation import _check_forward_args

        expected = [torch.empty(4, 8, device="meta")]
        _check_forward_args([torch.randn(4, 8)], expected)

    def test_concrete_mismatch_raises(self):
        from autoparallel.input_validation import _check_forward_args

        expected = [torch.empty(4, 8, device="meta")]
        with pytest.raises(ValueError, match="has shape"):
            _check_forward_args([torch.randn(3, 8)], expected)

    def test_symint_batch_dim_accepts_any_size(self):
        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import (
            DimDynamic,
            ShapeEnv,
            StatelessSymbolicContext,
        )

        from autoparallel.input_validation import _check_forward_args

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env, static_shapes=False)
        real = torch.empty(16, 128)
        sym_ctx = StatelessSymbolicContext(
            dynamic_sizes=[DimDynamic.DYNAMIC, DimDynamic.STATIC]
        )
        sym_tensor = fake_mode.from_tensor(real, symbolic_context=sym_ctx)

        # SymInt dim should accept any size
        for test_bs in [1, 16, 32, 256]:
            _check_forward_args([torch.randn(test_bs, 128)], [sym_tensor])

    def test_symint_rejects_wrong_static_dim(self):
        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import (
            DimDynamic,
            ShapeEnv,
            StatelessSymbolicContext,
        )

        from autoparallel.input_validation import _check_forward_args

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env, static_shapes=False)
        real = torch.empty(16, 128)
        sym_ctx = StatelessSymbolicContext(
            dynamic_sizes=[DimDynamic.DYNAMIC, DimDynamic.STATIC]
        )
        sym_tensor = fake_mode.from_tensor(real, symbolic_context=sym_ctx)

        with pytest.raises(ValueError, match="has shape"):
            _check_forward_args([torch.randn(16, 64)], [sym_tensor])

    def test_ndim_mismatch_raises(self):
        from autoparallel.input_validation import _check_forward_args

        expected = [torch.empty(4, 8, device="meta")]
        with pytest.raises(ValueError, match="dims"):
            _check_forward_args([torch.randn(4, 8, 1)], expected)


class TestMakeInputsDynamic:
    """Tests for _make_inputs_dynamic."""

    def test_all_dims_become_symbolic(self):
        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        from autoparallel.api import _make_inputs_dynamic

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env, static_shapes=False)

        with fake_mode:
            x = torch.randn(16, 128)
            y = torch.randn(16, 64)

        result = _make_inputs_dynamic((x, y), fake_mode)

        for t in result:
            for s in t.shape:
                assert isinstance(s, torch.SymInt), f"Expected SymInt, got {type(s)}"

    def test_non_tensor_leaves_unchanged(self):
        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        from autoparallel.api import _make_inputs_dynamic

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env, static_shapes=False)

        with fake_mode:
            x = torch.randn(16, 128)

        result = _make_inputs_dynamic((x, 42, "hello"), fake_mode)

        assert isinstance(result[0], torch.Tensor)
        assert result[1] == 42
        assert result[2] == "hello"

    def test_symbolic_dims_concretize_after_mm(self):
        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        from autoparallel.api import _make_inputs_dynamic

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env, static_shapes=False)

        with fake_mode:
            x = torch.randn(16, 128)
            w = torch.randn(128, 64)

        result = _make_inputs_dynamic((x,), fake_mode)
        sym_x = result[0]

        # Before mm, both dims are symbolic
        assert isinstance(sym_x.shape[0], torch.SymInt)
        assert isinstance(sym_x.shape[1], torch.SymInt)

        # After mm with concrete weight, dim 1 gets a guard
        with fake_mode:
            y = sym_x @ w

        # Output has symbolic batch, concrete model dim
        assert isinstance(y.shape[0], torch.SymInt)
        assert y.shape[1] == 64
        # The guard should have concretized dim 1's expression
        assert sym_x.shape[1].node.expr.is_number

    def test_uses_meta_tensors_no_allocation(self):
        """_make_inputs_dynamic should use meta tensors internally."""
        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        from autoparallel.api import _make_inputs_dynamic

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env, static_shapes=False)

        with fake_mode:
            # Large tensor — would OOM if actually allocated
            x = torch.randn(1000000, 1000000)

        # Should not raise OOM
        result = _make_inputs_dynamic((x,), fake_mode)
        assert isinstance(result[0].shape[0], torch.SymInt)


class TestConcretizeArgs:
    """Tests for _concretize_args in optimize_sharding."""

    def test_concretize_symints(self):
        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import (
            DimDynamic,
            ShapeEnv,
            StatelessSymbolicContext,
        )

        from autoparallel.optimize_sharding import _concretize_args

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env, static_shapes=False)
        real = torch.empty(16, 128)
        sym_ctx = StatelessSymbolicContext(
            dynamic_sizes=[DimDynamic.DYNAMIC, DimDynamic.DYNAMIC]
        )
        x = fake_mode.from_tensor(real, symbolic_context=sym_ctx)

        args = (x, x.shape[0], [x.shape[1], 32])
        result = _concretize_args(args)

        # FakeTensor should have concrete shapes
        assert isinstance(result[0], torch.Tensor)
        assert all(isinstance(s, int) for s in result[0].shape)
        assert result[0].shape == (16, 128)

        # SymInt should become int
        assert isinstance(result[1], int)
        assert result[1] == 16

        # Nested SymInt should become int
        assert isinstance(result[2][0], int)
        assert result[2][0] == 128
        assert result[2][1] == 32

    def test_passthrough_concrete(self):
        from autoparallel.optimize_sharding import _concretize_args

        args = (torch.randn(4, 4), 42, [1, 2, 3])
        result = _concretize_args(args)

        assert result[1] == 42
        assert result[2] == [1, 2, 3]


class TestProducesTensor:
    """Tests for _produces_tensor in optimize_sharding."""

    def test_tensor(self):
        from autoparallel.optimize_sharding import _produces_tensor

        assert _produces_tensor(torch.randn(4))

    def test_tuple_of_tensors(self):
        from autoparallel.optimize_sharding import _produces_tensor

        assert _produces_tensor((torch.randn(4), torch.randn(4)))

    def test_nested_with_none(self):
        from autoparallel.optimize_sharding import _produces_tensor

        assert _produces_tensor((torch.randn(4), None))

    def test_symint(self):
        from autoparallel.optimize_sharding import _produces_tensor

        assert not _produces_tensor(42)

    def test_none(self):
        from autoparallel.optimize_sharding import _produces_tensor

        assert not _produces_tensor(None)


class TestComputeLocalViewShape:
    """Tests for _compute_local_view_shape in apply_sharding."""

    def _make_output_spec(self, placements, mesh_shape):
        from torch.distributed._tensor.placement_types import DTensorSpec

        class FakeMesh:
            def __init__(self, shape):
                self._shape = shape
                self.ndim = len(shape)

            @property
            def shape(self):
                return self._shape

            def size(self, dim):
                return self._shape[dim]

        return DTensorSpec(mesh=FakeMesh(mesh_shape), placements=tuple(placements))

    def test_flatten_batch_seq(self):
        """[B, S, H] -> [B*S, H] with Shard(0) on dp."""
        from autoparallel.apply_sharding import _compute_local_view_shape

        output_spec = self._make_output_spec([Shard(0), Replicate()], (32, 8))

        # Use SymInts for local input shape
        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import (
            DimDynamic,
            ShapeEnv,
            StatelessSymbolicContext,
        )

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env, static_shapes=False)
        t = fake_mode.from_tensor(
            torch.empty(8, 256, 6144, device="meta"),
            symbolic_context=StatelessSymbolicContext(
                dynamic_sizes=[
                    DimDynamic.DYNAMIC,
                    DimDynamic.DYNAMIC,
                    DimDynamic.STATIC,
                ]
            ),
        )

        local_out = _compute_local_view_shape(
            global_input_shape=(256, 256, 6144),
            global_output_shape=(65536, 6144),
            local_input_shape=t.shape,
            output_spec=output_spec,
        )

        # Output dim 0 should be symbolic (product of batch and seq)
        assert isinstance(local_out[0], torch.SymInt)
        # Output dim 1 should be concrete
        assert local_out[1] == 6144

    def test_unflatten_batch_seq(self):
        """[B*S, H] -> [B, S, H] with Shard(0) on dp."""
        from autoparallel.apply_sharding import _compute_local_view_shape

        output_spec = self._make_output_spec([Shard(0), Replicate()], (32, 8))

        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import (
            DimDynamic,
            ShapeEnv,
            StatelessSymbolicContext,
        )

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env, static_shapes=False)
        t = fake_mode.from_tensor(
            torch.empty(2048, 6144, device="meta"),
            symbolic_context=StatelessSymbolicContext(
                dynamic_sizes=[DimDynamic.DYNAMIC, DimDynamic.STATIC]
            ),
        )

        local_out = _compute_local_view_shape(
            global_input_shape=(65536, 6144),
            global_output_shape=(256, 256, 6144),
            local_input_shape=t.shape,
            output_spec=output_spec,
        )

        # Split: sharded piece should be symbolic (B_local = B*S_local // S)
        assert isinstance(local_out[0], torch.SymInt)
        # Split: non-sharded piece should be concrete
        assert local_out[1] == 256
        assert local_out[2] == 6144

    def test_split_heads_with_tp(self):
        """[B, S, H] -> [B, S, nheads, head_dim] with TP on nheads."""
        from autoparallel.apply_sharding import _compute_local_view_shape

        output_spec = self._make_output_spec([Shard(0), Shard(2)], (32, 8))

        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import (
            DimDynamic,
            ShapeEnv,
            StatelessSymbolicContext,
        )

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env, static_shapes=False)
        t = fake_mode.from_tensor(
            torch.empty(8, 256, 6144, device="meta"),
            symbolic_context=StatelessSymbolicContext(
                dynamic_sizes=[
                    DimDynamic.DYNAMIC,
                    DimDynamic.DYNAMIC,
                    DimDynamic.STATIC,
                ]
            ),
        )

        local_out = _compute_local_view_shape(
            global_input_shape=(256, 256, 6144),
            global_output_shape=(256, 256, 48, 128),
            local_input_shape=t.shape,
            output_spec=output_spec,
        )

        # B and S should be symbolic (from local input)
        assert isinstance(local_out[0], torch.SymInt)
        assert isinstance(local_out[1], torch.SymInt)
        # nheads should be 48 // 8 = 6 (TP sharded)
        assert local_out[2] == 6
        # head_dim stays concrete
        assert local_out[3] == 128

    def test_merge_heads(self):
        """[B, S, nheads, head_dim] -> [B, S, H]"""
        from autoparallel.apply_sharding import _compute_local_view_shape

        output_spec = self._make_output_spec([Shard(0), Replicate()], (32, 8))

        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import (
            DimDynamic,
            ShapeEnv,
            StatelessSymbolicContext,
        )

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env, static_shapes=False)
        t = fake_mode.from_tensor(
            torch.empty(8, 256, 48, 128, device="meta"),
            symbolic_context=StatelessSymbolicContext(
                dynamic_sizes=[
                    DimDynamic.DYNAMIC,
                    DimDynamic.DYNAMIC,
                    DimDynamic.STATIC,
                    DimDynamic.STATIC,
                ]
            ),
        )

        local_out = _compute_local_view_shape(
            global_input_shape=(256, 256, 48, 128),
            global_output_shape=(256, 256, 6144),
            local_input_shape=t.shape,
            output_spec=output_spec,
        )

        assert isinstance(local_out[0], torch.SymInt)
        assert isinstance(local_out[1], torch.SymInt)
        # Flatten of nheads*head_dim = 48*128 = 6144
        assert local_out[2] == 6144


class TestReSymbolizeGraph:
    """Tests for _re_symbolize_graph."""

    def test_re_symbolize_replaces_old_symints(self):
        """Re-symbolize should produce fresh symbols not from the old ShapeEnv."""
        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        from autoparallel.apply_sharding import _re_symbolize_graph

        # Create a graph inside a FakeTensorMode (simulating the lowering)
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)

        with fake_mode:
            x = torch.randn(8, 64)
            w = torch.randn(64, 32)

        def f(x, w):
            return x @ w

        gm = make_fx(f, tracing_mode="symbolic")(x, w)

        # Re-symbolize
        fresh_gm = _re_symbolize_graph(gm)

        # Check placeholders have SymInt shapes
        for node in fresh_gm.graph.nodes:
            if node.op == "placeholder":
                val = node.meta.get("val")
                if isinstance(val, torch.Tensor):
                    has_symint = any(isinstance(s, torch.SymInt) for s in val.shape)
                    assert has_symint, f"{node.name} should have SymInt shapes"

        # The SymInts should be from a different ShapeEnv than the original
        for node in fresh_gm.graph.nodes:
            if node.op == "placeholder":
                val = node.meta.get("val")
                if isinstance(val, torch.Tensor):
                    for s in val.shape:
                        if isinstance(s, torch.SymInt):
                            assert s.node.shape_env is not shape_env


class TestConcretizeShape:
    """Tests for _concretize_shape."""

    def test_concrete_passthrough(self):
        from autoparallel.apply_sharding import _concretize_shape

        result = _concretize_shape((4, 8, 16))
        assert result == (4, 8, 16)
        assert all(isinstance(s, int) for s in result)

    def test_symint_to_hint(self):
        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import (
            DimDynamic,
            ShapeEnv,
            StatelessSymbolicContext,
        )

        from autoparallel.apply_sharding import _concretize_shape

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env, static_shapes=False)
        real = torch.empty(16, 128)
        sym_ctx = StatelessSymbolicContext(
            dynamic_sizes=[DimDynamic.DYNAMIC, DimDynamic.STATIC]
        )
        x = fake_mode.from_tensor(real, symbolic_context=sym_ctx)

        result = _concretize_shape(x.shape)
        assert result == (16, 128)
        assert all(isinstance(s, int) for s in result)


# ============================================================================
# Integration tests — require fake process group and mesh fixtures
# ============================================================================


@apply_cuda_patches
def test_dynamic_produces_same_placement_as_static_1d(device_mesh_1d):
    """ILP solution should be identical with dynamic=True vs dynamic=False."""
    dim1, dim2 = 1024, 4096
    bs = 2048 * device_mesh_1d.shape[0]

    def input_fn():
        return torch.randn(bs, dim1, device="cuda")

    with torch.device("meta"):
        model = FFN(dim1, dim2)

    placement = (Shard(0),)
    with AutoParallel(model, input_fn, device_mesh_1d) as autop_static:
        autop_static.add_input_constraints([placement])
        autop_static.add_output_constraints([placement])
        static_placement = autop_static.optimize_placement(verbose=False)

    with torch.device("meta"):
        model = FFN(dim1, dim2)

    with AutoParallel(model, input_fn, device_mesh_1d, dynamic=True) as autop_dynamic:
        autop_dynamic.add_input_constraints([placement])
        autop_dynamic.add_output_constraints([placement])
        dynamic_placement = autop_dynamic.optimize_placement(verbose=False)

    # Compare placements for all nodes
    for node_s, node_d in zip(
        autop_static.gm.graph.nodes, autop_dynamic.gm.graph.nodes
    ):
        if node_s not in static_placement:
            continue
        if node_d not in dynamic_placement:
            continue
        sp = static_placement[node_s]
        dp = dynamic_placement[node_d]
        assert sp.output_specs.placements == dp.output_specs.placements, (
            f"Placement mismatch for {node_s.name}: "
            f"static={sp.output_specs.placements} vs dynamic={dp.output_specs.placements}"
        )


@apply_cuda_patches
def test_dynamic_produces_same_placement_as_static_2d(device_mesh_2d):
    """ILP solution for transformer block should be identical with dynamic=True."""
    dim1 = 6144
    dim2 = dim1 * 4
    nheads = 48
    bs = 8 * device_mesh_2d.shape[0]

    def input_fn():
        return torch.randn(bs, 256, dim1, device="cuda", requires_grad=True)

    with torch.device("meta"):
        model = TransformerBlock(nheads, dim1, dim2)

    placement = (Shard(0), Replicate())
    with AutoParallel(model, input_fn, device_mesh_2d) as autop_static:
        autop_static.add_input_constraints([placement])
        autop_static.add_output_constraints([placement])
        static_placement = autop_static.optimize_placement(verbose=False)

    with torch.device("meta"):
        model = TransformerBlock(nheads, dim1, dim2)

    with AutoParallel(model, input_fn, device_mesh_2d, dynamic=True) as autop_dynamic:
        autop_dynamic.add_input_constraints([placement])
        autop_dynamic.add_output_constraints([placement])
        dynamic_placement = autop_dynamic.optimize_placement(verbose=False)

    param_nodes_s = get_param_nodes(autop_static.gm.graph)
    param_nodes_d = get_param_nodes(autop_dynamic.gm.graph)
    for node_s, node_d in zip(param_nodes_s, param_nodes_d):
        sp = static_placement[node_s].output_specs.placements
        dp = dynamic_placement[node_d].output_specs.placements
        assert (
            sp == dp
        ), f"Param placement mismatch for {node_s.name}: static={sp} vs dynamic={dp}"


@apply_cuda_patches
def test_dynamic_apply_placement_ffn(device_mesh_1d):
    """apply_placement should succeed with dynamic=True for a simple FFN."""
    dim1, dim2 = 1024, 4096
    bs = 2048 * device_mesh_1d.shape[0]

    def input_fn():
        return torch.randn(bs, dim1, device="cuda")

    with torch.device("meta"):
        model = FFN(dim1, dim2)

    placement = (Shard(0),)
    with AutoParallel(model, input_fn, device_mesh_1d, dynamic=True) as autop:
        autop.add_input_constraints([placement])
        autop.add_output_constraints([placement])
        autop.optimize_placement(verbose=False)
        parallel_model = autop.apply_placement()

    assert parallel_model is not None


@apply_cuda_patches
def test_dynamic_apply_placement_transformer(device_mesh_2d):
    """apply_placement should succeed with dynamic=True for transformer block."""
    dim1 = 6144
    dim2 = dim1 * 4
    nheads = 48
    bs = 8 * device_mesh_2d.shape[0]

    def input_fn():
        return torch.randn(bs, 256, dim1, device="cuda", requires_grad=True)

    with torch.device("meta"):
        model = TransformerBlock(nheads, dim1, dim2)

    placement = (Shard(0), Replicate())
    with AutoParallel(
        model, input_fn, device_mesh_2d, dynamic=True, compile=False
    ) as autop:
        autop.add_input_constraints([placement])
        autop.add_output_constraints([placement])
        autop.optimize_placement(verbose=False)
        parallel_model = autop.apply_placement()

    assert parallel_model is not None


@apply_cuda_patches
def test_dynamic_joint_graph_has_symbolic_shapes(device_mesh_2d):
    """The joint graph from dynamic=True should have symbolic shapes on inputs."""
    dim1 = 6144
    dim2 = dim1 * 4
    nheads = 48
    bs = 8 * device_mesh_2d.shape[0]

    def input_fn():
        return torch.randn(bs, 256, dim1, device="cuda", requires_grad=True)

    with torch.device("meta"):
        model = TransformerBlock(nheads, dim1, dim2)

    placement = (Shard(0), Replicate())
    with AutoParallel(
        model, input_fn, device_mesh_2d, dynamic=True, compile=False
    ) as autop:
        autop.add_input_constraints([placement])
        autop.add_output_constraints([placement])

        # Check the joint graph has symbolic shapes on input placeholders
        has_symint = False
        for node in autop.gm.graph.nodes:
            if node.op == "placeholder" and isinstance(
                node.meta.get("val"), torch.Tensor
            ):
                for s in node.meta["val"].shape:
                    if isinstance(s, torch.SymInt):
                        has_symint = True
                        break
        assert has_symint, "joint graph should have symbolic shapes on inputs"


@apply_cuda_patches
def test_dynamic_check_forward_args_accepts_different_batch(device_mesh_1d):
    """_check_forward_args should accept different batch sizes with dynamic shapes."""
    from autoparallel.input_validation import (
        _check_forward_args,
        _compute_expected_inputs,
    )

    dim1, dim2 = 1024, 4096
    bs = 2048 * device_mesh_1d.shape[0]

    def input_fn():
        return torch.randn(bs, dim1, device="cuda")

    with torch.device("meta"):
        model = FFN(dim1, dim2)

    placement = (Shard(0),)
    with AutoParallel(model, input_fn, device_mesh_1d, dynamic=True) as autop:
        autop.add_input_constraints([placement])
        autop.add_output_constraints([placement])
        autop.optimize_placement(verbose=False)

        expected = _compute_expected_inputs(
            autop._traced_inputs, autop.input_constraints, device_mesh_1d
        )

    # The expected shapes should have SymInt for the batch dim
    assert isinstance(expected[0].shape[0], torch.SymInt), "batch dim should be SymInt"

    # Different batch sizes should be accepted
    local_bs = bs // device_mesh_1d.size()
    for test_bs in [local_bs, local_bs * 2, 1]:
        _check_forward_args(
            [torch.randn(test_bs, dim1)],
            expected,
        )

    # Wrong non-batch dim should be rejected
    with pytest.raises(ValueError, match="has shape"):
        _check_forward_args(
            [torch.randn(local_bs, dim1 + 1)],
            expected,
        )
