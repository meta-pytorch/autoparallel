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


class ViewHeavyModel(nn.Module):
    """Model that exercises view/reshape with batch-dependent shapes."""

    def __init__(self, dim, nheads):
        super().__init__()
        self.nheads = nheads
        self.head_dim = dim // nheads
        self.proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        bs, seq, _ = x.shape
        # linear on last dim (no batch/seq flattening)
        y = self.proj(x)
        # unflatten: [B, S, D] -> [B, S, nheads, head_dim]
        y = y.view(bs, seq, self.nheads, self.head_dim)
        # permute + flatten: [B, nheads, S, head_dim] -> [B, S, D]
        y = y.permute(0, 2, 1, 3).contiguous().view(bs, seq, -1)
        return self.out_proj(y)


class FactoryOpModel(nn.Module):
    """Model that creates tensors with input-dependent shapes."""

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        out = self.linear(x)
        # Factory op with batch-dependent shape
        mask = torch.zeros(x.shape[0], x.shape[1], dtype=x.dtype, device=x.device)
        return out + mask.unsqueeze(-1)


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

    def test_concretized_symint_dim_is_enforced(self):
        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        from autoparallel.api import _make_inputs_dynamic
        from autoparallel.input_validation import _check_forward_args

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env, static_shapes=False)

        with fake_mode:
            x = torch.randn(16, 128)
            w = torch.randn(128, 64)

        (sym_x,) = _make_inputs_dynamic((x,), fake_mode)

        with fake_mode:
            _ = sym_x @ w

        with pytest.raises(ValueError, match="has shape"):
            _check_forward_args([torch.randn(7, 999)], [sym_x])


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

    def test_preserves_requires_grad(self):
        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        from autoparallel.api import _make_inputs_dynamic

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env, static_shapes=False)

        with fake_mode:
            x = torch.randn(16, 128, requires_grad=True)
            y = torch.randn(16, 64, requires_grad=False)

        sym_x, sym_y = _make_inputs_dynamic((x, y), fake_mode)

        assert sym_x.requires_grad
        assert not sym_y.requires_grad


class TestConcretizeArgs:
    """Tests for concretize_args in optimize_sharding."""

    def test_concretize_symints(self):
        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import (
            DimDynamic,
            ShapeEnv,
            StatelessSymbolicContext,
        )

        from autoparallel.optimize_sharding import concretize_args

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env, static_shapes=False)
        real = torch.empty(16, 128)
        sym_ctx = StatelessSymbolicContext(
            dynamic_sizes=[DimDynamic.DYNAMIC, DimDynamic.DYNAMIC]
        )
        x = fake_mode.from_tensor(real, symbolic_context=sym_ctx)

        args = (x, x.shape[0], [x.shape[1], 32])
        result = concretize_args(args)

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
        from autoparallel.optimize_sharding import concretize_args

        args = (torch.randn(4, 4), 42, [1, 2, 3])
        result = concretize_args(args)

        assert result[1] == 42
        assert result[2] == [1, 2, 3]


class TestConcretizeGm:
    """Tests for concretize_gm in optimize_sharding."""

    def test_structure_preserved(self):
        """Concretized graph has same nodes, ops, targets, edges."""
        from torch._subclasses import FakeTensorMode
        from torch.fx import Graph, GraphModule
        from torch.fx.experimental.symbolic_shapes import (
            DimDynamic,
            ShapeEnv,
            StatelessSymbolicContext,
        )

        from autoparallel.optimize_sharding import concretize_gm

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env, static_shapes=False)

        real = torch.empty(16, 128, device="meta")
        sym_ctx = StatelessSymbolicContext(
            dynamic_sizes=[DimDynamic.DYNAMIC, DimDynamic.STATIC]
        )
        sym_t = fake_mode.from_tensor(real, symbolic_context=sym_ctx)

        graph = Graph()
        x = graph.placeholder("x")
        x.meta["val"] = sym_t
        y = graph.call_function(torch.relu, (x,))
        with fake_mode:
            y.meta["val"] = torch.relu(sym_t)
        graph.output(y)
        gm = GraphModule({}, graph)

        concrete_gm, orig_to_concrete, concrete_to_orig = concretize_gm(gm)

        # Same number of nodes
        orig_nodes = list(gm.graph.nodes)
        conc_nodes = list(concrete_gm.graph.nodes)
        assert len(orig_nodes) == len(conc_nodes)

        # Same ops and targets
        for orig, conc in zip(orig_nodes, conc_nodes):
            assert orig.op == conc.op
            assert orig.target == conc.target

        # Bidirectional mapping is total
        assert len(orig_to_concrete) == len(orig_nodes)
        assert len(concrete_to_orig) == len(conc_nodes)
        for orig, conc in zip(orig_nodes, conc_nodes):
            assert orig_to_concrete[orig] is conc
            assert concrete_to_orig[conc] is orig

    def test_symints_concretized(self):
        """meta['val'] shapes in the concrete graph are plain ints."""
        from torch._subclasses import FakeTensorMode
        from torch.fx import Graph, GraphModule
        from torch.fx.experimental.symbolic_shapes import (
            DimDynamic,
            ShapeEnv,
            StatelessSymbolicContext,
        )

        from autoparallel.optimize_sharding import concretize_gm

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env, static_shapes=False)

        real = torch.empty(16, 128, device="meta")
        sym_ctx = StatelessSymbolicContext(
            dynamic_sizes=[DimDynamic.DYNAMIC, DimDynamic.DYNAMIC]
        )
        sym_t = fake_mode.from_tensor(real, symbolic_context=sym_ctx)
        assert any(isinstance(s, torch.SymInt) for s in sym_t.shape)

        graph = Graph()
        x = graph.placeholder("x")
        x.meta["val"] = sym_t
        graph.output(x)
        gm = GraphModule({}, graph)

        concrete_gm, _, _ = concretize_gm(gm)

        conc_placeholder = next(
            n for n in concrete_gm.graph.nodes if n.op == "placeholder"
        )
        val = conc_placeholder.meta["val"]
        assert all(isinstance(s, int) for s in val.shape)
        assert val.shape == (16, 128)

    def test_non_val_meta_preserved(self):
        """Non-val metadata (e.g., desc) is preserved in the concrete graph."""
        from torch._subclasses import FakeTensorMode
        from torch.fx import Graph, GraphModule
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        from autoparallel.optimize_sharding import concretize_gm

        fake_mode = FakeTensorMode(shape_env=ShapeEnv(), static_shapes=False)

        graph = Graph()
        x = graph.placeholder("x")
        x.meta["val"] = fake_mode.from_tensor(torch.empty(4, 8, device="meta"))
        x.meta["desc"] = "test_descriptor"
        x.meta["custom_key"] = 42
        graph.output(x)
        gm = GraphModule({}, graph)

        concrete_gm, _, _ = concretize_gm(gm)

        conc_x = next(n for n in concrete_gm.graph.nodes if n.op == "placeholder")
        assert conc_x.meta["desc"] == "test_descriptor"
        assert conc_x.meta["custom_key"] == 42


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


class TestShapeEnvSwap:
    """Tests for ShapeEnv swap in apply_sharding_to_model."""

    def test_parallel_graph_has_fresh_symints(self):
        """Parallel graph should have SymInts from a fresh ShapeEnv, not the joint graph's."""
        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch.fx.experimental.symbolic_shapes import (
            DimDynamic,
            ShapeEnv,
            StatelessSymbolicContext,
        )

        old_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=old_env, static_shapes=False)

        real_x = torch.empty(8, 64, device="meta")
        sym_ctx = StatelessSymbolicContext(
            dynamic_sizes=[DimDynamic.DYNAMIC, DimDynamic.DYNAMIC]
        )
        x = fake_mode.from_tensor(real_x, symbolic_context=sym_ctx)
        with fake_mode:
            x = x.to("cpu")
            w = torch.randn(64, 32)

        # Swap ShapeEnv, create fresh symbolic tensor, trace
        new_env = ShapeEnv()
        fake_mode.shape_env = new_env
        fake_mode.static_shapes = False

        real_local = torch.empty(4, 64, device="meta")
        fresh_ctx = StatelessSymbolicContext(
            dynamic_sizes=[DimDynamic.DYNAMIC, DimDynamic.STATIC]
        )
        x_fresh = fake_mode.from_tensor(real_local, symbolic_context=fresh_ctx)
        with fake_mode:
            x_fresh = x_fresh.to("cpu")

        gm = make_fx(lambda x, w: x @ w, tracing_mode="symbolic")(x_fresh, w)

        # Restore
        fake_mode.shape_env = old_env

        # Check: placeholders have SymInts from new_env, not old_env
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                val = node.meta.get("val")
                if isinstance(val, torch.Tensor):
                    for s in val.shape:
                        if isinstance(s, torch.SymInt):
                            assert id(s.node.shape_env) == id(
                                new_env
                            ), "SymInt should be from fresh ShapeEnv"
                            assert id(s.node.shape_env) != id(
                                old_env
                            ), "SymInt should NOT be from old ShapeEnv"

    def test_make_local_args_preserves_requires_grad(self):
        from unittest.mock import patch

        from torch._subclasses import FakeTensorMode
        from torch.distributed.device_mesh import DeviceMesh
        from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
        from torch.distributed.tensor.placement_types import Replicate
        from torch.fx import Graph, GraphModule
        from torch.fx.experimental.symbolic_shapes import (
            DimDynamic,
            ShapeEnv,
            StatelessSymbolicContext,
        )

        from autoparallel.apply_sharding import _make_local_args

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env, static_shapes=False)

        real = torch.empty(16, 128, device="meta", requires_grad=True)
        sym_ctx = StatelessSymbolicContext(
            dynamic_sizes=[DimDynamic.DYNAMIC, DimDynamic.STATIC]
        )
        tensor = fake_mode.from_tensor(real, symbolic_context=sym_ctx)

        graph = Graph()
        x = graph.placeholder("x")
        x.meta["val"] = tensor
        graph.output((x,))
        gm = GraphModule(torch.nn.Module(), graph)

        mesh = DeviceMesh("cpu", torch.arange(1))
        spec = DTensorSpec(
            mesh=mesh,
            placements=(Replicate(),),
            tensor_meta=TensorMeta(torch.Size([16, 128]), (128, 1), torch.float32),
        )
        sharding_placement = {x: type("Placement", (), {"input_specs": (spec,)})()}

        with patch(
            "autoparallel.apply_sharding.DTensor.from_local",
            side_effect=lambda local, mesh, placements: type(
                "DummyDTensor",
                (),
                {
                    "redistribute": lambda self, mesh, placements: self,
                    "to_local": lambda self: local,
                },
            )(),
        ):
            (local_arg,) = _make_local_args(gm, sharding_placement)

        assert local_arg.requires_grad


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


class TestLocalizeShapeArg:
    """Tests for _localize_shape_arg in apply_sharding."""

    def _make_node_with_shape(self, shape, fake_mode):
        """Create a minimal FX node with a FakeTensor meta['val']."""
        from torch.fx import Graph

        graph = Graph()
        node = graph.placeholder("x")
        real = torch.empty(shape, device="meta")
        node.meta["val"] = fake_mode.from_tensor(real)
        with fake_mode:
            node.meta["val"] = node.meta["val"].to("cpu")
        return node

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

    def test_fully_concrete_shape(self):
        """All concrete args: divide sharded dims by mesh size."""
        from torch._subclasses import FakeTensorMode

        from autoparallel.apply_sharding import _localize_shape_arg

        fake_mode = FakeTensorMode()
        # Node meta["val"] must be the OUTPUT shape (what the view produces)
        node = self._make_node_with_shape((256 * 256, 6144), fake_mode)
        spec = self._make_output_spec([Shard(0), Replicate()], (32, 8))

        result = _localize_shape_arg(node, [256 * 256, 6144], spec)

        assert result[0] == (256 * 256) // 32  # dim 0 sharded by dp
        assert result[1] == 6144  # dim 1 not sharded

    def test_mixed_symint_and_concrete(self):
        """SymInt args preserved, concrete args divided."""
        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import (
            DimDynamic,
            ShapeEnv,
            StatelessSymbolicContext,
        )

        from autoparallel.apply_sharding import _localize_shape_arg

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env, static_shapes=False)
        # Output shape: [batch*seq, hidden] — 2 dims
        node = self._make_node_with_shape((256 * 256, 6144), fake_mode)

        # Simulate interpreter producing SymInt for batch*seq, concrete for hidden
        real = torch.empty(2048, device="meta")
        sym_ctx = StatelessSymbolicContext(dynamic_sizes=[DimDynamic.DYNAMIC])
        sym_val = fake_mode.from_tensor(real, symbolic_context=sym_ctx)
        sym_batch_seq = sym_val.shape[0]

        spec = self._make_output_spec([Shard(0), Shard(1)], (32, 8))
        result = _localize_shape_arg(node, [sym_batch_seq, 6144], spec)

        # SymInt preserved (already local)
        assert isinstance(result[0], torch.SymInt)
        assert result[0] is sym_batch_seq
        # Concrete divided by tp mesh
        assert result[1] == 6144 // 8

    def test_multi_dim_sharding(self):
        """Same dim sharded on multiple mesh dims."""
        from torch._subclasses import FakeTensorMode

        from autoparallel.apply_sharding import _localize_shape_arg

        fake_mode = FakeTensorMode()
        node = self._make_node_with_shape((1024, 64), fake_mode)
        spec = self._make_output_spec([Shard(0), Shard(0)], (4, 64))

        result = _localize_shape_arg(node, [1024, 64], spec)

        # dim 0 divided by both mesh dims: 1024 // 4 // 64 = 4
        assert result[0] == 4
        assert result[1] == 64

    def test_uneven_shard_uses_ceiling_division(self):
        """Uneven shard dim uses ceiling division: 10 / 3 → 4."""
        from torch._subclasses import FakeTensorMode

        from autoparallel.apply_sharding import _localize_shape_arg

        fake_mode = FakeTensorMode()
        node = self._make_node_with_shape((10, 128), fake_mode)
        spec = self._make_output_spec([Shard(0)], (3,))

        result = _localize_shape_arg(node, [10, 128], spec)

        assert result[0] == 4  # ceil(10 / 3) = 4
        assert result[1] == 128


class TestHasRankVaryingSize:
    """Tests for _has_rank_varying_size in apply_sharding."""

    def _make_spec(self, placements, mesh_shape):
        from torch.distributed._tensor.placement_types import DTensorSpec

        class FakeMesh:
            def __init__(self, shape):
                self._shape = shape
                self.ndim = len(shape)

            @property
            def shape(self):
                return self._shape

        return DTensorSpec(mesh=FakeMesh(mesh_shape), placements=tuple(placements))

    def test_even_shard_not_varying(self):
        from autoparallel.apply_sharding import _has_rank_varying_size

        spec = self._make_spec([Shard(0)], (8,))
        assert not _has_rank_varying_size(0, (256, 128), spec)
        assert not _has_rank_varying_size(1, (256, 128), spec)

    def test_uneven_shard_is_varying(self):
        from autoparallel.apply_sharding import _has_rank_varying_size

        spec = self._make_spec([Shard(0)], (3,))
        assert _has_rank_varying_size(0, (10, 128), spec)
        assert not _has_rank_varying_size(1, (10, 128), spec)

    def test_replicate_not_varying(self):
        from autoparallel.apply_sharding import _has_rank_varying_size

        spec = self._make_spec([Replicate()], (8,))
        assert not _has_rank_varying_size(0, (10, 128), spec)

    def test_2d_mesh_uneven_on_second_dim(self):
        from autoparallel.apply_sharding import _has_rank_varying_size

        # Shard(0) on mesh dim 0 (size 32): 4096/32=128 → even
        # Shard(1) on mesh dim 1 (size 6): 4096/6 → uneven
        spec = self._make_spec([Shard(0), Shard(1)], (32, 6))
        assert not _has_rank_varying_size(0, (4096, 4096), spec)
        assert _has_rank_varying_size(1, (4096, 4096), spec)

    def test_uneven_param_dim_becomes_symbolic_in_make_local_args(self):
        """When a parameter is unevenly sharded, its local dim should be
        symbolic so all ranks can share the same graph."""
        from unittest.mock import patch

        from torch._subclasses import FakeTensorMode
        from torch.distributed.device_mesh import DeviceMesh
        from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
        from torch.distributed.tensor.placement_types import Replicate
        from torch.fx import Graph, GraphModule
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        from autoparallel.apply_sharding import _make_local_args

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env, static_shapes=False)

        # Parameter: concrete [400, 100], will be unevenly sharded on dim 0
        with fake_mode:
            param_tensor = torch.empty(400, 100, device="cpu")

        graph = Graph()
        p = graph.placeholder("param")
        p.meta["val"] = param_tensor
        graph.output((p,))
        gm = GraphModule(torch.nn.Module(), graph)

        mesh = DeviceMesh("cpu", torch.arange(7).reshape(1, 7))
        spec = DTensorSpec(
            mesh=mesh,
            placements=(Replicate(), Shard(0)),
            tensor_meta=TensorMeta(torch.Size([400, 100]), (100, 1), torch.float32),
        )
        sharding_placement = {
            p: type("P", (), {"input_specs": (spec,)})(),
        }

        # Swap ShapeEnv (as apply_sharding_to_model does)
        fake_mode.shape_env = ShapeEnv()

        # The blocker: _make_local_args is called inside `with fake_mode:`
        # (from _apply_placement_common in api.py). from_tensor doesn't create
        # SymInts when the mode context was entered before the ShapeEnv swap.
        with fake_mode:
            with patch(
                "autoparallel.apply_sharding.DTensor.from_local",
                side_effect=lambda local, mesh, placements: type(
                    "DummyDTensor",
                    (),
                    {
                        "redistribute": lambda self, mesh, placements: self,
                        "to_local": lambda self: local,
                    },
                )(),
            ):
                (local_param,) = _make_local_args(gm, sharding_placement)

        # dim 0 is unevenly sharded (400 % 7 != 0) so it should be symbolic
        assert any(isinstance(s, torch.SymInt) for s in local_param.shape), (
            f"Uneven param dim should be symbolic but got concrete shape "
            f"{local_param.shape}"
        )


class TestCheckForwardArgsMultiInput:
    """Tests for _check_forward_args with multiple inputs."""

    def test_two_inputs_different_batch_accepted(self):
        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import (
            DimDynamic,
            ShapeEnv,
            StatelessSymbolicContext,
        )

        from autoparallel.input_validation import _check_forward_args

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env, static_shapes=False)

        t1 = fake_mode.from_tensor(
            torch.empty(16, 128),
            symbolic_context=StatelessSymbolicContext(
                dynamic_sizes=[DimDynamic.DYNAMIC, DimDynamic.STATIC]
            ),
        )
        t2 = fake_mode.from_tensor(
            torch.empty(16, 64),
            symbolic_context=StatelessSymbolicContext(
                dynamic_sizes=[DimDynamic.DYNAMIC, DimDynamic.STATIC]
            ),
        )

        # Different batch sizes accepted
        _check_forward_args([torch.randn(32, 128), torch.randn(32, 64)], [t1, t2])

    def test_two_inputs_wrong_feature_rejected(self):
        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import (
            DimDynamic,
            ShapeEnv,
            StatelessSymbolicContext,
        )

        from autoparallel.input_validation import _check_forward_args

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env, static_shapes=False)

        t1 = fake_mode.from_tensor(
            torch.empty(16, 128),
            symbolic_context=StatelessSymbolicContext(
                dynamic_sizes=[DimDynamic.DYNAMIC, DimDynamic.STATIC]
            ),
        )
        t2 = fake_mode.from_tensor(
            torch.empty(16, 64),
            symbolic_context=StatelessSymbolicContext(
                dynamic_sizes=[DimDynamic.DYNAMIC, DimDynamic.STATIC]
            ),
        )

        # Wrong feature dim on second input rejected
        with pytest.raises(ValueError, match="has shape"):
            _check_forward_args([torch.randn(32, 128), torch.randn(32, 999)], [t1, t2])

    def test_arg_count_mismatch_rejected(self):
        from autoparallel.input_validation import _check_forward_args

        expected = [torch.empty(4, 8, device="meta")]
        with pytest.raises(ValueError, match="expected 1"):
            _check_forward_args([torch.randn(4, 8), torch.randn(4, 8)], expected)


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
        sharding_placement = autop.optimize_placement(verbose=False)
        parallel_model = autop.apply_placement(sharding_placement)

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
        sharding_placement = autop.optimize_placement(verbose=False)
        parallel_model = autop.apply_placement(sharding_placement)

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


@apply_cuda_patches
def test_dynamic_apply_placement_view_heavy(device_mesh_2d):
    """apply_placement with dynamic=True for a view-heavy model.

    Exercises the _localize_shape_arg path for view/reshape ops where
    shape args mix SymInts (batch-dependent, already local) and concrete
    ints (global constants needing division by mesh size).
    """
    dim = 6144
    nheads = 48
    bs = 8 * device_mesh_2d.shape[0]

    def input_fn():
        return torch.randn(bs, 256, dim, device="cuda", requires_grad=True)

    with torch.device("meta"):
        model = ViewHeavyModel(dim, nheads)

    placement = (Shard(0), Replicate())
    with AutoParallel(
        model, input_fn, device_mesh_2d, dynamic=True, compile=False
    ) as autop:
        autop.add_input_constraints([placement])
        autop.add_output_constraints([placement])
        sharding_placement = autop.optimize_placement(verbose=False)
        parallel_model = autop.apply_placement(sharding_placement)

    assert parallel_model is not None


@apply_cuda_patches
def test_dynamic_apply_placement_factory_op(device_mesh_1d):
    """apply_placement with dynamic=True for a model with factory ops.

    Exercises _localize_shape_arg for factory ops where shape depends
    on the symbolic batch dimension.
    """
    dim = 1024
    bs = 2048 * device_mesh_1d.shape[0]

    def input_fn():
        return torch.randn(bs, 64, dim, device="cuda", requires_grad=True)

    with torch.device("meta"):
        model = FactoryOpModel(dim)

    placement = (Shard(0),)
    with AutoParallel(
        model, input_fn, device_mesh_1d, dynamic=True, compile=False
    ) as autop:
        autop.add_input_constraints([placement])
        autop.add_output_constraints([placement])
        sharding_placement = autop.optimize_placement(verbose=False)
        parallel_model = autop.apply_placement(sharding_placement)

    assert parallel_model is not None


@apply_cuda_patches
def test_dynamic_vs_static_parity_view_heavy(device_mesh_2d):
    """ILP placement and apply_placement parity for view-heavy model."""
    dim = 6144
    nheads = 48
    bs = 8 * device_mesh_2d.shape[0]

    def input_fn():
        return torch.randn(bs, 256, dim, device="cuda", requires_grad=True)

    placement = (Shard(0), Replicate())

    # Static
    with torch.device("meta"):
        model = ViewHeavyModel(dim, nheads)
    with AutoParallel(model, input_fn, device_mesh_2d, dynamic=False) as autop_s:
        autop_s.add_input_constraints([placement])
        autop_s.add_output_constraints([placement])
        static_placement = autop_s.optimize_placement(verbose=False)
        autop_s.apply_placement(static_placement)

    # Dynamic
    with torch.device("meta"):
        model = ViewHeavyModel(dim, nheads)
    with AutoParallel(model, input_fn, device_mesh_2d, dynamic=True) as autop_d:
        autop_d.add_input_constraints([placement])
        autop_d.add_output_constraints([placement])
        dynamic_placement = autop_d.optimize_placement(verbose=False)
        autop_d.apply_placement(dynamic_placement)

    # Compare param placements — these must match exactly since they
    # determine the model's weight distribution.
    param_nodes_s = get_param_nodes(autop_s.gm.graph)
    param_nodes_d = get_param_nodes(autop_d.gm.graph)
    for node_s, node_d in zip(param_nodes_s, param_nodes_d):
        sp = static_placement[node_s].output_specs.placements
        dp = dynamic_placement[node_d].output_specs.placements
        assert sp == dp, (
            f"Param placement mismatch for {node_s.name}: "
            f"static={sp} vs dynamic={dp}"
        )

    # Compare intermediate node placements by target op. The ILP may
    # choose different (but equivalent-cost) strategies for intermediate
    # nodes, so we check that both graphs have the same set of ops in
    # their placement solutions.
    from collections import Counter

    s_targets = Counter(
        str(n.target)
        for n in autop_s.gm.graph.nodes
        if n.op == "call_function" and n in static_placement
    )
    d_targets = Counter(
        str(n.target)
        for n in autop_d.gm.graph.nodes
        if n.op == "call_function" and n in dynamic_placement
    )
    # All static ops should appear in the dynamic graph (dynamic may have
    # a few extra from symbolic decompositions)
    for target, count in s_targets.items():
        assert (
            target in d_targets
        ), f"Op {target} in static graph ({count}x) missing from dynamic"


# ===== Uneven sharding integration tests =====


@apply_cuda_patches
def test_uneven_sharding_optimizer_includes_uneven_placements(device_mesh_2d):
    """The optimizer should include Shard placements for dims that don't evenly
    divide the mesh size, as long as the local shard is non-empty."""
    # dim1=100, TP=8: 100 % 8 = 4, uneven but non-empty (ceil(100/8)=13)
    dim1 = 100
    dim2 = 400

    with torch.device("meta"):
        model = FFN(dim1, dim2)

    def input_fn():
        bs = 8 * device_mesh_2d.shape[0]
        return torch.randn(bs, dim1, device="cuda")

    placement = (Shard(0), Replicate())

    with AutoParallel(model, input_fn, device_mesh_2d, dynamic=True) as autop:
        autop.add_input_constraints([placement])
        autop.add_output_constraints([placement])

        # Check that Shard(1) on TP dim is available for w1 [dim2, dim1]
        # dim2=400, TP=8: 400 % 8 = 0 (even, always available)
        # dim1=100, TP=8: 100 % 8 = 4 (uneven, should now be available)
        param_nodes = get_param_nodes(autop.sharding_optimizer.graph)
        w1_strat = autop.sharding_optimizer.strats[param_nodes[0]]
        w1_shape = param_nodes[0].meta["val"].shape

        tp_shard_placements = set()
        for s in w1_strat.strategies:
            tp_placement = s.output_specs.placements[1]
            if tp_placement.is_shard():
                tp_shard_placements.add(tp_placement.dim)

        # Both Shard(0) and Shard(1) should be available on TP
        assert 0 in tp_shard_placements, (
            f"Shard(0) on TP missing for w1 {w1_shape} "
            f"(available: {tp_shard_placements})"
        )
        assert 1 in tp_shard_placements, (
            f"Shard(1) on TP missing for w1 {w1_shape} — "
            f"uneven sharding not enabled? "
            f"(available: {tp_shard_placements})"
        )


@apply_cuda_patches
def test_uneven_sharding_optimize_and_apply(device_mesh_2d):
    """Full pipeline: optimize + apply_placement with forced uneven sharding."""
    dim1 = 100
    dim2 = 400

    with torch.device("meta"):
        model = FFN(dim1, dim2)

    def input_fn():
        bs = 8 * device_mesh_2d.shape[0]
        return torch.randn(bs, dim1, device="cuda")

    placement = (Shard(0), Replicate())

    with AutoParallel(
        model, input_fn, device_mesh_2d, dynamic=True, compile=False
    ) as autop:
        autop.add_input_constraints([placement])
        autop.add_output_constraints([placement])

        # Force an uneven shard: w1 is [400, 100], Shard(1) on TP=8
        # 100 % 8 = 4, so local size is ceil(100/8)=13 for most ranks
        param_nodes = get_param_nodes(autop.sharding_optimizer.graph)
        autop.sharding_optimizer.add_node_constraint(
            param_nodes[0], placement=(Replicate(), Shard(1))
        )

        sharding_placement = autop.optimize_placement(verbose=False)
        autop.apply_placement(sharding_placement)

    # Verify the parallel graph has symbolic dims for the uneven param
    pgm = autop.joint_with_descriptors.graph_module
    param_placeholder = next(
        n for n in pgm.graph.nodes if n.op == "placeholder" and n.name == "primals_1"
    )
    val = param_placeholder.meta["val"]
    # The unevenly-sharded dim should be symbolic
    has_symbolic = any(isinstance(s, torch.SymInt) for s in val.shape)
    assert has_symbolic, (
        f"Unevenly-sharded param should have symbolic dims but got "
        f"concrete shape {val.shape}"
    )


@apply_cuda_patches
def test_uneven_sharding_parallel_graph_has_pad_ops(device_mesh_2d):
    """The parallel graph should contain pad + slice ops for uneven sharding."""
    dim1 = 100
    dim2 = 400

    with torch.device("meta"):
        model = FFN(dim1, dim2)

    def input_fn():
        bs = 8 * device_mesh_2d.shape[0]
        return torch.randn(bs, dim1, device="cuda")

    placement = (Shard(0), Replicate())

    with AutoParallel(
        model, input_fn, device_mesh_2d, dynamic=True, compile=False
    ) as autop:
        autop.add_input_constraints([placement])
        autop.add_output_constraints([placement])

        # Force Shard(1) on TP for w1 [400, 100]: 100 % 8 = 4
        param_nodes = get_param_nodes(autop.sharding_optimizer.graph)
        autop.sharding_optimizer.add_node_constraint(
            param_nodes[0], placement=(Replicate(), Shard(1))
        )

        sharding_placement = autop.optimize_placement(verbose=False)
        autop.apply_placement(sharding_placement)

    pgm = autop.joint_with_descriptors.graph_module

    # Check for pad op (constant_pad_nd) in the parallel graph
    pad_ops = [
        n
        for n in pgm.graph.nodes
        if n.op == "call_function"
        and n.target == torch.ops.aten.constant_pad_nd.default
    ]
    assert (
        len(pad_ops) > 0
    ), "Expected constant_pad_nd ops for uneven sharding but found none"

    # Check for slice/narrow op (unpad after all-gather)
    slice_ops = [
        n
        for n in pgm.graph.nodes
        if n.op == "call_function"
        and n.target in (torch.ops.aten.slice.Tensor, torch.ops.aten.narrow.default)
    ]
    assert (
        len(slice_ops) > 0
    ), "Expected slice/narrow ops for unpadding after all-gather but found none"

    # The pad op should have a symbolic pad size argument
    pad_node = pad_ops[0]
    pad_args = pad_node.args[1]  # the pad list
    has_symbolic_pad = any(isinstance(a, torch.fx.Node) for a in pad_args)
    assert has_symbolic_pad, (
        f"Pad size should be symbolic (FX Node) for rank-independent graph, "
        f"but got concrete pad args: {pad_args}"
    )
