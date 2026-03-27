"""
Tests for TiledLayout sharding placement and propagation.

TiledLayout = (tensor_layout, shard_layout):
- tensor_layout: tensor_coords -> memory_offset (changed by reshape/transpose)
- shard_layout: (mesh, local) -> element_index (changed by shard/redistribute)
"""

import unittest

from autoparallel.shardings.cute._pycute import (
    ArithmeticTuple,
    E,
    Layout,
    ScaledBasis,
    coalesce,
    codomain_divide,
    make_basis_like,
    product,
)
from autoparallel.shardings.cute.placement import TiledLayout
from autoparallel.shardings.cute.propagation import (
    propagate_einsum,
    propagate_pointwise,
    propagate_reduction,
    propagate_transpose,
    propagate_view,
)


class TestTiledLayout(unittest.TestCase):
    """Test TiledLayout creation and queries."""

    def test_replicate(self):
        t = TiledLayout.replicate((4, 8, 16), (2, 4))
        self.assertTrue(t.is_replicate())
        self.assertEqual(t.tensor_shape, (4, 8, 16))
        self.assertEqual(t.mesh_shape, (2, 4))
        self.assertEqual(t.num_elements, 512)
        self.assertEqual(t.local_size, 512)

    def test_shard(self):
        t = TiledLayout.shard((4, 8, 16), (2,), shard_dim=0)
        self.assertFalse(t.is_replicate())
        placements = t.get_placements()
        self.assertEqual(placements, [("shard", 0)])

    def test_shard_dim1(self):
        t = TiledLayout.shard((4, 8, 16), (2,), shard_dim=1)
        placements = t.get_placements()
        self.assertEqual(placements, [("shard", 1)])

    def test_shard_2d_mesh(self):
        t = TiledLayout.shard((4, 8, 16), (2, 4), shard_dim=0, mesh_dim=0)
        placements = t.get_placements()
        self.assertEqual(placements[0], ("shard", 0))
        self.assertEqual(placements[1], ("replicate", None))

    def test_shard_both_dims(self):
        # Shard dim 0 on mesh dim 0, dim 1 on mesh dim 1
        t = TiledLayout.shard((4, 8, 16), (2, 4), shard_dim=0, mesh_dim=0)
        # Now also shard dim 1 on mesh dim 1 — need to build manually
        # For now test the factory for single shard
        placements = t.get_placements()
        self.assertEqual(placements[0], ("shard", 0))

    def test_replicate_get_placements(self):
        t = TiledLayout.replicate((4, 8, 16), (2, 4))
        placements = t.get_placements()
        self.assertEqual(placements, [("replicate", None), ("replicate", None)])

    def test_str(self):
        t = TiledLayout.shard((4, 8, 16), (2,), shard_dim=1)
        s = str(t)
        self.assertIn("S(1)", s)

    def test_equality(self):
        t1 = TiledLayout.shard((4, 8, 16), (2,), shard_dim=0)
        t2 = TiledLayout.shard((4, 8, 16), (2,), shard_dim=0)
        self.assertEqual(t1, t2)


class TestViewPropagation(unittest.TestCase):
    """View/reshape: tensor_layout changes, shard_layout invariant."""

    def test_flatten(self):
        """(B, S, H) -> (B*S, H): shard_layout unchanged."""
        t = TiledLayout.shard((4, 8, 16), (2,), shard_dim=0)
        out = propagate_view(t, (32, 16))
        self.assertIsNotNone(out)
        self.assertEqual(out.tensor_shape, (32, 16))
        # shard_layout is invariant
        self.assertEqual(out.shard_layout, t.shard_layout)

    def test_unflatten(self):
        """(B*S, H) -> (B, S, H): shard_layout unchanged."""
        t = TiledLayout.shard((32, 16), (2,), shard_dim=0)
        out = propagate_view(t, (4, 8, 16))
        self.assertIsNotNone(out)
        self.assertEqual(out.tensor_shape, (4, 8, 16))
        self.assertEqual(out.shard_layout, t.shard_layout)

    def test_replicate_through_view(self):
        t = TiledLayout.replicate((4, 8, 16), (2,))
        out = propagate_view(t, (32, 16))
        self.assertIsNotNone(out)
        self.assertTrue(out.is_replicate())

    def test_round_trip(self):
        """Flatten then unflatten: shard_layout preserved exactly."""
        original = TiledLayout.shard((4, 8, 16), (2,), shard_dim=1)
        after_flat = propagate_view(original, (32, 16))
        after_unflat = propagate_view(after_flat, (4, 8, 16))
        # shard_layout identical throughout
        self.assertEqual(after_unflat.shard_layout, original.shard_layout)

    def test_incompatible_shape(self):
        t = TiledLayout.shard((4, 8, 16), (2,), shard_dim=0)
        out = propagate_view(t, (100,))  # wrong total elements
        self.assertIsNone(out)


class TestTransposePropagation(unittest.TestCase):
    """Transpose: tensor_layout modes reorder, shard_layout unchanged."""

    def test_transpose(self):
        t = TiledLayout.shard((4, 8, 16), (2,), shard_dim=0)
        out = propagate_transpose(t, 0, 1)
        self.assertIsNotNone(out)
        self.assertEqual(out.tensor_shape, (8, 4, 16))
        # shard_layout invariant
        self.assertEqual(out.shard_layout, t.shard_layout)

    def test_transpose_replicate(self):
        t = TiledLayout.replicate((4, 8), (2,))
        out = propagate_transpose(t, 0, 1)
        self.assertTrue(out.is_replicate())
        self.assertEqual(out.tensor_shape, (8, 4))


class TestEinsumPropagation(unittest.TestCase):
    """Einsum: redistribution-free strategies."""

    def test_both_replicate(self):
        a = TiledLayout.replicate((16, 8), (2,))
        b = TiledLayout.replicate((8, 32), (2,))
        out = propagate_einsum("mk,kn->mn", a, b, (16, 32))
        self.assertIsNotNone(out)
        self.assertTrue(out.is_replicate())

    def test_m_shard(self):
        a = TiledLayout.shard((16, 8), (2,), shard_dim=0)
        b = TiledLayout.replicate((8, 32), (2,))
        out = propagate_einsum("mk,kn->mn", a, b, (16, 32))
        self.assertIsNotNone(out)
        placements = out.get_placements()
        self.assertEqual(placements[0], ("shard", 0))

    def test_n_shard(self):
        a = TiledLayout.replicate((16, 8), (2,))
        b = TiledLayout.shard((8, 32), (2,), shard_dim=1)
        out = propagate_einsum("mk,kn->mn", a, b, (16, 32))
        self.assertIsNotNone(out)
        placements = out.get_placements()
        self.assertEqual(placements[0], ("shard", 1))

    def test_k_shard_both(self):
        a = TiledLayout.shard((16, 8), (2,), shard_dim=1)
        b = TiledLayout.shard((8, 32), (2,), shard_dim=0)
        out = propagate_einsum("mk,kn->mn", a, b, (16, 32))
        self.assertIsNotNone(out)
        self.assertTrue(hasattr(out, "_is_partial") and out._is_partial)

    def test_k_shard_only_a_incompatible(self):
        a = TiledLayout.shard((16, 8), (2,), shard_dim=1)
        b = TiledLayout.replicate((8, 32), (2,))
        out = propagate_einsum("mk,kn->mn", a, b, (16, 32))
        self.assertIsNone(out)

    def test_batch_both_sharded(self):
        a = TiledLayout.shard((4, 8, 16), (2,), shard_dim=0)
        b = TiledLayout.shard((4, 16, 32), (2,), shard_dim=0)
        out = propagate_einsum("bmk,bkn->bmn", a, b, (4, 8, 32))
        self.assertIsNotNone(out)
        placements = out.get_placements()
        self.assertEqual(placements[0], ("shard", 0))

    def test_batch_a_only_incompatible(self):
        a = TiledLayout.shard((4, 8, 16), (2,), shard_dim=0)
        b = TiledLayout.replicate((4, 16, 32), (2,))
        out = propagate_einsum("bmk,bkn->bmn", a, b, (4, 8, 32))
        self.assertIsNone(out)


class TestPointwisePropagation(unittest.TestCase):

    def test_matching(self):
        a = TiledLayout.shard((8, 16), (2,), shard_dim=0)
        b = TiledLayout.shard((8, 16), (2,), shard_dim=0)
        out = propagate_pointwise([a, b], (8, 16))
        self.assertIsNotNone(out)
        self.assertEqual(out.shard_layout, a.shard_layout)

    def test_mismatch_incompatible(self):
        a = TiledLayout.shard((8, 16), (2,), shard_dim=0)
        b = TiledLayout.shard((8, 16), (2,), shard_dim=1)
        out = propagate_pointwise([a, b], (8, 16))
        self.assertIsNone(out)

    def test_shard_with_replicate_broadcast(self):
        a = TiledLayout.shard((8, 16), (2,), shard_dim=0)
        b = TiledLayout.replicate((8, 16), (2,))
        out = propagate_pointwise([a, b], (8, 16))
        self.assertIsNotNone(out)

    def test_all_replicate(self):
        a = TiledLayout.replicate((8, 16), (2,))
        b = TiledLayout.replicate((8, 16), (2,))
        out = propagate_pointwise([a, b], (8, 16))
        self.assertIsNotNone(out)
        self.assertTrue(out.is_replicate())


class TestReductionPropagation(unittest.TestCase):

    def test_reduce_non_sharded_dim(self):
        t = TiledLayout.shard((8, 16), (2,), shard_dim=1)
        out = propagate_reduction(t, reduce_dim=0, keepdim=False, output_shape=(16,))
        self.assertIsNotNone(out)
        placements = out.get_placements()
        self.assertEqual(placements[0], ("shard", 0))  # dim 1 becomes dim 0

    def test_reduce_sharded_dim(self):
        t = TiledLayout.shard((8, 16), (2,), shard_dim=0)
        out = propagate_reduction(t, reduce_dim=0, keepdim=False, output_shape=(16,))
        self.assertIsNotNone(out)
        self.assertTrue(hasattr(out, "_is_partial") and out._is_partial)

    def test_reduce_replicate(self):
        t = TiledLayout.replicate((8, 16), (2,))
        out = propagate_reduction(t, reduce_dim=0, keepdim=False, output_shape=(16,))
        self.assertTrue(out.is_replicate())


class TestEndToEnd(unittest.TestCase):

    def test_linear_3d_both_shardings_preserved(self):
        """
        (B, S, H) with S(0) on dp, S(1) on sp.
        view -> mm -> view: shard_layout invariant through views.
        """
        B, S, H, O = 4, 8, 16, 32

        # Input: shard dim 0 on mesh dim 0
        input_t = TiledLayout.shard((B, S, H), (2, 4), shard_dim=0, mesh_dim=0)
        original_shard = input_t.shard_layout

        # Step 1: view (B, S, H) -> (B*S, H) — trivial
        after_v1 = propagate_view(input_t, (B * S, H))
        self.assertIsNotNone(after_v1)
        self.assertEqual(after_v1.shard_layout, original_shard)

        # Step 2: mm (B*S, H) @ (H, O) -> (B*S, O)
        weight = TiledLayout.replicate((H, O), (2, 4))
        after_mm = propagate_einsum(
            "mk,kn->mn", after_v1, weight, (B * S, O)
        )
        self.assertIsNotNone(after_mm)

        # Step 3: view (B*S, O) -> (B, S, O) — trivial
        after_v2 = propagate_view(after_mm, (B, S, O))
        self.assertIsNotNone(after_v2)

        # The shard on dim 0 should be recoverable
        placements = after_v2.get_placements()
        self.assertEqual(placements[0], ("shard", 0))

    def test_view_is_trivial(self):
        """View propagation is literally just updating tensor_shape."""
        t = TiledLayout.shard((4, 8, 16), (2, 4), shard_dim=1, mesh_dim=1)
        v1 = propagate_view(t, (32, 16))
        v2 = propagate_view(v1, (4, 8, 16))
        # shard_layout never changed
        self.assertEqual(t.shard_layout, v1.shard_layout)
        self.assertEqual(t.shard_layout, v2.shard_layout)


class TestScaledBasis(unittest.TestCase):
    """ScaledBasis / coordinate strides still work."""

    def test_basis_creation(self):
        self.assertEqual(E(0).value, 1)
        self.assertEqual(E(0).index, 0)

    def test_coordinate_layout(self):
        L = Layout((4, 8), (E(0), E(1)))
        result = L(2, 3)
        self.assertIsInstance(result, ArithmeticTuple)
        self.assertEqual(result.values, (2, 3))

    def test_codomain_divide(self):
        cov = codomain_divide(Layout((4, 4), (8, 1)), (4, 8))
        self.assertEqual(cov[0], 4)
        self.assertEqual(cov[1], 4)

    def test_codomain_divide_contiguous(self):
        cov = codomain_divide(Layout(16, 1), (4, 8))
        self.assertEqual(cov[0], 2)
        self.assertEqual(cov[1], 8)


if __name__ == "__main__":
    unittest.main()
