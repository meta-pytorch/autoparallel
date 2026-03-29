"""
Tests for ShardedLayout = hierarchical CuTe Layout (local, mesh) per dim.
hier_layout = logical_divide(Layout(tensor_shape), local_sizes) — view-invariant.
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
    max_common_layout,
    max_common_vector,
)
from autoparallel.shardings.cute.placement import ShardedLayout
from autoparallel.shardings.cute.propagation import (
    propagate_einsum,
    propagate_gather,
    propagate_permute,
    propagate_pointwise,
    propagate_reduction,
    propagate_slice,
    propagate_transpose,
    propagate_view,
)


class TestShardedLayout(unittest.TestCase):

    def test_replicate(self):
        t = ShardedLayout.replicate((4, 8, 16))
        self.assertTrue(t.is_replicate())
        self.assertEqual(t.tensor_shape, (4, 8, 16))

    def test_shard(self):
        t = ShardedLayout.shard((4, 8, 16), shard_dim=0, mesh_dim_size=2)
        self.assertFalse(t.is_replicate())
        placements = t.get_placements()
        self.assertEqual(placements[0], ("shard", 0, 2, 0))

    def test_shard_dim1(self):
        t = ShardedLayout.shard((4, 8, 16), shard_dim=1, mesh_dim_size=2)
        placements = t.get_placements()
        self.assertEqual(placements[0], ("shard", 1, 2, 0))

    def test_replicate_placements(self):
        t = ShardedLayout.replicate((4, 8, 16))
        placements = t.get_placements()
        self.assertEqual(placements, [("replicate", None, None)])

    def test_equality(self):
        t1 = ShardedLayout.shard((4, 8, 16), shard_dim=0, mesh_dim_size=2)
        t2 = ShardedLayout.shard((4, 8, 16), shard_dim=0, mesh_dim_size=2)
        self.assertEqual(t1, t2)

    def test_hier_layout(self):
        """hier_layout has hierarchical shape from logical_divide."""
        t = ShardedLayout.shard((4, 8, 16), shard_dim=1, mesh_dim_size=2)
        self.assertEqual(t.hier_layout.size(), 512)

    def test_shard_multi_s0s0(self):
        """S(0),S(0) on (8, 16) with mesh (2, 4)."""
        t = ShardedLayout.shard_multi((8, 16), [(0, 2), (0, 4)])
        self.assertFalse(t.is_replicate())
        placements = t.get_placements()
        self.assertEqual(placements[0][0], "shard")
        self.assertEqual(placements[0][1], 0)

    def test_shard_multi_s0s1(self):
        """S(0),S(1) on (4, 8, 16) with mesh (2, 4)."""
        t = ShardedLayout.shard_multi((4, 8, 16), [(0, 2), (1, 4)])
        placements = t.get_placements()
        self.assertEqual(placements[0], ("shard", 0, 2, 0))
        self.assertEqual(placements[1], ("shard", 1, 4, 1))

    def test_shard_multi_view_invariant(self):
        """S(0),S(1) through view: sharding preserved."""
        t = ShardedLayout.shard_multi((4, 8, 16), [(0, 2), (1, 4)])
        v = propagate_view(t, (32, 16))
        self.assertIsNotNone(v)
        # After merge, both mesh modes are on dim 0
        placements = v.get_placements()
        self.assertEqual(placements[0][0], "shard")
        self.assertEqual(placements[0][1], 0)


class TestViewPropagation(unittest.TestCase):

    def test_flatten(self):
        t = ShardedLayout.shard((4, 8, 16), shard_dim=0, mesh_dim_size=2)
        out = propagate_view(t, (32, 16))
        self.assertIsNotNone(out)
        self.assertEqual(out.tensor_shape, (32, 16))
        # Placements preserved
        self.assertEqual(out.get_placements()[0][0], "shard")

    def test_unflatten(self):
        t = ShardedLayout.shard((32, 16), shard_dim=0, mesh_dim_size=2)
        out = propagate_view(t, (4, 8, 16))
        self.assertIsNotNone(out)
        self.assertEqual(out.tensor_shape, (4, 8, 16))

    def test_replicate_through_view(self):
        t = ShardedLayout.replicate((4, 8, 16))
        out = propagate_view(t, (32, 16))
        self.assertIsNotNone(out)
        self.assertTrue(out.is_replicate())

    def test_round_trip(self):
        original = ShardedLayout.shard((4, 8, 16), shard_dim=1, mesh_dim_size=2)
        after_flat = propagate_view(original, (32, 16))
        after_unflat = propagate_view(after_flat, (4, 8, 16))
        # Placements round-trip
        self.assertEqual(original.get_placements(), after_unflat.get_placements())

    def test_incompatible(self):
        t = ShardedLayout.shard((4, 8, 16), shard_dim=0, mesh_dim_size=2)
        self.assertIsNone(propagate_view(t, (100,)))

    def test_hier_layout_invariant(self):
        """Placements preserved through compatible view chain."""
        t = ShardedLayout.shard((4, 8, 16), shard_dim=1, mesh_dim_size=4)
        orig_placements = t.get_placements()
        # View to (32, 16): merge dims 0,1 — compatible
        v1 = propagate_view(t, (32, 16))
        self.assertIsNotNone(v1)
        # View to (2, 16, 16): split — mesh=4 goes to dim 1 (size 16)
        v2 = propagate_view(v1, (2, 16, 16))
        self.assertIsNotNone(v2)
        self.assertEqual(v2.get_placements()[0], ("shard", 1, 4, 0))
        # View back to (4, 8, 16)
        v3 = propagate_view(v2, (4, 8, 16))
        self.assertIsNotNone(v3)
        self.assertEqual(v3.get_placements(), orig_placements)


class TestTransposePropagation(unittest.TestCase):

    def test_transpose(self):
        t = ShardedLayout.shard((4, 8, 16), shard_dim=0, mesh_dim_size=2)
        out = propagate_transpose(t, 0, 1)
        self.assertEqual(out.tensor_shape, (8, 4, 16))
        self.assertNotEqual(out.hier_layout, t.hier_layout)

    def test_transpose_replicate(self):
        t = ShardedLayout.replicate((4, 8))
        out = propagate_transpose(t, 0, 1)
        self.assertTrue(out.is_replicate())


class TestPermutePropagation(unittest.TestCase):

    def test_permute_3d(self):
        t = ShardedLayout.shard((4, 8, 16), shard_dim=0, mesh_dim_size=2)
        out = propagate_permute(t, (2, 0, 1))
        self.assertEqual(out.tensor_shape, (16, 4, 8))

    def test_permute_round_trip(self):
        t = ShardedLayout.shard((4, 8, 16), shard_dim=1, mesh_dim_size=2)
        p1 = propagate_permute(t, (2, 0, 1))
        p2 = propagate_permute(p1, (1, 2, 0))
        self.assertEqual(p2, t)


class TestSlicePropagation(unittest.TestCase):

    def test_slice_non_sharded_dim(self):
        t = ShardedLayout.shard((4, 8, 16), shard_dim=0, mesh_dim_size=2)
        out = propagate_slice(t, dim=1, index=3)
        self.assertIsNotNone(out)
        placements = out.get_placements()
        self.assertEqual(placements[0], ("shard", 0, 2, 0))

    def test_slice_sharded_dim(self):
        t = ShardedLayout.shard((4, 8, 16), shard_dim=0, mesh_dim_size=2)
        out = propagate_slice(t, dim=0, index=2)
        self.assertIsNone(out)

    def test_slice_replicate(self):
        t = ShardedLayout.replicate((4, 8, 16))
        out = propagate_slice(t, dim=1, index=3)
        self.assertIsNotNone(out)
        self.assertTrue(out.is_replicate())

    def test_slice_dim_before_shard(self):
        t = ShardedLayout.shard((4, 8, 16), shard_dim=1, mesh_dim_size=2)
        out = propagate_slice(t, dim=0, index=2)
        self.assertIsNotNone(out)
        placements = out.get_placements()
        self.assertEqual(placements[0], ("shard", 0, 2, 0))  # dim 1 -> dim 0


class TestGatherPropagation(unittest.TestCase):

    def test_gather_non_sharded(self):
        t = ShardedLayout.shard((4, 8, 16), shard_dim=0, mesh_dim_size=2)
        index = Layout(4, 1)
        out = propagate_gather(t, dim=1, index_layout=index)
        self.assertIsNotNone(out)

    def test_gather_sharded_dim(self):
        t = ShardedLayout.shard((4, 8, 16), shard_dim=1, mesh_dim_size=2)
        index = Layout(4, 1)
        out = propagate_gather(t, dim=1, index_layout=index)
        self.assertIsNone(out)

    def test_gather_replicate(self):
        t = ShardedLayout.replicate((4, 8, 16))
        index = Layout(4, 2)
        out = propagate_gather(t, dim=1, index_layout=index)
        self.assertIsNotNone(out)
        self.assertTrue(out.is_replicate())


class TestEinsumPropagation(unittest.TestCase):

    def test_both_replicate(self):
        a = ShardedLayout.replicate((16, 8))
        b = ShardedLayout.replicate((8, 32))
        out = propagate_einsum("mk,kn->mn", a, b, (16, 32))
        self.assertIsNotNone(out)
        self.assertTrue(out.is_replicate())

    def test_m_shard(self):
        a = ShardedLayout.shard((16, 8), shard_dim=0, mesh_dim_size=2)
        b = ShardedLayout.replicate((8, 32))
        out = propagate_einsum("mk,kn->mn", a, b, (16, 32))
        self.assertIsNotNone(out)
        placements = out.get_placements()
        self.assertEqual(placements[0], ("shard", 0, 2, 0))

    def test_n_shard(self):
        a = ShardedLayout.replicate((16, 8))
        b = ShardedLayout.shard((8, 32), shard_dim=1, mesh_dim_size=2)
        out = propagate_einsum("mk,kn->mn", a, b, (16, 32))
        self.assertIsNotNone(out)
        placements = out.get_placements()
        self.assertEqual(placements[0], ("shard", 1, 2, 0))

    def test_k_shard_both(self):
        a = ShardedLayout.shard((16, 8), shard_dim=1, mesh_dim_size=2)
        b = ShardedLayout.shard((8, 32), shard_dim=0, mesh_dim_size=2)
        out = propagate_einsum("mk,kn->mn", a, b, (16, 32))
        self.assertIsNotNone(out)
        self.assertTrue(hasattr(out, "_is_partial") and out._is_partial)

    def test_k_shard_only_a(self):
        a = ShardedLayout.shard((16, 8), shard_dim=1, mesh_dim_size=2)
        b = ShardedLayout.replicate((8, 32))
        out = propagate_einsum("mk,kn->mn", a, b, (16, 32))
        self.assertIsNone(out)

    def test_batch_both(self):
        a = ShardedLayout.shard((4, 8, 16), shard_dim=0, mesh_dim_size=2)
        b = ShardedLayout.shard((4, 16, 32), shard_dim=0, mesh_dim_size=2)
        out = propagate_einsum("bmk,bkn->bmn", a, b, (4, 8, 32))
        self.assertIsNotNone(out)
        placements = out.get_placements()
        self.assertEqual(placements[0], ("shard", 0, 2, 0))

    def test_batch_a_only(self):
        a = ShardedLayout.shard((4, 8, 16), shard_dim=0, mesh_dim_size=2)
        b = ShardedLayout.replicate((4, 16, 32))
        out = propagate_einsum("bmk,bkn->bmn", a, b, (4, 8, 32))
        self.assertIsNone(out)


class TestPointwisePropagation(unittest.TestCase):

    def test_matching(self):
        a = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=2)
        b = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=2)
        out = propagate_pointwise([a, b], (8, 16))
        self.assertIsNotNone(out)

    def test_mismatch(self):
        # Same mesh dim sharding different tensor dims — incompatible
        a = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=2)
        b = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=4)
        out = propagate_pointwise([a, b], (8, 16))
        # Different mesh sizes on same dim → broadcast rejects (both have mesh)
        self.assertIsNone(out)

    def test_mismatch_same_mesh_dim_different_tensor_dims(self):
        # A shards dim 0 on mesh dim 0, B shards dim 1 on mesh dim 0
        # Same mesh dim assigned to different tensor dims → incompatible
        a = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=2, mesh_dim=0)
        b = ShardedLayout.shard((8, 16), shard_dim=1, mesh_dim_size=2, mesh_dim=0)
        out = propagate_pointwise([a, b], (8, 16))
        self.assertIsNone(out)

    def test_shard_with_replicate(self):
        a = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=2)
        b = ShardedLayout.replicate((8, 16))
        out = propagate_pointwise([a, b], (8, 16))
        self.assertIsNotNone(out)

    def test_all_replicate(self):
        a = ShardedLayout.replicate((8, 16))
        b = ShardedLayout.replicate((8, 16))
        out = propagate_pointwise([a, b], (8, 16))
        self.assertTrue(out.is_replicate())


class TestReductionPropagation(unittest.TestCase):

    def test_reduce_non_sharded(self):
        t = ShardedLayout.shard((8, 16), shard_dim=1, mesh_dim_size=2)
        out = propagate_reduction(t, reduce_dim=0, keepdim=False, output_shape=(16,))
        placements = out.get_placements()
        self.assertEqual(placements[0], ("shard", 0, 2, 0))

    def test_reduce_sharded(self):
        t = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=2)
        out = propagate_reduction(t, reduce_dim=0, keepdim=False, output_shape=(16,))
        self.assertTrue(hasattr(out, "_is_partial") and out._is_partial)

    def test_reduce_replicate(self):
        t = ShardedLayout.replicate((8, 16))
        out = propagate_reduction(t, reduce_dim=0, keepdim=False, output_shape=(16,))
        self.assertTrue(out.is_replicate())


class TestEndToEnd(unittest.TestCase):

    def test_linear_3d(self):
        """view -> mm -> view with shard preserved."""
        B, S, H, O = 4, 8, 16, 32
        input_t = ShardedLayout.shard((B, S, H), shard_dim=0, mesh_dim_size=2)

        after_v1 = propagate_view(input_t, (B * S, H))
        self.assertIsNotNone(after_v1)

        weight = ShardedLayout.replicate((H, O))
        after_mm = propagate_einsum("mk,kn->mn", after_v1, weight, (B * S, O))
        self.assertIsNotNone(after_mm)

        after_v2 = propagate_view(after_mm, (B, S, O))
        self.assertIsNotNone(after_v2)
        placements = after_v2.get_placements()
        self.assertEqual(placements[0], ("shard", 0, 2, 0))

    def test_linear_3d_both_dims_sharded(self):
        """
        (B, S, H) sharded on dim 0 AND dim 1 across 2D mesh.
        view(B*S, H) -> mm -> view(B, S, O)
        Both shardings should be preserved through the entire chain.
        This is the KEY motivating example for CuTe layouts.
        """
        B, S, H, O = 4, 8, 16, 32
        dp_size, sp_size = 2, 4

        input_t = ShardedLayout.shard_multi(
            (B, S, H), [(0, dp_size), (1, sp_size)]
        )
        placements_in = input_t.get_placements()
        self.assertEqual(placements_in[0], ("shard", 0, dp_size, 0))
        self.assertEqual(placements_in[1], ("shard", 1, sp_size, 1))

        # Step 1: view (B, S, H) -> (B*S, H)
        after_v1 = propagate_view(input_t, (B * S, H))
        self.assertIsNotNone(after_v1)

        # Step 2: mm (B*S, H) @ (H, O) -> (B*S, O)
        weight = ShardedLayout.replicate((H, O))
        after_mm = propagate_einsum("mk,kn->mn", after_v1, weight, (B * S, O))
        self.assertIsNotNone(after_mm)

        # Step 3: view (B*S, O) -> (B, S, O)
        after_v2 = propagate_view(after_mm, (B, S, O))
        self.assertIsNotNone(after_v2)

        # Both shardings recovered!
        placements_out = after_v2.get_placements()
        self.assertEqual(placements_out[0], ("shard", 0, dp_size, 0))
        self.assertEqual(placements_out[1], ("shard", 1, sp_size, 1))


class TestScaledBasis(unittest.TestCase):

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


class TestMaxCommon(unittest.TestCase):
    """max_common_layout and max_common_vector."""

    def test_same_layout(self):
        a = Layout((4, 8), (8, 1))
        self.assertEqual(max_common_vector(a, a), 32)

    def test_col_vs_row_major(self):
        a = Layout((4, 4), (1, 4))
        b = Layout((4, 4), (4, 1))
        self.assertEqual(max_common_vector(a, b), 4)

    def test_col_major_vs_padded(self):
        a = Layout((4, 8), (1, 4))
        b = Layout((4, 8), (1, 5))
        self.assertEqual(max_common_vector(a, b), 4)

    def test_contiguous_vs_strided(self):
        a = Layout(16, 1)
        b = Layout(16, 2)
        self.assertEqual(max_common_vector(a, b), 1)

    def test_max_common_layout_returns_layout(self):
        a = Layout((4, 4), (1, 4))
        b = Layout((4, 4), (4, 1))
        mcl = max_common_layout(a, b)
        self.assertEqual(mcl.size(), 4)
        for i in range(mcl.size()):
            self.assertEqual(b(mcl(i)), i)


if __name__ == "__main__":
    unittest.main()
