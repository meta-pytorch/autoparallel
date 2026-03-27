"""
Tests for CuTe-based sharding placement and propagation.

All propagation rules assume inputs are already in the given placements
(no redistribution). They return output placements or None if incompatible.

Layout convention: rank-2 (device, local...) -> flat_offset.
"""

import unittest

from torch.distributed._pycute import Layout, coalesce
from autoparallel.shardings.cute.placement import CutePlacement
from autoparallel.shardings.cute.propagation import (
    propagate_einsum,
    propagate_pointwise,
    propagate_reduction,
    propagate_view,
)


class TestPycuteBasics(unittest.TestCase):
    """Sanity checks for torch.distributed._pycute."""

    def test_layout_creation(self):
        L = Layout(4, 2)
        self.assertEqual(L.size(), 4)
        self.assertEqual(L(0), 0)
        self.assertEqual(L(1), 2)

    def test_layout_2d(self):
        L = Layout((4, 8), (8, 1))  # row-major 4x8
        self.assertEqual(L.size(), 32)
        self.assertEqual(L(0, 0), 0)
        self.assertEqual(L(1, 0), 8)
        self.assertEqual(L(0, 1), 1)

    def test_coalesce(self):
        L = Layout((2, 6), (6, 1))
        c = coalesce(L)
        self.assertEqual(c.shape, 12)
        self.assertEqual(c.stride, 1)

    def test_stride_zero(self):
        L = Layout(4, 0)
        for i in range(4):
            self.assertEqual(L(i), 0)


class TestCutePlacement(unittest.TestCase):
    """Test CutePlacement with rank-2 (device, local) convention."""

    def test_replicate(self):
        p = CutePlacement.replicate(4)
        self.assertTrue(p.is_replicate())
        self.assertFalse(p.is_shard())
        self.assertEqual(p.mesh_dim_size, 4)
        self.assertEqual(str(p), "R")

    def test_shard(self):
        p = CutePlacement.shard(dim=1, tensor_dim_size=16, mesh_dim_size=4)
        self.assertFalse(p.is_replicate())
        self.assertTrue(p.is_shard())
        self.assertEqual(p.dim, 1)
        self.assertEqual(p.mesh_dim_size, 4)
        self.assertEqual(p.local_size, 4)
        # Layout: (device, local) -> flat. Device 0 holds [0..3], device 1 holds [4..7]
        self.assertEqual(p.layout(0, 0), 0)
        self.assertEqual(p.layout(1, 0), 4)
        self.assertEqual(p.layout(0, 3), 3)

    def test_to_placement_shard(self):
        p = CutePlacement.shard(0, 8, 2)
        tp = p.to_placement()
        self.assertIsNotNone(tp)
        self.assertTrue(tp.is_shard())
        self.assertEqual(tp.dim, 0)

    def test_to_placement_replicate(self):
        p = CutePlacement.replicate(4)
        tp = p.to_placement()
        self.assertIsNotNone(tp)
        self.assertTrue(tp.is_replicate())

    def test_complex_not_shard(self):
        # Hierarchical layout from flatten — not a simple shard
        p = CutePlacement(dim=0, layout=Layout((2, 4, 4), (4, 8, 1)))
        self.assertFalse(p.is_replicate())
        self.assertFalse(p.is_shard())
        self.assertIsNone(p.to_placement())

    def test_equality(self):
        p1 = CutePlacement.shard(1, 16, 4)
        p2 = CutePlacement.shard(1, 16, 4)
        self.assertEqual(p1, p2)

    def test_from_placement_shard(self):
        from torch.distributed.tensor.placement_types import Shard
        p = CutePlacement.from_placement(Shard(1), 16, 4)
        self.assertTrue(p.is_shard())
        self.assertEqual(p.dim, 1)

    def test_from_placement_replicate(self):
        from torch.distributed.tensor.placement_types import Replicate
        p = CutePlacement.from_placement(Replicate(), 16, 4)
        self.assertTrue(p.is_replicate())


class TestViewPropagation(unittest.TestCase):
    """Test view/reshape propagation — all redistribution-free."""

    def test_identity_view(self):
        placements = (CutePlacement.shard(0, 8, 2),)
        out = propagate_view(placements, (8, 4), (8, 4), (2,))
        self.assertIsNotNone(out)
        self.assertTrue(out[0].is_shard())
        self.assertEqual(out[0].dim, 0)

    def test_replicate_through_view(self):
        placements = (CutePlacement.replicate(4),)
        out = propagate_view(placements, (8, 4), (32,), (4,))
        self.assertIsNotNone(out)
        self.assertTrue(out[0].is_replicate())

    def test_flatten_leftmost_shard(self):
        """(B, S, H) -> (B*S, H) with Shard(0): contiguous shard survives."""
        placements = (CutePlacement.shard(0, 4, 2),)
        out = propagate_view(placements, (4, 8, 16), (32, 16), (2,))
        self.assertIsNotNone(out)
        self.assertEqual(out[0].dim, 0)
        self.assertTrue(out[0].is_shard())
        self.assertEqual(out[0].local_size, 16)  # B/D * S = 2*8

    def test_flatten_non_leftmost_shard(self):
        """(B, S, H) -> (B*S, H) with Shard(1): CuTe strided shard.

        KEY TEST: DTensor can't handle this, CuTe can.
        """
        B, S, H, D = 2, 8, 16, 4
        placements = (CutePlacement.shard(1, S, D),)
        out = propagate_view(placements, (B, S, H), (B * S, H), (D,))

        self.assertIsNotNone(out)
        self.assertEqual(out[0].dim, 0)
        self.assertFalse(out[0].is_shard())
        self.assertFalse(out[0].is_replicate())
        self.assertEqual(out[0].local_size, B * S // D)

        # Verify device 0 holds correct flat indices
        device0_indices = sorted(
            out[0].layout(0, i) for i in range(out[0].local_size)
        )
        expected = sorted(b * S + s for b in range(B) for s in range(S // D))
        self.assertEqual(device0_indices, expected)

    def test_flatten_shard_2d_mesh(self):
        """(B, S, H) -> (B*S, H) with 2D mesh [Shard(0), Shard(1)]."""
        B, S, H = 4, 8, 16
        placements = (
            CutePlacement.shard(0, B, 2),
            CutePlacement.shard(1, S, 4),
        )
        out = propagate_view(placements, (B, S, H), (B * S, H), (2, 4))
        self.assertIsNotNone(out)
        self.assertTrue(out[0].is_shard())
        self.assertFalse(out[1].is_shard())
        self.assertFalse(out[1].is_replicate())

    def test_split_shard(self):
        """(B*S, H) -> (B, S, H) with Shard(0): shard on first piece."""
        placements = (CutePlacement.shard(0, 32, 2),)
        out = propagate_view(placements, (32, 16), (4, 8, 16), (2,))
        self.assertIsNotNone(out)
        self.assertEqual(out[0].dim, 0)
        self.assertTrue(out[0].is_shard())

    def test_unsqueeze_middle(self):
        """S(1) on (8, 4) -> (8, 1, 4): shard skips size-1 dim."""
        placements = (CutePlacement.shard(1, 4, 2),)
        out = propagate_view(placements, (8, 4), (8, 1, 4), (2,))
        self.assertIsNotNone(out)
        self.assertEqual(out[0].dim, 2)
        self.assertTrue(out[0].is_shard())

    def test_unsqueeze_front(self):
        """S(0) on (8, 4) -> (1, 8, 4): shard skips size-1 dim."""
        placements = (CutePlacement.shard(0, 8, 2),)
        out = propagate_view(placements, (8, 4), (1, 8, 4), (2,))
        self.assertIsNotNone(out)
        self.assertEqual(out[0].dim, 1)
        self.assertTrue(out[0].is_shard())

    def test_non_divisible_returns_none(self):
        p = CutePlacement(dim=1, layout=Layout((3, 1), (1, 1)))
        out = propagate_view((p,), (4, 5), (20,), (3,))
        self.assertIsNone(out)

    def test_round_trip_shard0(self):
        """S(0): flatten then unflatten recovers original."""
        B, S, H, D = 4, 8, 16, 2
        original = (CutePlacement.shard(0, B, D),)
        after_flat = propagate_view(original, (B, S, H), (B * S, H), (D,))
        self.assertIsNotNone(after_flat)
        after_unflat = propagate_view(after_flat, (B * S, H), (B, S, H), (D,))
        self.assertIsNotNone(after_unflat)
        self.assertEqual(after_unflat[0].dim, 0)
        self.assertTrue(after_unflat[0].is_shard())

    def test_round_trip_shard1(self):
        """S(1): flatten then unflatten recovers original."""
        B, S, H, D = 4, 8, 16, 2
        original = (CutePlacement.shard(1, S, D),)
        after_flat = propagate_view(original, (B, S, H), (B * S, H), (D,))
        self.assertIsNotNone(after_flat)
        after_unflat = propagate_view(after_flat, (B * S, H), (B, S, H), (D,))
        self.assertIsNotNone(after_unflat)
        self.assertEqual(after_unflat[0].dim, 1)
        self.assertTrue(after_unflat[0].is_shard())


class TestEinsumPropagation(unittest.TestCase):
    """Test einsum/matmul — redistribution-free only."""

    def test_both_replicate(self):
        D = 2
        R = CutePlacement.replicate(D)
        out = propagate_einsum("mk,kn->mn", (R,), (R,), (16, 8), (8, 32), (D,))
        self.assertIsNotNone(out)
        self.assertTrue(out[0].is_replicate())

    def test_m_shard(self):
        D = 2
        pa = (CutePlacement.shard(0, 16, D),)
        pb = (CutePlacement.replicate(D),)
        out = propagate_einsum("mk,kn->mn", pa, pb, (16, 8), (8, 32), (D,))
        self.assertIsNotNone(out)
        self.assertEqual(out[0].dim, 0)

    def test_n_shard(self):
        D = 2
        pa = (CutePlacement.replicate(D),)
        pb = (CutePlacement.shard(1, 32, D),)
        out = propagate_einsum("mk,kn->mn", pa, pb, (16, 8), (8, 32), (D,))
        self.assertIsNotNone(out)
        self.assertEqual(out[0].dim, 1)

    def test_k_shard_both(self):
        D = 2
        pa = (CutePlacement.shard(1, 8, D),)
        pb = (CutePlacement.shard(0, 8, D),)
        out = propagate_einsum("mk,kn->mn", pa, pb, (16, 8), (8, 32), (D,))
        self.assertIsNotNone(out)
        self.assertTrue(hasattr(out[0], "_is_partial") and out[0]._is_partial)

    def test_k_shard_only_a_incompatible(self):
        D = 2
        pa = (CutePlacement.shard(1, 8, D),)
        pb = (CutePlacement.replicate(D),)
        out = propagate_einsum("mk,kn->mn", pa, pb, (16, 8), (8, 32), (D,))
        self.assertIsNone(out)

    def test_conflicting_incompatible(self):
        D = 2
        pa = (CutePlacement.shard(0, 16, D),)
        pb = (CutePlacement.shard(1, 32, D),)
        out = propagate_einsum("mk,kn->mn", pa, pb, (16, 8), (8, 32), (D,))
        self.assertIsNone(out)

    def test_batch_both_sharded(self):
        D = 2
        pa = (CutePlacement.shard(0, 4, D),)
        pb = (CutePlacement.shard(0, 4, D),)
        out = propagate_einsum("bmk,bkn->bmn", pa, pb, (4, 8, 16), (4, 16, 32), (D,))
        self.assertIsNotNone(out)
        self.assertEqual(out[0].dim, 0)

    def test_batch_a_only_incompatible(self):
        D = 2
        pa = (CutePlacement.shard(0, 4, D),)
        pb = (CutePlacement.replicate(D),)
        out = propagate_einsum("bmk,bkn->bmn", pa, pb, (4, 8, 16), (4, 16, 32), (D,))
        self.assertIsNone(out)

    def test_batch_m_shard(self):
        D = 2
        pa = (CutePlacement.shard(1, 8, D),)
        pb = (CutePlacement.replicate(D),)
        out = propagate_einsum("bmk,bkn->bmn", pa, pb, (4, 8, 16), (4, 16, 32), (D,))
        self.assertIsNotNone(out)
        self.assertEqual(out[0].dim, 1)

    def test_cute_batch_shard(self):
        """CuTe hierarchical shard on batch passes through."""
        D = 4
        cute_layout = Layout((D, 2, 2), (4, 4, 1))
        pa = (CutePlacement(dim=0, layout=cute_layout),)
        pb = (CutePlacement(dim=0, layout=cute_layout),)
        out = propagate_einsum("bmk,bkn->bmn", pa, pb, (4, 8, 16), (4, 16, 32), (D,))
        self.assertIsNotNone(out)
        self.assertEqual(out[0].dim, 0)
        self.assertEqual(out[0].layout, cute_layout)


class TestPointwisePropagation(unittest.TestCase):

    def test_matching_placements(self):
        D = 4
        p = CutePlacement.shard(0, 8, D)
        out = propagate_pointwise([(p,), (p,)], [(8, 16), (8, 16)], (D,))
        self.assertIsNotNone(out)
        self.assertEqual(out[0], p)

    def test_disagreeing_incompatible(self):
        D = 4
        p1 = CutePlacement.shard(0, 8, D)
        p2 = CutePlacement.shard(1, 16, D)
        out = propagate_pointwise([(p1,), (p2,)], [(8, 16), (8, 16)], (D,))
        self.assertIsNone(out)

    def test_shard_with_replicate_incompatible(self):
        D = 2
        p1 = CutePlacement.shard(0, 8, D)
        p2 = CutePlacement.replicate(D)
        out = propagate_pointwise([(p1,), (p2,)], [(8, 16), (8, 16)], (D,))
        self.assertIsNone(out)

    def test_shard_with_replicate_broadcast(self):
        D = 2
        p1 = CutePlacement.shard(0, 8, D)
        p2 = CutePlacement.replicate(D)
        out = propagate_pointwise([(p1,), (p2,)], [(8, 16), (1, 16)], (D,))
        self.assertIsNotNone(out)
        self.assertEqual(out[0], p1)

    def test_all_replicate(self):
        D = 4
        p = CutePlacement.replicate(D)
        out = propagate_pointwise([(p,), (p,)], [(8, 16), (8, 16)], (D,))
        self.assertIsNotNone(out)
        self.assertTrue(out[0].is_replicate())

    def test_cute_passes_through(self):
        D = 4
        cute = CutePlacement(dim=0, layout=Layout((D, 2, 2), (4, 8, 1)))
        out = propagate_pointwise([(cute,), (cute,)], [(16,), (16,)], (D,))
        self.assertIsNotNone(out)
        self.assertEqual(out[0], cute)


class TestReductionPropagation(unittest.TestCase):

    def test_reduce_non_sharded_dim(self):
        D = 2
        p = CutePlacement.shard(1, 16, D)
        out = propagate_reduction((p,), 0, False, (D,))
        self.assertEqual(out[0].dim, 0)
        self.assertTrue(out[0].is_shard())

    def test_reduce_sharded_dim(self):
        D = 2
        p = CutePlacement.shard(0, 8, D)
        out = propagate_reduction((p,), 0, False, (D,))
        self.assertTrue(hasattr(out[0], "_is_partial") and out[0]._is_partial)

    def test_reduce_keepdim(self):
        D = 2
        p = CutePlacement.shard(1, 16, D)
        out = propagate_reduction((p,), 0, True, (D,))
        self.assertEqual(out[0].dim, 1)

    def test_reduce_replicate(self):
        D = 4
        p = CutePlacement.replicate(D)
        out = propagate_reduction((p,), 0, False, (D,))
        self.assertTrue(out[0].is_replicate())

    def test_reduce_cute_non_sharded_dim(self):
        D = 4
        cute = CutePlacement(dim=1, layout=Layout((D, 2, 2), (4, 8, 1)))
        out = propagate_reduction((cute,), 0, False, (D,))
        self.assertEqual(out[0].dim, 0)
        self.assertEqual(out[0].layout, cute.layout)


class TestEndToEnd(unittest.TestCase):

    def test_linear_3d_preserves_both_shardings(self):
        """
        (B, S, H) sharded [S(0), S(1)] -> view -> mm -> view
        Both shardings preserved, full round-trip recovery.
        """
        B, S, H, O = 4, 8, 16, 32
        dp_size, sp_size = 2, 4
        mesh_sizes = (dp_size, sp_size)

        input_p = (
            CutePlacement.shard(0, B, dp_size),
            CutePlacement.shard(1, S, sp_size),
        )

        # Step 1: view (B, S, H) -> (B*S, H)
        after_v1 = propagate_view(input_p, (B, S, H), (B * S, H), mesh_sizes)
        self.assertIsNotNone(after_v1)
        self.assertTrue(after_v1[0].is_shard())       # dp: S(0)
        self.assertFalse(after_v1[1].is_shard())      # sp: CuTe
        self.assertFalse(after_v1[1].is_replicate())

        # Step 2: mm (B*S, H) @ (H, O) -> (B*S, O)
        weight_p = (CutePlacement.replicate(dp_size), CutePlacement.replicate(sp_size))
        after_mm = propagate_einsum(
            "mk,kn->mn", after_v1, weight_p, (B * S, H), (H, O), mesh_sizes
        )
        self.assertIsNotNone(after_mm)

        # Step 3: view (B*S, O) -> (B, S, O)
        after_v2 = propagate_view(after_mm, (B * S, O), (B, S, O), mesh_sizes)
        self.assertIsNotNone(after_v2)

        # Full round-trip: S(0), S(1)
        self.assertEqual(after_v2[0].dim, 0)
        self.assertTrue(after_v2[0].is_shard())
        self.assertEqual(after_v2[1].dim, 1)
        self.assertTrue(after_v2[1].is_shard())


if __name__ == "__main__":
    unittest.main()
