"""
Tests for CuTe-based sharding placement and propagation.

All propagation rules assume inputs are already in the given placements
(no redistribution). They return output placements or None if incompatible.
"""

import unittest

from torch.distributed._pycute import Layout, coalesce, flatten, product
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
        self.assertEqual(L(3), 6)

    def test_layout_2d(self):
        L = Layout((4, 8), (1, 4))  # col-major 4x8
        self.assertEqual(L.size(), 32)
        self.assertEqual(L(0, 0), 0)
        self.assertEqual(L(1, 0), 1)
        self.assertEqual(L(0, 1), 4)

    def test_coalesce(self):
        L = Layout((2, 6), (6, 1))  # row-major contiguous 12 elements
        c = coalesce(L)
        self.assertEqual(c.shape, 12)
        self.assertEqual(c.stride, 1)

    def test_coalesce_non_contiguous(self):
        L = Layout((2, 3), (1, 4))
        c = coalesce(L)
        self.assertEqual(c.shape, (2, 3))
        self.assertEqual(c.stride, (1, 4))

    def test_stride_zero(self):
        L = Layout(4, 0)
        self.assertEqual(L(0), 0)
        self.assertEqual(L(1), 0)
        self.assertEqual(L(3), 0)


class TestCutePlacement(unittest.TestCase):
    """Test CutePlacement creation and conversion."""

    def test_replicate(self):
        p = CutePlacement.replicate(4)
        self.assertTrue(p.is_replicate())
        self.assertFalse(p.is_shard())
        self.assertIsNone(p.dim)
        self.assertEqual(str(p), "R")

    def test_shard(self):
        p = CutePlacement.shard(dim=1, tensor_dim_size=16, mesh_dim_size=4)
        self.assertFalse(p.is_replicate())
        self.assertTrue(p.is_shard())
        self.assertEqual(p.dim, 1)
        self.assertEqual(p.layout(0), 0)
        self.assertEqual(p.layout(1), 4)
        self.assertEqual(p.layout(3), 12)

    def test_shard_to_placement(self):
        p = CutePlacement.shard(dim=0, tensor_dim_size=8, mesh_dim_size=2)
        tp = p.to_placement()
        self.assertIsNotNone(tp)
        self.assertTrue(tp.is_shard())
        self.assertEqual(tp.dim, 0)

    def test_replicate_to_placement(self):
        p = CutePlacement.replicate(4)
        tp = p.to_placement()
        self.assertIsNotNone(tp)
        self.assertTrue(tp.is_replicate())

    def test_complex_placement_to_placement(self):
        p = CutePlacement(dim=0, layout=Layout((2, 2), (8, 1)))
        self.assertFalse(p.is_replicate())
        self.assertFalse(p.is_shard())
        tp = p.to_placement()
        self.assertIsNone(tp)

    def test_equality(self):
        p1 = CutePlacement.shard(1, 16, 4)
        p2 = CutePlacement.shard(1, 16, 4)
        self.assertEqual(p1, p2)

    def test_replicate_equality(self):
        p1 = CutePlacement.replicate(4)
        p2 = CutePlacement.replicate(4)
        self.assertEqual(p1, p2)

    def test_from_placement_shard(self):
        from torch.distributed.tensor.placement_types import Shard
        s = Shard(1)
        p = CutePlacement.from_placement(s, tensor_dim_size=16, mesh_dim_size=4)
        self.assertTrue(p.is_shard())
        self.assertEqual(p.dim, 1)

    def test_from_placement_replicate(self):
        from torch.distributed.tensor.placement_types import Replicate
        r = Replicate()
        p = CutePlacement.from_placement(r, tensor_dim_size=16, mesh_dim_size=4)
        self.assertTrue(p.is_replicate())


class TestViewPropagation(unittest.TestCase):
    """Test view/reshape propagation — all redistribution-free."""

    def test_identity_view(self):
        """View that doesn't change shape: placement passes through."""
        placements = (CutePlacement.shard(0, 8, 2),)
        out = propagate_view(placements, (8, 4), (8, 4), (2,))
        self.assertIsNotNone(out)
        self.assertEqual(out[0].dim, 0)
        self.assertTrue(out[0].is_shard())

    def test_replicate_through_view(self):
        """Replicate passes through any view — no communication."""
        placements = (CutePlacement.replicate(4),)
        out = propagate_view(placements, (8, 4), (32,), (4,))
        self.assertIsNotNone(out)
        self.assertTrue(out[0].is_replicate())

    def test_flatten_leftmost_shard(self):
        """Flatten (B, S, H) -> (B*S, H) with Shard(0) on batch.

        Each device does local view (B/D, S, H) -> (B/D*S, H).
        Output is contiguous shard on dim 0. No communication.
        """
        placements = (CutePlacement.shard(0, 4, 2),)
        out = propagate_view(placements, (4, 8, 16), (32, 16), (2,))
        self.assertIsNotNone(out)
        self.assertEqual(out[0].dim, 0)
        self.assertTrue(out[0].is_shard())
        self.assertEqual(out[0].layout.size(), 16)  # B/D * S = 2*8 = 16

    def test_flatten_non_leftmost_shard(self):
        """Flatten (B, S, H) -> (B*S, H) with Shard(1) on sequence.

        KEY TEST: DTensor can't handle this, CuTe can.
        Each device does local view (B, S/D, H) -> (B*S/D, H).
        In global flat space, the device's elements are strided.
        No communication — just a different description of positions.
        """
        B, S, H = 2, 8, 16
        D = 4

        placements = (CutePlacement.shard(1, S, D),)
        out = propagate_view(placements, (B, S, H), (B * S, H), (D,))

        self.assertIsNotNone(out)
        self.assertEqual(out[0].dim, 0)
        self.assertFalse(out[0].is_shard())  # strided, not contiguous
        self.assertFalse(out[0].is_replicate())

        # Verify: device 0 holds S/D=2 elements from each of B=2 batches
        # Global flat indices: {b*S + s | b in [0,B), s in [0, S/D)}
        # = {0*8+0, 0*8+1, 1*8+0, 1*8+1} = {0, 1, 8, 9}
        layout = out[0].layout
        device0_indices = sorted(layout(i) for i in range(layout.size()))
        self.assertEqual(device0_indices, [0, 1, 8, 9])

    def test_flatten_shard_2d_mesh(self):
        """Flatten (B, S, H) -> (B*S, H) with 2D mesh [Shard(0), Shard(1)].

        Each device does a local view — no communication on either mesh dim.
        """
        B, S, H = 4, 8, 16
        placements = (
            CutePlacement.shard(0, B, 2),
            CutePlacement.shard(1, S, 4),
        )
        out = propagate_view(placements, (B, S, H), (B * S, H), (2, 4))

        self.assertIsNotNone(out)
        # Mesh dim 0: leftmost flatten -> contiguous shard
        self.assertEqual(out[0].dim, 0)
        self.assertTrue(out[0].is_shard())
        # Mesh dim 1: non-leftmost flatten -> strided CuTe shard
        self.assertEqual(out[1].dim, 0)
        self.assertFalse(out[1].is_shard())
        self.assertFalse(out[1].is_replicate())

    def test_split_shard_first_piece(self):
        """Split (B*S, H) -> (B, S, H) with Shard(0), B divisible by mesh.

        Each device does local view (B*S/D, H) -> (B/D, S, H).
        No communication.
        """
        BS, H = 32, 16
        B, S = 4, 8
        D = 2

        placements = (CutePlacement.shard(0, BS, D),)
        out = propagate_view(placements, (BS, H), (B, S, H), (D,))

        self.assertIsNotNone(out)
        self.assertEqual(out[0].dim, 0)
        self.assertTrue(out[0].is_shard())

    def test_non_divisible_returns_none(self):
        """Shard not evenly divisible -> incompatible (returns None)."""
        # Shard dim 1 (size 5) by 3 — can't evenly divide
        p = CutePlacement(dim=1, layout=Layout(3, 1))
        out = propagate_view((p,), (4, 5), (20,), (3,))
        self.assertIsNone(out)

    def test_divisible_flatten(self):
        """Shard dim 0 (size 6) by 3 in flatten — works."""
        placements = (CutePlacement.shard(0, 6, 3),)
        out = propagate_view(placements, (6, 4), (24,), (3,))
        self.assertIsNotNone(out)
        self.assertEqual(out[0].dim, 0)

    def test_round_trip_flatten_unflatten(self):
        """Flatten then unflatten recovers original Shard(0).

        (B, S, H) -> (B*S, H) -> (B, S, H)
        Both views are local — no communication at any step.
        """
        B, S, H = 4, 8, 16
        D = 2

        original = (CutePlacement.shard(0, B, D),)

        after_flatten = propagate_view(original, (B, S, H), (B * S, H), (D,))
        self.assertIsNotNone(after_flatten)
        self.assertTrue(after_flatten[0].is_shard())
        self.assertEqual(after_flatten[0].dim, 0)

        after_unflatten = propagate_view(
            after_flatten, (B * S, H), (B, S, H), (D,)
        )
        self.assertIsNotNone(after_unflatten)
        self.assertEqual(after_unflatten[0].dim, 0)
        self.assertTrue(after_unflatten[0].is_shard())


class TestEinsumPropagation(unittest.TestCase):
    """Test einsum/matmul propagation — all redistribution-free."""

    def test_mm_a_batch_shard_b_replicate(self):
        """bmk,bkn->bmn: A=Shard(batch), B=Replicate.

        Each device has its batch slice of A and full B.
        Computes C[d] = A[d] @ B[d] locally. No communication.
        """
        D = 2
        pa = (CutePlacement.shard(0, 4, D),)
        pb = (CutePlacement.replicate(D),)
        out = propagate_einsum(
            "bmk,bkn->bmn", pa, pb, (4, 8, 16), (4, 16, 32), (D,)
        )
        self.assertIsNotNone(out)
        self.assertEqual(out[0].dim, 0)
        self.assertFalse(out[0].is_replicate())

    def test_mm_a_replicate_b_batch_shard(self):
        """bmk,bkn->bmn: A=Replicate, B=Shard(batch).

        Symmetric to above. No communication.
        """
        D = 2
        pa = (CutePlacement.replicate(D),)
        pb = (CutePlacement.shard(0, 4, D),)
        out = propagate_einsum(
            "bmk,bkn->bmn", pa, pb, (4, 8, 16), (4, 16, 32), (D,)
        )
        self.assertIsNotNone(out)
        self.assertEqual(out[0].dim, 0)

    def test_mm_both_batch_shard(self):
        """bmk,bkn->bmn: Both sharded on batch.

        Each device has matching batch slices. No communication.
        """
        D = 2
        pa = (CutePlacement.shard(0, 4, D),)
        pb = (CutePlacement.shard(0, 4, D),)
        out = propagate_einsum(
            "bmk,bkn->bmn", pa, pb, (4, 8, 16), (4, 16, 32), (D,)
        )
        self.assertIsNotNone(out)
        self.assertEqual(out[0].dim, 0)

    def test_mm_m_shard(self):
        """bmk,bkn->bmn: A=Shard(m), B=Replicate.

        Each device has its M slice of A. Computes its M slice of output.
        B is fully available. No communication.
        """
        D = 2
        pa = (CutePlacement.shard(1, 8, D),)
        pb = (CutePlacement.replicate(D),)
        out = propagate_einsum(
            "bmk,bkn->bmn", pa, pb, (4, 8, 16), (4, 16, 32), (D,)
        )
        self.assertIsNotNone(out)
        self.assertEqual(out[0].dim, 1)  # m dim in output

    def test_mm_n_shard(self):
        """bmk,bkn->bmn: A=Replicate, B=Shard(n).

        Each device has its N slice of B. Computes its N slice of output.
        A is fully available. No communication.
        """
        D = 2
        pa = (CutePlacement.replicate(D),)
        pb = (CutePlacement.shard(2, 32, D),)
        out = propagate_einsum(
            "bmk,bkn->bmn", pa, pb, (4, 8, 16), (4, 16, 32), (D,)
        )
        self.assertIsNotNone(out)
        self.assertEqual(out[0].dim, 2)  # n dim in output

    def test_mm_k_shard_both(self):
        """bmk,bkn->bmn: Both sharded on K (contraction).

        Each device computes partial result. Needs all-reduce AFTER
        (inherent to algorithm, not redistribution).
        """
        D = 2
        pa = (CutePlacement.shard(2, 16, D),)
        pb = (CutePlacement.shard(1, 16, D),)
        out = propagate_einsum(
            "bmk,bkn->bmn", pa, pb, (4, 8, 16), (4, 16, 32), (D,)
        )
        self.assertIsNotNone(out)
        self.assertTrue(hasattr(out[0], "_is_partial") and out[0]._is_partial)

    def test_mm_k_shard_only_a_incompatible(self):
        """bmk,bkn->bmn: A=Shard(k), B=Replicate -> incompatible.

        Can't compute partial mm if only one input is sharded on K.
        """
        D = 2
        pa = (CutePlacement.shard(2, 16, D),)
        pb = (CutePlacement.replicate(D),)
        out = propagate_einsum(
            "bmk,bkn->bmn", pa, pb, (4, 8, 16), (4, 16, 32), (D,)
        )
        self.assertIsNone(out)

    def test_mm_conflicting_shardings_incompatible(self):
        """bmk,bkn->bmn: A=Shard(m), B=Shard(n) on same mesh dim -> incompatible."""
        D = 2
        pa = (CutePlacement.shard(1, 8, D),)  # m
        pb = (CutePlacement.shard(2, 32, D),)  # n
        out = propagate_einsum(
            "bmk,bkn->bmn", pa, pb, (4, 8, 16), (4, 16, 32), (D,)
        )
        self.assertIsNone(out)

    def test_mm_cute_batch_shard(self):
        """bmk,bkn->bmn: CuTe hierarchical shard on batch passes through."""
        D = 2
        cute_layout = Layout((2, 2), (4, 1))
        pa = (CutePlacement(dim=0, layout=cute_layout),)
        pb = (CutePlacement.replicate(D),)
        out = propagate_einsum(
            "bmk,bkn->bmn", pa, pb, (4, 8, 16), (4, 16, 32), (D,)
        )
        self.assertIsNotNone(out)
        self.assertEqual(out[0].dim, 0)
        self.assertEqual(out[0].layout, cute_layout)

    def test_simple_mm(self):
        """mk,kn->mn: A=Shard(m), B=Replicate."""
        D = 4
        pa = (CutePlacement.shard(0, 16, D),)
        pb = (CutePlacement.replicate(D),)
        out = propagate_einsum(
            "mk,kn->mn", pa, pb, (16, 8), (8, 32), (D,)
        )
        self.assertIsNotNone(out)
        self.assertEqual(out[0].dim, 0)

    def test_both_replicate(self):
        """Both replicate -> output replicate."""
        D = 2
        pa = (CutePlacement.replicate(D),)
        pb = (CutePlacement.replicate(D),)
        out = propagate_einsum(
            "mk,kn->mn", pa, pb, (16, 8), (8, 32), (D,)
        )
        self.assertIsNotNone(out)
        self.assertTrue(out[0].is_replicate())


class TestPointwisePropagation(unittest.TestCase):
    """Test pointwise op propagation — redistribution-free."""

    def test_matching_placements(self):
        """All inputs agree -> output gets same placement."""
        D = 4
        p = CutePlacement.shard(0, 8, D)
        out = propagate_pointwise([(p,), (p,)], [(8, 16), (8, 16)], (D,))
        self.assertIsNotNone(out)
        self.assertEqual(out[0], p)

    def test_disagreeing_placements_incompatible(self):
        """Inputs disagree on sharding -> incompatible (returns None)."""
        D = 4
        p1 = CutePlacement.shard(0, 8, D)
        p2 = CutePlacement.shard(1, 16, D)
        out = propagate_pointwise([(p1,), (p2,)], [(8, 16), (8, 16)], (D,))
        self.assertIsNone(out)

    def test_shard_with_replicate(self):
        """One sharded, one replicate -> output gets shard."""
        D = 2
        p1 = CutePlacement.shard(0, 8, D)
        p2 = CutePlacement.replicate(D)
        out = propagate_pointwise([(p1,), (p2,)], [(8, 16), (8, 16)], (D,))
        self.assertIsNotNone(out)
        self.assertEqual(out[0], p1)

    def test_broadcast_size1(self):
        """Input with size-1 on sharded dim -> treated as replicate."""
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

    def test_cute_placement_passes_through(self):
        D = 4
        cute_layout = Layout((2, 2), (8, 1))
        p = CutePlacement(dim=0, layout=cute_layout)
        out = propagate_pointwise([(p,), (p,)], [(16,), (16,)], (D,))
        self.assertIsNotNone(out)
        self.assertEqual(out[0], p)


class TestReductionPropagation(unittest.TestCase):
    """Test reduction op propagation."""

    def test_reduce_non_sharded_dim(self):
        D = 2
        p = CutePlacement.shard(1, 16, D)
        out = propagate_reduction((p,), reduce_dim=0, keepdim=False, mesh_sizes=(D,))
        self.assertEqual(out[0].dim, 0)
        self.assertTrue(out[0].is_shard())

    def test_reduce_sharded_dim(self):
        D = 2
        p = CutePlacement.shard(0, 8, D)
        out = propagate_reduction((p,), reduce_dim=0, keepdim=False, mesh_sizes=(D,))
        self.assertTrue(hasattr(out[0], "_is_partial") and out[0]._is_partial)

    def test_reduce_keepdim(self):
        D = 2
        p = CutePlacement.shard(1, 16, D)
        out = propagate_reduction((p,), reduce_dim=0, keepdim=True, mesh_sizes=(D,))
        self.assertEqual(out[0].dim, 1)

    def test_reduce_replicate(self):
        D = 4
        p = CutePlacement.replicate(D)
        out = propagate_reduction((p,), reduce_dim=0, keepdim=False, mesh_sizes=(D,))
        self.assertTrue(out[0].is_replicate())

    def test_reduce_cute_non_sharded_dim(self):
        D = 4
        cute_layout = Layout((2, 2), (8, 1))
        p = CutePlacement(dim=1, layout=cute_layout)
        out = propagate_reduction((p,), reduce_dim=0, keepdim=False, mesh_sizes=(D,))
        self.assertEqual(out[0].dim, 0)
        self.assertEqual(out[0].layout, cute_layout)


class TestEndToEnd(unittest.TestCase):
    """End-to-end: nn.Linear decomposition view -> mm -> view."""

    def test_linear_3d_preserves_both_shardings(self):
        """
        Input (B, S, H) sharded on B and S across 2D mesh.
        nn.Linear = view(B*S, H) -> mm(B*S, O) -> view(B, S, O)

        All three ops are redistribution-free with CuTe placements.
        """
        B, S, H, O = 4, 8, 16, 32
        dp_size, sp_size = 2, 4
        mesh_sizes = (dp_size, sp_size)

        input_placements = (
            CutePlacement.shard(0, B, dp_size),
            CutePlacement.shard(1, S, sp_size),
        )

        # Step 1: view (B, S, H) -> (B*S, H) — local view, no communication
        after_view1 = propagate_view(
            input_placements, (B, S, H), (B * S, H), mesh_sizes
        )
        self.assertIsNotNone(after_view1)

        # dp: Shard(0) -> contiguous shard on flat dim
        self.assertEqual(after_view1[0].dim, 0)
        self.assertTrue(after_view1[0].is_shard())

        # sp: Shard(1) -> CuTe strided shard on flat dim
        self.assertEqual(after_view1[1].dim, 0)
        self.assertFalse(after_view1[1].is_shard())
        self.assertFalse(after_view1[1].is_replicate())

        # Step 2: mm (B*S, H) @ (H, O) -> (B*S, O)
        # Weight replicated. A sharded on M dim. No communication.
        weight_placements = (
            CutePlacement.replicate(dp_size),
            CutePlacement.replicate(sp_size),
        )
        after_mm = propagate_einsum(
            "mk,kn->mn",
            after_view1,
            weight_placements,
            (B * S, H),
            (H, O),
            mesh_sizes,
        )
        self.assertIsNotNone(after_mm)

        # dp: M-shard passes through
        self.assertEqual(after_mm[0].dim, 0)
        self.assertTrue(after_mm[0].is_shard())

        # sp: CuTe M-shard passes through
        self.assertEqual(after_mm[1].dim, 0)
        self.assertFalse(after_mm[1].is_shard())
        self.assertFalse(after_mm[1].is_replicate())

        # Step 3: view (B*S, O) -> (B, S, O) — local view, no communication
        after_view2 = propagate_view(
            after_mm, (B * S, O), (B, S, O), mesh_sizes
        )
        self.assertIsNotNone(after_view2)

        # dp: should recover Shard(0) on B
        self.assertEqual(after_view2[0].dim, 0)
        self.assertTrue(after_view2[0].is_shard())

        # sp: should not be replicate (preserved through entire chain!)
        self.assertFalse(after_view2[1].is_replicate())


if __name__ == "__main__":
    unittest.main()
