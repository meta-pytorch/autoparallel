"""
Tests for CuTe-based sharding placement and propagation.
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
    """Sanity checks for the vendored pycute utilities."""

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
        # Can't merge: 2*1 != 4
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
        # Layout: device d -> offset d*4
        self.assertEqual(p.layout(0), 0)
        self.assertEqual(p.layout(1), 4)
        self.assertEqual(p.layout(3), 12)

    def test_shard_to_placement(self):
        p = CutePlacement.shard(dim=0, tensor_dim_size=8, mesh_dim_size=2)
        tp = p.to_placement()
        self.assertIsNotNone(tp)
        # Should be Shard(0)
        self.assertTrue(tp.is_shard())
        self.assertEqual(tp.dim, 0)

    def test_replicate_to_placement(self):
        p = CutePlacement.replicate(4)
        tp = p.to_placement()
        self.assertIsNotNone(tp)
        self.assertTrue(tp.is_replicate())

    def test_complex_placement_to_placement(self):
        # A hierarchical layout that can't be simplified to Shard
        p = CutePlacement(dim=0, layout=Layout((2, 2), (8, 1)))
        self.assertFalse(p.is_replicate())
        self.assertFalse(p.is_shard())
        tp = p.to_placement()
        self.assertIsNone(tp)  # Can't convert to standard placement

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
    """Test view/reshape propagation rules."""

    def test_identity_view(self):
        """View that doesn't change shape."""
        placements = (CutePlacement.shard(0, 8, 2),)
        inp_t, out_p = propagate_view(placements, (8, 4), (8, 4), (2,))
        self.assertEqual(out_p[0].dim, 0)
        self.assertTrue(out_p[0].is_shard())

    def test_replicate_through_view(self):
        """Replicate passes through any view."""
        placements = (CutePlacement.replicate(4),)
        inp_t, out_p = propagate_view(placements, (8, 4), (32,), (4,))
        self.assertTrue(out_p[0].is_replicate())

    def test_flatten_leftmost_shard(self):
        """Flatten (B, S, H) -> (B*S, H) with Shard(0) on batch."""
        # Shard on dim 0 (B=4), mesh_size=2
        placements = (CutePlacement.shard(0, 4, 2),)
        inp_t, out_p = propagate_view(placements, (4, 8, 16), (32, 16), (2,))
        # Shard on dim 0 should propagate: each device gets B/2 * S = 16 elements
        self.assertEqual(out_p[0].dim, 0)
        self.assertTrue(out_p[0].is_shard())
        # Each device holds 16 contiguous elements
        self.assertEqual(out_p[0].layout.size(), 16)

    def test_flatten_non_leftmost_shard(self):
        """
        Flatten (B, S, H) -> (B*S, H) with Shard(1) on sequence.
        This is the KEY test case that DTensor cannot handle.
        """
        B, S, H = 2, 8, 16
        D = 4  # mesh_dim_size

        placements = (CutePlacement.shard(1, S, D),)
        inp_t, out_p = propagate_view(placements, (B, S, H), (B * S, H), (D,))

        # Output should be a CutePlacement on dim 0 (the flat dim)
        self.assertEqual(out_p[0].dim, 0)
        # It should NOT be a simple shard (it's a strided pattern)
        self.assertFalse(out_p[0].is_shard())
        # It should NOT be replicate
        self.assertFalse(out_p[0].is_replicate())

        # Verify the layout is correct by checking what indices each device holds.
        # Flatten (B=2, S=8) -> (16,) with Shard on S by D=4.
        # Row-major flatten: flat_idx = b*S + s
        # Device d holds s in [d*2, d*2+2) for each batch b.
        # Device 0 holds flat indices: {0, 1, 8, 9}
        #
        # Layout shape = (pre_B, local_S) = (2, 2), stride = (S=8, 1)
        # CuTe colexicographic evaluation:
        #   L(0) = 0, L(1) = 8, L(2) = 1, L(3) = 9
        layout = out_p[0].layout
        device0_indices = sorted(layout(i) for i in range(layout.size()))
        self.assertEqual(device0_indices, [0, 1, 8, 9])

    def test_flatten_shard_2d_mesh(self):
        """
        Flatten (B, S, H) -> (B*S, H) with Shard(0) on dim0, Shard(1) on dim1.
        2D mesh: (dp=2, sp=4).
        """
        B, S, H = 4, 8, 16
        placements = (
            CutePlacement.shard(0, B, 2),  # Shard batch on mesh dim 0
            CutePlacement.shard(1, S, 4),  # Shard seq on mesh dim 1
        )
        mesh_sizes = (2, 4)
        inp_t, out_p = propagate_view(
            placements, (B, S, H), (B * S, H), mesh_sizes
        )

        # Mesh dim 0: Shard(0) -> leftmost in flatten -> contiguous shard
        self.assertEqual(out_p[0].dim, 0)
        self.assertTrue(out_p[0].is_shard())

        # Mesh dim 1: Shard(1) -> non-leftmost in flatten -> CuTe strided shard
        self.assertEqual(out_p[1].dim, 0)
        self.assertFalse(out_p[1].is_shard())
        self.assertFalse(out_p[1].is_replicate())

    def test_split_shard_first_piece(self):
        """Split (B*S, H) -> (B, S, H) with Shard(0), B evenly divides mesh."""
        BS, H = 32, 16
        B, S = 4, 8
        D = 2

        placements = (CutePlacement.shard(0, BS, D),)
        inp_t, out_p = propagate_view(placements, (BS, H), (B, S, H), (D,))

        # Shard on flat dim 0 -> should shard on first split dim (B=4, D=2)
        self.assertEqual(out_p[0].dim, 0)
        self.assertTrue(out_p[0].is_shard())

    def test_non_divisible_falls_back_to_replicate(self):
        """Shard that's not evenly divisible -> replicate in flatten output."""
        # Use a CutePlacement directly with a layout that has non-divisible shard
        # Shard dim 0 (size 6) by 4 devices: 6 % 4 != 0
        placements = (CutePlacement.shard(0, 6, 3),)
        # Flatten (6, 4) -> (24,) with dim 0 (size 6) sharded by 3
        inp_t, out_p = propagate_view(placements, (6, 4), (24,), (3,))
        # 6 is divisible by 3, so this should work
        self.assertEqual(out_p[0].dim, 0)
        self.assertFalse(out_p[0].is_replicate())

        # Now test actual non-divisible: shard dim 1 (size 5) by 3
        p2 = CutePlacement(dim=1, layout=Layout(3, 1))  # stride 1 but 5%3 != 0
        inp_t2, out_p2 = propagate_view((p2,), (4, 5), (20,), (3,))
        # 5 is not divisible by 3 -> replicate
        self.assertTrue(out_p2[0].is_replicate())

    def test_round_trip_flatten_unflatten(self):
        """
        Flatten then unflatten should recover the original shard on the
        leftmost dim. For non-leftmost dims, we get a CutePlacement
        after flatten, and after unflatten we should recover Shard.
        """
        B, S, H = 4, 8, 16
        D = 2

        # Start with Shard(0) on batch
        original = (CutePlacement.shard(0, B, D),)

        # Flatten (B, S, H) -> (B*S, H)
        _, after_flatten = propagate_view(
            original, (B, S, H), (B * S, H), (D,)
        )
        self.assertTrue(after_flatten[0].is_shard())
        self.assertEqual(after_flatten[0].dim, 0)

        # Unflatten (B*S, H) -> (B, S, H)
        _, after_unflatten = propagate_view(
            after_flatten, (B * S, H), (B, S, H), (D,)
        )
        # Should recover Shard(0) on batch dim
        self.assertEqual(after_unflatten[0].dim, 0)
        self.assertTrue(after_unflatten[0].is_shard())


class TestEinsumPropagation(unittest.TestCase):
    """Test einsum/matmul propagation rules."""

    def test_mm_batch_shard(self):
        """bmk,bkn->bmn with Shard on batch dim."""
        D = 2
        pa = (CutePlacement.shard(0, 4, D),)
        pb = (CutePlacement.replicate(D),)
        ta, tb, out = propagate_einsum(
            "bmk,bkn->bmn", pa, pb, (4, 8, 16), (4, 16, 32), (D,)
        )
        # Batch dim sharded: A stays, B gets shard on batch, output gets shard on batch
        self.assertEqual(out[0].dim, 0)  # batch dim in output
        self.assertFalse(out[0].is_replicate())

    def test_mm_m_shard(self):
        """bmk,bkn->bmn with Shard on M dim."""
        D = 2
        pa = (CutePlacement.shard(1, 8, D),)  # shard on m
        pb = (CutePlacement.replicate(D),)
        ta, tb, out = propagate_einsum(
            "bmk,bkn->bmn", pa, pb, (4, 8, 16), (4, 16, 32), (D,)
        )
        # M dim: A sharded, B replicate, output sharded on M
        self.assertEqual(out[0].dim, 1)  # m dim in output
        self.assertTrue(tb[0].is_replicate())

    def test_mm_n_shard(self):
        """bmk,bkn->bmn with Shard on N dim."""
        D = 2
        pa = (CutePlacement.replicate(D),)
        pb = (CutePlacement.shard(2, 32, D),)  # shard on n
        ta, tb, out = propagate_einsum(
            "bmk,bkn->bmn", pa, pb, (4, 8, 16), (4, 16, 32), (D,)
        )
        # N dim: B sharded, A replicate, output sharded on N
        self.assertEqual(out[0].dim, 2)  # n dim in output
        self.assertTrue(ta[0].is_replicate())

    def test_mm_k_shard(self):
        """bmk,bkn->bmn with Shard on K (contraction) dim."""
        D = 2
        pa = (CutePlacement.shard(2, 16, D),)  # shard on k
        pb = (CutePlacement.replicate(D),)
        ta, tb, out = propagate_einsum(
            "bmk,bkn->bmn", pa, pb, (4, 8, 16), (4, 16, 32), (D,)
        )
        # K dim: both inputs sharded, output is partial
        self.assertEqual(tb[0].dim, 1)  # k in B is dim 1
        self.assertTrue(hasattr(out[0], "_is_partial") and out[0]._is_partial)

    def test_mm_cute_batch_shard(self):
        """bmk,bkn->bmn with CuTe (hierarchical) shard on batch dim."""
        D = 2
        # A CutePlacement with hierarchical layout (from a previous view)
        cute_layout = Layout((2, 2), (4, 1))  # non-trivial layout
        pa = (CutePlacement(dim=0, layout=cute_layout),)
        pb = (CutePlacement.replicate(D),)
        ta, tb, out = propagate_einsum(
            "bmk,bkn->bmn", pa, pb, (4, 8, 16), (4, 16, 32), (D,)
        )
        # Batch dim: CuTe layout should pass through to output
        self.assertEqual(out[0].dim, 0)
        self.assertEqual(out[0].layout, cute_layout)

    def test_simple_mm(self):
        """mk,kn->mn (no batch dim)."""
        D = 4
        pa = (CutePlacement.shard(0, 16, D),)  # shard on m
        pb = (CutePlacement.replicate(D),)
        ta, tb, out = propagate_einsum(
            "mk,kn->mn", pa, pb, (16, 8), (8, 32), (D,)
        )
        self.assertEqual(out[0].dim, 0)  # m in output


class TestPointwisePropagation(unittest.TestCase):
    """Test pointwise op propagation."""

    def test_matching_placements(self):
        """All inputs agree -> output gets same placement."""
        D = 4
        p = CutePlacement.shard(0, 8, D)
        out = propagate_pointwise(
            [(p,), (p,)], [(8, 16), (8, 16)], (D,)
        )
        self.assertEqual(out[0], p)

    def test_disagreeing_placements(self):
        """Inputs disagree -> output is replicate."""
        D = 4
        p1 = CutePlacement.shard(0, 8, D)
        p2 = CutePlacement.shard(1, 16, D)
        out = propagate_pointwise(
            [(p1,), (p2,)], [(8, 16), (8, 16)], (D,)
        )
        self.assertTrue(out[0].is_replicate())

    def test_broadcast_size1(self):
        """Input with size-1 on sharded dim is treated as replicate."""
        D = 2
        p1 = CutePlacement.shard(0, 8, D)
        p2 = CutePlacement.replicate(D)  # size-1 dim -> replicate
        out = propagate_pointwise(
            [(p1,), (p2,)], [(8, 16), (1, 16)], (D,)
        )
        # p2 is replicate, so only p1's placement matters
        self.assertEqual(out[0], p1)

    def test_all_replicate(self):
        """All replicate -> output replicate."""
        D = 4
        p = CutePlacement.replicate(D)
        out = propagate_pointwise(
            [(p,), (p,)], [(8, 16), (8, 16)], (D,)
        )
        self.assertTrue(out[0].is_replicate())

    def test_cute_placement_passes_through(self):
        """CuTe (hierarchical) placement passes through pointwise."""
        D = 4
        cute_layout = Layout((2, 2), (8, 1))
        p = CutePlacement(dim=0, layout=cute_layout)
        out = propagate_pointwise(
            [(p,), (p,)], [(16,), (16,)], (D,)
        )
        self.assertEqual(out[0], p)


class TestReductionPropagation(unittest.TestCase):
    """Test reduction op propagation."""

    def test_reduce_non_sharded_dim(self):
        """Reducing a non-sharded dim: shard placement adjusts dim."""
        D = 2
        p = CutePlacement.shard(1, 16, D)
        out = propagate_reduction((p,), reduce_dim=0, keepdim=False, mesh_sizes=(D,))
        # Dim 1 becomes dim 0 when dim 0 is removed
        self.assertEqual(out[0].dim, 0)
        self.assertTrue(out[0].is_shard())

    def test_reduce_sharded_dim(self):
        """Reducing the sharded dim: output is partial."""
        D = 2
        p = CutePlacement.shard(0, 8, D)
        out = propagate_reduction((p,), reduce_dim=0, keepdim=False, mesh_sizes=(D,))
        self.assertTrue(hasattr(out[0], "_is_partial") and out[0]._is_partial)

    def test_reduce_keepdim(self):
        """Reducing with keepdim=True: dim indices unchanged."""
        D = 2
        p = CutePlacement.shard(1, 16, D)
        out = propagate_reduction((p,), reduce_dim=0, keepdim=True, mesh_sizes=(D,))
        # Dim 1 stays dim 1 since keepdim=True
        self.assertEqual(out[0].dim, 1)

    def test_reduce_replicate(self):
        """Reducing with replicate placement: stays replicate."""
        D = 4
        p = CutePlacement.replicate(D)
        out = propagate_reduction((p,), reduce_dim=0, keepdim=False, mesh_sizes=(D,))
        self.assertTrue(out[0].is_replicate())

    def test_reduce_cute_non_sharded_dim(self):
        """Reducing non-sharded dim with CuTe placement."""
        D = 4
        cute_layout = Layout((2, 2), (8, 1))
        p = CutePlacement(dim=1, layout=cute_layout)
        out = propagate_reduction((p,), reduce_dim=0, keepdim=False, mesh_sizes=(D,))
        self.assertEqual(out[0].dim, 0)
        self.assertEqual(out[0].layout, cute_layout)


class TestEndToEnd(unittest.TestCase):
    """End-to-end test: nn.Linear decomposition view -> mm -> view."""

    def test_linear_3d_preserves_both_shardings(self):
        """
        The key motivating example:
        input (B, S, H) sharded on both B and S across a 2D mesh.
        nn.Linear decomposes to: view(B*S, H) -> mm(B*S, O) -> view(B, S, O)
        With CuTe, both shardings should be preserved through the entire chain.
        """
        B, S, H, O = 4, 8, 16, 32
        dp_size = 2  # mesh dim 0: shard batch
        sp_size = 4  # mesh dim 1: shard sequence
        mesh_sizes = (dp_size, sp_size)

        # Initial placements: Shard(0) on dp, Shard(1) on sp
        input_placements = (
            CutePlacement.shard(0, B, dp_size),
            CutePlacement.shard(1, S, sp_size),
        )

        # Step 1: view (B, S, H) -> (B*S, H)
        _, after_view1 = propagate_view(
            input_placements, (B, S, H), (B * S, H), mesh_sizes
        )

        # Mesh dim 0 (dp): Shard(0) on B -> leftmost in flatten -> contiguous shard
        self.assertEqual(after_view1[0].dim, 0)
        self.assertTrue(after_view1[0].is_shard())

        # Mesh dim 1 (sp): Shard(1) on S -> non-leftmost -> CuTe hierarchical shard
        self.assertEqual(after_view1[1].dim, 0)
        self.assertFalse(after_view1[1].is_shard())
        self.assertFalse(after_view1[1].is_replicate())

        # Step 2: mm (B*S, H) @ (H, O) -> (B*S, O)
        # Weight is replicated
        weight_placements = (
            CutePlacement.replicate(dp_size),
            CutePlacement.replicate(sp_size),
        )
        ta, tw, after_mm = propagate_einsum(
            "mk,kn->mn",
            after_view1,
            weight_placements,
            (B * S, H),
            (H, O),
            mesh_sizes,
        )

        # dp (mesh dim 0): M-shard passes through mm
        self.assertEqual(after_mm[0].dim, 0)
        self.assertTrue(after_mm[0].is_shard())

        # sp (mesh dim 1): CuTe M-shard passes through mm
        self.assertEqual(after_mm[1].dim, 0)
        self.assertFalse(after_mm[1].is_shard())
        self.assertFalse(after_mm[1].is_replicate())

        # Step 3: view (B*S, O) -> (B, S, O)
        _, after_view2 = propagate_view(
            after_mm, (B * S, O), (B, S, O), mesh_sizes
        )

        # dp (mesh dim 0): should recover Shard(0) on B
        self.assertEqual(after_view2[0].dim, 0)
        self.assertTrue(after_view2[0].is_shard())

        # sp (mesh dim 1): the CuTe shard on flat dim 0 should split back
        # This will depend on whether the split propagation can recover
        # the original shard from the CuTe layout.
        # At minimum, it should not be replicate (the whole point!).
        self.assertFalse(after_view2[1].is_replicate())


if __name__ == "__main__":
    unittest.main()
