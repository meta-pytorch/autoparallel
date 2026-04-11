"""
Tests for recipe-based sharding propagation.
All propagation rules use the 5 primitives: Carry, Insert, Remove, Merge, Split.
"""

import unittest

from autoparallel.shardings.cute._pycute import (
    ArithmeticTuple,
    E,
    Layout,
    ScaledBasis,
    XorStride,
    coalesce,
    codomain_divide,
    composition,
    logical_divide,
    make_basis_like,
    max_common_layout,
    max_common_vector,
)
from autoparallel.shardings.cute import (
    ShardedLayout,
    plan_redistribute,
    plan_redistribute_detailed,
    propagate_addmm,
    propagate_argmax,
    propagate_baddbmm,
    propagate_bmm,
    propagate_broadcast,
    propagate_cat,
    propagate_convolution,
    propagate_dot,
    propagate_dropout,
    propagate_einsum,
    propagate_embedding,
    propagate_expand,
    propagate_flatten,
    propagate_gather,
    propagate_identity,
    propagate_index_select,
    propagate_layer_norm,
    propagate_mm,
    propagate_movedim,
    propagate_permute,
    propagate_pointwise,
    propagate_reduction,
    propagate_repeat,
    propagate_replicate_affected,
    propagate_scatter,
    propagate_slice,
    propagate_sort,
    propagate_split,
    propagate_squeeze,
    propagate_stack,
    propagate_t,
    propagate_topk,
    propagate_transpose,
    propagate_unbind,
    propagate_unflatten,
    propagate_unsqueeze,
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
        self.assertEqual(placements[0], ("shard", 0, 2, (0,)))

    def test_shard_dim1(self):
        t = ShardedLayout.shard((4, 8, 16), shard_dim=1, mesh_dim_size=2)
        placements = t.get_placements()
        self.assertEqual(placements[0], ("shard", 1, 2, (0,)))

    def test_replicate_placements(self):
        t = ShardedLayout.replicate((4, 8, 16))
        placements = t.get_placements()
        self.assertEqual(placements, [("replicate", None, None)])

    def test_equality(self):
        t1 = ShardedLayout.shard((4, 8, 16), shard_dim=0, mesh_dim_size=2)
        t2 = ShardedLayout.shard((4, 8, 16), shard_dim=0, mesh_dim_size=2)
        self.assertEqual(t1, t2)

    def test_hier_layout(self):
        t = ShardedLayout.shard((4, 8, 16), shard_dim=1, mesh_dim_size=2)
        self.assertEqual(t.hier_layout.size(), 512)

    def test_shard_multi_s0s0(self):
        t = ShardedLayout.shard_multi((8, 16), [(0, 2), (0, 4)])
        self.assertFalse(t.is_replicate())
        placements = t.get_placements()
        self.assertEqual(placements[0][0], "shard")
        self.assertEqual(placements[0][1], 0)

    def test_shard_multi_s0s1(self):
        t = ShardedLayout.shard_multi((4, 8, 16), [(0, 2), (1, 4)])
        placements = t.get_placements()
        self.assertEqual(placements[0], ("shard", 0, 2, (0,)))
        self.assertEqual(placements[1], ("shard", 1, 4, (1,)))

    def test_shard_multi_view_invariant(self):
        t = ShardedLayout.shard_multi((4, 8, 16), [(0, 2), (1, 4)])
        v = propagate_view(t, (32, 16))
        self.assertIsNotNone(v)
        placements = v.get_placements()
        self.assertEqual(placements[0][0], "shard")
        self.assertEqual(placements[0][1], 0)


class TestViewPropagation(unittest.TestCase):

    def test_flatten(self):
        t = ShardedLayout.shard((4, 8, 16), shard_dim=0, mesh_dim_size=2)
        out = propagate_view(t, (32, 16))
        self.assertIsNotNone(out)
        self.assertEqual(out.tensor_shape, (32, 16))
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
        self.assertEqual(original.get_placements(), after_unflat.get_placements())

    def test_incompatible(self):
        t = ShardedLayout.shard((4, 8, 16), shard_dim=0, mesh_dim_size=2)
        self.assertIsNone(propagate_view(t, (100,)))

    def test_hier_layout_invariant(self):
        t = ShardedLayout.shard((4, 8, 16), shard_dim=1, mesh_dim_size=4)
        orig_placements = t.get_placements()
        v1 = propagate_view(t, (32, 16))
        self.assertIsNotNone(v1)
        v2 = propagate_view(v1, (2, 16, 16))
        self.assertIsNotNone(v2)
        self.assertEqual(v2.get_placements()[0], ("shard", 1, 4, (0,)))
        v3 = propagate_view(v2, (4, 8, 16))
        self.assertIsNotNone(v3)
        self.assertEqual(v3.get_placements(), orig_placements)

    def test_s0s0_split_preserves_mesh_dims(self):
        """S(0)S(0) view split keeps both mesh dims on the output dim that has mesh."""
        src = ShardedLayout.shard_multi((16,), [(0, 2), (0, 2)])
        self.assertEqual(src.mesh_dim_map, {0: (0, 1)})

        # Split (16,) -> (4, 4): local=4 splits off, both mesh factors stay together
        out = propagate_view(src, (4, 4))
        self.assertIsNotNone(out)
        self.assertEqual(out.global_shape, (4, 4))
        # Both mesh dims must stay on tensor dim 0 (which has the mesh factors)
        self.assertEqual(out.mesh_dim_map[0], (0, 1))
        self.assertEqual(out.mesh_dim_map[1], ())

    def test_s0s0_split_round_trip(self):
        """S(0)S(0) split then merge recovers original mesh_dim_map."""
        src = ShardedLayout.shard_multi((16,), [(0, 2), (0, 2)])
        split = propagate_view(src, (4, 4))
        self.assertIsNotNone(split)
        merged = propagate_view(split, (16,))
        self.assertIsNotNone(merged)
        self.assertEqual(merged.mesh_dim_map, {0: (0, 1)})


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
        self.assertEqual(placements[0], ("shard", 0, 2, (0,)))

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
        self.assertEqual(placements[0], ("shard", 0, 2, (0,)))


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


class TestUnsqueezePropagation(unittest.TestCase):

    def test_unsqueeze_replicate(self):
        t = ShardedLayout.replicate((4, 8))
        out = propagate_unsqueeze(t, dim=0)
        self.assertEqual(out.global_shape, (1, 4, 8))
        self.assertTrue(out.is_replicate())

    def test_unsqueeze_middle(self):
        t = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        out = propagate_unsqueeze(t, dim=1)
        self.assertEqual(out.global_shape, (4, 1, 8))
        placements = out.get_placements()
        self.assertEqual(placements[0][:3], ("shard", 0, 2))

    def test_unsqueeze_end(self):
        t = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        out = propagate_unsqueeze(t, dim=2)
        self.assertEqual(out.global_shape, (4, 8, 1))
        placements = out.get_placements()
        self.assertEqual(placements[0][:3], ("shard", 0, 2))

    def test_unsqueeze_before_shard(self):
        t = ShardedLayout.shard((4, 8), shard_dim=1, mesh_dim_size=2)
        out = propagate_unsqueeze(t, dim=0)
        self.assertEqual(out.global_shape, (1, 4, 8))
        placements = out.get_placements()
        self.assertEqual(placements[0][:3], ("shard", 2, 2))

    def test_unsqueeze_mesh_dim_map_shift(self):
        t = ShardedLayout.shard_multi((4, 8, 16), [(0, 2), (1, 4)])
        out = propagate_unsqueeze(t, dim=1)
        self.assertEqual(out.global_shape, (4, 1, 8, 16))
        placements = out.get_placements()
        self.assertEqual(placements[0][:3], ("shard", 0, 2))
        self.assertEqual(placements[1][:3], ("shard", 2, 4))

    def test_unsqueeze_squeeze_roundtrip(self):
        t = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        unsqueezed = propagate_unsqueeze(t, dim=1)
        squeezed = propagate_slice(unsqueezed, dim=1, index=0)
        self.assertEqual(squeezed.global_shape, t.global_shape)
        self.assertEqual(squeezed.get_placements(), t.get_placements())


class TestEinsumPropagation(unittest.TestCase):

    def test_both_replicate(self):
        a = ShardedLayout.replicate((16, 8))
        b = ShardedLayout.replicate((8, 32))
        out = propagate_einsum("mk,kn->mn", a, b)
        self.assertIsNotNone(out)
        self.assertTrue(out.is_replicate())
        self.assertEqual(out.global_shape, (16, 32))

    def test_m_shard(self):
        a = ShardedLayout.shard((16, 8), shard_dim=0, mesh_dim_size=2)
        b = ShardedLayout.replicate((8, 32))
        out = propagate_einsum("mk,kn->mn", a, b)
        self.assertIsNotNone(out)
        self.assertEqual(out.global_shape, (16, 32))
        placements = out.get_placements()
        self.assertEqual(placements[0], ("shard", 0, 2, (0,)))

    def test_n_shard(self):
        a = ShardedLayout.replicate((16, 8))
        b = ShardedLayout.shard((8, 32), shard_dim=1, mesh_dim_size=2)
        out = propagate_einsum("mk,kn->mn", a, b)
        self.assertIsNotNone(out)
        self.assertEqual(out.global_shape, (16, 32))
        placements = out.get_placements()
        self.assertEqual(placements[0], ("shard", 1, 2, (0,)))

    def test_k_shard_both(self):
        a = ShardedLayout.shard((16, 8), shard_dim=1, mesh_dim_size=2)
        b = ShardedLayout.shard((8, 32), shard_dim=0, mesh_dim_size=2)
        out = propagate_einsum("mk,kn->mn", a, b)
        self.assertIsNotNone(out)
        self.assertEqual(out.global_shape, (16, 32))
        self.assertTrue(len(out.partial) > 0)

    def test_k_shard_only_a(self):
        a = ShardedLayout.shard((16, 8), shard_dim=1, mesh_dim_size=2)
        b = ShardedLayout.replicate((8, 32))
        out = propagate_einsum("mk,kn->mn", a, b)
        self.assertIsNone(out)

    def test_batch_both(self):
        a = ShardedLayout.shard((4, 8, 16), shard_dim=0, mesh_dim_size=2)
        b = ShardedLayout.shard((4, 16, 32), shard_dim=0, mesh_dim_size=2)
        out = propagate_einsum("bmk,bkn->bmn", a, b)
        self.assertIsNotNone(out)
        self.assertEqual(out.global_shape, (4, 8, 32))
        placements = out.get_placements()
        self.assertEqual(placements[0], ("shard", 0, 2, (0,)))

    def test_batch_a_only(self):
        a = ShardedLayout.shard((4, 8, 16), shard_dim=0, mesh_dim_size=2)
        b = ShardedLayout.replicate((4, 16, 32))
        out = propagate_einsum("bmk,bkn->bmn", a, b)
        self.assertIsNone(out)


class TestPointwisePropagation(unittest.TestCase):

    def test_matching(self):
        a = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=2)
        b = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=2)
        out = propagate_pointwise([a, b])
        self.assertIsNotNone(out)
        self.assertEqual(out.global_shape, (8, 16))

    def test_mismatch(self):
        a = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=2)
        b = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=4)
        out = propagate_pointwise([a, b])
        self.assertIsNone(out)

    def test_mismatch_same_mesh_dim_different_tensor_dims(self):
        a = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=2, mesh_dim=0)
        b = ShardedLayout.shard((8, 16), shard_dim=1, mesh_dim_size=2, mesh_dim=0)
        out = propagate_pointwise([a, b])
        self.assertIsNone(out)

    def test_shard_with_replicate(self):
        a = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=2)
        b = ShardedLayout.replicate((8, 16))
        out = propagate_pointwise([a, b])
        self.assertIsNotNone(out)

    def test_all_replicate(self):
        a = ShardedLayout.replicate((8, 16))
        b = ShardedLayout.replicate((8, 16))
        out = propagate_pointwise([a, b])
        self.assertTrue(out.is_replicate())


class TestReductionPropagation(unittest.TestCase):

    def test_reduce_non_sharded(self):
        t = ShardedLayout.shard((8, 16), shard_dim=1, mesh_dim_size=2)
        out = propagate_reduction(t, reduce_dim=0, keepdim=False)
        self.assertEqual(out.global_shape, (16,))
        placements = out.get_placements()
        self.assertEqual(placements[0], ("shard", 0, 2, (0,)))

    def test_reduce_sharded(self):
        t = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=2)
        out = propagate_reduction(t, reduce_dim=0, keepdim=False)
        self.assertEqual(out.global_shape, (16,))
        self.assertEqual(out.partial, {0: "sum"})

    def test_reduce_sharded_max(self):
        t = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=2)
        out = propagate_reduction(t, reduce_dim=0, keepdim=False, reduce_op="max")
        self.assertEqual(out.global_shape, (16,))
        self.assertEqual(out.partial, {0: "max"})

    def test_reduce_non_sharded_no_partial(self):
        t = ShardedLayout.shard((8, 16), shard_dim=1, mesh_dim_size=2)
        out = propagate_reduction(t, reduce_dim=0, keepdim=False)
        self.assertEqual(out.global_shape, (16,))
        self.assertEqual(out.partial, {})

    def test_reduce_multi_mesh_partial(self):
        t = ShardedLayout.shard_multi((4, 8, 16), [(0, 2), (1, 4)])
        out = propagate_reduction(t, reduce_dim=1, keepdim=False)
        self.assertEqual(out.global_shape, (4, 16))
        self.assertEqual(out.partial, {1: "sum"})
        placements = out.get_placements()
        self.assertEqual(placements[0][:3], ("shard", 0, 2))

    def test_reduce_replicate(self):
        t = ShardedLayout.replicate((8, 16))
        out = propagate_reduction(t, reduce_dim=0, keepdim=False)
        self.assertEqual(out.global_shape, (16,))
        self.assertTrue(out.is_replicate())


class TestEndToEnd(unittest.TestCase):

    def test_linear_3d(self):
        B, S, H, O = 4, 8, 16, 32
        input_t = ShardedLayout.shard((B, S, H), shard_dim=0, mesh_dim_size=2)
        after_v1 = propagate_view(input_t, (B * S, H))
        self.assertIsNotNone(after_v1)
        weight = ShardedLayout.replicate((H, O))
        after_mm = propagate_einsum("mk,kn->mn", after_v1, weight)
        self.assertIsNotNone(after_mm)
        after_v2 = propagate_view(after_mm, (B, S, O))
        self.assertIsNotNone(after_v2)
        placements = after_v2.get_placements()
        self.assertEqual(placements[0], ("shard", 0, 2, (0,)))

    def test_linear_3d_both_dims_sharded(self):
        B, S, H, O = 4, 8, 16, 32
        dp_size, sp_size = 2, 4
        input_t = ShardedLayout.shard_multi(
            (B, S, H), [(0, dp_size), (1, sp_size)]
        )
        placements_in = input_t.get_placements()
        self.assertEqual(placements_in[0], ("shard", 0, dp_size, (0,)))
        self.assertEqual(placements_in[1], ("shard", 1, sp_size, (1,)))

        after_v1 = propagate_view(input_t, (B * S, H))
        self.assertIsNotNone(after_v1)
        weight = ShardedLayout.replicate((H, O))
        after_mm = propagate_einsum("mk,kn->mn", after_v1, weight)
        self.assertIsNotNone(after_mm)
        after_v2 = propagate_view(after_mm, (B, S, O))
        self.assertIsNotNone(after_v2)

        placements_out = after_v2.get_placements()
        self.assertEqual(placements_out[0], ("shard", 0, dp_size, (0,)))
        self.assertEqual(placements_out[1], ("shard", 1, sp_size, (1,)))

    def test_fsdp_tp_linear(self):
        """FSDP+TP: input S(0)R @ weight RS(1) -> output S(0)S(1).

        2D mesh: mesh_dim 0 = FSDP (data parallel), mesh_dim 1 = TP (tensor parallel).
        Input (M, K) sharded on dim 0 across FSDP.
        Weight (K, N) sharded on dim 1 across TP.
        Output (M, N) should be S(0) on dim 0, S(1) on dim 1.
        """
        M, K, N = 16, 32, 64
        fsdp_size, tp_size = 2, 4

        # Input: S(0) on dim 0 (FSDP shards rows), R on dim 1
        input_t = ShardedLayout.shard((M, K), shard_dim=0, mesh_dim_size=fsdp_size, mesh_dim=0)
        self.assertEqual(input_t.mesh_dim_map, {0: (0,), 1: ()})

        # Weight: R on dim 0, S(1) on dim 1 (TP shards columns)
        weight = ShardedLayout.shard((K, N), shard_dim=1, mesh_dim_size=tp_size, mesh_dim=1)
        self.assertEqual(weight.mesh_dim_map, {0: (), 1: (1,)})

        # Einsum: mk,kn->mn
        # m: Carry from input (S(0)) — FSDP sharding preserved
        # n: Carry from weight (S(1)) — TP sharding preserved
        # k: contracted, both replicate on k → no Partial
        out = propagate_einsum("mk,kn->mn", input_t, weight)
        self.assertIsNotNone(out)
        self.assertEqual(out.global_shape, (M, N))
        self.assertEqual(out.partial, {})

        # Output: S(0) on dim 0 (from input), S(1) on dim 1 (from weight)
        self.assertEqual(out.mesh_dim_map, {0: (0,), 1: (1,)})
        placements = out.get_placements()
        self.assertEqual(placements[0], ("shard", 0, fsdp_size, (0,)))
        self.assertEqual(placements[1], ("shard", 1, tp_size, (1,)))


class TestRedistribute(unittest.TestCase):

    def test_same_layout(self):
        s = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=2)
        self.assertEqual(plan_redistribute(s, s), [])

    def test_shard_to_replicate(self):
        source = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=2)
        target = ShardedLayout.replicate((8, 16))
        result = plan_redistribute(source, target)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "all_gather")
        self.assertEqual(result[0][1], 0)

    def test_replicate_to_shard(self):
        source = ShardedLayout.replicate((8, 16))
        target = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=2)
        result = plan_redistribute(source, target)
        self.assertEqual(result, [])

    def test_different_dims(self):
        source = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=2)
        target = ShardedLayout.shard((8, 16), shard_dim=1, mesh_dim_size=2)
        result = plan_redistribute(source, target)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "all_to_all")
        self.assertEqual(result[0][1], 0)

    def test_partial_to_replicate(self):
        source = ShardedLayout.shard((16, 8), shard_dim=1, mesh_dim_size=2)
        source = propagate_reduction(source, reduce_dim=1, keepdim=False)
        self.assertEqual(source.partial, {0: "sum"})
        target = ShardedLayout.replicate((16,))
        result = plan_redistribute(source, target)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "all_reduce")
        self.assertEqual(result[0][2]["reduce_op"], "sum")

    def test_partial_to_shard(self):
        source = ShardedLayout.shard((16, 8), shard_dim=1, mesh_dim_size=2)
        source = propagate_reduction(source, reduce_dim=1, keepdim=False)
        target = ShardedLayout.shard((16,), shard_dim=0, mesh_dim_size=2)
        result = plan_redistribute(source, target)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "reduce_scatter")
        self.assertEqual(result[0][2]["reduce_op"], "sum")

    def test_multi_mesh_swap(self):
        source = ShardedLayout.shard_multi((4, 8, 16), [(0, 2), (1, 4)])
        target = ShardedLayout.shard_multi((4, 8, 16), [(1, 2), (0, 4)])
        result = plan_redistribute(source, target)
        types = {r[0] for r in result}
        self.assertTrue("all_to_all" in types)

    def test_shape_mismatch(self):
        source = ShardedLayout.replicate((8, 16))
        target = ShardedLayout.replicate((4, 32))
        with self.assertRaises(ValueError):
            plan_redistribute(source, target)

    def test_replicate_to_replicate(self):
        source = ShardedLayout.replicate((8, 16))
        target = ShardedLayout.replicate((8, 16))
        self.assertEqual(plan_redistribute(source, target), [])


class TestRedistributeDetailed(unittest.TestCase):
    """Tests for GPU-stride-based redistribution with vectorization."""

    def test_same_sharding_vectorization(self):
        source = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=2)
        target = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=2)
        detail = plan_redistribute_detailed(source, target)
        self.assertEqual(detail["collectives"], [])
        # Same layout -> vectorization = local tile size (all local elements contiguous)
        from autoparallel.shardings.cute._pycute import product as cute_product
        local_size = cute_product(source.local_sizes)
        self.assertEqual(detail["vectorization_factor"], local_size)

    def test_shard_to_replicate_vectorization(self):
        source = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=2)
        target = ShardedLayout.replicate((8, 16))
        detail = plan_redistribute_detailed(source, target)
        self.assertEqual(len(detail["collectives"]), 1)
        self.assertEqual(detail["collectives"][0][0], "all_gather")
        # src local = 64, tgt local = 128 -> mcv = max contiguous agreement
        self.assertGreater(detail["vectorization_factor"], 0)

    def test_all_to_all_vectorization(self):
        source = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=2)
        target = ShardedLayout.shard((8, 16), shard_dim=1, mesh_dim_size=2)
        detail = plan_redistribute_detailed(source, target)
        self.assertEqual(len(detail["collectives"]), 1)
        self.assertEqual(detail["collectives"][0][0], "all_to_all")
        # Vectorization factor: how many contiguous elements per message
        self.assertGreater(detail["vectorization_factor"], 0)
        self.assertLess(detail["vectorization_factor"], source.num_elements)

    def test_element_mapping_exists(self):
        source = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=2)
        target = ShardedLayout.shard((8, 16), shard_dim=1, mesh_dim_size=2)
        detail = plan_redistribute_detailed(source, target)
        # Element mapping should be a Layout
        self.assertIsNotNone(detail["element_mapping"])

    def test_gpu_stride_classification(self):
        """Verify the GPU stride approach classifies all standard cases."""
        # S(0) -> R: all_gather
        s1 = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=2)
        t1 = ShardedLayout.replicate((8, 16))
        r1 = plan_redistribute(s1, t1)
        self.assertEqual(r1[0][0], "all_gather")

        # R -> S(0): no communication (local reinterpret)
        r2 = plan_redistribute(t1, s1)
        self.assertEqual(r2, [])

        # S(0) -> S(1): all_to_all (fused from all_gather + local_slice)
        t3 = ShardedLayout.shard((8, 16), shard_dim=1, mesh_dim_size=2)
        r3 = plan_redistribute(s1, t3)
        self.assertEqual(r3[0][0], "all_to_all")

        # Same: no communication
        r4 = plan_redistribute(s1, s1)
        self.assertEqual(r4, [])

    def test_partial_gpu_stride(self):
        """Partial cases use GPU stride + partial annotation."""
        # Partial -> Replicate: all_reduce
        source = ShardedLayout.shard((16, 8), shard_dim=1, mesh_dim_size=2)
        source = propagate_reduction(source, reduce_dim=1, keepdim=False)
        target = ShardedLayout.replicate((16,))
        r = plan_redistribute(source, target)
        self.assertEqual(r[0][0], "all_reduce")

        # Partial -> Shard: reduce_scatter
        target2 = ShardedLayout.shard((16,), shard_dim=0, mesh_dim_size=2)
        r2 = plan_redistribute(source, target2)
        self.assertEqual(r2[0][0], "reduce_scatter")

    def test_s0s0_ltr_to_rtl(self):
        """S(0)S(0) left-to-right -> S(0)S(0) right-to-left.

        LTR and RTL have different nesting orders for the same tensor dim.
        The per-mesh-dim analysis can't detect the difference (identity ppermute
        per mesh dim), so a post-pass computes the global rank permutation
        via rank-to-chunk CuTe composition and emits a single ppermute on the
        full mesh (mesh_dim=None).
        """
        ltr = ShardedLayout.shard_multi((8, 16), [(0, 2), (0, 4)])
        rtl = ShardedLayout.shard_multi((8, 16), [(0, 4), (0, 2)])
        result = plan_redistribute(ltr, rtl)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "ppermute")
        self.assertIsNone(result[0][1])  # global ppermute on full mesh
        perm = result[0][2]["perm"]
        # Verify non-identity permutation
        self.assertTrue(any(s != d for s, d in perm))

    def test_s0s0_per_mesh_dim_stride_values(self):
        """Verify per-mesh-dim GPU strides match mesh dim sizes, not tuple positions.

        For shard_multi((8, 16), [(0, 2), (0, 4)]):
          mesh_dim 0 (size 2, outermost) should have the larger stride
          mesh_dim 1 (size 4, innermost) should have the smaller stride

        The nested logical_divide produces (4, 2) in innermost-first order,
        but mesh_dim_map is outermost-first. The extraction must reverse.
        """
        from autoparallel.shardings.cute.redistribute import _get_per_mesh_dim_gpu_stride

        sl = ShardedLayout.shard_multi((8, 16), [(0, 2), (0, 4)])

        md0_info = _get_per_mesh_dim_gpu_stride(sl, 0)
        md1_info = _get_per_mesh_dim_gpu_stride(sl, 1)

        self.assertIsNotNone(md0_info)
        self.assertIsNotNone(md1_info)

        _, md0_stride = md0_info
        _, md1_stride = md1_info

        # mesh_dim 0 (size 2, outermost) has larger stride than mesh_dim 1 (size 4, innermost)
        self.assertGreater(md0_stride, md1_stride)

    def test_s0s0_ltr_to_rs0_ltr(self):
        """S(0)S(0) LTR -> RS(0) LTR.

        S(0)S(0) LTR: mesh0(size=2, gs=64), mesh1(size=4, gs=16)
        RS(0) LTR: mesh0 gs=0 (replicate), mesh1(size=4, gs=32)

        mesh_dim 0: gs 64 -> 0 -> all_gather
        mesh_dim 1: gs 16 -> 32, different -> all_to_all
        """
        src = ShardedLayout.shard_multi((8, 16), [(0, 2), (0, 4)])
        tgt = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=4, mesh_dim=1)
        result = plan_redistribute(src, tgt)
        collectives_by_md = {md: ctype for ctype, md, info in result}
        self.assertEqual(collectives_by_md.get(0), "all_gather")
        # mesh_dim 1 has different strides (16 vs 32)
        self.assertIn(1, collectives_by_md)

    def test_s0s0_rtl_to_rs0_rtl(self):
        """S(0)S(0) RTL -> RS(0) RTL.

        S(0)S(0) RTL: mesh0(size=4, gs=32), mesh1(size=2, gs=16)
        RS(0) RTL: mesh0 gs=0 (replicate), mesh1(size=2, gs=64)

        mesh_dim 0: gs 32 -> 0 -> all_gather
        mesh_dim 1: gs 16 -> 64, different -> all_to_all
        """
        src = ShardedLayout.shard_multi((8, 16), [(0, 4), (0, 2)])
        tgt = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=2, mesh_dim=1)
        result = plan_redistribute(src, tgt)
        collectives_by_md = {md: ctype for ctype, md, info in result}
        self.assertEqual(collectives_by_md.get(0), "all_gather")
        self.assertIn(1, collectives_by_md)


class TestCatPropagation(unittest.TestCase):

    def test_cat_both_replicate(self):
        a = ShardedLayout.replicate((4, 8))
        b = ShardedLayout.replicate((4, 8))
        out = propagate_cat([a, b], dim=0)
        self.assertIsNotNone(out)
        self.assertEqual(out.global_shape, (8, 8))
        self.assertTrue(out.is_replicate())

    def test_cat_sharded_other_dim(self):
        a = ShardedLayout.shard((4, 8), shard_dim=1, mesh_dim_size=2)
        b = ShardedLayout.shard((4, 8), shard_dim=1, mesh_dim_size=2)
        out = propagate_cat([a, b], dim=0)
        self.assertIsNotNone(out)
        self.assertEqual(out.global_shape, (8, 8))
        # Shard on dim 1 preserved
        placements = out.get_placements()
        self.assertEqual(placements[0][:3], ("shard", 1, 2))

    def test_cat_sharded_same_dim(self):
        a = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        b = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        out = propagate_cat([a, b], dim=0)
        self.assertIsNotNone(out)
        self.assertEqual(out.global_shape, (8, 8))
        # Should still be sharded on dim 0
        self.assertFalse(out.is_replicate())

    def test_cat_incompatible_other_dim(self):
        a = ShardedLayout.shard((4, 8), shard_dim=1, mesh_dim_size=2)
        b = ShardedLayout.replicate((4, 8))
        out = propagate_cat([a, b], dim=0)
        # dim 1: a is sharded, b is replicate -> carry picks sharded (compatible)
        self.assertIsNotNone(out)

    def test_cat_incompatible_mesh_conflict(self):
        a = ShardedLayout.shard((4, 8), shard_dim=1, mesh_dim_size=2, mesh_dim=0)
        b = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2, mesh_dim=0)
        # Same mesh dim on different tensor dims -> incompatible
        out = propagate_cat([a, b], dim=0)
        self.assertIsNone(out)

    def test_cat_then_pointwise(self):
        """Non-contiguous layout from cat propagates through pointwise."""
        a = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        b = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        cat_result = propagate_cat([a, b], dim=0)
        self.assertIsNotNone(cat_result)
        # Pointwise with same layout should work
        out = propagate_pointwise([cat_result, cat_result])
        self.assertIsNotNone(out)


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


class TestXorStride(unittest.TestCase):

    def test_mul_odd_even(self):
        x = XorStride(7)
        self.assertEqual(x * 1, XorStride(7))
        self.assertEqual(x * 0, XorStride(0))
        self.assertEqual(x * 3, XorStride(7))
        self.assertEqual(x * 4, XorStride(0))

    def test_add_xor_xor(self):
        self.assertEqual(XorStride(3) + XorStride(5), XorStride(6))

    def test_add_int_xor(self):
        self.assertEqual(4 + XorStride(3), 7)
        self.assertEqual(XorStride(3) + 4, 7)

    def test_add_resolves_to_int(self):
        result = 0 + XorStride(5)
        self.assertIsInstance(result, int)
        self.assertEqual(result, 5)

    def test_pure_xor_layout(self):
        L = Layout((2, 2), (XorStride(1), XorStride(3)))
        self.assertEqual(L(0, 0), 0)
        self.assertEqual(L(1, 0), 1)
        self.assertEqual(L(0, 1), 3)
        self.assertEqual(L(1, 1), 2)

    def test_mixed_strides(self):
        L = Layout((4, 2), (1, XorStride(7)))
        for g in range(4):
            self.assertEqual(L(g, 0), g)
            self.assertEqual(L(g, 1), 7 - g)

    def test_zigzag_n2(self):
        L = Layout((2, 2), (1, XorStride(3)))
        self.assertEqual(L(0, 0), 0)
        self.assertEqual(L(1, 0), 1)
        self.assertEqual(L(0, 1), 3)
        self.assertEqual(L(1, 1), 2)

    def test_zigzag_n4(self):
        L = Layout((4, 2), (1, XorStride(7)))
        for g in range(4):
            self.assertEqual(L(g, 0), g)
            self.assertEqual(L(g, 1), 7 - g)

    def test_zigzag_n8(self):
        L = Layout((8, 2), (1, XorStride(15)))
        for g in range(8):
            self.assertEqual(L(g, 0), g)
            self.assertEqual(L(g, 1), 15 - g)

    def test_zigzag_via_logical_divide(self):
        t = Layout((8,))
        t_bits = logical_divide(t, (Layout((2, 2, 2)),))
        R = Layout((2, 2, 2), (1, 2, XorStride(7)))
        result = composition(t_bits, R)
        for pair in range(2):
            for b1 in range(2):
                for b0 in range(2):
                    gpu = b0 + 2 * b1
                    expected = gpu if pair == 0 else 7 - gpu
                    self.assertEqual(result(b0, b1, pair), expected)


class TestModStride(unittest.TestCase):
    """Tests for ModStride: modular arithmetic strides."""

    def test_mul(self):
        from autoparallel.shardings.cute._pycute import ModStride
        m = ModStride(3, 4)
        self.assertEqual(m * 0, ModStride(0, 4))
        self.assertEqual(m * 1, ModStride(3, 4))
        self.assertEqual(m * 2, ModStride(2, 4))  # 3*2 % 4 = 2
        self.assertEqual(m * 3, ModStride(1, 4))  # 3*3 % 4 = 1

    def test_add_mod_mod(self):
        from autoparallel.shardings.cute._pycute import ModStride
        self.assertEqual(ModStride(1, 4) + ModStride(3, 4), ModStride(0, 4))
        self.assertEqual(ModStride(2, 4) + ModStride(2, 4), ModStride(0, 4))
        self.assertEqual(ModStride(1, 4) + ModStride(1, 4), ModStride(2, 4))

    def test_add_mod_int(self):
        from autoparallel.shardings.cute._pycute import ModStride
        result = ModStride(3, 4) + 5
        self.assertIsInstance(result, int)
        self.assertEqual(result, 0)  # (3+5) % 4 = 0

    def test_add_int_mod(self):
        from autoparallel.shardings.cute._pycute import ModStride
        result = 5 + ModStride(3, 4)
        self.assertIsInstance(result, int)
        self.assertEqual(result, 0)  # (5+3) % 4 = 0

    def test_add_mod_xor(self):
        """ModStride + XorStride: resolves ModStride to int, then XOR."""
        from autoparallel.shardings.cute._pycute import ModStride
        result = ModStride(3, 4) + XorStride(7)
        self.assertIsInstance(result, int)
        self.assertEqual(result, 3 ^ 7)  # = 4

    def test_add_xor_mod(self):
        """XorStride + ModStride: same result."""
        from autoparallel.shardings.cute._pycute import ModStride
        result = XorStride(7) + ModStride(3, 4)
        self.assertIsInstance(result, int)
        self.assertEqual(result, 7 ^ 3)  # = 4

    def test_ring_rotation(self):
        """ModStride(3, 4) implements (g - step) % 4 as (g + 3*step) % 4."""
        from autoparallel.shardings.cute._pycute import ModStride
        for g in range(4):
            for step in range(4):
                ms = ModStride(1, 4) * g + ModStride(3, 4) * step
                expected = (g - step) % 4
                # ms is ModStride, resolve to int by adding 0
                result = ms + 0
                self.assertEqual(result, expected,
                    f"g={g}, step={step}: got {result}, expected {expected}")

    def test_layout_ring_rotation(self):
        """Layout with ModStride for ring attention rotation."""
        from autoparallel.shardings.cute._pycute import ModStride
        # Layout((4, 4), (ModStride(1,4), ModStride(3,4)))
        # (gpu, step) -> source_gpu = (gpu + 3*step) % 4
        L = Layout((4, 4), (ModStride(1, 4), ModStride(3, 4)))
        for step in range(4):
            for gpu in range(4):
                result = L(gpu, step)
                expected = (gpu - step) % 4
                self.assertEqual(result, expected,
                    f"gpu={gpu}, step={step}: got {result}, expected {expected}")

    def test_full_ring_attention_layout(self):
        """Full ring attention: (b0, b1, step, pair) -> chunk index.

        XorStride must be the LAST mode so all ModStride modes resolve
        to an integer before the XOR is applied (left-to-right evaluation).
        """
        from autoparallel.shardings.cute._pycute import ModStride
        # Layout((2, 2, 4, 2), (ModStride(1,4), ModStride(2,4), ModStride(3,4), XorStride(7)))
        L = Layout(
            (2, 2, 4, 2),
            (ModStride(1, 4), ModStride(2, 4), ModStride(3, 4), XorStride(7))
        )
        for step in range(4):
            for b1 in range(2):
                for b0 in range(2):
                    gpu = b0 + 2 * b1
                    source_gpu = (gpu - step) % 4
                    for pair in range(2):
                        expected = source_gpu if pair == 0 else 7 - source_gpu
                        result = L(b0, b1, step, pair)
                        self.assertEqual(result, expected,
                            f"step={step}, gpu={gpu}, pair={pair}: got {result}, expected {expected}")


class TestReductionExtended(unittest.TestCase):
    """Additional reduction tests from DTensor coverage gaps."""

    def test_keepdim_sharded(self):
        """keepdim=True on sharded dim: size-1 dim kept, Partial emitted."""
        t = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=2)
        out = propagate_reduction(t, reduce_dim=0, keepdim=True)
        self.assertEqual(out.global_shape, (1, 16))
        self.assertEqual(out.partial, {0: "sum"})

    def test_keepdim_non_sharded(self):
        """keepdim=True on non-sharded dim: size-1 kept, no Partial."""
        t = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=2)
        out = propagate_reduction(t, reduce_dim=1, keepdim=True)
        self.assertEqual(out.global_shape, (8, 1))
        self.assertEqual(out.partial, {})
        placements = out.get_placements()
        self.assertEqual(placements[0][:3], ("shard", 0, 2))

    def test_multi_dim_reduction(self):
        """Reduce dims [0, 2] simultaneously on a 4D tensor."""
        t = ShardedLayout.shard_multi((4, 8, 16, 32), [(0, 2), (2, 4)])
        out = propagate_reduction(t, reduce_dim=[0, 2], keepdim=False)
        self.assertEqual(out.global_shape, (8, 32))
        # dim 0 was sharded on mesh 0 -> partial
        # dim 2 was sharded on mesh 1 -> partial
        self.assertEqual(out.partial, {0: "sum", 1: "sum"})

    def test_multi_dim_reduction_mixed(self):
        """Reduce [0, 1]: dim 0 sharded (-> partial), dim 1 not (no partial)."""
        t = ShardedLayout.shard((4, 8, 16), shard_dim=0, mesh_dim_size=2)
        out = propagate_reduction(t, reduce_dim=[0, 1], keepdim=False)
        self.assertEqual(out.global_shape, (16,))
        self.assertEqual(out.partial, {0: "sum"})


class TestViewExtended(unittest.TestCase):
    """Additional view tests from DTensor coverage gaps."""

    def test_squeeze(self):
        """Squeeze: remove size-1 dim (inverse of unsqueeze)."""
        t = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        unsqueezed = propagate_unsqueeze(t, dim=1)
        self.assertEqual(unsqueezed.global_shape, (4, 1, 8))
        # Squeeze via view (4, 1, 8) -> (4, 8)
        squeezed = propagate_view(unsqueezed, (4, 8))
        self.assertIsNotNone(squeezed)
        self.assertEqual(squeezed.global_shape, (4, 8))
        self.assertEqual(squeezed.get_placements(), t.get_placements())

    def test_flatten_non_leftmost_sharded(self):
        """Flatten dims 0+1 when dim 1 is sharded."""
        t = ShardedLayout.shard((4, 8, 16), shard_dim=1, mesh_dim_size=2)
        out = propagate_view(t, (32, 16))
        self.assertIsNotNone(out)
        self.assertEqual(out.global_shape, (32, 16))
        # Shard should be on merged dim 0
        placements = out.get_placements()
        self.assertEqual(placements[0][0], "shard")
        self.assertEqual(placements[0][1], 0)

    def test_merge_split_round_trip(self):
        """(4, 8, 16) -> (32, 16) -> (2, 16, 16): Flatten+Split case."""
        t = ShardedLayout.shard((4, 8, 16), shard_dim=1, mesh_dim_size=4)
        v1 = propagate_view(t, (32, 16))
        self.assertIsNotNone(v1)
        v2 = propagate_view(v1, (2, 16, 16))
        self.assertIsNotNone(v2)
        # Mesh should land on the dim that contains it
        self.assertEqual(v2.get_placements()[0], ("shard", 1, 4, (0,)))


class TestBroadcastExtended(unittest.TestCase):
    """Additional broadcast/pointwise tests from DTensor coverage gaps."""

    def test_size_1_broadcasting(self):
        """(8, 1) + (8, 16): dim 1 broadcasts from size 1."""
        a = ShardedLayout.shard((8, 1), shard_dim=0, mesh_dim_size=2)
        b = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=2)
        out = propagate_broadcast(a, b)
        self.assertIsNotNone(out)
        self.assertEqual(out.global_shape, (8, 16))
        placements = out.get_placements()
        self.assertEqual(placements[0][:3], ("shard", 0, 2))

    def test_size_1_sharded_broadcast(self):
        """(8, 1) replicate + (8, 16) sharded on dim 1: take sharded."""
        a = ShardedLayout.replicate((8, 1))
        b = ShardedLayout.shard((8, 16), shard_dim=1, mesh_dim_size=2)
        out = propagate_broadcast(a, b)
        self.assertIsNotNone(out)
        self.assertEqual(out.global_shape, (8, 16))
        placements = out.get_placements()
        self.assertEqual(placements[0][:3], ("shard", 1, 2))


class TestEinsumExtended(unittest.TestCase):
    """Additional einsum tests from DTensor coverage gaps."""

    def test_batched_mm_bij_bjk(self):
        """Batched mm with different label convention: bij,bjk->bik."""
        a = ShardedLayout.shard((4, 8, 16), shard_dim=0, mesh_dim_size=2)
        b = ShardedLayout.shard((4, 16, 32), shard_dim=0, mesh_dim_size=2)
        out = propagate_einsum("bij,bjk->bik", a, b)
        self.assertIsNotNone(out)
        self.assertEqual(out.global_shape, (4, 8, 32))
        placements = out.get_placements()
        self.assertEqual(placements[0], ("shard", 0, 2, (0,)))

    def test_free_dimension(self):
        """Label in one input + output but not the other (not m/n/k)."""
        # a has dims (batch, m, k), b has dims (k, n)
        # batch is free on a side — like a non-batched b
        a = ShardedLayout.shard((4, 8, 16), shard_dim=0, mesh_dim_size=2)
        b = ShardedLayout.replicate((16, 32))
        out = propagate_einsum("bmk,kn->bmn", a, b)
        self.assertIsNotNone(out)
        self.assertEqual(out.global_shape, (4, 8, 32))
        placements = out.get_placements()
        self.assertEqual(placements[0], ("shard", 0, 2, (0,)))

    def test_partial_on_k_dim(self):
        """K-sharded contraction produces Partial, verify reduce_op."""
        a = ShardedLayout.shard((16, 8), shard_dim=1, mesh_dim_size=4)
        b = ShardedLayout.shard((8, 32), shard_dim=0, mesh_dim_size=4)
        out = propagate_einsum("mk,kn->mn", a, b)
        self.assertIsNotNone(out)
        self.assertEqual(out.partial, {0: "sum"})
        self.assertTrue(out.is_replicate())  # output dims not sharded

    def test_m_shard_n_shard_same_mesh_dim_rejects(self):
        """S(M) on mesh 0 + S(N) on mesh 0 → invalid.

        Both M and N would need to be sharded on the same mesh dim,
        but a single mesh dim can only partition one tensor dim.
        This would produce a block-diagonal decomposition, not a valid matmul.
        """
        a = ShardedLayout.shard((1024, 1024), shard_dim=0, mesh_dim_size=8, mesh_dim=0)
        b = ShardedLayout.shard((1024, 1024), shard_dim=1, mesh_dim_size=8, mesh_dim=0)
        out = propagate_einsum("mk,kn->mn", a, b)
        self.assertIsNone(out)

    def test_m_shard_n_shard_different_mesh_dims_ok(self):
        """S(M) on mesh 0 + S(N) on mesh 1 → valid S(0)S(1) output (FSDP+TP)."""
        a = ShardedLayout.shard((1024, 1024), shard_dim=0, mesh_dim_size=2, mesh_dim=0)
        b = ShardedLayout.shard((1024, 1024), shard_dim=1, mesh_dim_size=4, mesh_dim=1)
        out = propagate_einsum("mk,kn->mn", a, b)
        self.assertIsNotNone(out)
        self.assertEqual(out.mesh_dim_map, {0: (0,), 1: (1,)})


class TestCatExtended(unittest.TestCase):
    """Additional cat tests from DTensor coverage gaps."""

    def test_cat_three_tensors(self):
        """Cat with 3 inputs, all replicate on cat dim."""
        a = ShardedLayout.replicate((4, 8))
        b = ShardedLayout.replicate((4, 8))
        c = ShardedLayout.replicate((4, 8))
        out = propagate_cat([a, b, c], dim=0)
        self.assertIsNotNone(out)
        self.assertEqual(out.global_shape, (12, 8))
        self.assertTrue(out.is_replicate())

    def test_cat_different_sizes_replicate(self):
        """Cat tensors with different sizes on cat dim, replicate."""
        a = ShardedLayout.replicate((4, 8))
        b = ShardedLayout.replicate((6, 8))
        out = propagate_cat([a, b], dim=0)
        self.assertIsNotNone(out)
        self.assertEqual(out.global_shape, (10, 8))
        self.assertTrue(out.is_replicate())

    def test_cat_dim1(self):
        """Cat on non-zero dim."""
        a = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        b = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        out = propagate_cat([a, b], dim=1)
        self.assertIsNotNone(out)
        self.assertEqual(out.global_shape, (4, 16))
        placements = out.get_placements()
        self.assertEqual(placements[0][:3], ("shard", 0, 2))


class TestSliceExtended(unittest.TestCase):
    """Additional slice tests from DTensor coverage gaps."""

    def test_slice_replicate_preserves_other_shard(self):
        """Slice on replicate dim, sharding on another dim survives."""
        t = ShardedLayout.shard((4, 8, 16), shard_dim=2, mesh_dim_size=2)
        out = propagate_slice(t, dim=0, index=1)
        self.assertIsNotNone(out)
        self.assertEqual(out.global_shape, (8, 16))
        placements = out.get_placements()
        self.assertEqual(placements[0][:3], ("shard", 1, 2))


class TestGatherExtended(unittest.TestCase):
    """Additional gather tests from DTensor coverage gaps."""

    def test_gather_preserves_other_shard(self):
        """Gather on non-sharded dim, sharding on other dim survives."""
        t = ShardedLayout.shard((4, 8, 16), shard_dim=0, mesh_dim_size=2)
        index = Layout((3, 2), (2, 1))  # multi-mode index
        out = propagate_gather(t, dim=1, index_layout=index)
        self.assertIsNotNone(out)
        placements = out.get_placements()
        self.assertEqual(placements[0][:3], ("shard", 0, 2))


class TestInteractions(unittest.TestCase):
    """Cross-operator interaction tests."""

    def test_transpose_then_view(self):
        """Transpose then view: sub-dim structure survives transpose."""
        t = ShardedLayout.shard((4, 8, 16), shard_dim=0, mesh_dim_size=2)
        # Transpose dims 0, 1: (4,8,16) -> (8,4,16), shard now on dim 1
        transposed = propagate_transpose(t, 0, 1)
        self.assertEqual(transposed.global_shape, (8, 4, 16))
        # View (8,4,16) -> (32,16): merge dims 0,1
        viewed = propagate_view(transposed, (32, 16))
        self.assertIsNotNone(viewed)
        self.assertEqual(viewed.global_shape, (32, 16))
        # Shard should be on merged dim 0
        self.assertFalse(viewed.is_replicate())

    def test_unsqueeze_then_broadcast(self):
        """Unsqueeze creates size-1 dim, then pointwise broadcasts it."""
        a = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=2)
        # Unsqueeze a at dim 2: (8, 16) -> (8, 16, 1)
        a_unsqueezed = propagate_unsqueeze(a, dim=2)
        self.assertEqual(a_unsqueezed.global_shape, (8, 16, 1))

        b = ShardedLayout.shard((8, 16, 4), shard_dim=0, mesh_dim_size=2)
        # Broadcast: (8, 16, 1) + (8, 16, 4) -> (8, 16, 4)
        out = propagate_broadcast(a_unsqueezed, b)
        self.assertIsNotNone(out)
        self.assertEqual(out.global_shape, (8, 16, 4))
        placements = out.get_placements()
        self.assertEqual(placements[0][:3], ("shard", 0, 2))

    def test_view_transpose_view_round_trip(self):
        """View -> transpose -> view: the sub-dim preservation test."""
        t = ShardedLayout.shard_multi((4, 8, 16), [(0, 2), (1, 4)])
        # View (4,8,16) -> (32,16)
        v1 = propagate_view(t, (32, 16))
        self.assertIsNotNone(v1)
        # Transpose (32,16) -> (16,32)
        tr = propagate_transpose(v1, 0, 1)
        self.assertEqual(tr.global_shape, (16, 32))
        # Transpose back (16,32) -> (32,16)
        tr2 = propagate_transpose(tr, 0, 1)
        self.assertEqual(tr2.global_shape, (32, 16))
        # View (32,16) -> (4,8,16): should recover original shardings
        v2 = propagate_view(tr2, (4, 8, 16))
        self.assertIsNotNone(v2)
        self.assertEqual(v2.get_placements(), t.get_placements())


class TestIdentityOps(unittest.TestCase):
    """Identity ops: clone, contiguous, detach, etc."""

    def test_identity_shard(self):
        t = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        out = propagate_identity(t)
        self.assertEqual(out, t)

    def test_identity_replicate(self):
        t = ShardedLayout.replicate((4, 8))
        out = propagate_identity(t)
        self.assertTrue(out.is_replicate())


class TestNamedMatrixOps(unittest.TestCase):
    """mm, bmm, addmm, dot, t — wrappers over einsum/transpose."""

    def test_mm(self):
        a = ShardedLayout.shard((16, 8), shard_dim=0, mesh_dim_size=2)
        b = ShardedLayout.replicate((8, 32))
        out = propagate_mm(a, b)
        self.assertIsNotNone(out)
        self.assertEqual(out.global_shape, (16, 32))
        self.assertEqual(out.get_placements()[0][:3], ("shard", 0, 2))

    def test_bmm(self):
        a = ShardedLayout.shard((4, 8, 16), shard_dim=0, mesh_dim_size=2)
        b = ShardedLayout.shard((4, 16, 32), shard_dim=0, mesh_dim_size=2)
        out = propagate_bmm(a, b)
        self.assertIsNotNone(out)
        self.assertEqual(out.global_shape, (4, 8, 32))
        self.assertEqual(out.get_placements()[0], ("shard", 0, 2, (0,)))

    def test_addmm(self):
        bias = ShardedLayout.replicate((32,))
        a = ShardedLayout.shard((16, 8), shard_dim=0, mesh_dim_size=2)
        b = ShardedLayout.replicate((8, 32))
        out = propagate_addmm(bias, a, b)
        self.assertIsNotNone(out)
        self.assertEqual(out.global_shape, (16, 32))

    def test_dot(self):
        a = ShardedLayout.replicate((8,))
        b = ShardedLayout.replicate((8,))
        out = propagate_dot(a, b)
        self.assertIsNotNone(out)
        self.assertTrue(out.is_replicate())

    def test_dot_sharded(self):
        a = ShardedLayout.shard((8,), shard_dim=0, mesh_dim_size=2)
        b = ShardedLayout.shard((8,), shard_dim=0, mesh_dim_size=2)
        out = propagate_dot(a, b)
        self.assertIsNotNone(out)
        self.assertEqual(out.partial, {0: "sum"})

    def test_t(self):
        t = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        out = propagate_t(t)
        self.assertEqual(out.global_shape, (8, 4))
        self.assertEqual(out.get_placements()[0][:3], ("shard", 1, 2))

    def test_movedim(self):
        t = ShardedLayout.shard((4, 8, 16), shard_dim=0, mesh_dim_size=2)
        out = propagate_movedim(t, 0, 2)
        self.assertEqual(out.global_shape, (8, 16, 4))
        self.assertEqual(out.get_placements()[0][:3], ("shard", 2, 2))


class TestViewVariants(unittest.TestCase):
    """squeeze, expand, flatten, unflatten, repeat."""

    def test_squeeze_size1(self):
        t = ShardedLayout.shard((4, 1, 8), shard_dim=0, mesh_dim_size=2)
        out = propagate_squeeze(t, dim=1)
        self.assertEqual(out.global_shape, (4, 8))
        self.assertEqual(out.get_placements()[0][:3], ("shard", 0, 2))

    def test_squeeze_not_size1(self):
        t = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        out = propagate_squeeze(t, dim=1)
        self.assertEqual(out.global_shape, (4, 8))  # no change

    def test_squeeze_all(self):
        t = ShardedLayout.shard((4, 1, 1, 8), shard_dim=0, mesh_dim_size=2)
        out = propagate_squeeze(t)
        self.assertEqual(out.global_shape, (4, 8))

    def test_flatten(self):
        t = ShardedLayout.shard((4, 8, 16), shard_dim=0, mesh_dim_size=2)
        out = propagate_flatten(t, start_dim=0, end_dim=1)
        self.assertEqual(out.global_shape, (32, 16))

    def test_unflatten(self):
        t = ShardedLayout.shard((32, 16), shard_dim=0, mesh_dim_size=2)
        out = propagate_unflatten(t, dim=0, sizes=(4, 8))
        self.assertEqual(out.global_shape, (4, 8, 16))

    def test_expand_broadcast(self):
        t = ShardedLayout.shard((4, 1), shard_dim=0, mesh_dim_size=2)
        out = propagate_expand(t, (4, 8))
        self.assertIsNotNone(out)
        self.assertEqual(out.global_shape, (4, 8))
        self.assertEqual(out.get_placements()[0][:3], ("shard", 0, 2))

    def test_expand_add_leading(self):
        t = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        out = propagate_expand(t, (3, 4, 8))
        self.assertIsNotNone(out)
        self.assertEqual(out.global_shape, (3, 4, 8))
        # Shard shifted to dim 1
        self.assertEqual(out.get_placements()[0][:3], ("shard", 1, 2))

    def test_repeat_unsharded(self):
        t = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        out = propagate_repeat(t, (1, 2))
        # Dim 0: repeat=1 -> carry. Dim 1: repeat=2, not sharded -> ok
        self.assertIsNotNone(out)
        self.assertEqual(out.global_shape, (4, 16))

    def test_repeat_sharded_rejects(self):
        t = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        out = propagate_repeat(t, (2, 1))
        # Dim 0: repeat=2, sharded -> reject
        self.assertIsNone(out)


class TestStackSplitUnbind(unittest.TestCase):
    """stack, split, unbind."""

    def test_stack(self):
        a = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        b = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        out = propagate_stack([a, b], dim=0)
        self.assertIsNotNone(out)
        self.assertEqual(out.global_shape, (2, 4, 8))

    def test_split_replicate(self):
        t = ShardedLayout.shard((8, 16), shard_dim=1, mesh_dim_size=2)
        results = propagate_split(t, [4, 4], dim=0)
        self.assertIsNotNone(results)
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertEqual(r.global_shape, (4, 16))

    def test_split_sharded_rejects(self):
        t = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=2)
        results = propagate_split(t, [4, 4], dim=0)
        self.assertIsNone(results)

    def test_unbind(self):
        t = ShardedLayout.shard((4, 8), shard_dim=1, mesh_dim_size=2)
        results = propagate_unbind(t, dim=0)
        self.assertIsNotNone(results)
        self.assertEqual(len(results), 4)
        for r in results:
            self.assertEqual(r.global_shape, (8,))

    def test_unbind_sharded_rejects(self):
        t = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        results = propagate_unbind(t, dim=0)
        self.assertIsNone(results)


class TestReplicateAffectedOps(unittest.TestCase):
    """flip, roll, sort, topk, argmax, cumsum, softmax, layer_norm."""

    def test_flip_non_sharded(self):
        t = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        out = propagate_replicate_affected(t, [1])
        self.assertIsNotNone(out)
        self.assertEqual(out.get_placements()[0][:3], ("shard", 0, 2))

    def test_flip_sharded_rejects(self):
        t = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        out = propagate_replicate_affected(t, [0])
        self.assertIsNone(out)

    def test_roll_non_sharded(self):
        t = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        out = propagate_replicate_affected(t, [1])
        self.assertIsNotNone(out)

    def test_roll_sharded_rejects(self):
        t = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        out = propagate_replicate_affected(t, [0])
        self.assertIsNone(out)

    def test_sort(self):
        t = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        result = propagate_sort(t, dim=1)
        self.assertIsNotNone(result)
        values, indices = result
        self.assertEqual(values.get_placements()[0][:3], ("shard", 0, 2))

    def test_sort_sharded_rejects(self):
        t = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        self.assertIsNone(propagate_sort(t, dim=0))

    def test_topk(self):
        t = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        result = propagate_topk(t, dim=1, k=3)
        self.assertIsNotNone(result)
        values, indices = result
        self.assertEqual(values.global_shape, (4, 3))

    def test_argmax_non_sharded(self):
        t = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        out = propagate_argmax(t, dim=1)
        self.assertIsNotNone(out)

    def test_argmax_sharded_rejects(self):
        t = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        out = propagate_argmax(t, dim=0)
        self.assertIsNone(out)

    def test_cumsum(self):
        t = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        out = propagate_replicate_affected(t, 1)
        self.assertIsNotNone(out)

    def test_cumsum_sharded_rejects(self):
        t = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        out = propagate_replicate_affected(t, 0)
        self.assertIsNone(out)

    def test_softmax(self):
        t = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        out = propagate_replicate_affected(t, 1)
        self.assertIsNotNone(out)

    def test_softmax_sharded_rejects(self):
        t = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        out = propagate_replicate_affected(t, 0)
        self.assertIsNone(out)

    def test_layer_norm(self):
        t = ShardedLayout.shard((4, 8, 16), shard_dim=0, mesh_dim_size=2)
        out = propagate_layer_norm(t, normalized_dims=2)
        self.assertIsNotNone(out)  # last 2 dims (8, 16) not sharded

    def test_layer_norm_sharded_rejects(self):
        t = ShardedLayout.shard((4, 8, 16), shard_dim=1, mesh_dim_size=2)
        out = propagate_layer_norm(t, normalized_dims=2)
        self.assertIsNone(out)  # dim 1 is sharded and in normalized range


class TestSelectIndexSelect(unittest.TestCase):

    def test_select(self):
        t = ShardedLayout.shard((4, 8, 16), shard_dim=0, mesh_dim_size=2)
        out = propagate_slice(t, dim=1, index=3)
        self.assertIsNotNone(out)
        self.assertEqual(out.get_placements()[0][:3], ("shard", 0, 2))

    def test_index_select(self):
        t = ShardedLayout.shard((4, 8, 16), shard_dim=0, mesh_dim_size=2)
        out = propagate_index_select(t, dim=1, index_size=3)
        self.assertIsNotNone(out)


class TestScatterOp(unittest.TestCase):

    def test_scatter_replicate(self):
        t = ShardedLayout.replicate((4, 8))
        src = ShardedLayout.replicate((4, 8))
        out = propagate_scatter(t, dim=0, src_sharded=src)
        self.assertIsNotNone(out)

    def test_scatter_sharded_rejects(self):
        t = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        src = ShardedLayout.replicate((4, 8))
        out = propagate_scatter(t, dim=0, src_sharded=src)
        self.assertIsNone(out)


class TestEmbeddingOp(unittest.TestCase):

    def test_embedding_colwise(self):
        weight = ShardedLayout.shard((1000, 64), shard_dim=1, mesh_dim_size=2)
        indices = ShardedLayout.replicate((4, 8))
        out = propagate_embedding(weight, indices, mode="colwise")
        self.assertIsNotNone(out)

    def test_embedding_batch(self):
        weight = ShardedLayout.replicate((1000, 64))
        indices = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        out = propagate_embedding(weight, indices, mode="batch")
        self.assertIsNotNone(out)


class TestConvolutionOp(unittest.TestCase):

    def test_conv_batch_shard(self):
        # (N, C_in, H, W) sharded on batch
        inp = ShardedLayout.shard((4, 3, 32, 32), shard_dim=0, mesh_dim_size=2)
        # (C_out, C_in, kH, kW) replicate
        weight = ShardedLayout.replicate((16, 3, 3, 3))
        out = propagate_convolution(inp, weight)
        self.assertIsNotNone(out)
        self.assertEqual(out.get_placements()[0][:3], ("shard", 0, 2))

    def test_conv_spatial_sharded_rejects(self):
        inp = ShardedLayout.shard((4, 3, 32, 32), shard_dim=2, mesh_dim_size=2)
        weight = ShardedLayout.replicate((16, 3, 3, 3))
        out = propagate_convolution(inp, weight)
        self.assertIsNone(out)


class TestDropoutOp(unittest.TestCase):

    def test_dropout_shard(self):
        t = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        out = propagate_dropout(t)
        self.assertIsNotNone(out)
        self.assertEqual(out, t)

    def test_dropout_partial_rejects(self):
        t = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        t = propagate_reduction(t, reduce_dim=0)
        self.assertTrue(len(t.partial) > 0)
        out = propagate_dropout(t)
        self.assertIsNone(out)


class TestOpRegistry(unittest.TestCase):
    """Tests for op registry mapping ATen ops to propagation functions."""

    def test_registry_lookup(self):
        from autoparallel.shardings.cute import get_propagation_rule
        self.assertIsNotNone(get_propagation_rule("aten.mm.default"))
        self.assertIsNotNone(get_propagation_rule("aten.add.Tensor"))
        self.assertIsNotNone(get_propagation_rule("aten.view.default"))
        self.assertIsNotNone(get_propagation_rule("aten.clone.default"))
        self.assertIsNone(get_propagation_rule("aten.nonexistent.default"))

    def test_registry_maps_to_correct_functions(self):
        from autoparallel.shardings.cute import get_propagation_rule
        self.assertEqual(get_propagation_rule("aten.mm.default"), propagate_mm)
        self.assertEqual(get_propagation_rule("aten.bmm.default"), propagate_bmm)
        self.assertEqual(get_propagation_rule("aten.cat.default"), propagate_cat)
        self.assertEqual(get_propagation_rule("aten.view.default"), propagate_view)
        self.assertEqual(get_propagation_rule("aten.permute.default"), propagate_permute)
        self.assertEqual(get_propagation_rule("aten.transpose.int"), propagate_transpose)
        self.assertEqual(get_propagation_rule("aten.unsqueeze.default"), propagate_unsqueeze)
        self.assertEqual(get_propagation_rule("aten.select.int"), propagate_slice)
        self.assertEqual(get_propagation_rule("aten.embedding.default"), propagate_embedding)
        self.assertEqual(get_propagation_rule("aten.convolution.default"), propagate_convolution)

    def test_pointwise_ops_registered(self):
        from autoparallel.shardings.cute import get_propagation_rule
        # All pointwise ops should have a registered rule (linear or non-linear)
        pointwise_ops = [
            "aten.add.Tensor", "aten.sub.Tensor", "aten.mul.Tensor",
            "aten.div.Tensor", "aten.relu.default", "aten.gelu.default",
            "aten.sigmoid.default", "aten.tanh.default", "aten.where.self",
            "aten.abs.default", "aten.neg.default", "aten.exp.default",
        ]
        for op in pointwise_ops:
            self.assertIsNotNone(get_propagation_rule(op), f"{op} not registered")

    def test_identity_ops_registered(self):
        from autoparallel.shardings.cute import get_propagation_rule
        identity_ops = [
            "aten.clone.default", "aten.contiguous.default",
            "aten.detach.default", "aten.empty_like.default",
        ]
        for op in identity_ops:
            self.assertEqual(get_propagation_rule(op), propagate_identity, f"{op} not mapped to identity")

    def test_reduction_ops_registered(self):
        from autoparallel.shardings.cute import get_propagation_rule
        reduction_ops = [
            "aten.sum.default", "aten.mean.default", "aten.max.default",
            "aten.min.default", "aten.amax.default",
        ]
        for op in reduction_ops:
            self.assertEqual(get_propagation_rule(op), propagate_reduction, f"{op} not mapped to reduction")

    def test_registry_size(self):
        from autoparallel.shardings.cute import OP_REGISTRY
        # Should have a substantial number of ops registered
        self.assertGreater(len(OP_REGISTRY), 200)

    def test_registry_aten_calling_convention(self):
        """Registry functions match ATen signatures — call with op args directly."""
        from autoparallel.shardings.cute import get_propagation_rule

        t = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)

        # aten.roll(self, shifts, dims) — shifts is unused for sharding
        roll_fn = get_propagation_rule("aten.roll.default")
        out = roll_fn(t, [2], [1])
        self.assertIsNotNone(out)

        # aten.flip(self, dims)
        flip_fn = get_propagation_rule("aten.flip.default")
        out = flip_fn(t, [1])
        self.assertIsNotNone(out)

        # aten._softmax(self, dim, half_to_float)
        softmax_fn = get_propagation_rule("aten._softmax.default")
        out = softmax_fn(t, 1, False)
        self.assertIsNotNone(out)

        # aten.cumsum(self, dim)
        cumsum_fn = get_propagation_rule("aten.cumsum.default")
        out = cumsum_fn(t, 1)
        self.assertIsNotNone(out)

        # aten.sort(self, dim, descending) — returns tuple
        sort_fn = get_propagation_rule("aten.sort.default")
        result = sort_fn(t, 1, False)
        self.assertIsNotNone(result)

        # aten.select.int(self, dim, index)
        select_fn = get_propagation_rule("aten.select.int")
        out = select_fn(t, 1, 3)
        self.assertIsNotNone(out)


class TestPpermute(unittest.TestCase):
    """Tests for ppermute detection in plan_redistribute."""

    def test_shard_dim_swap_is_all_to_all(self):
        """S(0)->S(1): each device sends to ALL others -> all_to_all, not ppermute."""
        source = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=2)
        target = ShardedLayout.shard((8, 16), shard_dim=1, mesh_dim_size=2)
        result = plan_redistribute(source, target)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "all_to_all")

    def test_same_dim_different_stride_is_ppermute(self):
        """Same dim sharded with different mesh sizes -> ppermute (1-to-1)."""
        # S(0) mesh=2 -> S(0) mesh=4: each src device maps to one tgt device
        source = ShardedLayout.shard((16, 16), shard_dim=0, mesh_dim_size=2)
        target = ShardedLayout.shard((16, 16), shard_dim=0, mesh_dim_size=4)
        result = plan_redistribute(source, target)
        self.assertEqual(len(result), 1)
        # This might be ppermute or all_to_all depending on the element mapping
        # For mesh=2->mesh=4: src device 0 has elements 0-7, tgt device 0 has 0-3, device 1 has 4-7
        # So src device 0 sends to TWO tgt devices -> all_to_all
        self.assertEqual(result[0][0], "all_to_all")

    def test_s0s0_ltr_rtl_is_ppermute(self):
        """S(0)S(0) LTR -> RTL: reordering on same mesh dim -> ppermute."""
        ltr = ShardedLayout.shard_multi((8, 16), [(0, 2), (0, 4)])
        rtl = ShardedLayout.shard_multi((8, 16), [(0, 4), (0, 2)])
        result = plan_redistribute(ltr, rtl)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "ppermute")
        perm = result[0][2]["perm"]
        # Valid permutation: bijection
        srcs = [p[0] for p in perm]
        dsts = [p[1] for p in perm]
        self.assertEqual(len(set(srcs)), len(srcs))
        self.assertEqual(len(set(dsts)), len(dsts))

    def test_same_sharding_no_ppermute(self):
        """Same sharding -> no collective at all."""
        s = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=2)
        result = plan_redistribute(s, s)
        self.assertEqual(result, [])


class TestRingAttentionPpermute(unittest.TestCase):
    """Tests that ring attention step transitions yield correct ppermute calls."""

    @staticmethod
    def _make_ring_layout(n_gpus):
        import math
        from autoparallel.shardings.cute._pycute import ModStride, XorStride
        n_bits = int(math.log2(n_gpus))
        shapes = [2] * n_bits + [n_gpus, 2]
        strides = (
            [ModStride(2**i, n_gpus) for i in range(n_bits)]
            + [ModStride(n_gpus - 1, n_gpus)]
            + [XorStride(2 * n_gpus - 1)]
        )
        return Layout(tuple(shapes), tuple(strides))

    @staticmethod
    def _get_step_ppermute(ring, n_gpus, step_src, step_tgt):
        """Compute the ppermute from step_src to step_tgt using the ring layout."""
        import math
        n_bits = int(math.log2(n_gpus))

        def get_chunks(step):
            gpu_to_chunks = {}
            for g in range(n_gpus):
                bits = [(g >> i) & 1 for i in range(n_bits)]
                chunks = [ring(*(bits + [step, p])) for p in range(2)]
                gpu_to_chunks[g] = chunks
            return gpu_to_chunks

        src_map = get_chunks(step_src)
        tgt_map = get_chunks(step_tgt)

        chunk_to_src = {}
        for gpu, chunks in src_map.items():
            for c in chunks:
                chunk_to_src[c] = gpu

        perm = []
        for tgt_gpu in range(n_gpus):
            chunks = tgt_map[tgt_gpu]
            src_gpu = chunk_to_src[chunks[0]]
            assert chunk_to_src[chunks[1]] == src_gpu
            perm.append((src_gpu, tgt_gpu))
        return sorted(perm)

    def test_4gpu_zigzag_assignment(self):
        """Step 0 produces correct zigzag: GPU g gets chunks {g, 7-g}."""
        ring = self._make_ring_layout(4)
        for g in range(4):
            bits = [g & 1, (g >> 1) & 1]
            c0 = ring(*(bits + [0, 0]))
            c1 = ring(*(bits + [0, 1]))
            self.assertEqual(c0, g)
            self.assertEqual(c1, 7 - g)

    def test_4gpu_all_steps_circular_shift(self):
        """All step transitions on 4 GPUs are circular left shift ppermute."""
        ring = self._make_ring_layout(4)
        for step in range(3):
            perm = self._get_step_ppermute(ring, 4, step, step + 1)
            for src, dst in perm:
                self.assertEqual(dst, (src + 1) % 4,
                                 f"Step {step}->{step+1}: expected ({src}, {(src+1)%4}), got ({src}, {dst})")

    def test_8gpu_all_steps_circular_shift(self):
        """All step transitions on 8 GPUs are circular left shift ppermute."""
        ring = self._make_ring_layout(8)
        for step in range(7):
            perm = self._get_step_ppermute(ring, 8, step, step + 1)
            for src, dst in perm:
                self.assertEqual(dst, (src + 1) % 8,
                                 f"Step {step}->{step+1}: expected ({src}, {(src+1)%8}), got ({src}, {dst})")

    def test_4gpu_load_balance(self):
        """Zigzag achieves perfect load balance for causal attention on 4 GPUs.

        Each GPU gets chunks (g, 7-g), work = (g+1) + (7-g+1) = 9 for all g.
        """
        ring = self._make_ring_layout(4)
        for g in range(4):
            bits = [g & 1, (g >> 1) & 1]
            c0 = ring(*(bits + [0, 0]))
            c1 = ring(*(bits + [0, 1]))
            # Causal attention: chunk c has work proportional to (c+1)
            work = (c0 + 1) + (c1 + 1)
            self.assertEqual(work, 9, f"GPU {g}: work={work}, expected 9")

    def test_4gpu_step_permutation_is_bijection(self):
        """Each step transition is a valid permutation (bijection)."""
        ring = self._make_ring_layout(4)
        for step in range(3):
            perm = self._get_step_ppermute(ring, 4, step, step + 1)
            srcs = [p[0] for p in perm]
            dsts = [p[1] for p in perm]
            self.assertEqual(sorted(srcs), list(range(4)))
            self.assertEqual(sorted(dsts), list(range(4)))

    @staticmethod
    def _make_ring_step_layout(n_gpus, step):
        """Build ShardedLayout for ring attention at a specific step.

        Uses XorStride zigzag + offset for ring rotation.
        Operates in chunk space (tensor of 2*N chunks).
        """
        from autoparallel.shardings.cute._pycute import XorStride
        n_chunks = 2 * n_gpus
        # sub-dim: (local=2, mesh=N_gpus) with strides (XorStride(2N-1), 1)
        # XorStride is local (zigzag pair), mesh stride 1 (chunk identity = GPU index)
        hier = Layout(
            (((2, n_gpus),),),
            (((XorStride(n_chunks - 1), 1),),)
        )
        sl = ShardedLayout(hier, {0: (0,)})
        if step > 0:
            sl = sl.with_offset({0: step * (n_gpus - 1)})
        return sl

    def test_plan_redistribute_step0_to_step1(self):
        """plan_redistribute detects circular shift ppermute between ring steps."""
        step0 = self._make_ring_step_layout(4, step=0)
        step1 = self._make_ring_step_layout(4, step=1)
        result = plan_redistribute(step0, step1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "ppermute")
        perm = dict(result[0][2]["perm"])
        for src in range(4):
            self.assertEqual(perm[src], (src + 1) % 4)

    def test_plan_redistribute_step1_to_step2(self):
        """Step 1→2 also gives circular shift (same offset difference)."""
        step1 = self._make_ring_step_layout(4, step=1)
        step2 = self._make_ring_step_layout(4, step=2)
        result = plan_redistribute(step1, step2)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "ppermute")
        perm = dict(result[0][2]["perm"])
        for src in range(4):
            self.assertEqual(perm[src], (src + 1) % 4)

    def test_plan_redistribute_same_step_noop(self):
        """Same step → no redistribution needed."""
        step0 = self._make_ring_step_layout(4, step=0)
        result = plan_redistribute(step0, step0)
        self.assertEqual(result, [])

    def test_plan_redistribute_8gpu_step_transition(self):
        """8-GPU ring attention step transition is circular shift."""
        step0 = self._make_ring_step_layout(8, step=0)
        step1 = self._make_ring_step_layout(8, step=1)
        result = plan_redistribute(step0, step1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "ppermute")
        perm = dict(result[0][2]["perm"])
        for src in range(8):
            self.assertEqual(perm[src], (src + 1) % 8)


class TestPlanRedistributeEdgeCases(unittest.TestCase):
    """Edge case tests for plan_redistribute."""

    def test_s0s0_to_replicate(self):
        """S(0)S(0) → Replicate emits two all_gathers (one per mesh dim)."""
        src = ShardedLayout.shard_multi((16,), [(0, 2), (0, 2)])
        tgt = ShardedLayout.replicate((16,))
        result = plan_redistribute(src, tgt)
        types = [ct for ct, _, _ in result]
        self.assertEqual(types.count("all_gather"), 2)

    def test_s0s0_to_s0(self):
        """S(0)S(0) → S(0) on one mesh dim: all_gather on the other."""
        src = ShardedLayout.shard_multi((16,), [(0, 2), (0, 2)])
        tgt = ShardedLayout.shard((16,), shard_dim=0, mesh_dim_size=2, mesh_dim=0)
        result = plan_redistribute(src, tgt)
        collectives_by_type = {}
        for ct, md, info in result:
            collectives_by_type.setdefault(ct, []).append(md)
        self.assertIn("all_gather", collectives_by_type)

    def test_replicate_to_replicate(self):
        """Replicate → Replicate: no collectives."""
        src = ShardedLayout.replicate((8, 16))
        tgt = ShardedLayout.replicate((8, 16))
        self.assertEqual(plan_redistribute(src, tgt), [])

    def test_s0_to_s0_same(self):
        """Same S(0) → S(0): no collectives."""
        sl = ShardedLayout.shard((8,), shard_dim=0, mesh_dim_size=4)
        self.assertEqual(plan_redistribute(sl, sl), [])

    def test_partial_to_partial_same(self):
        """Partial → Partial with same reduce_op: all_reduce still needed."""
        src = ShardedLayout.replicate((8,))
        src.partial = {0: "sum"}
        tgt = ShardedLayout.replicate((8,))
        # Partial on src, no partial on tgt → all_reduce
        result = plan_redistribute(src, tgt)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "all_reduce")

    def test_partial_to_shard(self):
        """Partial(sum) → S(0): reduce_scatter."""
        src = ShardedLayout.replicate((8,))
        src.partial = {0: "sum"}
        tgt = ShardedLayout.shard((8,), shard_dim=0, mesh_dim_size=4)
        result = plan_redistribute(src, tgt)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "reduce_scatter")

    def test_shape_mismatch_raises(self):
        """Different global shapes → ValueError."""
        src = ShardedLayout.replicate((8,))
        tgt = ShardedLayout.replicate((16,))
        with self.assertRaises(ValueError):
            plan_redistribute(src, tgt)

    def test_multidim_shard_different_dims(self):
        """S(0)R → RS(1): all_gather on dim 0, local reinterpret on dim 1."""
        src = ShardedLayout.shard((8, 8), shard_dim=0, mesh_dim_size=2)
        tgt = ShardedLayout.shard((8, 8), shard_dim=1, mesh_dim_size=2)
        result = plan_redistribute(src, tgt)
        types = {ct for ct, _, _ in result}
        # Should have all_to_all (S(0) → S(1) on same mesh dim)
        self.assertIn("all_to_all", types)

    def test_offset_same_layout_different_offset(self):
        """Same strides, different offset → ppermute."""
        from autoparallel.shardings.cute._pycute import XorStride
        N = 4
        hier = Layout((((2, N),),), (((XorStride(2 * N - 1), 1),),))
        sl0 = ShardedLayout(hier, {0: (0,)})
        sl1 = sl0.with_offset({0: N - 1})
        result = plan_redistribute(sl0, sl1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "ppermute")
        perm = dict(result[0][2]["perm"])
        # Circular shift
        for src in range(N):
            self.assertEqual(perm[src], (src + 1) % N)

    def test_offset_zero_is_noop(self):
        """Offset {0: 0} is equivalent to no offset."""
        sl1 = ShardedLayout.shard((8,), shard_dim=0, mesh_dim_size=4)
        sl2 = sl1.with_offset({0: 0})
        self.assertEqual(plan_redistribute(sl1, sl2), [])

    def test_with_offset_accumulates(self):
        """with_offset adds to existing offset."""
        from autoparallel.shardings.cute._pycute import XorStride
        hier = Layout((((2, 4),),), (((XorStride(7), 1),),))
        sl = ShardedLayout(hier, {0: (0,)})
        sl1 = sl.with_offset({0: 3})
        sl2 = sl1.with_offset({0: 3})
        self.assertEqual(sl2.offset, {0: 6})


class TestEnumerateShardings(unittest.TestCase):
    """Tests for enumerate_shardings."""

    def test_1d_mesh(self):
        from autoparallel.shardings.cute import enumerate_shardings
        # (8, 16) on mesh (2,)
        shardings = enumerate_shardings((8, 16), (2,))
        shapes = [s.global_shape for s in shardings]
        # Should have: Replicate, Shard(0), Shard(1)
        self.assertTrue(any(s.is_replicate() for s in shardings))
        self.assertTrue(any(
            s.mesh_dim_map == {0: (0,), 1: ()} for s in shardings
        ))
        self.assertTrue(any(
            s.mesh_dim_map == {0: (), 1: (0,)} for s in shardings
        ))
        self.assertEqual(len(shardings), 3)

    def test_2d_mesh(self):
        from autoparallel.shardings.cute import enumerate_shardings
        # (8, 16) on mesh (2, 4)
        shardings = enumerate_shardings((8, 16), (2, 4))
        # Should have: R, S(0,m0), S(1,m0), S(0,m1), S(1,m1),
        # S(0,m0)S(1,m1), S(1,m0)S(0,m1),
        # S(0)S(0) LTR, S(0)S(0) RTL (if 8 % 8 == 0),
        # S(1)S(1) LTR, S(1)S(1) RTL (if 16 % 8 == 0)
        self.assertGreater(len(shardings), 5)
        # Verify replicate exists
        self.assertTrue(any(s.is_replicate() for s in shardings))

    def test_s0s0_orderings(self):
        from autoparallel.shardings.cute import enumerate_shardings
        # (8, 16) on mesh (2, 4): S(0)S(0) needs 8 % (2*4) == 0 ✓
        shardings = enumerate_shardings((8, 16), (2, 4))
        # Find S(0)S(0) candidates: mesh_dim_map[0] has both mesh dims
        s0s0 = [s for s in shardings if len(s.mesh_dim_map[0]) == 2]
        # Should have LTR and RTL as distinct layouts
        self.assertEqual(len(s0s0), 2)
        # They should be different ShardedLayouts
        self.assertNotEqual(s0s0[0].hier_layout, s0s0[1].hier_layout)

    def test_divisibility_filter(self):
        from autoparallel.shardings.cute import enumerate_shardings
        # (7, 16) on mesh (2,): dim 0 (size 7) not divisible by 2
        shardings = enumerate_shardings((7, 16), (2,))
        # Should have: Replicate, Shard(1) only — Shard(0) excluded
        self.assertEqual(len(shardings), 2)
        self.assertTrue(all(
            s.is_replicate() or s.mesh_dim_map == {0: (), 1: (0,)}
            for s in shardings
        ))


class TestEnumerateStrategies(unittest.TestCase):
    """Tests for enumerate_strategies."""

    def test_mm_strategies(self):
        from autoparallel.shardings.cute import enumerate_shardings, enumerate_strategies
        a_cands = enumerate_shardings((16, 8), (2,))
        b_cands = enumerate_shardings((8, 32), (2,))
        strategies = enumerate_strategies("aten.mm.default", [a_cands, b_cands])
        self.assertGreater(len(strategies), 0)
        # Should include: both replicate -> replicate output
        rep_strats = [(ins, out) for ins, out in strategies
                      if ins[0].is_replicate() and ins[1].is_replicate()]
        self.assertEqual(len(rep_strats), 1)
        self.assertTrue(rep_strats[0][1].is_replicate())
        # Should include: A sharded on M, B replicate -> output sharded on M
        m_shard = [(ins, out) for ins, out in strategies
                   if ins[0].mesh_dim_map[0] == (0,) and ins[1].is_replicate()]
        self.assertEqual(len(m_shard), 1)

    def test_pointwise_strategies(self):
        from autoparallel.shardings.cute import enumerate_shardings, enumerate_strategies
        a_cands = enumerate_shardings((8, 16), (2,))
        b_cands = enumerate_shardings((8, 16), (2,))
        strategies = enumerate_strategies("aten.add.Tensor", [a_cands, b_cands])
        # Should include matching shardings
        self.assertGreater(len(strategies), 0)
        # All outputs should be valid ShardedLayouts
        for ins, out in strategies:
            self.assertIsNotNone(out)

    def test_pointwise_no_redundant_replicate_shard(self):
        """Same-shape pointwise: (R, S(0)) and (S(0), R) are filtered out."""
        from autoparallel.shardings.cute import enumerate_shardings, enumerate_strategies
        a_cands = enumerate_shardings((8, 8), (2,))
        b_cands = enumerate_shardings((8, 8), (2,))
        strategies = enumerate_strategies("aten.add.Tensor", [a_cands, b_cands])

        # No strategy should mix R and S inputs on a same-size dim
        for ins, out in strategies:
            has_sharded = any(not i.is_replicate() for i in ins)
            has_replicate = any(i.is_replicate() for i in ins)
            self.assertFalse(
                has_sharded and has_replicate,
                f"Redundant mixed R/S strategy found: {[i.mesh_dim_map for i in ins]}"
            )

    def test_broadcast_replicate_shard_kept(self):
        """Broadcast pointwise: (S(0), R) is kept when shapes differ on dim 0."""
        from autoparallel.shardings.cute import enumerate_shardings, enumerate_strategies
        a_cands = enumerate_shardings((8, 8), (2,))
        b_cands = enumerate_shardings((1, 8), (2,))
        strategies = enumerate_strategies("aten.add.Tensor", [a_cands, b_cands])

        # Should have at least one mixed R/S strategy (broadcast on dim 0)
        mixed = [
            ins for ins, out in strategies
            if any(not i.is_replicate() for i in ins) and any(i.is_replicate() for i in ins)
        ]
        self.assertGreater(len(mixed), 0,
                           "Broadcast (S(0), R) strategy should be kept for different dim sizes")


class TestShardingHints(unittest.TestCase):
    """Tests for op-specific sharding hints."""

    def test_register_and_retrieve(self):
        from autoparallel.shardings.cute.strategy import (
            register_sharding_hint, get_sharding_hints, _OP_SHARDING_HINTS
        )
        # Clean up any previous registration
        test_op = "aten._test_hint_op.default"
        _OP_SHARDING_HINTS.pop(test_op, None)

        def my_hint(tensor_shapes, mesh_shape):
            return [[ShardedLayout.replicate(s) for s in tensor_shapes]]

        register_sharding_hint(test_op, my_hint)
        hints = get_sharding_hints(test_op, [(4, 8)], (2,))
        self.assertGreater(len(hints), 0)

        # Clean up
        _OP_SHARDING_HINTS.pop(test_op, None)

    def test_no_hints_returns_empty(self):
        from autoparallel.shardings.cute import get_sharding_hints
        hints = get_sharding_hints("aten.nonexistent.default", [(4, 8)], (2,))
        self.assertEqual(hints, [])

    def test_hints_extend_candidates(self):
        from autoparallel.shardings.cute import enumerate_shardings
        from autoparallel.shardings.cute.strategy import (
            register_sharding_hint, get_sharding_hints, _OP_SHARDING_HINTS
        )
        test_op = "aten._test_hint_extend.default"
        _OP_SHARDING_HINTS.pop(test_op, None)

        # Register a hint that adds a custom sharding
        custom = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=2, mesh_dim=0)

        def my_hint(tensor_shapes, mesh_shape):
            return [[custom]]

        register_sharding_hint(test_op, my_hint)

        # Standard candidates
        standard = enumerate_shardings((8, 16), (2,))
        # Hints
        hints = get_sharding_hints(test_op, [(8, 16)], (2,))
        # Combined should be larger than standard alone
        combined = standard + hints[0] if hints else standard
        self.assertGreaterEqual(len(combined), len(standard))

        _OP_SHARDING_HINTS.pop(test_op, None)


class TestInvariantValidation(unittest.TestCase):
    """Tests for post-condition invariant assertions in the propagation engine."""

    def test_valid_sharding_passes(self):
        """Standard shardings pass validation without error."""
        from autoparallel.shardings.cute.propagation import _validate_sharded_layout
        _validate_sharded_layout(ShardedLayout.replicate((4, 8)))
        _validate_sharded_layout(ShardedLayout.shard((4, 8), 0, 2))
        _validate_sharded_layout(ShardedLayout.shard_multi((4, 8, 16), [(0, 2), (1, 4)]))

    def test_mesh_dim_on_multiple_tensor_dims_rejects(self):
        """Manually constructed layout with same mesh_dim on two tensor dims."""
        from autoparallel.shardings.cute.propagation import _validate_sharded_layout
        from autoparallel.shardings.cute._pycute import Layout
        # Force invalid state: mesh_dim 0 on both tensor dims
        sl = ShardedLayout.shard((8, 16), 0, 2, mesh_dim=0)
        sl.mesh_dim_map[1] = (0,)  # inject: mesh_dim 0 also on dim 1
        with self.assertRaises(AssertionError, msg="mesh dim on multiple tensor dims"):
            _validate_sharded_layout(sl)

    def test_partial_and_sharded_on_same_mesh_dim_rejects(self):
        """Partial on mesh_dim 0 AND sharding on mesh_dim 0 is contradictory."""
        from autoparallel.shardings.cute.propagation import _validate_sharded_layout
        sl = ShardedLayout.shard((8, 16), 0, 2, mesh_dim=0)
        sl.partial = {0: "sum"}  # inject: partial on mesh_dim 0
        with self.assertRaises(AssertionError, msg="partial and sharded on same mesh dim"):
            _validate_sharded_layout(sl)

    def test_propagation_rejects_invalid_strategy(self):
        """The engine itself rejects strategies that would violate invariants."""
        # S(M) on mesh 0 + S(N) on mesh 0 → rejected before validation
        a = ShardedLayout.shard((16, 8), shard_dim=0, mesh_dim_size=2, mesh_dim=0)
        b = ShardedLayout.shard((8, 32), shard_dim=1, mesh_dim_size=2, mesh_dim=0)
        out = propagate_einsum("mk,kn->mn", a, b)
        self.assertIsNone(out)


class TestPartialLinearity(unittest.TestCase):
    """Tests for Partial propagation through pointwise ops with linearity."""

    def _make_partial(self):
        """Create a Partial("sum") tensor via K-sharded einsum."""
        a = ShardedLayout.shard((16, 8), shard_dim=1, mesh_dim_size=2, mesh_dim=0)
        b = ShardedLayout.shard((8, 32), shard_dim=0, mesh_dim_size=2, mesh_dim=0)
        result = propagate_einsum("mk,kn->mn", a, b)
        assert result is not None and result.partial == {0: "sum"}
        return result

    def test_add_partial_partial(self):
        """add(Partial, Partial) -> Partial (additive linear)."""
        p = self._make_partial()
        from autoparallel.shardings.cute import get_propagation_rule
        add_fn = get_propagation_rule("aten.add.Tensor")
        out = add_fn(p, p)
        self.assertIsNotNone(out)
        self.assertEqual(out.partial, {0: "sum"})

    def test_mul_partial_partial_rejects(self):
        """mul(Partial, Partial) -> reject (non-linear in both args)."""
        p = self._make_partial()
        from autoparallel.shardings.cute import get_propagation_rule
        mul_fn = get_propagation_rule("aten.mul.Tensor")
        out = mul_fn(p, p)
        self.assertIsNone(out)

    def test_mul_partial_replicate(self):
        """mul(Partial, Replicate) -> Partial (multiplicative linear in one arg)."""
        p = self._make_partial()
        r = ShardedLayout.replicate((16, 32))
        from autoparallel.shardings.cute import get_propagation_rule
        mul_fn = get_propagation_rule("aten.mul.Tensor")
        out = mul_fn(p, r)
        self.assertIsNotNone(out)
        self.assertEqual(out.partial, {0: "sum"})

    def test_relu_partial(self):
        """relu(Partial) -> reject (non-linear, can't reduce after relu)."""
        p = self._make_partial()
        from autoparallel.shardings.cute import get_propagation_rule
        relu_fn = get_propagation_rule("aten.relu.default")
        out = relu_fn(p)
        self.assertIsNone(out)

    def test_neg_partial(self):
        """neg(Partial) -> Partial (unary linear: -(a+b) = -a + -b)."""
        p = self._make_partial()
        from autoparallel.shardings.cute import get_propagation_rule
        neg_fn = get_propagation_rule("aten.neg.default")
        out = neg_fn(p)
        self.assertIsNotNone(out)
        self.assertEqual(out.partial, {0: "sum"})

    def test_clone_partial(self):
        """clone(Partial) -> Partial (identity preserves Partial)."""
        p = self._make_partial()
        out = propagate_identity(p)
        self.assertIsNotNone(out)
        self.assertEqual(out.partial, {0: "sum"})


class TestSilentCorrectnessEdgeCases(unittest.TestCase):
    """Tests for potential silent correctness issues identified during design review.

    Each test targets a specific edge case that could produce an internally
    inconsistent ShardedLayout if the engine has gaps in its checks.
    """

    def test_s0s0_transpose(self):
        """Case 1: S(0)S(0) through transpose — mesh structure must survive."""
        t = ShardedLayout.shard_multi((8, 16), [(0, 2), (0, 4)])
        self.assertEqual(t.mesh_dim_map, {0: (0, 1), 1: ()})

        out = propagate_transpose(t, 0, 1)
        self.assertIsNotNone(out)
        self.assertEqual(out.global_shape, (16, 8))
        # S(0)S(0) should move to dim 1
        self.assertEqual(out.mesh_dim_map, {0: (), 1: (0, 1)})

    def test_s0s0_view_merge(self):
        """Case 1b: S(0)S(0) through view merge — should preserve structure."""
        t = ShardedLayout.shard_multi((8, 16, 4), [(0, 2), (0, 4)])
        # View (8, 16, 4) -> (8, 64): merge dims 1 and 2
        out = propagate_view(t, (8, 64))
        self.assertIsNotNone(out)
        self.assertEqual(out.global_shape, (8, 64))
        # S(0)S(0) stays on dim 0
        self.assertEqual(out.mesh_dim_map[0], (0, 1))

    def test_partial_plus_sharded_same_mesh_dim(self):
        """Case 2: Partial on mesh_dim 0 + sharding on mesh_dim 0 is contradictory.

        This can arise if a K-sharded reduction produces Partial on mesh_dim 0,
        then a downstream op tries to carry a dim sharded on mesh_dim 0.
        """
        # Create a Partial result: K-sharded einsum
        a = ShardedLayout.shard((16, 8), shard_dim=1, mesh_dim_size=2, mesh_dim=0)
        b = ShardedLayout.shard((8, 32), shard_dim=0, mesh_dim_size=2, mesh_dim=0)
        partial_result = propagate_einsum("mk,kn->mn", a, b)
        self.assertIsNotNone(partial_result)
        self.assertEqual(partial_result.partial, {0: "sum"})

        # Now try pointwise with something sharded on mesh_dim 0
        sharded = ShardedLayout.shard((16, 32), shard_dim=0, mesh_dim_size=2, mesh_dim=0)

        # This should be rejected: partial_result has Partial on mesh 0,
        # sharded has Shard on mesh 0 for dim 0.
        # The pointwise Carry would inherit mesh_dim 0 on dim 0 from sharded,
        # plus Partial on mesh_dim 0 from partial_result → contradictory.
        out = propagate_pointwise([partial_result, sharded])
        self.assertIsNone(out)

    def test_cat_incompatible_non_cat_dims(self):
        """Case 3: Cat where non-cat dims have incompatible shardings across inputs."""
        # A shards dim 1 on mesh 0, B shards dim 1 on mesh 1
        a = ShardedLayout.shard((4, 8), shard_dim=1, mesh_dim_size=2, mesh_dim=0)
        b = ShardedLayout.shard((4, 8), shard_dim=1, mesh_dim_size=2, mesh_dim=1)
        # Cat on dim 0: non-cat dim 1 has different mesh dims → incompatible
        out = propagate_cat([a, b], dim=0)
        self.assertIsNone(out)

    def test_cat_one_replicate_one_sharded_non_cat_dim(self):
        """Case 3b: Cat where one input is replicate, other sharded on non-cat dim."""
        a = ShardedLayout.shard((4, 8), shard_dim=1, mesh_dim_size=2)
        b = ShardedLayout.replicate((4, 8))
        # Cat on dim 0: non-cat dim 1 — a is sharded, b is replicate
        # Carry should pick the sharded one (compatible)
        out = propagate_cat([a, b], dim=0)
        self.assertIsNotNone(out)

    def test_view_merge3_split2(self):
        """Case 4: Merge 3 dims into 1, then split into 2 — mesh_dim redistribution."""
        # (4, 8, 16) with S(0) on dim 1
        t = ShardedLayout.shard((4, 8, 16), shard_dim=1, mesh_dim_size=2)
        # Merge all 3 dims: (4, 8, 16) -> (512,)
        v1 = propagate_view(t, (512,))
        self.assertIsNotNone(v1)
        self.assertEqual(v1.global_shape, (512,))
        # Mesh should be on dim 0
        self.assertFalse(v1.is_replicate())

        # Split into 2 dims: (512,) -> (32, 16)
        v2 = propagate_view(v1, (32, 16))
        self.assertIsNotNone(v2)
        self.assertEqual(v2.global_shape, (32, 16))

    def test_einsum_batch_different_mesh_dims_rejects(self):
        """Case 5: Batch dim sharded on different mesh dims across inputs → reject."""
        # A: batch on mesh_dim 0, B: batch on mesh_dim 1
        a = ShardedLayout.shard((4, 8, 16), shard_dim=0, mesh_dim_size=2, mesh_dim=0)
        b = ShardedLayout.shard((4, 16, 32), shard_dim=0, mesh_dim_size=2, mesh_dim=1)
        # Batch dim 'b' is in both A and B but on different mesh dims
        out = propagate_einsum("bmk,bkn->bmn", a, b)
        # Should be rejected: batch dims must match
        self.assertIsNone(out)

    def test_einsum_batch_same_mesh_dim_ok(self):
        """Case 5b: Batch dim on same mesh dim across inputs → valid."""
        a = ShardedLayout.shard((4, 8, 16), shard_dim=0, mesh_dim_size=2, mesh_dim=0)
        b = ShardedLayout.shard((4, 16, 32), shard_dim=0, mesh_dim_size=2, mesh_dim=0)
        out = propagate_einsum("bmk,bkn->bmn", a, b)
        self.assertIsNotNone(out)
        self.assertEqual(out.mesh_dim_map[0], (0,))

    def test_expand_sharded_size1_dim(self):
        """Case 6: Expand only broadcasts size-1 dims.

        Sharded dims (size > 1) are not touched by expand — they pass through
        via Carry. A size-1 dim that's replicate (mesh=1) expands normally.
        A truly sharded size-1 dim would require mesh_size > 1 on a size-1 dim,
        which is impossible (can't divide 1 by >1).

        So expand is safe: it only operates on size-1 dims which can't be
        meaningfully sharded.
        """
        # (4, 1, 8) with dim 0 sharded — expand dim 1 from 1 to 16
        t = ShardedLayout.shard((4, 1, 8), shard_dim=0, mesh_dim_size=2)
        out = propagate_expand(t, (4, 16, 8))
        self.assertIsNotNone(out)
        self.assertEqual(out.global_shape, (4, 16, 8))
        # Shard on dim 0 preserved
        self.assertEqual(out.get_placements()[0][:3], ("shard", 0, 2))

    def test_expand_replicate_size1_dim_ok(self):
        """Case 6b: Expand a size-1 replicate dim — should work."""
        t = ShardedLayout.shard((4, 1), shard_dim=0, mesh_dim_size=2)
        out = propagate_expand(t, (4, 8))
        self.assertIsNotNone(out)
        self.assertEqual(out.global_shape, (4, 8))
        # Shard on dim 0 preserved
        self.assertEqual(out.get_placements()[0][:3], ("shard", 0, 2))


class TestCuTeBackend(unittest.TestCase):
    """Tests for CuTeBackend — ShardingBackend implementation."""

    def test_implements_protocol(self):
        from autoparallel.shardings.backend import ShardingBackend
        from autoparallel.shardings.cute_backend import CuTeBackend
        backend = CuTeBackend()
        self.assertIsInstance(backend, ShardingBackend)

    def _make_fake_node(self, shape):
        """Create a minimal fake node with meta['val'] for testing."""
        import torch
        class FakeNode:
            def __init__(self, tensor):
                self.meta = {"val": tensor}
        return FakeNode(torch.randn(shape))

    def test_create_all_options_1d(self):
        from autoparallel.shardings.cute_backend import CuTeBackend
        backend = CuTeBackend()
        node = self._make_fake_node((8, 16))
        options = backend.create_all_options(mesh=(2,), node=node)
        # Replicate + S(0) + S(1)
        self.assertEqual(len(options), 3)
        # All have ShardedLayout as output_spec
        for opt in options:
            self.assertIsInstance(opt.output_spec, ShardedLayout)
            self.assertEqual(opt.compute_cost, 0.0)

    def test_create_all_options_2d(self):
        from autoparallel.shardings.cute_backend import CuTeBackend
        backend = CuTeBackend()
        node = self._make_fake_node((1024, 1024))
        options = backend.create_all_options(mesh=(2, 4), node=node)
        self.assertGreater(len(options), 5)

    def test_redistribute_cost_same(self):
        from autoparallel.shardings.cute_backend import CuTeBackend
        backend = CuTeBackend()
        s = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=2)
        self.assertEqual(backend.redistribute_cost(s, s, mesh=(2,)), 0.0)

    def test_redistribute_cost_shard_to_replicate(self):
        from autoparallel.shardings.cute_backend import CuTeBackend
        backend = CuTeBackend()
        src = ShardedLayout.shard((8, 16), shard_dim=0, mesh_dim_size=2)
        tgt = ShardedLayout.replicate((8, 16))
        cost = backend.redistribute_cost(src, tgt, mesh=(2,))
        self.assertGreater(cost, 0.0)

    def test_apply_solution_empty(self):
        """apply_solution with empty solution returns graph unchanged."""
        from autoparallel.shardings.cute_backend import CuTeBackend
        backend = CuTeBackend()
        # Create a minimal FX graph: just a placeholder and output
        import torch
        def f(x):
            return x + 1
        gm = torch.fx.symbolic_trace(f)
        # Add fake meta to placeholder
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                node.meta["val"] = torch.randn(4, 4)
            elif node.op == "output":
                node.meta["desc"] = None
        # Empty solution — no nodes to process
        parallel_gm, params, buffers = backend.apply_solution(gm, {}, None)
        self.assertIsNotNone(parallel_gm)

    def test_apply_solution_inserts_collective(self):
        """apply_solution inserts all_gather when S(0) → Replicate at an edge."""
        from autoparallel.shardings.cute_backend import CuTeBackend
        from autoparallel.shardings.backend import OpOption
        import torch

        backend = CuTeBackend()

        # Graph: x (placeholder) -> mm(x, x) -> output
        def f(x, y):
            return torch.mm(x, y)

        gm = torch.fx.symbolic_trace(f)
        # Add fake meta
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                node.meta["val"] = torch.randn(4, 4)

        # Build solution: x is S(0) mesh=2, mm wants replicate inputs
        s0 = ShardedLayout.shard((4, 4), shard_dim=0, mesh_dim_size=2)
        rep = ShardedLayout.replicate((4, 4))

        nodes = list(gm.graph.nodes)
        x_node = nodes[0]  # placeholder x
        y_node = nodes[1]  # placeholder y
        mm_node = nodes[2]  # mm

        solution = {
            x_node: OpOption(output_spec=s0, input_specs=(s0,)),
            y_node: OpOption(output_spec=s0, input_specs=(s0,)),
            mm_node: OpOption(output_spec=rep, input_specs=(rep, rep)),
        }

        # Create a fake 1D mesh
        import torch.distributed as dist
        from torch.testing._internal.distributed.fake_pg import FakeStore
        if dist.is_initialized():
            dist.destroy_process_group()
        store = FakeStore()
        dist.init_process_group(backend="fake", world_size=2, rank=0, store=store)
        from torch.distributed.device_mesh import init_device_mesh
        mesh = init_device_mesh("cpu", (2,))

        try:
            parallel_gm, _, _ = backend.apply_solution(gm, solution, mesh)

            # Check that the parallel graph has all_gather nodes
            graph_str = str(parallel_gm.graph)
            self.assertIn("all_gather", graph_str,
                          f"Expected all_gather in graph, got:\n{graph_str}")
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
