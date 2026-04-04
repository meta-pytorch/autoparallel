"""
Tests for recipe-based sharding propagation (sharding_v2).
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
from autoparallel.shardings.cute.sharding_v2 import (
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
    propagate_cumsum,
    propagate_dot,
    propagate_dropout,
    propagate_einsum,
    propagate_embedding,
    propagate_expand,
    propagate_flatten,
    propagate_flip,
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
    propagate_roll,
    propagate_scatter,
    propagate_select,
    propagate_slice,
    propagate_softmax,
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

        LTR: specs [(0,2),(0,4)] -> mesh0(size=2, outermost, gs=64), mesh1(size=4, innermost, gs=16)
        RTL: specs [(0,4),(0,2)] -> mesh0(size=4, outermost, gs=32), mesh1(size=2, innermost, gs=16)

        mesh_dim 0 (outermost): gs 64 -> 32, different -> all_to_all
        mesh_dim 1 (innermost): gs 16 -> 16, same -> no_op
        """
        ltr = ShardedLayout.shard_multi((8, 16), [(0, 2), (0, 4)])
        rtl = ShardedLayout.shard_multi((8, 16), [(0, 4), (0, 2)])
        result = plan_redistribute(ltr, rtl)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "all_to_all")
        self.assertEqual(result[0][1], 0)  # mesh_dim 0 (outermost)

    def test_s0s0_per_mesh_dim_stride_values(self):
        """Verify per-mesh-dim GPU strides match mesh dim sizes, not tuple positions.

        For shard_multi((8, 16), [(0, 2), (0, 4)]):
          mesh_dim 0 (size 2, outermost) should have the larger stride
          mesh_dim 1 (size 4, innermost) should have the smaller stride

        The nested logical_divide produces (4, 2) in innermost-first order,
        but mesh_dim_map is outermost-first. The extraction must reverse.
        """
        from autoparallel.shardings.cute.sharding_v2 import _get_per_mesh_dim_gpu_stride

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
        out = propagate_flip(t, dims=[1])
        self.assertIsNotNone(out)
        self.assertEqual(out.get_placements()[0][:3], ("shard", 0, 2))

    def test_flip_sharded_rejects(self):
        t = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        out = propagate_flip(t, dims=[0])
        self.assertIsNone(out)

    def test_roll_non_sharded(self):
        t = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        out = propagate_roll(t, shifts=2, dims=[1])
        self.assertIsNotNone(out)

    def test_roll_sharded_rejects(self):
        t = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        out = propagate_roll(t, shifts=2, dims=[0])
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
        out = propagate_cumsum(t, dim=1)
        self.assertIsNotNone(out)

    def test_cumsum_sharded_rejects(self):
        t = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        out = propagate_cumsum(t, dim=0)
        self.assertIsNone(out)

    def test_softmax(self):
        t = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        out = propagate_softmax(t, dim=1)
        self.assertIsNotNone(out)

    def test_softmax_sharded_rejects(self):
        t = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=2)
        out = propagate_softmax(t, dim=0)
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
        out = propagate_select(t, dim=1, index=3)
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


if __name__ == "__main__":
    unittest.main()
