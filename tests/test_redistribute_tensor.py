"""
Tests for redistribute_tensor using LocalTensorMode.

Uses torch.distributed._local_tensor to simulate distributed collectives
on a single GPU. The funcol.* calls in redistribute_tensor get intercepted
by LocalTensorMode and execute local reimplementations from _local_tensor._c10d.
This tests the full pipeline end-to-end without NCCL.
"""

import unittest
from unittest.mock import patch
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.distributed._local_tensor import LocalTensorMode, LocalTensor
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel.shardings.cute import ShardedLayout, redistribute_tensor
from autoparallel.shardings.cute._pycute import Layout, XorStride


def _local_permute_tensor(tensor, src_dst, group, tag=""):
    """LocalTensor-aware permute_tensor for testing.

    funcol.permute_tensor uses dist.get_rank() to compute split sizes before
    calling all_to_all_single. Under LocalTensorMode with backend="fake",
    get_rank() always returns 0, so all ranks compute the same split sizes.
    This reimplementation directly permutes the per-rank tensors.
    """
    if isinstance(tensor, LocalTensor):
        # src_dst[src] = dst: rank src sends to rank dst
        # Invert: for each dst rank, find which src rank sends to it
        inv = {dst: src for src, dst in enumerate(src_dst)}
        new_shards = {}
        for r in tensor._local_tensors:
            src_rank = inv.get(r)
            if src_rank is not None and src_rank in tensor._local_tensors:
                new_shards[r] = tensor._local_tensors[src_rank].clone()
            else:
                new_shards[r] = tensor._local_tensors[r].clone()
        return LocalTensor(new_shards)
    return funcol.permute_tensor(tensor, src_dst, group, tag)


def _setup_fake_dist(world_size):
    """Setup fake distributed backend for single-process testing."""
    if dist.is_initialized():
        dist.destroy_process_group()
    store = FakeStore()
    dist.init_process_group(backend="fake", world_size=world_size, rank=0, store=store)


def _teardown_fake_dist():
    if dist.is_initialized():
        dist.destroy_process_group()


def _make_shards(global_tensor, layout, mesh_size):
    """Partition a global tensor into per-rank shards according to a ShardedLayout.

    Returns dict[rank -> local_tensor].
    """
    ndim = len(layout.global_shape)
    shards = {}

    for rank in range(mesh_size):
        # Find which tensor dim this rank shards, and compute the slice
        shard = global_tensor
        for dim in range(ndim):
            mesh_dims = layout.mesh_dim_map[dim]
            if mesh_dims:
                # This dim is sharded. Compute the slice for this rank.
                local_size = layout.local_sizes[dim]
                # For 1D mesh: rank directly indexes the shard
                start = rank * local_size
                end = start + local_size
                shard = shard.narrow(dim, start, local_size)
        shards[rank] = shard.contiguous()

    return shards


class TestRedistributeTensor(unittest.TestCase):
    """Tests for redistribute_tensor using LocalTensorMode."""

    @classmethod
    def setUpClass(cls):
        cls.world_size = 4
        _setup_fake_dist(cls.world_size)

    @classmethod
    def tearDownClass(cls):
        _teardown_fake_dist()

    def _run_redistribute(self, global_tensor, src_layout, tgt_layout):
        """Run redistribute_tensor under LocalTensorMode and verify results.

        Creates per-rank shards from src_layout, redistributes to tgt_layout,
        and verifies each rank's result matches the expected shard from tgt_layout.
        """
        with LocalTensorMode(frozenset(range(self.world_size))):
            from torch.distributed.device_mesh import init_device_mesh
            mesh = init_device_mesh("cpu", (self.world_size,))

            # Create source shards
            src_shards = _make_shards(global_tensor, src_layout, self.world_size)
            src_local = LocalTensor(src_shards)

            # Redistribute — returns a plain tensor (wait already called)
            result = redistribute_tensor(src_local, src_layout, tgt_layout, mesh)

            # Expected: shards from target layout
            tgt_shards = _make_shards(global_tensor, tgt_layout, self.world_size)

            # Verify each rank
            for rank in range(self.world_size):
                result_rank = result._local_tensors[rank] if isinstance(result, LocalTensor) else result
                expected = tgt_shards[rank]
                self.assertTrue(
                    torch.allclose(result_rank, expected, atol=1e-6),
                    f"Rank {rank}: got {result_rank}, expected {expected}"
                )

    def test_same_sharding_noop(self):
        """Same sharding → no communication, tensor unchanged."""
        t = torch.arange(8, dtype=torch.float32)
        src = ShardedLayout.shard((8,), shard_dim=0, mesh_dim_size=4)
        self._run_redistribute(t, src, src)

    def test_shard_to_replicate(self):
        """S(0) → Replicate via all_gather."""
        t = torch.arange(8, dtype=torch.float32)
        src = ShardedLayout.shard((8,), shard_dim=0, mesh_dim_size=4)
        tgt = ShardedLayout.replicate((8,))
        self._run_redistribute(t, src, tgt)

    def test_replicate_to_shard(self):
        """Replicate → S(0) via local slice (no communication)."""
        t = torch.arange(8, dtype=torch.float32)
        src = ShardedLayout.replicate((8,))
        tgt = ShardedLayout.shard((8,), shard_dim=0, mesh_dim_size=4)

        with LocalTensorMode(frozenset(range(self.world_size))):
            from torch.distributed.device_mesh import init_device_mesh
            mesh = init_device_mesh("cpu", (self.world_size,))

            # All ranks have the full tensor (replicated)
            shards = {r: t.clone() for r in range(self.world_size)}
            src_local = LocalTensor(shards)

            result = redistribute_tensor(src_local, src, tgt, mesh)
            # After local slice, each rank keeps its shard
            # Under fake PG (rank=0), mesh.get_local_rank(0) = 0 for all ranks
            # So all ranks get the slice for rank 0: t[0:2]
            for rank in range(self.world_size):
                r = result._local_tensors[rank] if isinstance(result, LocalTensor) else result
                self.assertEqual(r.shape[0], 2,
                                 f"Rank {rank}: expected local size 2, got {r.shape[0]}")

    def test_all_reduce_partial(self):
        """Partial("sum") → Replicate via all_reduce."""
        # Simulate: each rank has partial values that sum to the correct result
        t = torch.arange(8, dtype=torch.float32)
        src = ShardedLayout.replicate((8,))
        src.partial = {0: "sum"}
        tgt = ShardedLayout.replicate((8,))

        with LocalTensorMode(frozenset(range(self.world_size))):
            from torch.distributed.device_mesh import init_device_mesh
            mesh = init_device_mesh("cpu", (self.world_size,))

            # Each rank has 1/4 of the values (partial sum)
            shards = {r: t / self.world_size for r in range(self.world_size)}
            src_local = LocalTensor(shards)

            result = redistribute_tensor(src_local, src, tgt, mesh)

            # After all_reduce("sum"), each rank should have the full sum
            for rank in range(self.world_size):
                r = result._local_tensors[rank] if isinstance(result, LocalTensor) else result
                self.assertTrue(
                    torch.allclose(r, t, atol=1e-5),
                    f"Rank {rank}: got {r}, expected {t}"
                )

    def test_shard_to_replicate_2d(self):
        """2D tensor S(0) → Replicate via all_gather."""
        t = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        src = ShardedLayout.shard((4, 4), shard_dim=0, mesh_dim_size=4)
        tgt = ShardedLayout.replicate((4, 4))
        self._run_redistribute(t, src, tgt)

    def test_shard_dim1_to_replicate(self):
        """2D tensor S(1) → Replicate via all_gather on dim 1."""
        t = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        src = ShardedLayout.shard((4, 4), shard_dim=1, mesh_dim_size=4)
        tgt = ShardedLayout.replicate((4, 4))
        self._run_redistribute(t, src, tgt)

    def test_replicate_noop(self):
        """Replicate → Replicate = no-op."""
        t = torch.arange(8, dtype=torch.float32)
        src = ShardedLayout.replicate((8,))
        tgt = ShardedLayout.replicate((8,))
        with LocalTensorMode(frozenset(range(self.world_size))):
            from torch.distributed.device_mesh import init_device_mesh
            mesh = init_device_mesh("cpu", (self.world_size,))

            shards = {r: t.clone() for r in range(self.world_size)}
            src_local = LocalTensor(shards)

            result = redistribute_tensor(src_local, src, tgt, mesh)
            # Same layout → returned as-is (no collective)
            self.assertIs(result, src_local)

    def test_reduce_scatter_partial_to_shard(self):
        """Partial("sum") → S(0) via reduce_scatter."""
        t = torch.arange(8, dtype=torch.float32)
        src = ShardedLayout.replicate((8,))
        src.partial = {0: "sum"}
        tgt = ShardedLayout.shard((8,), shard_dim=0, mesh_dim_size=4)

        with LocalTensorMode(frozenset(range(self.world_size))):
            from torch.distributed.device_mesh import init_device_mesh
            mesh = init_device_mesh("cpu", (self.world_size,))

            # Each rank has 1/4 of the values (partial sum)
            shards = {r: t / self.world_size for r in range(self.world_size)}
            src_local = LocalTensor(shards)

            result = redistribute_tensor(src_local, src, tgt, mesh)

            # After reduce_scatter("sum"), rank r has t[r*2:(r+1)*2]
            for rank in range(self.world_size):
                r = result._local_tensors[rank] if isinstance(result, LocalTensor) else result
                expected = t[rank * 2:(rank + 1) * 2]
                self.assertTrue(
                    torch.allclose(r, expected, atol=1e-5),
                    f"Rank {rank}: got {r}, expected {expected}"
                )

    def test_shard_to_replicate_large(self):
        """Larger tensor S(0) → Replicate."""
        t = torch.randn(64, dtype=torch.float32)
        src = ShardedLayout.shard((64,), shard_dim=0, mesh_dim_size=4)
        tgt = ShardedLayout.replicate((64,))
        self._run_redistribute(t, src, tgt)

    def test_shard_to_replicate_3d(self):
        """3D tensor S(0) → Replicate."""
        t = torch.arange(32, dtype=torch.float32).reshape(4, 2, 4)
        src = ShardedLayout.shard((4, 2, 4), shard_dim=0, mesh_dim_size=4)
        tgt = ShardedLayout.replicate((4, 2, 4))
        self._run_redistribute(t, src, tgt)

    def test_all_to_all_s0_to_s1(self):
        """S(0) → S(1) via all_to_all on a 2D tensor."""
        t = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        src = ShardedLayout.shard((8, 4), shard_dim=0, mesh_dim_size=4)
        tgt = ShardedLayout.shard((8, 4), shard_dim=1, mesh_dim_size=4)
        self._run_redistribute(t, src, tgt)

    def test_all_to_all_s1_to_s0(self):
        """S(1) → S(0) via all_to_all on a 2D tensor."""
        t = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        src = ShardedLayout.shard((4, 8), shard_dim=1, mesh_dim_size=4)
        tgt = ShardedLayout.shard((4, 8), shard_dim=0, mesh_dim_size=4)
        self._run_redistribute(t, src, tgt)

    def test_all_to_all_s0_to_s1_3d(self):
        """S(0) → S(1) via all_to_all on a 3D tensor."""
        t = torch.arange(96, dtype=torch.float32).reshape(4, 8, 3)
        src = ShardedLayout.shard((4, 8, 3), shard_dim=0, mesh_dim_size=4)
        tgt = ShardedLayout.shard((4, 8, 3), shard_dim=1, mesh_dim_size=4)
        self._run_redistribute(t, src, tgt)

    def test_s0s0_ltr_to_rtl_symmetric_noop(self):
        """S(0)S(0) LTR → RTL on symmetric (2,2) mesh is a no-op.

        With same-size mesh dims, LTR and RTL produce identical layouts
        (same element-to-rank mapping), so no redistribution needed.
        """
        from autoparallel.shardings.cute import plan_redistribute
        ltr = ShardedLayout.shard_multi((16,), [(0, 2), (0, 2)])
        rtl = ShardedLayout.shard_multi((16,), [(0, 2), (0, 2)])
        collectives = plan_redistribute(ltr, rtl)
        self.assertEqual(collectives, [])

    def test_s0s0_ltr_to_rtl_asymmetric(self):
        """S(0)S(0) LTR → RTL on asymmetric (2,4) mesh via global ppermute."""
        from autoparallel.shardings.cute import plan_redistribute
        from autoparallel.shardings.cute.strategy import enumerate_shardings

        shardings = enumerate_shardings((16,), (2, 4))
        s0s0 = [s for s in shardings if len(s.mesh_dim_map.get(0, ())) > 1]
        self.assertEqual(len(s0s0), 2, "Expected 2 S(0)S(0) shardings on (2,4) mesh")

        ltr, rtl = s0s0[0], s0s0[1]
        collectives = plan_redistribute(ltr, rtl)
        # Should have a non-trivial global ppermute
        has_real_data_movement = any(
            ct == "ppermute" and any(s != d for s, d in info.get("perm", []))
            for ct, _, info in collectives
        )
        self.assertTrue(has_real_data_movement,
                        "LTR->RTL on asymmetric mesh should require data movement")

    def test_shard_to_replicate_2d_mesh(self):
        """S(0) on mesh_size=4 → Replicate on a 2D 4x4 tensor."""
        t = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        src = ShardedLayout.shard((4, 4), shard_dim=0, mesh_dim_size=4)
        tgt = ShardedLayout.replicate((4, 4))
        self._run_redistribute(t, src, tgt)

    def test_shard_to_shard_different_dim_size(self):
        """S(0) mesh=2 on dim 0 → S(0) mesh=2 on dim 1, same mesh, 4x4 tensor."""
        t = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        src = ShardedLayout.shard((4, 4), shard_dim=0, mesh_dim_size=2)
        tgt = ShardedLayout.shard((4, 4), shard_dim=1, mesh_dim_size=2)

        with LocalTensorMode(frozenset(range(2))):
            from torch.distributed.device_mesh import init_device_mesh
            mesh = init_device_mesh("cpu", (2,))

            src_shards = {0: t[:2, :].contiguous(), 1: t[2:, :].contiguous()}
            src_local = LocalTensor(src_shards)

            result = redistribute_tensor(src_local, src, tgt, mesh)

            tgt_shards = {0: t[:, :2].contiguous(), 1: t[:, 2:].contiguous()}
            for rank in range(2):
                r = result._local_tensors[rank]
                expected = tgt_shards[rank]
                self.assertTrue(
                    torch.allclose(r, expected, atol=1e-6),
                    f"Rank {rank}: got {r}, expected {expected}"
                )

    def test_partial_sum_to_shard_2d(self):
        """Partial(sum) → S(1) via reduce_scatter on a 2D tensor."""
        t = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        src = ShardedLayout.replicate((2, 4))
        src.partial = {0: "sum"}
        tgt = ShardedLayout.shard((2, 4), shard_dim=1, mesh_dim_size=4)

        with LocalTensorMode(frozenset(range(self.world_size))):
            from torch.distributed.device_mesh import init_device_mesh
            mesh = init_device_mesh("cpu", (self.world_size,))

            shards = {r: t / self.world_size for r in range(self.world_size)}
            src_local = LocalTensor(shards)

            result = redistribute_tensor(src_local, src, tgt, mesh)

            for rank in range(self.world_size):
                r = result._local_tensors[rank] if isinstance(result, LocalTensor) else result
                expected = t[:, rank:rank + 1]
                self.assertTrue(
                    torch.allclose(r, expected, atol=1e-5),
                    f"Rank {rank}: got {r}, expected {expected}"
                )

    def test_all_to_all_s0_to_s1_square(self):
        """S(0) → S(1) on a square 4x4 tensor, mesh=4."""
        t = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        src = ShardedLayout.shard((4, 4), shard_dim=0, mesh_dim_size=4)
        tgt = ShardedLayout.shard((4, 4), shard_dim=1, mesh_dim_size=4)
        self._run_redistribute(t, src, tgt)

    def test_offset_no_change(self):
        """Same layout with same offset → no-op."""
        N = 4
        hier = Layout((((2, N),),), (((XorStride(2 * N - 1), 1),),))
        sl = ShardedLayout(hier, {0: (0,)}).with_offset({0: 3})
        from autoparallel.shardings.cute import plan_redistribute
        self.assertEqual(plan_redistribute(sl, sl), [])

    @patch("autoparallel.shardings.cute.redistribute_tensor.funcol.permute_tensor",
           side_effect=_local_permute_tensor)
    def test_ring_attention_step0_to_step1(self, mock_permute):
        """Ring attention step 0 → step 1 via global ppermute (circular shift)."""
        N = self.world_size  # 4
        n_chunks = 2 * N  # 8

        # Build step 0 and step 1 layouts (chunk-level, 8 elements)
        hier = Layout((((2, N),),), (((XorStride(n_chunks - 1), 1),),))
        step0 = ShardedLayout(hier, {0: (0,)})
        step1 = step0.with_offset({0: N - 1})

        # Step 0: GPU g gets chunks {g, g^7} = {g, 7-g}
        # Step 1: GPU g gets chunks {(g+3)%4, ((g+3)%4)^7}
        # Transition: circular left shift — GPU g sends to GPU (g+1)%4

        with LocalTensorMode(frozenset(range(N))):
            from torch.distributed.device_mesh import init_device_mesh
            mesh = init_device_mesh("cpu", (N,))

            # Build step 0 shards: GPU g has value g (representing its chunk id)
            src_shards = {g: torch.tensor([float(g), float(g ^ (n_chunks - 1))])
                          for g in range(N)}
            src_local = LocalTensor(src_shards)

            result = redistribute_tensor(src_local, step0, step1, mesh)

            # After circular shift: GPU g receives from GPU (g-1)%4
            for g in range(N):
                r = result._local_tensors[g] if isinstance(result, LocalTensor) else result
                sender = (g - 1) % N
                expected = torch.tensor([float(sender), float(sender ^ (n_chunks - 1))])
                self.assertTrue(
                    torch.allclose(r, expected, atol=1e-6),
                    f"GPU {g}: got {r}, expected {expected} (from GPU {sender})"
                )

    @patch("autoparallel.shardings.cute.redistribute_tensor.funcol.permute_tensor",
           side_effect=_local_permute_tensor)
    def test_ring_attention_step1_to_step2(self, mock_permute):
        """Ring attention step 1 → step 2: same circular shift pattern."""
        N = self.world_size
        n_chunks = 2 * N

        hier = Layout((((2, N),),), (((XorStride(n_chunks - 1), 1),),))
        step1 = ShardedLayout(hier, {0: (0,)}).with_offset({0: N - 1})
        step2 = ShardedLayout(hier, {0: (0,)}).with_offset({0: 2 * (N - 1)})

        with LocalTensorMode(frozenset(range(N))):
            from torch.distributed.device_mesh import init_device_mesh
            mesh = init_device_mesh("cpu", (N,))

            src_shards = {g: torch.tensor([float(g)]) for g in range(N)}
            src_local = LocalTensor(src_shards)

            result = redistribute_tensor(src_local, step1, step2, mesh)

            for g in range(N):
                r = result._local_tensors[g] if isinstance(result, LocalTensor) else result
                sender = (g - 1) % N
                expected = torch.tensor([float(sender)])
                self.assertTrue(
                    torch.allclose(r, expected, atol=1e-6),
                    f"GPU {g}: got {r}, expected {expected}"
                )

    @patch("autoparallel.shardings.cute.redistribute_tensor.funcol.permute_tensor",
           side_effect=_local_permute_tensor)
    def test_ring_attention_full_rotation(self, mock_permute):
        """N steps of ring rotation returns data to original GPU."""
        N = self.world_size
        n_chunks = 2 * N

        hier = Layout((((2, N),),), (((XorStride(n_chunks - 1), 1),),))

        with LocalTensorMode(frozenset(range(N))):
            from torch.distributed.device_mesh import init_device_mesh
            mesh = init_device_mesh("cpu", (N,))

            # Start: GPU g has value g
            tensor = LocalTensor({g: torch.tensor([float(g)]) for g in range(N)})

            # Apply N steps of circular shift
            for step in range(N):
                src = ShardedLayout(hier, {0: (0,)}).with_offset({0: step * (N - 1)})
                tgt = ShardedLayout(hier, {0: (0,)}).with_offset({0: (step + 1) * (N - 1)})
                tensor = redistribute_tensor(tensor, src, tgt, mesh)

            # After N shifts, data should be back at original GPU
            for g in range(N):
                r = tensor._local_tensors[g] if isinstance(tensor, LocalTensor) else tensor
                expected = torch.tensor([float(g)])
                self.assertTrue(
                    torch.allclose(r, expected, atol=1e-6),
                    f"GPU {g}: got {r}, expected {expected} after full rotation"
                )


if __name__ == "__main__":
    unittest.main()
