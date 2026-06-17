# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import nn
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel import AutoParallel, with_sharding_constraint
from autoparallel.collectives import local_map
from autoparallel.ops import doc_packed_attn, permutation


def get_local_map_nodes(graph, is_backward=False):
    nodes = []
    for node in graph.nodes:
        if "local_map_kwargs" in node.meta:
            node_is_backward = node.meta.get("partitioner_tag", "") == "is_backward"
            if node_is_backward == is_backward:
                nodes.append(node)
    return nodes


def verify_local_map_placements(sharding_placement, node, expected_placements):
    spec = sharding_placement[node]
    if isinstance(spec.output_specs, tuple):
        output_spec = spec.output_specs[0]
    else:
        output_spec = spec.output_specs
    assert (
        output_spec.placements == expected_placements
    ), f"Expected placements {expected_placements}, got {output_spec.placements}"


class TestWithShardingConstraint:
    """Tests for the with_sharding_constraint operator."""

    def test_with_sharding_constraint_explicit_mesh(self, device_mesh_1d):
        """Test with_sharding_constraint with an explicit device mesh."""
        dim = 128

        class Model(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.linear1 = nn.Linear(dim, dim, bias=False)
                self.linear2 = nn.Linear(dim, dim, bias=False)

            def forward(self, x):
                x = self.linear1(x)
                # Constrain intermediate result to be sharded
                x = with_sharding_constraint(x, (Shard(0),), device_mesh_1d)
                x = self.linear2(x)
                return x

        def input_fn():
            return torch.rand(512, dim, device="cuda")

        with torch.device("meta"):
            model = Model(dim)

        with AutoParallel(model, input_fn, device_mesh_1d) as autop:
            autop.add_input_constraints([(Shard(0),)])
            autop.add_output_constraints([(Shard(0),)])
            sharding_placement = autop.optimize_placement()

            # Verify the with_sharding_constraint node has correct placement
            local_map_nodes = get_local_map_nodes(autop.gm.graph, is_backward=False)
            assert len(local_map_nodes) == 1, "Expected 1 forward local_map node"
            verify_local_map_placements(
                sharding_placement, local_map_nodes[0], (Shard(0),)
            )

            parallel_mod = autop.apply_placement(sharding_placement)

        assert parallel_mod is not None

    def test_with_sharding_constraint_between_local_maps(self, device_mesh_1d):
        """Test with_sharding_constraint between local_map regions."""
        dim = 128

        @local_map(
            out_placements=((Shard(0),),),
            in_placements=((Shard(0),),),
            redistribute_inputs=True,
            device_mesh=device_mesh_1d,
        )
        def compute1(x):
            return x + 1

        @local_map(
            out_placements=((Shard(0),),),
            in_placements=((Shard(0),),),
            redistribute_inputs=True,
            device_mesh=device_mesh_1d,
        )
        def compute2(x):
            return x * 2

        class Model(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.linear = nn.Linear(dim, dim, bias=False)

            def forward(self, x):
                x = self.linear(x)
                x = compute1(x)
                # Constraint applied between local_map regions (at DTensor level)
                x = with_sharding_constraint(x, (Shard(0),), device_mesh_1d)
                x = compute2(x)
                return x

        def input_fn():
            return torch.rand(512, dim, device="cuda")

        with torch.device("meta"):
            model = Model(dim)

        with AutoParallel(model, input_fn, device_mesh_1d) as autop:
            autop.add_input_constraints([(Shard(0),)])
            autop.add_output_constraints([(Shard(0),)])
            sharding_placement = autop.optimize_placement()

            # Verify all local_map nodes have correct placement
            # There are 3 forward local_map nodes: compute1, with_sharding_constraint, compute2
            local_map_nodes = get_local_map_nodes(autop.gm.graph, is_backward=False)
            assert (
                len(local_map_nodes) == 3
            ), f"Expected 3 forward local_map nodes, got {len(local_map_nodes)}"
            for node in local_map_nodes:
                verify_local_map_placements(sharding_placement, node, (Shard(0),))

            parallel_mod = autop.apply_placement(sharding_placement)

        assert parallel_mod is not None

    def test_with_sharding_constraint_replicate(self, device_mesh_1d):
        """Test with_sharding_constraint to force replication."""
        dim = 128

        class Model(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.linear1 = nn.Linear(dim, dim, bias=False)
                self.linear2 = nn.Linear(dim, dim, bias=False)

            def forward(self, x):
                x = self.linear1(x)
                # Force intermediate to be replicated
                x = with_sharding_constraint(x, (Replicate(),), device_mesh_1d)
                x = self.linear2(x)
                return x

        def input_fn():
            return torch.rand(512, dim, device="cuda")

        with torch.device("meta"):
            model = Model(dim)

        with AutoParallel(model, input_fn, device_mesh_1d) as autop:
            autop.add_input_constraints([(Shard(0),)])
            autop.add_output_constraints([(Shard(0),)])
            sharding_placement = autop.optimize_placement()

            # Verify the with_sharding_constraint node forces Replicate
            local_map_nodes = get_local_map_nodes(autop.gm.graph, is_backward=False)
            assert len(local_map_nodes) == 1, "Expected 1 forward local_map node"
            verify_local_map_placements(
                sharding_placement, local_map_nodes[0], (Replicate(),)
            )

            parallel_mod = autop.apply_placement(sharding_placement)

        assert parallel_mod is not None

    def test_with_sharding_constraint_2d_mesh(self, device_mesh_2d):
        """Test with_sharding_constraint on a 2D mesh."""
        dim = 128

        class Model(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.linear1 = nn.Linear(dim, dim, bias=False)
                self.linear2 = nn.Linear(dim, dim, bias=False)

            def forward(self, x):
                x = self.linear1(x)
                # Shard along batch dim on dp, replicate on tp
                x = with_sharding_constraint(x, (Shard(0), Replicate()), device_mesh_2d)
                x = self.linear2(x)
                return x

        def input_fn():
            return torch.rand(512, dim, device="cuda")

        with torch.device("meta"):
            model = Model(dim)

        with AutoParallel(model, input_fn, device_mesh_2d) as autop:
            autop.add_input_constraints([(Shard(0), Replicate())])
            autop.add_output_constraints([(Shard(0), Replicate())])
            sharding_placement = autop.optimize_placement()

            # Verify the with_sharding_constraint node has correct 2D placement
            local_map_nodes = get_local_map_nodes(autop.gm.graph, is_backward=False)
            assert len(local_map_nodes) == 1, "Expected 1 forward local_map node"
            verify_local_map_placements(
                sharding_placement, local_map_nodes[0], (Shard(0), Replicate())
            )

            parallel_mod = autop.apply_placement(sharding_placement)

        assert parallel_mod is not None

    def test_with_sharding_constraint_multiple(self, device_mesh_1d):
        """Test multiple with_sharding_constraint calls in sequence."""
        dim = 128

        class Model(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.linear1 = nn.Linear(dim, dim, bias=False)
                self.linear2 = nn.Linear(dim, dim, bias=False)
                self.linear3 = nn.Linear(dim, dim, bias=False)

            def forward(self, x):
                x = self.linear1(x)
                x = with_sharding_constraint(x, (Shard(0),), device_mesh_1d)
                x = self.linear2(x)
                x = with_sharding_constraint(x, (Replicate(),), device_mesh_1d)
                x = self.linear3(x)
                x = with_sharding_constraint(x, (Shard(0),), device_mesh_1d)
                return x

        def input_fn():
            return torch.rand(512, dim, device="cuda")

        with torch.device("meta"):
            model = Model(dim)

        with AutoParallel(model, input_fn, device_mesh_1d) as autop:
            autop.add_input_constraints([(Shard(0),)])
            autop.add_output_constraints([(Shard(0),)])
            sharding_placement = autop.optimize_placement()

            # Verify all 3 with_sharding_constraint nodes have correct placements
            local_map_nodes = get_local_map_nodes(autop.gm.graph, is_backward=False)
            assert (
                len(local_map_nodes) == 3
            ), f"Expected 3 forward local_map nodes, got {len(local_map_nodes)}"

            # Nodes should be in order: Shard(0), Replicate(), Shard(0)
            expected_placements = [(Shard(0),), (Replicate(),), (Shard(0),)]
            for node, expected in zip(local_map_nodes, expected_placements):
                verify_local_map_placements(sharding_placement, node, expected)

            parallel_mod = autop.apply_placement(sharding_placement)

        assert parallel_mod is not None

    def test_with_sharding_constraint_no_mesh_outside_local_map_raises(self):
        """Test that with_sharding_constraint raises error when no mesh is available."""
        x = torch.rand(10, 10)
        with pytest.raises(RuntimeError, match="No device mesh is currently active"):
            with_sharding_constraint(x, (Shard(0),))


class TestPermutation:
    def test_shape_preserved(self):
        """Permutation should preserve tensor shape."""
        x = torch.randn(5, 10, 3)
        result = permutation(x, axis=0)
        assert result.shape == x.shape

        result = permutation(x, axis=1)
        assert result.shape == x.shape

        result = permutation(x, axis=2)
        assert result.shape == x.shape

    def test_elements_preserved(self):
        """Permutation should preserve all elements (just reordered)."""
        x = torch.arange(24).reshape(4, 6)
        result = permutation(x, axis=0)

        # Sort along axis and compare
        x_sorted = x.sort(dim=0).values
        result_sorted = result.sort(dim=0).values
        assert torch.equal(x_sorted, result_sorted)

    def test_elements_preserved_axis1(self):
        """Permutation along axis=1 should preserve all elements."""
        x = torch.arange(24).reshape(4, 6)
        result = permutation(x, axis=1)

        x_sorted = x.sort(dim=1).values
        result_sorted = result.sort(dim=1).values
        assert torch.equal(x_sorted, result_sorted)

    def test_independent_false_same_permutation(self):
        """With independent=False, the same permutation is applied to all slices."""
        torch.manual_seed(42)
        x = torch.arange(12).reshape(3, 4)
        result = permutation(x, axis=0, independent=False)

        # argsort gives the indices that would sort each column
        # If the same permutation is applied to all columns, all columns
        # should have identical argsort indices
        sort_indices = result.argsort(0)
        assert (sort_indices == sort_indices[:, :1]).all()

    def test_independent_true_different_permutations(self):
        """With independent=True, different permutations for each slice."""
        torch.manual_seed(42)
        # Use a larger tensor to make it very unlikely all columns get same permutation
        x = torch.arange(100).reshape(10, 10)
        result = permutation(x, axis=0, independent=True)

        # Elements should still be preserved per column
        for col in range(x.shape[1]):
            x_col_sorted = x[:, col].sort().values
            result_col_sorted = result[:, col].sort().values
            assert torch.equal(x_col_sorted, result_col_sorted)

        # With independent=True, at least some columns should have different orderings
        # Check by looking at the relative positions
        col0_order = result[:, 0].argsort()
        different_order_found = False
        for col in range(1, x.shape[1]):
            col_order = result[:, col].argsort()
            if not torch.equal(col0_order, col_order):
                different_order_found = True
                break
        assert (
            different_order_found
        ), "Expected different permutations for different columns"

    def test_1d_tensor(self):
        """Permutation works on 1D tensors."""
        x = torch.arange(10)
        result = permutation(x, axis=0)
        assert result.shape == x.shape
        assert set(result.tolist()) == set(x.tolist())

    def test_negative_axis(self):
        """Permutation works with negative axis."""
        x = torch.randn(3, 4, 5)
        result = permutation(x, axis=-1)
        assert result.shape == x.shape

    def test_device_preserved(self):
        """Result should be on same device as input."""
        x = torch.randn(5, 10)
        result = permutation(x)
        assert result.device == x.device

    def test_dtype_preserved(self):
        """Result should have same dtype as input."""
        for dtype in [torch.float32, torch.float64, torch.int64, torch.int32]:
            if dtype.is_floating_point:
                x = torch.randn(5, 10, dtype=dtype)
            else:
                x = torch.randint(0, 100, (5, 10), dtype=dtype)
            result = permutation(x)
            assert result.dtype == dtype

    def test_reproducibility_with_seed(self):
        """Same seed should produce same permutation."""
        x = torch.arange(20).reshape(4, 5)

        torch.manual_seed(123)
        result1 = permutation(x, axis=0)

        torch.manual_seed(123)
        result2 = permutation(x, axis=0)

        assert torch.equal(result1, result2)

    def test_independent_reproducibility(self):
        """Same seed should produce same result with independent=True."""
        x = torch.arange(20).reshape(4, 5)

        torch.manual_seed(456)
        result1 = permutation(x, axis=0, independent=True)

        torch.manual_seed(456)
        result2 = permutation(x, axis=0, independent=True)

        assert torch.equal(result1, result2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_tensor(self):
        """Permutation works on CUDA tensors."""
        x = torch.randn(5, 10, device="cuda")
        result = permutation(x, axis=0)
        assert result.device == x.device
        assert result.shape == x.shape


# ---------------------------------------------------------------------------
# doc_packed_attn
# ---------------------------------------------------------------------------


def _build_padded_cu_seq_q(
    per_batch_doc_lens: list[list[int]],
    n_docs: int,
    seq_len: int,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Build a ``[B, n_docs+1]`` int32 cu_seq_q tensor from per-batch doc lengths.

    Rows with fewer than ``n_docs`` real documents are padded by repeating
    ``seq_len`` at the trailing positions, so those slots represent zero-length
    documents that the kernel skips.
    """
    B = len(per_batch_doc_lens)
    cu = torch.full((B, n_docs + 1), seq_len, dtype=torch.int32, device=device)
    for b, lens in enumerate(per_batch_doc_lens):
        assert (
            sum(lens) == seq_len
        ), f"batch {b}: doc lengths {lens} sum to {sum(lens)}, expected {seq_len}"
        assert (
            len(lens) <= n_docs
        ), f"batch {b}: {len(lens)} real docs > n_docs={n_docs}"
        cumsum = torch.tensor(
            [0] + list(torch.tensor(lens).cumsum(0).tolist()),
            dtype=torch.int32,
            device=device,
        )
        cu[b, : len(cumsum)] = cumsum
    return cu


def _per_doc_sdpa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    per_batch_doc_lens: list[list[int]],
    window_size: tuple[int, int],
) -> torch.Tensor:
    """Reference implementation: SDPA per document, then concat.

    Builds an explicit mask for sliding-window cases so it exactly matches
    flash-attention's per-doc causal+window semantics.
    """
    import torch.nn.functional as F

    is_full = window_size == (-1, -1)
    is_causal = window_size == (-1, 0)
    out = torch.zeros_like(q)
    for b in range(q.shape[0]):
        offset = 0
        for L in per_batch_doc_lens[b]:
            q_doc = q[b, offset : offset + L].transpose(0, 1).unsqueeze(0)
            k_doc = k[b, offset : offset + L].transpose(0, 1).unsqueeze(0)
            v_doc = v[b, offset : offset + L].transpose(0, 1).unsqueeze(0)
            if is_full:
                o = F.scaled_dot_product_attention(
                    q_doc, k_doc, v_doc, is_causal=False, enable_gqa=True
                )
            elif is_causal:
                o = F.scaled_dot_product_attention(
                    q_doc, k_doc, v_doc, is_causal=True, enable_gqa=True
                )
            else:
                window_left = window_size[0] + 1  # inclusive of self
                i_idx = torch.arange(L, device=q.device).unsqueeze(1)
                j_idx = torch.arange(L, device=q.device).unsqueeze(0)
                mask = (j_idx <= i_idx) & (i_idx - j_idx < window_left)
                o = F.scaled_dot_product_attention(
                    q_doc, k_doc, v_doc, attn_mask=mask, enable_gqa=True
                )
            out[b, offset : offset + L] = o.squeeze(0).transpose(0, 1)
            offset += L
    return out


class TestDocPackedAttn:
    """Tests for autoparallel.ops.doc_packed_attn.

    Phase-1 contract:
      - q/k/v: ``[B, S, H, D]`` (THD reshaped internally).
      - cu_seq_q: ``[B, n_docs+1]`` int32; padding is repeated ``S`` past real docs.
      - n_docs is uniform across batch elements.
      - Returns a single ``[B, S, H, D]`` tensor (the underlying op also yields
        ``lse``/``rng_state`` for backward but the public helper hides them).
    """

    # -- schema / fake mode ----------------------------------------------------

    def test_schema(self):
        """Op schema mirrors varlen_attn's argument style and returns 3 tensors."""
        schema = torch.ops.autoparallel.doc_packed_attn_op.default._schema
        s = str(schema)
        assert "Tensor query" in s
        assert "Tensor cu_seq_q" in s
        assert "SymInt n_docs" in s
        assert "SymInt[]? window_size" in s
        assert "bool enable_gqa" in s
        # 3-tuple return: (out, lse, rng_state)
        assert "-> (Tensor, Tensor, Tensor)" in s

    def test_fake_tensor_trace(self):
        """FakeTensorMode tracing produces a tensor with an autograd grad_fn."""
        from torch._subclasses import FakeTensorMode

        with FakeTensorMode():
            B, S, Hq, Hkv, D = 1, 128, 8, 4, 64
            q = torch.empty(
                B, S, Hq, D, dtype=torch.bfloat16, device="cuda", requires_grad=True
            )
            k = torch.empty(
                B, S, Hkv, D, dtype=torch.bfloat16, device="cuda", requires_grad=True
            )
            v = torch.empty(
                B, S, Hkv, D, dtype=torch.bfloat16, device="cuda", requires_grad=True
            )
            cu = torch.empty(B, 5, dtype=torch.int32, device="cuda")
            out = doc_packed_attn(
                q, k, v, cu, n_docs=3, window_size=(-1, 0), enable_gqa=True
            )
            assert out.shape == q.shape
            assert out.dtype == q.dtype
            assert out.grad_fn is not None

    # -- input validation ------------------------------------------------------

    def test_gqa_disabled_requires_matching_heads(self):
        """Without enable_gqa, Hq must equal Hkv."""
        from torch._subclasses import FakeTensorMode

        with FakeTensorMode():
            q = torch.empty(1, 128, 8, 64, dtype=torch.bfloat16, device="cuda")
            k = torch.empty(1, 128, 4, 64, dtype=torch.bfloat16, device="cuda")
            v = torch.empty(1, 128, 4, 64, dtype=torch.bfloat16, device="cuda")
            cu = torch.empty(1, 5, dtype=torch.int32, device="cuda")
            with pytest.raises(ValueError, match="enable_gqa=True"):
                doc_packed_attn(q, k, v, cu, n_docs=3)

    def test_gqa_requires_divisible_heads(self):
        """With enable_gqa, Hq must be a multiple of Hkv."""
        from torch._subclasses import FakeTensorMode

        with FakeTensorMode():
            q = torch.empty(1, 128, 9, 64, dtype=torch.bfloat16, device="cuda")
            k = torch.empty(1, 128, 4, 64, dtype=torch.bfloat16, device="cuda")
            v = torch.empty(1, 128, 4, 64, dtype=torch.bfloat16, device="cuda")
            cu = torch.empty(1, 5, dtype=torch.int32, device="cuda")
            with pytest.raises(ValueError, match="multiple of kv heads"):
                doc_packed_attn(q, k, v, cu, n_docs=3, enable_gqa=True)

    # -- parity (B=1, B>1, uniform & mixed doc counts) -------------------------

    # Curated cases: (name, per_batch_doc_lens, n_docs, window_size)
    _PARITY_CASES = [
        # B=1
        ("B1_causal", [[100, 80, 76]], 3, (-1, 0)),
        ("B1_full", [[100, 80, 76]], 3, (-1, -1)),
        ("B1_swa_W32", [[100, 80, 76]], 3, (31, 0)),
        ("B1_single_doc", [[256]], 1, (-1, 0)),
        # B>1, uniform doc counts (no padding)
        ("B2_uniform", [[100, 80, 76], [128, 64, 64]], 3, (-1, 0)),
        ("B2_uniform_swa", [[100, 80, 76], [128, 64, 64]], 3, (31, 0)),
        (
            "B4_uniform",
            [
                [64, 64, 64, 64],
                [128, 64, 32, 32],
                [200, 24, 24, 8],
                [80, 80, 48, 48],
            ],
            4,
            (-1, 0),
        ),
        # B>1, mixed doc counts (zero-length-doc padding)
        (
            "B2_mixed_3v5",
            [[100, 80, 76], [60, 60, 60, 60, 16]],
            5,
            (-1, 0),
        ),
        (
            "B3_mixed_2v4v6",
            [
                [200, 56],
                [100, 60, 50, 46],
                [60, 50, 50, 40, 30, 26],
            ],
            6,
            (-1, 0),
        ),
    ]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "name, per_batch_doc_lens, n_docs, window_size",
        _PARITY_CASES,
        ids=[c[0] for c in _PARITY_CASES],
    )
    def test_parity_vs_per_doc_sdpa(
        self, name, per_batch_doc_lens, n_docs, window_size
    ):
        """Forward + backward match per-document SDPA reference at bf16 precision."""
        torch.manual_seed(42)
        S = 256
        B = len(per_batch_doc_lens)
        Hq, Hkv, D = 8, 4, 64
        cu_seq_q = _build_padded_cu_seq_q(per_batch_doc_lens, n_docs, S)

        q = torch.randn(
            B, S, Hq, D, dtype=torch.bfloat16, device="cuda", requires_grad=True
        )
        k = torch.randn(
            B, S, Hkv, D, dtype=torch.bfloat16, device="cuda", requires_grad=True
        )
        v = torch.randn(
            B, S, Hkv, D, dtype=torch.bfloat16, device="cuda", requires_grad=True
        )

        out = doc_packed_attn(
            q, k, v, cu_seq_q, n_docs, window_size=window_size, enable_gqa=True
        )
        grad = torch.randn_like(out)
        out.backward(grad)

        q_ref = q.detach().clone().requires_grad_(True)
        k_ref = k.detach().clone().requires_grad_(True)
        v_ref = v.detach().clone().requires_grad_(True)
        out_ref = _per_doc_sdpa_reference(
            q_ref, k_ref, v_ref, per_batch_doc_lens, window_size
        )
        out_ref.backward(grad)

        # bf16 epsilon ~ 8e-3; allow ~4x headroom because flash and SDPA
        # accumulate dk across the sequence dim in different orders, and
        # GQA broadcasting compounds rounding for the longest single-doc cases.
        tol = 3e-2
        assert torch.isfinite(out).all()
        assert (out - out_ref).abs().max().item() < tol, (
            f"forward parity failure: max_abs_diff="
            f"{(out - out_ref).abs().max().item():.2e}"
        )
        for name_g, g, g_ref in [
            ("q", q.grad, q_ref.grad),
            ("k", k.grad, k_ref.grad),
            ("v", v.grad, v_ref.grad),
        ]:
            diff = (g - g_ref).abs().max().item()
            assert diff < tol, f"{name_g}.grad parity failure: max_abs_diff={diff:.2e}"

    # -- padding semantics -----------------------------------------------------

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_zero_length_padding_does_not_affect_real_docs(self):
        """Padding past n_docs (zero-length trailing docs) must not change output.

        Two cu_seq_q tensors with identical real-doc boundaries but different
        padding (n_docs=3 tight vs n_docs=8 padded) should produce bit-identical
        results because the trailing entries describe zero-length documents.
        """
        torch.manual_seed(0)
        B, S, Hq, Hkv, D = 1, 256, 8, 4, 64
        per_batch = [[100, 80, 76]]

        q = torch.randn(B, S, Hq, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, S, Hkv, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, S, Hkv, D, dtype=torch.bfloat16, device="cuda")

        cu_tight = _build_padded_cu_seq_q(per_batch, n_docs=3, seq_len=S)
        cu_padded = _build_padded_cu_seq_q(per_batch, n_docs=8, seq_len=S)

        out_tight = doc_packed_attn(
            q, k, v, cu_tight, n_docs=3, window_size=(-1, 0), enable_gqa=True
        )
        out_padded = doc_packed_attn(
            q, k, v, cu_padded, n_docs=8, window_size=(-1, 0), enable_gqa=True
        )
        assert torch.equal(out_tight, out_padded), (
            "zero-length padding changed the output — kernel may not handle "
            "trailing empty segments correctly"
        )

    # -- no recompute in backward ---------------------------------------------

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_backward_does_not_recompute_forward(self):
        """Profile N iterations: flash-fwd count must equal iteration count.

        If the backward recomputed the forward, fwd_count would be 2*N. We
        save (out, lse, rng_state) from forward and dispatch directly to
        _flash_attention_backward, so fwd_count == bwd_count == N.
        """
        from torch.profiler import ProfilerActivity, profile

        torch.manual_seed(0)
        B, S, Hq, Hkv, D = 1, 256, 8, 4, 64
        per_batch = [[100, 80, 76]]
        cu_seq_q = _build_padded_cu_seq_q(per_batch, n_docs=3, seq_len=S)

        q = torch.randn(
            B, S, Hq, D, dtype=torch.bfloat16, device="cuda", requires_grad=True
        )
        k = torch.randn(
            B, S, Hkv, D, dtype=torch.bfloat16, device="cuda", requires_grad=True
        )
        v = torch.randn(
            B, S, Hkv, D, dtype=torch.bfloat16, device="cuda", requires_grad=True
        )

        n_iters = 3
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(n_iters):
                out = doc_packed_attn(
                    q, k, v, cu_seq_q, 3, window_size=(-1, 0), enable_gqa=True
                )
                out.backward(torch.randn_like(out))
                q.grad = k.grad = v.grad = None

        fwd_count = bwd_count = 0
        for row in prof.key_averages():
            if "flash_attention_forward" in row.key:
                fwd_count = row.count
            elif "flash_attention_backward" in row.key:
                bwd_count = row.count
        assert fwd_count == n_iters, (
            f"flash_attention_forward called {fwd_count} times for {n_iters} "
            f"iterations — backward is recomputing the forward"
        )
        assert bwd_count == n_iters, (
            f"flash_attention_backward called {bwd_count} times for {n_iters} "
            f"iterations"
        )


class TestDocPackedAttnShardingStrategy:
    """Tests for AP's sharding strategy on autoparallel::doc_packed_attn_op.

    These verify the strategy enumerates the right placement options and that
    the solver picks the expected DP / DP+TP shardings when constraints pin
    the inputs. They use ``JustAttn`` (no surrounding linear layers) so that
    the test is independent of how the solver scores larger graphs.
    """

    class _JustAttn(nn.Module):
        """Minimal model: a single doc_packed_attn call, no linears.

        Used to isolate AP's choice for the attention op from upstream/down-
        stream sharding decisions (which a real transformer would dominate).
        """

        def __init__(self, n_docs: int):
            super().__init__()
            self.n_docs = n_docs

        def forward(self, q, k, v, cu_seq_q):
            return doc_packed_attn(
                q,
                k,
                v,
                cu_seq_q,
                n_docs=self.n_docs,
                window_size=(-1, 0),
                enable_gqa=True,
            )

    @staticmethod
    def _input_fn_factory(B, S, Hq, Hkv, D, n_docs):
        """Returns an input_fn that builds q/k/v plus uniform-doc cu_seq_q."""

        def input_fn():
            q = torch.randn(B, S, Hq, D, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.bfloat16)
            per_doc = S // n_docs
            base = torch.arange(n_docs + 1, dtype=torch.int32, device="cuda") * per_doc
            cu = base.unsqueeze(0).expand(B, -1).contiguous()
            return q, k, v, cu

        return input_fn

    @staticmethod
    def _find_doc_packed_node(gm):
        for node in gm.graph.nodes:
            if (
                node.op == "call_function"
                and getattr(node.target, "name", lambda: "")()
                == "autoparallel::doc_packed_attn_op"
            ):
                return node
        raise AssertionError("autoparallel::doc_packed_attn_op not found in graph")

    def test_dp_only_pins_shard0(self, device_mesh_1d):
        """1D dp mesh, all inputs pinned Shard(0): op picks Shard(0) end-to-end."""
        # B must be divisible by mesh size (256 in the fixture).
        B, S, Hq, Hkv, D, n_docs = 256, 256, 8, 4, 64, 4
        model = self._JustAttn(n_docs).to(dtype=torch.bfloat16)
        input_fn = self._input_fn_factory(B, S, Hq, Hkv, D, n_docs)

        with AutoParallel(
            model, input_fn, device_mesh_1d, repeated_subgraphs=False
        ) as autop:
            autop.add_input_constraints(
                [
                    (Shard(0),),  # q
                    (Shard(0),),  # k
                    (Shard(0),),  # v
                    (Shard(0),),  # cu_seq_q
                ]
            )
            autop.add_output_constraints([(Shard(0),)])
            sol = autop.optimize_placement()

            node = self._find_doc_packed_node(autop.gm)
            spec = sol[node]
            in_placements = [s.placements for s in spec.input_specs]
            out_placements = [s.placements for s in spec.output_specs]

            # All four inputs must be Shard(0).
            for i, p in enumerate(in_placements):
                assert p == (Shard(0),), f"input {i} placement {p} != (Shard(0),)"

            # out: Shard(0); lse [Hq, B*S] sharded on dim 1; rng replicated.
            assert out_placements[0] == (Shard(0),), out_placements[0]
            assert out_placements[1] == (Shard(1),), out_placements[1]
            assert out_placements[2] == (Replicate(),), out_placements[2]

    def test_dp_tp_pins_shard0_shard2(self, device_mesh_2d):
        """2D dp×tp mesh, q/k/v pinned (Shard(0), Shard(2)).

        cu_seq_q has no head dim so it's (Shard(0), Replicate()). Output
        inherits q's placement. lse rearranges to (Shard(1), Shard(0))
        because T == B*S occupies dim 1 and Hq occupies dim 0.
        """
        # Mesh is (32, 8): dp=32 must divide B; tp=8 must divide Hq and Hkv.
        dp_size, tp_size = device_mesh_2d.shape
        B, S, Hq, Hkv, D, n_docs = 32, 256, 8, 8, 64, 4  # Hq=Hkv=tp_size
        if Hq % tp_size != 0 or Hkv % tp_size != 0:
            pytest.skip(f"tp_size={tp_size} doesn't divide Hq={Hq} or Hkv={Hkv}")
        if B % dp_size != 0:
            pytest.skip(f"dp_size={dp_size} doesn't divide B={B}")

        model = self._JustAttn(n_docs).to(dtype=torch.bfloat16)
        input_fn = self._input_fn_factory(B, S, Hq, Hkv, D, n_docs)

        with AutoParallel(
            model, input_fn, device_mesh_2d, repeated_subgraphs=False
        ) as autop:
            autop.add_input_constraints(
                [
                    (Shard(0), Shard(2)),
                    (Shard(0), Shard(2)),
                    (Shard(0), Shard(2)),
                    (Shard(0), Replicate()),
                ]
            )
            autop.add_output_constraints([(Shard(0), Shard(2))])
            sol = autop.optimize_placement()

            node = self._find_doc_packed_node(autop.gm)
            spec = sol[node]
            in_placements = [s.placements for s in spec.input_specs]
            out_placements = [s.placements for s in spec.output_specs]

            assert in_placements[0] == (Shard(0), Shard(2)), in_placements[0]  # q
            assert in_placements[1] == (Shard(0), Shard(2)), in_placements[1]  # k
            assert in_placements[2] == (Shard(0), Shard(2)), in_placements[2]  # v
            assert in_placements[3] == (Shard(0), Replicate()), in_placements[3]  # cu

            assert out_placements[0] == (Shard(0), Shard(2)), out_placements[0]  # out
            assert out_placements[1] == (Shard(1), Shard(0)), out_placements[1]  # lse
            assert out_placements[2] == (Replicate(), Replicate()), out_placements[2]

    def test_apply_placement_succeeds(self, device_mesh_1d):
        """End-to-end smoke test that apply_placement produces a runnable module."""
        B, S, Hq, Hkv, D, n_docs = 256, 256, 8, 4, 64, 4
        model = self._JustAttn(n_docs).to(dtype=torch.bfloat16)
        input_fn = self._input_fn_factory(B, S, Hq, Hkv, D, n_docs)

        with AutoParallel(
            model, input_fn, device_mesh_1d, repeated_subgraphs=False
        ) as autop:
            autop.add_input_constraints(
                [(Shard(0),), (Shard(0),), (Shard(0),), (Shard(0),)]
            )
            autop.add_output_constraints([(Shard(0),)])
            sol = autop.optimize_placement()
            parallel_mod = autop.apply_placement(sol)
        assert parallel_mod is not None
