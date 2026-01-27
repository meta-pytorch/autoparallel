# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from autoparallel.ops import permutation


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
