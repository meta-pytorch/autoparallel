# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch


def permutation(x: torch.Tensor, axis: int = 0, independent: bool = False):
    """Randomly permute elements of a tensor along an axis.

    Similar to jax.random.permutation, this function returns a tensor with
    elements shuffled along the specified axis.

    Args:
        x: Input tensor to permute.
        axis: The axis along which to permute. Defaults to 0.
        independent: If False (default), applies the same random permutation
            to all slices along the axis (like shuffling rows of a matrix
            together). If True, generates independent random permutations
            for each slice, meaning each position along other dimensions
            gets its own random ordering.

    Returns:
        A tensor with the same shape as x, with elements permuted along
        the specified axis.

    Examples:
        >>> x = torch.arange(12).reshape(3, 4)
        >>> # Shuffle rows (axis=0), same permutation for all columns
        >>> permutation(x, axis=0, independent=False)
        >>> # Shuffle rows independently for each column
        >>> permutation(x, axis=0, independent=True)
    """
    if independent is False:
        idxs = torch.randperm(x.shape[axis], device=x.device)
        return x.index_select(axis, idxs)

    # generate random permutation matrix which is independent per axis
    idxs = torch.rand_like(x, dtype=torch.float32).argsort(axis)
    return x.gather(axis, idxs)
