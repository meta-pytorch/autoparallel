# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from torch.distributed.tensor._dtensor_spec import ShardOrderEntry

from autoparallel.apply_sharding import _compute_shard_order


class TestComputeShardOrder:
    def test_sorted_order(self):
        shard_order = (ShardOrderEntry(tensor_dim=0, mesh_dims=(2, 0, 1)),)
        result = _compute_shard_order(shard_order, reverse=False)
        assert result == (ShardOrderEntry(tensor_dim=0, mesh_dims=(0, 1, 2)),)

    def test_reversed_order(self):
        shard_order = (ShardOrderEntry(tensor_dim=0, mesh_dims=(2, 0, 1)),)
        result = _compute_shard_order(shard_order, reverse=True)
        assert result == (ShardOrderEntry(tensor_dim=0, mesh_dims=(2, 1, 0)),)

    def test_multiple_entries(self):
        shard_order = (
            ShardOrderEntry(tensor_dim=0, mesh_dims=(3, 1)),
            ShardOrderEntry(tensor_dim=1, mesh_dims=(2, 0)),
        )
        result = _compute_shard_order(shard_order, reverse=False)
        assert result == (
            ShardOrderEntry(tensor_dim=0, mesh_dims=(1, 3)),
            ShardOrderEntry(tensor_dim=1, mesh_dims=(0, 2)),
        )

    def test_already_sorted_is_idempotent(self):
        shard_order = (ShardOrderEntry(tensor_dim=0, mesh_dims=(0, 1, 2)),)
        result = _compute_shard_order(shard_order, reverse=False)
        assert result == shard_order

    def test_empty(self):
        assert _compute_shard_order((), reverse=False) == ()
        assert _compute_shard_order((), reverse=True) == ()
