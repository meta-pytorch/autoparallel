# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch.distributed.tensor._dtensor_spec import ShardOrderEntry
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from autoparallel.apply_sharding import (
    _compute_shard_order,
    _filter_specs_for_local_map,
)


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


def _make_symint(val: int) -> torch.SymInt:
    shape_env = ShapeEnv()
    from torch._dynamo.source import ConstantSource

    sym = shape_env.create_symbol(val, source=ConstantSource(source_name=f"s{val}"))
    return shape_env.create_symintnode(sym, hint=val)


class TestFilterSpecsForLocalMap:
    def test_tensors_only(self):
        flat_args = [torch.tensor(1.0), torch.tensor(2.0)]
        curr_specs = ["spec_a", "spec_b"]
        tgt_specs = ["spec_c", "spec_d"]
        c, t = _filter_specs_for_local_map(flat_args, curr_specs, tgt_specs)
        assert c == ["spec_a", "spec_b"]
        assert t == ["spec_c", "spec_d"]

    def test_mixed_tensor_and_symint(self):
        s = _make_symint(3)
        flat_args = [torch.tensor(1.0), s, torch.tensor(2.0)]
        # specs only have entries for tensor args (SymInts are excluded)
        curr_specs = ["spec_a", "spec_b"]
        tgt_specs = ["spec_c", "spec_d"]
        c, t = _filter_specs_for_local_map(flat_args, curr_specs, tgt_specs)
        assert c == ["spec_a", "spec_b"]
        assert t == ["spec_c", "spec_d"]

    def test_symint_with_non_none_spec_raises(self):
        s = _make_symint(3)
        flat_args = [s]
        # No specs for SymInt-only args — passes with empty specs
        c, t = _filter_specs_for_local_map(flat_args, [], [])
        assert c == []
        assert t == []

    def test_unexpected_type_raises(self):
        flat_args = ["unexpected_string"]
        with pytest.raises(ValueError, match="Unexpected local_map HOP argument"):
            _filter_specs_for_local_map(flat_args, [], [])
