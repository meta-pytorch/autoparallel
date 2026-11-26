# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.distributed.tensor._dtensor_spec import (
    DTensorSpec,
    ShardOrderEntry,
    TensorMeta,
)
from torch.distributed.tensor.placement_types import Replicate, Shard, _StridedShard

from autoparallel.apply_sharding import ApplyShardingInterpreter


def test_redistribute_a_strided_shard_source(device_mesh_2d):
    """``redistribute_tensor`` with a ``_StridedShard`` spec.

    A spec holding a ``_StridedShard`` encodes its shard order in its placements, and
    ``DTensorSpec`` rejects an explicit ``shard_order`` next to one. The interpreter used to
    pass the shard order it computed anyway, so redistributing any strided layout raised
    ``ValueError`` before a collective was issued.
    """
    mesh = device_mesh_2d
    rows = mesh.size()
    meta = TensorMeta(torch.Size([rows, 4]), (4, 1), torch.float32)
    # Strided placements that also carry the shard order their strategy computed, as the
    # solver's specs do.
    source = DTensorSpec(
        mesh,
        (_StridedShard(0, split_factor=mesh.shape[1]), Shard(0)),
        tensor_meta=meta,
        shard_order=(ShardOrderEntry(tensor_dim=0, mesh_dims=(0, 1)),),
        use_strided_shard_as_shard_order=False,
    )
    target = DTensorSpec(mesh, (Replicate(), Replicate()), tensor_meta=meta)

    interpreter = ApplyShardingInterpreter.__new__(ApplyShardingInterpreter)
    interpreter.param_placement_order = {}
    out = interpreter.redistribute_tensor(torch.randn(1, 4), source, target, node=None)

    # Replicated from a mesh.size()-way shard of dim 0: every row.
    assert out.shape == (rows, 4)
