# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
from conftest import apply_cuda_patches
from torch import nn
from torch.distributed.tensor.experimental import register_sharding
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel.api import AutoParallel

_LIB = torch.library.Library("autoparallel_test", "DEF")
_LIB.define("fragile(Tensor x) -> Tensor")


@torch.library.impl(_LIB, "fragile", "CompositeExplicitAutograd")
def _fragile(x):
    return x.clone()


@torch.library.register_fake("autoparallel_test::fragile")
def _fragile_fake(x):
    # A kernel-validity check that only some shardings violate (as _grouped_mm's 16-byte
    # stride alignment does).
    torch._check(x.shape[0] % 16 == 0, lambda: "rows must be a multiple of 16")
    return torch.empty_like(x)


@register_sharding(torch.ops.autoparallel_test.fragile.default)
def _fragile_sharding(x):
    return [([Shard(0)], [Shard(0)]), ([Replicate()], [Replicate()])]


class Model(nn.Module):
    def forward(self, x):
        return torch.ops.autoparallel_test.fragile(x) * 2


@apply_cuda_patches
def test_a_strategy_whose_cost_estimation_raises_is_dropped(device_mesh_1d):
    """Each of the 256 ranks would hold 2 rows, which the op refuses: ``Shard(0)`` is not an
    available strategy, ``Replicate`` is, and the solve completes with it."""
    mesh = device_mesh_1d

    def input_fn():
        return torch.rand(512, 8, device="cuda")

    with torch.device("meta"):
        model = Model()
    with AutoParallel(model, input_fn, mesh) as autop:
        autop.add_input_constraints([(Replicate(),)])
        autop.add_output_constraints([(Replicate(),)])
        autop.optimize_placement()
        strats = autop.sharding_optimizer.strats
        (node,) = [
            n
            for n in strats
            if getattr(n, "target", None) is torch.ops.autoparallel_test.fragile.default
        ]
        outputs = {s.output_specs.placements for s in strats[node].strategies}
        assert outputs == {(Replicate(),)}
