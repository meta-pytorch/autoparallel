# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
from unittest.mock import patch

import torch
from conftest import apply_cuda_patches
from torch import nn
from torch._functorch._aot_autograd.fx_utils import (
    get_plain_input_and_grad_nodes,
    get_plain_output_and_tangent_nodes,
)
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel import UNCONSTRAINED
from autoparallel.api import AutoParallel


class BoundaryModel(nn.Module):
    def forward(self, default, replicated, free, enabled):
        result = default + replicated + free
        if enabled:
            result = result + 1
        return result, free + 1, None


@apply_cuda_patches
def test_boundary_constraint_states(device_mesh_1d):
    with torch.device("meta"):
        model = BoundaryModel()

    def input_fn():
        return (
            torch.randn(512, 8, device="cuda"),
            torch.randn(512, 8, device="cuda"),
            torch.randn(8, device="cuda"),
            True,
        )

    with AutoParallel(
        model, input_fn, device_mesh_1d, repeated_subgraphs=False
    ) as autop:
        autop.add_input_constraints([None, (Replicate(),), UNCONSTRAINED, None])
        autop.add_output_constraints([None, UNCONSTRAINED, None])

        optimizer = autop.sharding_optimizer
        input_nodes = get_plain_input_and_grad_nodes(optimizer.graph)
        output_nodes = get_plain_output_and_tangent_nodes(optimizer.graph)
        constrained_names = set(optimizer._node_constraint_names.values())
        assert (
            input_nodes[next(desc for desc in input_nodes if desc.idx == 2)][0].name
            not in constrained_names
        )
        assert (
            output_nodes[next(desc for desc in output_nodes if desc.idx == 1)][0].name
            not in constrained_names
        )

        solution = autop.optimize_placement()
        default_input = input_nodes[
            next(desc for desc in input_nodes if desc.idx == 0)
        ][0]
        replicated_input = input_nodes[
            next(desc for desc in input_nodes if desc.idx == 1)
        ][0]
        default_output = output_nodes[
            next(desc for desc in output_nodes if desc.idx == 0)
        ][0]
        assert solution[
            optimizer._concrete_to_orig[default_input]
        ].output_specs.placements == (Shard(0),)
        assert solution[
            optimizer._concrete_to_orig[replicated_input]
        ].output_specs.placements == (Replicate(),)
        assert solution[
            optimizer._concrete_to_orig[default_output]
        ].output_specs.placements == (Shard(0),)

        with tempfile.NamedTemporaryFile(suffix=".ap") as saved:
            optimizer.save(saved.name)
            loaded = type(optimizer).load(saved.name)
        input_log = next(
            kwargs
            for name, kwargs in loaded._constraint_log
            if name == "add_sharded_input_constraint"
        )
        assert input_log["input_placements"][2] is UNCONSTRAINED

        markers = {
            node.name: node.meta["is_tensor_value"] for node in loaded.graph.nodes
        }
        with patch("torch.save") as save:
            loaded.save("unused.ap")
        resaved_graph = save.call_args.args[0]["graph"]
        assert {
            node.name: node.meta["is_tensor_value"] for node in resaved_graph.nodes
        } == markers

        autop.apply_placement(solution)
