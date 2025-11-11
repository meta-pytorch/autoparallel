# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for activation checkpointing functionality.
"""

import pytest
import torch
from torch.utils.checkpoint import CheckpointPolicy

from autoparallel._testing.models.llama3 import Transformer, TransformerModelArgs
from autoparallel.activation_checkpointing import _apply_ac_policy


@pytest.fixture(scope="module")
def llama3_model():
    """Create a small Llama3 model for testing."""
    torch.manual_seed(1999)
    model_args = TransformerModelArgs(
        dim=64, n_layers=2, n_heads=4, vocab_size=256, rope_theta=500000
    )
    return Transformer(model_args)


def create_joint_graph_from_model(model, input_args):
    """Create a joint graph from a model for testing activation checkpointing functions."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx

    def simple_fwd_fn(*inputs):
        return model(*inputs)

    # Create fake tensor mode with consistent device handling
    with FakeTensorMode(allow_non_fake_inputs=True) as fake_mode:
        # Create fake inputs that match the input structure
        fake_input_args = tuple(fake_mode.from_tensor(arg) for arg in input_args)

        # Create a simple forward graph first
        fwd_graph = make_fx(simple_fwd_fn)(*fake_input_args)

        # Create a mock joint graph with forward and backward sections
        joint_graph = torch.fx.Graph()

        # Copy forward nodes
        value_remap = {}
        for node in fwd_graph.graph.nodes:
            if node.op == "placeholder":
                new_node = joint_graph.placeholder(node.target)
                new_node.meta.update(node.meta)
                value_remap[node] = new_node
            elif node.op == "call_function":
                new_args = tuple(value_remap.get(arg, arg) for arg in node.args)
                new_node = joint_graph.call_function(node.target, new_args, node.kwargs)
                new_node.meta.update(node.meta)
                value_remap[node] = new_node
            elif node.op == "output":
                # Add backward nodes just manually for testing purpose(marked as backward)
                output_node = value_remap[node.args[0]]

                # Add a sum operation for loss
                sum_node = joint_graph.call_function(
                    torch.ops.aten.sum.default, (output_node,)
                )
                sum_node.meta["val"] = torch.tensor(1.0)

                # Add backward nodes
                bw_node = joint_graph.call_function(
                    torch.ops.aten.mul.Tensor, (sum_node, 1.0)
                )
                bw_node.meta["partitioner_tag"] = "is_backward"
                bw_node.meta["val"] = torch.tensor(1.0)

                # Add tangent placeholder
                tangent_node = joint_graph.placeholder("tangents_1")
                tangent_node.meta["val"] = output_node.meta.get(
                    "val", torch.randn(2, 8, 64)
                )

                # Create output
                joint_graph.output([output_node, bw_node])
                break

        return joint_graph


def create_joint_graph_llama3(llama3_model):
    """Create a joint graph from Llama3 model."""
    batch_size = 2
    seq_len = 8
    vocab_size = llama3_model.model_args.vocab_size

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    return create_joint_graph_from_model(llama3_model, (input_ids,))


class TestACPolicy:
    """Test AC policy application."""

    def test_apply_ac_policy(self, llama3_model):
        """Test _apply_ac_policy function."""
        graph = create_joint_graph_llama3(llama3_model)

        # Define save list with operations that might be in the graph
        save_list = {
            torch.ops.aten.mm.default,
            torch.ops.aten.addmm.default,
        }

        _apply_ac_policy(graph, save_list)

        marked_nodes_to_save = [
            node
            for node in graph.nodes
            if node.meta.get("recompute") == CheckpointPolicy.MUST_SAVE
        ]

        # Count total mm.default nodes in the graph to verify every-other-node policy
        total_mm_nodes = len(
            [node for node in graph.nodes if node.target == torch.ops.aten.mm.default]
        )

        # The policy should save every other mm.default node
        expected_saved_nodes = (
            total_mm_nodes + 1
        ) // 2  # ceiling division for odd counts

        # Verify the every-other-node policy is working correctly
        assert (
            len(marked_nodes_to_save) == expected_saved_nodes
        ), f"Expected {expected_saved_nodes} nodes to be saved, but got {len(marked_nodes_to_save)}"
