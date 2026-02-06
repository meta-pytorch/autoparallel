# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import functools

import pytest
import torch
from torch import nn
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

from autoparallel.api import AutoParallel


@pytest.fixture(scope="module", autouse=True)
def init_pg():
    world_size = 32
    fake_store = FakeStore()
    if torch.distributed.is_initialized():
        return
    torch.distributed.init_process_group(
        "fake", store=fake_store, rank=0, world_size=world_size
    )


@pytest.fixture(scope="module")
def device_mesh_1d():
    world_size = torch.distributed.get_world_size()
    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda", (world_size,), mesh_dim_names=("dp",)
    )
    return mesh


def _custom_policy_fn(ctx, op, *args, **kwargs):
    """Custom policy that saves mm ops and recomputes attention ops."""
    if op in (
        torch.ops.aten._scaled_dot_product_flash_attention.default,
        torch.ops.aten._scaled_dot_product_efficient_attention.default,
        torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    ):
        return CheckpointPolicy.MUST_RECOMPUTE
    return CheckpointPolicy.MUST_SAVE


class AttentionBlock(nn.Module):
    """Simple attention block for testing activation checkpointing."""

    def __init__(self, nheads, dim):
        super().__init__()
        self.nheads = nheads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def _compute_attention(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
        k = k.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
        v = v.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)

        o = nn.functional.scaled_dot_product_attention(q, k, v)
        o = o.permute(0, 2, 1, 3).flatten(-2)

        o = self.wo(o)
        return o

    def forward(self, x, context_fn=None):
        if context_fn is not None:
            o = torch.utils.checkpoint.checkpoint(
                self._compute_attention, x, use_reentrant=False, context_fn=context_fn
            )
        else:
            # Basic checkpoint without custom policy
            o = torch.utils.checkpoint.checkpoint(
                self._compute_attention, x, use_reentrant=False
            )
        return o + x


class AttentionBlockWithCustomPolicy(nn.Module):
    """Attention block that uses custom checkpoint policy."""

    def __init__(self, nheads, dim):
        super().__init__()
        self.block = AttentionBlock(nheads, dim)
        self.context_fn = functools.partial(
            create_selective_checkpoint_contexts, _custom_policy_fn
        )

    def forward(self, x):
        return self.block(x, context_fn=self.context_fn)


class AttentionBlockWithBasicCheckpoint(nn.Module):
    """Attention block that uses basic checkpoint (no custom policy)."""

    def __init__(self, nheads, dim):
        super().__init__()
        self.block = AttentionBlock(nheads, dim)

    def forward(self, x):
        return self.block(x, context_fn=None)


def _get_checkpoint_tagged_nodes(graph, include_backward=False):
    """Get nodes that have checkpoint-related metadata."""
    tagged_nodes = []
    for node in graph.nodes:
        if "checkpoint" in node.meta.get("stack_trace", ""):
            is_bwd = node.meta.get("partitioner_tag", "") == "is_backward"
            if include_backward or not is_bwd:
                tagged_nodes.append(node)
    return tagged_nodes


def _validate_checkpoint_tags_custom_policy(graph, policy_fn):
    """Validate that nodes have the expected checkpoint tags based on custom policy."""
    tagged_nodes = _get_checkpoint_tagged_nodes(graph, include_backward=False)
    assert len(tagged_nodes) > 0, "Expected some nodes with checkpoint tags"

    for node in tagged_nodes:
        if "getitem" in str(node.target):
            # getitem nodes inherit tag from parent
            expected = policy_fn(None, node.args[0].target, (), ())
        elif "alias" in str(node.target) and "getitem" in str(node.args[0].target):
            # alias nodes that depend on getitem inherit from grandparent
            expected = policy_fn(None, node.args[0].args[0].target, (), ())
        else:
            expected = policy_fn(None, node.target, (), ())

        actual = node.meta.get("recompute")
        assert actual == expected, f"Node {node}: expected {expected}, got {actual}"

    return tagged_nodes


def _validate_mm_has_checkpoint_tag(graph):
    """Validate that mm nodes have checkpoint policy set."""
    mm_nodes = graph.find_nodes(op="call_function", target=torch.ops.aten.mm.default)
    assert len(mm_nodes) > 0, "Expected at least one mm node"

    # Check that at least one mm node has a checkpoint policy
    has_policy = any(node.meta.get("recompute") is not None for node in mm_nodes)
    assert has_policy, "Expected at least one mm node with checkpoint policy"
    return mm_nodes


def test_ac_custom_policy_checkpoint_tags(device_mesh_1d):
    """Test that custom policy checkpoint tags are preserved through AutoParallel."""
    nheads = 8
    dim = 128
    bs = 32
    seq_len = 64

    def input_fn():
        return torch.rand(bs, seq_len, dim, device="cuda")

    with torch.device("meta"):
        model = AttentionBlockWithCustomPolicy(nheads, dim)

    with AutoParallel(model, input_fn, device_mesh_1d, compile=False) as autop:
        x_sharding = (Shard(0),)
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([x_sharding])

        sharding_placement = autop.optimize_placement()
        parallel_mod = autop.apply_placement(sharding_placement)

        # Validate checkpoint tags in the graph
        graph = autop.parallel_gm.graph
        _validate_checkpoint_tags_custom_policy(graph, _custom_policy_fn)
        _validate_mm_has_checkpoint_tag(graph)

    assert parallel_mod is not None


def test_ac_basic_checkpoint_tags(device_mesh_1d):
    """Test that basic checkpoint (no custom policy) works with AutoParallel."""
    nheads = 8
    dim = 128
    bs = 32
    seq_len = 64

    def input_fn():
        return torch.rand(bs, seq_len, dim, device="cuda")

    with torch.device("meta"):
        model = AttentionBlockWithBasicCheckpoint(nheads, dim)

    with AutoParallel(model, input_fn, device_mesh_1d, compile=False) as autop:
        x_sharding = (Shard(0),)
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([x_sharding])

        sharding_placement = autop.optimize_placement()
        parallel_mod = autop.apply_placement(sharding_placement)

        # For basic checkpoint, nodes should still have stack_trace with "checkpoint"
        graph = autop.parallel_gm.graph
        tagged_nodes = _get_checkpoint_tagged_nodes(graph)
        assert (
            len(tagged_nodes) > 0
        ), "Expected some nodes with checkpoint in stack_trace"

    assert parallel_mod is not None


def test_ac_torch_compile_preserves_custom_policy_tags(device_mesh_1d):
    """Test that torch.compile preserves checkpoint tags from custom policy."""
    nheads = 8
    dim = 128
    bs = 32
    seq_len = 64

    def input_fn():
        return torch.rand(bs, seq_len, dim, device="cuda")

    with torch.device("meta"):
        model = AttentionBlockWithCustomPolicy(nheads, dim)

    captured_graphs = []

    def boxed_nop_preserve_node_meta(gm: torch.fx.GraphModule, example_inputs):
        def run(args):
            with torch.fx.traceback.preserve_node_meta():
                # nonlocal captured_graphs
                # captured_graphs.append(gm)
                return torch.fx.Interpreter(gm).boxed_run(args)

        run._boxed_call = True  # type: ignore[attr-defined]
        return run

    # First, apply autoparallel without compile to check tags
    with AutoParallel(model, input_fn, device_mesh_1d, compile=False) as autop:
        x_sharding = (Shard(0),)
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([x_sharding])

        autop.compiler_fn = boxed_nop_preserve_node_meta

        sharding_placement = autop.optimize_placement()
        parallel_mod = autop.apply_placement(sharding_placement)

        # Validate checkpoint tags before compile
        graph = autop.parallel_gm.graph
        pre_compile_tagged = _validate_checkpoint_tags_custom_policy(
            graph, _custom_policy_fn
        )
        pre_compile_mm = _validate_mm_has_checkpoint_tag(graph)

    def capture_backend(gm, example_inputs):
        """Backend that captures the graph for inspection."""
        # captured_graphs.append(gm)
        # Return the graph module as-is (eager execution)
        return gm

    # Initialize the model and run it to trigger compilation
    parallel_mod.to_empty(device="cuda")
    # Initialize weights with simple values
    for p in parallel_mod.parameters():
        p.data.fill_(0.01)

    local_bs = bs // device_mesh_1d.size()
    x = torch.rand(local_bs, seq_len, dim, device="cuda")
    _ = parallel_mod(x)

    # Apply torch.compile with our capturing backend
    # compiled_mod = torch.compile(parallel_mod, backend=capture_backend)
    compiled_mod = torch.compile(parallel_mod)

    # Run the compiled model to trigger compilation
    _ = compiled_mod(x)

    # Verify we captured a graph
    assert len(captured_graphs) > 0, "Expected to capture at least one graph"

    # Check that the captured graph has nodes with checkpoint-related metadata
    # The compiled graph should preserve the recompute tags with custom policy values
    compiled_graph = captured_graphs[0].graph
    compiled_tagged = []
    compiled_mm_with_tags = []
    for node in compiled_graph.nodes:
        # Check for recompute tag (the key metadata for checkpoint behavior)
        if node.meta.get("recompute") is not None:
            compiled_tagged.append(node)
            # Check if this is an mm node
            if node.target == torch.ops.aten.mm.default:
                compiled_mm_with_tags.append(node)

    # Verify that checkpoint tags are preserved in the compiled graph
    assert len(compiled_tagged) > 0, (
        "Expected nodes with 'recompute' metadata in compiled graph. "
        "This indicates checkpoint tags were not preserved through torch.compile."
    )

    # Verify mm nodes have the expected MUST_SAVE policy
    assert (
        len(compiled_mm_with_tags) > 0
    ), "Expected mm nodes with 'recompute' metadata in compiled graph."
    for mm_node in compiled_mm_with_tags:
        assert (
            mm_node.meta.get("recompute") == CheckpointPolicy.MUST_SAVE
        ), f"mm node {mm_node} should have MUST_SAVE policy"

    assert compiled_mod is not None


def test_ac_torch_compile_preserves_basic_checkpoint_tags(device_mesh_1d):
    """Test that torch.compile preserves checkpoint info from basic checkpoint."""
    nheads = 8
    dim = 128
    bs = 32
    seq_len = 64

    def input_fn():
        return torch.rand(bs, seq_len, dim, device="cuda")

    with torch.device("meta"):
        model = AttentionBlockWithBasicCheckpoint(nheads, dim)

    with AutoParallel(model, input_fn, device_mesh_1d, compile=False) as autop:
        x_sharding = (Shard(0),)
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([x_sharding])

        sharding_placement = autop.optimize_placement()
        parallel_mod = autop.apply_placement(sharding_placement)

        graph = autop.parallel_gm.graph
        pre_compile_tagged = _get_checkpoint_tagged_nodes(graph)

    # Verify we have checkpoint-tagged nodes before compile
    assert (
        len(pre_compile_tagged) > 0
    ), "Expected checkpoint-tagged nodes before compile"

    # Capture the graph that torch.compile sees using a custom backend
    captured_graphs = []

    def capture_backend(gm, example_inputs):
        """Backend that captures the graph for inspection."""
        captured_graphs.append(gm)
        # Return the graph module as-is (eager execution)
        return gm

    # Apply torch.compile with our capturing backend
    compiled_mod = torch.compile(parallel_mod, backend=capture_backend)

    # Initialize the model and run it to trigger compilation
    parallel_mod.to_empty(device="cuda")
    # Initialize weights with simple values
    for p in parallel_mod.parameters():
        p.data.fill_(0.01)

    # Run the compiled model to trigger compilation
    local_bs = bs // device_mesh_1d.size()
    x = torch.rand(local_bs, seq_len, dim, device="cuda")
    _ = compiled_mod(x)

    # Verify we captured a graph
    assert len(captured_graphs) > 0, "Expected to capture at least one graph"

    # Check that the captured graph has nodes with checkpoint-related metadata
    # The compiled graph should preserve the recompute tags
    compiled_graph = captured_graphs[0].graph
    compiled_tagged = []
    for node in compiled_graph.nodes:
        # Check for recompute tag (the key metadata for checkpoint behavior)
        if node.meta.get("recompute") is not None:
            compiled_tagged.append(node)

    # Verify that checkpoint tags are preserved in the compiled graph
    assert len(compiled_tagged) > 0, (
        "Expected nodes with 'recompute' metadata in compiled graph. "
        "This indicates checkpoint tags were not preserved through torch.compile."
    )

    assert compiled_mod is not None


def test_ac_compile_true_applies_torch_compile(device_mesh_1d):
    """Test that compile=True in AutoParallel applies torch.compile to the model."""
    nheads = 8
    dim = 128
    bs = 32
    seq_len = 64

    def input_fn():
        return torch.rand(bs, seq_len, dim, device="cuda")

    with torch.device("meta"):
        model = AttentionBlockWithCustomPolicy(nheads, dim)

    with AutoParallel(model, input_fn, device_mesh_1d, compile=True) as autop:
        x_sharding = (Shard(0),)
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([x_sharding])

        sharding_placement = autop.optimize_placement()
        parallel_mod = autop.apply_placement(sharding_placement)

        # Validate checkpoint tags in the graph before compile wrapping
        graph = autop.parallel_gm.graph
        _validate_checkpoint_tags_custom_policy(graph, _custom_policy_fn)
        _validate_mm_has_checkpoint_tag(graph)

    # The returned model should be wrapped with torch.compile
    # We can check this by looking at the type
    assert parallel_mod is not None
    # torch.compile returns an OptimizedModule
    assert (
        hasattr(parallel_mod, "_orig_mod") or "Optimized" in type(parallel_mod).__name__
    )


def test_ac_backward_nodes_have_matching_seq_nr(device_mesh_1d):
    """Test that backward nodes have seq_nr matching their forward counterparts."""
    nheads = 8
    dim = 128
    bs = 32
    seq_len = 64

    def input_fn():
        return torch.rand(bs, seq_len, dim, device="cuda")

    with torch.device("meta"):
        model = AttentionBlockWithCustomPolicy(nheads, dim)

    with AutoParallel(model, input_fn, device_mesh_1d, compile=False) as autop:
        x_sharding = (Shard(0),)
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([x_sharding])

        sharding_placement = autop.optimize_placement()
        _ = autop.apply_placement(sharding_placement)

        graph = autop.parallel_gm.graph

        # Collect seq_nr from forward checkpoint nodes
        fwd_seq_nrs = set()
        for node in graph.nodes:
            if "checkpoint" in node.meta.get("stack_trace", ""):
                is_bwd = node.meta.get("partitioner_tag", "") == "is_backward"
                if not is_bwd:
                    if "seq_nr" in node.meta:
                        fwd_seq_nrs.add(node.meta["seq_nr"])

        # Verify backward nodes have matching seq_nr
        for node in graph.nodes:
            if "checkpoint" in node.meta.get("stack_trace", ""):
                is_bwd = node.meta.get("partitioner_tag", "") == "is_backward"
                if is_bwd and "seq_nr" in node.meta:
                    assert node.meta["seq_nr"] in fwd_seq_nrs, (
                        f"Backward node {node} has seq_nr {node.meta['seq_nr']} "
                        f"not found in forward nodes"
                    )
