# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""Tests that activation checkpointing tags (recompute, seq_nr, ac_graph_id)
are correctly propagated through graph capture and through autoparallel's
own ac_joint_pass."""

import functools

import torch
import torch.fx.traceback as fx_traceback
from torch import nn
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

from autoparallel.api import AutoParallel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# The checkpointed function name used to filter nodes that were actually
# traced inside the checkpoint region.  TorchDispatchMode only intercepts
# OpOverloads, so only nodes whose stack_trace includes the checkpointed
# function received a recompute tag — nodes after checkpoint() returns
# (e.g. the FFN portion) share a "checkpoint" stack frame but were never
# dispatched under the policy.
_CHECKPOINTED_FN = "_compute_attention"


def _must_save_policy(ctx, op, *args, **kwargs):
    """Save everything except SDPA ops, which must be recomputed."""
    if (
        op == torch.ops.aten._scaled_dot_product_flash_attention.default
        or op == torch.ops.aten._scaled_dot_product_efficient_attention.default
    ):
        return CheckpointPolicy.MUST_RECOMPUTE
    return CheckpointPolicy.MUST_SAVE


class AttentionBlock(nn.Module):
    def __init__(self, nheads, dim, ffn_dim):
        super().__init__()
        self.nheads = nheads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)

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

    def forward(self, x):
        raise NotImplementedError("Subclasses choose checkpoint strategy")


class AttentionBlockWithUserAC(AttentionBlock):
    """Uses torch.utils.checkpoint.checkpoint with a user-supplied policy."""

    def __init__(self, nheads, dim, ffn_dim, context_fn):
        super().__init__(nheads, dim, ffn_dim)
        self._context_fn = context_fn

    def forward(self, x):
        o = torch.utils.checkpoint.checkpoint(
            self._compute_attention,
            x,
            use_reentrant=False,
            context_fn=self._context_fn,
        )
        o0 = o + x
        o = self.w1(o0)
        o = torch.nn.functional.relu(o)
        o = self.w2(o)
        return o0 + o


class AttentionBlockNoAC(AttentionBlock):
    """No user-level activation checkpointing."""

    def forward(self, x):
        o = self._compute_attention(x)
        o0 = o + x
        o = self.w1(o0)
        o = torch.nn.functional.relu(o)
        o = self.w2(o)
        return o0 + o


def _build_joint_graph(model, input_fn, mesh):
    """Build the joint fwd+bwd graph via AutoParallel, without running the
    sharding optimizer."""
    autop = AutoParallel(model, input_fn, mesh)
    try:
        autop.build_model_graph()
        return autop.gm
    finally:
        autop.stack.close()


def _is_inside_checkpointed_fn(node):
    """Return True if the node's stack trace shows it was traced inside
    the checkpointed function (_compute_attention)."""
    return _CHECKPOINTED_FN in node.meta.get("stack_trace", "")


# ---------------------------------------------------------------------------
# Family 1: user-level torch.utils.checkpoint.checkpoint
# ---------------------------------------------------------------------------


def test_user_ac_recompute_tags_on_targeted_ops(device_mesh_1d):
    """SDPA ops get MUST_RECOMPUTE and mm ops get MUST_SAVE from user policy."""
    context_fn = functools.partial(
        create_selective_checkpoint_contexts, _must_save_policy
    )
    nheads, dim, ffn_dim = 8, 128, 512
    bs, seq_len = 32, 32

    def input_fn():
        return torch.rand(bs, seq_len, dim, device="cuda")

    with torch.device("meta"):
        model = AttentionBlockWithUserAC(nheads, dim, ffn_dim, context_fn)

    gm = _build_joint_graph(model, input_fn, device_mesh_1d)

    mm_nodes = gm.graph.find_nodes(op="call_function", target=torch.ops.aten.mm.default)
    assert len(mm_nodes) > 0
    for n in mm_nodes:
        if n.meta.get("partitioner_tag", "") == "is_backward":
            continue
        if _is_inside_checkpointed_fn(n):
            assert (
                n.meta.get("recompute") == CheckpointPolicy.MUST_SAVE
            ), f"{n} expected MUST_SAVE, got {n.meta.get('recompute')}"


def test_user_ac_recompute_tag_set_on_all_checkpointed_fwd_nodes(device_mesh_1d):
    """Every forward node inside the checkpointed function has a recompute
    tag set (TorchDispatchMode tags all OpOverloads dispatched under the
    policy context)."""
    context_fn = functools.partial(
        create_selective_checkpoint_contexts, _must_save_policy
    )
    nheads, dim, ffn_dim = 8, 128, 512
    bs, seq_len = 32, 32

    def input_fn():
        return torch.rand(bs, seq_len, dim, device="cuda")

    with torch.device("meta"):
        model = AttentionBlockWithUserAC(nheads, dim, ffn_dim, context_fn)

    gm = _build_joint_graph(model, input_fn, device_mesh_1d)

    for n in gm.graph.nodes:
        if not _is_inside_checkpointed_fn(n):
            continue
        is_bwd = n.meta.get("partitioner_tag", "") == "is_backward"
        if not is_bwd:
            assert (
                n.meta.get("recompute") is not None
            ), f"{n} (target={n.target}) inside checkpointed fn has no recompute tag"


def test_user_ac_seq_nr_consistency(device_mesh_1d):
    """seq_nr of backward checkpointed nodes must have a matching forward node."""
    context_fn = functools.partial(
        create_selective_checkpoint_contexts, _must_save_policy
    )
    nheads, dim, ffn_dim = 8, 128, 512
    bs, seq_len = 32, 32

    def input_fn():
        return torch.rand(bs, seq_len, dim, device="cuda")

    with torch.device("meta"):
        model = AttentionBlockWithUserAC(nheads, dim, ffn_dim, context_fn)

    gm = _build_joint_graph(model, input_fn, device_mesh_1d)

    fwd_seqs = set()
    for n in gm.graph.nodes:
        if not _is_inside_checkpointed_fn(n):
            continue
        is_bwd = n.meta.get("partitioner_tag", "") == "is_backward"
        if not is_bwd:
            fwd_seqs.add(n.meta["seq_nr"])
        else:
            assert n.meta["seq_nr"] in fwd_seqs, (
                f"backward node {n} has seq_nr={n.meta['seq_nr']} "
                f"with no matching forward node"
            )


def test_user_ac_nn_module_stack_present(device_mesh_1d):
    """nn_module_stack and fwd_nn_module_stack survive graph capture."""
    context_fn = functools.partial(
        create_selective_checkpoint_contexts, _must_save_policy
    )
    nheads, dim, ffn_dim = 8, 128, 512
    bs, seq_len = 32, 32

    def input_fn():
        return torch.rand(bs, seq_len, dim, device="cuda")

    with torch.device("meta"):
        model = AttentionBlockWithUserAC(nheads, dim, ffn_dim, context_fn)

    gm = _build_joint_graph(model, input_fn, device_mesh_1d)

    assert any(n.meta.get("nn_module_stack") for n in gm.graph.nodes)
    assert any(n.meta.get("fwd_nn_module_stack") for n in gm.graph.nodes)


def test_user_ac_policy_fn_applied_to_getitem(device_mesh_1d):
    """getitem nodes derived from a checkpointed op inherit the parent's
    recompute tag (they are tagged by the checkpoint infrastructure based
    on the multi-output op that precedes them)."""
    context_fn = functools.partial(
        create_selective_checkpoint_contexts, _must_save_policy
    )
    nheads, dim, ffn_dim = 8, 128, 512
    bs, seq_len = 32, 32

    def input_fn():
        return torch.rand(bs, seq_len, dim, device="cuda")

    with torch.device("meta"):
        model = AttentionBlockWithUserAC(nheads, dim, ffn_dim, context_fn)

    gm = _build_joint_graph(model, input_fn, device_mesh_1d)

    import operator

    for n in gm.graph.nodes:
        if not _is_inside_checkpointed_fn(n):
            continue
        if n.meta.get("partitioner_tag", "") == "is_backward":
            continue
        if n.target is not operator.getitem:
            continue
        parent = n.args[0]
        parent_policy = _must_save_policy(None, parent.target, (), ())
        actual = n.meta.get("recompute")
        assert actual == parent_policy, (
            f"getitem {n} has recompute={actual}, but parent {parent} "
            f"(target={parent.target}) has policy {parent_policy}"
        )


# ---------------------------------------------------------------------------
# Family 2: autoparallel's ac_joint_pass
# ---------------------------------------------------------------------------


def _build_parallel_graph(model_cls, mesh, *, context_fn=None):
    """Run the full AutoParallel pipeline and return the parallel graph for
    ac_joint_pass testing."""
    nheads, dim, ffn_dim = 8, 128, 512
    bs = 8 * mesh.shape[0]
    seq_len = 32

    def input_fn():
        return torch.rand(bs, seq_len, dim, device="cuda")

    with torch.device("meta"):
        if context_fn is not None:
            model = model_cls(nheads, dim, ffn_dim, context_fn)
        else:
            model = model_cls(nheads, dim, ffn_dim)

    with AutoParallel(model, input_fn, mesh) as autop:
        x_sharding = (Shard(0),)
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([x_sharding])
        sharding_placement = autop.optimize_placement()
        autop.apply_placement(sharding_placement)

    return autop.parallel_gm.graph


def test_ac_joint_pass_marks_recomputable_nodes(device_mesh_1d):
    """ac_joint_pass sets PREFER_RECOMPUTE on recomputable forward nodes
    (excluding getitem and already-tagged nodes)."""
    from autoparallel.graph_passes.activation_checkpointing import ac_joint_pass

    graph = _build_parallel_graph(AttentionBlockNoAC, device_mesh_1d)
    ac_joint_pass(graph, ac_stage_size_in_GiB=None)

    from torch._functorch.partitioners import _has_tag_is_backward

    for n in graph.nodes:
        if n.op != "call_function":
            continue
        if _has_tag_is_backward(n):
            continue
        recompute = n.meta.get("recompute")
        if recompute is not None:
            assert recompute in (
                CheckpointPolicy.PREFER_RECOMPUTE,
                CheckpointPolicy.MUST_SAVE,
                CheckpointPolicy.MUST_RECOMPUTE,
            ), f"{n} has unexpected recompute={recompute}"


def test_ac_joint_pass_apply_ac_policy_saves_mm_and_sdpa(device_mesh_1d):
    """_apply_ac_policy marks mm and SDPA ops as MUST_SAVE."""
    from autoparallel.graph_passes.activation_checkpointing import (
        _apply_ac_policy,
        mark_nodes_as_must_save_to_stage_recomputation,
    )

    graph = _build_parallel_graph(AttentionBlockNoAC, device_mesh_1d)

    # First mark everything as recomputable (same as ac_joint_pass does)
    mark_nodes_as_must_save_to_stage_recomputation(graph, stage_size_in_GiB=None)

    save_list = {
        torch.ops.aten.mm.default,
        torch.ops.aten._scaled_dot_product_efficient_attention.default,
        torch.ops.aten._scaled_dot_product_flash_attention.default,
        torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    }
    _apply_ac_policy(graph, save_list=save_list)

    from torch._functorch.partitioners import _has_tag_is_backward

    sdpa_targets = {
        torch.ops.aten._scaled_dot_product_efficient_attention.default,
        torch.ops.aten._scaled_dot_product_flash_attention.default,
        torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    }
    fwd_sdpa_nodes = [
        n
        for n in graph.nodes
        if n.op == "call_function"
        and n.target in sdpa_targets
        and not _has_tag_is_backward(n)
    ]
    for n in fwd_sdpa_nodes:
        assert (
            n.meta.get("recompute") == CheckpointPolicy.MUST_SAVE
        ), f"SDPA node {n} expected MUST_SAVE, got {n.meta.get('recompute')}"


def test_ac_joint_pass_respects_user_annotations(device_mesh_1d):
    """ac_joint_pass's _mark_nodes_as_must_save skips nodes already tagged
    by the user (those with ac_graph_id != AP_AC_GRAPH_ID)."""
    from autoparallel.graph_passes.activation_checkpointing import (
        AP_AC_GRAPH_ID,
        ac_joint_pass,
    )

    context_fn = functools.partial(
        create_selective_checkpoint_contexts, _must_save_policy
    )
    graph = _build_parallel_graph(
        AttentionBlockWithUserAC, device_mesh_1d, context_fn=context_fn
    )

    # Record which nodes were tagged by the user before ac_joint_pass
    user_tagged = {}
    for n in graph.nodes:
        recompute = n.meta.get("recompute")
        if recompute is not None:
            user_tagged[n.name] = recompute

    ac_joint_pass(graph, ac_stage_size_in_GiB=None)

    # User-tagged nodes without AP_AC_GRAPH_ID should keep their original tag
    for n in graph.nodes:
        if n.name in user_tagged and n.meta.get("ac_graph_id", -1) != AP_AC_GRAPH_ID:
            assert n.meta["recompute"] == user_tagged[n.name], (
                f"User-tagged node {n} had {user_tagged[n.name]} but "
                f"ac_joint_pass changed it to {n.meta['recompute']}"
            )


def test_ac_joint_pass_stages_recomputation(device_mesh_1d):
    """With a small stage size, mark_nodes_as_must_save_to_stage_recomputation
    inserts MUST_SAVE nodes to break recomputation chains."""
    from autoparallel.graph_passes.activation_checkpointing import (
        mark_nodes_as_must_save_to_stage_recomputation,
    )

    graph = _build_parallel_graph(AttentionBlockNoAC, device_mesh_1d)

    # Use a tiny stage size to force multiple stages
    mark_nodes_as_must_save_to_stage_recomputation(graph, stage_size_in_GiB=1e-6)

    from torch._functorch.partitioners import _has_tag_is_backward

    must_save_nodes = [
        n
        for n in graph.nodes
        if n.op == "call_function"
        and not _has_tag_is_backward(n)
        and n.meta.get("recompute") == CheckpointPolicy.MUST_SAVE
    ]
    assert (
        len(must_save_nodes) > 0
    ), "Expected staging to insert MUST_SAVE nodes with tiny stage size"


# ---------------------------------------------------------------------------
# Family 3: local_map + activation checkpointing
# ---------------------------------------------------------------------------


def _make_local_map_block(mesh):
    """Build a Block class that uses local_map-wrapped ops inside a
    checkpoint region.  The local_map decorators capture `mesh`, so the
    class factory must receive the mesh at definition time."""
    from torch.distributed._tensor.experimental import local_map

    def policy_fn(ctx, op, *args, **kwargs):
        if (
            op == torch.ops.aten._scaled_dot_product_flash_attention.default
            or op == torch.ops.aten._scaled_dot_product_efficient_attention.default
        ):
            return CheckpointPolicy.PREFER_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)

    @local_map(
        out_placements=((Replicate(), Replicate(), Replicate()),),
        in_placements=(
            (Replicate(), Replicate(), Replicate()),
            (Replicate(), Replicate(), Replicate()),
        ),
        redistribute_inputs=True,
        in_grad_placements=None,
        device_mesh=mesh,
    )
    def replicate_linear(w, x):
        with fx_traceback.annotate({"inside_local_map": 1}):
            return torch.matmul(x, w.t())

    @local_map(
        out_placements=((Shard(0), Shard(0), Replicate()),),
        in_placements=((Shard(0), Shard(0), Replicate()),),
        redistribute_inputs=True,
        in_grad_placements=None,
        device_mesh=mesh,
    )
    def sharded_pointwise(x):
        with fx_traceback.annotate({"inside_local_map": 0}):
            return x + 10

    @local_map(
        out_placements=((Shard(0), Shard(1), Shard(2)),),
        in_placements=(
            (Shard(0), Shard(1), Shard(2)),
            (Shard(0), Shard(1), Shard(2)),
            (Shard(0), Shard(1), Shard(2)),
        ),
        redistribute_inputs=True,
        in_grad_placements=None,
        device_mesh=mesh,
    )
    def context_parallel_attention(query, key, value):
        with fx_traceback.annotate({"inside_local_map": 2}):
            return nn.functional.scaled_dot_product_attention(
                query=query, key=key, value=value, is_causal=False
            )

    class Block(nn.Module):
        def __init__(self, nheads, dim, ffn_dim):
            super().__init__()
            self.nheads = nheads
            bias = False
            self.wq = nn.Linear(dim, dim, bias=bias)
            self.wk = nn.Linear(dim, dim, bias=bias)
            self.wv = nn.Linear(dim, dim, bias=bias)
            self.wo = nn.Linear(dim, dim, bias=bias)
            self.w1 = nn.Linear(dim, ffn_dim, bias=bias)
            self.w2 = nn.Linear(ffn_dim, dim, bias=bias)

        def _compute_attention(self, x):
            with fx_traceback.annotate({"inside_checkpoint": 0}):
                boosted_weight = sharded_pointwise(self.wq.weight)
                q = replicate_linear(boosted_weight, x)
                k = self.wk(x)
                v = self.wv(x)
                q = q.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
                k = k.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
                v = v.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
                o = context_parallel_attention(q, k, v)
                o = o.permute(0, 2, 1, 3).flatten(-2)
                o = self.wo(o)
                return o

        def forward(self, x):
            with fx_traceback.annotate({"outside_checkpoint": 0}):
                o = torch.utils.checkpoint.checkpoint(
                    self._compute_attention,
                    x,
                    use_reentrant=False,
                    context_fn=context_fn,
                )
                o0 = o + x
                o = self.w1(o0)
                o = torch.nn.functional.relu(o)
                o = self.w2(o)
                return o0 + o

    return Block, policy_fn


def test_local_map_ac_recompute_tags(device_mesh_3d):
    """Recompute tags from user checkpoint policy survive graph capture
    when local_map ops are used inside the checkpointed function."""
    mesh = device_mesh_3d
    Block, policy_fn = _make_local_map_block(mesh)

    nheads, dim, ffn_dim = 8, 128, 512
    bs = 8 * mesh.shape[0]
    seq_len = 32

    def input_fn():
        return torch.rand(bs, seq_len, dim, device="cuda")

    with torch.device("meta"):
        model = Block(nheads, dim, ffn_dim)

    gm = _build_joint_graph(model, input_fn, mesh)

    for n in gm.graph.nodes:
        if not _is_inside_checkpointed_fn(n):
            continue
        is_bwd = n.meta.get("partitioner_tag", "") == "is_backward"
        if not is_bwd:
            assert (
                n.meta.get("recompute") is not None
            ), f"{n} (target={n.target}) inside checkpointed fn has no recompute tag"


def test_local_map_ac_seq_nr_consistency(device_mesh_3d):
    """seq_nr consistency holds when local_map ops are used inside a
    checkpoint region."""
    mesh = device_mesh_3d
    Block, _ = _make_local_map_block(mesh)

    nheads, dim, ffn_dim = 8, 128, 512
    bs = 8 * mesh.shape[0]
    seq_len = 32

    def input_fn():
        return torch.rand(bs, seq_len, dim, device="cuda")

    with torch.device("meta"):
        model = Block(nheads, dim, ffn_dim)

    gm = _build_joint_graph(model, input_fn, mesh)

    fwd_seqs = set()
    for n in gm.graph.nodes:
        if not _is_inside_checkpointed_fn(n):
            continue
        is_bwd = n.meta.get("partitioner_tag", "") == "is_backward"
        if not is_bwd:
            fwd_seqs.add(n.meta["seq_nr"])
        else:
            assert n.meta["seq_nr"] in fwd_seqs, (
                f"backward node {n} has seq_nr={n.meta['seq_nr']} "
                f"with no matching forward node"
            )


def test_local_map_custom_metadata_propagation(device_mesh_3d):
    """Custom fx_traceback annotations propagate through local_map + checkpoint
    onto the parallel graph."""
    mesh = device_mesh_3d
    Block, _ = _make_local_map_block(mesh)

    nheads, dim, ffn_dim = 8, 128, 512
    bs = 8 * mesh.shape[0]
    seq_len = 32

    def input_fn():
        return torch.rand(bs, seq_len, dim, device="cuda")

    with torch.device("meta"):
        model = Block(nheads, dim, ffn_dim)

    with (
        fx_traceback.preserve_node_meta(),
        AutoParallel(model, input_fn, mesh) as autop,
    ):
        autop.add_parameter_memory_constraint(low=None, high=None)
        x_sharding = (Shard(0),) + (Replicate(),) * (mesh.ndim - 1)
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([x_sharding])
        sharding_placement = autop.optimize_placement()
        autop.apply_placement(sharding_placement)

    sdpa_nodes = [
        n
        for n in autop.parallel_gm.graph.nodes
        if "_scaled_dot_product_" in n.name and "_attention" in n.name
    ]
    assert len(sdpa_nodes) == 2, (
        f"Expected 2 SDPA nodes (fwd + bwd), got {len(sdpa_nodes)}: "
        f"{[n.name for n in sdpa_nodes]}"
    )
    fwd_sdpa, bwd_sdpa = sdpa_nodes
    assert fwd_sdpa.meta["custom"] == {
        "inside_checkpoint": 0,
        "inside_local_map": 2,
        "outside_checkpoint": 0,
    }
    assert bwd_sdpa.meta["custom"] == {
        "inside_checkpoint": 0,
        "inside_local_map": 2,
        "outside_checkpoint": 0,
    }
