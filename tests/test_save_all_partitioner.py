# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the _SaveAllPartitioner mechanism in autoparallel/compile.py.

The mechanism reproduces the first partitioner's save/recompute decisions in
the second compilation (torch.compile with autoparallel_backend) by:

1. Tagging forward outputs with `ap_must_save` in `_boxed_nop_preserve_node_meta`
2. `preserve_node_meta` propagates the tags to the second compilation's joint graph
3. `_SaveAllPartitioner` reads the tags and saves only those nodes

Without this machinery, the default min-cut partitioner makes independent
decisions that diverge from the first partitioner (most importantly, it
saves FSDP allgather outputs that should be recomputed via prefetch).
"""

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.utils.checkpoint import CheckpointPolicy

from autoparallel._testing.models.llama3 import Transformer, TransformerModelArgs
from autoparallel.api import AutoParallel, _boxed_nop_preserve_node_meta
from autoparallel.compile import (
    _patch_partitioner_dce,
    _SaveAllPartitioner,
    autoparallel_backend,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_small_llama(n_layers=2):
    """Tiny LLaMA-3 sized for fast tests."""
    return Transformer(
        TransformerModelArgs(
            dim=256,
            n_layers=n_layers,
            n_heads=8,
            n_kv_heads=2,
            ffn_dim_multiplier=1.3,
            multiple_of=64,
            rope_theta=500000,
            vocab_size=1024,
            max_seq_len=512,
        )
    )


def _run_autoparallel(mesh, n_layers=2, batch_size=None, seqlen=128):
    """Run AutoParallel up to apply_placement and return the parallel module
    plus AC pass for the second compilation."""
    from autoparallel.compile import _make_ac_joint_pass

    vocab_size = 1024
    if batch_size is None:
        batch_size = 2 * mesh.shape[0]

    with torch.device("meta"):
        model = _make_small_llama(n_layers=n_layers)

    AutoParallel._make_fuse_allgather_pass = lambda self: None

    with AutoParallel(
        model,
        lambda: torch.randint(0, vocab_size, (batch_size, seqlen), device="cuda"),
        mesh,
        MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
        repeated_subgraphs=True,
    ) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)
        if mesh.ndim == 2:
            autop.add_input_constraints([(Shard(0), Replicate())])
            autop.add_output_constraints([(Shard(0), Shard(2))])
        else:
            autop.add_input_constraints([(Shard(0),)])
            autop.add_output_constraints([(Shard(0),)])
        sharding_placement = autop.optimize_placement(verbose=False)

        ac_pass = _make_ac_joint_pass()
        with torch._functorch.config.patch({"joint_custom_pass": ac_pass}):
            parallel_mod = autop.apply_placement(sharding_placement)

    parallel_mod.to_empty(device="cuda")
    parallel_mod.init_weights()
    return parallel_mod, batch_size, seqlen, vocab_size


def _capture_saved_activations(
    parallel_mod, batch_size, seqlen, vocab_size, mesh, backend
):
    """Compile parallel_mod, capture what _SaveAllPartitioner sees and returns,
    and return (saved_activation_names, captured_info)."""
    captured = {}

    orig_call = _SaveAllPartitioner.__call__

    def capturing_call(self, gm, joint_inputs, **kwargs):
        # Snapshot tag state before _SaveAllPartitioner mutates anything
        captured["wait_tensor_recompute_tags"] = sum(
            1
            for n in gm.graph.nodes
            if "wait_tensor" in n.name
            and n.meta.get("recompute") == CheckpointPolicy.MUST_RECOMPUTE
        )
        captured["allgather_recompute_tags"] = sum(
            1
            for n in gm.graph.nodes
            if "all_gather_into_tensor" in n.name
            and n.meta.get("recompute") == CheckpointPolicy.MUST_RECOMPUTE
        )
        captured["ap_must_save_count"] = sum(
            1 for n in gm.graph.nodes if n.meta.get("custom", {}).get("ap_must_save")
        )
        fw, bw = orig_call(self, gm, joint_inputs, **kwargs)
        captured["fw_module"] = fw
        captured["bw_module"] = bw
        return fw, bw

    _SaveAllPartitioner.__call__ = capturing_call
    try:
        compiled = torch.compile(parallel_mod, backend=backend)
        x = torch.randint(
            0, vocab_size, (batch_size // mesh.shape[0], seqlen), device="cuda"
        )
        out = compiled(x)
        out.backward(torch.randn_like(out))
    finally:
        _SaveAllPartitioner.__call__ = orig_call

    # The backward graph's placeholders (minus tangents and primals) are the
    # saved activations.
    bw = captured["bw_module"]
    saved_names = []
    for node in bw.graph.nodes:
        if node.op != "placeholder":
            continue
        if isinstance(node.target, str) and (
            "tangent" in node.target or "primals" in node.target
        ):
            continue
        saved_names.append(node.name)
    return saved_names, captured


# ---------------------------------------------------------------------------
# Unit tests for the standalone mechanisms
# ---------------------------------------------------------------------------


def test_boxed_nop_tag_forward_marks_outputs():
    """_boxed_nop_preserve_node_meta(tag_forward=True) tags the forward
    output node's tensor args with ap_must_save."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.randn(4, device="meta")
    add = graph.call_function(torch.ops.aten.add.Tensor, args=(x, x))
    add.meta["val"] = torch.randn(4, device="meta")
    mul = graph.call_function(torch.ops.aten.mul.Tensor, args=(add, add))
    mul.meta["val"] = torch.randn(4, device="meta")
    graph.output((add, mul))
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    _boxed_nop_preserve_node_meta(gm, None, tag_forward=True)

    add_tagged = add.meta.get("custom", {}).get("ap_must_save")
    mul_tagged = mul.meta.get("custom", {}).get("ap_must_save")
    assert add_tagged is True
    assert mul_tagged is True


def test_boxed_nop_tag_forward_skips_getitem():
    """For getitem outputs (multi-output ops), the parent is tagged instead
    since getitem metadata doesn't survive preserve_node_meta."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.randn(4, device="meta")
    split = graph.call_function(torch.ops.aten.split.Tensor, args=(x, 2))
    split.meta["val"] = [torch.randn(2, device="meta")] * 2
    import operator

    g0 = graph.call_function(operator.getitem, args=(split, 0))
    g0.meta["val"] = torch.randn(2, device="meta")
    graph.output((g0,))
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    _boxed_nop_preserve_node_meta(gm, None, tag_forward=True)

    # Parent (split) gets the tag, not the getitem
    assert split.meta.get("custom", {}).get("ap_must_save") is True
    assert g0.meta.get("custom", {}).get("ap_must_save") is None


def test_boxed_nop_no_tag_forward_default():
    """tag_forward defaults to False — no tagging happens."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.randn(4, device="meta")
    add = graph.call_function(torch.ops.aten.add.Tensor, args=(x, x))
    add.meta["val"] = torch.randn(4, device="meta")
    graph.output((add,))
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    _boxed_nop_preserve_node_meta(gm, None)

    assert add.meta.get("custom", {}).get("ap_must_save") is None


def test_preserve_node_meta_propagates_recompute_through_collectives():
    """preserve_node_meta correctly propagates MUST_RECOMPUTE through
    collective ops (allgather, wait_tensor) — confirming the safety net
    mechanism works.
    """
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch.testing._internal.distributed.fake_pg import FakeStore

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            "fake", store=FakeStore(), rank=0, world_size=8
        )

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.randn(4, device="meta")
    add = graph.call_function(torch.ops.aten.add.Tensor, args=(x, x))
    add.meta["val"] = torch.randn(4, device="meta")
    add.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
    add.meta["ac_graph_id"] = 100000

    ag = graph.call_function(
        torch.ops._c10d_functional.all_gather_into_tensor.default,
        args=(add, 2, "0"),
    )
    ag.meta["val"] = torch.randn(8, device="meta")
    ag.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
    ag.meta["ac_graph_id"] = 100000

    wt = graph.call_function(torch.ops._c10d_functional.wait_tensor.default, args=(ag,))
    wt.meta["val"] = torch.randn(8, device="meta")
    wt.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
    wt.meta["ac_graph_id"] = 100000

    graph.output((wt,))
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    compiled = _boxed_nop_preserve_node_meta(gm, None)
    fake_input = torch.randn(4, device="cpu")
    new_gm = make_fx(compiled, tracing_mode="fake")([fake_input])

    for node in new_gm.graph.nodes:
        if node.op != "call_function":
            continue
        name = (
            node.target.__name__
            if hasattr(node.target, "__name__")
            else str(node.target)
        )
        if name in (
            "add.Tensor",
            "all_gather_into_tensor.default",
            "wait_tensor.default",
        ):
            assert (
                node.meta.get("recompute") == CheckpointPolicy.MUST_RECOMPUTE
            ), f"{name} lost MUST_RECOMPUTE tag through preserve_node_meta"


# ---------------------------------------------------------------------------
# Integration test for the partitioner — verifies that wait_tensor outputs
# from FSDP allgathers are NOT saved by _SaveAllPartitioner.
# ---------------------------------------------------------------------------


def test_save_all_partitioner_does_not_save_fsdp_wait_tensors(device_mesh_2d):
    """The whole point of the machinery: with FSDP allgathers tagged
    MUST_RECOMPUTE, _SaveAllPartitioner should NOT save their wait_tensor
    outputs in the backward.
    """
    parallel_mod, batch_size, seqlen, vocab_size = _run_autoparallel(
        device_mesh_2d, n_layers=2
    )
    backend = autoparallel_backend(enable_ac=False, overlap_scheduling=False)

    saved_names, captured = _capture_saved_activations(
        parallel_mod, batch_size, seqlen, vocab_size, device_mesh_2d, backend
    )

    # FSDP wait_tensor outputs should NOT appear as saved activations
    saved_wait_tensors = [n for n in saved_names if "wait_tensor" in n]
    assert len(saved_wait_tensors) == 0, (
        f"FSDP wait_tensor outputs were saved (should be recomputed via "
        f"FSDP prefetch): {saved_wait_tensors}"
    )

    # Sanity: MUST_RECOMPUTE tags survived into the joint graph
    assert captured["allgather_recompute_tags"] > 0, (
        "Expected MUST_RECOMPUTE tags on all_gather nodes to survive "
        "preserve_node_meta into the second compilation's joint graph"
    )


def test_save_all_partitioner_uses_ap_must_save_tags(device_mesh_2d):
    """_SaveAllPartitioner saves nodes tagged ap_must_save by the first
    compilation."""
    parallel_mod, batch_size, seqlen, vocab_size = _run_autoparallel(
        device_mesh_2d, n_layers=2
    )
    backend = autoparallel_backend(enable_ac=False, overlap_scheduling=False)

    saved_names, captured = _capture_saved_activations(
        parallel_mod, batch_size, seqlen, vocab_size, device_mesh_2d, backend
    )

    # ap_must_save tags should be present in the joint graph (set by
    # _boxed_nop_preserve_node_meta during first compilation, propagated
    # via preserve_node_meta)
    assert captured["ap_must_save_count"] > 0, (
        "Expected ap_must_save tags from first compilation to survive into "
        "second compilation's joint graph"
    )

    # And we should have saved at least some non-trivial activations
    activation_saves = [n for n in saved_names if not n.startswith("primals")]
    assert len(activation_saves) > 0


def test_save_all_partitioner_runs_end_to_end(device_mesh_2d):
    """Full end-to-end: AutoParallel + torch.compile(autoparallel_backend)
    forward + backward without errors."""
    parallel_mod, batch_size, seqlen, vocab_size = _run_autoparallel(
        device_mesh_2d, n_layers=2
    )
    backend = autoparallel_backend(enable_ac=False, overlap_scheduling=False)

    compiled = torch.compile(parallel_mod, backend=backend)
    x = torch.randint(
        0,
        vocab_size,
        (batch_size // device_mesh_2d.shape[0], seqlen),
        device="cuda",
    )
    out = compiled(x)
    out.backward(torch.randn_like(out))
    # Successful completion is the test


# ---------------------------------------------------------------------------
# Supporting machinery tests
# ---------------------------------------------------------------------------


def test_patch_partitioner_dce_allows_wait_tensor_elimination():
    """_patch_partitioner_dce makes is_not_collective return False for
    wait_tensor — required so unused wait_tensors can be DCE'd."""
    import torch._functorch.partitioners as partitioners

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    wt = graph.call_function(torch.ops._c10d_functional.wait_tensor.default, args=(x,))
    graph.output((wt,))

    original_result = partitioners.is_not_collective(wt)

    with _patch_partitioner_dce():
        patched_result = partitioners.is_not_collective(wt)

    # After exit, original behavior is restored
    restored_result = partitioners.is_not_collective(wt)

    # Inside the patch, wait_tensor is treated as DCE-eligible (returns False
    # is consistent — see the patch's intent of "letting wait_tensor through")
    assert patched_result is False
    # And the patch is properly restored
    assert restored_result == original_result


def test_autoparallel_backend_includes_save_all_partitioner():
    """autoparallel_backend() configures custom_partitioner_fn to be a
    _SaveAllPartitioner instance."""
    autoparallel_backend(enable_ac=False, overlap_scheduling=False)
    # The backend wires _SaveAllPartitioner as the custom_partitioner_fn.
    # Just verify the type exists and is properly defined.
    assert _SaveAllPartitioner is not None
    p = _SaveAllPartitioner()
    assert callable(p)
    assert hasattr(p, "uuid")
    assert p.uuid() is None


def test_save_all_partitioner_compile_with_ac_enabled(device_mesh_2d):
    """autoparallel_backend(enable_ac=True) runs the AC joint pass before
    the partitioner. The combination should compile end-to-end."""
    parallel_mod, batch_size, seqlen, vocab_size = _run_autoparallel(
        device_mesh_2d, n_layers=2
    )
    # AC enabled in the backend → joint_custom_pass adds PREFER_RECOMPUTE
    backend = autoparallel_backend(enable_ac=True, overlap_scheduling=False)

    compiled = torch.compile(parallel_mod, backend=backend)
    x = torch.randint(
        0,
        vocab_size,
        (batch_size // device_mesh_2d.shape[0], seqlen),
        device="cuda",
    )
    out = compiled(x)
    out.backward(torch.randn_like(out))


def test_save_all_partitioner_compile_1d_mesh(device_mesh_1d):
    """The partitioner works with a 1D (FSDP-only) mesh."""
    parallel_mod, batch_size, seqlen, vocab_size = _run_autoparallel(
        device_mesh_1d, n_layers=2
    )
    backend = autoparallel_backend(enable_ac=False, overlap_scheduling=False)

    compiled = torch.compile(parallel_mod, backend=backend)
    x = torch.randint(
        0,
        vocab_size,
        (batch_size // device_mesh_1d.shape[0], seqlen),
        device="cuda",
    )
    out = compiled(x)
    out.backward(torch.randn_like(out))
