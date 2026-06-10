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

import operator

import pytest
import torch
from conftest import apply_cuda_patches
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.utils.checkpoint import CheckpointPolicy

from autoparallel._testing.models.llama3 import Transformer, TransformerModelArgs
from autoparallel.api import _boxed_nop_preserve_node_meta
from autoparallel.compile import (
    _patch_partitioner_dce,
    _SaveAllPartitioner,
    autoparallel_backend,
)

# ---------------------------------------------------------------------------
# Helpers
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
    from autoparallel.api import AutoParallel
    from autoparallel.compile import _make_ac_joint_pass

    vocab_size = 1024
    if batch_size is None:
        batch_size = 2 * mesh.shape[0]

    with torch.device("meta"):
        model = _make_small_llama(n_layers=n_layers)

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


# Module-scoped caches for parallel_mod. Each integration test below builds
# the same one (deterministic for a given mesh shape) — caching saves ~30s
# per test. The cache is keyed by mesh.ndim since the helper only varies
# along that dimension; both 1D and 2D meshes get their own entry.
_parallel_mod_cache: dict = {}


@pytest.fixture(scope="module")
@apply_cuda_patches
def parallel_mod_2d(device_mesh_2d):
    if "2d" not in _parallel_mod_cache:
        _parallel_mod_cache["2d"] = _run_autoparallel(device_mesh_2d, n_layers=2)
    return _parallel_mod_cache["2d"]


@pytest.fixture(scope="module")
@apply_cuda_patches
def parallel_mod_1d(device_mesh_1d):
    if "1d" not in _parallel_mod_cache:
        _parallel_mod_cache["1d"] = _run_autoparallel(device_mesh_1d, n_layers=2)
    return _parallel_mod_cache["1d"]


def _capture_partitioner_call(
    parallel_mod, batch_size, seqlen, vocab_size, mesh, enable_ac=False
):
    """Run the second compilation with _SaveAllPartitioner as the
    partition_fn but use identity compilers (no Inductor codegen). This
    keeps the partitioner under test in the loop while skipping the
    expensive Triton kernel compilation.

    The dict contains:
      - wait_tensor_recompute_tags: count of MUST_RECOMPUTE on wait_tensors
      - allgather_recompute_tags: count of MUST_RECOMPUTE on all_gathers
      - ap_must_save_count: count of nodes tagged ap_must_save
      - fw_module, bw_module: the partitioned graph modules
      - saved_activation_names: backward inputs that aren't primals or tangents
    """
    from autoparallel.api import _suppress_wait_tensor_side_effect
    from autoparallel.compile import _make_ac_joint_pass, _patch_partitioner_dce

    captured = {}
    partitioner = _SaveAllPartitioner()

    def capturing_partition_fn(
        joint_module, joint_inputs, *, num_fwd_outputs, **kwargs
    ):
        captured["wait_tensor_recompute_tags"] = sum(
            1
            for n in joint_module.graph.nodes
            if "wait_tensor" in n.name
            and n.meta.get("recompute") == CheckpointPolicy.MUST_RECOMPUTE
        )
        captured["allgather_recompute_tags"] = sum(
            1
            for n in joint_module.graph.nodes
            if "all_gather_into_tensor" in n.name
            and n.meta.get("recompute") == CheckpointPolicy.MUST_RECOMPUTE
        )
        captured["ap_must_save_count"] = sum(
            1
            for n in joint_module.graph.nodes
            if n.meta.get("custom", {}).get("ap_must_save")
        )
        fw, bw = partitioner(
            joint_module,
            joint_inputs,
            num_fwd_outputs=num_fwd_outputs,
            **kwargs,
        )
        captured["fw_module"] = fw
        captured["bw_module"] = bw
        return fw, bw

    def capture_only_backend(gm, example_inputs):
        from torch._functorch.aot_autograd import aot_module_simplified

        return aot_module_simplified(
            gm,
            example_inputs,
            fw_compiler=lambda g, i: g,
            bw_compiler=lambda g, i: g,
            partition_fn=capturing_partition_fn,
        )

    functorch_patches = {}
    if enable_ac:
        functorch_patches["joint_custom_pass"] = _make_ac_joint_pass()

    with (
        _suppress_wait_tensor_side_effect(),
        _patch_partitioner_dce(),
        torch._functorch.config.patch(functorch_patches),
    ):
        compiled = torch.compile(parallel_mod, backend=capture_only_backend)
        x = torch.randint(
            0, vocab_size, (batch_size // mesh.shape[0], seqlen), device="cuda"
        )
        out = compiled(x)
        out.backward(torch.randn_like(out))

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
    captured["saved_activation_names"] = saved_names
    return captured


def _capture_first_partitioner_saves(mesh, n_layers=2, seqlen=128):
    """Run AutoParallel and capture the first partitioner's saved values
    (the forward outputs beyond num_fwd_outputs)."""
    from torch._functorch.aot_autograd import aot_compile_joint_with_descriptors

    from autoparallel.api import AutoParallel
    from autoparallel.compile import _make_ac_joint_pass

    vocab_size = 1024
    batch_size = 2 * mesh.shape[0]

    with torch.device("meta"):
        model = _make_small_llama(n_layers=n_layers)

    captured = {}

    def capturing_fw_compiler(fx_g, example_inputs, **kwargs):
        # The compiled forward returns [*model_outputs, *saved_values]
        output_node = next(n for n in fx_g.graph.nodes if n.op == "output")
        captured["fw_outputs"] = list(output_node.args[0])
        from autoparallel.api import _boxed_nop_preserve_node_meta

        return _boxed_nop_preserve_node_meta(fx_g, example_inputs, **kwargs)

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
        from functools import partial

        with torch._functorch.config.patch({"joint_custom_pass": ac_pass}):
            # Replicate apply_placement's compile call so we can intercept
            # the fw_compiler.
            autop._apply_placement_common(sharding_placement)
            from autoparallel.graph_passes.activation_checkpointing import (
                mark_fsdp_all_gather_recomputation,
            )

            mark_fsdp_all_gather_recomputation(
                autop.parallel_gm.graph, autop.reshard_after_forward
            )
            aot_compile_joint_with_descriptors(
                autop.joint_with_descriptors,
                fw_compiler=partial(capturing_fw_compiler, tag_forward=True),
                bw_compiler=autop.compiler_fn,
            )

    fw_metadata = autop.joint_with_descriptors._aot_state.fw_metadata
    num_fwd_outputs = fw_metadata.num_forward_returns
    # Saved values = forward outputs beyond num_fwd_outputs (the model outputs).
    # Filter out primal/placeholder pass-throughs since the second
    # partitioner's saved set doesn't include those either (they're inputs).
    fw_outputs = captured["fw_outputs"]
    saved = [
        n
        for n in fw_outputs[num_fwd_outputs:]
        if isinstance(n, torch.fx.Node) and n.op != "placeholder"
    ]
    return saved


def _saved_names_from_default_compile(
    parallel_mod, batch_size, seqlen, vocab_size, mesh, enable_ac=True
):
    """Compile with the default min-cut partitioner (NOT _SaveAllPartitioner)
    and capture which backward inputs are saved. This is the "without the fix"
    baseline that motivates _SaveAllPartitioner.

    By default this enables AC (joint_custom_pass = ac_joint_pass) — the
    motivating bad case is min-cut + AC tags driving force_save_collectives
    to save FSDP allgather outputs.
    """
    from autoparallel.compile import _make_ac_joint_pass

    captured = {}

    def simple_backend(gm, example_inputs):
        from torch._functorch.aot_autograd import aot_module_simplified
        from torch._functorch.partitioners import min_cut_rematerialization_partition

        def fw(g, i):
            return g

        def bw(g, i):
            captured["bw_module"] = g
            return g

        return aot_module_simplified(
            gm,
            example_inputs,
            fw_compiler=fw,
            bw_compiler=bw,
            partition_fn=min_cut_rematerialization_partition,
        )

    functorch_patches = {}
    if enable_ac:
        functorch_patches["joint_custom_pass"] = _make_ac_joint_pass()

    with torch._functorch.config.patch(functorch_patches):
        compiled = torch.compile(parallel_mod, backend=simple_backend)
        x = torch.randint(
            0, vocab_size, (batch_size // mesh.shape[0], seqlen), device="cuda"
        )
        out = compiled(x)
        out.backward(torch.randn_like(out))

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
    return saved_names


def _simple_partition(saved_node_meta):
    graph = torch.fx.Graph()
    x = graph.placeholder("primals_1")
    x.meta["val"] = torch.randn(4, device="meta")
    tangent = graph.placeholder("tangents_1")
    tangent.meta["val"] = torch.randn(4, device="meta")
    saved = graph.call_function(torch.ops.aten.add.Tensor, args=(x, x))
    saved.meta["val"] = torch.randn(4, device="meta")
    saved.meta.update(saved_node_meta)
    bwd = graph.call_function(torch.ops.aten.mul.Tensor, args=(saved, tangent))
    bwd.meta["val"] = torch.randn(4, device="meta")
    output = graph.output((saved, bwd))
    output.meta["desc"] = [None, None]
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    return _SaveAllPartitioner()(
        gm, [torch.randn(4), torch.randn(4)], num_fwd_outputs=1
    )


def _multi_output_partition(
    parent_meta,
    saved_indices=None,
    extra_parent_consumer=False,
):
    """Build a tiny joint graph with a multi-output forward op (split) and
    run _SaveAllPartitioner on it.

    parent_meta is merged into the multi-output op's meta. Optionally tag
    saved_indices on the parent (mirroring what tag_forward does). If
    extra_parent_consumer is True, the parent is also referenced directly
    in the output (a non-getitem user that survives DCE), so the partitioner
    observes split.users containing both getitems and a non-getitem node.
    """
    graph = torch.fx.Graph()
    x = graph.placeholder("primals_1")
    x.meta["val"] = torch.randn(6, device="meta")
    tangent = graph.placeholder("tangents_1")
    tangent.meta["val"] = torch.randn(6, device="meta")

    split = graph.call_function(torch.ops.aten.split.Tensor, args=(x, 2))
    split.meta["val"] = [torch.randn(2, device="meta")] * 3
    custom = split.meta.setdefault("custom", {})
    custom.update(parent_meta.get("custom", {}))
    if "recompute" in parent_meta:
        split.meta["recompute"] = parent_meta["recompute"]
    if saved_indices is not None:
        custom["ap_must_save_getitem_indices"] = saved_indices

    g0 = graph.call_function(operator.getitem, args=(split, 0))
    g0.meta["val"] = torch.randn(2, device="meta")
    g1 = graph.call_function(operator.getitem, args=(split, 1))
    g1.meta["val"] = torch.randn(2, device="meta")
    g2 = graph.call_function(operator.getitem, args=(split, 2))
    g2.meta["val"] = torch.randn(2, device="meta")

    # Concatenate g0+g1+g2 + tangent so all getitems feed into backward
    cat = graph.call_function(torch.ops.aten.cat.default, args=([g0, g1, g2],))
    cat.meta["val"] = torch.randn(6, device="meta")
    bwd = graph.call_function(torch.ops.aten.mul.Tensor, args=(cat, tangent))
    bwd.meta["val"] = torch.randn(6, device="meta")

    # Optionally make split a direct forward output (non-getitem user).
    if extra_parent_consumer:
        output_args = (cat, split, bwd)
        num_fwd_outputs = 2
    else:
        output_args = (cat, bwd)
        num_fwd_outputs = 1
    output = graph.output(output_args)
    output.meta["desc"] = [None] * len(output_args)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    return _SaveAllPartitioner()(
        gm,
        [torch.randn(6), torch.randn(6)],
        num_fwd_outputs=num_fwd_outputs,
    )


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

    assert add.meta.get("custom", {}).get("ap_must_save") is True
    assert mul.meta.get("custom", {}).get("ap_must_save") is True


def test_boxed_nop_tag_forward_skips_getitem():
    """For getitem outputs (multi-output ops), the parent is tagged instead
    since getitem metadata doesn't survive preserve_node_meta."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.randn(4, device="meta")
    split = graph.call_function(torch.ops.aten.split.Tensor, args=(x, 2))
    split.meta["val"] = [torch.randn(2, device="meta")] * 2
    g0 = graph.call_function(operator.getitem, args=(split, 0))
    g0.meta["val"] = torch.randn(2, device="meta")
    graph.output((g0,))
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    _boxed_nop_preserve_node_meta(gm, None, tag_forward=True)

    # Parent (split) gets the tag, not the getitem
    assert split.meta.get("custom", {}).get("ap_must_save") is True
    assert split.meta.get("custom", {}).get("ap_must_save_getitem_indices") == [0]
    assert g0.meta.get("custom", {}).get("ap_must_save") is None


def test_boxed_nop_tag_forward_records_getitem_indices():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.randn(6, device="meta")
    split = graph.call_function(torch.ops.aten.split.Tensor, args=(x, 2))
    split.meta["val"] = [torch.randn(2, device="meta")] * 3
    g0 = graph.call_function(operator.getitem, args=(split, 0))
    g0.meta["val"] = torch.randn(2, device="meta")
    g2 = graph.call_function(operator.getitem, args=(split, 2))
    g2.meta["val"] = torch.randn(2, device="meta")
    graph.output((g0, g2))
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    _boxed_nop_preserve_node_meta(gm, None, tag_forward=True)

    assert split.meta["custom"]["ap_must_save_getitem_indices"] == [0, 2]


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


def test_save_all_partitioner_saves_ap_must_save_despite_prefer_recompute():
    fw, bw = _simple_partition(
        {
            "custom": {"ap_must_save": True},
            "recompute": CheckpointPolicy.PREFER_RECOMPUTE,
        }
    )

    fw_outputs = next(n for n in fw.graph.nodes if n.op == "output").args[0]
    bw_placeholders = [n.name for n in bw.graph.nodes if n.op == "placeholder"]
    assert "add_tensor" in [n.name for n in fw_outputs if isinstance(n, torch.fx.Node)]
    assert "add_tensor" in bw_placeholders


def test_save_all_partitioner_honors_must_save():
    fw, bw = _simple_partition({"recompute": CheckpointPolicy.MUST_SAVE})

    fw_outputs = next(n for n in fw.graph.nodes if n.op == "output").args[0]
    bw_placeholders = [n.name for n in bw.graph.nodes if n.op == "placeholder"]
    assert "add_tensor" in [n.name for n in fw_outputs if isinstance(n, torch.fx.Node)]
    assert "add_tensor" in bw_placeholders


def test_save_all_partitioner_does_not_save_must_recompute():
    fw, bw = _simple_partition(
        {
            "custom": {"ap_must_save": True},
            "recompute": CheckpointPolicy.MUST_RECOMPUTE,
        }
    )

    bw_placeholders = [n.name for n in bw.graph.nodes if n.op == "placeholder"]
    assert "add_tensor" not in bw_placeholders


def test_save_all_partitioner_must_recompute_blocks_multi_output_save():
    """MUST_RECOMPUTE on a multi-output op blocks saving its getitem
    children, even when ap_must_save is also set. Documents the invariant
    that the first partitioner's recompute decision wins over its own
    save tags (which can happen if both are set during graph passes)."""
    fw, bw = _multi_output_partition(
        parent_meta={
            "custom": {"ap_must_save": True},
            "recompute": CheckpointPolicy.MUST_RECOMPUTE,
        },
        saved_indices=[0, 1, 2],
    )

    # No getitem children should appear in backward inputs
    bw_placeholders = [n.name for n in bw.graph.nodes if n.op == "placeholder"]
    assert not any(
        name.startswith("getitem") for name in bw_placeholders
    ), f"MUST_RECOMPUTE should block multi-output save, got: {bw_placeholders}"


def test_save_all_partitioner_must_recompute_blocks_opaque_save():
    """MUST_RECOMPUTE blocks saving even on nodes that would otherwise be
    saved as opaque. The _must_recompute check must run before is_opaque_node
    so the first partitioner's recompute intent is honored uniformly."""
    # Build a fake graph with a node we'll force is_opaque_node to recognize.
    graph = torch.fx.Graph()
    x = graph.placeholder("primals_1")
    x.meta["val"] = torch.randn(4, device="meta")
    tangent = graph.placeholder("tangents_1")
    tangent.meta["val"] = torch.randn(4, device="meta")
    opaque_like = graph.call_function(torch.ops.aten.add.Tensor, args=(x, x))
    opaque_like.meta["val"] = torch.randn(4, device="meta")
    opaque_like.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
    bwd = graph.call_function(torch.ops.aten.mul.Tensor, args=(opaque_like, tangent))
    bwd.meta["val"] = torch.randn(4, device="meta")
    output = graph.output((opaque_like, bwd))
    output.meta["desc"] = [None, None]
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    # Force is_opaque_node to return True for our op. _SaveAllPartitioner
    # imports it from torch._functorch.partitioners, so patch there.
    import torch._functorch.partitioners as partitioners

    original = partitioners.is_opaque_node
    partitioners.is_opaque_node = lambda n: n.name == "add_tensor"
    try:
        fw, bw = _SaveAllPartitioner()(
            gm, [torch.randn(4), torch.randn(4)], num_fwd_outputs=1
        )
    finally:
        partitioners.is_opaque_node = original

    # add_tensor was tagged MUST_RECOMPUTE; even though is_opaque_node
    # returned True, it should NOT be saved
    bw_placeholders = [n.name for n in bw.graph.nodes if n.op == "placeholder"]
    assert "add_tensor" not in bw_placeholders, (
        f"MUST_RECOMPUTE on an opaque-classified node was ignored; "
        f"backward placeholders: {bw_placeholders}"
    )


def test_save_all_partitioner_multi_output_with_non_getitem_user():
    """A multi-output op tagged ap_must_save must save its getitem children
    (not the parent tuple) even when it has a non-getitem user. The save
    logic must look at the node's value type, not just its users."""
    fw, bw = _multi_output_partition(
        parent_meta={"custom": {"ap_must_save": True}},
        saved_indices=[0, 1, 2],
        extra_parent_consumer=True,
    )

    # Backward should receive the getitem children, NOT the split tuple
    bw_placeholders = [n for n in bw.graph.nodes if n.op == "placeholder"]
    # No placeholder should have a tuple/list value (which would mean we
    # saved the multi-output op directly)
    for n in bw_placeholders:
        val = n.meta.get("val")
        assert not isinstance(val, (list, tuple)), (
            f"Backward got placeholder {n.name} with tuple/list val — multi-output op "
            f"was saved directly instead of its getitem children"
        )
    # And at least one getitem should be present (the save took effect)
    getitem_names = [n.name for n in bw_placeholders if n.name.startswith("getitem")]
    assert len(getitem_names) > 0, (
        f"Expected getitem children to be saved, got placeholders: "
        f"{[n.name for n in bw_placeholders]}"
    )


def test_save_all_partitioner_replays_only_indexed_getitems():
    """When tag_forward records specific getitem indices on a multi-output
    parent, only those indices should appear as saved backward inputs;
    other indices should not be saved (and would be recomputed if needed).

    Build a graph where backward consumes BOTH getitem 0 and getitem 1, but
    we only tag index 0 as ap_must_save. The partitioner should save
    getitem 0 only.
    """
    graph = torch.fx.Graph()
    x = graph.placeholder("primals_1")
    x.meta["val"] = torch.randn(4, device="meta")
    t0 = graph.placeholder("tangents_1")
    t0.meta["val"] = torch.randn(2, device="meta")
    t1 = graph.placeholder("tangents_2")
    t1.meta["val"] = torch.randn(2, device="meta")

    # Multi-output op: split into 2 chunks. Mark only index 0 as ap_must_save.
    split = graph.call_function(torch.ops.aten.split.Tensor, args=(x, 2))
    split.meta["val"] = [torch.randn(2, device="meta")] * 2
    custom = split.meta.setdefault("custom", {})
    custom["ap_must_save"] = True
    custom["ap_must_save_getitem_indices"] = [0]

    g0 = graph.call_function(operator.getitem, args=(split, 0))
    g0.meta["val"] = torch.randn(2, device="meta")
    g1 = graph.call_function(operator.getitem, args=(split, 1))
    g1.meta["val"] = torch.randn(2, device="meta")

    # Forward outputs: a value that depends on both g0 and g1 (so both
    # getitems are required in forward).
    add_fw = graph.call_function(torch.ops.aten.add.Tensor, args=(g0, g1))
    add_fw.meta["val"] = torch.randn(2, device="meta")

    # Backward ops: independent muls using g0 and g1 respectively, so each
    # getitem is a real backward dependency.
    bwd0 = graph.call_function(torch.ops.aten.mul.Tensor, args=(g0, t0))
    bwd0.meta["val"] = torch.randn(2, device="meta")
    bwd1 = graph.call_function(torch.ops.aten.mul.Tensor, args=(g1, t1))
    bwd1.meta["val"] = torch.randn(2, device="meta")

    output = graph.output((add_fw, bwd0, bwd1))
    output.meta["desc"] = [None, None, None]
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    fw, bw = _SaveAllPartitioner()(
        gm,
        [torch.randn(4), torch.randn(2), torch.randn(2)],
        num_fwd_outputs=1,
    )

    bw_placeholders = [n.name for n in bw.graph.nodes if n.op == "placeholder"]
    # getitem 0 should be a saved backward input (we tagged index 0)
    assert (
        "getitem" in bw_placeholders
    ), f"Expected getitem (index 0) to be saved; got {bw_placeholders}"
    # getitem 1 should NOT be a saved input — it's not in the indices list
    assert "getitem_1" not in bw_placeholders, (
        f"Expected getitem_1 (index 1) NOT to be saved (index restriction); "
        f"got {bw_placeholders}"
    )


def test_save_all_partitioner_must_save_overrides_getitem_indices():
    """MUST_SAVE on a multi-output parent should save ALL getitem children
    even if ap_must_save_getitem_indices restricts to a subset. MUST_SAVE
    is a stronger directive than ap_must_save's index-specific replay."""
    graph = torch.fx.Graph()
    x = graph.placeholder("primals_1")
    x.meta["val"] = torch.randn(4, device="meta")
    t0 = graph.placeholder("tangents_1")
    t0.meta["val"] = torch.randn(2, device="meta")
    t1 = graph.placeholder("tangents_2")
    t1.meta["val"] = torch.randn(2, device="meta")

    split = graph.call_function(torch.ops.aten.split.Tensor, args=(x, 2))
    split.meta["val"] = [torch.randn(2, device="meta")] * 2
    # Both MUST_SAVE AND restricted indices. MUST_SAVE should win.
    split.meta["recompute"] = CheckpointPolicy.MUST_SAVE
    custom = split.meta.setdefault("custom", {})
    custom["ap_must_save"] = True
    custom["ap_must_save_getitem_indices"] = [0]

    g0 = graph.call_function(operator.getitem, args=(split, 0))
    g0.meta["val"] = torch.randn(2, device="meta")
    g1 = graph.call_function(operator.getitem, args=(split, 1))
    g1.meta["val"] = torch.randn(2, device="meta")

    add_fw = graph.call_function(torch.ops.aten.add.Tensor, args=(g0, g1))
    add_fw.meta["val"] = torch.randn(2, device="meta")
    bwd0 = graph.call_function(torch.ops.aten.mul.Tensor, args=(g0, t0))
    bwd0.meta["val"] = torch.randn(2, device="meta")
    bwd1 = graph.call_function(torch.ops.aten.mul.Tensor, args=(g1, t1))
    bwd1.meta["val"] = torch.randn(2, device="meta")

    output = graph.output((add_fw, bwd0, bwd1))
    output.meta["desc"] = [None, None, None]
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    fw, bw = _SaveAllPartitioner()(
        gm,
        [torch.randn(4), torch.randn(2), torch.randn(2)],
        num_fwd_outputs=1,
    )

    bw_placeholders = [n.name for n in bw.graph.nodes if n.op == "placeholder"]
    # Both getitems should be saved (MUST_SAVE overrides the index restriction)
    assert "getitem" in bw_placeholders, f"Expected getitem; got {bw_placeholders}"
    assert "getitem_1" in bw_placeholders, (
        f"Expected getitem_1 even though indices=[0] (MUST_SAVE should override); "
        f"got {bw_placeholders}"
    )


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


def test_patch_partitioner_dce_allows_wait_tensor_elimination():
    """_patch_partitioner_dce overrides is_not_collective for wait_tensor,
    and the override is properly reverted on exit."""
    import torch._functorch.partitioners as partitioners

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    wt = graph.call_function(torch.ops._c10d_functional.wait_tensor.default, args=(x,))
    graph.output((wt,))

    original_result = partitioners.is_not_collective(wt)

    with _patch_partitioner_dce():
        patched_result = partitioners.is_not_collective(wt)

    # After exit, the original function is restored
    restored_result = partitioners.is_not_collective(wt)

    # Inside the patch, wait_tensor goes through our shortcut
    assert patched_result is False
    # And the patch is properly restored
    assert restored_result == original_result


# ---------------------------------------------------------------------------
# Integration tests for the partitioner behavior on a real model
# ---------------------------------------------------------------------------


@apply_cuda_patches
def test_save_all_partitioner_does_not_save_fsdp_wait_tensors(
    parallel_mod_2d, device_mesh_2d
):
    """The whole point of the machinery: with FSDP allgathers tagged
    MUST_RECOMPUTE, _SaveAllPartitioner should NOT save their wait_tensor
    outputs in the backward.
    """
    parallel_mod, batch_size, seqlen, vocab_size = parallel_mod_2d

    captured = _capture_partitioner_call(
        parallel_mod,
        batch_size,
        seqlen,
        vocab_size,
        device_mesh_2d,
        enable_ac=False,
    )

    # FSDP wait_tensor outputs should NOT appear as saved activations
    saved_wait_tensors = [
        n for n in captured["saved_activation_names"] if "wait_tensor" in n
    ]
    assert len(saved_wait_tensors) == 0, (
        f"FSDP wait_tensor outputs were saved (should be recomputed via "
        f"FSDP prefetch): {saved_wait_tensors}"
    )

    # Sanity: MUST_RECOMPUTE tags survived into the joint graph
    assert captured["allgather_recompute_tags"] > 0, (
        "Expected MUST_RECOMPUTE tags on all_gather nodes to survive "
        "preserve_node_meta into the second compilation's joint graph"
    )

    # And ap_must_save tags should be present (sanity check on tag
    # propagation — used to be a separate test).
    assert captured["ap_must_save_count"] > 0, (
        "Expected ap_must_save tags from first compilation to survive into "
        "second compilation's joint graph"
    )
    activation_saves = [
        n for n in captured["saved_activation_names"] if not n.startswith("primals")
    ]
    assert len(activation_saves) > 0


@apply_cuda_patches
def test_default_partitioner_diverges_from_save_all_partitioner(
    parallel_mod_2d, device_mesh_2d
):
    """The default min-cut partitioner produces a different save list than
    _SaveAllPartitioner when AC is active. This is the motivating reason
    _SaveAllPartitioner exists: min-cut + AC tags can save FSDP allgather
    outputs (or other tensors that the first partitioner chose to recompute).

    We assert "differ" rather than checking for a specific wait_tensor name
    because the exact divergence depends on model size, mesh shape, and the
    AC stage budget. For the production LLaMA-3 8B config (32 layers, dim=4096,
    128 GPUs), the divergence manifests as ~1.2 GB of FSDP wait_tensor outputs
    being saved per 4 layers — see [memory_gap_investigation memory note].
    """
    parallel_mod, batch_size, seqlen, vocab_size = parallel_mod_2d

    # Default partitioner with AC enabled in second compilation (bad case)
    default_saves = _saved_names_from_default_compile(
        parallel_mod,
        batch_size,
        seqlen,
        vocab_size,
        device_mesh_2d,
        enable_ac=True,
    )

    # _SaveAllPartitioner with AC enabled in second compilation (our fix)
    save_all_captured = _capture_partitioner_call(
        parallel_mod,
        batch_size,
        seqlen,
        vocab_size,
        device_mesh_2d,
        enable_ac=True,
    )
    save_all_saves = save_all_captured["saved_activation_names"]

    # The two partitioners pick different things. Equal save count would
    # only happen by coincidence — even when counts match the choice of
    # tensors differs.
    default_set = set(default_saves)
    save_all_set = set(save_all_saves)
    only_in_default = default_set - save_all_set
    only_in_save_all = save_all_set - default_set
    assert only_in_default or only_in_save_all, (
        f"Default partitioner and _SaveAllPartitioner produced identical "
        f"saves ({sorted(default_set)}). If this is reproducible, the "
        f"motivating divergence may have been fixed upstream and "
        f"_SaveAllPartitioner could be reevaluated."
    )


@apply_cuda_patches
def test_save_all_partitioner_reproduces_first_partitioner_saves(
    parallel_mod_2d, device_mesh_2d
):
    """_SaveAllPartitioner's saved set should approximately match what the
    first partitioner chose. The two operate on different graphs (the first
    inside apply_placement, the second inside torch.compile), so we compare
    by tensor shape/dtype histograms rather than node names — names will
    differ across compilations but the underlying tensors should match.
    """
    # First partitioner: capture the forward outputs beyond num_fwd_outputs.
    first_saves = _capture_first_partitioner_saves(device_mesh_2d, n_layers=2)

    # Second partitioner: reuse the cached parallel_mod and capture the
    # backward inputs (saved values).
    parallel_mod, batch_size, seqlen, vocab_size = parallel_mod_2d
    captured = _capture_partitioner_call(
        parallel_mod,
        batch_size,
        seqlen,
        vocab_size,
        device_mesh_2d,
        enable_ac=False,
    )
    bw = captured["bw_module"]
    second_saves = []
    for node in bw.graph.nodes:
        if node.op != "placeholder":
            continue
        if isinstance(node.target, str) and (
            "tangent" in node.target or "primals" in node.target
        ):
            continue
        second_saves.append(node)

    def _shape_sig(node):
        val = node.meta.get("val")
        if val is None or not hasattr(val, "shape"):
            return None
        return (tuple(val.shape), str(val.dtype))

    # Counts should be close. Retracing through Dynamo can add or eliminate
    # a small number of view/reshape nodes.
    diff = abs(len(first_saves) - len(second_saves))
    assert diff <= 2, (
        f"First partitioner saved {len(first_saves)} values, but "
        f"_SaveAllPartitioner saved {len(second_saves)} (diff {diff} > 2). "
        f"They should match closely."
    )

    # Shape/dtype histograms should match closely too.
    from collections import Counter

    first_shapes = Counter(_shape_sig(n) for n in first_saves)
    second_shapes = Counter(_shape_sig(n) for n in second_saves)
    # Drop entries with no shape (sym ints, opaque, etc.)
    first_shapes.pop(None, None)
    second_shapes.pop(None, None)

    # Symmetric difference should be small
    diff_count = sum((first_shapes - second_shapes).values()) + sum(
        (second_shapes - first_shapes).values()
    )
    assert diff_count <= 4, (
        f"Shape histograms diverged too much (diff_count={diff_count}).\n"
        f"  Only in first:  {first_shapes - second_shapes}\n"
        f"  Only in second: {second_shapes - first_shapes}"
    )


@apply_cuda_patches
def test_save_all_partitioner_compile_with_ac_enabled(parallel_mod_2d, device_mesh_2d):
    """End-to-end smoke test: AutoParallel + torch.compile(autoparallel_backend)
    with AC enabled. Exercises the full Inductor pipeline including AC joint
    pass and codegen — kept around so a regression in the backend wiring
    surfaces here even if the unit tests still pass."""
    parallel_mod, batch_size, seqlen, vocab_size = parallel_mod_2d
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


@apply_cuda_patches
def test_save_all_partitioner_compile_1d_mesh(parallel_mod_1d, device_mesh_1d):
    """The partitioner works with a 1D (FSDP-only) mesh."""
    parallel_mod, batch_size, seqlen, vocab_size = parallel_mod_1d
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
