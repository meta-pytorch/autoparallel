# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the monkey-patches installed by
autoparallel.graph_passes.auto_bucketing._patch_fsdp_bucketing.

The patches are installed at module import time, so importing
auto_bucketing replaces the originals in PyTorch's bucketing/fsdp modules.
"""

import torch
import torch._inductor.fx_passes.bucketing as bucketing_mod
import torch._inductor.fx_passes.fsdp as fsdp_mod

# Importing auto_bucketing triggers _patch_fsdp_bucketing() at import time.
import autoparallel.graph_passes.auto_bucketing as ab  # noqa: F401


def _make_fake_tensor_meta(shape, dtype=torch.bfloat16):
    """Return a FakeTensor-like object usable as node.meta['val'].

    The bucketing helpers only access .numel(), .element_size(), and .dtype.
    A real torch.empty meta tensor satisfies that contract.
    """
    return torch.empty(shape, dtype=dtype, device="meta")


def _make_ag_node(g, x, group_name, shape, dtype=torch.bfloat16, group_size=8):
    """Build an all_gather_into_tensor call_function node with FSDP-shaped
    input chain: exactly one placeholder ancestor (which is_fsdp_all_gather
    requires)."""
    ag = g.call_function(
        torch.ops._c10d_functional.all_gather_into_tensor.default,
        (x, group_size, group_name),
    )
    ag.meta["val"] = _make_fake_tensor_meta(shape, dtype)
    return ag


def _make_wait_node(g, ag):
    w = g.call_function(torch.ops._c10d_functional.wait_tensor.default, (ag,))
    w.meta["val"] = ag.meta["val"]
    return w


def _build_gm(graph):
    return torch.fx.GraphModule(torch.nn.Module(), graph)


# ---------- identify_fsdp_groups tests ----------


def test_identify_fsdp_groups_picks_primary_group():
    """With AGs split across two PGs, only the group with more AGs is returned."""
    g = torch.fx.Graph()
    # Group "dp" gets 5 AGs, group "tp" gets 2. "dp" should win.
    placeholders = [g.placeholder(f"p{i}") for i in range(7)]
    for ph in placeholders:
        ph.meta["val"] = _make_fake_tensor_meta((1024,), torch.bfloat16)

    for i in range(5):
        _make_ag_node(g, placeholders[i], "dp", (8192,), group_size=8)
    for i in range(5, 7):
        _make_ag_node(g, placeholders[i], "tp", (8192,), group_size=8)
    g.output(())
    gm = _build_gm(g)

    groups, group_size = fsdp_mod.identify_fsdp_groups(gm)

    assert list(groups) == ["dp"], f"expected only primary 'dp', got {list(groups)}"
    assert group_size == 8


def test_identify_fsdp_groups_handles_no_fsdp_ags():
    """Empty graph or graphs with no FSDP AGs return an empty OrderedSet."""
    g = torch.fx.Graph()
    g.output(())
    gm = _build_gm(g)

    groups, group_size = fsdp_mod.identify_fsdp_groups(gm)

    assert len(groups) == 0
    assert group_size is None


def test_identify_fsdp_groups_ties_pick_one():
    """Equal AG counts: most_common is deterministic but the exact group
    chosen depends on insertion order. We only require that exactly one is
    returned (not both)."""
    g = torch.fx.Graph()
    placeholders = [g.placeholder(f"p{i}") for i in range(4)]
    for ph in placeholders:
        ph.meta["val"] = _make_fake_tensor_meta((1024,), torch.bfloat16)
    for ph in placeholders[:2]:
        _make_ag_node(g, ph, "dp", (8192,))
    for ph in placeholders[2:]:
        _make_ag_node(g, ph, "tp", (8192,))
    g.output(())
    gm = _build_gm(g)

    groups, _ = fsdp_mod.identify_fsdp_groups(gm)
    assert len(groups) == 1
    assert list(groups)[0] in ("dp", "tp")


# ---------- greedy_bucket tests ----------


def _build_ag_chain_graph(n_ags, ag_shape_bytes_each=1024, group_name="dp"):
    """Build a graph with n_ags AGs in the same group + their waits.

    Each AG has its own placeholder input (so descendant relations between
    AGs are absent — every AG is independent).

    Returns (gm, ag_nodes, wait_nodes).
    """
    g = torch.fx.Graph()
    ag_nodes = []
    wait_nodes = []
    # Use shape such that numel * element_size == ag_shape_bytes_each.
    # bfloat16 = 2 bytes, so numel = bytes / 2.
    numel = max(1, ag_shape_bytes_each // 2)
    for i in range(n_ags):
        ph = g.placeholder(f"p{i}")
        ph.meta["val"] = _make_fake_tensor_meta((numel,), torch.bfloat16)
        ag = _make_ag_node(g, ph, group_name, (numel,), group_size=8)
        w = _make_wait_node(g, ag)
        ag_nodes.append(ag)
        wait_nodes.append(w)
    g.output(tuple(wait_nodes))
    return _build_gm(g), ag_nodes, wait_nodes


def _call_greedy_bucket(gm, bucket_cap_mb, *, filter_wait_node=None):
    """Invoke the patched greedy_bucket via the public API surface."""
    return bucketing_mod.greedy_bucket_collective_by_mb(
        gm,
        lambda _idx: bucket_cap_mb,
        # filter_node accepts the collective node (args[0] of the wait).
        # We accept all AGs for these tests.
        filter_node=bucketing_mod.is_all_gather_into_tensor,
        node_group_key=bucketing_mod._ag_group_key,
        filter_wait_node=filter_wait_node,
    )


def test_greedy_bucket_merges_within_caps():
    """4 small AGs, plenty of room: one bucket containing all of them."""
    gm, ag_nodes, _ = _build_ag_chain_graph(n_ags=4, ag_shape_bytes_each=1024)
    # max_topo_span large enough to fit all (graph has ~12 nodes).
    ab.aten_autobucketing_config.max_topo_span = 1000
    try:
        buckets = _call_greedy_bucket(gm, bucket_cap_mb=10.0)
    finally:
        ab.aten_autobucketing_config.max_topo_span = 1500

    assert len(buckets) == 1, f"expected 1 bucket, got {len(buckets)}"
    assert len(buckets[0]) == 4, f"expected 4 AGs in bucket, got {len(buckets[0])}"


def test_greedy_bucket_splits_on_bytes_cap():
    """Each AG = 4 MB, cap = 10 MB → buckets of at most 2 AGs each."""
    # 5 MB per AG → second AG exceeds 10 MB cap.
    bytes_per_ag = 5 * 1024 * 1024
    gm, ag_nodes, _ = _build_ag_chain_graph(n_ags=6, ag_shape_bytes_each=bytes_per_ag)
    ab.aten_autobucketing_config.max_topo_span = 1000
    try:
        buckets = _call_greedy_bucket(gm, bucket_cap_mb=10.0)
    finally:
        ab.aten_autobucketing_config.max_topo_span = 1500

    assert len(buckets) >= 2, f"expected >=2 buckets, got {len(buckets)}: {buckets}"
    # Each bucket must respect the 10 MB cap, so each bucket has ≤ 2 AGs.
    for b in buckets:
        assert len(b) <= 2, f"bucket has {len(b)} > 2 AGs (exceeds 10 MB cap)"
    # Single-element buckets are dropped, so each surviving bucket has ≥ 2.
    for b in buckets:
        assert len(b) >= 2, f"surviving bucket has only {len(b)} member"


def test_greedy_bucket_splits_on_span_cap():
    """4 AGs spaced out in the graph, max_topo_span forces splits below
    the byte cap."""
    # Each AG with its placeholder, wait, plus a string of no-op compute
    # nodes between them so the topo distance between AGs is large.
    g = torch.fx.Graph()
    ag_nodes = []
    wait_nodes = []
    bytes_per_ag = 1024  # tiny — bytes cap won't fire
    numel = bytes_per_ag // 2
    for i in range(4):
        ph = g.placeholder(f"p{i}")
        ph.meta["val"] = _make_fake_tensor_meta((numel,), torch.bfloat16)
        ag = _make_ag_node(g, ph, "dp", (numel,), group_size=8)
        w = _make_wait_node(g, ag)
        ag_nodes.append(ag)
        wait_nodes.append(w)
        # Pad: add 50 unrelated ops between this wait and the next AG.
        x = w
        for j in range(50):
            x = g.call_function(torch.ops.aten.relu.default, (x,))
            x.meta["val"] = w.meta["val"]
    g.output(tuple(wait_nodes))
    gm = _build_gm(g)

    # With ~50+ nodes per AG, total span across 4 AGs ≈ 200.
    # max_topo_span = 80 means at most 2 AGs per bucket.
    ab.aten_autobucketing_config.max_topo_span = 80
    try:
        buckets = _call_greedy_bucket(gm, bucket_cap_mb=10.0)
    finally:
        ab.aten_autobucketing_config.max_topo_span = 1500

    assert (
        len(buckets) >= 2
    ), f"expected >=2 buckets due to span cap, got {len(buckets)}: {buckets}"
    for b in buckets:
        # Each bucket spans ~50 nodes per pair → 2 AGs at most fit in 80.
        assert len(b) <= 2, f"bucket of {len(b)} AGs violates max_topo_span=80"


def test_greedy_bucket_span_cap_none_disables_span():
    """max_topo_span=None should restore byte-only behavior:
    same 4 spaced-out AGs that split at span=80 above now go into one bucket."""
    g = torch.fx.Graph()
    wait_nodes = []
    for i in range(4):
        ph = g.placeholder(f"p{i}")
        ph.meta["val"] = _make_fake_tensor_meta((512,), torch.bfloat16)
        ag = _make_ag_node(g, ph, "dp", (512,), group_size=8)
        w = _make_wait_node(g, ag)
        wait_nodes.append(w)
        x = w
        for j in range(50):
            x = g.call_function(torch.ops.aten.relu.default, (x,))
            x.meta["val"] = w.meta["val"]
    g.output(tuple(wait_nodes))
    gm = _build_gm(g)

    saved = ab.aten_autobucketing_config.max_topo_span
    ab.aten_autobucketing_config.max_topo_span = None
    try:
        buckets = _call_greedy_bucket(gm, bucket_cap_mb=10.0)
    finally:
        ab.aten_autobucketing_config.max_topo_span = saved

    assert (
        len(buckets) == 1
    ), f"expected 1 bucket with span cap disabled, got {len(buckets)}"
    assert len(buckets[0]) == 4


def test_greedy_bucket_skips_descendant_collectives():
    """Two AGs where AG2 depends on AG1's wait must not bucket together
    (would create a cycle on merge)."""
    g = torch.fx.Graph()
    p1 = g.placeholder("p1")
    p1.meta["val"] = _make_fake_tensor_meta((512,), torch.bfloat16)
    ag1 = _make_ag_node(g, p1, "dp", (512,), group_size=8)
    w1 = _make_wait_node(g, ag1)
    # AG2 takes w1 as input → AG2 is a descendant of AG1.
    ag2 = _make_ag_node(g, w1, "dp", (512,), group_size=8)
    w2 = _make_wait_node(g, ag2)
    g.output((w2,))
    gm = _build_gm(g)

    ab.aten_autobucketing_config.max_topo_span = 1000
    try:
        buckets = _call_greedy_bucket(gm, bucket_cap_mb=10.0)
    finally:
        ab.aten_autobucketing_config.max_topo_span = 1500

    # Either no buckets or a single-element bucket (which the impl drops).
    # The two AGs must never end up together.
    for b in buckets:
        assert not (
            ag1 in b and ag2 in b
        ), "AG1 and AG2 ended up in same bucket despite descendant relation"
