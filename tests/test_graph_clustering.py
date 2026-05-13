# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import operator
import re
from collections import Counter

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel._testing.models.dsv3 import DeepSeekV3Model, make_dsv3_config
from autoparallel._testing.models.llama3 import Transformer, TransformerModelArgs
from autoparallel.api import AutoParallel
from autoparallel.graph_passes.graph_clustering import get_identical_regions


def _get_layer_index(node):
    """Extract the transformer layer index from a node's nn_module_stack."""
    nn_stack = node.meta.get("nn_module_stack") or node.meta.get("fwd_nn_module_stack")
    if nn_stack:
        for _key, (module_path, _cls) in nn_stack.items():
            m = re.search(r"layers\.(\d+)", module_path)
            if m:
                return int(m.group(1))
    return None


def _clustered_nodes(clusters):
    """Return every FX node that appears in any final clustered region."""
    clustered = set()
    for group in clusters:
        for region in group:
            clustered.update(region)
    return clustered


def _clustering_stats(graph, strats, clusters, n_layers):
    """Compute per-layer clustering coverage from precomputed clusters."""
    clustered_nodes = _clustered_nodes(clusters)
    regions_per_group = []
    for group in clusters:
        regions_per_group.append(len(group))

    per_layer_clustered = Counter()
    per_layer_total = Counter()
    total_with_strats = 0
    for node in graph.nodes:
        if node not in strats:
            continue
        total_with_strats += 1
        layer_idx = _get_layer_index(node)
        if layer_idx is not None:
            per_layer_total[layer_idx] += 1
            if node in clustered_nodes:
                per_layer_clustered[layer_idx] += 1

    return {
        "total_nodes": total_with_strats,
        "clustered_nodes": len(clustered_nodes),
        "cluster_groups": len(clusters),
        "per_layer_clustered": dict(per_layer_clustered),
        "per_layer_total": dict(per_layer_total),
        "regions_per_group": regions_per_group,
    }


def _assert_layer_coverage(stats, n_layers, min_coverage, label, layers=None):
    """Assert each repeated layer has enough nodes in clustered regions."""
    layers = range(n_layers) if layers is None else layers
    layer_totals = [stats["per_layer_total"].get(i, 0) for i in layers]
    if len(set(layer_totals)) != 1:
        raise AssertionError(
            f"{label}: layers have different node counts: {layer_totals}"
        )

    total = layer_totals[0]
    if total == 0:
        raise AssertionError(f"{label}: no layer nodes found in clustering stats")
    for layer_idx in layers:
        clustered = stats["per_layer_clustered"].get(layer_idx, 0)
        coverage = clustered / total
        if coverage < min_coverage:
            raise AssertionError(
                f"{label}: layer {layer_idx} clustering coverage too low: "
                f"{clustered}/{total} = {coverage:.1%}, "
                f"expected >= {min_coverage:.1%}"
            )


def _assert_cross_layer_cluster(
    clusters, expected_layers, min_region_size, phase, label
):
    """Assert there is one large same-phase region for each expected layer."""
    expected_layers = set(expected_layers)
    for group in clusters:
        if len(group) != len(expected_layers):
            continue
        region_layers = []
        for region in group:
            tags = {n.meta.get("partitioner_tag") for n in region}
            tags.discard(None)
            if tags != {phase}:
                break
            layers = {idx for n in region if (idx := _get_layer_index(n)) is not None}
            if len(region) < min_region_size or len(layers) != 1:
                break
            region_layers.append(next(iter(layers)))
        else:
            if set(region_layers) == expected_layers:
                return
    raise AssertionError(
        f"{label}: missing {phase} cross-layer cluster for layers "
        f"{sorted(expected_layers)} with min region size {min_region_size}"
    )


def _assert_no_forward_backward_mixing(clusters, label):
    """Assert clustered regions never contain both forward and backward nodes."""
    for i, group in enumerate(clusters):
        for j, region in enumerate(group):
            tags = set(n.meta.get("partitioner_tag") for n in region)
            tags.discard(None)
            if len(tags) > 1:
                raise AssertionError(
                    f"{label}: cluster group {i}, region {j} mixes phases: {tags}"
                )


def _run_clustering(autop, n_layers, input_sharding, output_sharding=None):
    """Build AutoParallel state and return clustering stats plus raw clusters."""
    if output_sharding is None:
        output_sharding = input_sharding
    with autop:
        autop.add_input_constraints([input_sharding])
        autop.add_output_constraints([output_sharding])

        graph = autop.sharding_optimizer.graph
        strats = autop.sharding_optimizer.strats
        clusters = get_identical_regions(graph, strats)
        stats = _clustering_stats(graph, strats, clusters, n_layers)
    return stats, clusters


def _assert_model_clustering(
    stats,
    clusters,
    *,
    label,
    n_layers,
    min_coverage,
    forward_layers,
    backward_layers,
    coverage_layers=None,
    min_region_size=100,
):
    """Assert coverage, phase separation, and large fwd/bwd layer clusters."""
    _assert_layer_coverage(
        stats,
        n_layers,
        min_coverage=min_coverage,
        label=label,
        layers=coverage_layers,
    )
    _assert_cross_layer_cluster(
        clusters,
        forward_layers,
        min_region_size=min_region_size,
        phase="is_forward",
        label=label,
    )
    _assert_cross_layer_cluster(
        clusters,
        backward_layers,
        min_region_size=min_region_size,
        phase="is_backward",
        label=label,
    )
    _assert_no_forward_backward_mixing(clusters, label=label)


def _setup_llama_autop(device_mesh_2d, n_layers=4):
    """Set up AutoParallel with a small LLaMA model."""
    vocab_size = 2048
    seqlen = 512
    batch_size = 2 * device_mesh_2d.shape[0]

    model_args = TransformerModelArgs(
        dim=256,
        n_layers=n_layers,
        n_heads=16,
        n_kv_heads=4,
        vocab_size=vocab_size,
        rope_theta=500000,
        max_seq_len=seqlen,
    )
    with torch.device("meta"):
        model = Transformer(model_args)

    def input_fn():
        return torch.randint(0, vocab_size, (batch_size, seqlen), device="cuda")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )

    autop = AutoParallel(
        model, input_fn, device_mesh_2d, mp_policy, repeated_subgraphs=True
    )
    return autop, model_args


def _setup_ds3_local_map_autop(device_mesh_2d):
    local_batch_size = 8
    seq_len = 2048
    global_batch_size = (
        local_batch_size * device_mesh_2d.shape[0] * device_mesh_2d.shape[1]
    )
    config = make_dsv3_config(max_seq_len=seq_len)
    with torch.device("meta"):
        model = DeepSeekV3Model(
            config,
            mesh=device_mesh_2d,
            compute_dtype=torch.bfloat16,
        )
    for module in model.modules():
        if hasattr(module, "axis_name"):
            module.axis_name = device_mesh_2d.mesh_dim_names[1]

    def input_fn():
        return torch.randint(
            0,
            config.vocab_size,
            (global_batch_size, config.rope.max_seq_len),
            device="cuda",
        )

    return AutoParallel(model, input_fn, device_mesh_2d, dynamic=True), config


def test_clustering_high_coverage(device_mesh_2d):
    """The vast majority of layer-specific nodes should be clustered.

    With identical transformer layers, clustering should cover nearly all
    layer nodes. A small gap is acceptable due to the layer-0 boundary
    asymmetry in the backward pass.
    """
    n_layers = 4
    autop, _ = _setup_llama_autop(device_mesh_2d, n_layers=n_layers)
    stats, clusters = _run_clustering(
        autop,
        n_layers,
        input_sharding=(Shard(0), Replicate()),
        output_sharding=(Shard(0), Shard(2)),
    )
    _assert_model_clustering(
        stats,
        clusters,
        label="LLaMA",
        n_layers=n_layers,
        min_coverage=0.50,
        forward_layers=range(n_layers),
        # Layer 0 has known backward boundary asymmetry.
        backward_layers=range(1, n_layers),
    )

    autop, config = _setup_ds3_local_map_autop(device_mesh_2d)
    n_layers = len(config.layers)
    stats, clusters = _run_clustering(
        autop,
        n_layers,
        input_sharding=(Shard(0), Shard(0)),
    )
    _assert_model_clustering(
        stats,
        clusters,
        label="DS3",
        n_layers=n_layers,
        min_coverage=0.75,
        coverage_layers=range(1, n_layers),
        forward_layers=range(1, n_layers),
        backward_layers=range(1, n_layers),
    )


def test_clustering_no_forward_backward_mixing(device_mesh_2d):
    """Each cluster group's regions should contain only forward or only
    backward nodes, never a mix. Expansion must not cross the phase boundary
    by following saved-tensor edges from backward into forward."""
    n_layers = 4
    autop, _ = _setup_llama_autop(device_mesh_2d, n_layers=n_layers)
    with autop:
        x_sharding = (Shard(0), Replicate())
        out_sharding = (Shard(0), Shard(2))
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([out_sharding])

        clusters = get_identical_regions(
            autop.sharding_optimizer.graph, autop.sharding_optimizer.strats
        )

    for i, group in enumerate(clusters):
        for j, region in enumerate(group):
            tags = set(n.meta.get("partitioner_tag") for n in region)
            tags.discard(None)
            assert len(tags) <= 1, f"Cluster group {i}, region {j} mixes phases: {tags}"


def test_getitem_siblings_are_clustered(device_mesh_2d):
    """Getitem nodes that project sibling outputs from tuple-returning ops
    (e.g. SDPA returning (output, logsumexp, rng_state)) should be clustered
    together with their producer when the producer is already clustered."""
    n_layers = 4
    autop, _ = _setup_llama_autop(device_mesh_2d, n_layers=n_layers)
    with autop:
        x_sharding = (Shard(0), Replicate())
        out_sharding = (Shard(0), Shard(2))
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([out_sharding])

        graph = autop.sharding_optimizer.graph
        strats = autop.sharding_optimizer.strats
        clusters = get_identical_regions(graph, strats)

    clustered_nodes = set()
    for group in clusters:
        for region in group:
            clustered_nodes.update(region)

    # Find all getitem nodes that have strategies and belong to layers
    unclustered_getitems = []
    for node in graph.nodes:
        if node.target is not operator.getitem:
            continue
        if node not in strats:
            continue
        layer_idx = _get_layer_index(node)
        if layer_idx is not None and node not in clustered_nodes:
            unclustered_getitems.append(node)

    assert len(unclustered_getitems) == 0, (
        f"{len(unclustered_getitems)} getitem nodes in layers are not clustered: "
        f"{[n.name for n in unclustered_getitems[:5]]}"
    )


def test_getitem_siblings_cluster_consistency(device_mesh_2d):
    """Getitem siblings added by _extend_with_sibling_getitems should appear
    in the same cluster group as their producer, with one per region."""
    n_layers = 4
    autop, _ = _setup_llama_autop(device_mesh_2d, n_layers=n_layers)
    with autop:
        x_sharding = (Shard(0), Replicate())
        out_sharding = (Shard(0), Shard(2))
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([out_sharding])

        graph = autop.sharding_optimizer.graph
        strats = autop.sharding_optimizer.strats
        clusters = get_identical_regions(graph, strats)

    for i, group in enumerate(clusters):
        num_regions = len(group)
        # Collect all getitem nodes across all regions in this group
        getitems_in_group = []
        for region in group:
            for node in region:
                if node.target is operator.getitem:
                    getitems_in_group.append(node)

        if not getitems_in_group:
            continue

        # Each getitem index should appear exactly once per region
        # Group getitems by their tuple index
        by_index = Counter(n.args[1] for n in getitems_in_group)
        for idx, count in by_index.items():
            assert count == num_regions, (
                f"Cluster group {i}: getitem[{idx}] appears {count} times "
                f"but there are {num_regions} regions"
            )
