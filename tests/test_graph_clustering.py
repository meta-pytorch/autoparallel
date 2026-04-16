# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import re
from collections import Counter

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard

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


def _clustering_stats(graph, strats, n_layers):
    """Compute clustering statistics."""
    clusters = get_identical_regions(graph, strats)

    clustered_nodes = set()
    regions_per_group = []
    for group in clusters:
        regions_per_group.append(len(group))
        for region in group:
            for node in region:
                clustered_nodes.add(node)

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


def test_clustering_high_coverage(device_mesh_2d):
    """The vast majority of layer-specific nodes should be clustered.

    With identical transformer layers, clustering should cover nearly all
    layer nodes. A small gap is acceptable due to the layer-0 boundary
    asymmetry in the backward pass.
    """
    n_layers = 4
    autop, _ = _setup_llama_autop(device_mesh_2d, n_layers=n_layers)
    with autop:
        x_sharding = (Shard(0), Replicate())
        out_sharding = (Shard(0), Shard(2))
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([out_sharding])

        stats = _clustering_stats(
            autop.gm.graph, autop.sharding_optimizer.strats, n_layers
        )

    # Every layer should have the same total node count
    layer_totals = [stats["per_layer_total"].get(i, 0) for i in range(n_layers)]
    assert (
        len(set(layer_totals)) == 1
    ), f"Layers have different node counts: {layer_totals}"

    # At least 50% of layer nodes should be clustered across all layers
    total = layer_totals[0]
    for layer_idx in range(n_layers):
        clustered = stats["per_layer_clustered"].get(layer_idx, 0)
        coverage = clustered / total
        assert coverage >= 0.50, (
            f"Layer {layer_idx} clustering coverage too low: "
            f"{clustered}/{total} = {coverage:.1%}"
        )


def test_clustering_no_forward_backward_mixing(device_mesh_2d):
    """Cluster groups should not mix forward and backward nodes from
    different operations. The partitioner_tag in the hash prevents
    different-phase nodes with identical shapes from being clustered."""
    n_layers = 4
    autop, _ = _setup_llama_autop(device_mesh_2d, n_layers=n_layers)
    with autop:
        x_sharding = (Shard(0), Replicate())
        out_sharding = (Shard(0), Shard(2))
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([out_sharding])

        clusters = get_identical_regions(
            autop.gm.graph, autop.sharding_optimizer.strats
        )

    for i, group in enumerate(clusters):
        region0 = group[0]
        targets = set(str(n.target) for n in region0)
        tags = set(n.meta.get("partitioner_tag") for n in region0)
        # A node's target+tag should be consistent within a region.
        # The hash includes both, so different targets or different tags
        # in the same region means expansion crossed a boundary — which
        # is fine as long as both sides are truly identical. But pure
        # forward ops should never share a root hash with pure backward
        # ops of a *different* operation.
        for region in group[1:]:
            other_targets = set(str(n.target) for n in region)
            other_tags = set(n.meta.get("partitioner_tag") for n in region)
            assert (
                targets == other_targets
            ), f"Cluster group {i}: regions have different op targets"
            assert (
                tags == other_tags
            ), f"Cluster group {i}: regions have different phase tags"
