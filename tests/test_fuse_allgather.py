# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for chained allgather fusion pass.

These tests build minimal FX graphs that mimic chained allgather patterns
without running the full AutoParallel pipeline or needing real process groups.
"""

import pytest
import torch
import torch.fx

from autoparallel.api import _suppress_wait_tensor_side_effect
from autoparallel.graph_passes.fuse_allgather import fuse_chained_allgathers

AG = torch.ops._c10d_functional.all_gather_into_tensor.default
WAIT = torch.ops._c10d_functional.wait_tensor.default
PERMUTE = torch.ops.aten.permute.default


@pytest.fixture(autouse=True)
def suppress_wait_side_effect():
    """Allow DCE to remove wait_tensor nodes, matching the runtime environment."""
    with _suppress_wait_tensor_side_effect():
        yield


def _count_ops(graph, target):
    return len(graph.find_nodes(op="call_function", target=target))


def _add_placeholder(graph, name, shape):
    node = graph.placeholder(name)
    node.meta["val"] = torch.empty(*shape)
    return node


def _add_all_gather(graph, input_node, group_size, group_name):
    in_shape = input_node.meta["val"].shape
    out_shape = (in_shape[0] * group_size, *in_shape[1:])
    node = graph.call_function(AG, args=(input_node, group_size, group_name))
    node.meta["val"] = torch.empty(*out_shape, dtype=input_node.meta["val"].dtype)
    return node


def _add_wait_tensor(graph, input_node):
    node = graph.call_function(WAIT, args=(input_node,))
    node.meta["val"] = input_node.meta["val"].clone()
    return node


def _add_permute(graph, input_node, dims):
    node = graph.call_function(PERMUTE, args=(input_node, dims))
    in_val = input_node.meta["val"]
    node.meta["val"] = in_val.permute(dims)
    return node


def _add_chained_allgather(graph, input_node, size1=16, pg1="dp", size2=8, pg2="tp"):
    """Build: input -> ag1 -> wait -> permute([1,0]) -> permute([1,0]) -> ag2 -> wait."""
    ag1 = _add_all_gather(graph, input_node, size1, pg1)
    wait1 = _add_wait_tensor(graph, ag1)
    p1 = _add_permute(graph, wait1, [1, 0])
    p2 = _add_permute(graph, p1, [1, 0])
    ag2 = _add_all_gather(graph, p2, size2, pg2)
    wait2 = _add_wait_tensor(graph, ag2)
    return wait2


def test_basic_fusion():
    """Detects chained allgathers and fuses into a single full-mesh allgather."""
    graph = torch.fx.Graph()
    x = _add_placeholder(graph, "x", (8, 4096))
    result = _add_chained_allgather(graph, x)
    graph.output((result,))

    assert _count_ops(graph, AG) == 2
    assert _count_ops(graph, WAIT) == 2
    assert _count_ops(graph, PERMUTE) == 2

    fusions = fuse_chained_allgathers(
        graph, full_group_size=128, full_group_name="full"
    )

    assert fusions == 1
    assert _count_ops(graph, AG) == 1
    assert _count_ops(graph, WAIT) == 1
    assert _count_ops(graph, PERMUTE) == 0

    ag_node = graph.find_nodes(op="call_function", target=AG)[0]
    assert ag_node.args[1] == 128
    assert ag_node.args[2] == "full"


def test_multiple_chains():
    """Multiple independent chains in the same graph are all fused."""
    graph = torch.fx.Graph()
    x = _add_placeholder(graph, "x", (8, 4096))
    y = _add_placeholder(graph, "y", (8, 4096))
    r1 = _add_chained_allgather(graph, x)
    r2 = _add_chained_allgather(graph, y)
    graph.output((r1, r2))

    assert _count_ops(graph, AG) == 4

    fusions = fuse_chained_allgathers(
        graph, full_group_size=128, full_group_name="full"
    )

    assert fusions == 2
    assert _count_ops(graph, AG) == 2
    assert _count_ops(graph, WAIT) == 2


def test_no_fusion_wrong_permute():
    """Non-[1,0] permutes prevent fusion (view chain doesn't trace through)."""
    graph = torch.fx.Graph()
    x = _add_placeholder(graph, "x", (8, 4096))
    ag1 = _add_all_gather(graph, x, 16, "dp")
    wait1 = _add_wait_tensor(graph, ag1)
    p1 = _add_permute(graph, wait1, [1, 0])
    p2 = _add_permute(graph, p1, [0, 1])  # identity, not [1, 0]
    ag2 = _add_all_gather(graph, p2, 8, "tp")
    wait2 = _add_wait_tensor(graph, ag2)
    graph.output((wait2,))

    fusions = fuse_chained_allgathers(
        graph, full_group_size=128, full_group_name="full"
    )

    assert fusions == 0
    assert _count_ops(graph, AG) == 2


def test_no_fusion_wait_multiple_users():
    """If the first wait has other users, the intermediate result is consumed elsewhere."""
    graph = torch.fx.Graph()
    x = _add_placeholder(graph, "x", (8, 4096))
    ag1 = _add_all_gather(graph, x, 16, "dp")
    wait1 = _add_wait_tensor(graph, ag1)
    p1 = _add_permute(graph, wait1, [1, 0])
    p2 = _add_permute(graph, p1, [1, 0])
    ag2 = _add_all_gather(graph, p2, 8, "tp")
    wait2 = _add_wait_tensor(graph, ag2)
    # wait1 also used directly in output
    graph.output((wait2, wait1))

    fusions = fuse_chained_allgathers(
        graph, full_group_size=128, full_group_name="full"
    )

    assert fusions == 0
    assert _count_ops(graph, AG) == 2


def test_no_fusion_group_size_mismatch():
    """If size1 * size2 != full_group_size, no fusion occurs."""
    graph = torch.fx.Graph()
    x = _add_placeholder(graph, "x", (8, 4096))
    result = _add_chained_allgather(graph, x, size1=16, size2=8)
    graph.output((result,))

    # Wrong full_group_size
    fusions = fuse_chained_allgathers(graph, full_group_size=64, full_group_name="full")

    assert fusions == 0
    assert _count_ops(graph, AG) == 2


def test_no_fusion_same_group():
    """Two allgathers on the same process group are not fused."""
    graph = torch.fx.Graph()
    x = _add_placeholder(graph, "x", (8, 4096))
    result = _add_chained_allgather(graph, x, size1=4, pg1="dp", size2=4, pg2="dp")
    graph.output((result,))

    fusions = fuse_chained_allgathers(graph, full_group_size=16, full_group_name="full")

    assert fusions == 0


def test_subgroup_order_validation():
    """When subgroup_order is provided, only matching groups in valid order fuse."""
    graph = torch.fx.Graph()
    x = _add_placeholder(graph, "x", (8, 4096))
    result = _add_chained_allgather(graph, x, size1=16, pg1="dp", size2=8, pg2="tp")
    graph.output((result,))

    # Unknown subgroup names — should not fuse
    fusions = fuse_chained_allgathers(
        graph,
        full_group_size=128,
        full_group_name="full",
        subgroup_order={"other1": 0, "other2": 1},
    )
    assert fusions == 0
    assert _count_ops(graph, AG) == 2

    # dp(0)→tp(1) ascending chain: needs reversed_full_group_name
    fusions = fuse_chained_allgathers(
        graph,
        full_group_size=128,
        full_group_name="full",
        subgroup_order={"dp": 0, "tp": 1},
        reversed_full_group_name="full_rev",
    )
    assert fusions == 1
    assert _count_ops(graph, AG) == 1

    # Verify the fused AG uses the reversed group
    ag_node = graph.find_nodes(op="call_function", target=AG)[0]
    assert ag_node.args[2] == "full_rev"


def test_descending_subgroup_order_fuses():
    """Descending mesh-dim order (tp→dp) fuses with the default flat mesh."""
    graph = torch.fx.Graph()
    x = _add_placeholder(graph, "x", (8, 4096))
    result = _add_chained_allgather(graph, x, size1=8, pg1="tp", size2=16, pg2="dp")
    graph.output((result,))

    fusions = fuse_chained_allgathers(
        graph,
        full_group_size=128,
        full_group_name="full",
        subgroup_order={"dp": 0, "tp": 1},
    )

    assert fusions == 1
    assert _count_ops(graph, AG) == 1

    ag_node = graph.find_nodes(op="call_function", target=AG)[0]
    assert ag_node.args[2] == "full"


def test_with_cast_before_allgather():
    """The cast before the first allgather is preserved and becomes the fused ag's input."""
    graph = torch.fx.Graph()
    x = _add_placeholder(graph, "x", (8, 4096))
    # Simulate dtype cast
    cast = graph.call_function(
        torch.ops.prims.convert_element_type.default, args=(x, torch.bfloat16)
    )
    cast.meta["val"] = x.meta["val"].to(torch.bfloat16)
    result = _add_chained_allgather(graph, cast)
    graph.output((result,))

    fusions = fuse_chained_allgathers(
        graph, full_group_size=128, full_group_name="full"
    )

    assert fusions == 1
    assert _count_ops(graph, AG) == 1

    # Cast should still be present
    cast_count = _count_ops(graph, torch.ops.prims.convert_element_type.default)
    assert cast_count == 1

    # The allgather input should be the cast output
    ag_node = graph.find_nodes(op="call_function", target=AG)[0]
    assert ag_node.args[0].target == torch.ops.prims.convert_element_type.default


def test_standalone_allgather_untouched():
    """A single allgather without a chain is not affected."""
    graph = torch.fx.Graph()
    x = _add_placeholder(graph, "x", (8, 4096))
    ag = _add_all_gather(graph, x, 8, "tp")
    wait = _add_wait_tensor(graph, ag)
    graph.output((wait,))

    fusions = fuse_chained_allgathers(
        graph, full_group_size=128, full_group_name="full"
    )

    assert fusions == 0
    assert _count_ops(graph, AG) == 1


def test_direct_chain_no_views():
    """Two allgathers directly chained (ag1 -> wait -> ag2) with no view ops.

    Without subgroup_order, this is NOT fuseable (can't validate direction).
    """
    graph = torch.fx.Graph()
    x = _add_placeholder(graph, "x", (8, 4096))
    ag1 = _add_all_gather(graph, x, 16, "dp")
    wait1 = _add_wait_tensor(graph, ag1)
    ag2 = _add_all_gather(graph, wait1, 8, "tp")
    wait2 = _add_wait_tensor(graph, ag2)
    graph.output((wait2,))

    fusions = fuse_chained_allgathers(
        graph, full_group_size=128, full_group_name="full"
    )

    assert fusions == 0
    assert _count_ops(graph, AG) == 2
    assert _count_ops(graph, WAIT) == 2


def test_direct_chain_ascending_with_reversed_group():
    """Direct dp→tp chain fuses when subgroup_order and reversed group are provided."""
    graph = torch.fx.Graph()
    x = _add_placeholder(graph, "x", (8, 4096))
    ag1 = _add_all_gather(graph, x, 16, "dp")
    wait1 = _add_wait_tensor(graph, ag1)
    ag2 = _add_all_gather(graph, wait1, 8, "tp")
    wait2 = _add_wait_tensor(graph, ag2)
    graph.output((wait2,))

    fusions = fuse_chained_allgathers(
        graph,
        full_group_size=128,
        full_group_name="full",
        subgroup_order={"dp": 0, "tp": 1},
        reversed_full_group_name="full_rev",
    )

    assert fusions == 1
    assert _count_ops(graph, AG) == 1
    assert _count_ops(graph, WAIT) == 1

    ag_node = graph.find_nodes(op="call_function", target=AG)[0]
    assert ag_node.args[1] == 128
    assert ag_node.args[2] == "full_rev"


def test_direct_chain_descending_with_subgroup_order():
    """Direct tp→dp chain fuses with the default flat mesh."""
    graph = torch.fx.Graph()
    x = _add_placeholder(graph, "x", (8, 4096))
    ag1 = _add_all_gather(graph, x, 8, "tp")
    wait1 = _add_wait_tensor(graph, ag1)
    ag2 = _add_all_gather(graph, wait1, 16, "dp")
    wait2 = _add_wait_tensor(graph, ag2)
    graph.output((wait2,))

    fusions = fuse_chained_allgathers(
        graph,
        full_group_size=128,
        full_group_name="full",
        subgroup_order={"dp": 0, "tp": 1},
    )

    assert fusions == 1
    assert _count_ops(graph, AG) == 1

    ag_node = graph.find_nodes(op="call_function", target=AG)[0]
    assert ag_node.args[2] == "full"


def test_ascending_without_reversed_group_does_not_fuse():
    """dp→tp chain without reversed_full_group_name cannot fuse."""
    graph = torch.fx.Graph()
    x = _add_placeholder(graph, "x", (8, 4096))
    ag1 = _add_all_gather(graph, x, 16, "dp")
    wait1 = _add_wait_tensor(graph, ag1)
    ag2 = _add_all_gather(graph, wait1, 8, "tp")
    wait2 = _add_wait_tensor(graph, ag2)
    graph.output((wait2,))

    fusions = fuse_chained_allgathers(
        graph,
        full_group_size=128,
        full_group_name="full",
        subgroup_order={"dp": 0, "tp": 1},
        # no reversed_full_group_name
    )

    assert fusions == 0
    assert _count_ops(graph, AG) == 2


def test_noop_view_chain():
    """A no-op view/reshape between allgathers does not reconcile rank ordering.

    Even though a view op is present, if strides never change the chain is
    semantically equivalent to a direct chain and must not be fused.
    """
    graph = torch.fx.Graph()
    x = _add_placeholder(graph, "x", (8, 4096))
    ag1 = _add_all_gather(graph, x, 16, "dp")
    wait1 = _add_wait_tensor(graph, ag1)
    # No-op view: same shape, same strides
    view = graph.call_function(torch.ops.aten.view.default, args=(wait1, [128, 4096]))
    view.meta["val"] = wait1.meta["val"].clone()
    ag2 = _add_all_gather(graph, view, 8, "tp")
    wait2 = _add_wait_tensor(graph, ag2)
    graph.output((wait2,))

    fusions = fuse_chained_allgathers(
        graph, full_group_size=128, full_group_name="full"
    )

    assert fusions == 0
    assert _count_ops(graph, AG) == 2


def test_unsqueeze_squeeze_chain():
    """A temporary stride change without dimension reorder is not fuseable."""
    graph = torch.fx.Graph()
    x = _add_placeholder(graph, "x", (8, 4096))
    ag1 = _add_all_gather(graph, x, 16, "dp")
    wait1 = _add_wait_tensor(graph, ag1)
    unsqueeze = graph.call_function(torch.ops.aten.unsqueeze.default, args=(wait1, 0))
    unsqueeze.meta["val"] = wait1.meta["val"].unsqueeze(0)
    squeeze = graph.call_function(torch.ops.aten.squeeze.dim, args=(unsqueeze, 0))
    squeeze.meta["val"] = unsqueeze.meta["val"].squeeze(0)
    ag2 = _add_all_gather(graph, squeeze, 8, "tp")
    wait2 = _add_wait_tensor(graph, ag2)
    graph.output((wait2,))

    fusions = fuse_chained_allgathers(
        graph, full_group_size=128, full_group_name="full"
    )

    assert fusions == 0
    assert _count_ops(graph, AG) == 2


def test_fuse_with_device_mesh(device_mesh_2d):
    """Integration test: build fuse pass from a real 2D DeviceMesh and verify
    that direct dp→tp chains fuse using the column-major process group."""
    import torch.distributed as dist

    mesh = device_mesh_2d
    dp_size = mesh.size(0)
    tp_size = mesh.size(1)
    full_size = dp_size * tp_size

    # Row-major flat mesh (default)
    flat_mesh = mesh._flatten()
    full_group_name = flat_mesh.get_group().group_name

    # Column-major process group via sort_ranks=False
    col_major_ranks = mesh.mesh.T.contiguous().view(-1).tolist()
    reversed_pg = dist.new_group(ranks=col_major_ranks, sort_ranks=False)
    reversed_full_group_name = reversed_pg.group_name

    assert full_group_name != reversed_full_group_name

    subgroup_order = {
        mesh.get_group(mesh_dim=dim).group_name: dim for dim in range(mesh.ndim)
    }
    dp_pg = mesh.get_group(mesh_dim=0).group_name
    tp_pg = mesh.get_group(mesh_dim=1).group_name
    assert dp_pg != tp_pg

    # Build a direct dp→tp chain using real PG names.
    graph = torch.fx.Graph()
    x = _add_placeholder(graph, "x", (1, 4096))
    ag1 = _add_all_gather(graph, x, dp_size, dp_pg)
    wait1 = _add_wait_tensor(graph, ag1)
    ag2 = _add_all_gather(graph, wait1, tp_size, tp_pg)
    wait2 = _add_wait_tensor(graph, ag2)
    graph.output((wait2,))

    fusions = fuse_chained_allgathers(
        graph,
        full_group_size=full_size,
        full_group_name=full_group_name,
        subgroup_order=subgroup_order,
        reversed_full_group_name=reversed_full_group_name,
    )

    assert fusions == 1
    assert _count_ops(graph, AG) == 1

    ag_node = graph.find_nodes(op="call_function", target=AG)[0]
    assert ag_node.args[1] == full_size
    assert ag_node.args[2] == reversed_full_group_name


def test_col_major_rank_ordering_correctness():
    """Verify that allgather on the column-major process group produces the
    same result as two sequential allgathers (dp then tp) for reverse shard
    order data.

    For reverse shard order S(0)S(0) on a (dp, tp) mesh, tp is the outer
    shard and dp is the inner shard.  Rank (dp=i, tp=j) holds the shard at
    position j*dp_size + i in the full tensor.

    Sequential AG(dp) then AG(tp) recovers the full tensor in correct order.
    A single AG on the column-major group must produce the same ordering.
    """
    dp_size, tp_size = 4, 2
    full_size = dp_size * tp_size
    chunk = 3  # rows per shard

    # Per-rank shards: rank r at (dp=r//tp, tp=r%tp) holds shard j*dp+i
    rank_to_shard = {}
    for r in range(full_size):
        dp_idx, tp_idx = r // tp_size, r % tp_size
        shard_pos = tp_idx * dp_size + dp_idx
        rank_to_shard[r] = torch.arange(
            shard_pos * chunk, (shard_pos + 1) * chunk, dtype=torch.float
        )

    # --- Sequential: AG(dp) then AG(tp) ---
    # AG(dp): for each tp group, gather dp shards in rank order.
    after_ag_dp = {}
    for tp_idx in range(tp_size):
        dp_group = sorted(r for r in range(full_size) if r % tp_size == tp_idx)
        gathered = torch.cat([rank_to_shard[r] for r in dp_group])
        for r in dp_group:
            after_ag_dp[r] = gathered

    # AG(tp): for each dp group, gather tp results in rank order.
    for dp_idx in range(dp_size):
        tp_group = sorted(r for r in range(full_size) if r // tp_size == dp_idx)
        sequential_result = torch.cat([after_ag_dp[r] for r in tp_group])
        break  # all ranks get the same result

    # --- Fused: single AG on column-major group ---
    col_major_ranks = []
    for tp_idx in range(tp_size):
        for dp_idx in range(dp_size):
            col_major_ranks.append(dp_idx * tp_size + tp_idx)
    fused_result = torch.cat([rank_to_shard[r] for r in col_major_ranks])

    # Both must produce the original tensor in order [0, 1, 2, ..., N*chunk-1]
    expected = torch.arange(full_size * chunk, dtype=torch.float)
    torch.testing.assert_close(sequential_result, expected)
    torch.testing.assert_close(fused_result, expected)
