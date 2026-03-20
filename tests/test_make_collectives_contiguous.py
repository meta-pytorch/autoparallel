# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.fx

from autoparallel.graph_passes.make_collectives_contiguous import (
    make_collectives_contiguous,
)


def _count_ops(gm, target):
    return len(gm.graph.find_nodes(op="call_function", target=target))


def _build_graph_with_collective(collective_target):
    """Build a simple FX graph: placeholder -> collective -> output."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.randn(8)
    collective = graph.call_function(collective_target, args=(x, 2, "0"))
    collective.meta["val"] = torch.randn(16)
    output = graph.output(collective)
    output.meta["val"] = collective.meta["val"]
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    return gm


def test_all_gather_gets_contiguous_clone():
    target = torch.ops._c10d_functional.all_gather_into_tensor.default
    gm = _build_graph_with_collective(target)

    assert _count_ops(gm, torch.ops.aten.clone.default) == 0
    make_collectives_contiguous(gm)
    assert _count_ops(gm, torch.ops.aten.clone.default) == 1

    # The clone should be the input to the collective
    for node in gm.graph.nodes:
        if node.target == target:
            clone_node = node.args[0]
            assert clone_node.target == torch.ops.aten.clone.default
            assert clone_node.kwargs["memory_format"] == torch.contiguous_format


def test_reduce_scatter_gets_contiguous_clone():
    target = torch.ops._c10d_functional.reduce_scatter_tensor.default
    gm = _build_graph_with_collective(target)

    make_collectives_contiguous(gm)
    assert _count_ops(gm, torch.ops.aten.clone.default) == 1


def test_already_contiguous_clone_is_not_duplicated():
    """If the input is already a contiguous clone, don't insert another."""
    target = torch.ops._c10d_functional.all_gather_into_tensor.default
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.randn(8)
    clone = graph.call_function(
        torch.ops.aten.clone.default,
        args=(x,),
        kwargs={"memory_format": torch.contiguous_format},
    )
    clone.meta["val"] = x.meta["val"]
    collective = graph.call_function(target, args=(clone, 2, "0"))
    collective.meta["val"] = torch.randn(16)
    output = graph.output(collective)
    output.meta["val"] = collective.meta["val"]
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    make_collectives_contiguous(gm)
    # Should still be exactly 1 clone, not 2
    assert _count_ops(gm, torch.ops.aten.clone.default) == 1


def test_non_collective_ops_untouched():
    """Ops that aren't collectives should not get a clone inserted."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.randn(4, 4)
    add = graph.call_function(torch.ops.aten.add.Tensor, args=(x, x))
    add.meta["val"] = torch.randn(4, 4)
    output = graph.output(add)
    output.meta["val"] = add.meta["val"]
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    make_collectives_contiguous(gm)
    assert _count_ops(gm, torch.ops.aten.clone.default) == 0


def test_multiple_collectives():
    """Each collective gets its own contiguous clone."""
    ag_target = torch.ops._c10d_functional.all_gather_into_tensor.default
    rs_target = torch.ops._c10d_functional.reduce_scatter_tensor.default

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.randn(8)
    ag = graph.call_function(ag_target, args=(x, 2, "0"))
    ag.meta["val"] = torch.randn(16)
    rs = graph.call_function(rs_target, args=(ag, "sum", 2, "0"))
    rs.meta["val"] = torch.randn(8)
    output = graph.output(rs)
    output.meta["val"] = rs.meta["val"]
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    make_collectives_contiguous(gm)
    assert _count_ops(gm, torch.ops.aten.clone.default) == 2


def test_shared_input_gets_separate_clones():
    """When two collectives share the same input, each gets its own clone."""
    target = torch.ops._c10d_functional.all_gather_into_tensor.default

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.randn(8)
    ag1 = graph.call_function(target, args=(x, 2, "0"))
    ag1.meta["val"] = torch.randn(16)
    ag2 = graph.call_function(target, args=(x, 4, "1"))
    ag2.meta["val"] = torch.randn(32)
    output = graph.output((ag1, ag2))
    output.meta["val"] = (ag1.meta["val"], ag2.meta["val"])
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    make_collectives_contiguous(gm)
    assert _count_ops(gm, torch.ops.aten.clone.default) == 2
