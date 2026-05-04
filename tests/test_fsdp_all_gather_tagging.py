# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for FSDP all-gather tagging (mark_fsdp_all_gather_recomputation).

These tests build minimal FX graphs that mimic the FSDP all-gather pattern
without running the full AutoParallel pipeline.
"""

import pytest
import torch
import torch.fx
from torch.utils.checkpoint import CheckpointPolicy

from autoparallel.graph_passes.activation_checkpointing import (
    AP_AC_GRAPH_ID,
    force_recompute_fsdp_all_gather,
    force_save_fsdp_all_gather,
    mark_fsdp_all_gather_recomputation,
)
from autoparallel.graph_passes.autobucketing_inductor import bucket_utils

# ---------------------------------------------------------------------------
# Helpers for building minimal FSDP-like graphs
# ---------------------------------------------------------------------------


def _new_graph() -> torch.fx.Graph:
    return torch.fx.Graph()


def _add_placeholder(graph: torch.fx.Graph, name: str) -> torch.fx.Node:
    node = graph.placeholder(name)
    node.meta["val"] = torch.empty(16)
    return node


def _add_all_gather(graph: torch.fx.Graph, input_node: torch.fx.Node) -> torch.fx.Node:
    node = graph.call_function(
        torch.ops._c10d_functional.all_gather_into_tensor.default,
        args=(input_node, 4, "0"),
    )
    node.meta["val"] = torch.empty(64)
    return node


def _add_wait_tensor(graph: torch.fx.Graph, input_node: torch.fx.Node) -> torch.fx.Node:
    node = graph.call_function(
        torch.ops._c10d_functional.wait_tensor.default,
        args=(input_node,),
    )
    node.meta["val"] = torch.empty(64)
    return node


def _add_mm(graph: torch.fx.Graph, a: torch.fx.Node, b: torch.fx.Node) -> torch.fx.Node:
    node = graph.call_function(torch.ops.aten.mm.default, args=(a, b))
    node.meta["val"] = torch.empty(8, 8)
    return node


def _add_output(graph: torch.fx.Graph, nodes: list[torch.fx.Node]) -> torch.fx.Node:
    return graph.output(tuple(nodes))


def _build_simple_fsdp_graph() -> torch.fx.Graph:
    """Build: placeholder -> all_gather -> wait_tensor -> mm (with another placeholder).

    The mm node simulates a downstream use in backward (no must_be_in_forward tag).
    """
    graph = _new_graph()
    param = _add_placeholder(graph, "param")
    activation = _add_placeholder(graph, "activation")
    ag = _add_all_gather(graph, param)
    wait = _add_wait_tensor(graph, ag)
    out = _add_mm(graph, wait, activation)
    _add_output(graph, [out])
    return graph


class _SchedulerNode:
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name


class _FusedSchedulerNode(_SchedulerNode):
    pass


class _GroupedSchedulerNode(_SchedulerNode):
    pass


# ---------------------------------------------------------------------------
# Tests for force_recompute_fsdp_all_gather
# ---------------------------------------------------------------------------


def test_force_recompute_tags_ag_and_wait():
    """all_gather and wait_tensor nodes get MUST_RECOMPUTE + ac_graph_id."""
    graph = _build_simple_fsdp_graph()
    force_recompute_fsdp_all_gather(graph)

    for node in graph.nodes:
        if node.target == torch.ops._c10d_functional.all_gather_into_tensor.default:
            assert node.meta["recompute"] is CheckpointPolicy.MUST_RECOMPUTE
            assert node.meta["ac_graph_id"] == AP_AC_GRAPH_ID
        elif node.target == torch.ops._c10d_functional.wait_tensor.default:
            assert node.meta["recompute"] is CheckpointPolicy.MUST_RECOMPUTE
            assert node.meta["ac_graph_id"] == AP_AC_GRAPH_ID


def test_force_recompute_tags_slice_after_wait():
    """A slice node directly after wait_tensor also gets MUST_RECOMPUTE."""
    graph = _new_graph()
    param = _add_placeholder(graph, "param")
    activation = _add_placeholder(graph, "activation")
    ag = _add_all_gather(graph, param)
    wait = _add_wait_tensor(graph, ag)
    sliced = graph.call_function(torch.ops.aten.slice.Tensor, args=(wait, 0, 0, 32))
    sliced.meta["val"] = torch.empty(32)
    out = _add_mm(graph, sliced, activation)
    _add_output(graph, [out])

    force_recompute_fsdp_all_gather(graph)

    for node in graph.nodes:
        if node.target == torch.ops.aten.slice.Tensor:
            assert node.meta["recompute"] is CheckpointPolicy.MUST_RECOMPUTE
            assert node.meta["ac_graph_id"] == AP_AC_GRAPH_ID


def test_force_recompute_tags_dtype_cast_before_ag():
    """A convert_element_type before all_gather also gets MUST_RECOMPUTE."""
    graph = _new_graph()
    param = _add_placeholder(graph, "param")
    activation = _add_placeholder(graph, "activation")
    cast = graph.call_function(
        torch.ops.prims.convert_element_type.default,
        args=(param, torch.float32),
    )
    cast.meta["val"] = torch.empty(16)
    ag = _add_all_gather(graph, cast)
    wait = _add_wait_tensor(graph, ag)
    out = _add_mm(graph, wait, activation)
    _add_output(graph, [out])

    force_recompute_fsdp_all_gather(graph)

    for node in graph.nodes:
        if node.target == torch.ops.prims.convert_element_type.default:
            assert node.meta["recompute"] is CheckpointPolicy.MUST_RECOMPUTE
            assert node.meta["ac_graph_id"] == AP_AC_GRAPH_ID


def test_force_recompute_ignores_non_fsdp_all_gather():
    """all_gather nodes that don't trace back to a placeholder are skipped."""
    graph = _new_graph()
    a = _add_placeholder(graph, "a")
    b = _add_placeholder(graph, "b")
    # mm has two inputs, so the chain from placeholder is not single-input
    mm = _add_mm(graph, a, b)
    ag = _add_all_gather(graph, mm)
    wait = _add_wait_tensor(graph, ag)
    _add_output(graph, [wait])

    force_recompute_fsdp_all_gather(graph)

    for node in graph.nodes:
        assert "recompute" not in node.meta


# ---------------------------------------------------------------------------
# Tests for force_save_fsdp_all_gather
# ---------------------------------------------------------------------------


def test_force_save_tags_last_chain_node():
    """The last non-view node after wait_tensor gets MUST_SAVE."""
    graph = _new_graph()
    param = _add_placeholder(graph, "param")
    activation = _add_placeholder(graph, "activation")
    ag = _add_all_gather(graph, param)
    wait = _add_wait_tensor(graph, ag)
    # mm is multi-input so it breaks the chain — wait is the last chain node
    out = _add_mm(graph, wait, activation)
    # Mark mm as not-must-be-in-forward so the assertion in force_save passes
    _add_output(graph, [out])

    force_save_fsdp_all_gather(graph)

    for node in graph.nodes:
        if node.target == torch.ops._c10d_functional.wait_tensor.default:
            assert node.meta["recompute"] is CheckpointPolicy.MUST_SAVE
            assert node.meta["ac_graph_id"] == AP_AC_GRAPH_ID


# ---------------------------------------------------------------------------
# Tests for mark_fsdp_all_gather_recomputation (the dispatch function)
# ---------------------------------------------------------------------------


def test_mark_fsdp_recompute_when_reshard_after_forward():
    """reshard_after_forward=True → all_gather/wait get MUST_RECOMPUTE."""
    graph = _build_simple_fsdp_graph()
    mark_fsdp_all_gather_recomputation(graph, reshard_after_forward=True)

    for node in graph.nodes:
        if node.target == torch.ops._c10d_functional.all_gather_into_tensor.default:
            assert node.meta["recompute"] is CheckpointPolicy.MUST_RECOMPUTE
        elif node.target == torch.ops._c10d_functional.wait_tensor.default:
            assert node.meta["recompute"] is CheckpointPolicy.MUST_RECOMPUTE


def test_mark_fsdp_save_when_no_reshard_after_forward():
    """reshard_after_forward=False → wait chain end gets MUST_SAVE."""
    graph = _build_simple_fsdp_graph()
    mark_fsdp_all_gather_recomputation(graph, reshard_after_forward=False)

    found_save = False
    for node in graph.nodes:
        if node.meta.get("recompute") is CheckpointPolicy.MUST_SAVE:
            found_save = True
    assert found_save, "Expected at least one MUST_SAVE node"


def test_no_tags_without_fsdp_pattern():
    """Graph without any FSDP all-gather pattern gets no tags at all."""
    graph = _new_graph()
    a = _add_placeholder(graph, "a")
    b = _add_placeholder(graph, "b")
    out = _add_mm(graph, a, b)
    _add_output(graph, [out])

    mark_fsdp_all_gather_recomputation(graph, reshard_after_forward=True)

    for node in graph.nodes:
        assert "recompute" not in node.meta
        assert "ac_graph_id" not in node.meta


# ---------------------------------------------------------------------------
# Tests for autobucketing scheduler helpers
# ---------------------------------------------------------------------------


def test_get_op_idx():
    assert bucket_utils.get_op_idx(_SchedulerNode("op142")) == 142


def test_get_op_idx_rejects_non_op_name():
    with pytest.raises(KeyError, match="Expected op name"):
        bucket_utils.get_op_idx(_SchedulerNode("buf142"))


def test_get_op_idx_rejects_fused_and_grouped_snodes(monkeypatch):
    monkeypatch.setattr(
        bucket_utils.scheduler, "FusedSchedulerNode", _FusedSchedulerNode
    )
    monkeypatch.setattr(
        bucket_utils.scheduler, "GroupedSchedulerNode", _GroupedSchedulerNode
    )

    for node_cls in (_FusedSchedulerNode, _GroupedSchedulerNode):
        with pytest.raises(TypeError, match="Expected an unfused scheduler node"):
            bucket_utils.get_op_idx(node_cls("op142"))
