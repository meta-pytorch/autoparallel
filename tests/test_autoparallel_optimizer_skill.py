# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from autoparallel.cost_models.collective_runtime_estimation import (
    get_nccl_topo_config,
    set_nccl_topo_config,
)
from autoparallel.graph_passes.auto_bucketing import aten_autobucketing_config

ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / ".agents/skills/autoparallel-optimizer"
EVAL_ROOT = ROOT / "tests/agent_skills/autoparallel_optimizer"


def _load_collector_module():
    path = SKILL_ROOT / "scripts/collect_reordered_metrics.py"
    spec = importlib.util.spec_from_file_location(
        "autoparallel_skill_collect_reordered_metrics", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


collector_module = _load_collector_module()
ReorderedMetricsCollector = collector_module.ReorderedMetricsCollector
classify_graph_phase = collector_module.classify_graph_phase


class _UnusedMesh:
    pass


def _make_graph(*, backward=False):
    gm = make_fx(lambda x: torch.sin(x), tracing_mode="fake")(torch.ones(4))
    if backward:
        placeholder = next(node for node in gm.graph.nodes if node.op == "placeholder")
        placeholder.name = "tangents_1"
        placeholder.target = "tangents_1"
        gm.recompile()
    return gm


def _use_zero_runtime_estimator(collector):
    def estimator(_node):
        return 0.0

    collector.runtime_estimator = estimator
    collector.reordering_config.custom_runtime_estimation = estimator


@pytest.fixture(autouse=True)
def preserve_collector_globals():
    topology = get_nccl_topo_config()
    max_topo_span = aten_autobucketing_config.max_topo_span
    post_grad_pass = torch._inductor.config.post_grad_custom_post_pass
    yield
    set_nccl_topo_config(topology)
    aten_autobucketing_config.max_topo_span = max_topo_span
    torch._inductor.config.post_grad_custom_post_pass = post_grad_pass


def test_classify_graph_phase_uses_tangents_and_partitioner_tags():
    forward = _make_graph()
    backward = _make_graph(backward=True)

    assert classify_graph_phase(forward.graph) == "forward"
    assert classify_graph_phase(backward.graph) == "backward"

    call_node = next(node for node in forward.graph.nodes if node.op == "call_function")
    call_node.meta["partitioner_tag"] = "is_backward"
    assert classify_graph_phase(forward.graph) == "backward"


def test_classify_graph_phase_treats_mixed_tags_without_tangents_as_forward():
    gm = make_fx(lambda x: torch.cos(torch.sin(x)), tracing_mode="fake")(torch.ones(4))
    call_nodes = [node for node in gm.graph.nodes if node.op == "call_function"]
    call_nodes[0].meta["partitioner_tag"] = "is_forward"
    call_nodes[1].meta["partitioner_tag"] = "is_backward"

    assert classify_graph_phase(gm.graph) == "forward"


def test_collector_restores_global_state_after_exception():
    previous_topology = object()
    previous_span = aten_autobucketing_config.max_topo_span
    set_nccl_topo_config(previous_topology)
    collector = ReorderedMetricsCollector(
        _UnusedMesh(),
        nccl_topology=None,
        reordering_overrides={"max_topo_span": 23},
    )

    with pytest.raises(ValueError, match="stop"):
        with collector:
            assert get_nccl_topo_config() is None
            assert aten_autobucketing_config.max_topo_span == 23
            raise ValueError("stop")

    assert get_nccl_topo_config() is previous_topology
    assert aten_autobucketing_config.max_topo_span == previous_span
    assert torch._inductor.config.post_grad_custom_post_pass is None


def test_collector_records_fallback_when_detection_fails(monkeypatch):
    monkeypatch.setattr(collector_module, "detect_nccl_topo_config", lambda _mesh: None)
    collector = ReorderedMetricsCollector(_UnusedMesh())

    with pytest.warns(RuntimeWarning, match="detection returned None"):
        with collector:
            assert collector.cost_model == "default"
            assert collector.cost_model_status == "pytorch_default_fallback"


def test_collector_exposes_explicit_topology_as_cost_model():
    topology = object()
    collector = ReorderedMetricsCollector(_UnusedMesh(), nccl_topology=topology)

    with collector:
        assert collector.cost_model is topology
        assert get_nccl_topo_config() is topology
        assert collector.cost_model_status == "nccl_explicit"


def test_collector_rejects_unknown_reordering_option():
    collector = ReorderedMetricsCollector(
        _UnusedMesh(),
        nccl_topology=None,
        reordering_overrides={"not_an_option": True},
    )

    with pytest.raises(ValueError, match="not_an_option"):
        with collector:
            pass


def test_collector_rejects_conflicting_post_grad_pass():
    with torch._inductor.config.patch(
        {"post_grad_custom_post_pass": lambda graph: graph}
    ):
        with pytest.raises(RuntimeError, match="already configured"):
            with ReorderedMetricsCollector(_UnusedMesh(), nccl_topology=None):
                pass


def test_collector_callback_is_safe_for_nested_config_copy():
    collector = ReorderedMetricsCollector(_UnusedMesh(), nccl_topology=None)

    with collector:
        _use_zero_runtime_estimator(collector)
        torch._inductor.config.get_config_copy()
        callback = torch._inductor.config.post_grad_custom_post_pass
        assert callback is collector_module._run_active_post_grad_pass
        callback(_make_graph().graph)

    assert [record["phase"] for record in collector.records] == ["forward"]


def test_collector_rejects_inductor_overlap_scheduling():
    collector = ReorderedMetricsCollector(_UnusedMesh(), nccl_topology=None)
    forward = _make_graph()

    with collector:
        with torch._inductor.config.patch(
            {"aten_distributed_optimizations.enable_overlap_scheduling": True}
        ):
            with pytest.raises(RuntimeError, match="only overlap scheduler"):
                collector._post_grad_pass(forward.graph)

    assert collector.records == []


def test_collector_writes_forward_backward_metrics_and_traces(tmp_path):
    collector = ReorderedMetricsCollector(
        _UnusedMesh(), nccl_topology=None, trace_dir=tmp_path / "traces"
    )

    with collector:
        _use_zero_runtime_estimator(collector)
        collector._post_grad_pass(_make_graph().graph)
        collector._post_grad_pass(_make_graph(backward=True).graph)

    collector.require_forward_backward()
    output = tmp_path / "metrics.json"
    collector.write_json(output)
    payload = json.loads(output.read_text())

    assert payload["cost_model"]["status"] == "pytorch_default_explicit"
    assert [record["phase"] for record in payload["graphs"]] == [
        "forward",
        "backward",
    ]
    metric_fields = {
        "total_time",
        "compute_time",
        "communication_time",
        "exposed_comm_time",
        "peak_memory",
    }
    assert metric_fields <= payload["graphs"][0].keys()
    assert payload["graphs"][0]["placeholder_count"] == 1
    assert "placeholders" not in payload["graphs"][0]
    assert (tmp_path / "traces/forward_0.json").is_file()
    assert (tmp_path / "traces/backward_0.json").is_file()


def test_collector_can_include_placeholder_signatures():
    collector = ReorderedMetricsCollector(
        _UnusedMesh(), nccl_topology=None, include_placeholder_signatures=True
    )

    with collector:
        _use_zero_runtime_estimator(collector)
        collector._post_grad_pass(_make_graph().graph)

    record = collector.records[0]
    assert record["placeholder_count"] == 1
    assert len(record["placeholders"]) == 1
    assert set(record["placeholders"][0]) == {"name", "shape", "dtype"}


def test_collector_requires_both_graph_phases():
    collector = ReorderedMetricsCollector(_UnusedMesh(), nccl_topology=None)
    collector.records.append({"phase": "forward"})

    with pytest.raises(RuntimeError, match="backward"):
        collector.require_forward_backward()


def test_blind_eval_cases_and_rubrics_are_consistent():
    cases = json.loads((EVAL_ROOT / "cases.json").read_text())
    rubric = json.loads((EVAL_ROOT / "rubric.json").read_text())

    assert cases["schema_version"] == rubric["schema_version"] == 1
    assert cases["protocol"]["agent_visible_fields"] == ["prompt"]
    assert cases["protocol"]["repeat_count"] >= 3
    assert cases["protocol"]["baseline"] == {
        "enabled": True,
        "invocations": ["implicit"],
        "same_prompt": True,
    }

    case_ids = [case["id"] for case in cases["cases"]]
    rubric_ids = [item["id"] for item in rubric["rubrics"]]
    assert len(case_ids) == len(set(case_ids))
    assert len(rubric_ids) == len(set(rubric_ids))
    assert {case["rubric_id"] for case in cases["cases"]} == set(rubric_ids)
    rubrics_by_id = {item["id"]: item for item in rubric["rubrics"]}
    evidence_fields = set(cases["protocol"]["grader_visible_fields"])

    leaked_answer_terms = (
        "SPMD_MODELING_ASSUMPTION",
        "overlap_scheduling=False",
        "add_node_constraint",
        "remove_constraints",
        "(2, 8)",
    )
    for case in cases["cases"]:
        assert case["invocation"] in {"explicit", "implicit"}
        assert case["mode"] in {"explain", "plan", "evaluate", "negative"}
        assert case["prompt"].strip()
        assert not any(term in case["prompt"] for term in leaked_answer_terms)
        assert (
            case["skill_expected"] == rubrics_by_id[case["rubric_id"]]["skill_expected"]
        )
        for copy in case["workspace_copy"]:
            source = ROOT / copy["source"]
            assert source.is_file()
            assert not Path(copy["destination"]).is_absolute()
            assert ".." not in Path(copy["destination"]).parts

    for item in rubric["rubrics"]:
        criterion_ids = [criterion["id"] for criterion in item["criteria"]]
        assert criterion_ids
        assert len(criterion_ids) == len(set(criterion_ids))
        assert all(criterion["weight"] > 0 for criterion in item["criteria"])
        assert any(criterion["critical"] for criterion in item["criteria"])
        assert all(
            set(criterion["evidence"]) <= evidence_fields
            for criterion in item["criteria"]
        )


def test_claude_and_codex_use_the_same_skill():
    claude_skill = ROOT / ".claude/skills/autoparallel-optimizer"

    assert claude_skill.is_symlink()
    assert claude_skill.resolve() == SKILL_ROOT.resolve()
