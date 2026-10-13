# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import json
import subprocess
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


def _load_evaluation_module():
    path = SKILL_ROOT / "scripts/run_behavioral_evaluation.py"
    spec = importlib.util.spec_from_file_location(
        "autoparallel_skill_run_behavioral_evaluation", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluation_module = _load_evaluation_module()
BehavioralEvaluationRunner = evaluation_module.BehavioralEvaluationRunner
plan_evaluation_runs = evaluation_module.plan_runs
prepare_evaluation_workspace = evaluation_module.prepare_workspace
score_evaluation_grade = evaluation_module.score_grade
summarize_evaluation_results = evaluation_module.summarize_results
plan_pairwise_comparisons = evaluation_module.plan_pairwise_comparisons
collapse_reciprocal_pairwise_results = (
    evaluation_module.collapse_reciprocal_pairwise_results
)
summarize_pairwise_results = evaluation_module.summarize_pairwise_results
grader_tool_trace = evaluation_module._grader_tool_trace
grader_artifacts = evaluation_module._grader_artifacts
trace_mentions_skill = evaluation_module._trace_mentions_skill
skill_trace_fields = evaluation_module._skill_trace_fields
source_revision = evaluation_module._source_revision
tracked_source_dirty = evaluation_module._tracked_source_dirty
fixtures_sha256 = evaluation_module._fixtures_sha256
validate_resume_state = evaluation_module._validate_resume_state
load_completed_run = evaluation_module._load_completed_run
load_completed_pair = evaluation_module._load_completed_pair
acquire_campaign_lock = evaluation_module._acquire_campaign_lock
evaluation_failed = evaluation_module._evaluation_failed
grading_evidence_instructions = evaluation_module.GRADING_EVIDENCE_INSTRUCTIONS


def _load_comparison_module():
    path = SKILL_ROOT / "scripts/compare_behavioral_evaluations.py"
    spec = importlib.util.spec_from_file_location(
        "autoparallel_skill_compare_behavioral_evaluations", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


comparison_module = _load_comparison_module()
compare_evaluation_campaigns = comparison_module.compare_campaigns


def _load_harness_validator_module():
    path = SKILL_ROOT / "scripts/validate_evaluation_harness.py"
    spec = importlib.util.spec_from_file_location(
        "autoparallel_skill_validate_evaluation_harness", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


harness_validator_module = _load_harness_validator_module()
validate_harness_source = harness_validator_module.validate_harness_source
harness_handoff_requirements = harness_validator_module.HANDOFF_REQUIREMENTS


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
    bucket_all_all_reduce_groups = (
        aten_autobucketing_config.bucket_all_all_reduce_groups
    )
    post_grad_pass = torch._inductor.config.post_grad_custom_post_pass
    yield
    set_nccl_topo_config(topology)
    aten_autobucketing_config.max_topo_span = max_topo_span
    aten_autobucketing_config.bucket_all_all_reduce_groups = (
        bucket_all_all_reduce_groups
    )
    torch._inductor.config.post_grad_custom_post_pass = post_grad_pass


def test_classify_graph_phase_uses_tangents_and_partitioner_tags():
    forward = _make_graph()
    backward = _make_graph(backward=True)

    assert classify_graph_phase(forward.graph) == "forward"
    assert classify_graph_phase(backward.graph) == "backward"

    call_node = next(node for node in forward.graph.nodes if node.op == "call_function")
    call_node.meta["partitioner_tag"] = "is_backward"
    assert classify_graph_phase(forward.graph) == "backward"


def test_evaluation_harness_validator_checks_collector_contract():
    valid = """
def evaluate(model, input_fn, mesh, mp_policy, local_inputs, *, metrics_path):
    collector = ReorderedMetricsCollector(mesh)
    with collector:
        with AutoParallel(
            model, input_fn, mesh, mp_policy=mp_policy,
            cost_model=collector.cost_model,
        ) as autop:
            placement = autop.optimize_placement()
            parallel_model = autop.apply_placement(placement)
        compiled = torch.compile(
            parallel_model,
            backend=autoparallel_backend(overlap_scheduling=False),
        )
        compiled(*local_inputs).backward()
    collector.require_forward_backward()
    collector.write_json(path=metrics_path)
"""
    assert validate_harness_source(valid) == []

    invalid = (EVAL_ROOT / "fixtures/evaluation_harness.py").read_text()
    issues = set(validate_harness_source(invalid))
    assert {
        "COLLECTOR_CONTEXT_MISSING",
        "COST_MODEL_NOT_SHARED",
        "OVERLAP_SCHEDULING_NOT_DISABLED",
        "BACKWARD_NOT_EXECUTED",
        "FORWARD_BACKWARD_NOT_REQUIRED",
        "METRICS_PATH_NOT_EXPLICIT",
        "METRICS_NOT_WRITTEN",
    } <= issues
    assert harness_handoff_requirements == [
        "Reordered graph metrics are estimates distinct from the ILP objective and "
        "measured runtime.",
        "The pass adds no overlap control dependencies; later Inductor scheduling "
        "or fusion can change execution order.",
    ]


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
        reordering_overrides={
            "max_topo_span": 23,
            "bucket_all_all_reduce_groups": True,
        },
    )

    with pytest.raises(ValueError, match="stop"):
        with collector:
            assert get_nccl_topo_config() is None
            assert aten_autobucketing_config.max_topo_span == 23
            assert aten_autobucketing_config.bucket_all_all_reduce_groups is True
            raise ValueError("stop")

    assert get_nccl_topo_config() is previous_topology
    assert aten_autobucketing_config.max_topo_span == previous_span
    assert aten_autobucketing_config.bucket_all_all_reduce_groups is False
    assert torch._inductor.config.post_grad_custom_post_pass is None


def test_collector_records_fallback_when_detection_fails(monkeypatch):
    monkeypatch.setattr(collector_module, "detect_nccl_topo_config", lambda _mesh: None)
    collector = ReorderedMetricsCollector(_UnusedMesh())

    with pytest.warns(RuntimeWarning, match="detection returned None"):
        with collector:
            assert collector.cost_model == "default"
            assert collector.cost_model_status == "pytorch_default_fallback"
            assert collector.cost_model_source == "detection_failed"


def test_collector_exposes_explicit_topology_as_cost_model():
    topology = object()
    collector = ReorderedMetricsCollector(_UnusedMesh(), nccl_topology=topology)

    with collector:
        assert collector.cost_model is topology
        assert get_nccl_topo_config() is topology
        assert collector.cost_model_status == "nccl_explicit"
        assert collector.cost_model_source == "explicit"


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
    assert payload["cost_model"]["source"] == "explicit"
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
    assert payload["graphs"][0]["bucketing_diagnostics"] == {
        "collective_counts_after": {},
        "collective_counts_before": {},
        "group_selection": None,
        "greedy_bucket_calls": [],
    }
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


def test_collector_can_measure_without_reordering():
    collector = ReorderedMetricsCollector(
        _UnusedMesh(),
        nccl_topology=None,
        reordering_overrides={"enabled": False},
    )

    with collector:
        _use_zero_runtime_estimator(collector)
        collector._post_grad_pass(_make_graph().graph)

    assert collector.reordering_config.enabled is False
    assert "bucketing_diagnostics" not in collector.records[0]


def test_collector_requires_both_graph_phases():
    collector = ReorderedMetricsCollector(_UnusedMesh(), nccl_topology=None)
    collector.records.append({"phase": "forward"})

    with pytest.raises(RuntimeError, match="backward"):
        collector.require_forward_backward()


def test_blind_eval_cases_and_rubrics_are_consistent():
    cases = json.loads((EVAL_ROOT / "cases.json").read_text())
    rubric = json.loads((EVAL_ROOT / "rubric.json").read_text())

    assert cases["schema_version"] == rubric["schema_version"] == 1
    assert cases["protocol"]["agent_visible_fields"] == [
        "agent_preamble",
        "prompt",
    ]
    assert "Read-only inspection" in cases["protocol"]["agent_preamble"]
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
        if case["invocation"] == "explicit":
            assert "Do not execute commands" not in case["prompt"]
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


def test_behavioral_rubric_keeps_coverage_details_out_of_hard_gates():
    document = json.loads((EVAL_ROOT / "rubric.json").read_text())
    criteria = {
        (rubric["id"], criterion["id"]): criterion
        for rubric in document["rubrics"]
        for criterion in rubric["criteria"]
    }

    coverage_details = {
        ("full_shard_overlap_uncertainty", "uses_independent_optimizer_state"),
        ("gradient_accumulation_cycle_boundary", "uses_compatible_ddp_template"),
        ("pareto_dominance_overrides_latency_tolerance", "uses_robust_envelopes"),
        ("fsdp_template_compatibility", "includes_safe_template_costs"),
        (
            "torchbench_tied_weight_placement",
            "routes_to_qualified_fallback_until_retest",
        ),
    }
    assert all(not criteria[key]["critical"] for key in coverage_details)

    hard_gates = {
        ("gradient_accumulation_cycle_boundary", "selects_ddp"),
        ("torchbench_tied_weight_placement", "treats_alias_break_as_semantic_failure"),
        ("torchbench_qat_buffer_alias", "classifies_application_failure"),
    }
    assert all(criteria[key]["critical"] for key in hard_gates)


def test_realizability_guidance_uses_balanced_uneven_shard_extents():
    guidance = (SKILL_ROOT / "references/realizability-failures.md").read_text()

    assert "q, r = divmod(logical_size, shard_count)" in guidance
    assert "five ranks of 2,501 and three ranks of 2,500" in guidance


def test_realizability_guidance_requires_targeted_alias_regression():
    guidance = (SKILL_ROOT / "references/realizability-failures.md").read_text()

    assert "minimized regression" in guidance
    assert "genuinely missing" in guidance and "still\nfails" in guidance
    assert "two successive updates" in guidance and "different aliases" in guidance
    assert "unpartitioned native reference" in guidance


def test_realizability_guidance_constrains_functionalized_mutations():
    guidance = (SKILL_ROOT / "references/realizability-failures.md").read_text()

    assert "`InputMutationAOTOutput` descriptor" in guidance
    assert "`mutated_input` descriptor" in guidance
    assert "default placement-equality constraint" in guidance
    assert "out-of-graph epilogue" in guidance
    assert "mutated input and normal\noutputs" in guidance


def test_mesh_guidance_keeps_candidate_evidence_independent():
    guidance = (SKILL_ROOT / "references/mesh-selection.md").read_text()

    assert "each mesh as a separate planning experiment" in guidance
    assert "Do not pool placements" in guidance and "timings across meshes" in guidance
    assert "current ILP solve cost is impractical" in guidance


def test_full_shard_guidance_requires_independent_endpoint_artifacts():
    guidance = (SKILL_ROOT / "references/full-shard-planning.md").read_text()

    assert "serial placements and optimizer" in guidance
    assert "optimistic endpoint" in guidance and "different paths" in guidance
    assert "one mutated optimizer object" in guidance


def test_optimizer_envelope_guidance_requires_regime_match():
    guidance = (SKILL_ROOT / "SKILL.md").read_text()

    assert "Before applying a frozen envelope" in guidance
    assert "**platform**" in guidance and "realized\n  kernel family" in guidance
    assert "**optimizer contract**" in guidance
    assert "**scale**" in guidance and "tensor-count ranges" in guidance
    assert "must not use\n  `executed_on_target` or `measured`" in guidance

    workflow = (SKILL_ROOT / "references/workflow.md").read_text()
    assert "calibration platform:" in workflow
    assert "optimizer contract:" in workflow
    assert "calibration scale:" in workflow


def test_decision_guidance_requires_complete_contract_bookkeeping():
    guidance = (SKILL_ROOT / "SKILL.md").read_text()

    assert (
        "process launch,\n  compilation, allocator, or system-state variance"
        in guidance
    )
    assert "collective group sizes and scale" in guidance
    assert "explicit NCCL topology for the allocated communicator" in guidance
    assert (
        "qualitative ordering and say that exact\n  values were not supplied"
        in guidance
    )
    assert (
        "both the partial-boundary\n  ordering and the full-step ordering" in guidance
    )
    assert (
        "partial-boundary ordering,\n  and full-step latency plus memory ordering"
        in guidance
    )
    assert "same clipped ID for its exact retries and linked baselines" in guidance
    assert "same clipping\n  frequency and optimizer-update frequency" in guidance
    assert "global_batch % effective_dp_degree == 0" in guidance
    assert "post-lowering or\n  post-wrapping per-FQN invariant check" in guidance
    assert "optimization\n  hypothesis and provisional starting point" in guidance
    assert "optional ILP compression" in guidance
    assert "clustering role: optional ILP compression" in guidance
    assert (
        "local batch, collective scale, and potential\n  selected placement" in guidance
    )
    assert "valid_meta_uses: legality/feasibility checking" in guidance
    assert "explicit `partial_boundary:`\n  and `full_boundary:` fields" in guidance
    assert "`verdict:\n  inconclusive`" in guidance
    assert "Do not infer execution provenance from supplied timing numbers" in guidance
    assert "Only use\n  `executed_on_target`" in guidance
    assert "`STARTUP_NOT_AMORTIZED`, not `PERFORMANCE_BAR_MISSED`" in guidance
    assert "**world size\n  does not define global batch**" in guidance
    assert "achieved byte-weighted parameter-storage fraction" in guidance
    assert "output-shape parity, output-value parity, and gradient parity" in guidance
    assert "read-only hypothetical or arithmetic comparison" in guidance
    assert "should answer directly and omit the card" in guidance
    assert "`local_norm_work` and `placement_dependent_global_reduction`" in guidance
    assert "supplied latency and forward/backward peak are\n  **measured**" in guidance
    assert (
        "after lowering or\n  wrapping and before execution evidence is accepted"
        in guidance
    )

    full_shard = (SKILL_ROOT / "references/full-shard-planning.md").read_text()
    assert "Do not leave endpoint bookkeeping implicit" in full_shard
    assert "topology, activation checkpointing, and\nautobucketing fixed" in full_shard


def test_optimizer_objective_rubric_does_not_require_unsupplied_values():
    document = json.loads((EVAL_ROOT / "rubric.json").read_text())
    criterion = next(
        criterion
        for rubric in document["rubrics"]
        if rubric["id"] == "optimizer_objective_mismatch"
        for criterion in rubric["criteria"]
        if criterion["id"] == "records_stable_finding"
    )

    assert "must not be invented" in criterion["description"]
    assert (
        "qualitative win/loss and memory ordering is full credit"
        in criterion["description"]
    )


def test_counterfactual_guidance_recovers_baseline_before_ablation():
    guidance = (SKILL_ROOT / "references/workflow.md").read_text()

    assert "first remove exactly the names returned" in guidance
    assert "original output and memory constraints unchanged" in guidance
    assert "Establish that recovery before re-adding" in guidance


def test_global_spmd_guidance_qualifies_custom_collective_fallback():
    guidance = (SKILL_ROOT / "references/global-spmd-modeling.md").read_text()

    assert "internal\ncommunication cost unpriced" in guidance
    assert "manual or native distributed path as the qualified fallback" in guidance
    assert "classify AutoParallel as ineligible\nor inconclusive" in guidance


def test_global_spmd_guidance_validates_no_input_random_factories():
    guidance = (SKILL_ROOT / "references/global-spmd-modeling.md").read_text()

    assert "mesh from planner context" in guidance
    assert "replicated-only strategy" in guidance
    assert "local-shape rewriting" in guidance
    assert "Equal startup seeds\ndo not prove" in guidance
    assert "advance RNG state differently by\nrank" in guidance
    assert (
        "single-rank\nshape/range test is only planning or smoke evidence" in guidance
    )


def test_skill_guidance_bounds_operator_strategy_coverage():
    guidance = (SKILL_ROOT / "SKILL.md").read_text()

    assert "schema and in-contract semantics are equivalent" in guidance
    assert "allowlisted replicate-only strategy" in guidance
    assert "Reject invalid\n  input options candidate-locally" in guidance
    assert "On both 1D and 2D meshes, lower every emitted strategy" in guidance
    assert "output-structure, shape, value, and gradient checks" in guidance


def test_behavioral_evaluation_plan_is_isolated_and_repeated():
    cases = json.loads((EVAL_ROOT / "cases.json").read_text())

    runs = plan_evaluation_runs(cases)

    assert len(runs) == 180
    assert sum(run.variant == "skill" for run in runs) == 93
    assert sum(run.variant == "baseline" for run in runs) == 87
    assert all(1 <= run.repeat <= 3 for run in runs)
    assert len({run.id for run in runs}) == len(runs)


def test_behavioral_evaluation_agent_preamble_allows_read_only_guidance(tmp_path):
    cases = json.loads((EVAL_ROOT / "cases.json").read_text())
    rubric = json.loads((EVAL_ROOT / "rubric.json").read_text())
    runner = BehavioralEvaluationRunner(
        repo_root=ROOT,
        skill_root=SKILL_ROOT,
        output_dir=tmp_path,
        command=["codex"],
        model=None,
        reasoning_effort="medium",
        timeout_seconds=1,
        grade=True,
        rubric_document=rubric,
        agent_preamble=cases["protocol"]["agent_preamble"],
    )

    prompt = runner._agent_prompt(cases["cases"][0])

    assert prompt.startswith("Read-only inspection")
    assert prompt.endswith(cases["cases"][0]["prompt"])


def test_behavioral_graders_do_not_require_repeating_supplied_facts():
    assert (
        "facts supplied by the task as available evidence"
        in grading_evidence_instructions
    )
    assert "need not repeat or paraphrase" in grading_evidence_instructions
    assert "Apply this standard consistently" in grading_evidence_instructions

    source = (SKILL_ROOT / "scripts/run_behavioral_evaluation.py").read_text()
    assert source.count("GRADING_EVIDENCE_INSTRUCTIONS") == 3


def test_behavioral_evaluation_workspace_hides_answers(tmp_path):
    cases = json.loads((EVAL_ROOT / "cases.json").read_text())
    case = cases["cases"][0]
    workspace = tmp_path / "skill-workspace"

    prepare_evaluation_workspace(
        workspace,
        repo_root=ROOT,
        skill_root=SKILL_ROOT,
        case=case,
        include_skill=True,
    )

    copied_skill = workspace / ".agents/skills/autoparallel-optimizer"
    assert (copied_skill / "SKILL.md").is_file()
    assert not (copied_skill / "references/agent-routing-evals.json").exists()
    assert (workspace / "global_spmd_local_shuffle.py").is_file()

    baseline_workspace = tmp_path / "baseline-workspace"
    prepare_evaluation_workspace(
        baseline_workspace,
        repo_root=ROOT,
        skill_root=SKILL_ROOT,
        case=case,
        include_skill=False,
    )
    assert not (baseline_workspace / ".agents").exists()
    assert (baseline_workspace / "global_spmd_local_shuffle.py").is_file()


def test_behavioral_evaluation_scores_critical_criteria():
    rubric_document = json.loads((EVAL_ROOT / "rubric.json").read_text())
    rubric = rubric_document["rubrics"][0]
    perfect_grade = {
        "criteria": [
            {"id": item["id"], "score": 1, "evidence": "present"}
            for item in rubric["criteria"]
        ]
    }

    perfect = score_evaluation_grade(
        rubric,
        rubric_document["scoring"],
        perfect_grade,
    )
    assert perfect["passed"]
    assert perfect["weighted_score"] == 1.0

    perfect_grade["criteria"][0]["score"] = 0.5
    failed = score_evaluation_grade(
        rubric,
        rubric_document["scoring"],
        perfect_grade,
    )
    assert not failed["passed"]
    assert not failed["critical_pass"]

    summary = summarize_evaluation_results(
        [
            {
                "run_id": "case-skill-01",
                "case_id": "case",
                "variant": "skill",
                "skill_expected": True,
                "skill_usage_observed": True,
                "skill_file_read_observed": True,
                "skill_access_failure_observed": False,
                "score": perfect,
            },
            {
                "run_id": "case-baseline-01",
                "case_id": "case",
                "variant": "baseline",
                "skill_expected": True,
                "skill_usage_observed": None,
                "skill_file_read_observed": None,
                "skill_access_failure_observed": None,
                "score": failed,
            },
        ]
    )
    assert summary["comparison"]["mean_score_delta"] > 0
    assert summary["comparison"]["pass_rate_delta"] == 1.0
    assert summary["comparison"]["paired_run_count"] == 1
    assert summary["comparison"]["mean_score_delta_case_bootstrap_95"] == {
        "lower": summary["comparison"]["mean_score_delta"],
        "upper": summary["comparison"]["mean_score_delta"],
        "samples": 20_000,
        "seed": 0,
        "unit": "case",
    }
    assert summary["variants"]["skill"]["weighted_coverage_mean"] == 1.0
    assert summary["variants"]["skill"]["critical_criteria_pass_rate"] == 1.0
    assert summary["variants"]["skill"]["skill_expected_run_count"] == 1
    assert summary["variants"]["skill"]["skill_usage_observed_count"] == 1
    assert summary["variants"]["skill"]["skill_usage_observed_rate"] == 1.0
    assert summary["variants"]["skill"]["skill_file_read_observed_count"] == 1
    assert summary["variants"]["skill"]["skill_file_read_observed_rate"] == 1.0
    assert summary["variants"]["skill"]["skill_access_failure_observed_count"] == 0
    assert summary["variants"]["skill"]["skill_access_failure_observed_rate"] == 0.0
    assert summary["failed_run_count"] == 0


def test_behavioral_evaluation_pairwise_comparison_is_blind_and_paired():
    results = [
        {
            "run_id": "case-skill-01",
            "case_id": "case",
            "variant": "skill",
            "repeat": 1,
            "agent_exit_code": 0,
        },
        {
            "run_id": "case-baseline-01",
            "case_id": "case",
            "variant": "baseline",
            "repeat": 1,
            "agent_exit_code": 0,
        },
        {
            "run_id": "unpaired-skill-01",
            "case_id": "unpaired",
            "variant": "skill",
            "repeat": 1,
            "agent_exit_code": 0,
        },
    ]

    pairs = plan_pairwise_comparisons(results)
    assert len(pairs) == 1
    assert pairs[0][0]["variant"] == "skill"
    assert pairs[0][1]["variant"] == "baseline"

    judgments = [
        {
            "judgment_id": "case-01-skill-first",
            "case_id": "case",
            "repeat": 1,
            "a_variant": "skill",
            "winner": "a",
            "winner_variant": "skill",
        },
        {
            "judgment_id": "case-01-baseline-first",
            "case_id": "case",
            "repeat": 1,
            "a_variant": "baseline",
            "winner": "b",
            "winner_variant": "skill",
        },
        {
            "judgment_id": "other-01-skill-first",
            "case_id": "other",
            "repeat": 1,
            "a_variant": "skill",
            "winner": "b",
            "winner_variant": "baseline",
        },
        {
            "judgment_id": "other-01-baseline-first",
            "case_id": "other",
            "repeat": 1,
            "a_variant": "baseline",
            "winner": "a",
            "winner_variant": "baseline",
        },
        {
            "judgment_id": "biased-01-skill-first",
            "case_id": "biased",
            "repeat": 1,
            "a_variant": "skill",
            "winner": "a",
            "winner_variant": "skill",
        },
        {
            "judgment_id": "biased-01-baseline-first",
            "case_id": "biased",
            "repeat": 1,
            "a_variant": "baseline",
            "winner": "a",
            "winner_variant": "baseline",
        },
    ]
    consensus = collapse_reciprocal_pairwise_results(judgments)
    summary = summarize_pairwise_results(consensus, position_results=judgments)
    assert summary["skill_wins"] == 1
    assert summary["baseline_wins"] == 1
    assert summary["ties"] == 1
    assert summary["skill_decisive_win_rate"] == 0.5
    interval = summary["skill_decisive_win_rate_wilson_95"]
    assert interval["lower"] < 0.5 < interval["upper"]
    assert summary["pair_count"] == 3
    assert summary["judgment_count"] == 6
    assert summary["reciprocal_order_agreement_count"] == 2
    assert summary["reciprocal_order_disagreement_count"] == 1
    assert summary["position_wins"] == {"a": 4, "b": 2, "tie": 0}
    assert summary["case_majority"]["case_count"] == 3
    assert summary["case_majority"]["skill_wins"] == 1
    assert summary["case_majority"]["baseline_wins"] == 1
    assert summary["case_majority"]["skill_decisive_win_rate"] == 0.5


def test_behavioral_evaluation_grader_receives_trace_and_artifacts(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "\n".join(
            (
                json.dumps({"type": "thread.started", "thread_id": "hidden"}),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "command_execution",
                            "command": "python -m pytest",
                            "exit_code": 0,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "agent_message",
                            "text": "Using autoparallel-optimizer now.",
                        },
                    }
                ),
            )
        )
    )
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "result.py").write_text("value = 1\n")

    assert grader_tool_trace(trace) == [
        {
            "type": "command_execution",
            "command": "python -m pytest",
            "exit_code": 0,
        }
    ]
    artifacts = grader_artifacts(artifact_root)
    assert artifacts[0]["path"] == "result.py"
    assert artifacts[0]["text"] == "value = 1\n"
    assert trace_mentions_skill(trace)


def test_behavioral_evaluation_distinguishes_skill_mention_from_file_read(tmp_path):
    trace = tmp_path / "skill-trace.jsonl"
    trace.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "item": {
                            "type": "agent_message",
                            "text": "Using the AutoParallel optimizer skill.",
                        }
                    }
                ),
                json.dumps(
                    {
                        "item": {
                            "type": "command_execution",
                            "command": "cat .agents/skills/autoparallel-optimizer/SKILL.md",
                            "aggregated_output": (
                                "---\nname: autoparallel-optimizer\n---\n"
                                "# AutoParallel Optimizer\n"
                            ),
                            "exit_code": 0,
                        }
                    }
                ),
            )
        )
    )

    assert skill_trace_fields(trace, response_path=None, include=True) == {
        "skill_usage_observed": True,
        "skill_file_read_observed": True,
        "skill_access_failure_observed": False,
    }
    assert skill_trace_fields(trace, response_path=None, include=False) == {
        "skill_usage_observed": None,
        "skill_file_read_observed": None,
        "skill_access_failure_observed": None,
    }

    response = tmp_path / "response.md"
    response.write_text(
        "The skill file could not be read because the sandbox blocked it."
    )
    assert skill_trace_fields(trace, response_path=response, include=True)[
        "skill_access_failure_observed"
    ]

    response.write_text("The skill guidance was read successfully.")
    trace.write_text(
        trace.read_text()
        + "\n"
        + json.dumps(
            {
                "item": {
                    "type": "command_execution",
                    "command": "grep blocked evaluator.py",
                    "aggregated_output": '"inspection command was blocked"',
                    "exit_code": 0,
                }
            }
        )
    )
    assert not skill_trace_fields(trace, response_path=response, include=True)[
        "skill_access_failure_observed"
    ]


def test_behavioral_evaluation_records_source_state():
    assert (
        source_revision(ROOT)
        == subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    assert isinstance(tracked_source_dirty(ROOT), bool)


def test_behavioral_evaluation_fails_for_incomplete_pairwise_grading():
    assert not evaluation_failed({"failed_run_count": 0})
    assert evaluation_failed({"failed_run_count": 1})
    assert evaluation_failed(
        {
            "failed_run_count": 0,
            "pairwise_comparison": {
                "failed_pair_count": 1,
                "failed_pair_judgment_count": 0,
            },
        }
    )
    assert evaluation_failed(
        {
            "failed_run_count": 0,
            "pairwise_comparison": {
                "failed_pair_count": 0,
                "failed_pair_judgment_count": 1,
            },
        }
    )


def test_behavioral_evaluation_hashes_fixture_contents(tmp_path):
    fixture = tmp_path / "fixture.py"
    fixture.write_text("value = 1\n")
    suite = {
        "cases": [
            {"workspace_copy": [{"source": "fixture.py", "destination": "model.py"}]}
        ]
    }

    original = fixtures_sha256(tmp_path, suite)
    fixture.write_text("value = 2\n")

    assert fixtures_sha256(tmp_path, suite) != original


def test_behavioral_evaluation_resume_validates_and_reuses_completed_work(tmp_path):
    config = {"schema_version": 1, "cases_sha256": "cases"}
    protocol = {"repeat_count": 1}
    (tmp_path / "evaluation-config.json").write_text(json.dumps(config))
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    validate_resume_state(tmp_path, config=config, protocol=protocol)

    with pytest.raises(ValueError, match="cases_sha256"):
        validate_resume_state(
            tmp_path,
            config=config | {"cases_sha256": "different"},
            protocol=protocol,
        )

    spec = evaluation_module.EvaluationRun("case", "skill", 1)
    run_dir = tmp_path / "runs" / spec.id
    run_dir.mkdir(parents=True)
    (run_dir / "response.md").write_text("answer")
    (run_dir / "tool-trace.jsonl").write_text("")
    (run_dir / "workspace-diff.json").write_text("{}")
    candidate_sha256 = evaluation_module._json_sha256(
        {
            "response": "answer",
            "workspace_diff": {},
            "tool_trace": [],
            "changed_artifacts": [],
        }
    )
    result = {
        "run_id": spec.id,
        "case_id": spec.case_id,
        "variant": spec.variant,
        "repeat": spec.repeat,
        "agent_exit_code": 0,
        "pairwise_candidate_sha256": candidate_sha256,
        "score": {"weighted_score": 1.0},
    }
    (run_dir / "result.json").write_text(json.dumps(result))
    assert load_completed_run(tmp_path, spec, require_score=True) == result

    pair_dir = tmp_path / "pairs" / "case-01-baseline-first"
    pair_dir.mkdir(parents=True)
    pair = {
        "pair_id": "case-01",
        "judgment_id": "case-01-baseline-first",
        "case_id": "case",
        "repeat": 1,
        "a_variant": "baseline",
        "b_variant": "skill",
        "winner": "b",
        "winner_variant": "skill",
        "a_candidate_sha256": "baseline-evidence",
        "b_candidate_sha256": candidate_sha256,
    }
    baseline_result = result | {
        "variant": "baseline",
        "pairwise_candidate_sha256": "baseline-evidence",
    }
    (pair_dir / "result.json").write_text(json.dumps(pair))
    assert (
        load_completed_pair(
            tmp_path,
            result,
            baseline_result,
            expected_a_variant="baseline",
        )
        == pair
    )
    assert (
        load_completed_pair(
            tmp_path,
            result,
            baseline_result,
            expected_a_variant="skill",
        )
        is None
    )

    changed_result = result | {"pairwise_candidate_sha256": "changed"}
    assert (
        load_completed_pair(
            tmp_path,
            changed_result,
            baseline_result,
            expected_a_variant="baseline",
        )
        is None
    )

    first_lock = acquire_campaign_lock(tmp_path)
    with pytest.raises(ValueError, match="already active"):
        acquire_campaign_lock(tmp_path)
    first_lock.close()
    acquire_campaign_lock(tmp_path).close()


def test_behavioral_evaluation_comparison_flags_incompatible_rubrics(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    common_config = {
        "model": "test-model",
        "reasoning_effort": "medium",
        "repeats": 1,
        "baseline_enabled": True,
        "grading_enabled": True,
        "pairwise_grading_enabled": True,
        "pairwise_ordering": "reciprocal_ab_v2",
        "codex_command": "codex",
        "timeout_seconds": 1800,
        "case_ids": None,
        "cases_sha256": "same-cases",
        "fixtures_sha256": "same-fixtures",
        "evaluator_sha256": "same-evaluator",
        "skill_sha256": "same-skill",
    }
    (left / "evaluation-config.json").write_text(
        json.dumps(common_config | {"rubric_sha256": "old"})
    )
    (right / "evaluation-config.json").write_text(
        json.dumps(common_config | {"rubric_sha256": "new"})
    )
    left_results = [
        {
            "case_id": "case",
            "variant": variant,
            "repeat": 1,
            "score": {"weighted_score": score},
        }
        for variant, score in (("skill", 0.8), ("baseline", 0.7))
    ]
    right_results = [
        {
            "case_id": "case",
            "variant": variant,
            "repeat": 1,
            "score": {"weighted_score": score},
        }
        for variant, score in (("skill", 0.9), ("baseline", 0.75))
    ]
    (left / "results.json").write_text(json.dumps(left_results))
    (right / "results.json").write_text(json.dumps(right_results))
    (left / "summary.json").write_text(json.dumps({"pairwise_comparison": None}))
    (right / "summary.json").write_text(json.dumps({"pairwise_comparison": None}))

    comparison = compare_evaluation_campaigns(left, right)

    assert not comparison["scores_are_directly_comparable"]
    assert not comparison["compatibility"]["rubric_sha256"]
    assert comparison["variant_changes"]["skill"][
        "mean_case_score_change"
    ] == pytest.approx(0.1)
    assert comparison["paired_effect"]["left_mean_delta"] == pytest.approx(0.1)
    assert comparison["paired_effect"]["right_mean_delta"] == pytest.approx(0.15)
    assert comparison["paired_effect"]["mean_delta_of_deltas"] == pytest.approx(0.05)

    (right / "evaluation-config.json").write_text(
        json.dumps(
            common_config | {"rubric_sha256": "old", "skill_sha256": "changed-skill"}
        )
    )
    treatment_comparison = compare_evaluation_campaigns(left, right)
    assert treatment_comparison["scores_are_directly_comparable"]
    assert not treatment_comparison["compatibility"]["skill_sha256"]

    (right / "evaluation-config.json").write_text(
        json.dumps(
            common_config
            | {
                "rubric_sha256": "old",
                "skill_sha256": "changed-skill",
                "evaluator_sha256": "changed-evaluator",
            }
        )
    )
    evaluator_comparison = compare_evaluation_campaigns(left, right)
    assert not evaluator_comparison["scores_are_directly_comparable"]
    assert not evaluator_comparison["compatibility"]["evaluator_sha256"]

    for root in (left, right):
        config = json.loads((root / "evaluation-config.json").read_text())
        config.pop("fixtures_sha256")
        (root / "evaluation-config.json").write_text(json.dumps(config))
    missing_provenance = compare_evaluation_campaigns(left, right)
    assert not missing_provenance["scores_are_directly_comparable"]
    assert not missing_provenance["compatibility"]["fixtures_sha256"]


def test_claude_and_codex_use_the_same_skill():
    claude_skill = ROOT / ".claude/skills/autoparallel-optimizer"

    assert claude_skill.is_symlink()
    assert claude_skill.resolve() == SKILL_ROOT.resolve()
