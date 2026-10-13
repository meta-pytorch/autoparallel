from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import random
import shlex
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence, TextIO

PAIRWISE_ORDERING = "reciprocal_ab_v2"
GRADING_EVIDENCE_INSTRUCTIONS = (
    "Treat facts supplied by the task as available evidence. A response need not "
    "repeat or paraphrase a supplied fact to use it correctly unless the criterion "
    "explicitly requires stating, reporting, recording, or retaining that fact. "
    "Apply this standard consistently to every candidate."
)


@dataclass(frozen=True)
class EvaluationRun:
    case_id: str
    variant: str
    repeat: int

    @property
    def id(self) -> str:
        return f"{self.case_id}-{self.variant}-{self.repeat:02d}"


def plan_runs(
    suite: Mapping[str, Any],
    *,
    repeats: int | None = None,
    include_baseline: bool = True,
    case_ids: set[str] | None = None,
) -> list[EvaluationRun]:
    protocol = suite["protocol"]
    repeat_count = repeats or protocol["repeat_count"]
    baseline_invocations = set(protocol["baseline"]["invocations"])
    runs = []
    for case in suite["cases"]:
        if case_ids is not None and case["id"] not in case_ids:
            continue
        variants = ["skill"]
        if (
            include_baseline
            and protocol["baseline"]["enabled"]
            and case["invocation"] in baseline_invocations
        ):
            variants.append("baseline")
        for variant in variants:
            for repeat in range(1, repeat_count + 1):
                runs.append(EvaluationRun(case["id"], variant, repeat))
    return runs


def prepare_workspace(
    workspace: Path,
    *,
    repo_root: Path,
    skill_root: Path,
    case: Mapping[str, Any],
    include_skill: bool,
) -> None:
    workspace.mkdir(parents=True)
    if include_skill:
        destination = workspace / ".agents/skills/autoparallel-optimizer"
        shutil.copytree(skill_root, destination, ignore=_skill_copy_ignores)
    for item in case["workspace_copy"]:
        source = (repo_root / item["source"]).resolve()
        destination = _safe_workspace_path(workspace, item["destination"])
        if not source.is_file() or not source.is_relative_to(repo_root.resolve()):
            raise ValueError(f"Invalid evaluation fixture: {item['source']}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def score_grade(
    rubric: Mapping[str, Any],
    scoring: Mapping[str, Any],
    grade: Mapping[str, Any],
) -> dict[str, Any]:
    values = {item["id"]: item for item in grade.get("criteria", [])}
    expected_ids = {item["id"] for item in rubric["criteria"]}
    if set(values) != expected_ids:
        raise ValueError("Grader criterion IDs do not match the rubric")
    numerator = 0.0
    denominator = 0.0
    critical_pass = True
    criteria = []
    for criterion in rubric["criteria"]:
        result = values[criterion["id"]]
        score = result.get("score")
        if score not in {0, 0.5, 1}:
            raise ValueError(f"Invalid criterion score: {score!r}")
        weight = criterion["weight"]
        numerator += score * weight
        denominator += weight
        if criterion["critical"] and score != 1:
            critical_pass = False
        criteria.append(
            {
                "id": criterion["id"],
                "score": score,
                "weight": weight,
                "critical": criterion["critical"],
                "evidence": result.get("evidence", ""),
            }
        )
    weighted_score = numerator / denominator
    return {
        "weighted_score": weighted_score,
        "critical_pass": critical_pass,
        "passed": weighted_score >= scoring["pass_threshold"] and critical_pass,
        "criteria": criteria,
    }


def summarize_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for result in results:
        if "score" not in result:
            continue
        grouped.setdefault((result["case_id"], result["variant"]), []).append(result)
    cases: dict[str, dict[str, Any]] = {}
    variants: dict[str, list[Mapping[str, Any]]] = {}
    for (case_id, variant), items in sorted(grouped.items()):
        variants.setdefault(variant, []).extend(items)
        cases.setdefault(case_id, {})[variant] = {
            "run_count": len(items),
            "mean_score": mean(item["score"]["weighted_score"] for item in items),
            "pass_rate": mean(float(item["score"]["passed"]) for item in items),
            "weighted_coverage_mean": mean(
                item["score"]["weighted_score"] for item in items
            ),
            "critical_criteria_pass_rate": mean(
                float(item["score"]["critical_pass"]) for item in items
            ),
            "protocol_pass_rate": mean(
                float(item["score"]["passed"]) for item in items
            ),
            **_skill_usage_summary(items),
        }
    variant_summary = {
        variant: {
            "run_count": len(items),
            "mean_score": mean(item["score"]["weighted_score"] for item in items),
            "pass_rate": mean(float(item["score"]["passed"]) for item in items),
            "weighted_coverage_mean": mean(
                item["score"]["weighted_score"] for item in items
            ),
            "critical_criteria_pass_rate": mean(
                float(item["score"]["critical_pass"]) for item in items
            ),
            "protocol_pass_rate": mean(
                float(item["score"]["passed"]) for item in items
            ),
            **_skill_usage_summary(items),
        }
        for variant, items in sorted(variants.items())
    }
    raw_comparison = None
    if "skill" in variant_summary and "baseline" in variant_summary:
        raw_comparison = {
            "mean_score_delta": (
                variant_summary["skill"]["mean_score"]
                - variant_summary["baseline"]["mean_score"]
            ),
            "pass_rate_delta": (
                variant_summary["skill"]["pass_rate"]
                - variant_summary["baseline"]["pass_rate"]
            ),
        }
    paired = _paired_scored_results(results)
    comparison = _summarize_paired_results(paired) if paired else None
    return {
        "schema_version": 1,
        "run_count": len(results),
        "scored_run_count": sum("score" in result for result in results),
        "failed_run_count": sum(
            result.get("agent_exit_code") not in {None, 0} or "runner_error" in result
            for result in results
        ),
        "variants": variant_summary,
        "comparison": comparison,
        "raw_unpaired_comparison": raw_comparison,
        "cases": cases,
    }


def _paired_scored_results(
    results: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    grouped: dict[tuple[str, Any], dict[str, Mapping[str, Any]]] = {}
    for result in results:
        if "score" not in result or result.get("variant") not in {"skill", "baseline"}:
            continue
        key = (result["case_id"], result.get("repeat"))
        grouped.setdefault(key, {})[result["variant"]] = result
    return [
        (items["skill"], items["baseline"])
        for _, items in sorted(grouped.items())
        if set(items) == {"skill", "baseline"}
    ]


def _skill_usage_summary(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = [
        item
        for item in items
        if item.get("variant") == "skill" and item.get("skill_expected") is True
    ]
    if not expected:
        return {}
    usage_count = sum(item.get("skill_usage_observed") is True for item in expected)
    file_read_count = sum(
        item.get("skill_file_read_observed") is True for item in expected
    )
    access_failure_count = sum(
        item.get("skill_access_failure_observed") is True for item in expected
    )
    return {
        "skill_expected_run_count": len(expected),
        "skill_usage_observed_count": usage_count,
        "skill_usage_observed_rate": usage_count / len(expected),
        "skill_file_read_observed_count": file_read_count,
        "skill_file_read_observed_rate": file_read_count / len(expected),
        "skill_access_failure_observed_count": access_failure_count,
        "skill_access_failure_observed_rate": access_failure_count / len(expected),
    }


def _summarize_paired_results(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    skill_scores = [item[0]["score"]["weighted_score"] for item in pairs]
    baseline_scores = [item[1]["score"]["weighted_score"] for item in pairs]
    by_case: dict[str, list[float]] = {}
    for skill, baseline in pairs:
        by_case.setdefault(skill["case_id"], []).append(
            skill["score"]["weighted_score"] - baseline["score"]["weighted_score"]
        )
    case_deltas = {case_id: mean(values) for case_id, values in by_case.items()}
    epsilon = 1e-12
    return {
        "paired_run_count": len(pairs),
        "paired_case_count": len(by_case),
        "skill_weighted_coverage_mean": mean(skill_scores),
        "baseline_weighted_coverage_mean": mean(baseline_scores),
        "mean_score_delta": mean(skill_scores) - mean(baseline_scores),
        "mean_score_delta_case_bootstrap_95": _bootstrap_mean_interval(
            list(case_deltas.values())
        ),
        "skill_critical_criteria_pass_rate": mean(
            float(item[0]["score"]["critical_pass"]) for item in pairs
        ),
        "baseline_critical_criteria_pass_rate": mean(
            float(item[1]["score"]["critical_pass"]) for item in pairs
        ),
        "skill_protocol_pass_rate": mean(
            float(item[0]["score"]["passed"]) for item in pairs
        ),
        "baseline_protocol_pass_rate": mean(
            float(item[1]["score"]["passed"]) for item in pairs
        ),
        "pass_rate_delta": mean(float(item[0]["score"]["passed"]) for item in pairs)
        - mean(float(item[1]["score"]["passed"]) for item in pairs),
        "case_wins": sum(delta > epsilon for delta in case_deltas.values()),
        "case_ties": sum(abs(delta) <= epsilon for delta in case_deltas.values()),
        "case_losses": sum(delta < -epsilon for delta in case_deltas.values()),
        "case_deltas": dict(sorted(case_deltas.items())),
    }


class BehavioralEvaluationRunner:
    def __init__(
        self,
        *,
        repo_root: Path,
        skill_root: Path,
        output_dir: Path,
        command: Sequence[str],
        model: str | None,
        reasoning_effort: str,
        timeout_seconds: int,
        grade: bool,
        rubric_document: Mapping[str, Any],
        agent_preamble: str | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.skill_root = skill_root.resolve()
        self.output_dir = output_dir.resolve()
        self.command = list(command)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self.grade = grade
        self.rubrics = {item["id"]: item for item in rubric_document["rubrics"]}
        self.scoring = rubric_document["scoring"]
        self.agent_preamble = agent_preamble

    def _agent_prompt(self, case: Mapping[str, Any]) -> str:
        if not self.agent_preamble:
            return case["prompt"]
        return f"{self.agent_preamble}\n\n{case['prompt']}"

    def run(self, spec: EvaluationRun, case: Mapping[str, Any]) -> dict[str, Any]:
        run_dir = self.output_dir / "runs" / spec.id
        run_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f"autoparallel-eval-{spec.id}-"
        ) as root:
            workspace = Path(root) / "workspace"
            prepare_workspace(
                workspace,
                repo_root=self.repo_root,
                skill_root=self.skill_root,
                case=case,
                include_skill=spec.variant == "skill",
            )
            before = _snapshot_workspace(workspace)
            response_path = run_dir / "response.md"
            trace_path = run_dir / "tool-trace.jsonl"
            completed = self._invoke(
                workspace=workspace,
                prompt=self._agent_prompt(case),
                response_path=response_path,
                trace_path=trace_path,
            )
            after = _snapshot_workspace(workspace)
            workspace_diff = _workspace_diff(before, after)
            _write_json(run_dir / "workspace-diff.json", workspace_diff)
            _copy_changed_artifacts(workspace, run_dir / "artifacts", workspace_diff)
        result: dict[str, Any] = {
            "run_id": spec.id,
            "case_id": spec.case_id,
            "variant": spec.variant,
            "repeat": spec.repeat,
            "mode": case["mode"],
            "invocation": case["invocation"],
            "skill_expected": case["skill_expected"],
            "rubric_id": case["rubric_id"],
            "agent_exit_code": completed.returncode,
            **_skill_trace_fields(
                trace_path,
                response_path=response_path,
                include=spec.variant == "skill",
            ),
            "pairwise_candidate_sha256": _json_sha256(_pairwise_candidate(run_dir)),
        }
        if completed.returncode == 0 and self.grade:
            grade = self._grade(run_dir, self.rubrics[case["rubric_id"]])
            result["score"] = score_grade(
                self.rubrics[case["rubric_id"]], self.scoring, grade
            )
        _write_json(run_dir / "result.json", result)
        return result

    def _invoke(
        self,
        *,
        workspace: Path,
        prompt: str,
        response_path: Path,
        trace_path: Path,
        output_schema: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            *self.command,
            "exec",
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox",
            "workspace-write",
            "-C",
            str(workspace),
            "--thread-source",
            "skill-eval",
            "-c",
            f'model_reasoning_effort="{self.reasoning_effort}"',
            "--json",
            "--output-last-message",
            str(response_path),
        ]
        if self.model:
            command.extend(["--model", self.model])
        if output_schema:
            command.extend(["--output-schema", str(output_schema)])
        command.append("-")
        completed = subprocess.run(
            command,
            cwd=workspace,
            capture_output=True,
            text=True,
            input=prompt,
            timeout=self.timeout_seconds,
            check=False,
        )
        trace_path.write_text(completed.stdout)
        if completed.stderr:
            (trace_path.parent / "agent-stderr.txt").write_text(completed.stderr)
        return completed

    def _grade(self, run_dir: Path, rubric: Mapping[str, Any]) -> Mapping[str, Any]:
        response = (run_dir / "response.md").read_text()
        workspace_diff = json.loads((run_dir / "workspace-diff.json").read_text())
        tool_trace = _grader_tool_trace(run_dir / "tool-trace.jsonl")
        artifacts = _grader_artifacts(run_dir / "artifacts")
        schema = _grader_schema(rubric)
        schema_path = run_dir / "grader-schema.json"
        grade_path = run_dir / "grade.json"
        _write_json(schema_path, schema)
        prompt = (
            "Grade the response against every supplied criterion. Use only scores "
            "0, 0.5, or 1 and cite concise evidence. Return every criterion exactly "
            "once. "
            + GRADING_EVIDENCE_INSTRUCTIONS
            + "\n\nRUBRIC:\n"
            + json.dumps(rubric, indent=2, sort_keys=True)
            + "\n\nRESPONSE:\n"
            + response
            + "\n\nWORKSPACE DIFF:\n"
            + json.dumps(workspace_diff, indent=2, sort_keys=True)
            + "\n\nTOOL TRACE:\n"
            + json.dumps(tool_trace, indent=2, sort_keys=True)
            + "\n\nCHANGED ARTIFACTS:\n"
            + json.dumps(artifacts, indent=2, sort_keys=True)
        )
        with tempfile.TemporaryDirectory(prefix="autoparallel-eval-grader-") as root:
            workspace = Path(root)
            trace_path = run_dir / "grader-trace.jsonl"
            completed = self._invoke(
                workspace=workspace,
                prompt=prompt,
                response_path=grade_path,
                trace_path=trace_path,
                output_schema=schema_path,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"Grader failed for {run_dir.name}")
        return json.loads(grade_path.read_text())

    def grade_pair(
        self,
        skill_result: Mapping[str, Any],
        baseline_result: Mapping[str, Any],
        case: Mapping[str, Any],
        *,
        a_variant: str | None = None,
    ) -> dict[str, Any]:
        pair_id = f"{case['id']}-{skill_result['repeat']:02d}"
        if a_variant is None:
            a_variant, b_variant = _blind_pair_order(pair_id)
        elif a_variant in {"skill", "baseline"}:
            b_variant = "baseline" if a_variant == "skill" else "skill"
        else:
            raise ValueError(f"Invalid candidate A variant: {a_variant}")
        judgment_id = f"{pair_id}-{a_variant}-first"
        pair_dir = self.output_dir / "pairs" / judgment_id
        pair_dir.mkdir(parents=True, exist_ok=True)
        by_variant = {"skill": skill_result, "baseline": baseline_result}
        candidate_a = _pairwise_candidate(
            self.output_dir / "runs" / by_variant[a_variant]["run_id"]
        )
        candidate_b = _pairwise_candidate(
            self.output_dir / "runs" / by_variant[b_variant]["run_id"]
        )
        rubric = self.rubrics[case["rubric_id"]]
        schema_path = pair_dir / "grader-schema.json"
        grade_path = pair_dir / "grade.json"
        _write_json(schema_path, _pairwise_grader_schema(rubric))
        prompt = (
            "Blindly compare two responses to the same task. Judge only the supplied "
            "rubric and evidence. Prefer the response that more completely satisfies "
            "critical criteria, then weighted coverage; return tie when they are "
            "materially equivalent. Do not speculate about which response used a "
            "skill. Return every criterion preference exactly once. "
            + GRADING_EVIDENCE_INSTRUCTIONS
            + "\n\nTASK:\n"
            + self._agent_prompt(case)
            + "\n\nRUBRIC:\n"
            + json.dumps(rubric, indent=2, sort_keys=True)
            + "\n\nCANDIDATE A:\n"
            + json.dumps(candidate_a, indent=2, sort_keys=True)
            + "\n\nCANDIDATE B:\n"
            + json.dumps(candidate_b, indent=2, sort_keys=True)
        )
        with tempfile.TemporaryDirectory(prefix="autoparallel-eval-pair-") as root:
            completed = self._invoke(
                workspace=Path(root),
                prompt=prompt,
                response_path=grade_path,
                trace_path=pair_dir / "grader-trace.jsonl",
                output_schema=schema_path,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"Pairwise grader failed for {judgment_id}")
        grade = json.loads(grade_path.read_text())
        criterion_ids = {item["id"] for item in rubric["criteria"]}
        graded_ids = [item["id"] for item in grade["criterion_preferences"]]
        if len(graded_ids) != len(set(graded_ids)) or set(graded_ids) != criterion_ids:
            raise ValueError(
                f"Pairwise grader criterion IDs mismatch for {judgment_id}"
            )
        winner = grade["winner"]
        result = {
            "pair_id": pair_id,
            "judgment_id": judgment_id,
            "case_id": case["id"],
            "repeat": skill_result["repeat"],
            "a_variant": a_variant,
            "b_variant": b_variant,
            "winner": winner,
            "winner_variant": (
                "tie" if winner == "tie" else {"a": a_variant, "b": b_variant}[winner]
            ),
            "confidence": grade["confidence"],
            "rationale": grade["rationale"],
            "criterion_preferences": grade["criterion_preferences"],
            "a_candidate_sha256": _json_sha256(candidate_a),
            "b_candidate_sha256": _json_sha256(candidate_b),
        }
        _write_json(pair_dir / "result.json", result)
        return result


def _skill_copy_ignores(path: str, names: list[str]) -> set[str]:
    ignored = {name for name in names if name == "__pycache__" or name.endswith(".pyc")}
    if Path(path).name == "references":
        ignored.add("agent-routing-evals.json")
    return ignored


def _safe_workspace_path(workspace: Path, relative: str) -> Path:
    destination = (workspace / relative).resolve()
    if not destination.is_relative_to(workspace.resolve()):
        raise ValueError(f"Workspace destination escapes isolation: {relative}")
    return destination


def _snapshot_workspace(workspace: Path) -> dict[str, dict[str, Any]]:
    snapshot = {}
    for path in sorted(item for item in workspace.rglob("*") if item.is_file()):
        relative = path.relative_to(workspace).as_posix()
        data = path.read_bytes()
        snapshot[relative] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }
    return snapshot


def _workspace_diff(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    added = [{"path": path, **after[path]} for path in sorted(set(after) - set(before))]
    deleted = [
        {"path": path, **before[path]} for path in sorted(set(before) - set(after))
    ]
    modified = [
        {"path": path, "before": before[path], "after": after[path]}
        for path in sorted(set(before) & set(after))
        if before[path] != after[path]
    ]
    return {"added": added, "modified": modified, "deleted": deleted}


def _copy_changed_artifacts(
    workspace: Path, destination: Path, workspace_diff: Mapping[str, Any]
) -> None:
    changed = [item["path"] for item in workspace_diff["added"]]
    changed.extend(item["path"] for item in workspace_diff["modified"])
    for relative in changed:
        source = _safe_workspace_path(workspace, relative)
        target = _safe_workspace_path(destination, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _grader_schema(rubric: Mapping[str, Any]) -> dict[str, Any]:
    criterion_ids = [item["id"] for item in rubric["criteria"]]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["criteria"],
        "properties": {
            "criteria": {
                "type": "array",
                "minItems": len(criterion_ids),
                "maxItems": len(criterion_ids),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "score", "evidence"],
                    "properties": {
                        "id": {"type": "string", "enum": criterion_ids},
                        "score": {"type": "number", "enum": [0, 0.5, 1]},
                        "evidence": {"type": "string"},
                    },
                },
            }
        },
    }


def _pairwise_grader_schema(rubric: Mapping[str, Any]) -> dict[str, Any]:
    criterion_ids = [item["id"] for item in rubric["criteria"]]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "winner",
            "confidence",
            "rationale",
            "criterion_preferences",
        ],
        "properties": {
            "winner": {"type": "string", "enum": ["a", "b", "tie"]},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "rationale": {"type": "string"},
            "criterion_preferences": {
                "type": "array",
                "minItems": len(criterion_ids),
                "maxItems": len(criterion_ids),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "preference", "evidence"],
                    "properties": {
                        "id": {"type": "string", "enum": criterion_ids},
                        "preference": {
                            "type": "string",
                            "enum": ["a", "b", "tie"],
                        },
                        "evidence": {"type": "string"},
                    },
                },
            },
        },
    }


def _pairwise_candidate(run_dir: Path) -> dict[str, Any]:
    return {
        "response": _truncate_text((run_dir / "response.md").read_text(), 20_000),
        "workspace_diff": json.loads((run_dir / "workspace-diff.json").read_text()),
        "tool_trace": _grader_tool_trace(run_dir / "tool-trace.jsonl"),
        "changed_artifacts": _grader_artifacts(run_dir / "artifacts"),
    }


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _blind_pair_order(pair_id: str) -> tuple[str, str]:
    if hashlib.sha256(pair_id.encode()).digest()[0] % 2:
        return "skill", "baseline"
    return "baseline", "skill"


def plan_pairwise_comparisons(
    results: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    grouped: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    for result in results:
        if result.get("variant") not in {"skill", "baseline"}:
            continue
        if result.get("agent_exit_code") != 0 or "runner_error" in result:
            continue
        key = (result["case_id"], result["repeat"])
        grouped.setdefault(key, {})[result["variant"]] = result
    return [
        (items["skill"], items["baseline"])
        for _, items in sorted(grouped.items())
        if set(items) == {"skill", "baseline"}
    ]


def collapse_reciprocal_pairwise_results(
    results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for result in results:
        key = (result["case_id"], result["repeat"])
        grouped.setdefault(key, []).append(result)
    collapsed = []
    for (case_id, repeat), judgments in sorted(grouped.items()):
        by_a_variant = {item["a_variant"]: item for item in judgments}
        if len(judgments) != 2 or set(by_a_variant) != {"skill", "baseline"}:
            continue
        winners = {item["winner_variant"] for item in judgments}
        order_agreement = len(winners) == 1
        collapsed.append(
            {
                "pair_id": f"{case_id}-{repeat:02d}",
                "case_id": case_id,
                "repeat": repeat,
                "winner_variant": next(iter(winners)) if order_agreement else "tie",
                "order_agreement": order_agreement,
                "judgment_ids": sorted(item["judgment_id"] for item in judgments),
            }
        )
    return collapsed


def summarize_pairwise_results(
    results: Sequence[Mapping[str, Any]],
    *,
    position_results: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    by_case: dict[str, dict[str, int]] = {}
    counts = {"skill": 0, "baseline": 0, "tie": 0}
    position_wins = {"a": 0, "b": 0, "tie": 0}
    by_skill_position = {
        "a": {"skill": 0, "baseline": 0, "tie": 0},
        "b": {"skill": 0, "baseline": 0, "tie": 0},
    }
    for result in results:
        winner = result["winner_variant"]
        counts[winner] += 1
        case_counts = by_case.setdefault(
            result["case_id"], {"skill": 0, "baseline": 0, "tie": 0}
        )
        case_counts[winner] += 1
    for result in position_results if position_results is not None else results:
        winner = result["winner_variant"]
        position_wins[result["winner"]] += 1
        skill_position = "a" if result["a_variant"] == "skill" else "b"
        by_skill_position[skill_position][winner] += 1
    decisive = counts["skill"] + counts["baseline"]
    position_summary = {}
    for position, position_counts in by_skill_position.items():
        position_decisive = position_counts["skill"] + position_counts["baseline"]
        position_summary[position] = {
            "pair_count": sum(position_counts.values()),
            "skill_wins": position_counts["skill"],
            "baseline_wins": position_counts["baseline"],
            "ties": position_counts["tie"],
            "skill_decisive_win_rate": (
                position_counts["skill"] / position_decisive
                if position_decisive
                else None
            ),
        }
    case_majority_counts = {"skill": 0, "baseline": 0, "tie": 0}
    for case_counts in by_case.values():
        if case_counts["skill"] > case_counts["baseline"]:
            case_majority_counts["skill"] += 1
        elif case_counts["baseline"] > case_counts["skill"]:
            case_majority_counts["baseline"] += 1
        else:
            case_majority_counts["tie"] += 1
    decisive_cases = case_majority_counts["skill"] + case_majority_counts["baseline"]
    agreement_count = sum(result.get("order_agreement") is True for result in results)
    disagreement_count = sum(
        result.get("order_agreement") is False for result in results
    )
    return {
        "pair_count": len(results),
        "judgment_count": len(
            position_results if position_results is not None else results
        ),
        "reciprocal_order_agreement_count": agreement_count,
        "reciprocal_order_disagreement_count": disagreement_count,
        "skill_wins": counts["skill"],
        "baseline_wins": counts["baseline"],
        "ties": counts["tie"],
        "skill_decisive_win_rate": counts["skill"] / decisive if decisive else None,
        "skill_decisive_win_rate_wilson_95": _wilson_interval(
            counts["skill"], decisive
        ),
        "position_wins": position_wins,
        "by_skill_position": position_summary,
        "case_majority": {
            "case_count": len(by_case),
            "skill_wins": case_majority_counts["skill"],
            "baseline_wins": case_majority_counts["baseline"],
            "ties": case_majority_counts["tie"],
            "skill_decisive_win_rate": (
                case_majority_counts["skill"] / decisive_cases
                if decisive_cases
                else None
            ),
            "skill_decisive_win_rate_wilson_95": _wilson_interval(
                case_majority_counts["skill"], decisive_cases
            ),
        },
        "cases": dict(sorted(by_case.items())),
    }


def _bootstrap_mean_interval(
    values: Sequence[float], *, samples: int = 20_000, seed: int = 0
) -> dict[str, Any] | None:
    if not values:
        return None
    rng = random.Random(seed)
    means = sorted(mean(rng.choice(values) for _ in values) for _ in range(samples))
    return {
        "lower": means[int(0.025 * samples)],
        "upper": means[int(0.975 * samples)],
        "samples": samples,
        "seed": seed,
        "unit": "case",
    }


def _wilson_interval(
    successes: int, trials: int, *, z: float = 1.959963984540054
) -> dict[str, float] | None:
    if trials == 0:
        return None
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    center = (proportion + z**2 / (2 * trials)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1 - proportion) / trials + z**2 / (4 * trials**2))
        / denominator
    )
    return {"lower": center - radius, "upper": center + radius}


def _grader_tool_trace(path: Path) -> list[dict[str, Any]]:
    relevant_types = {"command_execution", "file_change", "mcp_tool_call"}
    items: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = event.get("item")
        if isinstance(item, Mapping) and item.get("type") in relevant_types:
            normalized = dict(item)
            for field in ("aggregated_output", "output"):
                if isinstance(normalized.get(field), str):
                    normalized[field] = _truncate_text(normalized[field], 4_000)
            items.append(normalized)
    return items


def _trace_mentions_skill(path: Path) -> bool:
    return (
        _skill_trace_fields(path, response_path=None, include=True)[
            "skill_usage_observed"
        ]
        is True
    )


def _skill_trace_fields(
    path: Path, *, response_path: Path | None, include: bool
) -> dict[str, bool | None]:
    if not include:
        return {
            "skill_usage_observed": None,
            "skill_file_read_observed": None,
            "skill_access_failure_observed": None,
        }
    trace = path.read_text().lower()
    usage_observed = any(
        marker in trace
        for marker in ("autoparallel-optimizer", "autoparallel optimizer skill")
    )
    file_read_observed = False
    narrative = []
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = event.get("item")
        if not isinstance(item, Mapping):
            continue
        if item.get("type") == "agent_message":
            narrative.append(str(item.get("text", "")).lower())
        if item.get("type") != "command_execution":
            continue
        command = str(item.get("command", "")).lower()
        output = str(item.get("aggregated_output", "")).lower()
        if (
            "autoparallel-optimizer/skill.md" in command
            and item.get("exit_code") == 0
            and (
                "name: autoparallel-optimizer" in output
                or "# autoparallel optimizer" in output
            )
        ):
            file_read_observed = True
    if response_path is not None and response_path.is_file():
        narrative.append(response_path.read_text().lower())
    combined_text = "\n".join(narrative)
    access_failure_observed = "skill" in combined_text and any(
        phrase in combined_text
        for phrase in (
            "instructions were unavailable",
            "skill file could not be read",
            "skill guidance could not be read",
            "inspection command was blocked",
            "sandbox rejected the read",
        )
    )
    return {
        "skill_usage_observed": usage_observed,
        "skill_file_read_observed": file_read_observed,
        "skill_access_failure_observed": access_failure_observed,
    }


def _source_revision(repo_root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _tracked_source_dirty(repo_root: Path) -> bool | None:
    completed = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode not in {0, 1}:
        return None
    return completed.returncode == 1


def _truncate_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    retained = limit // 2
    omitted = len(value) - 2 * retained
    return (
        value[:retained]
        + f"\n... <{omitted} characters omitted> ...\n"
        + value[-retained:]
    )


def _grader_artifacts(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    artifacts = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        data = path.read_bytes()
        artifact: dict[str, Any] = {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        if len(data) <= 100_000:
            try:
                artifact["text"] = data.decode("utf-8")
            except UnicodeDecodeError:
                pass
        artifacts.append(artifact)
    return artifacts


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "agent-routing-evals.json" or "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _fixtures_sha256(repo_root: Path, suite: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    relative_paths = sorted(
        {item["source"] for case in suite["cases"] for item in case["workspace_copy"]}
    )
    for relative in relative_paths:
        path = (repo_root / relative).resolve()
        if not path.is_file() or not path.is_relative_to(repo_root.resolve()):
            raise ValueError(f"Invalid evaluation fixture: {relative}")
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _validate_resume_state(
    output_dir: Path,
    *,
    config: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    if not output_dir.is_dir():
        raise ValueError(f"Resume output directory does not exist: {output_dir}")
    try:
        existing_config = json.loads(
            (output_dir / "evaluation-config.json").read_text()
        )
        existing_protocol = json.loads((output_dir / "protocol.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError(f"Resume metadata is missing or invalid: {error}") from error
    if existing_config != config:
        differing = sorted(set(existing_config) | set(config))
        differing = [
            key for key in differing if existing_config.get(key) != config.get(key)
        ]
        raise ValueError(f"Resume configuration differs: {differing}")
    if existing_protocol != protocol:
        raise ValueError("Resume protocol differs")


def _load_completed_run(
    output_dir: Path, spec: EvaluationRun, *, require_score: bool
) -> dict[str, Any] | None:
    path = output_dir / "runs" / spec.id / "result.json"
    try:
        result = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    expected = {
        "run_id": spec.id,
        "case_id": spec.case_id,
        "variant": spec.variant,
        "repeat": spec.repeat,
    }
    if not isinstance(result, dict) or any(
        result.get(key) != value for key, value in expected.items()
    ):
        return None
    if result.get("agent_exit_code") != 0 or "runner_error" in result:
        return None
    if not isinstance(result.get("pairwise_candidate_sha256"), str):
        return None
    try:
        current_candidate_sha256 = _json_sha256(_pairwise_candidate(path.parent))
    except (OSError, json.JSONDecodeError):
        return None
    if result["pairwise_candidate_sha256"] != current_candidate_sha256:
        return None
    if require_score and "score" not in result:
        return None
    return result


def _load_completed_pair(
    output_dir: Path,
    skill_result: Mapping[str, Any],
    baseline_result: Mapping[str, Any],
    *,
    expected_a_variant: str,
) -> dict[str, Any] | None:
    pair_id = f"{skill_result['case_id']}-{skill_result['repeat']:02d}"
    judgment_id = f"{pair_id}-{expected_a_variant}-first"
    path = output_dir / "pairs" / judgment_id / "result.json"
    try:
        result = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    expected_b_variant = "baseline" if expected_a_variant == "skill" else "skill"
    candidate_hashes = {
        "skill": skill_result.get("pairwise_candidate_sha256"),
        "baseline": baseline_result.get("pairwise_candidate_sha256"),
    }
    if not isinstance(result, dict) or any(
        (
            result.get("pair_id") != pair_id,
            result.get("judgment_id") != judgment_id,
            result.get("case_id") != skill_result["case_id"],
            result.get("repeat") != skill_result["repeat"],
            result.get("a_variant") != expected_a_variant,
            result.get("b_variant") != expected_b_variant,
            result.get("winner") not in {"a", "b", "tie"},
            result.get("winner_variant") not in {"skill", "baseline", "tie"},
            result.get("a_candidate_sha256") != candidate_hashes[expected_a_variant],
            result.get("b_candidate_sha256") != candidate_hashes[expected_b_variant],
            "runner_error" in result,
        )
    ):
        return None
    expected_winner = (
        "tie"
        if result["winner"] == "tie"
        else {"a": expected_a_variant, "b": expected_b_variant}[result["winner"]]
    )
    if result["winner_variant"] != expected_winner:
        return None
    return result


def _clear_incomplete_artifact(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _acquire_campaign_lock(output_dir: Path) -> TextIO:
    lock = (output_dir / ".behavioral-evaluation.lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise ValueError(
            f"Evaluation campaign is already active: {output_dir}"
        ) from error
    return lock


def _evaluation_failed(summary: Mapping[str, Any]) -> bool:
    if summary.get("failed_run_count", 0):
        return True
    pairwise = summary.get("pairwise_comparison")
    return isinstance(pairwise, Mapping) and any(
        pairwise.get(field, 0)
        for field in ("failed_pair_count", "failed_pair_judgment_count")
    )


def main() -> None:
    skill_root = Path(__file__).resolve().parents[1]
    repo_root = skill_root.parents[2]
    default_eval_root = repo_root / "tests/agent_skills/autoparallel_optimizer"
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path, nargs="?")
    parser.add_argument("--cases", type=Path, default=default_eval_root / "cases.json")
    parser.add_argument(
        "--rubric", type=Path, default=default_eval_root / "rubric.json"
    )
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--no-grade", action="store_true")
    parser.add_argument("--pairwise-grade", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--codex-command", default="codex")
    parser.add_argument("--model")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()

    suite = json.loads(args.cases.read_text())
    rubric = json.loads(args.rubric.read_text())
    runs = plan_runs(
        suite,
        repeats=args.repeats,
        include_baseline=not args.no_baseline,
        case_ids=set(args.case_ids) if args.case_ids else None,
    )
    if args.plan:
        print(json.dumps([run.__dict__ | {"run_id": run.id} for run in runs], indent=2))
        return
    if args.output_dir is None:
        parser.error("output_dir is required unless --plan is used")
    if args.jobs < 1 or args.timeout_seconds < 1:
        parser.error("--jobs and --timeout-seconds must be positive")
    if args.pairwise_grade and (args.no_grade or args.no_baseline):
        parser.error("--pairwise-grade requires individual grading and baselines")
    unknown_cases = set(args.case_ids or ()) - {case["id"] for case in suite["cases"]}
    if unknown_cases:
        parser.error(f"unknown case IDs: {sorted(unknown_cases)}")

    config = {
        "schema_version": 1,
        "source_revision": _source_revision(repo_root),
        "tracked_source_dirty": _tracked_source_dirty(repo_root),
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "fixtures_sha256": _fixtures_sha256(repo_root, suite),
        "rubric_sha256": hashlib.sha256(args.rubric.read_bytes()).hexdigest(),
        "skill_sha256": _tree_sha256(skill_root),
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "repeats": args.repeats or suite["protocol"]["repeat_count"],
        "case_ids": sorted(args.case_ids) if args.case_ids else None,
        "baseline_enabled": not args.no_baseline,
        "grading_enabled": not args.no_grade,
        "pairwise_grading_enabled": args.pairwise_grade,
        "pairwise_ordering": PAIRWISE_ORDERING,
        "codex_command": args.codex_command,
        "timeout_seconds": args.timeout_seconds,
    }
    if not args.resume:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        campaign_lock = _acquire_campaign_lock(args.output_dir)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    if args.resume:
        try:
            _validate_resume_state(
                args.output_dir,
                config=config,
                protocol=suite["protocol"],
            )
        except ValueError as error:
            parser.error(str(error))
    else:
        _write_json(args.output_dir / "protocol.json", suite["protocol"])
        _write_json(args.output_dir / "evaluation-config.json", config)
    runner = BehavioralEvaluationRunner(
        repo_root=repo_root,
        skill_root=skill_root,
        output_dir=args.output_dir,
        command=shlex.split(args.codex_command),
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        timeout_seconds=args.timeout_seconds,
        grade=not args.no_grade,
        rubric_document=rubric,
        agent_preamble=suite["protocol"].get("agent_preamble"),
    )
    cases = {case["id"]: case for case in suite["cases"]}
    results = []
    pending_runs = []
    for spec in runs:
        completed = (
            _load_completed_run(
                args.output_dir,
                spec,
                require_score=not args.no_grade,
            )
            if args.resume
            else None
        )
        if completed is not None:
            results.append(completed)
        else:
            _clear_incomplete_artifact(args.output_dir / "runs" / spec.id)
            pending_runs.append(spec)
    reused_run_count = len(results)
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        run_futures = {
            executor.submit(runner.run, spec, cases[spec.case_id]): spec
            for spec in pending_runs
        }
        for future in as_completed(run_futures):
            spec = run_futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                results.append(
                    {
                        "run_id": spec.id,
                        "case_id": spec.case_id,
                        "variant": spec.variant,
                        "repeat": spec.repeat,
                        "runner_error": f"{type(error).__name__}: {error}",
                    }
                )
    results.sort(key=lambda item: item["run_id"])
    _write_json(args.output_dir / "results.json", results)
    summary = summarize_results(results)
    summary["reused_run_count"] = reused_run_count
    if args.pairwise_grade:
        pairwise_results = []
        pair_specs = plan_pairwise_comparisons(results)
        pending_pairs = []
        reused_by_pair: dict[str, int] = {}
        for skill_result, baseline_result in pair_specs:
            pair_id = f"{skill_result['case_id']}-{skill_result['repeat']:02d}"
            for a_variant in ("skill", "baseline"):
                completed = (
                    _load_completed_pair(
                        args.output_dir,
                        skill_result,
                        baseline_result,
                        expected_a_variant=a_variant,
                    )
                    if args.resume
                    else None
                )
                if completed is not None:
                    pairwise_results.append(completed)
                    reused_by_pair[pair_id] = reused_by_pair.get(pair_id, 0) + 1
                else:
                    judgment_id = f"{pair_id}-{a_variant}-first"
                    _clear_incomplete_artifact(args.output_dir / "pairs" / judgment_id)
                    pending_pairs.append(
                        (skill_result, baseline_result, a_variant, judgment_id)
                    )
        reused_pair_judgment_count = len(pairwise_results)
        reused_pair_count = sum(count == 2 for count in reused_by_pair.values())
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            pair_futures = {
                executor.submit(
                    runner.grade_pair,
                    skill_result,
                    baseline_result,
                    cases[skill_result["case_id"]],
                    a_variant=a_variant,
                ): (skill_result, baseline_result, a_variant, judgment_id)
                for skill_result, baseline_result, a_variant, judgment_id in pending_pairs
            }
            for future in as_completed(pair_futures):
                skill_result, _, a_variant, judgment_id = pair_futures[future]
                try:
                    pairwise_results.append(future.result())
                except Exception as error:
                    pairwise_results.append(
                        {
                            "pair_id": (
                                f"{skill_result['case_id']}-"
                                f"{skill_result['repeat']:02d}"
                            ),
                            "judgment_id": judgment_id,
                            "case_id": skill_result["case_id"],
                            "repeat": skill_result["repeat"],
                            "a_variant": a_variant,
                            "runner_error": f"{type(error).__name__}: {error}",
                        }
                    )
        pairwise_results.sort(key=lambda item: item["judgment_id"])
        _write_json(args.output_dir / "pairwise-results.json", pairwise_results)
        scored_judgments = [
            item for item in pairwise_results if "winner_variant" in item
        ]
        consensus_pairs = collapse_reciprocal_pairwise_results(scored_judgments)
        _write_json(args.output_dir / "pairwise-consensus.json", consensus_pairs)
        summary["pairwise_comparison"] = summarize_pairwise_results(
            consensus_pairs,
            position_results=scored_judgments,
        )
        summary["pairwise_comparison"]["failed_pair_count"] = len(pair_specs) - len(
            consensus_pairs
        )
        summary["pairwise_comparison"]["failed_pair_judgment_count"] = 2 * len(
            pair_specs
        ) - len(scored_judgments)
        summary["pairwise_comparison"]["reused_pair_count"] = reused_pair_count
        summary["pairwise_comparison"][
            "reused_pair_judgment_count"
        ] = reused_pair_judgment_count
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    campaign_lock.close()
    if _evaluation_failed(summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
