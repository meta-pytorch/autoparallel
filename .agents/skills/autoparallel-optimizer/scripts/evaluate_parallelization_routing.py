from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

_FIELDS = (
    "invoke_skill",
    "autoparallel_disposition",
    "autoparallel_outcome",
    "competitiveness_verdict",
    "benchmark_action",
    "selected_strategy",
    "report_required",
)


def evaluate_responses(
    suite: Mapping[str, Any], responses: Mapping[str, Any]
) -> dict[str, Any]:
    if suite.get("schema_version") != 1:
        raise ValueError("unsupported routing evaluation schema")
    suite_cases = suite.get("cases", [])
    response_items = [case for case in responses.get("cases", []) if "id" in case]
    expected_cases = {case["id"]: case for case in suite_cases}
    if len(expected_cases) != len(suite_cases):
        raise ValueError("routing evaluation suite contains duplicate case IDs")
    response_cases = {case["id"]: case for case in response_items}
    duplicate_case_ids = sorted(
        {
            case["id"]
            for case in response_items
            if sum(item["id"] == case["id"] for item in response_items) > 1
        }
    )
    field_matches = {field: 0 for field in _FIELDS}
    results = []
    for case_id, case in expected_cases.items():
        response = response_cases.get(case_id)
        mismatches = []
        if response is None:
            mismatches.append("missing response")
        else:
            expected = case["expected"]
            for field in _FIELDS:
                if response.get(field) == expected[field]:
                    field_matches[field] += 1
                else:
                    mismatches.append(
                        f"{field}: expected {expected[field]!r}, "
                        f"got {response.get(field)!r}"
                    )
        results.append(
            {
                "id": case_id,
                "passed": not mismatches,
                "mismatches": mismatches,
            }
        )
    unexpected = sorted(set(response_cases) - set(expected_cases))
    passed = sum(result["passed"] for result in results)
    field_scores = {
        field: matches / len(expected_cases) if expected_cases else 0.0
        for field, matches in field_matches.items()
    }
    return {
        "schema_version": 1,
        "passed": (
            passed == len(results) and not unexpected and not duplicate_case_ids
        ),
        "score": passed / len(results) if results else 0.0,
        "field_score": (
            sum(field_matches.values()) / (len(expected_cases) * len(_FIELDS))
            if expected_cases
            else 0.0
        ),
        "field_scores": field_scores,
        "passed_case_count": passed,
        "case_count": len(results),
        "duplicate_case_ids": duplicate_case_ids,
        "unexpected_case_ids": unexpected,
        "results": results,
    }


def response_template(suite: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cases": [
            {
                "id": case["id"],
                **{field: None for field in _FIELDS},
            }
            for case in suite.get("cases", [])
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    default_suite = (
        Path(__file__).resolve().parents[1] / "references/agent-routing-evals.json"
    )
    parser.add_argument("responses", type=Path, nargs="?")
    parser.add_argument("--suite", type=Path, default=default_suite)
    parser.add_argument("--template", action="store_true")
    args = parser.parse_args()

    suite = json.loads(args.suite.read_text())
    if args.template:
        print(json.dumps(response_template(suite), indent=2, sort_keys=True))
        return
    if args.responses is None:
        parser.error("responses is required unless --template is used")
    result = evaluate_responses(suite, json.loads(args.responses.read_text()))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
