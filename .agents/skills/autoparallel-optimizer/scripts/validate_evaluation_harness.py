# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Iterable

HANDOFF_REQUIREMENTS = [
    "Reordered graph metrics are estimates distinct from the ILP objective and "
    "measured runtime.",
    "The pass adds no overlap control dependencies; later Inductor scheduling "
    "or fusion can change execution order.",
]


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _root_name(node: ast.expr) -> str | None:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _calls(node: ast.AST) -> Iterable[ast.Call]:
    return (item for item in ast.walk(node) if isinstance(item, ast.Call))


def validate_harness_source(source: str) -> list[str]:
    tree = ast.parse(source)
    calls = list(_calls(tree))
    collector_names: set[str] = set()
    collector_contexts: list[ast.With | ast.AsyncWith] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if (
                isinstance(value, ast.Call)
                and _call_name(value) == "ReorderedMetricsCollector"
            ):
                collector_names.update(
                    target.id for target in targets if isinstance(target, ast.Name)
                )
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        matched = False
        for item in node.items:
            context = item.context_expr
            if (
                isinstance(context, ast.Call)
                and _call_name(context) == "ReorderedMetricsCollector"
            ):
                matched = True
                if isinstance(item.optional_vars, ast.Name):
                    collector_names.add(item.optional_vars.id)
            elif isinstance(context, ast.Name) and context.id in collector_names:
                matched = True
        if matched:
            collector_contexts.append(node)

    issues = []
    if not collector_contexts:
        issues.append("COLLECTOR_CONTEXT_MISSING")

    complete_lifetime = False
    for collector_context in collector_contexts:
        names = {_call_name(call) for call in _calls(collector_context)}
        if {"AutoParallel", "compile", "backward"} <= names:
            complete_lifetime = True
    if collector_contexts and not complete_lifetime:
        issues.append("COLLECTOR_LIFETIME_INCOMPLETE")

    shared_cost_model = any(
        _call_name(call) == "AutoParallel"
        and any(
            keyword.arg == "cost_model"
            and isinstance(keyword.value, ast.Attribute)
            and keyword.value.attr == "cost_model"
            and _root_name(keyword.value) in collector_names
            for keyword in call.keywords
        )
        for call in calls
    )
    if not shared_cost_model:
        issues.append("COST_MODEL_NOT_SHARED")

    overlap_disabled = any(
        _call_name(call) == "autoparallel_backend"
        and any(
            keyword.arg == "overlap_scheduling"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is False
            for keyword in call.keywords
        )
        for call in calls
    )
    if not overlap_disabled:
        issues.append("OVERLAP_SCHEDULING_NOT_DISABLED")

    if not any(_call_name(call) == "backward" for call in calls):
        issues.append("BACKWARD_NOT_EXECUTED")
    if not any(
        _call_name(call) == "require_forward_backward"
        and _root_name(call.func) in collector_names
        for call in calls
    ):
        issues.append("FORWARD_BACKWARD_NOT_REQUIRED")

    explicit_metrics_path = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            argument.arg == "metrics_path"
            for argument in (
                node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            )
        )
        for node in ast.walk(tree)
    )
    if not explicit_metrics_path:
        issues.append("METRICS_PATH_NOT_EXPLICIT")
    if not any(
        _call_name(call) == "write_json"
        and _root_name(call.func) in collector_names
        and (
            (
                bool(call.args)
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id == "metrics_path"
            )
            or any(
                keyword.arg == "path"
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == "metrics_path"
                for keyword in call.keywords
            )
        )
        for call in calls
    ):
        issues.append("METRICS_NOT_WRITTEN")

    if any(
        _call_name(call) == "aten_autobucketing_reordering_pass"
        or any(keyword.arg == "post_grad_custom_post_pass" for keyword in call.keywords)
        for call in calls
    ):
        issues.append("DUPLICATE_SCHEDULER_CONFIGURATION")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("harness", type=Path)
    args = parser.parse_args()
    try:
        issues = validate_harness_source(args.harness.read_text())
    except (OSError, SyntaxError) as error:
        print(
            json.dumps(
                {
                    "valid": False,
                    "error": str(error),
                    "handoff_requirements": HANDOFF_REQUIREMENTS,
                },
                sort_keys=True,
            )
        )
        raise SystemExit(1) from error
    print(
        json.dumps(
            {
                "valid": not issues,
                "issues": issues,
                "handoff_requirements": HANDOFF_REQUIREMENTS,
            },
            sort_keys=True,
        )
    )
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
