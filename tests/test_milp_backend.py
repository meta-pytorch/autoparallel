# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pulp
import pytest

from autoparallel.optimize_sharding import ShardingOptimizer, _make_milp_solver

HIGHS = pulp.HiGHS(msg=False).available()
needs_highs = pytest.mark.skipif(not HIGHS, reason="highspy not installed")
BACKENDS = ["cbc", pytest.param("highs", marks=needs_highs)]


def test_an_unknown_backend_is_refused():
    with pytest.raises(ValueError, match="milp_backend must be one of"):
        ShardingOptimizer(None, None, milp_backend="gurobi")


def test_highs_without_highspy_is_refused(monkeypatch):
    monkeypatch.setattr(pulp.HiGHS, "available", lambda self: False)
    with pytest.raises(RuntimeError, match="needs highspy"):
        ShardingOptimizer(None, None, milp_backend="highs")


@pytest.mark.parametrize("backend", BACKENDS)
def test_each_backend_gets_the_time_limit(backend):
    solver = _make_milp_solver(backend, msg=False, time_limit=5)
    assert solver.timeLimit == 5
    assert isinstance(solver, pulp.HiGHS if backend == "highs" else pulp.PULP_CBC_CMD)


def _placement_problem(memory_budget=8):
    """Three nodes with two strategies each, a cost for neighbours that disagree,
    and a memory row that rules out the cheapest choice. The optimum is 6
    (a=1, b=1, c=1: cost 3 + 2 + 1, memory 7), found by enumeration below."""
    prob = pulp.LpProblem("placement", pulp.LpMinimize)
    x = {
        (n, s): pulp.LpVariable(f"x_{n}_{s}", cat="Binary")
        for n in "abc"
        for s in (0, 1)
    }
    cost = {
        ("a", 0): 1,
        ("a", 1): 3,
        ("b", 0): 1,
        ("b", 1): 2,
        ("c", 0): 4,
        ("c", 1): 1,
    }
    memory = {
        ("a", 0): 5,
        ("a", 1): 1,
        ("b", 0): 5,
        ("b", 1): 1,
        ("c", 0): 1,
        ("c", 1): 5,
    }
    switch = []
    for left, right in (("a", "b"), ("b", "c")):
        for s in (0, 1):
            y = pulp.LpVariable(f"y_{left}{s}_{right}{1 - s}", lowBound=0)
            prob += y >= x[left, s] + x[right, 1 - s] - 1
            switch.append(y)
    prob += pulp.lpSum(cost[k] * v for k, v in x.items()) + 2 * pulp.lpSum(switch)
    for n in "abc":
        prob += pulp.lpSum(x[n, s] for s in (0, 1)) == 1
    prob += pulp.lpSum(memory[k] * v for k, v in x.items()) <= memory_budget
    return prob, x, cost, memory


def _best_by_enumeration(cost, memory, memory_budget=8):
    values = []
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                choice = {"a": a, "b": b, "c": c}
                if sum(memory[n, s] for n, s in choice.items()) > memory_budget:
                    continue
                value = sum(cost[n, s] for n, s in choice.items())
                values.append(value + 2 * (a != b) + 2 * (b != c))
    return min(values)


def _optimizer(prob, x, backend):
    keys = {(i, s, 0, 0): x[n, s] for i, n in enumerate("abc") for s in (0, 1)}

    def no_walk():
        raise AssertionError("no point to check constraints against")

    return SimpleNamespace(
        prob=prob,
        solver_time_limit_seconds=60,
        milp_backend=backend,
        decision_vars={k: SimpleNamespace(var=v) for k, v in keys.items()},
        _root_to_linked={},
        _apply_memory_constraint=lambda: None,
        get_violated_constraints_log=(
            no_walk if backend == "highs" else (lambda: "violated constraints")
        ),
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_each_backend_finds_the_optimum(backend):
    prob, x, cost, memory = _placement_problem()
    optimizer = _optimizer(prob, x, backend)

    ShardingOptimizer._solve(optimizer)

    assert pulp.value(prob.objective) == pytest.approx(
        _best_by_enumeration(cost, memory)
    )
    # One strategy per node, even when HiGHS returns 0.9999999 for a 1.
    assert sorted(k[0] for k in optimizer.selected_keys) == [0, 1, 2]


@pytest.mark.parametrize("backend", BACKENDS)
def test_each_backend_reports_infeasibility(backend):
    prob, x, *_ = _placement_problem(memory_budget=2)
    optimizer = _optimizer(prob, x, backend)

    with pytest.raises(RuntimeError, match="could not find a feasible solution"):
        ShardingOptimizer._solve(optimizer)


def test_a_binary_within_the_integrality_tolerance_is_selected():
    prob, x, *_ = _placement_problem()
    optimizer = _optimizer(prob, x, "cbc")
    optimizer.prob = SimpleNamespace(
        solve=lambda solver: None,
        solutionTime=0.1,
        status=pulp.LpStatusOptimal,
        sol_status=pulp.LpSolutionOptimal,
    )
    values = {(0, 0): 0.9999999, (0, 1): 1e-7, (1, 0): 1.0000001, (1, 1): 0.0}
    values.update({(2, 0): 0.0, (2, 1): 1.0})
    for (i, s, _, _), dv in optimizer.decision_vars.items():
        dv.var.varValue = values[i, s]

    ShardingOptimizer._solve(optimizer)

    assert optimizer.selected_keys == [(0, 0, 0, 0), (1, 0, 0, 0), (2, 1, 0, 0)]


def test_a_node_without_a_selected_strategy_is_an_error():
    optimizer = SimpleNamespace(
        # Nodes 0 and 1 have decision variables; the output (2) has none.
        nodes=["a", "b", "output"],
        decision_vars={(0, 0, 0, 0): None, (1, 0, 0, 0): None},
        selected_keys=[(0, 0, 0, 0)],
        _resolve_decision_var=lambda key: SimpleNamespace(strategy="S"),
    )
    with pytest.raises(RuntimeError, match=r"no strategy for 1 nodes, e.g. \['b'\]"):
        ShardingOptimizer._extract_and_validate_solution(optimizer)

    optimizer.selected_keys.append((1, 0, 0, 0))
    assert ShardingOptimizer._extract_and_validate_solution(optimizer) == {
        "a": "S",
        "b": "S",
    }
