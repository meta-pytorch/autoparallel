# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pulp
import pytest

from autoparallel.optimize_sharding import ShardingOptimizer


class _Problem:
    def __init__(self, status, solution_status, solution_time):
        self.status = status
        self.sol_status = solution_status
        self.solutionTime = solution_time

    def solve(self, solver):
        assert solver.timeLimit == 7


def _optimizer(problem):
    return SimpleNamespace(
        prob=problem,
        solver_time_limit_seconds=7,
        _apply_memory_constraint=lambda: None,
        get_violated_constraints_log=lambda: "",
    )


@pytest.mark.parametrize(
    "status, solution_status",
    [
        (pulp.LpStatusNotSolved, pulp.LpSolutionNoSolutionFound),
        (pulp.LpStatusInfeasible, pulp.LpSolutionInfeasible),
        (pulp.LpStatusOptimal, pulp.LpSolutionIntegerFeasible),
    ],
)
def test_sharding_optimizer_reports_deadline_as_timeout(status, solution_status):
    optimizer = _optimizer(_Problem(status, solution_status, solution_time=7))

    with pytest.raises(TimeoutError, match="did not prove an optimal solution"):
        ShardingOptimizer._solve(optimizer)


def test_sharding_optimizer_preserves_fast_infeasibility():
    optimizer = _optimizer(
        _Problem(
            pulp.LpStatusInfeasible,
            pulp.LpSolutionInfeasible,
            solution_time=1,
        )
    )

    with pytest.raises(RuntimeError, match="could not find a feasible solution"):
        ShardingOptimizer._solve(optimizer)


@pytest.mark.parametrize(
    "status, solution_status, message",
    [
        (
            pulp.LpStatusNotSolved,
            pulp.LpSolutionNoSolutionFound,
            "solve failed",
        ),
        (
            pulp.LpStatusOptimal,
            pulp.LpSolutionIntegerFeasible,
            "non-optimal solution",
        ),
    ],
)
def test_sharding_optimizer_preserves_early_solver_failures(
    status, solution_status, message
):
    optimizer = _optimizer(_Problem(status, solution_status, solution_time=1))

    with pytest.raises(RuntimeError, match=message):
        ShardingOptimizer._solve(optimizer)
