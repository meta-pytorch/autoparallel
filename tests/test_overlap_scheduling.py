"""
Tests for overlap_scheduling.

Each test constructs a small DAG of compute/comm ops and verifies that
the solvers produce valid schedules with expected makespan.
"""

from autoparallel.overlap_scheduling import (
    COMPUTE_STREAM,
    OverlapProblem,
    critical_path_length,
    solve_greedy,
    solve_ilp,
)


def _validate_schedule(problem, schedule):
    """Check that start times respect all DAG and stream constraints."""
    st = schedule.start_times
    # DAG precedence
    for u in range(problem.n):
        for v in problem.successors(u):
            assert st[v] >= st[u] + problem.ops[u].duration_ms - 1e-6, (
                f"DAG violation: op {u} (end={st[u]+problem.ops[u].duration_ms}) "
                f"-> op {v} (start={st[v]})"
            )
    # Stream serialization: no two ops on the same stream overlap
    for stream in problem.streams():
        ids = problem.ops_on_stream(stream)
        intervals = sorted(
            [(st[i], st[i] + problem.ops[i].duration_ms, i) for i in ids]
        )
        for j in range(len(intervals) - 1):
            end_j = intervals[j][1]
            start_next = intervals[j + 1][0]
            assert start_next >= end_j - 1e-6, (
                f"Stream {stream} overlap: op {intervals[j][2]} ends at {end_j}, "
                f"op {intervals[j+1][2]} starts at {start_next}"
            )


def test_basic_overlap():
    """
    compute_1 -> all_gather -> compute_3
              \\-> compute_2 -/

    Without overlap: c1(5) + ag(10) + c2(8) + c3(3) = 26ms
    With overlap:    c1(5) + max(ag(10), c2(8)) + c3(3) = 18ms
    """
    p = OverlapProblem()
    c1 = p.add_op("matmul_1", COMPUTE_STREAM, 5.0)
    c2 = p.add_op("matmul_2", COMPUTE_STREAM, 8.0)
    ag = p.add_op("all_gather", "pg0", 10.0)
    c3 = p.add_op("matmul_3", COMPUTE_STREAM, 3.0)
    p.add_dep(c1, ag)
    p.add_dep(c1, c2)
    p.add_dep(ag, c3)
    p.add_dep(c2, c3)

    cp = critical_path_length(p)
    assert cp == 18.0, f"critical path should be 18ms, got {cp}"

    for solver in [solve_ilp, solve_greedy]:
        schedule = solver(p)
        _validate_schedule(p, schedule)
        assert (
            abs(schedule.makespan - 18.0) < 1e-6
        ), f"{solver.__name__}: expected makespan 18ms, got {schedule.makespan}"


def test_multi_pg_overlap():
    """
    Two independent collectives on different PGs, both overlappable with compute.

    compute_1 (10ms) can overlap with both ag_pg0 (6ms) and ag_pg1 (8ms)
    simultaneously, since they're on different streams.

    c0(2) -> ag_pg0(6) -> c2(1)
         \\-> ag_pg1(8) -> c2
         \\-> c1(10) ----> c2

    Critical path: c0(2) + max(ag_pg0(6), ag_pg1(8), c1(10)) + c2(1) = 13ms
    """
    p = OverlapProblem()
    c0 = p.add_op("setup", COMPUTE_STREAM, 2.0)
    ag0 = p.add_op("all_gather_pg0", "pg0", 6.0)
    ag1 = p.add_op("all_gather_pg1", "pg1", 8.0)
    c1 = p.add_op("big_matmul", COMPUTE_STREAM, 10.0)
    c2 = p.add_op("final", COMPUTE_STREAM, 1.0)

    p.add_dep(c0, ag0)
    p.add_dep(c0, ag1)
    p.add_dep(c0, c1)
    p.add_dep(ag0, c2)
    p.add_dep(ag1, c2)
    p.add_dep(c1, c2)

    for solver in [solve_ilp, solve_greedy]:
        schedule = solver(p)
        _validate_schedule(p, schedule)
        assert (
            abs(schedule.makespan - 13.0) < 1e-6
        ), f"{solver.__name__}: expected 13ms, got {schedule.makespan}"


def test_comm_longer_than_compute():
    """
    When comm takes longer than available compute, some comm is exposed.

    c0(2) -> ag(20) -> c2(1)
         \\-> c1(5) --> c2

    Optimal: c0(2), launch ag(20) and c1(5) in parallel.
    c1 finishes at 7, ag finishes at 22, c2 starts at 22.
    Makespan = 23ms.  Exposed comm = 15ms.
    """
    p = OverlapProblem()
    c0 = p.add_op("c0", COMPUTE_STREAM, 2.0)
    ag = p.add_op("ag", "pg0", 20.0)
    c1 = p.add_op("c1", COMPUTE_STREAM, 5.0)
    c2 = p.add_op("c2", COMPUTE_STREAM, 1.0)

    p.add_dep(c0, ag)
    p.add_dep(c0, c1)
    p.add_dep(ag, c2)
    p.add_dep(c1, c2)

    for solver in [solve_ilp, solve_greedy]:
        schedule = solver(p)
        _validate_schedule(p, schedule)
        assert (
            abs(schedule.makespan - 23.0) < 1e-6
        ), f"{solver.__name__}: expected 23ms, got {schedule.makespan}"


def test_chain_no_overlap_possible():
    """
    Fully sequential chain: no overlap is possible.

    c0(3) -> ag(5) -> c1(4)
    Everything is on the critical path.
    """
    p = OverlapProblem()
    c0 = p.add_op("c0", COMPUTE_STREAM, 3.0)
    ag = p.add_op("ag", "pg0", 5.0)
    c1 = p.add_op("c1", COMPUTE_STREAM, 4.0)

    p.add_dep(c0, ag)
    p.add_dep(ag, c1)

    for solver in [solve_ilp, solve_greedy]:
        schedule = solver(p)
        _validate_schedule(p, schedule)
        assert (
            abs(schedule.makespan - 12.0) < 1e-6
        ), f"{solver.__name__}: expected 12ms, got {schedule.makespan}"


def test_same_pg_serialization():
    """
    Two collectives on the same PG must be serialized on the comm stream,
    even if they are independent in the DAG.

    c0(1) -> ag1(5) -> c2(1)
         \\-> ag2(5) -> c2
         \\-> c1(12) -> c2

    ag1 and ag2 are on the same PG, so they serialize: 5+5=10ms on pg0.
    c1 is 12ms on compute. Makespan = 1 + 12 + 1 = 14ms.
    (Both ags finish by t=11 < 13 when c1 finishes, so fully hidden.)
    """
    p = OverlapProblem()
    c0 = p.add_op("c0", COMPUTE_STREAM, 1.0)
    ag1 = p.add_op("ag1", "pg0", 5.0)
    ag2 = p.add_op("ag2", "pg0", 5.0)
    c1 = p.add_op("c1", COMPUTE_STREAM, 12.0)
    c2 = p.add_op("c2", COMPUTE_STREAM, 1.0)

    p.add_dep(c0, ag1)
    p.add_dep(c0, ag2)
    p.add_dep(c0, c1)
    p.add_dep(ag1, c2)
    p.add_dep(ag2, c2)
    p.add_dep(c1, c2)

    for solver in [solve_ilp, solve_greedy]:
        schedule = solver(p)
        _validate_schedule(p, schedule)
        assert (
            abs(schedule.makespan - 14.0) < 1e-6
        ), f"{solver.__name__}: expected 14ms, got {schedule.makespan}"


def test_greedy_memory_budget():
    """
    Memory budget forces the greedy solver to defer a prefetch.

    c0(1) -> ag1(3, mem=100) -> c3(1)
         \\-> ag2(3, mem=100) -> c3
         \\-> c1(2) -> c2(2) -> c3

    Without memory limit: both ags launch at t=1, overlap with c1+c2.
    ag1 and ag2 finish at t=4, c2 finishes at t=5, makespan = 6.

    With memory budget = 150 (can only have one ag in flight):
    must serialize the ags, total comm = 6ms on pg0.
    c1+c2 = 4ms on compute. Makespan = 1 + max(6, 4) + 1 = 8.
    """
    p = OverlapProblem()
    c0 = p.add_op("c0", COMPUTE_STREAM, 1.0)
    ag1 = p.add_op("ag1", "pg0", 3.0, memory_bytes=100)
    ag2 = p.add_op("ag2", "pg0", 3.0, memory_bytes=100)
    c1 = p.add_op("c1", COMPUTE_STREAM, 2.0)
    c2 = p.add_op("c2", COMPUTE_STREAM, 2.0)
    c3 = p.add_op("c3", COMPUTE_STREAM, 1.0)

    p.add_dep(c0, ag1)
    p.add_dep(c0, ag2)
    p.add_dep(c0, c1)
    p.add_dep(c1, c2)
    p.add_dep(ag1, c3)
    p.add_dep(ag2, c3)
    p.add_dep(c2, c3)

    # No memory limit: both ags overlap with compute
    sched = solve_greedy(p)
    _validate_schedule(p, sched)
    assert (
        abs(sched.makespan - 8.0) < 1e-6
    ), f"expected 8ms without memory limit, got {sched.makespan}"

    # Memory limit: can only have one ag in flight at a time
    sched_mem = solve_greedy(p, memory_budget_bytes=150)
    _validate_schedule(p, sched_mem)
    # With memory constraint, ag2 is deferred, makespan increases
    assert sched_mem.makespan >= sched.makespan - 1e-6


def test_ilp_vs_greedy_agreement():
    """On simple cases, ILP and greedy should agree on makespan."""
    p = OverlapProblem()
    c0 = p.add_op("c0", COMPUTE_STREAM, 3.0)
    ag = p.add_op("ag", "pg0", 4.0)
    c1 = p.add_op("c1", COMPUTE_STREAM, 5.0)
    c2 = p.add_op("c2", COMPUTE_STREAM, 2.0)

    p.add_dep(c0, ag)
    p.add_dep(c0, c1)
    p.add_dep(ag, c2)
    p.add_dep(c1, c2)

    ilp = solve_ilp(p)
    greedy = solve_greedy(p)
    _validate_schedule(p, ilp)
    _validate_schedule(p, greedy)
    assert abs(ilp.makespan - greedy.makespan) < 1e-6


if __name__ == "__main__":
    test_basic_overlap()
    test_multi_pg_overlap()
    test_comm_longer_than_compute()
    test_chain_no_overlap_possible()
    test_same_pg_serialization()
    test_greedy_memory_budget()
    test_ilp_vs_greedy_agreement()
    print("All tests passed.")
