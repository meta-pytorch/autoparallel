"""
Tests for overlap_scheduling.

Each test constructs a small DAG of compute/comm ops and verifies that
the solvers produce valid schedules with expected makespan.

The FX graph tests build actual torch.fx graphs with c10d functional
collectives, run the full build_problem -> solve -> apply_schedule pipeline,
and verify the resulting graph order.
"""

from typing import Callable

import torch
import torch.fx as fx

from autoparallel.overlap_scheduling import (
    COMPUTE_STREAM,
    NodeInfo,
    NodeKind,
    OverlapProblem,
    build_overlap_problem,
    critical_path_length,
    reorder_for_overlap,
    solve_greedy,
    solve_ilp,
)

# c10d functional ops for building test graphs
_AG = torch.ops._c10d_functional.all_gather_into_tensor.default
_RS = torch.ops._c10d_functional.reduce_scatter_tensor.default
_WAIT = torch.ops._c10d_functional.wait_tensor.default


def _validate_schedule(problem, schedule):
    """Check that start times respect all DAG and stream constraints."""
    st = schedule.start_times
    ct = schedule.completion_times
    # DAG precedence: successor can't start until predecessor completes
    for u in range(problem.n):
        for v in problem.successors(u):
            assert ct[u] <= st[v] + 1e-6, (
                f"DAG violation: op {u} (completion={ct[u]}) "
                f"-> op {v} (start={st[v]})"
            )
    # Stream serialization: no two ops on the same stream overlap.
    # For comm ops the execution interval on the comm stream is
    # [completion - duration, completion], since the comm may be queued
    # after dispatch.
    for stream in problem.streams():
        ids = problem.ops_on_stream(stream)
        if stream == COMPUTE_STREAM:
            intervals = sorted(
                [(st[i], st[i] + problem.ops[i].duration_ms, i) for i in ids]
            )
        else:
            intervals = sorted(
                [
                    (
                        ct[i] - problem.ops[i].duration_ms,
                        ct[i],
                        i,
                    )
                    for i in ids
                ]
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

    Without memory limit: both ags launch at t=1, serialize on pg0
    (1-4, 4-7), compute c1+c2 runs 1-5, c3 waits for ag2 at t=7.
    Makespan = 8.

    With memory budget = 150 (can only have one ag in flight):
    ag2 is deferred until budget allows, makespan increases.
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

    # No memory limit (solve_greedy called directly): both ags overlap with compute
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


# ---------------------------------------------------------------------------
# FX graph integration tests
# ---------------------------------------------------------------------------


def _get_call_function_names(graph: fx.Graph) -> list[str]:
    """Extract ordered names of call_function nodes."""
    return [n.name for n in graph.nodes if n.op == "call_function"]


def _check_fx_topo_order(graph: fx.Graph) -> None:
    """Verify every node appears after all its inputs in the graph."""
    seen: set[fx.Node] = set()
    for node in graph.nodes:
        for inp in node.all_input_nodes:
            assert inp in seen, f"{node.name} appears before its input {inp.name}"
        seen.add(node)


def _make_classify(
    costs: dict[str, float],
) -> Callable:
    """Build a classify function from a {node_name: duration_ms} dict.

    Comm starts are identified by target; their PG name is extracted from
    the last positional arg.  Everything else goes on the compute stream.
    """

    def classify(node: fx.Node) -> NodeInfo:
        if node.op in ("placeholder", "output", "get_attr"):
            return NodeInfo(NodeKind.SKIP, COMPUTE_STREAM, 0.0)
        if node.target is _WAIT:
            return NodeInfo(NodeKind.COMM_WAIT, COMPUTE_STREAM, 0.0)
        if node.target in (_AG, _RS):
            pg_name = str(node.args[-1])
            duration = costs.get(node.name, 1.0)
            return NodeInfo(NodeKind.COMM_START, pg_name, duration)
        duration = costs.get(node.name, 0.0)
        return NodeInfo(NodeKind.COMPUTE, COMPUTE_STREAM, duration)

    return classify


def _build_basic_overlap_graph() -> fx.Graph:
    """
    mm1(x, w) -> ag(mm1) -> wait -> mm3(wait, mm2)
                             mm2(x, w) ---/

    Original order: mm1, ag, wait, mm2, mm3
    mm2 is independent of ag/wait and can overlap with the all_gather.
    """
    graph = fx.Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("w")
    mm1 = graph.call_function(torch.ops.aten.mm.default, (x, w))
    mm1.name = "mm1"
    ag = graph.call_function(_AG, (mm1, 2, "0"))
    ag.name = "ag"
    wait = graph.call_function(_WAIT, (ag,))
    wait.name = "wait"
    mm2 = graph.call_function(torch.ops.aten.mm.default, (x, w))
    mm2.name = "mm2"
    mm3 = graph.call_function(torch.ops.aten.mm.default, (wait, mm2))
    mm3.name = "mm3"
    graph.output(mm3)
    return graph


def test_fx_basic_reorder():
    """
    After reordering, mm2 should move before wait to overlap with ag.
    Expected order: mm1, ag, mm2, wait, mm3
    """
    costs = {"mm1": 5.0, "ag": 10.0, "mm2": 8.0, "mm3": 3.0}
    classify = _make_classify(costs)

    for solver_name in ["ilp", "greedy"]:
        graph = _build_basic_overlap_graph()
        assert _get_call_function_names(graph) == [
            "mm1",
            "ag",
            "wait",
            "mm2",
            "mm3",
        ]

        schedule = reorder_for_overlap(graph, classify, solver=solver_name)

        _check_fx_topo_order(graph)
        assert _get_call_function_names(graph) == [
            "mm1",
            "ag",
            "mm2",
            "wait",
            "mm3",
        ], f"{solver_name} produced wrong order: {_get_call_function_names(graph)}"
        assert (
            abs(schedule.makespan - 18.0) < 1e-6
        ), f"{solver_name}: expected 18ms, got {schedule.makespan}"


def test_fx_build_problem_correctness():
    """Verify that build_overlap_problem produces the right abstract DAG."""
    graph = _build_basic_overlap_graph()
    costs = {"mm1": 5.0, "ag": 10.0, "mm2": 8.0, "mm3": 3.0}
    classify = _make_classify(costs)

    problem, op_to_node, node_to_op = build_overlap_problem(graph, classify)

    # 4 ops: mm1, ag, mm2, mm3 (wait is dissolved)
    assert problem.n == 4

    # Check streams
    compute_ops = problem.ops_on_stream(COMPUTE_STREAM)
    comm_ops = problem.ops_on_stream("0")
    assert len(compute_ops) == 3  # mm1, mm2, mm3
    assert len(comm_ops) == 1  # ag

    # Check that wait node maps to same op as ag
    wait_node = None
    ag_node = None
    for node in graph.nodes:
        if node.name == "wait":
            wait_node = node
        if node.name == "ag":
            ag_node = node
    assert node_to_op[wait_node] == node_to_op[ag_node]

    # Check dependencies: mm3 should depend on ag (via dissolved wait) and mm2
    mm3_op = node_to_op[next(n for n in graph.nodes if n.name == "mm3")]
    mm3_preds = set(problem.predecessors(mm3_op))
    ag_op = node_to_op[ag_node]
    mm2_op = node_to_op[next(n for n in graph.nodes if n.name == "mm2")]
    assert ag_op in mm3_preds, "mm3 should depend on ag (via dissolved wait)"
    assert mm2_op in mm3_preds, "mm3 should depend on mm2"


def _build_two_layer_graph() -> fx.Graph:
    """
    Two-layer pattern with reduce_scatters on the same PG.

    mm1(x, w1) -> rs1(mm1) -> wait1 -> mm2(wait1, w2) -> rs2(mm2) -> wait2 -> mm3(wait2, w3)

    Original order is fully sequential; no overlap possible because each
    layer depends on the previous layer's reduce_scatter result... unless
    we add an independent branch.

    We add an independent mm_side that can overlap with rs1:

    mm1 -> rs1 -> wait1 --> mm2 -> rs2 -> wait2 -> mm3(wait2, w3)
       \\-> mm_side(x, w2) -/

    mm_side can run during rs1, overlapping comm with compute.
    """
    graph = fx.Graph()
    x = graph.placeholder("x")
    w1 = graph.placeholder("w1")
    w2 = graph.placeholder("w2")
    w3 = graph.placeholder("w3")

    mm1 = graph.call_function(torch.ops.aten.mm.default, (x, w1))
    mm1.name = "mm1"
    rs1 = graph.call_function(_RS, (mm1, "sum", 2, "0"))
    rs1.name = "rs1"
    wait1 = graph.call_function(_WAIT, (rs1,))
    wait1.name = "wait1"
    # mm_side is independent of rs1/wait1 — placed after wait1 in original
    # order but doesn't actually depend on it
    mm_side = graph.call_function(torch.ops.aten.mm.default, (x, w2))
    mm_side.name = "mm_side"
    mm2 = graph.call_function(torch.ops.aten.mm.default, (wait1, mm_side))
    mm2.name = "mm2"
    rs2 = graph.call_function(_RS, (mm2, "sum", 2, "0"))
    rs2.name = "rs2"
    wait2 = graph.call_function(_WAIT, (rs2,))
    wait2.name = "wait2"
    mm3 = graph.call_function(torch.ops.aten.mm.default, (wait2, w3))
    mm3.name = "mm3"
    graph.output(mm3)
    return graph


def test_fx_two_layer_reorder():
    """
    mm_side should move before wait1 to overlap with rs1.

    Original: mm1, rs1, wait1, mm_side, mm2, rs2, wait2, mm3
    Expected: mm1, rs1, mm_side, wait1, mm2, rs2, wait2, mm3
    """
    costs = {
        "mm1": 5.0,
        "rs1": 4.0,
        "mm_side": 5.0,
        "mm2": 5.0,
        "rs2": 4.0,
        "mm3": 2.0,
    }
    classify = _make_classify(costs)

    for solver_name in ["ilp", "greedy"]:
        graph = _build_two_layer_graph()
        assert _get_call_function_names(graph) == [
            "mm1",
            "rs1",
            "wait1",
            "mm_side",
            "mm2",
            "rs2",
            "wait2",
            "mm3",
        ]

        _ = reorder_for_overlap(graph, classify, solver=solver_name)
        names = _get_call_function_names(graph)
        _check_fx_topo_order(graph)

        # mm_side must come before wait1 (overlap with rs1)
        assert names.index("mm_side") < names.index(
            "wait1"
        ), f"{solver_name}: mm_side should be before wait1, got {names}"
        # rs1 must come before mm_side (rs1 launches first, then compute overlaps)
        assert names.index("rs1") < names.index(
            "mm_side"
        ), f"{solver_name}: rs1 should be before mm_side, got {names}"


def _build_multi_pg_graph() -> fx.Graph:
    """
    Two collectives on different PGs, both overlappable with big_mm.

    mm0(x, w) -> ag0(mm0, pg="0") -> wait0 \\
             \\-> ag1(mm0, pg="1") -> wait1 -> mm_final(wait0, wait1, big_mm)
             \\-> big_mm(x, w) ----------/

    Original order: mm0, ag0, wait0, ag1, wait1, big_mm, mm_final
    Both ag0 and ag1 can overlap with big_mm since they're on different PGs.
    """
    graph = fx.Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("w")

    mm0 = graph.call_function(torch.ops.aten.mm.default, (x, w))
    mm0.name = "mm0"
    ag0 = graph.call_function(_AG, (mm0, 2, "0"))
    ag0.name = "ag0"
    wait0 = graph.call_function(_WAIT, (ag0,))
    wait0.name = "wait0"
    ag1 = graph.call_function(_AG, (mm0, 2, "1"))
    ag1.name = "ag1"
    wait1 = graph.call_function(_WAIT, (ag1,))
    wait1.name = "wait1"
    big_mm = graph.call_function(torch.ops.aten.mm.default, (x, w))
    big_mm.name = "big_mm"
    # mm_final depends on all three results
    mm_final = graph.call_function(
        torch.ops.aten.mm.default,
        (wait0, wait1),
    )
    # Also add big_mm as a dependency via a dummy add
    mm_final_2 = graph.call_function(
        torch.ops.aten.add.Tensor,
        (mm_final, big_mm),
    )
    mm_final.name = "mm_final"
    mm_final_2.name = "mm_final_2"
    graph.output(mm_final_2)
    return graph


def test_fx_multi_pg_reorder():
    """
    Both collectives and big_mm should be launched right after mm0.
    big_mm should come before both waits.

    Original: mm0, ag0, wait0, ag1, wait1, big_mm, mm_final, mm_final_2
    Expected: big_mm moves before wait0 and wait1.
    """
    costs = {
        "mm0": 2.0,
        "ag0": 6.0,
        "ag1": 8.0,
        "big_mm": 10.0,
        "mm_final": 1.0,
        "mm_final_2": 0.5,
    }
    classify = _make_classify(costs)

    for solver_name in ["ilp", "greedy"]:
        graph = _build_multi_pg_graph()
        schedule = reorder_for_overlap(graph, classify, solver=solver_name)
        names = _get_call_function_names(graph)
        _check_fx_topo_order(graph)

        # big_mm should come before both waits
        assert names.index("big_mm") < names.index(
            "wait0"
        ), f"{solver_name}: big_mm before wait0, got {names}"
        assert names.index("big_mm") < names.index(
            "wait1"
        ), f"{solver_name}: big_mm before wait1, got {names}"

        # Makespan: mm0(2) + max(ag0(6), ag1(8), big_mm(10)) + mm_final(1) + mm_final_2(0.5) = 13.5
        assert (
            abs(schedule.makespan - 13.5) < 1e-6
        ), f"{solver_name}: expected 13.5ms, got {schedule.makespan}"


def test_fx_no_reorder_needed():
    """When the graph is already optimal, the order should be preserved."""
    graph = fx.Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("w")
    # Fully sequential: no overlap possible
    mm1 = graph.call_function(torch.ops.aten.mm.default, (x, w))
    mm1.name = "mm1"
    ag = graph.call_function(_AG, (mm1, 2, "0"))
    ag.name = "ag"
    wait = graph.call_function(_WAIT, (ag,))
    wait.name = "wait"
    mm2 = graph.call_function(torch.ops.aten.mm.default, (wait, w))
    mm2.name = "mm2"
    graph.output(mm2)

    original_order = _get_call_function_names(graph)
    classify = _make_classify({"mm1": 3.0, "ag": 5.0, "mm2": 4.0})
    reorder_for_overlap(graph, classify)
    assert _get_call_function_names(graph) == original_order
    _check_fx_topo_order(graph)


def test_fx_duplicate_deps():
    """
    When a node consumes a wait result via multiple args, the dependency
    should be deduplicated (not cause issues in the solver).
    """
    graph = fx.Graph()
    x = graph.placeholder("x")
    mm1 = graph.call_function(torch.ops.aten.mm.default, (x, x))
    mm1.name = "mm1"
    ag = graph.call_function(_AG, (mm1, 2, "0"))
    ag.name = "ag"
    wait = graph.call_function(_WAIT, (ag,))
    wait.name = "wait"
    # mm2 uses wait result in both args -> duplicate dependency on ag
    mm2 = graph.call_function(torch.ops.aten.mm.default, (wait, wait))
    mm2.name = "mm2"
    graph.output(mm2)

    classify = _make_classify({"mm1": 3.0, "ag": 5.0, "mm2": 4.0})

    # Should not crash due to duplicate edges
    problem, _, node_to_op = build_overlap_problem(graph, classify)
    ag_op = node_to_op[next(n for n in graph.nodes if n.name == "ag")]
    mm2_op = node_to_op[next(n for n in graph.nodes if n.name == "mm2")]
    # Should have exactly one edge from ag to mm2
    assert problem.predecessors(mm2_op).count(ag_op) == 1

    reorder_for_overlap(graph, classify)
    _check_fx_topo_order(graph)


def _build_comm_prep_graph() -> fx.Graph:
    """
    Graph where a "comm prep" op (reshape) sits between a parameter and an AG,
    with heavy compute ops on the main path.

    heavy1(x, w) -> heavy2(heavy1, w) -> mm_final(heavy2, wait)
    param -> reshape(param) -> ag(reshape) -> wait -/

    Original order: heavy1, heavy2, reshape, ag, wait, mm_final
    reshape should move before heavy1 to allow ag to overlap with heavy1+heavy2.
    """
    graph = fx.Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("w")
    param = graph.placeholder("param")

    heavy1 = graph.call_function(torch.ops.aten.mm.default, (x, w))
    heavy1.name = "heavy1"
    heavy2 = graph.call_function(torch.ops.aten.mm.default, (heavy1, w))
    heavy2.name = "heavy2"
    # reshape is independent — depends only on param (placeholder)
    reshape = graph.call_function(torch.ops.aten.reshape.default, (param, [-1]))
    reshape.name = "reshape"
    ag = graph.call_function(_AG, (reshape, 2, "0"))
    ag.name = "ag"
    wait = graph.call_function(_WAIT, (ag,))
    wait.name = "wait"
    mm_final = graph.call_function(torch.ops.aten.mm.default, (heavy2, wait))
    mm_final.name = "mm_final"
    graph.output(mm_final)
    return graph


def test_fx_comm_prep_reorder():
    """
    Comm prep ops (reshape feeding an AG, descending from placeholders)
    should move freely with their comms, not be pinned to original order.

    Original: heavy1, heavy2, reshape, ag, wait, mm_final

    Without comm_prep awareness, order-preservation would keep reshape
    after heavy2, preventing any overlap (37.1ms actual execution).

    The ILP finds the optimal: reshape, ag, heavy1, heavy2, wait, mm_final
    giving 22.1ms (ag fully hidden behind heavy1+heavy2).

    The greedy solver picks heavy1 first (lower ALAP = more urgent), then
    reshape, ag, heavy2: giving 27.1ms.  Both solvers benefit from comm_prep
    awareness — reshape moves before heavy2, enabling ag overlap.
    """
    costs = {
        "heavy1": 10.0,
        "heavy2": 10.0,
        "reshape": 0.1,
        "ag": 15.0,
        "mm_final": 2.0,
    }
    classify = _make_classify(costs)

    for solver_name in ["ilp", "greedy"]:
        graph = _build_comm_prep_graph()
        assert _get_call_function_names(graph) == [
            "heavy1",
            "heavy2",
            "reshape",
            "ag",
            "wait",
            "mm_final",
        ]

        schedule = reorder_for_overlap(graph, classify, solver=solver_name)
        names = _get_call_function_names(graph)
        _check_fx_topo_order(graph)

        # Both solvers: reshape and ag must move before heavy2 to enable overlap.
        # Without comm_prep fix, reshape would stay after heavy2 (original order).
        assert names.index("reshape") < names.index(
            "heavy2"
        ), f"{solver_name}: reshape should move before heavy2, got {names}"
        assert names.index("ag") < names.index(
            "heavy2"
        ), f"{solver_name}: ag should move before heavy2, got {names}"

        if solver_name == "ilp":
            # ILP finds optimal: reshape before heavy1
            assert names.index("reshape") < names.index(
                "heavy1"
            ), f"ilp: reshape should move before heavy1, got {names}"
            assert (
                abs(schedule.makespan - 22.1) < 1e-6
            ), f"ilp: expected 22.1ms, got {schedule.makespan}"
        else:
            # Greedy picks heavy1 first (lower ALAP), then reshape
            assert (
                abs(schedule.makespan - 27.1) < 1e-6
            ), f"greedy: expected 27.1ms, got {schedule.makespan}"


if __name__ == "__main__":
    test_basic_overlap()
    test_multi_pg_overlap()
    test_comm_longer_than_compute()
    test_chain_no_overlap_possible()
    test_same_pg_serialization()
    test_greedy_memory_budget()
    test_ilp_vs_greedy_agreement()
    test_fx_basic_reorder()
    test_fx_build_problem_correctness()
    test_fx_two_layer_reorder()
    test_fx_multi_pg_reorder()
    test_fx_no_reorder_needed()
    test_fx_duplicate_deps()
    test_fx_comm_prep_reorder()
    print("All tests passed.")
