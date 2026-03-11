"""
Reorder compute and communication ops in a dataflow graph to maximize overlap.

Execution model:
- One compute stream serializes all compute and miscellaneous ops.
- One communication stream per process group serializes collective ops on that PG.
- Different streams execute concurrently.
- DAG edges (dependencies) must be respected.
- Goal: minimize makespan (wall-clock time from first op to last).

Two solvers:
- solve_ilp:  exact solution via mixed-integer linear program (PuLP/CBC).
              Practical when the number of unordered same-stream pairs is < ~5000.
- solve_greedy: ASAP heuristic with ALAP-priority tie-breaking.
                Fast, memory-aware, usually near-optimal.

FX graph integration:
- build_overlap_problem: translate an FX graph + cost model into an OverlapProblem.
  Comm starts become ops on their PG stream; wait nodes are erased and replaced
  by dependency edges from the comm op to the wait's consumers.
- apply_schedule: reorder FX nodes according to the solved schedule, reinserting
  start/wait nodes at their optimal positions.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import pulp
import torch.fx as fx

# ---------------------------------------------------------------------------
# Problem formulation
# ---------------------------------------------------------------------------

COMPUTE_STREAM = "compute"
LAUNCH_OVERHEAD = 0.0


@dataclass
class Op:
    id: int
    name: str
    stream: str  # COMPUTE_STREAM or a process group name
    duration_ms: float
    memory_bytes: int = 0


class OverlapProblem:
    """DAG of operations on typed streams.  Build with add_op / add_dep."""

    def __init__(self) -> None:
        self.ops: list[Op] = []
        self._succ: list[list[int]] = []
        self._pred: list[list[int]] = []
        self._edges: set[tuple[int, int]] = set()

    def add_op(
        self,
        name: str,
        stream: str,
        duration_ms: float,
        memory_bytes: int = 0,
    ) -> int:
        idx = len(self.ops)
        self.ops.append(Op(idx, name, stream, duration_ms, memory_bytes))
        self._succ.append([])
        self._pred.append([])
        return idx

    def add_dep(self, src: int, dst: int) -> None:
        if (src, dst) in self._edges:
            return
        self._edges.add((src, dst))
        self._succ[src].append(dst)
        self._pred[dst].append(src)

    @property
    def n(self) -> int:
        return len(self.ops)

    def successors(self, i: int) -> list[int]:
        return self._succ[i]

    def predecessors(self, i: int) -> list[int]:
        return self._pred[i]

    def streams(self) -> set[str]:
        return {op.stream for op in self.ops}

    def ops_on_stream(self, stream: str) -> list[int]:
        return [op.id for op in self.ops if op.stream == stream]


@dataclass
class Schedule:
    start_times: dict[int, float]
    completion_times: dict[int, float]
    makespan: float

    def order(self) -> list[int]:
        """Op ids sorted by start time."""
        return sorted(self.start_times, key=lambda i: self.start_times[i])


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def topo_sort(problem: OverlapProblem) -> list[int]:
    in_deg = [0] * problem.n
    for i in range(problem.n):
        for j in problem.successors(i):
            in_deg[j] += 1
    queue = [i for i in range(problem.n) if in_deg[i] == 0]
    result: list[int] = []
    while queue:
        u = queue.pop(0)
        result.append(u)
        for v in problem.successors(u):
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)
    assert len(result) == problem.n, "cycle in DAG"
    return result


def compute_asap(problem: OverlapProblem) -> dict[int, float]:
    """As-Soon-As-Possible start times, ignoring stream contention."""
    asap: dict[int, float] = {}
    for u in topo_sort(problem):
        asap[u] = max(
            (asap[p] + problem.ops[p].duration_ms for p in problem.predecessors(u)),
            default=0.0,
        )
    return asap


def compute_alap(problem: OverlapProblem) -> dict[int, float]:
    """As-Late-As-Possible start times, ignoring stream contention."""
    asap = compute_asap(problem)
    horizon = max(asap[i] + problem.ops[i].duration_ms for i in range(problem.n))
    alap: dict[int, float] = {}
    for u in reversed(topo_sort(problem)):
        latest = min(
            (alap[s] for s in problem.successors(u)),
            default=horizon,
        )
        alap[u] = latest - problem.ops[u].duration_ms
    return alap


def critical_path_length(problem: OverlapProblem) -> float:
    """Lower bound on makespan: longest weighted path through the DAG."""
    asap = compute_asap(problem)
    return max(asap[i] + problem.ops[i].duration_ms for i in range(problem.n))


def topo_peak_memory(problem: OverlapProblem) -> int:
    """Peak memory when ops run in topological order (no overlap).

    Uses the same liveness model as solve_greedy: each op's output occupies
    memory_bytes from when it runs until its last consumer runs.
    """
    order = topo_sort(problem)
    remaining = [len(problem.successors(i)) for i in range(problem.n)]
    current = 0
    peak = 0
    for i in order:
        current += problem.ops[i].memory_bytes
        peak = max(peak, current)
        for p in problem.predecessors(i):
            remaining[p] -= 1
            if remaining[p] == 0:
                current -= problem.ops[p].memory_bytes
    return peak


def graph_order_peak_memory(problem: OverlapProblem) -> int:
    """Peak memory when ops run in their original graph order (by op id).

    Since build_overlap_problem adds ops in FX graph node order, iterating
    by op id reproduces the original (unoptimized) schedule's memory profile.
    This gives a tighter bound than topo_peak_memory when many ops (e.g.
    backward all-gathers) have no predecessors in the problem.
    """
    remaining = [len(problem.successors(i)) for i in range(problem.n)]
    current = 0
    peak = 0
    for i in range(problem.n):
        current += problem.ops[i].memory_bytes
        peak = max(peak, current)
        for p in problem.predecessors(i):
            remaining[p] -= 1
            if remaining[p] == 0:
                current -= problem.ops[p].memory_bytes
    return peak


# ---------------------------------------------------------------------------
# ILP solver
# ---------------------------------------------------------------------------


def _compute_ancestors(problem: OverlapProblem) -> list[set[int]]:
    ancestors: list[set[int]] = [set() for _ in range(problem.n)]
    for u in topo_sort(problem):
        for v in problem.successors(u):
            ancestors[v].add(u)
            ancestors[v] |= ancestors[u]
    return ancestors


def solve_ilp(problem: OverlapProblem) -> Schedule:
    """
    Minimize makespan via mixed-integer linear program.

    For each pair of ops on the same stream that are NOT ordered by the DAG,
    a binary variable decides which runs first.  DAG edges and stream
    serialization become linear constraints.

    Complexity is dominated by the number of binary variables (= number of
    unordered same-stream pairs).  Practical for < ~5000 such pairs.
    Memory constraints are not modeled here; use solve_greedy if memory is
    a concern.
    """
    n = problem.n
    big_m = sum(op.duration_ms for op in problem.ops) + 1.0
    ancestors = _compute_ancestors(problem)

    lp = pulp.LpProblem("overlap_schedule", pulp.LpMinimize)
    t = [pulp.LpVariable(f"t_{i}", lowBound=0) for i in range(n)]
    T = pulp.LpVariable("makespan", lowBound=0)
    lp += T

    # Makespan: T >= completion time of every op
    for i in range(n):
        lp += T >= t[i] + problem.ops[i].duration_ms

    # DAG precedence
    for u in range(n):
        for v in problem.successors(u):
            lp += t[v] >= t[u] + problem.ops[u].duration_ms

    # Stream serialization via disjunctive constraints
    n_binary = 0
    for stream in problem.streams():
        ids = problem.ops_on_stream(stream)
        for ia in range(len(ids)):
            a = ids[ia]
            for ib in range(ia + 1, len(ids)):
                b = ids[ib]
                if a in ancestors[b] or b in ancestors[a]:
                    continue  # already ordered by DAG
                x = pulp.LpVariable(f"ord_{a}_{b}", cat=pulp.LpBinary)
                lp += t[b] >= t[a] + problem.ops[a].duration_ms - big_m * (1 - x)
                lp += t[a] >= t[b] + problem.ops[b].duration_ms - big_m * x
                n_binary += 1

    lp.solve(pulp.PULP_CBC_CMD(msg=0))
    assert lp.status == pulp.constants.LpStatusOptimal

    start_times = {i: pulp.value(t[i]) for i in range(n)}
    completion_times = {
        i: start_times[i] + problem.ops[i].duration_ms for i in range(n)
    }
    return Schedule(
        start_times=start_times,
        completion_times=completion_times,
        makespan=pulp.value(T),
    )


# ---------------------------------------------------------------------------
# Greedy solver
# ---------------------------------------------------------------------------


def solve_greedy(
    problem: OverlapProblem,
    memory_budget_bytes: int | None = None,
    max_in_flight: int | None = 2,
) -> Schedule:
    """
    ASAP greedy scheduler with dispatch-aware timing model.

    Models the FX execution model where the compute stream sequentially
    dispatches every op:
    - Compute ops block the compute stream for their full duration.
    - Comm ops block the compute stream for only LAUNCH_OVERHEAD, then
      run asynchronously on their comm stream.
    - Predecessors are considered done at their completion time (which
      for comm ops may be later than dispatch + LAUNCH_OVERHEAD).

    The sort key picks the op with earliest dispatch time (ties broken by
    comm-before-compute, then lowest ALAP).  start_times stores the
    dispatch time for comm ops (when the compute stream queues the comm),
    while completion_times stores when the comm finishes on its stream.
    This way apply_schedule places comm start nodes at their dispatch
    position in the FX graph, not at the (potentially much later) time
    the comm actually begins on a busy comm stream.

    max_in_flight limits, per comm stream, how many dispatched comms can
    have unconsumed results (remaining_consumers > 0).  Leaf comms (e.g.
    reduce-scatters with no downstream consumers) don't count toward this
    limit.  This naturally interleaves prefetches with compute: when the
    limit is reached, compute ops run first, consuming earlier comm results
    and making room for new prefetches.  Default is 2 (one in progress +
    one prefetched ahead).
    """
    alap = compute_alap(problem)
    n = problem.n

    in_deg = [0] * n
    for i in range(n):
        for j in problem.successors(i):
            in_deg[j] += 1

    ready: set[int] = {i for i in range(n) if in_deg[i] == 0}
    completion: dict[int, float] = {}
    start_times: dict[int, float] = {}
    compute_time: float = 0.0
    comm_avail: dict[str, float] = defaultdict(float)

    # Memory tracking: each op's output is live until its last consumer runs
    current_memory = 0
    remaining_consumers = [len(problem.successors(i)) for i in range(n)]

    # In-flight comm tracking: comms dispatched but not yet fully consumed
    in_flight: dict[str, int] = defaultdict(int)

    def _net_memory_delta(op_id: int) -> int:
        """Memory change from scheduling op: output allocated, consumed predecessors freed."""
        delta = problem.ops[op_id].memory_bytes
        for p in problem.predecessors(op_id):
            if remaining_consumers[p] == 1:  # this op is the last consumer
                delta -= problem.ops[p].memory_bytes
        return delta

    while ready:
        best_op: int | None = None
        best_key = (float("inf"), True, float("inf"))

        for op_id in ready:
            op = problem.ops[op_id]
            pred_done = max(
                (completion[p] for p in problem.predecessors(op_id)),
                default=0.0,
            )
            start = max(compute_time, pred_done)

            if memory_budget_bytes is not None:
                delta = _net_memory_delta(op_id)
                if delta > 0 and current_memory + delta > memory_budget_bytes:
                    continue

            # Block non-leaf comms when too many are in-flight on this stream
            if (
                max_in_flight is not None
                and op.stream != COMPUTE_STREAM
                and remaining_consumers[op_id] > 0  # leaf comms always allowed
                and in_flight[op.stream] >= max_in_flight
            ):
                continue

            key = (start, op.stream == COMPUTE_STREAM, alap[op_id])
            if key < best_key:
                best_key = key
                best_op = op_id

        if best_op is None:
            # All ready ops blocked by budget or in-flight limit.  Prefer
            # compute ops: they sit on the critical path and will consume
            # in-flight comms, making room for new prefetches.
            compute_ready = [
                i for i in ready if problem.ops[i].stream == COMPUTE_STREAM
            ]
            pool = compute_ready or list(ready)
            best_op = min(pool, key=lambda i: alap[i])

        op = problem.ops[best_op]
        pred_done = max(
            (completion[p] for p in problem.predecessors(best_op)),
            default=0.0,
        )

        if op.stream == COMPUTE_STREAM:
            start = max(compute_time, pred_done)
            compute_time = start + op.duration_ms
            start_times[best_op] = start
            completion[best_op] = start + op.duration_ms
        else:
            dispatch = max(compute_time, pred_done)
            comm_start = max(dispatch, comm_avail[op.stream])
            comm_end = comm_start + op.duration_ms
            compute_time = dispatch + LAUNCH_OVERHEAD
            comm_avail[op.stream] = comm_end
            start_times[best_op] = dispatch
            completion[best_op] = comm_end
            if remaining_consumers[best_op] > 0:
                in_flight[op.stream] += 1

        ready.remove(best_op)

        current_memory += op.memory_bytes
        for p in problem.predecessors(best_op):
            remaining_consumers[p] -= 1
            if remaining_consumers[p] == 0:
                current_memory -= problem.ops[p].memory_bytes
                if problem.ops[p].stream != COMPUTE_STREAM:
                    in_flight[problem.ops[p].stream] -= 1

        for s in problem.successors(best_op):
            in_deg[s] -= 1
            if in_deg[s] == 0:
                ready.add(s)

    return Schedule(
        start_times=start_times,
        completion_times=completion,
        makespan=max(completion.values()),
    )


# ---------------------------------------------------------------------------
# FX graph integration
# ---------------------------------------------------------------------------


class NodeKind(Enum):
    COMPUTE = "compute"
    COMM_START = "comm_start"
    COMM_WAIT = "comm_wait"
    OTHER = "other"
    SKIP = "skip"


@dataclass
class NodeInfo:
    kind: NodeKind
    stream: str  # COMPUTE_STREAM or PG name
    duration_ms: float
    memory_bytes: int = 0


def build_overlap_problem(
    graph: fx.Graph,
    classify: Callable[[fx.Node], NodeInfo],
) -> tuple[OverlapProblem, dict[int, fx.Node], dict[fx.Node, int]]:
    """
    Translate an FX graph into an OverlapProblem.

    classify(node) returns a NodeInfo describing the node's kind, stream,
    and cost.  Comm starts become ops on their PG stream.  Wait nodes are
    dissolved: their consumers get a direct dependency on the comm op.

    Returns:
        problem:     the OverlapProblem
        op_to_node:  maps op id -> FX node (for comm ops, maps to the start node)
        node_to_op:  maps FX node -> op id (wait nodes map to their start's op)
    """
    problem = OverlapProblem()
    op_to_node: dict[int, fx.Node] = {}
    node_to_op: dict[fx.Node, int] = {}

    # First pass: create ops (skip waits — they'll be wired as deps)
    wait_to_start: dict[fx.Node, fx.Node] = {}
    for node in graph.nodes:
        info = classify(node)
        if info.kind == NodeKind.SKIP:
            continue
        if info.kind == NodeKind.COMM_WAIT:
            # wait_tensor's first arg is the comm start node
            start_node = node.args[0]
            assert isinstance(start_node, fx.Node)
            wait_to_start[node] = start_node
            node_to_op[node] = node_to_op[start_node]
            continue

        op_id = problem.add_op(
            name=node.name,
            stream=info.stream,
            duration_ms=info.duration_ms,
            memory_bytes=info.memory_bytes,
        )
        op_to_node[op_id] = node
        node_to_op[node] = op_id

    # Second pass: add dependency edges
    for node in graph.nodes:
        if node not in node_to_op:
            continue
        dst_op = node_to_op[node]
        for inp in node.all_input_nodes:
            # If inp is a wait node, the real dependency is on the comm start
            src_node = wait_to_start.get(inp, inp)
            if src_node not in node_to_op:
                continue
            src_op = node_to_op[src_node]
            if src_op != dst_op:
                problem.add_dep(src_op, dst_op)

    return problem, op_to_node, node_to_op


def apply_schedule(
    graph: fx.Graph,
    schedule: Schedule,
    problem: OverlapProblem,
    op_to_node: dict[int, fx.Node],
    node_to_op: dict[fx.Node, int],
) -> None:
    """
    Reorder FX graph nodes to match the solved schedule.

    Uses priority-based topological sort to guarantee all FX edges are
    respected while matching schedule start times as closely as possible.

    Sort key is a 4-tuple: (time, node_type, tiebreaker, original_order).
    node_type separates comm starts (0) from compute (1) from waits (2)
    so that at the same dispatch time, comms are launched before compute.
    For comm starts, the tiebreaker is completion_times which reflects the
    solver's intended comm stream order — crucial when multiple comms share
    the same dispatch time (LAUNCH_OVERHEAD ≈ 0).

    Compute ops preserve their original relative order: the solver decides
    time slots (where comms are inserted), but compute ops fill those slots
    in original graph order.  This prevents the solver's ALAP-based
    tie-breaking from reordering independent compute ops within a layer,
    which would change tensor lifetimes and blow up peak memory.
    """
    original_order = {node: i for i, node in enumerate(graph.nodes)}

    # Identify "comm prep" compute ops: trivial ops (e.g. reshape, view)
    # that sit between SKIP'd inputs (parameters) and comm starts, with no
    # real compute in their dependency chain.  These should move freely
    # with their comms rather than being pinned to original order.
    #
    # Symmetrically, "comm post" ops sit between comm results and graph
    # outputs (e.g. reshape after a reduce-scatter wait).  They should
    # also move freely so reduce-scatter waits can be pushed later.
    comm_adjacent: set[int] = set()
    changed = True
    while changed:
        changed = False
        for op_id in range(problem.n):
            if op_id in comm_adjacent or problem.ops[op_id].stream != COMPUTE_STREAM:
                continue
            succs = problem.successors(op_id)
            preds = problem.predecessors(op_id)
            # Comm prep: all preds are SKIP'd or already adjacent,
            # all succs are comms or already adjacent, must have succs.
            is_prep = (
                len(succs) > 0
                and all(
                    problem.ops[s].stream != COMPUTE_STREAM or s in comm_adjacent
                    for s in succs
                )
                and all(p in comm_adjacent for p in preds)
            )
            # Comm post: all succs are SKIP'd/absent or already adjacent,
            # all preds are comms or already adjacent, must have preds.
            is_post = (
                len(preds) > 0
                and all(
                    problem.ops[p].stream != COMPUTE_STREAM or p in comm_adjacent
                    for p in preds
                )
                and all(s in comm_adjacent for s in succs)
            )
            if is_prep or is_post:
                comm_adjacent.add(op_id)
                changed = True

    # Remap compute ops: assign solver time slots in original graph order.
    # Solver time slots are monotonically increasing (compute stream is
    # serial), so mapping them to original-order compute ops preserves
    # relative order while keeping the same time scale as comm/wait keys.
    # Comm-adjacent ops are excluded: they use solver start_times directly
    # so they can move with their associated comms.
    compute_entries: list[tuple[fx.Node, int]] = []
    for node in graph.nodes:
        if node in node_to_op:
            op_id = node_to_op[node]
            if (
                op_to_node.get(op_id) is node
                and problem.ops[op_id].stream == COMPUTE_STREAM
                and op_id not in comm_adjacent
            ):
                compute_entries.append((node, op_id))

    solver_slots = sorted(schedule.start_times[op_id] for _, op_id in compute_entries)
    orig_sorted = sorted(compute_entries, key=lambda x: original_order[x[0]])
    compute_mapped_time: dict[fx.Node, float] = {}
    for i, (node, _) in enumerate(orig_sorted):
        compute_mapped_time[node] = solver_slots[i]

    sort_key: dict[fx.Node, tuple[float, int, float, int]] = {}
    for node in graph.nodes:
        if node.op == "placeholder":
            sort_key[node] = (float("-inf"), 0, 0.0, original_order[node])
        elif node.op == "output":
            sort_key[node] = (float("inf"), 0, 0.0, original_order[node])
        elif node in node_to_op:
            op_id = node_to_op[node]
            if op_to_node.get(op_id) is node:
                op = problem.ops[op_id]
                if op.stream != COMPUTE_STREAM:
                    # Comm start: dispatch time as primary, completion
                    # as tiebreaker so same-dispatch-time comms appear
                    # in the solver's intended comm stream order.
                    sort_key[node] = (
                        schedule.start_times[op_id],
                        0,
                        schedule.completion_times[op_id],
                        original_order[node],
                    )
                else:
                    if op_id in comm_adjacent:
                        # Comm-adjacent: use solver time to move with its comm
                        sort_key[node] = (
                            schedule.start_times[op_id],
                            1,
                            0.0,
                            original_order[node],
                        )
                    else:
                        sort_key[node] = (
                            compute_mapped_time[node],
                            1,
                            0.0,
                            original_order[node],
                        )
            else:
                # Wait node: defer after same-time ops so comm launches
                # and compute run before waits block the compute stream.
                if len(problem.successors(op_id)) == 0:
                    # Leaf comm (e.g. reduce-scatter in backward): no
                    # downstream compute needs this result.  Push to end.
                    sort_key[node] = (
                        schedule.makespan,
                        2,
                        0.0,
                        original_order[node],
                    )
                else:
                    sort_key[node] = (
                        schedule.completion_times[op_id],
                        2,
                        0.0,
                        original_order[node],
                    )
        else:
            # Node not in the problem (SKIP'd get_attr, etc.): keep near
            # original position using a small scaled value so it sorts
            # early but after placeholders.
            sort_key[node] = (
                original_order[node] * 1e-9,
                0,
                0.0,
                original_order[node],
            )

    # Priority-based Kahn's topological sort: respects all FX edges while
    # scheduling nodes in sort_key order when unconstrained.
    in_degree: dict[fx.Node, int] = {}
    for node in graph.nodes:
        in_degree[node] = len(node.all_input_nodes)

    heap: list[tuple[tuple[float, int, float, int], int, fx.Node]] = []
    for node in graph.nodes:
        if in_degree[node] == 0:
            heapq.heappush(heap, (sort_key[node], id(node), node))

    ordered: list[fx.Node] = []
    while heap:
        _, _, node = heapq.heappop(heap)
        ordered.append(node)
        for user in node.users:
            in_degree[user] -= 1
            if in_degree[user] == 0:
                heapq.heappush(heap, (sort_key[user], id(user), user))

    # Reorder the graph to match
    output_node = next(n for n in graph.nodes if n.op == "output")
    for node in ordered:
        if node.op not in ("placeholder", "output"):
            output_node.prepend(node)

    graph.lint()


def _compute_fx_base_peak(
    graph: fx.Graph,
    node_info: dict[fx.Node, NodeInfo],
) -> int:
    """Peak memory when FX nodes run in their original graph order."""
    fx_remaining = {node: len(node.users) for node in graph.nodes}
    current = 0
    peak = 0
    for node in graph.nodes:
        current += node_info[node].memory_bytes
        peak = max(peak, current)
        for inp in node.all_input_nodes:
            fx_remaining[inp] -= 1
            if fx_remaining[inp] == 0:
                current -= node_info[inp].memory_bytes
    return peak


def _greedy_reorder(
    graph: fx.Graph,
    classify: Callable[[fx.Node], NodeInfo],
    max_in_flight: int | None = 2,
    memory_headroom_fraction: float = 0.0,
) -> Schedule:
    """Unified greedy scheduler that directly produces FX node order.

    Unlike solve_greedy + apply_schedule, this operates on FX nodes
    directly via Kahn's topological sort with the same dispatch-aware
    timing model.  Memory-budgeted compute reordering replaces the
    comm-adjacent fixpoint: among ready compute ops, default to original
    graph order, but allow an ALAP-preferred candidate when memory stays
    within a configurable headroom.  This naturally handles comm-prep/post
    ops (tiny memory, always pass the budget check).
    """
    # Classify all nodes upfront
    node_info: dict[fx.Node, NodeInfo] = {}
    wait_to_start: dict[fx.Node, fx.Node] = {}
    for node in graph.nodes:
        info = classify(node)
        node_info[node] = info
        if info.kind == NodeKind.COMM_WAIT:
            start_node = node.args[0]
            assert isinstance(start_node, fx.Node)
            wait_to_start[node] = start_node

    original_order: dict[fx.Node, int] = {node: i for i, node in enumerate(graph.nodes)}

    # Compute ALAP on abstract problem for priority
    problem, op_to_node, node_to_op = build_overlap_problem(graph, classify)
    alap = compute_alap(problem)
    alap_priority: dict[fx.Node, float] = {}
    for node in graph.nodes:
        if node in node_to_op:
            alap_priority[node] = alap[node_to_op[node]]
        else:
            alap_priority[node] = float("inf")

    # In-flight tracking: for each comm start, count non-output FX users
    # of its wait node (abstract successors in the dissolved-wait model)
    abstract_remaining: dict[fx.Node, int] = {}
    for wait, start in wait_to_start.items():
        count = sum(1 for u in wait.users if u.op != "output")
        abstract_remaining.setdefault(start, 0)
        abstract_remaining[start] += count

    # Memory budget via dry-run in original order
    base_peak = _compute_fx_base_peak(graph, node_info)
    memory_budget = base_peak * (1.0 + memory_headroom_fraction)

    # Scheduling state
    compute_time: float = 0.0
    comm_avail: dict[str, float] = defaultdict(float)
    in_flight: dict[str, int] = defaultdict(int)
    current_memory: int = 0
    completion: dict[fx.Node, float] = {}
    start_times_fx: dict[fx.Node, float] = {}

    # FX liveness tracking for memory
    fx_remaining: dict[fx.Node, int] = {node: len(node.users) for node in graph.nodes}

    # Kahn's algorithm state
    in_degree: dict[fx.Node, int] = {
        node: len(node.all_input_nodes) for node in graph.nodes
    }
    ready: set[fx.Node] = {node for node in graph.nodes if in_degree[node] == 0}
    ordered: list[fx.Node] = []

    def _pred_done(node: fx.Node) -> float:
        return max(
            (completion[inp] for inp in node.all_input_nodes if inp in completion),
            default=0.0,
        )

    def _net_memory_delta(node: fx.Node) -> int:
        delta = node_info[node].memory_bytes
        for inp in node.all_input_nodes:
            if fx_remaining[inp] == 1:
                delta -= node_info[inp].memory_bytes
        return delta

    def _emit(node: fx.Node) -> None:
        nonlocal current_memory, compute_time
        ordered.append(node)
        ready.discard(node)
        info = node_info[node]

        if info.kind == NodeKind.SKIP:
            completion[node] = compute_time
        elif info.kind == NodeKind.COMM_START:
            pred_done = _pred_done(node)
            dispatch = max(compute_time, pred_done)
            comm_start = max(dispatch, comm_avail[info.stream])
            comm_end = comm_start + info.duration_ms
            compute_time = dispatch + LAUNCH_OVERHEAD
            comm_avail[info.stream] = comm_end
            start_times_fx[node] = dispatch
            completion[node] = comm_end
            if abstract_remaining.get(node, 0) > 0:
                in_flight[info.stream] += 1
        elif info.kind == NodeKind.COMM_WAIT:
            start = wait_to_start[node]
            comm_done = completion[start]
            compute_time = max(compute_time, comm_done)
            start_times_fx[node] = compute_time
            completion[node] = compute_time
        else:  # COMPUTE, OTHER
            pred_done = _pred_done(node)
            start = max(compute_time, pred_done)
            compute_time = start + info.duration_ms
            start_times_fx[node] = start
            completion[node] = start + info.duration_ms

        # Memory tracking
        current_memory += info.memory_bytes
        for inp in node.all_input_nodes:
            fx_remaining[inp] -= 1
            if fx_remaining[inp] == 0:
                current_memory -= node_info[inp].memory_bytes

        # In-flight tracking: decrement when a consumer of a wait is emitted
        if node.op != "output":
            for inp in node.all_input_nodes:
                if node_info[inp].kind == NodeKind.COMM_WAIT:
                    start = wait_to_start[inp]
                    if abstract_remaining.get(start, 0) > 0:
                        abstract_remaining[start] -= 1
                        if abstract_remaining[start] == 0:
                            in_flight[node_info[start].stream] -= 1

        # Update Kahn's in-degree
        for user in node.users:
            in_degree[user] -= 1
            if in_degree[user] == 0:
                ready.add(user)

    # Main scheduling loop
    while ready:
        candidates: list[tuple[tuple[float, int, float], fx.Node]] = []

        # Skip nodes (placeholders, get_attr, etc.)
        for node in ready:
            info = node_info[node]
            if info.kind == NodeKind.SKIP:
                if node.op == "output":
                    candidates.append(((float("inf"), 3, 0.0), node))
                else:
                    candidates.append(
                        ((float("-inf"), -1, float(original_order[node])), node)
                    )

        # Comm starts
        for node in ready:
            info = node_info[node]
            if info.kind != NodeKind.COMM_START:
                continue
            if (
                max_in_flight is not None
                and abstract_remaining.get(node, 0) > 0
                and in_flight[info.stream] >= max_in_flight
            ):
                continue
            pred_done = _pred_done(node)
            dispatch = max(compute_time, pred_done)
            candidates.append(((dispatch, 0, alap_priority[node]), node))

        # Wait nodes
        for node in ready:
            info = node_info[node]
            if info.kind != NodeKind.COMM_WAIT:
                continue
            start = wait_to_start[node]
            if abstract_remaining.get(start, 0) == 0:
                key_time = float("inf")
            else:
                key_time = completion[start]
            candidates.append(((key_time, 2, 0.0), node))

        # Compute nodes: pick one candidate via default/ALAP selection
        compute_ready = [
            n for n in ready if node_info[n].kind in (NodeKind.COMPUTE, NodeKind.OTHER)
        ]
        if compute_ready:
            default = min(compute_ready, key=lambda n: original_order[n])
            alap_best = min(
                compute_ready,
                key=lambda n: (alap_priority[n], original_order[n]),
            )
            if alap_best != default:
                delta = _net_memory_delta(alap_best)
                if current_memory + delta <= memory_budget:
                    chosen = alap_best
                else:
                    chosen = default
            else:
                chosen = default
            pred_done = _pred_done(chosen)
            start = max(compute_time, pred_done)
            candidates.append(((start, 1, alap_priority[chosen]), chosen))

        if not candidates:
            # All comms blocked by in-flight limit, no compute/waits ready.
            # Force any ready node (shouldn't normally happen).
            best = min(ready, key=lambda n: alap_priority.get(n, float("inf")))
        else:
            _, best = min(candidates)
        _emit(best)

    # Reorder the FX graph to match
    output_node = next(n for n in graph.nodes if n.op == "output")
    for node in ordered:
        if node.op not in ("placeholder", "output"):
            output_node.prepend(node)
    graph.lint()

    # Build Schedule for return
    sched_start: dict[int, float] = {}
    sched_completion: dict[int, float] = {}
    for node in ordered:
        if node in node_to_op:
            op_id = node_to_op[node]
            if op_to_node.get(op_id) is node:
                sched_start[op_id] = start_times_fx.get(node, 0.0)
                sched_completion[op_id] = completion.get(node, 0.0)
    makespan = max(completion.values()) if completion else 0.0
    return Schedule(
        start_times=sched_start,
        completion_times=sched_completion,
        makespan=makespan,
    )


def reorder_for_overlap(
    graph: fx.Graph,
    classify: Callable[[fx.Node], NodeInfo],
    solver: str = "greedy",
    memory_budget_bytes: int | None = None,
    max_in_flight: int | None = 2,
    memory_headroom_fraction: float = 0.0,
) -> Schedule:
    """
    Full pipeline: build problem from FX graph, solve, and reorder in place.

    Returns the Schedule for inspection.
    """
    if solver == "greedy":
        return _greedy_reorder(graph, classify, max_in_flight, memory_headroom_fraction)
    # ILP path: keep existing build_problem -> solve_ilp -> apply_schedule
    problem, op_to_node, node_to_op = build_overlap_problem(graph, classify)
    schedule = solve_ilp(problem)
    apply_schedule(graph, schedule, problem, op_to_node, node_to_op)
    return schedule
