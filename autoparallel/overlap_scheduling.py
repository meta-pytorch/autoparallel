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
    return Schedule(start_times=start_times, makespan=pulp.value(T))


# ---------------------------------------------------------------------------
# Greedy solver
# ---------------------------------------------------------------------------


def solve_greedy(
    problem: OverlapProblem,
    memory_budget_bytes: int | None = None,
) -> Schedule:
    """
    ASAP greedy scheduler with ALAP-priority tie-breaking.

    Simulates concurrent stream execution.  At each step picks the op
    that can start earliest, breaking ties by urgency (lowest ALAP = most
    urgent).  Optionally respects a memory budget.
    """
    alap = compute_alap(problem)
    n = problem.n

    in_deg = [0] * n
    for i in range(n):
        for j in problem.successors(i):
            in_deg[j] += 1

    ready: set[int] = {i for i in range(n) if in_deg[i] == 0}
    completion: dict[int, float] = {}
    stream_avail: dict[str, float] = defaultdict(float)

    # Memory tracking: each op's output is live until its last consumer runs
    current_memory = 0
    remaining_consumers = [len(problem.successors(i)) for i in range(n)]

    while ready:
        best_op: int | None = None
        best_key = (float("inf"), float("inf"))

        for op_id in ready:
            op = problem.ops[op_id]
            pred_done = max(
                (completion[p] for p in problem.predecessors(op_id)),
                default=0.0,
            )
            start = max(stream_avail[op.stream], pred_done)

            if memory_budget_bytes is not None:
                if current_memory + op.memory_bytes > memory_budget_bytes:
                    continue

            key = (start, alap[op_id])
            if key < best_key:
                best_key = key
                best_op = op_id

        # If everything exceeds memory, force the most urgent op
        if best_op is None:
            best_op = min(ready, key=lambda i: alap[i])

        op = problem.ops[best_op]
        pred_done = max(
            (completion[p] for p in problem.predecessors(best_op)),
            default=0.0,
        )
        start = max(stream_avail[op.stream], pred_done)
        end = start + op.duration_ms

        completion[best_op] = end
        stream_avail[op.stream] = end
        ready.remove(best_op)

        current_memory += op.memory_bytes
        for p in problem.predecessors(best_op):
            remaining_consumers[p] -= 1
            if remaining_consumers[p] == 0:
                current_memory -= problem.ops[p].memory_bytes

        for s in problem.successors(best_op):
            in_deg[s] -= 1
            if in_deg[s] == 0:
                ready.add(s)

    start_times = {i: completion[i] - problem.ops[i].duration_ms for i in range(n)}
    return Schedule(start_times=start_times, makespan=max(completion.values()))


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
    op_to_node: dict[int, fx.Node],
    node_to_op: dict[fx.Node, int],
) -> None:
    """
    Reorder FX graph nodes to match the solved schedule.

    Each FX node is assigned the start time of its corresponding op.
    Wait nodes are placed at the completion time of their comm op
    (= start_time + duration), so they appear just before the first
    consumer that needs the result.
    """
    node_times: dict[fx.Node, float] = {}
    for node in graph.nodes:
        if node not in node_to_op:
            # placeholders get -inf, output gets +inf
            if node.op == "placeholder":
                node_times[node] = float("-inf")
            elif node.op == "output":
                node_times[node] = float("inf")
            continue

        op_id = node_to_op[node]
        op = None
        # Find the op — node_to_op maps wait nodes to their start's op
        for i, o in enumerate(schedule.start_times):
            if o == op_id:
                break
        # Check if this is a wait node (shares op_id with start, but is a
        # different FX node)
        if op_to_node.get(op_id) is not node and node.op != "placeholder":
            # This is a wait node: place it at the comm op's completion time
            op = schedule.start_times[op_id]
            # TODO: look up the actual Op to get duration; using a sentinel
            # that sorts after the start but before consumers
            node_times[node] = op + 0.0001
        else:
            node_times[node] = schedule.start_times[op_id]

    # Reorder: move nodes to match schedule order
    output_node = next(n for n in graph.nodes if n.op == "output")
    sorted_nodes = sorted(
        (n for n in graph.nodes if n.op not in ("placeholder", "output")),
        key=lambda n: node_times.get(n, 0.0),
    )
    for node in sorted_nodes:
        output_node.prepend(node)

    graph.lint()
