# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Collective costs from measured timings, e.g. nccl-tests sweeps of a cluster.

A table maps (collective, n_nodes, ppn) to a curve of (bytes, time_us) points.
Collectives use the names and byte conventions of collective_comm_cost():
"allgather" (output bytes), "reduce_scatter" (input bytes), "allreduce", and
"all_to_all" (per-rank buffer bytes), which match the size column of
nccl-tests. The keys are only meaningful for the gpus_per_node the table was
measured with, since it fixes how mesh dimensions map to (n_nodes, ppn).

Lookups interpolate linearly in bytes, hold the smallest measured size below
it, and extend the last segment above the largest. Multi-node counts without
a curve are interpolated linearly in n_nodes between measured counts, or
extrapolated from the two largest ones with the slope clamped at >= 0.
A single rank (n_nodes=1, ppn=1) communicates nothing and costs zero.
Anything else, including untabulated single-node keys, has no cost and raises.
"""

from __future__ import annotations

import bisect
import collections
import json
import re
import statistics
from dataclasses import dataclass, field

COLLECTIVES = ("allgather", "reduce_scatter", "allreduce", "all_to_all")

_SCHEMA_VERSION = 1

_NCCL_TESTS_BINARIES = {
    "all_gather_perf": "allgather",
    "reduce_scatter_perf": "reduce_scatter",
    "all_reduce_perf": "allreduce",
    "alltoall_perf": "all_to_all",
}

CurveKey = tuple[str, int, int]


@dataclass
class MeasuredCommTable:
    gpus_per_node: int
    # (collective, n_nodes, ppn) -> (ascending sizes in bytes, times in us).
    # Raw measurements; lookups use the envelope in _monotone_times.
    curves: dict[CurveKey, tuple[tuple[int, ...], tuple[float, ...]]]
    provenance: dict[str, str] = field(default_factory=dict)
    _monotone_times: dict[CurveKey, tuple[float, ...]] = field(init=False, repr=False)
    # (collective, ppn) -> ascending measured node counts above one
    _multi_node_counts: dict[tuple[str, int], list[int]] = field(init=False, repr=False)

    def __post_init__(self):
        self._monotone_times = {}
        self._multi_node_counts = collections.defaultdict(list)
        for key, (sizes, times) in self.curves.items():
            collective, n_nodes, ppn = key
            if collective not in COLLECTIVES:
                raise ValueError(f"Unknown collective {collective!r} in {key}")
            if len(sizes) < 2 or len(sizes) != len(times):
                raise ValueError(f"{key} needs at least two (bytes, time_us) points")
            if any(s1 <= s0 for s0, s1 in zip(sizes, sizes[1:])):
                raise ValueError(f"{key} sizes must be strictly increasing")
            # Noise only adds time, so take the lower non-decreasing envelope:
            # a slow outlier at a small size must not inflate larger sizes.
            envelope = list(times)
            for i in reversed(range(len(envelope) - 1)):
                envelope[i] = min(envelope[i], envelope[i + 1])
            self._monotone_times[key] = tuple(envelope)
            if n_nodes > 1:
                bisect.insort(self._multi_node_counts[(collective, ppn)], n_nodes)

    def time_us(self, collective: str, n_nodes: int, ppn: int, n_bytes: int) -> float:
        if n_nodes == ppn == 1:
            return 0.0
        nodes = self._bracketing_nodes(collective, n_nodes, ppn)
        if nodes is None:
            raise KeyError(
                f"No measured {collective} timings cover n_nodes={n_nodes}, "
                f"ppn={ppn} (gpus_per_node={self.gpus_per_node})"
            )
        lo, hi = nodes
        t_hi = self._curve_time((collective, hi, ppn), n_bytes)
        if lo == hi:
            return t_hi
        t_lo = self._curve_time((collective, lo, ppn), n_bytes)
        slope = (t_hi - t_lo) / (hi - lo)
        if n_nodes > hi:
            slope = max(slope, 0.0)
        return t_hi + (n_nodes - hi) * slope

    def coverage(self, n_nodes: int, ppn: int) -> str:
        """Label for how every collective is priced at (n_nodes, ppn).

        One of "measured", "interpolated", "extrapolated", "missing" when
        some collective has no cost, or "no_communication" for a single rank.
        """
        if n_nodes == ppn == 1:
            return "no_communication"
        brackets = []
        for collective in COLLECTIVES:
            nodes = self._bracketing_nodes(collective, n_nodes, ppn)
            if nodes is None:
                return "missing"
            brackets.append(nodes)
        if any(hi < n_nodes for _, hi in brackets):
            return "extrapolated"
        if any(lo != hi for lo, hi in brackets):
            return "interpolated"
        return "measured"

    def _bracketing_nodes(
        self, collective: str, n_nodes: int, ppn: int
    ) -> tuple[int, int] | None:
        if (collective, n_nodes, ppn) in self.curves:
            return n_nodes, n_nodes
        counts = self._multi_node_counts.get((collective, ppn), [])
        if n_nodes == 1 or len(counts) < 2 or n_nodes < counts[0]:
            return None
        i = min(bisect.bisect(counts, n_nodes), len(counts) - 1)
        return counts[i - 1], counts[i]

    def _curve_time(self, key: CurveKey, n_bytes: int) -> float:
        sizes, times = self.curves[key][0], self._monotone_times[key]
        if n_bytes <= sizes[0]:
            return times[0]
        i = min(bisect.bisect(sizes, n_bytes), len(sizes) - 1)
        slope = (times[i] - times[i - 1]) / (sizes[i] - sizes[i - 1])
        return times[i] + (n_bytes - sizes[i]) * slope

    def save(self, path: str) -> None:
        # One curve per line keeps the file reviewable and diffable.
        curves = ",\n    ".join(
            json.dumps(
                {
                    "collective": collective,
                    "n_nodes": n_nodes,
                    "ppn": ppn,
                    "bytes": list(sizes),
                    "time_us": list(times),
                }
            )
            for (collective, n_nodes, ppn), (sizes, times) in sorted(
                self.curves.items()
            )
        )
        with open(path, "w") as f:
            f.write(
                f'{{\n  "schema_version": {_SCHEMA_VERSION},\n'
                f'  "gpus_per_node": {self.gpus_per_node},\n'
                f'  "provenance": {json.dumps(self.provenance)},\n'
                f'  "curves": [\n    {curves}\n  ]\n}}\n'
            )

    @classmethod
    def load(cls, path: str) -> MeasuredCommTable:
        with open(path) as f:
            data = json.load(f)
        if data.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError(
                f"{path}: unsupported schema_version {data.get('schema_version')}"
            )
        curves = {
            (c["collective"], c["n_nodes"], c["ppn"]): (
                tuple(c["bytes"]),
                tuple(c["time_us"]),
            )
            for c in data["curves"]
        }
        return cls(data["gpus_per_node"], curves, data.get("provenance", {}))

    @classmethod
    def from_nccl_tests(
        cls,
        paths: list[str],
        gpus_per_node: int,
        provenance: dict[str, str] | None = None,
    ) -> MeasuredCommTable:
        """Build a table from the stdout of unsplit nccl-tests runs.

        Each file is one run of all_gather_perf, reduce_scatter_perf,
        all_reduce_perf or alltoall_perf. The collective, node count and ranks
        per node come from its header; times are the out-of-place column.
        Repeated runs of a key, e.g. on different allocations, are combined
        by taking the median time at each size.
        """
        runs = collections.defaultdict(list)
        for path in paths:
            key, curve = _parse_nccl_tests(path)
            runs[key].append(curve)
        curves = {}
        for key, key_runs in runs.items():
            sizes = key_runs[0][0]
            if any(run_sizes != sizes for run_sizes, _ in key_runs):
                raise ValueError(f"Runs of {key} measured different sizes")
            times = zip(*(run_times for _, run_times in key_runs))
            curves[key] = (sizes, tuple(statistics.median(t) for t in times))
        return cls(gpus_per_node, curves, provenance or {})


def _parse_nccl_tests(
    path: str,
) -> tuple[CurveKey, tuple[tuple[int, ...], tuple[float, ...]]]:
    collective = None
    hosts: list[str] = []
    points: dict[int, float] = {}
    with open(path) as f:
        for line in f:
            if line.startswith("# Collective test starting:"):
                binary = re.match(r"\w+?_perf", line.split(":")[1].strip())
                if binary is None or binary.group(0) not in _NCCL_TESTS_BINARIES:
                    raise ValueError(f"{path}: unsupported test in {line.strip()!r}")
                collective = _NCCL_TESTS_BINARIES[binary.group(0)]
            elif rank := re.match(r"#\s+Rank\s+\d+ .* on (\S+) device", line):
                hosts.append(rank.group(1))
            elif re.match(r"\s+\d", line):
                fields = line.split()
                if fields[8] not in ("0", "N/A"):
                    raise ValueError(f"{path}: validation errors in {line.strip()!r}")
                # Sizes too small to split across ranks are reported as 0.
                if int(fields[0]) > 0:
                    points[int(fields[0])] = float(fields[5])
    if collective is None or not hosts or not points:
        raise ValueError(f"{path}: not an nccl-tests output")
    ranks_per_host = collections.Counter(hosts).values()
    if len(set(ranks_per_host)) != 1:
        raise ValueError(f"{path}: ranks are not spread evenly across hosts")
    key = (collective, len(ranks_per_host), len(hosts) // len(ranks_per_host))
    sizes = tuple(sorted(points))
    return key, (sizes, tuple(points[s] for s in sizes))
