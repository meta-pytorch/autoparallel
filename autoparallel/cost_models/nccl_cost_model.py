# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
NCCL-based communication cost model for AutoParallel.

Ports the cost model from NCCL's tuning.cc to Python, covering 3 protocols
(LL, LL128, Simple), 6 algorithms (Ring, Tree, CollNet Direct/Chain, NVLS,
NVLS Tree) with per-architecture tuning constants and empirical correction
factors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.distributed.tensor import DeviceMesh


class GpuArch(Enum):
    AMPERE = 0  # A100
    HOPPER = 1  # H100/H200
    BLACKWELL = 2  # B200/GB200


class NCCLFunc(Enum):
    ALLGATHER = 0
    REDUCESCATTER = 1
    ALLREDUCE = 2


class NCCLAlgo(Enum):
    TREE = 0
    RING = 1
    COLLNET_DIRECT = 2
    COLLNET_CHAIN = 3
    NVLS = 4
    NVLS_TREE = 5


class NCCLProto(Enum):
    LL = 0
    LL128 = 1
    SIMPLE = 2


_DEFAULT_CHANNELS = {
    GpuArch.AMPERE: 12,
    GpuArch.HOPPER: 16,
    GpuArch.BLACKWELL: 16,
}


@dataclass
class NCCLTopoConfig:
    arch: GpuArch
    num_nodes: int
    gpus_per_node: int
    # Total per-GPU intra-node BW in GB/s, derived from NCCL's per-link NVLink
    # constants × default channel count (A100=87.7, H100=320, B200=640).
    bw_intra: float
    # Total per-GPU inter-node BW in GB/s (e.g. 25 for 200Gbps, 50 for 400Gbps)
    bw_inter: float
    # Number of NCCL channels (None = arch default: Ampere=12, Hopper/Blackwell=16)
    num_channels: int | None = None
    has_nvswitch: bool = False  # Enables NVLS (Hopper+ only)
    has_collnet: bool = False  # Enables CollNet Direct/Chain (SHARP)
    # Additional network latency beyond base hw latency (us)
    net_latency: float = 0.0


@dataclass
class MeshDimTopo:
    n_ranks: int  # devices in this mesh dimension's communicator
    n_nodes: int  # nodes spanned
    ppn: int  # ranks per node (n_ranks / n_nodes)
    bw_intra: float  # per-channel intra-node BW (GB/s)
    bw_inter: float  # per-channel inter-node BW (GB/s)
    n_channels: int


# ---------------------------------------------------------------------------
# Tuning constants ported from NCCL tuning.cc
# All per-protocol tuples are ordered (LL, LL128, SIMPLE).
# ---------------------------------------------------------------------------

# baseLatencies[algo][proto] (lines 143-148)
_BASE_LATENCIES = {
    NCCLAlgo.TREE: (6.8, 14.0, 8.4),
    NCCLAlgo.RING: (6.6, 14.0, 8.4),
    NCCLAlgo.COLLNET_DIRECT: (0.0, 0.0, 0.0),
    NCCLAlgo.COLLNET_CHAIN: (0.0, 0.0, 0.0),
    NCCLAlgo.NVLS: (0.0, 0.0, 0.0),
    NCCLAlgo.NVLS_TREE: (0.0, 0.0, 0.0),
}

# hwLatencies[hw][algo][proto] (lines 151-168)
_HW_LAT_NVLINK = {
    NCCLAlgo.TREE: (0.6, 1.25, 4.0),
    NCCLAlgo.RING: (0.6, 1.9, 3.4),
    NCCLAlgo.COLLNET_DIRECT: (0.0, 0.0, 3.7),
    NCCLAlgo.COLLNET_CHAIN: (0.0, 0.0, 2.8),
    NCCLAlgo.NVLS: (0.0, 0.0, 25.0),
    NCCLAlgo.NVLS_TREE: (0.0, 0.0, 25.0),
}
_HW_LAT_PCI = {
    NCCLAlgo.TREE: (1.0, 1.9, 4.0),
    NCCLAlgo.RING: (1.0, 2.5, 5.7),
    NCCLAlgo.COLLNET_DIRECT: (0.0, 0.0, 3.7),
    NCCLAlgo.COLLNET_CHAIN: (0.0, 0.0, 2.8),
    NCCLAlgo.NVLS: (0.0, 0.0, 0.0),
    NCCLAlgo.NVLS_TREE: (0.0, 0.0, 0.0),
}
_HW_LAT_NET = {
    NCCLAlgo.TREE: (5.0, 8.5, 14.0),
    NCCLAlgo.RING: (2.7, 4.0, 14.0),
    NCCLAlgo.COLLNET_DIRECT: (0.0, 0.0, 31.0),
    NCCLAlgo.COLLNET_CHAIN: (0.0, 0.0, 30.0),
    NCCLAlgo.NVLS: (0.0, 0.0, 18.0),
    NCCLAlgo.NVLS_TREE: (0.0, 0.0, 20.9),
}

# perChMaxTreeBws[arch][nodeBucket] (lines 188-193), nodeBucket: N1/N2/N4+
_PER_CH_MAX_TREE_BWS = {
    GpuArch.AMPERE: (24.0, 23.6, 17.8),
    GpuArch.HOPPER: (38.7, 41.4, 36.0),
    GpuArch.BLACKWELL: (70.0, 42.8, 24.0),
}

# perChMaxNVLSTreeBws[arch][nodeBucket] (lines 194-199)
_PER_CH_MAX_NVLS_TREE_BWS = {
    GpuArch.AMPERE: (24.0, 23.6, 17.8),
    GpuArch.HOPPER: (0.0, 57.7, 45.5),
    GpuArch.BLACKWELL: (0.0, 96.0, 43.8),
}

# llMaxBws[index1][nodeBucket] (lines 170-175)
# Single-node: indexed by GPU arch. Multi-node: NCCL switches to CPU vendor
# (Intel=row 0, AMD=row 1). We default to Intel for multi-node.
_LL_MAX_BWS = {
    GpuArch.AMPERE: (87.7, 39.0, 20.4),
    GpuArch.HOPPER: (141.0, 39.0, 20.4),
    GpuArch.BLACKWELL: (282.0, 39.0, 20.4),
}

# perChMaxRingLL128Bws[arch][nodeBucket] (lines 176-181)
_PER_CH_MAX_RING_LL128_BWS = {
    GpuArch.AMPERE: (20.0, 20.0, 20.0),
    GpuArch.HOPPER: (36.7, 36.7, 36.7),
    GpuArch.BLACKWELL: (40.0, 40.0, 40.0),
}

# perChMaxTreeLL128Bws[arch][nodeBucket] (lines 182-187)
_PER_CH_MAX_TREE_LL128_BWS = {
    GpuArch.AMPERE: (20.0, 20.0, 20.0),
    GpuArch.HOPPER: (36.7, 36.7, 29.0),
    GpuArch.BLACKWELL: (55.6, 31.67, 20.0),
}

# nvlsEfficiency[arch] (lines 135-140)
_NVLS_EFFICIENCY = {
    GpuArch.AMPERE: 0.0,
    GpuArch.HOPPER: 0.85,
    GpuArch.BLACKWELL: 0.74,
}

# treeCorrectionFactor[proto][24] (lines 581-585)
_TREE_CORRECTION_FACTOR = {
    NCCLProto.LL: (
        1.0,
        1.0,
        1.0,
        1.0,
        0.9,
        0.8,
        0.7,
        0.7,
        0.7,
        0.7,
        0.6,
        0.5,
        0.4,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ),
    NCCLProto.LL128: (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.9,
        0.8,
        0.8,
        0.8,
        0.7,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.8,
        0.9,
        0.9,
        0.9,
        0.9,
        1.0,
        1.0,
        1.0,
    ),
    NCCLProto.SIMPLE: (
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.8,
        0.7,
        0.6,
        0.6,
        0.5,
        0.5,
        0.5,
        0.5,
        0.6,
        0.7,
        0.8,
        0.7,
        0.7,
        0.8,
        0.9,
        0.9,
        0.9,
    ),
}


def _node_bucket(n_nodes: int) -> int:
    """Map node count to NCCL's index2: N1->0, N2->1, N4+->2."""
    if n_nodes <= 1:
        return 0
    if n_nodes <= 2:
        return 1
    return 2


def _get_n_channels(config: NCCLTopoConfig) -> int:
    if config.num_channels is not None:
        return config.num_channels
    return _DEFAULT_CHANNELS[config.arch]


def derive_mesh_dim_topo(
    config: NCCLTopoConfig,
    mesh_shape: tuple[int, ...],
    dim_idx: int,
) -> MeshDimTopo:
    """Derive per-mesh-dimension NCCL topology parameters."""
    dim_size = mesh_shape[dim_idx]
    inner_product = math.prod(mesh_shape[dim_idx + 1 :])
    ppn = max(1, min(config.gpus_per_node // inner_product, dim_size))
    n_nodes = dim_size // ppn
    n_channels = _get_n_channels(config)
    bw_intra = config.bw_intra / n_channels
    bw_inter = config.bw_inter / n_channels
    return MeshDimTopo(
        n_ranks=dim_size,
        n_nodes=n_nodes,
        ppn=ppn,
        bw_intra=bw_intra,
        bw_inter=bw_inter,
        n_channels=n_channels,
    )


# ---------------------------------------------------------------------------
# Bandwidth computation — port of tuning.cc lines 285-353
# ---------------------------------------------------------------------------


def _compute_algo_bw(
    func: NCCLFunc,
    algo: NCCLAlgo,
    proto: NCCLProto,
    topo: MeshDimTopo,
    config: NCCLTopoConfig,
) -> float:
    """Compute effective algorithm bandwidth in GB/s for a given algo+proto."""
    n_ranks = topo.n_ranks
    n_nodes = topo.n_nodes
    ppn = topo.ppn
    n_ch = topo.n_channels
    bw_intra = topo.bw_intra
    bw_inter = topo.bw_inter
    arch = config.arch

    if func == NCCLFunc.ALLREDUCE:
        nsteps = 2 * (n_ranks - 1)
    else:
        nsteps = n_ranks - 1

    node_bucket = _node_bucket(n_nodes)

    # NVLS/NVLS_TREE/CollNet only support Simple protocol
    if proto != NCCLProto.SIMPLE:
        if algo in (
            NCCLAlgo.NVLS,
            NCCLAlgo.NVLS_TREE,
            NCCLAlgo.COLLNET_DIRECT,
            NCCLAlgo.COLLNET_CHAIN,
        ):
            return 0.0

    # NVSwitch aggregates all ppn NICs for inter-node Ring/Tree traffic.
    if config.has_nvswitch and n_nodes > 1:
        bw_inter_agg = min(bw_intra, bw_inter * ppn)
    else:
        bw_inter_agg = bw_inter

    # --- Phase 1: compute raw bus BW per algorithm ---

    if algo == NCCLAlgo.RING:
        bw = bw_intra if n_nodes <= 2 else bw_inter_agg
        bus_bw = n_ch * bw

    elif algo == NCCLAlgo.TREE:
        if func != NCCLFunc.ALLREDUCE:
            return 0.0
        bw = bw_intra if n_nodes <= 2 else bw_inter_agg
        bus_bw = n_ch * bw

    elif algo == NCCLAlgo.NVLS:
        eff = _NVLS_EFFICIENCY[arch]
        if n_ch < 2:
            return 0.0
        # NVLS multi-node AG/RS requires collnet (NCCL lines 327-334)
        if (
            func in (NCCLFunc.ALLGATHER, NCCLFunc.REDUCESCATTER)
            and n_nodes > 1
            and not config.has_collnet
        ):
            return 0.0
        intra_bw = bw_intra * eff * (n_ch - 1) / n_ch
        if func == NCCLFunc.ALLREDUCE:
            intra_bw *= 2.0
        else:
            intra_bw *= (ppn - 1) / ppn
        if n_nodes > 1:
            bw = min(intra_bw, bw_inter)
        else:
            bw = intra_bw
        bus_bw = n_ch * bw

    elif algo == NCCLAlgo.NVLS_TREE:
        if func != NCCLFunc.ALLREDUCE:
            return 0.0
        eff = _NVLS_EFFICIENCY[arch]
        if n_ch < 2:
            return 0.0
        intra_bw = bw_intra * eff * (n_ch - 1) / n_ch
        intra_bw *= 2.0  # AllReduce pipelines two operations
        inter_bw = bw_inter_agg * (2 if n_nodes <= 2 else 1)
        per_ch_max = _PER_CH_MAX_NVLS_TREE_BWS[arch][node_bucket]
        bw = min(intra_bw, inter_bw, per_ch_max)
        bus_bw = n_ch * bw

    elif algo == NCCLAlgo.COLLNET_DIRECT:
        bw = bw_intra if n_nodes <= 2 else bw_inter_agg
        if func in (NCCLFunc.ALLGATHER, NCCLFunc.REDUCESCATTER):
            # AG/RS: no bus-to-algo conversion (NCCL line 347 skips it)
            bus_bw = ppn * min(bw_intra * n_ch, bw_inter * n_ch * 0.9)
            return bus_bw
        else:
            bus_bw = n_ch * bw
            factor = ppn / n_ch
            factor -= (factor - 1) / 2
            bus_bw /= factor
            if arch in (GpuArch.HOPPER, GpuArch.BLACKWELL):
                bus_bw *= 0.85

    elif algo == NCCLAlgo.COLLNET_CHAIN:
        if func != NCCLFunc.ALLREDUCE:
            return 0.0
        bw = bw_intra if n_nodes <= 2 else bw_inter_agg
        bus_bw = n_ch * bw

    else:
        return 0.0

    # --- Phase 2: protocol-specific BW adjustments (NCCL lines 305-313) ---

    ll_max_bw = _LL_MAX_BWS[arch][node_bucket]

    if algo == NCCLAlgo.RING and proto == NCCLProto.LL:
        bus_bw = min(ll_max_bw, bus_bw * 0.5)
    if algo == NCCLAlgo.RING and proto == NCCLProto.LL128:
        per_ch_max = _PER_CH_MAX_RING_LL128_BWS[arch][node_bucket]
        bus_bw = min(bus_bw * 0.92, n_ch * per_ch_max)
    if algo == NCCLAlgo.TREE and func == NCCLFunc.ALLREDUCE:
        per_ch_max_tree = _PER_CH_MAX_TREE_BWS[arch][node_bucket]
        bus_bw = min(bus_bw * 0.92, n_ch * per_ch_max_tree)
    if algo == NCCLAlgo.TREE and proto == NCCLProto.LL:
        bus_bw = min(bus_bw / 3.8, ll_max_bw)
    if algo == NCCLAlgo.TREE and proto == NCCLProto.LL128:
        per_ch_max_tree_ll128 = _PER_CH_MAX_TREE_LL128_BWS[arch][node_bucket]
        bus_bw = min(
            bus_bw * (7.0 / 9.0 if n_nodes == 1 else 120.0 / 128.0),
            n_ch * per_ch_max_tree_ll128,
        )

    # --- Phase 3: bus-to-algo conversion (NCCL lines 346-352) ---
    # For non-Ring AG/RS, no ratio applied (bus_bw IS algo_bw).
    if algo == NCCLAlgo.RING or func not in (
        NCCLFunc.ALLGATHER,
        NCCLFunc.REDUCESCATTER,
    ):
        if algo in (NCCLAlgo.RING, NCCLAlgo.NVLS, NCCLAlgo.NVLS_TREE):
            bus_bw *= n_ranks / nsteps
        else:
            bus_bw *= 0.5

    return bus_bw


# ---------------------------------------------------------------------------
# Latency computation — port of tuning.cc lines 354-402
# ---------------------------------------------------------------------------

# getNetOverhead() base value (1.0 us for Intel, the common default)
_NET_OVERHEAD_BASE = 1.0


# CE-path AllToAll bandwidth for NVSwitch-connected Hopper+ GPUs (GB/s).
# On these architectures, NCCL dispatches AllToAll to Copy Engines, achieving
# much higher throughput than SM-driven channels.
# H100: fitted from nccl-tests alltoall_perf on 8xH100 NVSwitch (>=32MB msgs).
# Blackwell: estimated proportionally from bw_intra ratio; needs profiling.
_A2A_CE_BW = {
    GpuArch.HOPPER: 380.0,
    GpuArch.BLACKWELL: 675.0,
}
_A2A_CE_LATENCY = 50.0  # us, measured from 0-byte AllToAll on H100

# NVSwitch empirical bandwidth (GB/s, algo_bw) and latency (us) for
# AG/RS/AR on Hopper+, measured from nccl-tests on H100 NVSwitch at
# 2/4/8 GPUs (1GB messages, out-of-place). AG and RS values are averaged
# since they are symmetric on NVSwitch. For intermediate GPU counts,
# linearly interpolated; clamped at the boundary measurements.
# Blackwell: scaled from Hopper by bw_intra ratio; needs profiling.
_NVSWITCH_BW_POINTS: dict[NCCLFunc, tuple[tuple[int, float], ...]] = {
    NCCLFunc.ALLGATHER: ((2, 537.0), (4, 434.0), (8, 394.0)),
    NCCLFunc.REDUCESCATTER: ((2, 537.0), (4, 434.0), (8, 394.0)),
    NCCLFunc.ALLREDUCE: ((2, 324.0), (4, 236.0), (8, 267.0)),
}
_NVSWITCH_LAT_POINTS: dict[NCCLFunc, tuple[tuple[int, float], ...]] = {
    NCCLFunc.ALLGATHER: ((2, 8.0), (4, 17.0), (8, 30.0)),
    NCCLFunc.REDUCESCATTER: ((2, 8.0), (4, 17.0), (8, 30.0)),
    NCCLFunc.ALLREDUCE: ((2, 10.0), (4, 15.0), (8, 31.0)),
}
_BLACKWELL_BW_SCALE = 640.0 / 320.0  # Blackwell/Hopper bw_intra ratio


def _interp_clamped(points: tuple[tuple[int, float], ...], x: int) -> float:
    """Linearly interpolate between data points, clamping at boundaries."""
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        if x <= x1:
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return points[-1][1]


# SM-driven AllToAll bandwidth correction for non-CE-path cases (older arch,
# no NVSwitch, or multi-node). P2P decomposition has worse pipelining
# (SLICESTEPS=1/CHUNKSTEPS=1) and more link contention than Ring collectives.
_A2A_BW_CORRECTION = 0.7


def _compute_algo_latency(
    func: NCCLFunc,
    algo: NCCLAlgo,
    proto: NCCLProto,
    topo: MeshDimTopo,
    config: NCCLTopoConfig,
) -> float:
    """Compute latency in microseconds for a given algo+proto."""
    n_ranks = topo.n_ranks
    n_nodes = topo.n_nodes
    ppn = topo.ppn
    p = proto.value

    base_lat = _BASE_LATENCIES[algo][p]
    # We assume NVLink intra-node
    intra_lat = _HW_LAT_NVLINK[algo][p]

    # ppn==1 case: use Tree NET latency for inter
    if ppn == 1:
        inter_lat_base = _HW_LAT_NET[NCCLAlgo.TREE][p]
    else:
        inter_lat_base = _HW_LAT_NET[algo][p]
    inter_lat = inter_lat_base + config.net_latency
    # Simple protocol doubles net_latency (flush extra latency, line 360)
    if proto == NCCLProto.SIMPLE:
        inter_lat += config.net_latency

    if func == NCCLFunc.ALLREDUCE:
        nsteps = 2 * (n_ranks - 1)
    else:
        nsteps = n_ranks - 1

    if algo == NCCLAlgo.RING:
        if n_nodes > 1:
            net_overhead = _NET_OVERHEAD_BASE
            if proto == NCCLProto.SIMPLE:
                net_overhead *= 3
        else:
            net_overhead = 0.0
        intra_lat = max(intra_lat, net_overhead)
        if n_nodes == 1:
            n_inter_steps = 0
        elif func == NCCLFunc.ALLREDUCE:
            n_inter_steps = 2 * (n_nodes - 1)
        else:
            n_inter_steps = n_nodes - 1
        lat = (
            base_lat + (nsteps - n_inter_steps) * intra_lat + n_inter_steps * inter_lat
        )

    elif algo == NCCLAlgo.TREE:
        lat = base_lat + 2 * (
            (n_ranks // n_nodes - 1) * intra_lat + _log2i(n_nodes) * inter_lat
        )

    elif algo == NCCLAlgo.COLLNET_DIRECT:
        lat = base_lat + 2 * (min(1, ppn - 1) * intra_lat + (ppn - 1) * 0.4) + inter_lat

    elif algo == NCCLAlgo.COLLNET_CHAIN:
        lat = base_lat + 2 * (ppn - 1) * intra_lat + inter_lat

    elif algo == NCCLAlgo.NVLS:
        # NVLS replaces base latency
        lat = intra_lat + (inter_lat if n_nodes > 1 else 0.0)

    elif algo == NCCLAlgo.NVLS_TREE:
        lat = base_lat + intra_lat + 2 * _log2i(n_nodes) * inter_lat

    else:
        lat = 0.0

    return lat


def _log2i(n: int) -> int:
    """Integer log2, matching NCCL's log2i."""
    if n <= 0:
        return 0
    return n.bit_length() - 1


# ---------------------------------------------------------------------------
# Time estimation and algorithm selection — port of ncclTopoGetAlgoTime
# ---------------------------------------------------------------------------


def _nccl_algo_time(
    func: NCCLFunc,
    algo: NCCLAlgo,
    proto: NCCLProto,
    n_bytes: int,
    topo: MeshDimTopo,
    config: NCCLTopoConfig,
) -> float:
    """Compute estimated time in microseconds for one algorithm+protocol.

    Returns float('inf') if the algorithm is disabled (zero bandwidth).
    """
    bw = _compute_algo_bw(func, algo, proto, topo, config)
    lat = _compute_algo_latency(func, algo, proto, topo, config)

    if bw <= 0:
        return float("inf")

    # Tree correction factor (NCCL line 595: logSize < 23 for Tree)
    log_size = _log2i(n_bytes >> 6)
    if algo == NCCLAlgo.TREE and func == NCCLFunc.ALLREDUCE and 0 <= log_size < 23:
        bw *= _TREE_CORRECTION_FACTOR[proto][log_size]

    # NVLS_Tree correction for Blackwell (line 596: logSize < 24)
    if (
        algo == NCCLAlgo.NVLS_TREE
        and func == NCCLFunc.ALLREDUCE
        and config.arch == GpuArch.BLACKWELL
        and 0 <= log_size < 24
    ):
        bw *= _TREE_CORRECTION_FACTOR[proto][log_size]

    # Ring plateau effect for multi-node Simple allreduce (lines 597-599)
    if (
        algo == NCCLAlgo.RING
        and proto == NCCLProto.SIMPLE
        and topo.n_nodes > 1
        and func == NCCLFunc.ALLREDUCE
        and n_bytes / (topo.n_channels * topo.n_ranks) >= 64
    ):
        lat *= 1.4

    return lat + n_bytes / (1000.0 * bw)


def _eligible_algos(
    func: NCCLFunc, config: NCCLTopoConfig, n_nodes: int
) -> list[NCCLAlgo]:
    """Return algorithms eligible for a given collective, respecting feature flags."""
    if func in (NCCLFunc.ALLGATHER, NCCLFunc.REDUCESCATTER):
        # AG/RS: Ring, NVLS, CollNet Direct
        algos = [NCCLAlgo.RING]
        if config.has_nvswitch and config.arch in (GpuArch.HOPPER, GpuArch.BLACKWELL):
            algos.append(NCCLAlgo.NVLS)
        if config.has_collnet:
            algos.append(NCCLAlgo.COLLNET_DIRECT)
    else:
        # AllReduce: Ring, Tree, CollNet Direct, CollNet Chain, NVLS, NVLS Tree
        algos = [NCCLAlgo.RING, NCCLAlgo.TREE]
        if config.has_collnet:
            algos.append(NCCLAlgo.COLLNET_DIRECT)
            algos.append(NCCLAlgo.COLLNET_CHAIN)
        if config.has_nvswitch and config.arch in (GpuArch.HOPPER, GpuArch.BLACKWELL):
            algos.append(NCCLAlgo.NVLS)
            if n_nodes > 1:
                algos.append(NCCLAlgo.NVLS_TREE)
    return algos


def _eligible_protos(algo: NCCLAlgo) -> list[NCCLProto]:
    """Return protocols eligible for a given algorithm."""
    if algo in (
        NCCLAlgo.NVLS,
        NCCLAlgo.NVLS_TREE,
        NCCLAlgo.COLLNET_DIRECT,
        NCCLAlgo.COLLNET_CHAIN,
    ):
        return [NCCLProto.SIMPLE]
    return [NCCLProto.LL, NCCLProto.LL128, NCCLProto.SIMPLE]


def nccl_collective_time(
    func: NCCLFunc,
    n_bytes: int,
    topo: MeshDimTopo,
    config: NCCLTopoConfig,
) -> float:
    """Pick the best algorithm+protocol and return estimated time in microseconds.

    For Hopper+ with NVSwitch on a single node, uses empirical bandwidth
    and latency fitted from nccl-tests at 2/4/8 GPUs, with linear
    interpolation for intermediate counts. Blackwell values are scaled
    from Hopper by the bw_intra ratio. The algo selection loop below is
    only used for non-NVSwitch or multi-node cases.
    """
    # NVSwitch empirical path for Hopper+ intra-node
    if (
        config.arch in (GpuArch.HOPPER, GpuArch.BLACKWELL)
        and config.has_nvswitch
        and topo.n_nodes == 1
    ):
        bw = _interp_clamped(_NVSWITCH_BW_POINTS[func], topo.n_ranks)
        if config.arch == GpuArch.BLACKWELL:
            bw *= _BLACKWELL_BW_SCALE
        lat = _interp_clamped(_NVSWITCH_LAT_POINTS[func], topo.n_ranks)
        return lat + n_bytes / (1000.0 * bw)

    algos = _eligible_algos(func, config, topo.n_nodes)
    best = float("inf")
    for algo in algos:
        for proto in _eligible_protos(algo):
            t = _nccl_algo_time(func, algo, proto, n_bytes, topo, config)
            if t < best:
                best = t
    return best


# ---------------------------------------------------------------------------
# Public API wrappers — all return time in microseconds
# ---------------------------------------------------------------------------


def nccl_allgather_cost(
    n_bytes: int, topo: MeshDimTopo, config: NCCLTopoConfig
) -> float:
    return nccl_collective_time(NCCLFunc.ALLGATHER, n_bytes, topo, config)


def nccl_allreduce_cost(
    n_bytes: int, topo: MeshDimTopo, config: NCCLTopoConfig
) -> float:
    return nccl_collective_time(NCCLFunc.ALLREDUCE, n_bytes, topo, config)


def nccl_reduce_scatter_cost(
    n_bytes: int, topo: MeshDimTopo, config: NCCLTopoConfig
) -> float:
    return nccl_collective_time(NCCLFunc.REDUCESCATTER, n_bytes, topo, config)


def nccl_all_to_all_cost(
    n_bytes: int, topo: MeshDimTopo, config: NCCLTopoConfig
) -> float:
    """Cost model for AllToAll using architecture-aware bandwidth estimation.

    NCCL has no collective-level cost model for AllToAll — it is excluded from
    the tuning loop (NCCL_NUM_FUNCTIONS=5 covers only BR/RD/AG/RS/AR). At
    runtime, AllToAll is either decomposed into nRanks P2P Send/Recv pairs or
    dispatched to the Copy Engine (CE) path on Hopper+ with CUDA 12.5+.

    Two paths:
      - CE path (Hopper+ with NVSwitch, intra-node): uses empirical per-arch
        bandwidth fitted from nccl-tests alltoall_perf. CE bypasses SM-driven
        channels and achieves much higher throughput (~380 GB/s on H100 vs.
        225 GB/s SM-driven).
      - SM-driven fallback (older arch, no NVSwitch, or multi-node): reuses
        Ring latency/bandwidth constants with nsteps=nRanks and bus-to-algo
        ratio=1.0, discounted by _A2A_BW_CORRECTION for worse pipelining and
        link contention.

    TODO: potential improvements, roughly in priority order:
      1. Profile AllToAll on Blackwell NVSwitch hardware to replace the
         estimated _A2A_CE_BW[BLACKWELL] with a measured value.
      2. Profile multi-node AllToAll to validate or replace _A2A_BW_CORRECTION.
      3. Fit a size-dependent discount curve to bus_bw for the SM-driven path
         (similar to NCCL's treeCorrectionFactor).
      4. LL AllToAll for small messages: separate latency-dominated model
         below some size threshold. Unlikely to matter for sharding decisions
         which typically involve large tensors.
    """
    n_ranks = topo.n_ranks
    n_nodes = topo.n_nodes
    _s = NCCLProto.SIMPLE.value

    # CE path: Hopper+ with NVSwitch, intra-node AllToAll
    if (
        config.arch in (GpuArch.HOPPER, GpuArch.BLACKWELL)
        and config.has_nvswitch
        and n_nodes == 1
    ):
        bus_bw = _A2A_CE_BW[config.arch]
        return _A2A_CE_LATENCY + n_bytes / (1000.0 * bus_bw)

    # SM-driven fallback: Ring-style, ratio = 1.0 (no bus-to-algo boost),
    # discounted by correction factor for pipelining / contention.
    bw = topo.bw_intra if n_nodes <= 2 else topo.bw_inter
    bus_bw = topo.n_channels * bw * _A2A_BW_CORRECTION

    if bus_bw <= 0:
        return float("inf")

    # Latency: Ring-style with nsteps = nRanks, Simple protocol
    nsteps = n_ranks
    base_lat = _BASE_LATENCIES[NCCLAlgo.RING][_s]
    intra_lat = _HW_LAT_NVLINK[NCCLAlgo.RING][_s]

    if topo.ppn == 1:
        inter_lat_base = _HW_LAT_NET[NCCLAlgo.TREE][_s]
    else:
        inter_lat_base = _HW_LAT_NET[NCCLAlgo.RING][_s]
    inter_lat = inter_lat_base + config.net_latency
    inter_lat += config.net_latency  # Simple protocol doubles net_latency

    net_overhead = _NET_OVERHEAD_BASE * 3 if n_nodes > 1 else 0.0
    intra_lat = max(intra_lat, net_overhead)

    n_inter_steps = 0 if n_nodes == 1 else n_nodes - 1
    lat = base_lat + (nsteps - n_inter_steps) * intra_lat + n_inter_steps * inter_lat

    return lat + n_bytes / (1000.0 * bus_bw)


# ---------------------------------------------------------------------------
# Default configs for common GPU architectures
# ---------------------------------------------------------------------------


def detect_nccl_topo_config(mesh: "DeviceMesh") -> NCCLTopoConfig | None:
    """Auto-detect GPU architecture and build an NCCLTopoConfig from the mesh.

    Returns None if the GPU is unrecognized (caller falls back to PyTorch model).
    """
    import torch

    device_name = torch.cuda.get_device_name(0)
    total_gpus = mesh.size()
    gpus_per_node = torch.cuda.device_count()
    num_nodes = total_gpus // gpus_per_node

    if "A100" in device_name:
        return a100_topo_config(num_nodes=num_nodes, gpus_per_node=gpus_per_node)
    elif "H100" in device_name or "H200" in device_name:
        return h100_topo_config(num_nodes=num_nodes, gpus_per_node=gpus_per_node)
    elif "B200" in device_name or "GB200" in device_name:
        return gb200_topo_config(num_nodes=num_nodes, gpus_per_node=gpus_per_node)
    return None


def a100_topo_config(
    num_nodes: int = 1,
    gpus_per_node: int = 8,
    bw_inter: float = 25.0,
    **kwargs,
) -> NCCLTopoConfig:
    """DGX A100: 8x A100 per node, NVLink 87.7 GB/s intra-node."""
    return NCCLTopoConfig(
        arch=GpuArch.AMPERE,
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        bw_intra=87.7,
        bw_inter=bw_inter,
        **kwargs,
    )


def h100_topo_config(
    num_nodes: int = 1,
    gpus_per_node: int = 8,
    bw_inter: float = 50.0,
    has_nvswitch: bool = True,
    **kwargs,
) -> NCCLTopoConfig:
    """DGX H100: 8x H100 per node, NVSwitch 320 GB/s intra-node.

    bw_intra derived from NCCL's SM90 NVLink BW (20.0 GB/s per channel from
    the graph search speed array) × 16 default channels.
    """
    return NCCLTopoConfig(
        arch=GpuArch.HOPPER,
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        bw_intra=320.0,
        bw_inter=bw_inter,
        has_nvswitch=has_nvswitch,
        **kwargs,
    )


def gb200_topo_config(
    num_nodes: int = 1,
    gpus_per_node: int = 72,
    bw_inter: float = 50.0,
    has_nvswitch: bool = True,
    **kwargs,
) -> NCCLTopoConfig:
    """GB200 NVL72: 72x B200 per rack, NVSwitch 640 GB/s intra-node.

    bw_intra derived from NCCL's SM100 NVLink BW (40.0 GB/s per channel from
    the graph search speed array) × 16 default channels.
    """
    return NCCLTopoConfig(
        arch=GpuArch.BLACKWELL,
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        bw_intra=640.0,
        bw_inter=bw_inter,
        has_nvswitch=has_nvswitch,
        **kwargs,
    )
