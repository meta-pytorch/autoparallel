# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
NCCL-based communication cost model for AutoParallel.

Ports the Simple-protocol cost model from NCCL's tuning.cc to Python,
covering 6 algorithms (Ring, Tree, CollNet Direct/Chain, NVLS, NVLS Tree)
with per-architecture tuning constants and empirical correction factors.
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
    # Total per-GPU intra-node BW in GB/s (e.g. A100=87.7, H100=225, B200=400)
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
# Tuning constants ported from NCCL tuning.cc (Simple protocol column only)
# ---------------------------------------------------------------------------

# baseLatencies[algo][SIMPLE] (lines 143-148)
_BASE_LATENCIES = {
    NCCLAlgo.TREE: 8.4,
    NCCLAlgo.RING: 8.4,
    NCCLAlgo.COLLNET_DIRECT: 0.0,
    NCCLAlgo.COLLNET_CHAIN: 0.0,
    NCCLAlgo.NVLS: 0.0,
    NCCLAlgo.NVLS_TREE: 0.0,
}

# hwLatencies[hw][algo][SIMPLE] (lines 151-168)
# hw: NVLINK, PCI, NET
_HW_LAT_NVLINK = {
    NCCLAlgo.TREE: 4.0,
    NCCLAlgo.RING: 3.4,
    NCCLAlgo.COLLNET_DIRECT: 3.7,
    NCCLAlgo.COLLNET_CHAIN: 2.8,
    NCCLAlgo.NVLS: 25.0,
    NCCLAlgo.NVLS_TREE: 25.0,
}
_HW_LAT_PCI = {
    NCCLAlgo.TREE: 4.0,
    NCCLAlgo.RING: 5.7,
    NCCLAlgo.COLLNET_DIRECT: 3.7,
    NCCLAlgo.COLLNET_CHAIN: 2.8,
    NCCLAlgo.NVLS: 0.0,
    NCCLAlgo.NVLS_TREE: 0.0,
}
_HW_LAT_NET = {
    NCCLAlgo.TREE: 14.0,
    NCCLAlgo.RING: 14.0,
    NCCLAlgo.COLLNET_DIRECT: 31.0,
    NCCLAlgo.COLLNET_CHAIN: 30.0,
    NCCLAlgo.NVLS: 18.0,
    NCCLAlgo.NVLS_TREE: 20.9,
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

# nvlsEfficiency[arch] (lines 135-140)
_NVLS_EFFICIENCY = {
    GpuArch.AMPERE: 0.0,
    GpuArch.HOPPER: 0.85,
    GpuArch.BLACKWELL: 0.74,
}

# treeCorrectionFactor[SIMPLE][24] (line 584)
_TREE_CORRECTION_FACTOR_SIMPLE = (
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
)


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
    topo: MeshDimTopo,
    config: NCCLTopoConfig,
) -> float:
    """Compute effective algorithm bandwidth in GB/s for a given algo."""
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

    if algo == NCCLAlgo.RING:
        bw = bw_intra if n_nodes <= 2 else bw_inter
        bus_bw = n_ch * bw
        # Bus-to-algo conversion
        bus_bw *= n_ranks / nsteps

    elif algo == NCCLAlgo.TREE:
        # Tree is only used for AllReduce
        if func != NCCLFunc.ALLREDUCE:
            return 0.0
        bw = bw_intra if n_nodes <= 2 else bw_inter
        bus_bw = n_ch * bw
        per_ch_max = _PER_CH_MAX_TREE_BWS[arch][node_bucket]
        bus_bw = min(bus_bw * 0.92, n_ch * per_ch_max)
        bus_bw *= 0.5

    elif algo == NCCLAlgo.NVLS:
        eff = _NVLS_EFFICIENCY[arch]
        if n_ch < 2:
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
        # Bus-to-algo conversion for AG/RS
        if func != NCCLFunc.ALLREDUCE:
            bus_bw *= n_ranks / nsteps

    elif algo == NCCLAlgo.NVLS_TREE:
        # Only for AllReduce
        if func != NCCLFunc.ALLREDUCE:
            return 0.0
        eff = _NVLS_EFFICIENCY[arch]
        if n_ch < 2:
            return 0.0
        intra_bw = bw_intra * eff * (n_ch - 1) / n_ch
        intra_bw *= 2.0  # AllReduce pipelines two operations
        inter_bw = bw_inter * (2 if n_nodes <= 2 else 1)
        per_ch_max = _PER_CH_MAX_NVLS_TREE_BWS[arch][node_bucket]
        bw = min(intra_bw, inter_bw, per_ch_max)
        bus_bw = n_ch * bw
        bus_bw *= 0.5

    elif algo == NCCLAlgo.COLLNET_DIRECT:
        bw = bw_intra if n_nodes <= 2 else bw_inter
        if func in (NCCLFunc.ALLGATHER, NCCLFunc.REDUCESCATTER):
            # AG/RS: ppn * min(bwIntra, bwInter * 0.9), no bus-to-algo conversion
            bus_bw = ppn * min(bw_intra * n_ch, bw_inter * n_ch * 0.9)
            return bus_bw
        else:
            # AllReduce
            bus_bw = n_ch * bw
            factor = ppn / n_ch
            factor -= (factor - 1) / 2
            bus_bw /= factor
            if arch in (GpuArch.HOPPER, GpuArch.BLACKWELL):
                bus_bw *= 0.85
            bus_bw *= 0.5

    elif algo == NCCLAlgo.COLLNET_CHAIN:
        # Only Simple protocol, only for AllReduce
        if func != NCCLFunc.ALLREDUCE:
            return 0.0
        bw = bw_intra if n_nodes <= 2 else bw_inter
        bus_bw = n_ch * bw
        bus_bw *= 0.5

    else:
        return 0.0

    return bus_bw


# ---------------------------------------------------------------------------
# Latency computation — port of tuning.cc lines 354-402
# ---------------------------------------------------------------------------

# getNetOverhead() * 3 for Simple protocol when nNodes > 1
# We use 1.0 us base (Intel default)
_NET_OVERHEAD_SIMPLE = 1.0 * 3


def _compute_algo_latency(
    func: NCCLFunc,
    algo: NCCLAlgo,
    topo: MeshDimTopo,
    config: NCCLTopoConfig,
) -> float:
    """Compute latency in microseconds for a given algo."""
    n_ranks = topo.n_ranks
    n_nodes = topo.n_nodes
    ppn = topo.ppn

    base_lat = _BASE_LATENCIES[algo]
    # We assume NVLink intra-node
    intra_lat = _HW_LAT_NVLINK[algo]

    # ppn==1 case: use Tree NET latency for inter
    if ppn == 1:
        inter_lat_base = _HW_LAT_NET[NCCLAlgo.TREE]
    else:
        inter_lat_base = _HW_LAT_NET[algo]
    inter_lat = inter_lat_base + config.net_latency
    # Simple protocol doubles net_latency (flush extra latency, line 360)
    inter_lat += config.net_latency

    if func == NCCLFunc.ALLREDUCE:
        nsteps = 2 * (n_ranks - 1)
    else:
        nsteps = n_ranks - 1

    if algo == NCCLAlgo.RING:
        net_overhead = _NET_OVERHEAD_SIMPLE if n_nodes > 1 else 0.0
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
        # Tree only for AllReduce
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
    n_bytes: int,
    topo: MeshDimTopo,
    config: NCCLTopoConfig,
) -> float:
    """Compute estimated time in microseconds for one algorithm.

    Returns float('inf') if the algorithm is disabled (zero bandwidth).
    """
    bw = _compute_algo_bw(func, algo, topo, config)
    lat = _compute_algo_latency(func, algo, topo, config)

    if bw <= 0:
        return float("inf")

    # Tree correction factor (line 595)
    log_size = _log2i(n_bytes >> 6)
    if algo == NCCLAlgo.TREE and func == NCCLFunc.ALLREDUCE and 0 <= log_size < 24:
        bw *= _TREE_CORRECTION_FACTOR_SIMPLE[log_size]

    # NVLS_Tree correction for Blackwell (line 596)
    if (
        algo == NCCLAlgo.NVLS_TREE
        and func == NCCLFunc.ALLREDUCE
        and config.arch == GpuArch.BLACKWELL
        and 0 <= log_size < 24
    ):
        bw *= _TREE_CORRECTION_FACTOR_SIMPLE[log_size]

    # Ring plateau effect for multi-node Simple allreduce (lines 597-599)
    if (
        algo == NCCLAlgo.RING
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


def nccl_collective_time(
    func: NCCLFunc,
    n_bytes: int,
    topo: MeshDimTopo,
    config: NCCLTopoConfig,
) -> float:
    """Pick the best algorithm and return estimated time in microseconds."""
    algos = _eligible_algos(func, config, topo.n_nodes)
    best = float("inf")
    for algo in algos:
        t = _nccl_algo_time(func, algo, n_bytes, topo, config)
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
    """Cost model for AllToAll based on NCCL's P2P ring decomposition.

    NCCL decomposes AllToAll into nRanks P2P Send/Recv pairs, giving
    nsteps = nRanks (vs nRanks-1 for AG/RS) and bus-to-algo ratio = 1.0
    (no nRanks/(nRanks-1) boost).
    """
    n_ranks = topo.n_ranks
    n_nodes = topo.n_nodes

    # Bandwidth: Ring-style, ratio = 1.0 (no bus-to-algo boost)
    bw = topo.bw_intra if n_nodes <= 2 else topo.bw_inter
    bus_bw = topo.n_channels * bw

    if bus_bw <= 0:
        return float("inf")

    # Latency: Ring-style with nsteps = nRanks
    nsteps = n_ranks
    base_lat = _BASE_LATENCIES[NCCLAlgo.RING]
    intra_lat = _HW_LAT_NVLINK[NCCLAlgo.RING]

    if topo.ppn == 1:
        inter_lat_base = _HW_LAT_NET[NCCLAlgo.TREE]
    else:
        inter_lat_base = _HW_LAT_NET[NCCLAlgo.RING]
    inter_lat = inter_lat_base + config.net_latency
    inter_lat += config.net_latency  # Simple protocol doubles net_latency

    net_overhead = _NET_OVERHEAD_SIMPLE if n_nodes > 1 else 0.0
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
    """DGX H100: 8x H100 per node, NVSwitch 225 GB/s intra-node."""
    return NCCLTopoConfig(
        arch=GpuArch.HOPPER,
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        bw_intra=225.0,
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
    """GB200 NVL72: 72x B200 per rack, NVSwitch 400 GB/s intra-node."""
    return NCCLTopoConfig(
        arch=GpuArch.BLACKWELL,
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        bw_intra=400.0,
        bw_inter=bw_inter,
        has_nvswitch=has_nvswitch,
        **kwargs,
    )
