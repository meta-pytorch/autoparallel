# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from autoparallel.cost_models.nccl_cost_model import (
    _A2A_MULTI_NODE_LAT_POINTS,
    _A2A_NIC_OVERHEAD,
    _A2A_NIC_RAMP_FACTOR,
    _AGRS_MULTI_NODE_BW_POINTS,
    _AGRS_MULTI_NODE_LAT_POINTS,
    _AGRS_MULTI_NODE_RAMP,
    _AR_MULTI_NODE_BW_POINTS,
    _AR_MULTI_NODE_LAT_POINTS,
    _AR_MULTI_NODE_RAMP_N2,
    _AR_MULTI_NODE_RAMP_N4,
    _AR_MULTI_NODE_RAMP_N16,
    _BLACKWELL_BW_SCALE,
    _RING_CORRECTION_FACTOR,
    GpuArch,
    NCCLAlgo,
    NCCLFunc,
    NCCLProto,
    NCCLTopoConfig,
    _compute_algo_bw,
    _compute_algo_latency,
    _eligible_algos,
    _eligible_protos,
    _interp_clamped,
    _log2i,
    _nccl_algo_time,
    a100_topo_config,
    derive_mesh_dim_topo,
    detect_nccl_topo_config,
    gb200_topo_config,
    h100_topo_config,
    nccl_all_to_all_cost,
    nccl_allgather_cost,
    nccl_allreduce_cost,
    nccl_collective_time,
    nccl_reduce_scatter_cost,
)

# ---- derive_mesh_dim_topo tests ----


class TestDeriveMeshDimTopo:
    def test_single_dim_single_node(self):
        config = a100_topo_config()
        topo = derive_mesh_dim_topo(config, (8,), 0)
        assert topo.n_ranks == 8
        assert topo.n_nodes == 1
        assert topo.ppn == 8
        assert topo.n_channels == 12

    def test_two_dim_mesh_inner(self):
        config = a100_topo_config(num_nodes=2)
        # mesh (2, 8): dim 1 is inner, inner_product = 1
        topo = derive_mesh_dim_topo(config, (2, 8), 1)
        assert topo.n_ranks == 8
        assert topo.ppn == 8
        assert topo.n_nodes == 1

    def test_two_dim_mesh_outer(self):
        config = a100_topo_config(num_nodes=2)
        # mesh (2, 8): dim 0 is outer, inner_product = 8
        topo = derive_mesh_dim_topo(config, (2, 8), 0)
        assert topo.n_ranks == 2
        assert topo.ppn == 1
        assert topo.n_nodes == 2

    def test_bw_per_channel(self):
        config = h100_topo_config(num_channels=16)
        topo = derive_mesh_dim_topo(config, (8,), 0)
        assert topo.bw_intra == pytest.approx(320.0 / 16)
        assert topo.bw_inter == pytest.approx(50.0 / 16)

    def test_custom_channels(self):
        config = a100_topo_config(num_channels=8)
        topo = derive_mesh_dim_topo(config, (8,), 0)
        assert topo.n_channels == 8

    def test_three_dim_mesh(self):
        # (2, 4, 4) on 2 nodes with 8 gpus each => 32 total
        config = NCCLTopoConfig(
            arch=GpuArch.HOPPER,
            num_nodes=4,
            gpus_per_node=8,
            bw_intra=320.0,
            bw_inter=50.0,
        )
        # dim 2: inner_product=1, ppn=min(8//1,4)=4, n_nodes=1
        topo2 = derive_mesh_dim_topo(config, (2, 4, 4), 2)
        assert topo2.ppn == 4
        assert topo2.n_nodes == 1

        # dim 1: inner_product=4, ppn=min(8//4,4)=2, n_nodes=2
        topo1 = derive_mesh_dim_topo(config, (2, 4, 4), 1)
        assert topo1.ppn == 2
        assert topo1.n_nodes == 2

        # dim 0: inner_product=16, ppn=max(1, min(8//16,2))=1, n_nodes=2
        topo0 = derive_mesh_dim_topo(config, (2, 4, 4), 0)
        assert topo0.ppn == 1
        assert topo0.n_nodes == 2


# ---- Single-algorithm BW/latency sanity tests ----


class TestAlgoBwLatency:
    def test_ring_bw_single_node(self):
        config = a100_topo_config()
        topo = derive_mesh_dim_topo(config, (8,), 0)
        bw = _compute_algo_bw(
            NCCLFunc.ALLREDUCE, NCCLAlgo.RING, NCCLProto.SIMPLE, topo, config
        )
        # Ring AR: busBw = nCh * bw * nRanks / (2*(nRanks-1))
        expected = 12 * (87.7 / 12) * 8 / (2 * 7)
        assert bw == pytest.approx(expected)

    def test_ring_bw_allgather(self):
        config = a100_topo_config()
        topo = derive_mesh_dim_topo(config, (8,), 0)
        bw = _compute_algo_bw(
            NCCLFunc.ALLGATHER, NCCLAlgo.RING, NCCLProto.SIMPLE, topo, config
        )
        # Ring AG: busBw = nCh * bw * nRanks / (nRanks-1)
        expected = 12 * (87.7 / 12) * 8 / 7
        assert bw == pytest.approx(expected)

    def test_tree_bw_allreduce(self):
        config = a100_topo_config()
        topo = derive_mesh_dim_topo(config, (8,), 0)
        bw = _compute_algo_bw(
            NCCLFunc.ALLREDUCE, NCCLAlgo.TREE, NCCLProto.SIMPLE, topo, config
        )
        # Tree AR: busBw = min(nCh*bw*0.92, nCh*perChMax) * 0.5
        per_ch_max = 24.0  # Ampere, N1
        raw = 12 * (87.7 / 12)
        capped = min(raw * 0.92, 12 * per_ch_max)
        expected = capped * 0.5
        assert bw == pytest.approx(expected)

    def test_tree_not_available_for_ag(self):
        config = a100_topo_config()
        topo = derive_mesh_dim_topo(config, (8,), 0)
        bw = _compute_algo_bw(
            NCCLFunc.ALLGATHER, NCCLAlgo.TREE, NCCLProto.SIMPLE, topo, config
        )
        assert bw == 0.0

    def test_nvls_bw_hopper(self):
        config = h100_topo_config()
        topo = derive_mesh_dim_topo(config, (8,), 0)
        bw = _compute_algo_bw(
            NCCLFunc.ALLREDUCE, NCCLAlgo.NVLS, NCCLProto.SIMPLE, topo, config
        )
        assert bw > 0

    def test_nvls_bw_ampere_zero(self):
        config = a100_topo_config(has_nvswitch=True)
        topo = derive_mesh_dim_topo(config, (8,), 0)
        # Ampere nvlsEfficiency = 0.0, so intraBw should be 0
        bw = _compute_algo_bw(
            NCCLFunc.ALLREDUCE, NCCLAlgo.NVLS, NCCLProto.SIMPLE, topo, config
        )
        assert bw == 0.0

    def test_ring_latency_single_node(self):
        config = a100_topo_config()
        topo = derive_mesh_dim_topo(config, (8,), 0)
        lat = _compute_algo_latency(
            NCCLFunc.ALLREDUCE, NCCLAlgo.RING, NCCLProto.SIMPLE, topo, config
        )
        # nNodes=1: nInterSteps=0, lat = baseLat + nsteps*intraLat
        nsteps = 2 * 7
        expected = 8.4 + nsteps * 3.4  # NVLink ring lat = 3.4
        assert lat == pytest.approx(expected)

    def test_ring_latency_multi_node(self):
        config = a100_topo_config(num_nodes=2, gpus_per_node=1)
        topo = derive_mesh_dim_topo(config, (2,), 0)
        lat = _compute_algo_latency(
            NCCLFunc.ALLREDUCE, NCCLAlgo.RING, NCCLProto.SIMPLE, topo, config
        )
        # gpus_per_node=1 => ppn=1, n_nodes=2, nRanks=2, nsteps=2, nInterSteps=2
        # netOverhead = 1.0*3 = 3.0, intraLat = max(3.4, 3.0) = 3.4
        # ppn=1 => interLat base = hwLat[NET][TREE] = 14.0
        # lat = 8.4 + (2-2)*3.4 + 2*14.0 + 4.7*2 = 45.8
        assert lat == pytest.approx(8.4 + 2 * 14.0 + 4.7 * 2)

    def test_tree_latency_single_node(self):
        config = a100_topo_config()
        topo = derive_mesh_dim_topo(config, (8,), 0)
        lat = _compute_algo_latency(
            NCCLFunc.ALLREDUCE, NCCLAlgo.TREE, NCCLProto.SIMPLE, topo, config
        )
        # nNodes=1: lat = baseLat + 2*((ppn-1)*intraLat + log2(1)*interLat)
        # = 8.4 + 2*(7*4.0 + 0) = 8.4 + 56 = 64.4
        assert lat == pytest.approx(8.4 + 2 * 7 * 4.0)


# ---- Algorithm selection tests ----


class TestAlgorithmSelection:
    def test_nvls_wins_intra_node_hopper(self):
        config = h100_topo_config()
        topo = derive_mesh_dim_topo(config, (8,), 0)
        n_bytes = 1 << 20  # 1MB
        ring_time = _nccl_algo_time(
            NCCLFunc.ALLREDUCE, NCCLAlgo.RING, NCCLProto.SIMPLE, n_bytes, topo, config
        )
        nvls_time = _nccl_algo_time(
            NCCLFunc.ALLREDUCE, NCCLAlgo.NVLS, NCCLProto.SIMPLE, n_bytes, topo, config
        )
        # NVLS should be competitive or better for large-ish messages on Hopper
        assert nvls_time < ring_time * 2  # At least in the same ballpark

    def test_tree_competes_multi_node_ar(self):
        config = a100_topo_config(num_nodes=4)
        topo = derive_mesh_dim_topo(config, (4,), 0)
        n_bytes = 1 << 20
        ring_time = _nccl_algo_time(
            NCCLFunc.ALLREDUCE, NCCLAlgo.RING, NCCLProto.SIMPLE, n_bytes, topo, config
        )
        tree_time = _nccl_algo_time(
            NCCLFunc.ALLREDUCE, NCCLAlgo.TREE, NCCLProto.SIMPLE, n_bytes, topo, config
        )
        # Tree should be competitive for multi-node AllReduce
        assert tree_time < float("inf")
        assert ring_time < float("inf")


# ---- Feature flag filtering ----


class TestFeatureFlags:
    def test_nvls_disabled_without_nvswitch(self):
        config = h100_topo_config(has_nvswitch=False)
        algos = _eligible_algos(NCCLFunc.ALLREDUCE, config, 1)
        assert NCCLAlgo.NVLS not in algos
        assert NCCLAlgo.NVLS_TREE not in algos

    def test_nvls_enabled_with_nvswitch_hopper(self):
        config = h100_topo_config()
        algos = _eligible_algos(NCCLFunc.ALLREDUCE, config, 1)
        assert NCCLAlgo.NVLS in algos

    def test_nvls_tree_requires_multi_node(self):
        config = h100_topo_config()
        algos_1n = _eligible_algos(NCCLFunc.ALLREDUCE, config, 1)
        algos_2n = _eligible_algos(NCCLFunc.ALLREDUCE, config, 2)
        assert NCCLAlgo.NVLS_TREE not in algos_1n
        assert NCCLAlgo.NVLS_TREE in algos_2n

    def test_collnet_disabled_without_flag(self):
        config = a100_topo_config(has_collnet=False)
        algos = _eligible_algos(NCCLFunc.ALLREDUCE, config, 1)
        assert NCCLAlgo.COLLNET_DIRECT not in algos
        assert NCCLAlgo.COLLNET_CHAIN not in algos

    def test_collnet_enabled_with_flag(self):
        config = a100_topo_config(has_collnet=True)
        algos = _eligible_algos(NCCLFunc.ALLREDUCE, config, 2)
        assert NCCLAlgo.COLLNET_DIRECT in algos
        assert NCCLAlgo.COLLNET_CHAIN in algos

    def test_nvls_not_eligible_ampere(self):
        config = a100_topo_config(has_nvswitch=True)
        algos = _eligible_algos(NCCLFunc.ALLREDUCE, config, 1)
        assert NCCLAlgo.NVLS not in algos

    def test_ag_rs_eligible_algos(self):
        config = h100_topo_config(has_collnet=True)
        algos = _eligible_algos(NCCLFunc.ALLGATHER, config, 2)
        assert NCCLAlgo.RING in algos
        assert NCCLAlgo.NVLS in algos
        assert NCCLAlgo.COLLNET_DIRECT in algos
        # Tree and CollNet Chain not available for AG/RS
        assert NCCLAlgo.TREE not in algos
        assert NCCLAlgo.COLLNET_CHAIN not in algos


# ---- Protocol eligibility ----


class TestProtocolEligibility:
    def test_ring_all_protos(self):
        protos = _eligible_protos(NCCLAlgo.RING)
        assert NCCLProto.LL in protos
        assert NCCLProto.LL128 in protos
        assert NCCLProto.SIMPLE in protos

    def test_tree_all_protos(self):
        protos = _eligible_protos(NCCLAlgo.TREE)
        assert NCCLProto.LL in protos
        assert NCCLProto.LL128 in protos
        assert NCCLProto.SIMPLE in protos

    def test_nvls_simple_only(self):
        assert _eligible_protos(NCCLAlgo.NVLS) == [NCCLProto.SIMPLE]

    def test_nvls_tree_simple_only(self):
        assert _eligible_protos(NCCLAlgo.NVLS_TREE) == [NCCLProto.SIMPLE]

    def test_collnet_simple_only(self):
        assert _eligible_protos(NCCLAlgo.COLLNET_DIRECT) == [NCCLProto.SIMPLE]
        assert _eligible_protos(NCCLAlgo.COLLNET_CHAIN) == [NCCLProto.SIMPLE]


# ---- LL/LL128 bandwidth tests ----


class TestLLBandwidth:
    def test_ring_ll_capped_by_ll_max(self):
        """Ring+LL bandwidth should be capped by llMaxBws."""
        config = a100_topo_config()
        topo = derive_mesh_dim_topo(config, (8,), 0)
        bw_ll = _compute_algo_bw(
            NCCLFunc.ALLGATHER, NCCLAlgo.RING, NCCLProto.LL, topo, config
        )
        bw_simple = _compute_algo_bw(
            NCCLFunc.ALLGATHER, NCCLAlgo.RING, NCCLProto.SIMPLE, topo, config
        )
        # LL: min(llMaxBw, busBw * 0.5) then bus-to-algo
        # Simple: busBw then bus-to-algo
        assert bw_ll < bw_simple
        assert bw_ll > 0

    def test_ring_ll128_capped(self):
        """Ring+LL128 bandwidth should be capped by perChMaxRingLL128Bw."""
        config = a100_topo_config()
        topo = derive_mesh_dim_topo(config, (8,), 0)
        bw_ll128 = _compute_algo_bw(
            NCCLFunc.ALLGATHER, NCCLAlgo.RING, NCCLProto.LL128, topo, config
        )
        bw_simple = _compute_algo_bw(
            NCCLFunc.ALLGATHER, NCCLAlgo.RING, NCCLProto.SIMPLE, topo, config
        )
        # LL128: min(busBw * 0.92, nCh * perChMax) then bus-to-algo
        assert bw_ll128 <= bw_simple
        assert bw_ll128 > 0

    def test_tree_ll_much_lower_than_simple(self):
        """Tree+LL should have much lower BW than Tree+Simple."""
        config = a100_topo_config()
        topo = derive_mesh_dim_topo(config, (8,), 0)
        bw_ll = _compute_algo_bw(
            NCCLFunc.ALLREDUCE, NCCLAlgo.TREE, NCCLProto.LL, topo, config
        )
        bw_simple = _compute_algo_bw(
            NCCLFunc.ALLREDUCE, NCCLAlgo.TREE, NCCLProto.SIMPLE, topo, config
        )
        # Tree+LL: min(busBw/3.8, llMaxBw) * 0.5
        assert bw_ll < bw_simple
        assert bw_ll > 0

    def test_nvls_ll_returns_zero(self):
        """NVLS + non-Simple protocol should return 0."""
        config = h100_topo_config()
        topo = derive_mesh_dim_topo(config, (8,), 0)
        bw_ll = _compute_algo_bw(
            NCCLFunc.ALLREDUCE, NCCLAlgo.NVLS, NCCLProto.LL, topo, config
        )
        bw_ll128 = _compute_algo_bw(
            NCCLFunc.ALLREDUCE, NCCLAlgo.NVLS, NCCLProto.LL128, topo, config
        )
        assert bw_ll == 0.0
        assert bw_ll128 == 0.0


# ---- LL latency tests ----


class TestLLLatency:
    def test_ring_ll_lower_latency_than_simple(self):
        """Ring+LL should have lower latency than Ring+Simple."""
        config = a100_topo_config()
        topo = derive_mesh_dim_topo(config, (8,), 0)
        lat_ll = _compute_algo_latency(
            NCCLFunc.ALLREDUCE, NCCLAlgo.RING, NCCLProto.LL, topo, config
        )
        lat_simple = _compute_algo_latency(
            NCCLFunc.ALLREDUCE, NCCLAlgo.RING, NCCLProto.SIMPLE, topo, config
        )
        # LL base latency (6.6) + intra (0.6) vs Simple base (8.4) + intra (3.4)
        assert lat_ll < lat_simple

    def test_ring_ll_multi_node_net_overhead_lower(self):
        """Ring+LL multi-node: net overhead is not multiplied by 3."""
        config = a100_topo_config(num_nodes=4, gpus_per_node=1)
        topo = derive_mesh_dim_topo(config, (4,), 0)
        lat_ll = _compute_algo_latency(
            NCCLFunc.ALLREDUCE, NCCLAlgo.RING, NCCLProto.LL, topo, config
        )
        lat_simple = _compute_algo_latency(
            NCCLFunc.ALLREDUCE, NCCLAlgo.RING, NCCLProto.SIMPLE, topo, config
        )
        assert lat_ll < lat_simple

    def test_inter_lat_no_double_for_ll(self):
        """LL protocol should not double the net_latency (no flush overhead)."""
        config = a100_topo_config(num_nodes=2, gpus_per_node=1, net_latency=5.0)
        topo = derive_mesh_dim_topo(config, (2,), 0)
        lat_ll = _compute_algo_latency(
            NCCLFunc.ALLREDUCE, NCCLAlgo.RING, NCCLProto.LL, topo, config
        )
        lat_simple = _compute_algo_latency(
            NCCLFunc.ALLREDUCE, NCCLAlgo.RING, NCCLProto.SIMPLE, topo, config
        )
        # Simple doubles net_latency (adds 5.0 extra), LL doesn't
        assert lat_simple > lat_ll


# ---- Tree correction factor ----


class TestTreeCorrection:
    def test_tree_correction_applied(self):
        config = a100_topo_config(num_nodes=2)
        topo = derive_mesh_dim_topo(config, (2,), 0)
        # Small message: correction factor < 1 for some sizes
        small = 1 << 10  # 1KB, log2(1024/64) = 4 -> factor = 0.9
        large = 1 << 18  # 256KB, log2(256K/64) = 12 -> factor = 0.5

        time_small = _nccl_algo_time(
            NCCLFunc.ALLREDUCE, NCCLAlgo.TREE, NCCLProto.SIMPLE, small, topo, config
        )
        time_large = _nccl_algo_time(
            NCCLFunc.ALLREDUCE, NCCLAlgo.TREE, NCCLProto.SIMPLE, large, topo, config
        )

        # Both should be finite
        assert time_small < float("inf")
        assert time_large < float("inf")

        # Verify correction is actually applied by comparing with uncorrected
        bw_raw = _compute_algo_bw(
            NCCLFunc.ALLREDUCE, NCCLAlgo.TREE, NCCLProto.SIMPLE, topo, config
        )
        lat = _compute_algo_latency(
            NCCLFunc.ALLREDUCE, NCCLAlgo.TREE, NCCLProto.SIMPLE, topo, config
        )
        time_uncorrected = lat + large / (1000.0 * bw_raw)
        # With correction factor of 0.5, bw is halved => comm time doubles
        assert time_large > time_uncorrected


# ---- Ring correction factor ----


class TestRingCorrection:
    def test_multi_node_ring_has_correction(self):
        """Multi-node Ring AG at 2M should be slower than uncorrected."""
        config = h100_topo_config(num_nodes=2, gpus_per_node=8)
        topo = derive_mesh_dim_topo(config, (16,), 0)
        n_bytes = 2 * (1 << 20)  # 2MB total => 128KB per GPU => logSize=11 => 0.05

        bw_raw = _compute_algo_bw(
            NCCLFunc.ALLGATHER, NCCLAlgo.RING, NCCLProto.SIMPLE, topo, config
        )
        lat = _compute_algo_latency(
            NCCLFunc.ALLGATHER, NCCLAlgo.RING, NCCLProto.SIMPLE, topo, config
        )
        time_uncorrected = lat + n_bytes / (1000.0 * bw_raw)

        time_corrected = _nccl_algo_time(
            NCCLFunc.ALLGATHER, NCCLAlgo.RING, NCCLProto.SIMPLE, n_bytes, topo, config
        )
        assert time_corrected > time_uncorrected

    def test_single_node_ring_unaffected(self):
        """Single-node Ring should not have any Ring correction applied."""
        config = h100_topo_config()
        topo = derive_mesh_dim_topo(config, (8,), 0)
        n_bytes = 2 * (1 << 20)

        bw_raw = _compute_algo_bw(
            NCCLFunc.ALLGATHER, NCCLAlgo.RING, NCCLProto.SIMPLE, topo, config
        )
        lat = _compute_algo_latency(
            NCCLFunc.ALLGATHER, NCCLAlgo.RING, NCCLProto.SIMPLE, topo, config
        )
        time_uncorrected = lat + n_bytes / (1000.0 * bw_raw)

        time_actual = _nccl_algo_time(
            NCCLFunc.ALLGATHER, NCCLAlgo.RING, NCCLProto.SIMPLE, n_bytes, topo, config
        )
        assert time_actual == pytest.approx(time_uncorrected)

    def test_large_messages_no_effect(self):
        """Large messages (>=1G per GPU) should have correction ~1.0."""
        config = h100_topo_config(num_nodes=2, gpus_per_node=8)
        topo = derive_mesh_dim_topo(config, (16,), 0)
        n_bytes = 16 * (1 << 30)  # 16GB total => 1GB per GPU => logSize=24

        bw_raw = _compute_algo_bw(
            NCCLFunc.ALLGATHER, NCCLAlgo.RING, NCCLProto.SIMPLE, topo, config
        )
        lat = _compute_algo_latency(
            NCCLFunc.ALLGATHER, NCCLAlgo.RING, NCCLProto.SIMPLE, topo, config
        )
        time_uncorrected = lat + n_bytes / (1000.0 * bw_raw)

        time_actual = _nccl_algo_time(
            NCCLFunc.ALLGATHER, NCCLAlgo.RING, NCCLProto.SIMPLE, n_bytes, topo, config
        )
        # logSize >= 24 means no correction applied (factor stays 1.0)
        assert time_actual == pytest.approx(time_uncorrected)

    def test_correction_factor_values(self):
        """Verify the correction table has expected shape and bounds."""
        assert len(_RING_CORRECTION_FACTOR) == 24
        assert _RING_CORRECTION_FACTOR[0] == 1.00
        assert _RING_CORRECTION_FACTOR[23] == 1.00
        for f in _RING_CORRECTION_FACTOR:
            assert 0.0 < f <= 1.0

    def test_depth_scaling_8_node(self):
        """8-node Ring should be slower than 4-node at the same per-GPU size."""
        n_bytes = 64 * (1 << 20)  # 64M total
        # 4-node: 32 GPUs, per_gpu = 2M, logSize=15
        config_4n = h100_topo_config(num_nodes=4)
        topo_4n = derive_mesh_dim_topo(config_4n, (32,), 0)
        # 8-node: 64 GPUs, per_gpu = 1M, logSize=14
        config_8n = h100_topo_config(num_nodes=8)
        topo_8n = derive_mesh_dim_topo(config_8n, (64,), 0)

        time_4n = _nccl_algo_time(
            NCCLFunc.ALLGATHER,
            NCCLAlgo.RING,
            NCCLProto.SIMPLE,
            n_bytes,
            topo_4n,
            config_4n,
        )
        time_8n = _nccl_algo_time(
            NCCLFunc.ALLGATHER,
            NCCLAlgo.RING,
            NCCLProto.SIMPLE,
            n_bytes,
            topo_8n,
            config_8n,
        )
        # 8-node should be slower (more pipeline stages, depth-adjusted correction)
        assert time_8n > time_4n

    def test_depth_scaling_amplifies_correction(self):
        """At 8 nodes, the effective correction should be stronger than table value."""
        config = h100_topo_config(num_nodes=8)
        topo = derive_mesh_dim_topo(config, (64,), 0)
        n_bytes = 8 * (1 << 20)  # 8M total => 128K/GPU, logSize=11, table f=0.10

        bw_raw = _compute_algo_bw(
            NCCLFunc.ALLGATHER, NCCLAlgo.RING, NCCLProto.SIMPLE, topo, config
        )
        lat = _compute_algo_latency(
            NCCLFunc.ALLGATHER, NCCLAlgo.RING, NCCLProto.SIMPLE, topo, config
        )
        # Time with table correction only (f=0.10)
        time_table_only = lat + n_bytes / (1000.0 * bw_raw * 0.10)
        # Time with depth-amplified correction (f=0.10^1.2=0.063)
        time_actual = _nccl_algo_time(
            NCCLFunc.ALLGATHER, NCCLAlgo.RING, NCCLProto.SIMPLE, n_bytes, topo, config
        )
        # Depth scaling makes it slower than table-only
        assert time_actual > time_table_only


class TestRingPlateau:
    def test_ring_plateau_multi_node(self):
        config = a100_topo_config(num_nodes=4, gpus_per_node=1)
        topo = derive_mesh_dim_topo(config, (4,), 0)
        n_ch = topo.n_channels
        n_ranks = topo.n_ranks
        # Need n_bytes / (nChannels * nRanks) >= 64
        threshold = 64 * n_ch * n_ranks
        # Just below and above threshold
        below = threshold - 1
        above = threshold

        time_below = _nccl_algo_time(
            NCCLFunc.ALLREDUCE, NCCLAlgo.RING, NCCLProto.SIMPLE, below, topo, config
        )
        time_above = _nccl_algo_time(
            NCCLFunc.ALLREDUCE, NCCLAlgo.RING, NCCLProto.SIMPLE, above, topo, config
        )

        # The latency inflation of 1.4x should cause a jump at the threshold
        bw = _compute_algo_bw(
            NCCLFunc.ALLREDUCE, NCCLAlgo.RING, NCCLProto.SIMPLE, topo, config
        )
        lat_base = _compute_algo_latency(
            NCCLFunc.ALLREDUCE, NCCLAlgo.RING, NCCLProto.SIMPLE, topo, config
        )

        # Account for Ring correction factor (multi-node)
        def _ring_corrected_bw(n_bytes_val):
            log_pg = _log2i((n_bytes_val // n_ranks) >> 6)
            if 0 <= log_pg < 24:
                return bw * _RING_CORRECTION_FACTOR[log_pg]
            elif log_pg < 0:
                return bw * _RING_CORRECTION_FACTOR[0]
            return bw

        # Below: no plateau inflation
        expected_below = lat_base + below / (1000.0 * _ring_corrected_bw(below))
        # Above: lat *= 1.4
        expected_above = lat_base * 1.4 + above / (1000.0 * _ring_corrected_bw(above))

        assert time_below == pytest.approx(expected_below)
        assert time_above == pytest.approx(expected_above)

    def test_ring_plateau_only_simple(self):
        """Ring plateau should only apply to Simple protocol."""
        config = a100_topo_config(num_nodes=4, gpus_per_node=1)
        topo = derive_mesh_dim_topo(config, (4,), 0)
        n_bytes = 1 << 20  # well above threshold

        bw_ll = _compute_algo_bw(
            NCCLFunc.ALLREDUCE, NCCLAlgo.RING, NCCLProto.LL, topo, config
        )
        lat_ll = _compute_algo_latency(
            NCCLFunc.ALLREDUCE, NCCLAlgo.RING, NCCLProto.LL, topo, config
        )
        time_ll = _nccl_algo_time(
            NCCLFunc.ALLREDUCE, NCCLAlgo.RING, NCCLProto.LL, n_bytes, topo, config
        )
        if bw_ll > 0:
            # LL should NOT have plateau inflation, but Ring correction applies
            log_pg = _log2i((n_bytes // topo.n_ranks) >> 6)
            corr = _RING_CORRECTION_FACTOR[log_pg] if 0 <= log_pg < 24 else 1.0
            expected_ll = lat_ll + n_bytes / (1000.0 * bw_ll * corr)
            assert time_ll == pytest.approx(expected_ll)


# ---- Monotonicity ----


class TestMonotonicity:
    @pytest.mark.parametrize(
        "func", [NCCLFunc.ALLGATHER, NCCLFunc.ALLREDUCE, NCCLFunc.REDUCESCATTER]
    )
    def test_cost_increases_with_size(self, func):
        config = h100_topo_config()
        topo = derive_mesh_dim_topo(config, (8,), 0)
        sizes = [1 << k for k in range(6, 28, 2)]
        times = [nccl_collective_time(func, s, topo, config) for s in sizes]
        for i in range(1, len(times)):
            assert times[i] >= times[i - 1], (
                f"Cost decreased from {sizes[i-1]} to {sizes[i]}: "
                f"{times[i-1]} > {times[i]}"
            )

    @pytest.mark.parametrize(
        "func", [NCCLFunc.ALLGATHER, NCCLFunc.ALLREDUCE, NCCLFunc.REDUCESCATTER]
    )
    def test_cost_positive(self, func):
        config = a100_topo_config()
        topo = derive_mesh_dim_topo(config, (8,), 0)
        t = nccl_collective_time(func, 1 << 20, topo, config)
        assert t > 0
        assert t < float("inf")


# ---- Public API wrapper tests ----


class TestPublicAPI:
    def test_allgather_cost(self):
        config = a100_topo_config()
        topo = derive_mesh_dim_topo(config, (8,), 0)
        cost = nccl_allgather_cost(1 << 20, topo, config)
        assert cost > 0
        assert cost < float("inf")

    def test_allreduce_cost(self):
        config = a100_topo_config()
        topo = derive_mesh_dim_topo(config, (8,), 0)
        cost = nccl_allreduce_cost(1 << 20, topo, config)
        assert cost > 0
        assert cost < float("inf")

    def test_reduce_scatter_cost(self):
        config = a100_topo_config()
        topo = derive_mesh_dim_topo(config, (8,), 0)
        cost = nccl_reduce_scatter_cost(1 << 20, topo, config)
        assert cost > 0
        assert cost < float("inf")

    def test_allreduce_more_expensive_than_reduce_scatter(self):
        config = a100_topo_config()
        topo = derive_mesh_dim_topo(config, (8,), 0)
        n_bytes = 1 << 24
        ar = nccl_allreduce_cost(n_bytes, topo, config)
        rs = nccl_reduce_scatter_cost(n_bytes, topo, config)
        assert ar > rs


# ---- detect_nccl_topo_config tests ----


class TestDetectNCCLTopoConfig:
    def _make_mock_mesh(self, total_gpus):
        from unittest.mock import MagicMock

        mesh = MagicMock()
        mesh.size.return_value = total_gpus
        return mesh

    def test_a100_detected(self):
        from unittest.mock import patch

        mesh = self._make_mock_mesh(16)
        with (
            patch("torch.cuda.get_device_name", return_value="NVIDIA A100-SXM4-80GB"),
            patch("torch.cuda.device_count", return_value=8),
        ):
            config = detect_nccl_topo_config(mesh)
        assert config is not None
        assert config.arch == GpuArch.AMPERE
        assert config.num_nodes == 2
        assert config.gpus_per_node == 8

    def test_h100_detected(self):
        from unittest.mock import patch

        mesh = self._make_mock_mesh(8)
        with (
            patch("torch.cuda.get_device_name", return_value="NVIDIA H100 80GB HBM3"),
            patch("torch.cuda.device_count", return_value=8),
        ):
            config = detect_nccl_topo_config(mesh)
        assert config is not None
        assert config.arch == GpuArch.HOPPER
        assert config.num_nodes == 1
        assert config.gpus_per_node == 8

    def test_h200_detected(self):
        from unittest.mock import patch

        mesh = self._make_mock_mesh(8)
        with (
            patch("torch.cuda.get_device_name", return_value="NVIDIA H200"),
            patch("torch.cuda.device_count", return_value=8),
        ):
            config = detect_nccl_topo_config(mesh)
        assert config is not None
        assert config.arch == GpuArch.HOPPER

    def test_b200_detected(self):
        from unittest.mock import patch

        mesh = self._make_mock_mesh(72)
        with (
            patch("torch.cuda.get_device_name", return_value="NVIDIA B200"),
            patch("torch.cuda.device_count", return_value=72),
        ):
            config = detect_nccl_topo_config(mesh)
        assert config is not None
        assert config.arch == GpuArch.BLACKWELL
        assert config.num_nodes == 1
        assert config.gpus_per_node == 72

    def test_gb200_detected(self):
        from unittest.mock import patch

        mesh = self._make_mock_mesh(144)
        with (
            patch("torch.cuda.get_device_name", return_value="NVIDIA GB200"),
            patch("torch.cuda.device_count", return_value=72),
        ):
            config = detect_nccl_topo_config(mesh)
        assert config is not None
        assert config.arch == GpuArch.BLACKWELL
        assert config.num_nodes == 2

    def test_unknown_gpu_returns_none(self):
        from unittest.mock import patch

        mesh = self._make_mock_mesh(8)
        with (
            patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090"),
            patch("torch.cuda.device_count", return_value=8),
        ):
            config = detect_nccl_topo_config(mesh)
        assert config is None


# ---- AllToAll cost model tests ----


class TestAllToAllCost:
    def test_positive_and_finite(self):
        config = a100_topo_config()
        topo = derive_mesh_dim_topo(config, (8,), 0)
        cost = nccl_all_to_all_cost(1 << 20, topo, config)
        assert cost > 0
        assert cost < float("inf")

    def test_monotonicity(self):
        config = h100_topo_config()
        topo = derive_mesh_dim_topo(config, (8,), 0)
        sizes = [1 << k for k in range(6, 28, 2)]
        times = [nccl_all_to_all_cost(s, topo, config) for s in sizes]
        for i in range(1, len(times)):
            assert times[i] >= times[i - 1], (
                f"AllToAll cost decreased from {sizes[i-1]} to {sizes[i]}: "
                f"{times[i-1]} > {times[i]}"
            )

    def test_more_expensive_than_allgather(self):
        config = a100_topo_config()
        topo = derive_mesh_dim_topo(config, (8,), 0)
        n_bytes = 1 << 24
        a2a = nccl_all_to_all_cost(n_bytes, topo, config)
        ag = nccl_allgather_cost(n_bytes, topo, config)
        assert a2a > ag

    def test_multi_node_latency_sublinear(self):
        """Multi-node AllToAll latency should grow sublinearly with nodes."""
        lat_2n = _interp_clamped(_A2A_MULTI_NODE_LAT_POINTS, 2)
        lat_4n = _interp_clamped(_A2A_MULTI_NODE_LAT_POINTS, 4)
        lat_8n = _interp_clamped(_A2A_MULTI_NODE_LAT_POINTS, 8)
        assert lat_2n < lat_4n < lat_8n
        # Sublinear: 4x nodes should give less than 4x latency
        assert lat_8n < 4 * lat_2n

    def test_multi_node_bw_scales_with_bw_inter(self):
        """Multi-node NVSwitch AllToAll BW should scale with bw_inter, not bw_intra."""
        config_50 = h100_topo_config(num_nodes=2, bw_inter=50.0)
        config_25 = h100_topo_config(num_nodes=2, bw_inter=25.0)
        topo_50 = derive_mesh_dim_topo(config_50, (16,), 0)
        topo_25 = derive_mesh_dim_topo(config_25, (16,), 0)
        n_bytes = 1 << 30  # 1GB — BW-dominated
        cost_50 = nccl_all_to_all_cost(n_bytes, topo_50, config_50)
        cost_25 = nccl_all_to_all_cost(n_bytes, topo_25, config_25)
        # Doubling NIC BW should roughly halve the BW term
        ratio = cost_25 / cost_50
        assert ratio > 1.5

    def test_nic_bottleneck_path_for_nvswitch_multi_node(self):
        """NVSwitch multi-node should use the NIC-bottleneck path."""
        config = h100_topo_config(num_nodes=2)
        topo = derive_mesh_dim_topo(config, (16,), 0)
        n_bytes = 1 << 30
        cost = nccl_all_to_all_cost(n_bytes, topo, config)
        # Verify against expected NIC-bottleneck formula
        lat = _interp_clamped(_A2A_MULTI_NODE_LAT_POINTS, 2)
        remote_fraction = (16 - 8) / 16
        nic_bw = 50.0 * (1.0 - _A2A_NIC_OVERHEAD / 2)
        expected = lat + n_bytes * remote_fraction / (1000.0 * nic_bw)
        assert cost == pytest.approx(expected)

    def test_sm_fallback_for_no_nvswitch(self):
        """Without NVSwitch, multi-node should use SM-driven fallback."""
        config = h100_topo_config(num_nodes=2, has_nvswitch=False)
        topo = derive_mesh_dim_topo(config, (16,), 0)
        n_bytes = 1 << 20
        cost = nccl_all_to_all_cost(n_bytes, topo, config)
        # SM-driven fallback uses Ring-style latency + channel-based BW;
        # NIC-bottleneck path would give a very different value.
        # Just verify it's finite and positive.
        assert cost > 0
        assert cost < float("inf")
        # SM-driven uses bw_intra for <=2 nodes, which is much higher than
        # NIC-bottleneck's bw_inter, so cost should be lower.
        config_nvs = h100_topo_config(num_nodes=2, has_nvswitch=True)
        topo_nvs = derive_mesh_dim_topo(config_nvs, (16,), 0)
        cost_nvs = nccl_all_to_all_cost(n_bytes, topo_nvs, config_nvs)
        # The two paths model different things so costs differ
        assert cost != pytest.approx(cost_nvs)

    def test_nic_ramp_slows_mid_range(self):
        """Mid-range AllToAll should be slower than uncorrected NIC model."""
        config = h100_topo_config(num_nodes=2)
        topo = derive_mesh_dim_topo(config, (16,), 0)
        # 2M total, per_gpu = 128K, logSize = 11, ramp factor = 0.43
        n_bytes = 2 * (1 << 20)
        cost = nccl_all_to_all_cost(n_bytes, topo, config)
        # Uncorrected NIC model (no ramp)
        lat = _interp_clamped(_A2A_MULTI_NODE_LAT_POINTS, 2)
        remote_fraction = (16 - 8) / 16
        nic_bw = 50.0 * (1.0 - _A2A_NIC_OVERHEAD / 2)
        uncorrected = lat + n_bytes * remote_fraction / (1000.0 * nic_bw)
        assert cost > uncorrected

    def test_nic_ramp_factor_table_shape(self):
        """Ramp factor table should have expected shape and bounds."""
        assert len(_A2A_NIC_RAMP_FACTOR) == 24
        assert _A2A_NIC_RAMP_FACTOR[0] == 1.0
        assert _A2A_NIC_RAMP_FACTOR[23] == 1.0
        for f in _A2A_NIC_RAMP_FACTOR:
            assert 0.0 < f <= 1.0

    def test_nic_ramp_weaker_at_more_nodes(self):
        """More nodes should weaken the ramp correction (NIC saturates sooner)."""
        # 4M total: per_gpu is 256K at 2n (idx 12, factor 0.62) and
        # 128K at 4n (idx 11, factor 0.43). But 4n applies linear deficit
        # dampening: 1-(1-0.43)/2 = 0.715 — weaker than 0.62.
        # Both are slower than uncorrected, but 4n should have a milder
        # correction relative to its uncorrected BW term.
        n_bytes = 4 * (1 << 20)
        config_2n = h100_topo_config(num_nodes=2)
        config_4n = h100_topo_config(num_nodes=4)
        topo_2n = derive_mesh_dim_topo(config_2n, (16,), 0)
        topo_4n = derive_mesh_dim_topo(config_4n, (32,), 0)
        cost_2n = nccl_all_to_all_cost(n_bytes, topo_2n, config_2n)
        cost_4n = nccl_all_to_all_cost(n_bytes, topo_4n, config_4n)
        # Compute uncorrected BW times for each
        lat_2n = _interp_clamped(_A2A_MULTI_NODE_LAT_POINTS, 2)
        lat_4n = _interp_clamped(_A2A_MULTI_NODE_LAT_POINTS, 4)
        rf_2n = (16 - 8) / 16
        rf_4n = (32 - 8) / 32
        nic_2n = 50.0 * (1.0 - _A2A_NIC_OVERHEAD / 2)
        nic_4n = 50.0 * (1.0 - _A2A_NIC_OVERHEAD / 4)
        uncorr_bw_2n = n_bytes * rf_2n / (1000.0 * nic_2n)
        uncorr_bw_4n = n_bytes * rf_4n / (1000.0 * nic_4n)
        # Effective correction = (cost - lat) / uncorrected_bw_time
        # Larger value = weaker correction (closer to 1.0)
        eff_corr_2n = uncorr_bw_2n / (cost_2n - lat_2n)
        eff_corr_4n = uncorr_bw_4n / (cost_4n - lat_4n)
        assert eff_corr_4n > eff_corr_2n


# ---- NVSwitch empirical path tests ----


class TestNVSwitchEmpiricalPath:
    def test_allreduce_more_expensive_than_allgather_h100_nvswitch(self):
        """AllReduce should be more expensive than AllGather on H100 NVSwitch.

        Benchmarks show AllReduce is 47-84% more expensive than AllGather at
        large message sizes on 8xH100 NVSwitch. The formula-based model gets
        this ranking inverted; the empirical path fixes it.
        """
        config = h100_topo_config()
        topo = derive_mesh_dim_topo(config, (8,), 0)
        n_bytes = 1 << 24  # 16MB
        ar = nccl_allreduce_cost(n_bytes, topo, config)
        ag = nccl_allgather_cost(n_bytes, topo, config)
        assert ar > ag

    def test_allgather_equals_reduce_scatter_h100_nvswitch(self):
        """AG and RS should have the same cost on NVSwitch (symmetric BW)."""
        config = h100_topo_config()
        topo = derive_mesh_dim_topo(config, (8,), 0)
        n_bytes = 1 << 24
        ag = nccl_allgather_cost(n_bytes, topo, config)
        rs = nccl_reduce_scatter_cost(n_bytes, topo, config)
        assert ag == pytest.approx(rs)

    def test_not_used_for_multi_node(self):
        """Multi-node should fall through to the algo selection loop."""
        config = h100_topo_config(num_nodes=2)
        topo = derive_mesh_dim_topo(config, (2, 8), 0)
        # Multi-node dim: n_nodes=2, should NOT use empirical path.
        # Just verify it returns a finite positive value (algo loop works).
        cost = nccl_allgather_cost(1 << 20, topo, config)
        assert cost > 0
        assert cost < float("inf")

    def test_not_used_for_ampere(self):
        """Ampere should not use the NVSwitch empirical path."""
        config = a100_topo_config(has_nvswitch=True)
        topo = derive_mesh_dim_topo(config, (8,), 0)
        cost = nccl_allgather_cost(1 << 20, topo, config)
        assert cost > 0
        assert cost < float("inf")

    def test_ag_faster_with_fewer_gpus(self):
        """AllGather is cheaper with fewer GPUs (higher per-GPU algo_bw)."""
        config = h100_topo_config()
        n_bytes = 1 << 24  # 16MB
        cost_2 = nccl_allgather_cost(
            n_bytes, derive_mesh_dim_topo(config, (2,), 0), config
        )
        cost_4 = nccl_allgather_cost(
            n_bytes, derive_mesh_dim_topo(config, (4,), 0), config
        )
        cost_8 = nccl_allgather_cost(
            n_bytes, derive_mesh_dim_topo(config, (8,), 0), config
        )
        assert cost_2 < cost_4 < cost_8

    def test_rs_faster_with_fewer_gpus(self):
        """ReduceScatter is cheaper with fewer GPUs."""
        config = h100_topo_config()
        n_bytes = 1 << 24
        cost_2 = nccl_reduce_scatter_cost(
            n_bytes, derive_mesh_dim_topo(config, (2,), 0), config
        )
        cost_4 = nccl_reduce_scatter_cost(
            n_bytes, derive_mesh_dim_topo(config, (4,), 0), config
        )
        cost_8 = nccl_reduce_scatter_cost(
            n_bytes, derive_mesh_dim_topo(config, (8,), 0), config
        )
        assert cost_2 < cost_4 < cost_8

    def test_ar_4gpu_slower_than_8gpu(self):
        """AllReduce at 4 GPUs is empirically slower than at 8 on NVSwitch.

        The lower BW at 4 GPUs (236 vs 267 GB/s) only dominates at large
        message sizes where the latency term becomes negligible.
        """
        config = h100_topo_config()
        n_bytes = 1 << 26  # 64MB — well above the crossover point
        cost_4 = nccl_allreduce_cost(
            n_bytes, derive_mesh_dim_topo(config, (4,), 0), config
        )
        cost_8 = nccl_allreduce_cost(
            n_bytes, derive_mesh_dim_topo(config, (8,), 0), config
        )
        assert cost_4 > cost_8

    def test_interpolation_between_measured_points(self):
        """GPU count between measured points should give intermediate cost."""
        config = h100_topo_config()
        n_bytes = 1 << 24
        cost_2 = nccl_allgather_cost(
            n_bytes, derive_mesh_dim_topo(config, (2,), 0), config
        )
        cost_3 = nccl_allgather_cost(
            n_bytes, derive_mesh_dim_topo(config, (3,), 0), config
        )
        cost_4 = nccl_allgather_cost(
            n_bytes, derive_mesh_dim_topo(config, (4,), 0), config
        )
        assert cost_2 < cost_3 < cost_4


# ---- NIC aggregation for NVSwitch multi-node ----


class TestNICAggregation:
    def test_ring_bw_increases_with_nvswitch_multi_node(self):
        """Ring BW should increase when NVSwitch aggregates NICs at 4+ nodes."""
        config_nvs = h100_topo_config(num_nodes=4)
        config_no_nvs = h100_topo_config(num_nodes=4, has_nvswitch=False)
        # Flat mesh: all 32 GPUs in one communicator, ppn=8
        topo_nvs = derive_mesh_dim_topo(config_nvs, (32,), 0)
        topo_no_nvs = derive_mesh_dim_topo(config_no_nvs, (32,), 0)
        bw_nvs = _compute_algo_bw(
            NCCLFunc.ALLGATHER, NCCLAlgo.RING, NCCLProto.SIMPLE, topo_nvs, config_nvs
        )
        bw_no_nvs = _compute_algo_bw(
            NCCLFunc.ALLGATHER,
            NCCLAlgo.RING,
            NCCLProto.SIMPLE,
            topo_no_nvs,
            config_no_nvs,
        )
        assert bw_nvs > bw_no_nvs

    def test_ag_cost_scales_sublinearly_with_nodes(self):
        """AG cost at 4 nodes should be less than 2x AG cost at 2 nodes."""
        config_2n = h100_topo_config(num_nodes=2)
        config_4n = h100_topo_config(num_nodes=4)
        n_bytes = 1 << 30  # 1GB
        topo_2n = derive_mesh_dim_topo(config_2n, (16,), 0)
        topo_4n = derive_mesh_dim_topo(config_4n, (32,), 0)
        cost_2n = nccl_allgather_cost(n_bytes, topo_2n, config_2n)
        cost_4n = nccl_allgather_cost(n_bytes, topo_4n, config_4n)
        assert cost_4n < 2 * cost_2n

    def test_nvls_tree_wins_over_nvls_multi_node_ar(self):
        """NVLS_TREE should beat NVLS for multi-node AR on H100 NVSwitch."""
        config = h100_topo_config(num_nodes=4)
        topo = derive_mesh_dim_topo(config, (32,), 0)
        n_bytes = 1 << 30
        nvls_time = _nccl_algo_time(
            NCCLFunc.ALLREDUCE, NCCLAlgo.NVLS, NCCLProto.SIMPLE, n_bytes, topo, config
        )
        nvls_tree_time = _nccl_algo_time(
            NCCLFunc.ALLREDUCE,
            NCCLAlgo.NVLS_TREE,
            NCCLProto.SIMPLE,
            n_bytes,
            topo,
            config,
        )
        assert nvls_tree_time < nvls_time

    def test_a100_no_nvswitch_unaffected(self):
        """A100 (no NVSwitch) Ring BW should be unchanged by aggregation."""
        config = a100_topo_config(num_nodes=4, gpus_per_node=8)
        topo = derive_mesh_dim_topo(config, (32,), 0)
        bw = _compute_algo_bw(
            NCCLFunc.ALLGATHER, NCCLAlgo.RING, NCCLProto.SIMPLE, topo, config
        )
        # Without NVSwitch: bw_inter used directly, no aggregation
        bw_inter_per_ch = 25.0 / 12
        expected = 12 * bw_inter_per_ch * 32 / 31
        assert bw == pytest.approx(expected)


# ---- LL wins for small messages at multi-node ----


class TestLLWinsSmallMessages:
    def test_ll_wins_small_multi_node_ag(self):
        """For small multi-node AG, LL or LL128 should beat Simple."""
        config = a100_topo_config(num_nodes=4, gpus_per_node=1)
        topo = derive_mesh_dim_topo(config, (4,), 0)
        n_bytes = 1 << 10  # 1KB
        time_simple = _nccl_algo_time(
            NCCLFunc.ALLGATHER, NCCLAlgo.RING, NCCLProto.SIMPLE, n_bytes, topo, config
        )
        time_ll = _nccl_algo_time(
            NCCLFunc.ALLGATHER, NCCLAlgo.RING, NCCLProto.LL, n_bytes, topo, config
        )
        # LL should have lower latency, winning for small messages
        assert time_ll < time_simple

    def test_simple_wins_large_message(self):
        """For large messages, Simple should beat LL (higher BW)."""
        config = a100_topo_config()
        topo = derive_mesh_dim_topo(config, (8,), 0)
        n_bytes = 1 << 24  # 16MB
        time_simple = _nccl_algo_time(
            NCCLFunc.ALLGATHER, NCCLAlgo.RING, NCCLProto.SIMPLE, n_bytes, topo, config
        )
        time_ll = _nccl_algo_time(
            NCCLFunc.ALLGATHER, NCCLAlgo.RING, NCCLProto.LL, n_bytes, topo, config
        )
        assert time_simple < time_ll

    def test_collective_time_picks_best_proto(self):
        """nccl_collective_time should pick the best protocol automatically."""
        config = a100_topo_config(num_nodes=4, gpus_per_node=1)
        topo = derive_mesh_dim_topo(config, (4,), 0)
        n_bytes = 1 << 10
        best = nccl_collective_time(NCCLFunc.ALLGATHER, n_bytes, topo, config)
        # Should be at most the Simple time
        time_simple = _nccl_algo_time(
            NCCLFunc.ALLGATHER, NCCLAlgo.RING, NCCLProto.SIMPLE, n_bytes, topo, config
        )
        assert best <= time_simple


# ---- Multi-node AllReduce empirical path tests ----


class TestARMultiNodeEmpirical:
    def test_empirical_path_used_for_multi_node_h100_ar(self):
        """Multi-node H100 AllReduce should use the empirical path."""
        config = h100_topo_config(num_nodes=2)
        topo = derive_mesh_dim_topo(config, (16,), 0)
        n_bytes = 1 << 30  # 1GB — BW-dominated, ramp ~1.0
        cost = nccl_allreduce_cost(n_bytes, topo, config)
        # Verify against the empirical formula
        bw = _interp_clamped(_AR_MULTI_NODE_BW_POINTS, 2)
        lat = _interp_clamped(_AR_MULTI_NODE_LAT_POINTS, 2)
        log_per_gpu = _log2i((n_bytes // 16) >> 6)
        ramp = _AR_MULTI_NODE_RAMP_N2[log_per_gpu]
        expected = lat + n_bytes / (1000.0 * bw * ramp)
        assert cost == pytest.approx(expected)

    def test_not_used_for_single_node(self):
        """Single-node H100 AllReduce should use the NVSwitch empirical path."""
        config = h100_topo_config()
        topo = derive_mesh_dim_topo(config, (8,), 0)
        n_bytes = 1 << 30
        cost = nccl_allreduce_cost(n_bytes, topo, config)
        # Should match single-node NVSwitch path, not multi-node AR path
        from autoparallel.cost_models.nccl_cost_model import (
            _NVSWITCH_BW_POINTS,
            _NVSWITCH_LAT_POINTS,
        )

        bw = _interp_clamped(_NVSWITCH_BW_POINTS[NCCLFunc.ALLREDUCE], 8)
        lat = _interp_clamped(_NVSWITCH_LAT_POINTS[NCCLFunc.ALLREDUCE], 8)
        expected = lat + n_bytes / (1000.0 * bw)
        assert cost == pytest.approx(expected)

    def test_not_used_for_ampere(self):
        """Ampere AllReduce should fall through to the algo selection loop."""
        config = a100_topo_config(num_nodes=2, gpus_per_node=8)
        topo = derive_mesh_dim_topo(config, (16,), 0)
        n_bytes = 1 << 30
        cost = nccl_allreduce_cost(n_bytes, topo, config)
        # Should NOT match the empirical formula
        bw = _interp_clamped(_AR_MULTI_NODE_BW_POINTS, 2)
        lat = _interp_clamped(_AR_MULTI_NODE_LAT_POINTS, 2)
        empirical = lat + n_bytes / (1000.0 * bw)
        assert cost != pytest.approx(empirical)

    def test_not_used_without_nvswitch(self):
        """H100 without NVSwitch should fall through to the algo loop."""
        config = h100_topo_config(num_nodes=2, has_nvswitch=False)
        topo = derive_mesh_dim_topo(config, (16,), 0)
        n_bytes = 1 << 30
        cost = nccl_allreduce_cost(n_bytes, topo, config)
        bw = _interp_clamped(_AR_MULTI_NODE_BW_POINTS, 2)
        lat = _interp_clamped(_AR_MULTI_NODE_LAT_POINTS, 2)
        empirical = lat + n_bytes / (1000.0 * bw)
        assert cost != pytest.approx(empirical)

    def test_monotonicity(self):
        """Multi-node AllReduce cost should increase with message size."""
        config = h100_topo_config(num_nodes=2)
        topo = derive_mesh_dim_topo(config, (16,), 0)
        sizes = [1 << k for k in range(6, 34, 2)]
        times = [nccl_allreduce_cost(s, topo, config) for s in sizes]
        for i in range(1, len(times)):
            assert times[i] >= times[i - 1], (
                f"Cost decreased from {sizes[i-1]} to {sizes[i]}: "
                f"{times[i-1]} > {times[i]}"
            )

    def test_ramp_tables_shape(self):
        """Ramp tables should have 24 entries with values in (0, 1]."""
        for table in [
            _AR_MULTI_NODE_RAMP_N2,
            _AR_MULTI_NODE_RAMP_N4,
            _AR_MULTI_NODE_RAMP_N16,
        ]:
            assert len(table) == 24
            for f in table:
                assert 0.0 < f <= 1.0

    def test_n16_ramp_used_for_16_nodes(self):
        """16-node AllReduce should use the N16 ramp table."""
        config = h100_topo_config(num_nodes=16)
        topo = derive_mesh_dim_topo(config, (128,), 0)
        n_bytes = 1 << 30
        cost = nccl_allreduce_cost(n_bytes, topo, config)
        bw = _interp_clamped(_AR_MULTI_NODE_BW_POINTS, 16)
        lat = _interp_clamped(_AR_MULTI_NODE_LAT_POINTS, 16)
        log_per_gpu = _log2i((n_bytes // 128) >> 6)
        ramp = _AR_MULTI_NODE_RAMP_N16[log_per_gpu]
        expected = lat + n_bytes / (1000.0 * bw * ramp)
        assert cost == pytest.approx(expected)

    def test_monotonicity_16_nodes(self):
        """16-node AllReduce cost should increase with message size."""
        config = h100_topo_config(num_nodes=16)
        topo = derive_mesh_dim_topo(config, (128,), 0)
        sizes = [1 << k for k in range(6, 34, 2)]
        times = [nccl_allreduce_cost(s, topo, config) for s in sizes]
        for i in range(1, len(times)):
            assert times[i] >= times[i - 1], (
                f"Cost decreased from {sizes[i-1]} to {sizes[i]}: "
                f"{times[i-1]} > {times[i]}"
            )

    def test_blackwell_bw_scales(self):
        """Blackwell multi-node AR should scale BW by _BLACKWELL_BW_SCALE."""
        config_h = h100_topo_config(num_nodes=2)
        config_b = gb200_topo_config(num_nodes=2, gpus_per_node=8)
        topo_h = derive_mesh_dim_topo(config_h, (16,), 0)
        topo_b = derive_mesh_dim_topo(config_b, (16,), 0)
        n_bytes = 1 << 32  # Large enough that ramp ~1.0 and lat negligible
        cost_h = nccl_allreduce_cost(n_bytes, topo_h, config_h)
        cost_b = nccl_allreduce_cost(n_bytes, topo_b, config_b)
        # BW term dominates: cost_h / cost_b ~ _BLACKWELL_BW_SCALE
        ratio = cost_h / cost_b
        assert ratio == pytest.approx(_BLACKWELL_BW_SCALE, rel=0.05)


# ---- Multi-node AG/RS empirical path tests ----


class TestAGRSMultiNodeEmpirical:
    def test_empirical_path_used_for_16_node_ag(self):
        """16-node H100 AllGather should use the empirical path."""
        config = h100_topo_config(num_nodes=16)
        topo = derive_mesh_dim_topo(config, (128,), 0)
        n_bytes = 1 << 33  # 8GB — BW-dominated
        cost = nccl_allgather_cost(n_bytes, topo, config)
        bw = _interp_clamped(_AGRS_MULTI_NODE_BW_POINTS, 16)
        lat = _interp_clamped(_AGRS_MULTI_NODE_LAT_POINTS, 16)
        expected = lat + n_bytes / (1000.0 * bw)  # ramp=1.0 at large sizes
        assert cost == pytest.approx(expected)

    def test_not_used_for_8_nodes(self):
        """8-node AG/RS should use the algo loop, not the empirical path."""
        config = h100_topo_config(num_nodes=8)
        topo = derive_mesh_dim_topo(config, (64,), 0)
        n_bytes = 1 << 30
        cost = nccl_allgather_cost(n_bytes, topo, config)
        # Should NOT match empirical formula (algo loop handles ≤8 nodes)
        bw = _interp_clamped(_AGRS_MULTI_NODE_BW_POINTS, 8)
        lat = _interp_clamped(_AGRS_MULTI_NODE_LAT_POINTS, 8)
        empirical = lat + n_bytes / (1000.0 * bw)
        assert cost != pytest.approx(empirical)

    def test_ag_rs_symmetric(self):
        """AG and RS should have the same cost (symmetric on NVSwitch)."""
        config = h100_topo_config(num_nodes=16)
        topo = derive_mesh_dim_topo(config, (128,), 0)
        n_bytes = 1 << 30
        ag = nccl_allgather_cost(n_bytes, topo, config)
        rs = nccl_reduce_scatter_cost(n_bytes, topo, config)
        assert ag == pytest.approx(rs)

    def test_monotonicity(self):
        """16-node AG cost should increase with message size."""
        config = h100_topo_config(num_nodes=16)
        topo = derive_mesh_dim_topo(config, (128,), 0)
        sizes = [1 << k for k in range(6, 34, 2)]
        times = [nccl_allgather_cost(s, topo, config) for s in sizes]
        for i in range(1, len(times)):
            assert times[i] >= times[i - 1], (
                f"Cost decreased from {sizes[i-1]} to {sizes[i]}: "
                f"{times[i-1]} > {times[i]}"
            )

    def test_ramp_table_shape(self):
        """AGRS ramp table should have 24 entries with values in (0, 1]."""
        assert len(_AGRS_MULTI_NODE_RAMP) == 24
        for f in _AGRS_MULTI_NODE_RAMP:
            assert 0.0 < f <= 1.0

    def test_not_used_without_nvswitch(self):
        """H100 without NVSwitch should use the algo loop."""
        config = h100_topo_config(num_nodes=16, has_nvswitch=False)
        topo = derive_mesh_dim_topo(config, (128,), 0)
        n_bytes = 1 << 30
        cost = nccl_allgather_cost(n_bytes, topo, config)
        assert cost > 0
        assert cost < float("inf")
