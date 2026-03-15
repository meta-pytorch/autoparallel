# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from autoparallel.cost_models.nccl_cost_model import (
    GpuArch,
    NCCLAlgo,
    NCCLFunc,
    NCCLTopoConfig,
    _compute_algo_bw,
    _compute_algo_latency,
    _eligible_algos,
    _nccl_algo_time,
    a100_topo_config,
    derive_mesh_dim_topo,
    detect_nccl_topo_config,
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
        assert topo.bw_intra == pytest.approx(225.0 / 16)
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
            bw_intra=225.0,
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
        bw = _compute_algo_bw(NCCLFunc.ALLREDUCE, NCCLAlgo.RING, topo, config)
        # Ring AR: busBw = nCh * bw * nRanks / (2*(nRanks-1))
        expected = 12 * (87.7 / 12) * 8 / (2 * 7)
        assert bw == pytest.approx(expected)

    def test_ring_bw_allgather(self):
        config = a100_topo_config()
        topo = derive_mesh_dim_topo(config, (8,), 0)
        bw = _compute_algo_bw(NCCLFunc.ALLGATHER, NCCLAlgo.RING, topo, config)
        # Ring AG: busBw = nCh * bw * nRanks / (nRanks-1)
        expected = 12 * (87.7 / 12) * 8 / 7
        assert bw == pytest.approx(expected)

    def test_tree_bw_allreduce(self):
        config = a100_topo_config()
        topo = derive_mesh_dim_topo(config, (8,), 0)
        bw = _compute_algo_bw(NCCLFunc.ALLREDUCE, NCCLAlgo.TREE, topo, config)
        # Tree AR: busBw = min(nCh*bw*0.92, nCh*perChMax) * 0.5
        per_ch_max = 24.0  # Ampere, N1
        raw = 12 * (87.7 / 12)
        capped = min(raw * 0.92, 12 * per_ch_max)
        expected = capped * 0.5
        assert bw == pytest.approx(expected)

    def test_tree_not_available_for_ag(self):
        config = a100_topo_config()
        topo = derive_mesh_dim_topo(config, (8,), 0)
        bw = _compute_algo_bw(NCCLFunc.ALLGATHER, NCCLAlgo.TREE, topo, config)
        assert bw == 0.0

    def test_nvls_bw_hopper(self):
        config = h100_topo_config()
        topo = derive_mesh_dim_topo(config, (8,), 0)
        bw = _compute_algo_bw(NCCLFunc.ALLREDUCE, NCCLAlgo.NVLS, topo, config)
        assert bw > 0

    def test_nvls_bw_ampere_zero(self):
        config = a100_topo_config(has_nvswitch=True)
        topo = derive_mesh_dim_topo(config, (8,), 0)
        # Ampere nvlsEfficiency = 0.0, so intraBw should be 0
        bw = _compute_algo_bw(NCCLFunc.ALLREDUCE, NCCLAlgo.NVLS, topo, config)
        assert bw == 0.0

    def test_ring_latency_single_node(self):
        config = a100_topo_config()
        topo = derive_mesh_dim_topo(config, (8,), 0)
        lat = _compute_algo_latency(NCCLFunc.ALLREDUCE, NCCLAlgo.RING, topo, config)
        # nNodes=1: nInterSteps=0, lat = baseLat + nsteps*intraLat
        nsteps = 2 * 7
        expected = 8.4 + nsteps * 3.4  # NVLink ring lat = 3.4
        assert lat == pytest.approx(expected)

    def test_ring_latency_multi_node(self):
        config = a100_topo_config(num_nodes=2, gpus_per_node=1)
        topo = derive_mesh_dim_topo(config, (2,), 0)
        lat = _compute_algo_latency(NCCLFunc.ALLREDUCE, NCCLAlgo.RING, topo, config)
        # gpus_per_node=1 => ppn=1, n_nodes=2, nRanks=2, nsteps=2, nInterSteps=2
        # netOverhead = 1.0*3 = 3.0, intraLat = max(3.4, 3.0) = 3.4
        # ppn=1 => interLat base = hwLat[NET][TREE] = 14.0
        # lat = 8.4 + (2-2)*3.4 + 2*14.0 = 36.4
        assert lat == pytest.approx(8.4 + 2 * 14.0)

    def test_tree_latency_single_node(self):
        config = a100_topo_config()
        topo = derive_mesh_dim_topo(config, (8,), 0)
        lat = _compute_algo_latency(NCCLFunc.ALLREDUCE, NCCLAlgo.TREE, topo, config)
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
            NCCLFunc.ALLREDUCE, NCCLAlgo.RING, n_bytes, topo, config
        )
        nvls_time = _nccl_algo_time(
            NCCLFunc.ALLREDUCE, NCCLAlgo.NVLS, n_bytes, topo, config
        )
        # NVLS should be competitive or better for large-ish messages on Hopper
        assert nvls_time < ring_time * 2  # At least in the same ballpark

    def test_tree_competes_multi_node_ar(self):
        config = a100_topo_config(num_nodes=4)
        topo = derive_mesh_dim_topo(config, (4,), 0)
        n_bytes = 1 << 20
        ring_time = _nccl_algo_time(
            NCCLFunc.ALLREDUCE, NCCLAlgo.RING, n_bytes, topo, config
        )
        tree_time = _nccl_algo_time(
            NCCLFunc.ALLREDUCE, NCCLAlgo.TREE, n_bytes, topo, config
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


# ---- Tree correction factor ----


class TestTreeCorrection:
    def test_tree_correction_applied(self):
        config = a100_topo_config(num_nodes=2)
        topo = derive_mesh_dim_topo(config, (2,), 0)
        # Small message: correction factor < 1 for some sizes
        small = 1 << 10  # 1KB, log2(1024/64) = 4 -> factor = 0.9
        large = 1 << 18  # 256KB, log2(256K/64) = 12 -> factor = 0.5

        time_small = _nccl_algo_time(
            NCCLFunc.ALLREDUCE, NCCLAlgo.TREE, small, topo, config
        )
        time_large = _nccl_algo_time(
            NCCLFunc.ALLREDUCE, NCCLAlgo.TREE, large, topo, config
        )

        # Both should be finite
        assert time_small < float("inf")
        assert time_large < float("inf")

        # Verify correction is actually applied by comparing with uncorrected
        bw_raw = _compute_algo_bw(NCCLFunc.ALLREDUCE, NCCLAlgo.TREE, topo, config)
        lat = _compute_algo_latency(NCCLFunc.ALLREDUCE, NCCLAlgo.TREE, topo, config)
        time_uncorrected = lat + large / (1000.0 * bw_raw)
        # With correction factor of 0.5, bw is halved => comm time doubles
        assert time_large > time_uncorrected


# ---- Ring plateau ----


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
            NCCLFunc.ALLREDUCE, NCCLAlgo.RING, below, topo, config
        )
        time_above = _nccl_algo_time(
            NCCLFunc.ALLREDUCE, NCCLAlgo.RING, above, topo, config
        )

        # The latency inflation of 1.4x should cause a jump at the threshold
        bw = _compute_algo_bw(NCCLFunc.ALLREDUCE, NCCLAlgo.RING, topo, config)
        lat_base = _compute_algo_latency(
            NCCLFunc.ALLREDUCE, NCCLAlgo.RING, topo, config
        )
        # Below: no inflation
        expected_below = lat_base + below / (1000.0 * bw)
        # Above: lat *= 1.4
        expected_above = lat_base * 1.4 + above / (1000.0 * bw)

        assert time_below == pytest.approx(expected_below)
        assert time_above == pytest.approx(expected_above)


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
