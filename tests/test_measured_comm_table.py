# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import json

import pytest

from autoparallel.cost_models.measured_comm_table import COLLECTIVES, MeasuredCommTable
from autoparallel.cost_models.nccl_cost_model import (
    NCCLFunc,
    derive_mesh_dim_topo,
    h100_topo_config,
    nccl_all_to_all_cost,
    nccl_collective_time,
    nccl_cost_coverage,
)

_SIZES = (1024, 2048, 4096)


def _affine(lat, us_per_byte):
    return _SIZES, tuple(lat + us_per_byte * s for s in _SIZES)


def _table(gpus_per_node=8, keys=((1, 8), (2, 1), (4, 1), (2, 8), (4, 8))):
    """Every collective on each (n_nodes, ppn) key, cost growing with n_nodes."""
    curves = {
        (c, n_nodes, ppn): _affine(10.0 * n_nodes + i, 0.01 * n_nodes * ppn)
        for i, c in enumerate(COLLECTIVES)
        for n_nodes, ppn in keys
    }
    return MeasuredCommTable(gpus_per_node, curves, {"source": "test"})


class TestLookup:
    def test_bytes_interpolation(self):
        table = _table()
        lat, slope = 10.0, 0.08  # allgather, 1 node, ppn 8
        for n_bytes in (*_SIZES, 1536, 3000):
            expected = lat + slope * n_bytes
            assert table.time_us("allgather", 1, 8, n_bytes) == pytest.approx(expected)
        # Held below the smallest size, last segment extended above the largest.
        assert table.time_us("allgather", 1, 8, 8) == pytest.approx(lat + slope * 1024)
        assert table.time_us("allgather", 1, 8, 1 << 20) == pytest.approx(
            lat + slope * (1 << 20)
        )

    def test_node_interpolation_is_linear(self):
        table = _table()
        t2 = table.time_us("reduce_scatter", 2, 1, 2048)
        t4 = table.time_us("reduce_scatter", 4, 1, 2048)
        assert table.time_us("reduce_scatter", 3, 1, 2048) == pytest.approx(
            (t2 + t4) / 2
        )

    def test_node_extrapolation_from_two_largest(self):
        table = _table()
        t2 = table.time_us("allreduce", 2, 8, 4096)
        t4 = table.time_us("allreduce", 4, 8, 4096)
        assert table.time_us("allreduce", 16, 8, 4096) == pytest.approx(
            t4 + (16 - 4) * (t4 - t2) / 2
        )

    def test_extrapolation_never_decreases(self):
        curves = {
            ("allgather", 2, 1): _affine(50.0, 0.02),
            ("allgather", 4, 1): _affine(40.0, 0.02),
        }
        table = MeasuredCommTable(8, curves)
        assert table.time_us("allgather", 64, 1, 2048) == table.time_us(
            "allgather", 4, 1, 2048
        )

    def test_uncovered_keys_raise(self):
        table = _table(keys=((1, 8), (4, 1), (8, 1), (2, 8)))
        for n_nodes, ppn in [
            (1, 4),  # single-node keys are never derived
            (2, 1),  # below the smallest measured node count
            (4, 2),  # untabulated ppn
            (8, 8),  # one measured node count cannot be extrapolated
        ]:
            with pytest.raises(KeyError, match="No measured allgather"):
                table.time_us("allgather", n_nodes, ppn, 1024)

    def test_lower_envelope(self):
        sizes = (64, 128, 256, 512, 1024)
        curves = {("allgather", 1, 8): (sizes, (50.0, 30.0, 40.0, 35.0, 60.0))}
        table = MeasuredCommTable(8, curves)
        assert [table.time_us("allgather", 1, 8, s) for s in sizes] == [
            30.0,
            30.0,
            35.0,
            35.0,
            60.0,
        ]
        assert table.curves == curves

    def test_coverage(self):
        table = _table()
        assert table.coverage(1, 8) == "measured"
        assert table.coverage(4, 1) == "measured"
        assert table.coverage(3, 1) == "interpolated"
        assert table.coverage(8, 1) == "extrapolated"
        assert table.coverage(1, 4) == "missing"
        partial = MeasuredCommTable(
            8, {k: v for k, v in table.curves.items() if k != ("all_to_all", 2, 1)}
        )
        assert partial.coverage(2, 1) == "missing"
        assert partial.coverage(3, 1) == "missing"

    def test_single_rank_costs_nothing(self):
        table = _table()
        assert table.coverage(1, 1) == "no_communication"
        for collective in COLLECTIVES:
            assert table.time_us(collective, 1, 1, 1 << 20) == 0.0
        config = h100_topo_config(num_nodes=8, measured_table=table)
        # The size-1 dimension of (4, 8, 1) is a single rank.
        assert nccl_cost_coverage(config, (4, 8, 1))[2] == "no_communication"

    @pytest.mark.parametrize(
        "curve, match",
        [
            ((("broadcast", 1, 8), _affine(1.0, 0.1)), "Unknown collective"),
            ((("allgather", 1, 8), ((1024,), (1.0,))), "at least two"),
            ((("allgather", 1, 8), ((2048, 1024), (1.0, 2.0))), "strictly increasing"),
        ],
    )
    def test_invalid_curves(self, curve, match):
        with pytest.raises(ValueError, match=match):
            MeasuredCommTable(8, dict([curve]))


class TestSerialization:
    def test_round_trip(self, tmp_path):
        table = _table()
        path = str(tmp_path / "table.json")
        table.save(path)
        with open(path) as f:
            data = json.load(f)
        assert data["schema_version"] == 1
        assert len(data["curves"]) == len(table.curves)
        loaded = MeasuredCommTable.load(path)
        assert loaded.gpus_per_node == table.gpus_per_node
        assert loaded.curves == table.curves
        assert loaded.provenance == table.provenance

    def test_rejects_unknown_schema(self, tmp_path):
        path = tmp_path / "table.json"
        path.write_text(json.dumps({"schema_version": 2, "curves": []}))
        with pytest.raises(ValueError, match="schema_version"):
            MeasuredCommTable.load(str(path))


_NCCL_TESTS_OUTPUT = """\
# Collective test starting: {binary}
# nThread 1 nGpus 1 minBytes 8 maxBytes 4096 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid   1001 on host-a device  0 [0000:1a:00] NVIDIA H100 80GB HBM3
#  Rank  1 Group  0 Pid   1002 on host-a device  1 [0000:40:00] NVIDIA H100 80GB HBM3
#  Rank  2 Group  0 Pid   2001 on {third_host} device  0 [0000:1a:00] NVIDIA H100 80GB HBM3
#  Rank  3 Group  0 Pid   2002 on host-b device  1 [0000:40:00] NVIDIA H100 80GB HBM3
#
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
           0             0     float    none      -1     0.29    0.00    0.00      0     0.26    0.00    0.00      0
        2048           128     float    none      -1    {t0:.2f}    0.10    0.08      0    19.90    0.10    0.08      0
        4096           256     float    none      -1    {t1:.2f}    0.19    0.14  {wrong}    20.10    0.20    0.15    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 0.11
#
# Collective test concluded: {binary}
"""


def _write_output(
    tmp_path, name, binary, third_host="host-b", wrong="0", times=(20.5, 21.25)
):
    path = tmp_path / name
    path.write_text(
        _NCCL_TESTS_OUTPUT.format(
            binary=binary, third_host=third_host, wrong=wrong, t0=times[0], t1=times[1]
        )
    )
    return str(path)


class TestFromNcclTests:
    def test_parse(self, tmp_path):
        paths = [
            _write_output(tmp_path, "ag.txt", "all_gather_perf_mpi"),
            _write_output(tmp_path, "a2a.txt", "alltoall_perf"),
        ]
        table = MeasuredCommTable.from_nccl_tests(paths, 2, {"nccl": "2.27.5"})
        assert table.curves == {
            ("allgather", 2, 2): ((2048, 4096), (20.5, 21.25)),
            ("all_to_all", 2, 2): ((2048, 4096), (20.5, 21.25)),
        }
        assert table.provenance == {"nccl": "2.27.5"}

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"binary": "broadcast_perf"}, "unsupported test"),
            ({"binary": "all_gather_perf", "wrong": "3"}, "validation errors"),
            ({"binary": "all_gather_perf", "third_host": "host-a"}, "evenly"),
        ],
    )
    def test_rejects_bad_runs(self, tmp_path, kwargs, match):
        path = _write_output(tmp_path, "run.txt", **kwargs)
        with pytest.raises(ValueError, match=match):
            MeasuredCommTable.from_nccl_tests([path], 2)

    def test_repeated_runs_take_the_median(self, tmp_path):
        paths = [
            _write_output(tmp_path, f"run{i}.txt", "all_gather_perf", times=times)
            for i, times in enumerate([(20.0, 40.0), (26.0, 30.0), (21.0, 33.0)])
        ]
        table = MeasuredCommTable.from_nccl_tests(paths, 2)
        assert table.curves == {("allgather", 2, 2): ((2048, 4096), (21.0, 33.0))}

    def test_repeated_runs_need_the_same_sizes(self, tmp_path):
        paths = [
            _write_output(tmp_path, f"run{i}.txt", "all_gather_perf") for i in range(2)
        ]
        with open(paths[1]) as f:
            text = f.read()
        with open(paths[1], "w") as f:
            f.write(
                text.replace("        4096           256", "        8192           512")
            )
        with pytest.raises(ValueError, match="different sizes"):
            MeasuredCommTable.from_nccl_tests(paths, 2)


class TestNCCLCostModelIntegration:
    def test_table_replaces_model(self):
        table = _table()
        config = h100_topo_config(num_nodes=4, measured_table=table)
        for dim, (n_nodes, ppn) in enumerate([(4, 1), (1, 8)]):
            topo = derive_mesh_dim_topo(config, (4, 8), dim)
            for func, name in [
                (NCCLFunc.ALLGATHER, "allgather"),
                (NCCLFunc.REDUCESCATTER, "reduce_scatter"),
                (NCCLFunc.ALLREDUCE, "allreduce"),
            ]:
                assert nccl_collective_time(func, 3000, topo, config) == (
                    table.time_us(name, n_nodes, ppn, 3000)
                )
            assert nccl_all_to_all_cost(3000, topo, config) == table.time_us(
                "all_to_all", n_nodes, ppn, 3000
            )

    def test_coverage(self):
        config = h100_topo_config(num_nodes=4, measured_table=_table())
        assert nccl_cost_coverage(config, (4, 8)) == ("measured",) * 3
        assert nccl_cost_coverage(config, (24, 8)) == (
            "extrapolated",
            "measured",
            "extrapolated",
        )
        # (3 nodes, ppn 2), (1 node, ppn 4), then the flattened (3 nodes, ppn 8).
        assert nccl_cost_coverage(config, (6, 4)) == (
            "missing",
            "missing",
            "interpolated",
        )

    def test_gpus_per_node_must_match(self):
        with pytest.raises(ValueError, match="measured with 8 GPUs per node"):
            h100_topo_config(gpus_per_node=4, measured_table=_table())

    def test_collective_comm_cost_uses_table(self, monkeypatch):
        import autoparallel.cost_models.collective_runtime_estimation as cre

        table = _table()
        monkeypatch.setattr(cre, "_nccl_topo_config", cre.get_nccl_topo_config())
        cre.set_nccl_topo_config(h100_topo_config(num_nodes=4, measured_table=table))
        assert cre.collective_comm_cost("all_to_all", 3000, (4, 8), 0) == (
            table.time_us("all_to_all", 4, 1, 3000)
        )
        assert cre.collective_comm_cost("allreduce", 3000, (4, 8), 1) == (
            table.time_us("allreduce", 1, 8, 3000)
        )

    def test_autoparallel_rejects_uncovered_mesh(self, device_mesh_1d):
        from unittest.mock import patch

        import torch
        from torch import nn

        import autoparallel.cost_models.collective_runtime_estimation as cre
        from autoparallel.api import AutoParallel

        def input_fn():
            return (torch.rand(256, 8, device="cuda"),)

        # 256 ranks at 8 GPUs/node need (32 nodes, ppn 8), which needs at
        # least two multi-node counts for ppn 8.
        config = h100_topo_config(
            num_nodes=32, measured_table=_table(keys=((1, 8), (2, 8)))
        )
        previous = cre.get_nccl_topo_config()
        with patch("torch.cuda.current_device", return_value=0):
            with torch.device("meta"):
                model = nn.Linear(8, 8)
            autop = AutoParallel(model, input_fn, device_mesh_1d, cost_model=config)
            with pytest.raises(ValueError, match=r"does not cover mesh \(256,\)"):
                autop.__enter__()
        assert cre.get_nccl_topo_config() is previous
