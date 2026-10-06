# Measuring Communication Costs for Your Cluster

AutoParallel's default cost model prices collectives with an analytic NCCL
model calibrated on one H100 cluster. If your GPUs, network, or NCCL version
differ, you can measure collective times on your own cluster and give
AutoParallel a table of measurements instead. It then prices every
collective from the table.

The procedure below measures a cluster once. The resulting JSON file then
works for every power-of-two `DeviceMesh` up to the largest node count you
measured.

## What gets measured

A table maps `(collective, n_nodes, ppn)` to a curve of `(bytes, time_us)`
points:

- **collective:** all-gather, reduce-scatter, all-reduce, and all-to-all,
  measured with nccl-tests' `all_gather_perf`, `reduce_scatter_perf`,
  `all_reduce_perf`, and `alltoall_perf`. Every key needs all four.
- **n_nodes, ppn:** the communicator a mesh dimension runs on: how many
  nodes it spans and how many of its ranks share a node.
- **bytes:** the size column of nccl-tests, which uses the same convention
  as AutoParallel: output bytes for all-gather, input bytes for
  reduce-scatter, the buffer for all-reduce, and the per-rank buffer for
  all-to-all.

AutoParallel assumes ranks are laid out node by node, and maps each mesh
dimension to a key. With `G` GPUs per node, dimension `i` of size `d_i` has
`ppn = max(1, min(G / (product of the later dimensions), d_i))` and
`n_nodes = d_i / ppn`. Collectives fused across dimensions run on the
flattened mesh, so it needs a key too. For 8 GPUs per node:

| Mesh | dim 0 | dim 1 | flattened |
|---|---|---|---|
| `(8, 8)` | (8 nodes, ppn 1) | (1 node, ppn 8) | (8 nodes, ppn 8) |
| `(16, 4)` | (8 nodes, ppn 2) | (1 node, ppn 4) | (8 nodes, ppn 8) |
| `(32, 2)` | (8 nodes, ppn 4) | (1 node, ppn 2) | (8 nodes, ppn 8) |

### The canonical grid

Rather than measuring per mesh, measure every key a power-of-two mesh can
produce:

- **single node:** `ppn` = 2, 4, ..., `G`
- **multi-node:** `n_nodes` = 2, 4, 8, ..., `N`, each with `ppn` = 1, 2, 4,
  ..., `G`, where `N` is the node count of your largest job
- **sizes:** 8 B to 8 GB, doubling, which covers the multi-GB buckets the
  compiler builds for FSDP collectives

Lookups between measured node counts are interpolated linearly in `n_nodes`.
Above `N` they are extrapolated from the two largest counts, which is
reported as `extrapolated` and becomes unreliable beyond about 2x `N`.
Single-node keys are never derived, so they must be measured.

## 1. Build nccl-tests against PyTorch's NCCL

Algorithm selection and tuning change between NCCL releases, so measure with
the NCCL your training uses:

```bash
python -c "import torch; print(torch.cuda.nccl.version())"
```

PyTorch wheels ship NCCL as the `nvidia-nccl` pip package, which includes
the headers. It only ships the versioned `libnccl.so.2`, so give the
nccl-tests build a directory with a `libnccl.so` link:

```bash
NCCL_PIP=$(python -c "import nvidia.nccl; print(list(nvidia.nccl.__path__)[0])")
mkdir -p nccl-home/lib
ln -s "$NCCL_PIP/include" nccl-home/include
ln -s "$NCCL_PIP/lib/libnccl.so.2" nccl-home/lib/libnccl.so

git clone https://github.com/NVIDIA/nccl-tests.git
make -C nccl-tests -j MPI=1 MPI_HOME=/path/to/mpi CUDA_HOME=/path/to/cuda \
    NCCL_HOME=$PWD/nccl-home NAME_SUFFIX=_mpi

export LD_LIBRARY_PATH=$NCCL_PIP/lib:/path/to/cuda/lib64:$LD_LIBRARY_PATH
```

`MPI=1` is needed for multi-node runs. If your PyTorch build uses a
different NCCL, build against that release instead. Check that a run prints
the expected version with `NCCL_DEBUG=VERSION`.

## 2. Run the sweep

`autoparallel/tools/nccl_tests_sweep.sh` runs the grid inside an exclusive
Slurm allocation of `N` nodes:

```bash
NCCL_TESTS_BUILD=$PWD/nccl-tests/build OUT_DIR=results RUN_TAG=run1 \
    sbatch --nodes=32 --gpus-per-node=8 --ntasks-per-node=8 --exclusive \
    autoparallel/tools/nccl_tests_sweep.sh
```

Each run is one process per GPU (`srun --ntasks-per-node=<ppn>
--gpus-per-node=<ppn> ... -g 1`) with `-b 8 -e 8G -f 2 -w 5 -n 20`, and its
stdout goes to one file per key. `DRY_RUN=1` prints the commands instead,
and the header of the script lists the other settings. If `N` is not a
power of two, add it explicitly, e.g. `NODE_COUNTS="1 2 4 8 16 24"`.

Guidelines:

- **Use the NCCL environment of your training jobs** (`NCCL_IB_HCA`,
  `NCCL_SOCKET_IFNAME`, and so on). Do not set `NCCL_ALGO` or `NCCL_PROTO`
  unless your training does.
- **Repeat the sweep on about three separate allocations** with different
  `RUN_TAG`s. On the cluster the default model was calibrated on, the same
  keys measured on two allocations differed by a median of 20-27% at small
  sizes for some node counts, and agreed within a few percent for others,
  depending on where the nodes sat in the network. The converter takes the
  median of repeated runs.
- **Do not set `NCCL_TESTS_SPLIT_MASK`.** The converter reads the
  communicator from the list of ranks in each run's header.

## 3. Convert the results to JSON

```bash
python -m autoparallel.tools.nccl_tests_to_json results/*.txt \
    --gpus-per-node 8 \
    --provenance "gpu=NVIDIA H100 80GB HBM3" \
    --provenance nccl_version=2.30.7 \
    --provenance "network=8x400G InfiniBand per node, rail-optimized" \
    --provenance date=2026-10-06 \
    -o my_cluster.json
```

The converter reads the collective, node count, and ranks per node from each
file, keeps the out-of-place times, drops sizes too small to split across
the ranks, and rejects runs that report validation errors. Provenance
entries are free-form; record whatever you need to trace the table back to
its measurements.

## 4. Use the table

```python
from autoparallel.api import AutoParallel
from autoparallel.cost_models import (
    MeasuredCommTable,
    h100_topo_config,
    nccl_cost_coverage,
)

table = MeasuredCommTable.load("my_cluster.json")
config = h100_topo_config(num_nodes=16, measured_table=table)
print(nccl_cost_coverage(config, (16, 8)))  # ('measured', 'measured', 'measured')

with AutoParallel(model, input_fn, mesh, mp_policy, cost_model=config) as autop:
    ...
```

The table replaces the model for every collective, so only the config's
`gpus_per_node` matters, and it must match the table's. If the table cannot
price a dimension of the mesh, `AutoParallel` raises an error that lists the
coverage of each dimension instead of mixing measured and modeled costs.
Extrapolated dimensions produce a warning.

If your script also schedules the compiled graph with
`make_custom_runtime_estimation`, as `examples/example_llama3.py` does,
install the same config with `set_nccl_topo_config(config)` (from
`autoparallel.cost_models.collective_runtime_estimation`) so the scheduler
prices collectives from the table too.

## Limitations

- **Isolated collectives.** The measurements are collectives running alone.
  Contention with computation or with other collectives running at the same
  time, and overlap between them, are not modeled.
- **Idle GPUs during runs with `ppn < G`.** In training, every GPU on a node
  runs a sibling communicator at the same time; in the sweep, the other GPUs
  are idle. This matters little on nodes with one NIC per GPU, but makes the
  table optimistic on nodes that share NICs or on oversubscribed networks.
- **Extrapolation carries trends forward.** On the H100 cluster the default
  model was calibrated on, all-gather bandwidth with 8 ranks per node halved
  between 16 and 32 nodes, and extrapolating beyond 32 nodes halves it again.
  Measure up to the size of your largest job.
- **Measurements age.** Re-measure after NCCL upgrades or network changes.
