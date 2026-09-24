# NCCL cost-model calibration

This benchmark produces the `h100_nvswitch_roce_400g` communication-cost
profile for 8×H100 NVSwitch nodes connected by RoCE 400G. The profile is
opt-in and leaves AutoParallel's default NCCL model unchanged.

The formal campaign measures 1, 2, 4, 8, 16, and 32 nodes, with three
replicates per allocation. Replicates 1 and 2 fit the model; replicate 3 is
held out for validation before the final table is refit from all replicates.
Each allocation measures 1, 2, 4, and 8 ranks per node, message sizes from
4 KiB through 1 GiB, and auto-selected AllGather, ReduceScatter, AllReduce,
and AllToAll. It also records forced NCCL algorithm/protocol combinations for
diagnosis.

Run a short gate before launching the formal campaign:

```bash
./launch.sh 2 1 true auto
```

Launch the full 18-job campaign with:

```bash
./launch_campaign.sh
```

`CONDA_FBPKG_ID` can override the default TorchTitan environment package.
Record the returned job IDs in a JSON manifest with this shape:

```json
{"formal": {"1": ["job-r1", "job-r2", "job-r3"], "2": []}}
```

After downloading each job's output into `RESULTS_ROOT/JOB_ID`, generate the
validation report, JSON artifact, and Python model table with:

```bash
python tools/fit_profile.py \
  --manifest campaign_jobs.json \
  --results-root RESULTS_ROOT \
  --json-output generated/h100_nvswitch_roce_400g.json \
  --python-output ../../autoparallel/cost_models/h100_nvswitch_roce_400g.py
```

The fitter targets the maximum across ranks of each rank's median CUDA-event
latency. For every collective and `(nodes, ranks-per-node)` topology, it emits
a startup latency, peak bandwidth, and a 24-point log-message-size efficiency
ramp. Enable the generated model with:

```bash
AUTOPARALLEL_NCCL_COST_MODEL_PROFILE=h100_nvswitch_roce_400g
```
