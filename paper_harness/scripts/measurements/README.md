# Reproducible measurement scripts

These scripts compose the existing permanent-harness lifecycle. They do not
define metrics, training settings, scheduler resources, or comparison rules.
Those remain in the selected campaign and in `harness.cli`.

`measurement.py` exposes five stages:

1. `prepare` calls the existing `harness.cli package` path (which performs the
   full source, asset, config, and parity validation) and then `render-mast`.
   The attempt path must not exist.
2. `submit` calls the existing sealed-attempt submission, immediately sets and
   reads back `CRITICAL/99`, fetches the actual submitted workspace and payload
   packages, compares them to the audited inputs, and reruns the existing exact
   package preflight.
3. `monitor` stores structured MAST status/history polls and fails closed on a
   retry, new attempt/epoch, failed or shrunk task, or elastic allocation. With
   `--run-root`, it also records every phase/arm marker and allocation-record
   count. Terminal stdout, stderr, definition, priority, history, and status are
   retained.
4. `retrieve` requires the exact `ws://` URI from the submitted `mount.sh`,
   fetches `oil.oilfs:stable` into the attempt, verifies two identical source
   manifests, copies to the fresh canonical `ATTEMPT/run`, verifies the local
   manifest/checksum, and unmounts.
5. `analyze` invokes the analyzer from the sealed harness payload and requires
   an explicit tlparse executable. It adds no metric or acceptance logic.

The portable end-to-end wrapper fixes only the existing campaign and point for
LLaMA3 8B A/B/C at DP4 x TP8 (`32gpu`). It is preparation-only unless
`--submit` is explicitly present:

```bash
paper_harness/scripts/measurements/run_llama3_4x8.sh \
  --attempt /path/to/task/attempts/001 \
  --python /path/to/compatible/python \
  --torchtitan-root /path/to/clean/torchtitan \
  --autoparallel-root /path/to/clean/autoparallel \
  --tokenizer-root /path/to/tokenizer \
  --replay-root /path/to/replay \
  --seed-checkpoint-root /path/to/seed-checkpoint
```

Add the following only when a MAST run is authorized:

```bash
  --submit \
  --oilfs-uri ws://workspace/from-the-exact-submitted-mount \
  --oilfs-user "$USER" \
  --tlparse-bin /path/to/tlparse
```

Equivalent environment variables are `ATTEMPT_ROOT`, `HARNESS_PYTHON`,
`TORCHTITAN_ROOT`, `AUTOPARALLEL_ROOT`, `LLAMA_TOKENIZER_ROOT`,
`LLAMA_REPLAY_ROOT`, `LLAMA_SEED_CHECKPOINT_ROOT`, `OILFS_URI`, `OILFS_USER`,
`TLPARSE_BIN`, `MEASUREMENT_MODE`, and `POLL_INTERVAL_SECONDS`.

Use `--dry-run` to print the composed commands without validating, packaging,
or submitting anything. The campaign remains the sole source for the three
arms, batch/sequence settings, model, data, compile stack, profiler phases,
hardware, and comparison contract.
