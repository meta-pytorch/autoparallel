# Workload adapter provenance

These adapters preserve the real model/config entry points used by the accepted
experiments. They contain configuration glue only; TorchTitan and AutoParallel
remain external source inputs.

- `llama3_2d/perf_configs.py` derives from the final 2026-08-17 paired scaling
  adapter, SHA-256
  `1b9c878176b7437e746cc335e991def46d670e5ff2cc6d20203aed3a306612f6`.
  The permanent version adds only a GraphTrainer-manual registry function that
  calls the existing `_graph_config` with `enable_autoparallel=False`.
- `llama3_3d_legacy/` is copied from the final 2026-08-31 fair DP2×CP2×TP2
  harness. Its five source-file hashes are retained by Git history and every
  packaged payload manifest.
- `llama3_seqlen/perf_configs.py` derives from historical SHA-256
  `c5389e515d0455d65dcc47a30e8a970f7745f5d3ccfd152bbc45da2435881ecf`.
  The permanent version removes only per-yield tensor hashing and JSONL writes.
- `deepseek_v3/perf_configs.py` derives from branch
  `kaijian/deepseek-v3-16b-four-arm-harness`, original SHA-256
  `30beb7f3dca7111ce885346d0ad050517c470863407099be9ad67840b6bef104`.
  The permanent version removes only per-yield tensor hashing and JSONL writes.

Any further integration change requires a new versioned profile and user gate.
