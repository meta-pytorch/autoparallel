# LLaMA 3 replanning workload

This workload compares a fresh ILP placement with exact replay of the canonical
4x8, sequence-2048, local-batch-2 placement. It uses TorchTitan's official
GraphTrainer LLaMA 8B model spec and the `apgt_v1` integration contract. Both
arms consume the same pre-tokenized C4 tensors from memory.

Build the input asset before pinning or validating either campaign:

```bash
python -m workloads.llama3_replanning.prepare_replay \
  --output-root "$REPLAY_ASSET" \
  --tokenizer-dir "$LLAMA_TOKENIZER"
sha256sum "$REPLAY_ASSET/manifest.json"
```

The default builder cases exactly match the paper sweep: sequence lengths 2K,
4K, 8K, and 16K at local batch 2, plus local batches 4 and 8 at sequence 2K.
It downloads shard 0 from the pinned C4 revision, verifies the shard's known
size and SHA-256, and uses TorchTitan's `HuggingFaceTextDataset` packing. The
large asset is intentionally not stored in git.

Replace the source and replay-manifest placeholders in both campaign files,
then run `llama3_8b_replanning_canonical.toml`. Promote
`performance/canonical_fresh/canonical_2k.json` from that successful job into a
read-only asset directory and pin its SHA-256 in
`llama3_8b_replanning_same_mesh.toml`. The input preflight validates the replay
file, tensors, tokenizer tree, and canonical placement before training starts.

The formal phases run 25 steps and measure steps 6-25. A separate six-step
phase records rank-0 Kineto and `TORCH_TRACE`; its timings are not performance
measurements. Fresh and replay arms run sequentially in the same allocation.

The 2x8 and 8x8 cluster-scale rows are not present. See
`CROSS_SCALE_USER_GATE.md` for the exact mesh-compatibility blocker and the API
decision required before adding them.
