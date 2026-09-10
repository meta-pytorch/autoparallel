# Submission stack provenance

The harness originated from `AlbedoWang/AutoParallel-Harness` commit `854c63e69e1f2e515ad6c14e799b3c241dcbfd93`. It is checked into `paper_harness/` so the paper branch records the exact launcher and analysis code without modifying either runtime source checkout.

## AutoParallel

The `kaijian/paper-submission` branch is an explicit linear stack:

1. PR523 stack head: `946f0643bea1e40165a63696b3e18982aea07838`.
2. Context-parallel integration head: `ec819dbe1d0827e3b52a0225ec1fcb47f8bd9732`.
3. N-D ordered FSDP shard-layout fix: `6f649f835e7d531be7d05c09927ed5cef1352b29`.
4. NCCL topology-cost fix, cherry-picked from `b8ace2a55d465787840aa41466f50be4fd9f73c1` as `34085887102c6d5a627ca8229f92e9ae44ed1426`.
5. The permanent harness snapshot and its documentation under `paper_harness/`.

The immutable experiment source is `5102d629c0a97ec604b12c328b40147d214ecbe7`. Current AutoParallel `main` is not merged implicitly.

## TorchTitan

The companion `AlbedoWang/torchtitan` branch `kaijian/paper-submission` is pinned at `6ced255cc55dec8a469bba2584c6c2efb35122ac`, rebased on official TorchTitan main `f93fd4ccff855b7e2a8f7d959e6532a9fa743f9e`. Its stack contains the GraphTrainer AutoParallel integration, LLaMA3/DeepSeek V3/Muse integration, solver configuration bridge, AP-specific Inductor settings, FQN restoration, and overlap-ordering fixes.

The final compatibility commit migrates the integration to latest TorchTitan's token-based training configuration and keeps the cross-entropy path valid for both current token-major inputs and historical fixed-shape LLaMA batches.

## Pinning policy

`experiment_lock.toml` is authoritative for both source commits and `torchtitan_conda_prod:946`. Every campaign must repeat those exact pins, and validation rejects dirty trees, source drift, runtime drift, or a changed frozen harness core. A later source or harness change requires a new explicit lock/version rather than editing a packaged attempt.
