# Submission stack provenance

The harness is vendored from `AlbedoWang/AutoParallel-Harness` commit `854c63e69e1f2e515ad6c14e799b3c241dcbfd93`. The snapshot is kept under `paper_harness/` so the paper branch records the exact launcher and analysis code without making TorchTitan or AutoParallel runtime sources mutable inputs.

## AutoParallel

The `kaijian/paper-submission` branch is an explicit linear stack:

1. PR523 stack head: `946f0643bea1e40165a63696b3e18982aea07838`.
2. Context-parallel integration head: `ec819dbe1d0827e3b52a0225ec1fcb47f8bd9732`.
3. N-D ordered FSDP shard-layout fix: `6f649f835e7d531be7d05c09927ed5cef1352b29`.
4. NCCL topology-cost fix, cherry-picked from `b8ace2a55d465787840aa41466f50be4fd9f73c1` as `34085887102c6d5a627ca8229f92e9ae44ed1426`.
5. The permanent harness snapshot and its documentation under `paper_harness/`.

Current AutoParallel `main` is not merged implicitly. This keeps the paper stack attributable and lets a campaign pin any other clean AutoParallel commit instead.

## TorchTitan

The companion `AlbedoWang/torchtitan` branch `kaijian/paper-submission` ends at `d0ced23d8b41895bfb2e2d8a0c0305d18ed08ccb`. Its stack contains the prior GraphTrainer AutoParallel integration, the validated LLaMA3/DeepSeek V3 3D integration snapshot, and the solver configuration bridge consumed by this harness.

The snapshot-restoration commit is `5f032d1337ddf2cf6ee1c334a3dd1a9816490ed3`. It reconstructs the final retained 3D integration source atop the available Git history because the original local final commit object was no longer available. The next commit, `d0ced23d8b41895bfb2e2d8a0c0305d18ed08ccb`, adds only the typed solver configuration and shared forwarding logic.

## Pinning policy

Campaigns pin immutable source commits and receive clean checkout paths at validation/package time. Existing historical campaign pins are provenance records and are not rewritten to the submission stack. New paper campaigns should pin the desired AutoParallel and TorchTitan submission commits explicitly, and any later source change should produce new commit pins rather than editing either checkout in place.
