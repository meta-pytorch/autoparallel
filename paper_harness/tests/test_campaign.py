from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from harness.campaign import CampaignError, load_campaign, settings_to_argv
from harness.cli import _submitted_job_id
from harness.parity import validate_pair
from harness.sources import inspect_source


REPO_ROOT = Path(__file__).resolve().parents[1]


class CampaignTests(unittest.TestCase):
    def test_submitted_job_id_uses_torchx_mast_handle(self) -> None:
        output = """Current Session ID: session-id

mast_conda://torchx/llama3-paper-wangkj-grfhpnvn
launched app: `mast_conda://torchx/llama3-paper-wangkj-grfhpnvn`
"""
        self.assertEqual(
            _submitted_job_id(output), "llama3-paper-wangkj-grfhpnvn"
        )

    def test_reproduction_campaigns_are_fixed_to_32_gpus(self) -> None:
        expected_init_timeouts = {
            "repro_llama3_8b_2d_32gpu.toml": 1200,
            "repro_llama3_8b_seqlen_4k_32gpu.toml": 1200,
            "repro_muse_glimmer_30b_32gpu.toml": 1800,
        }
        for path in sorted((REPO_ROOT / "campaigns").glob("repro_*.toml")):
            with self.subTest(path=path.name):
                campaign = load_campaign(path)
                self.assertEqual(campaign.world_size, 32)
                self.assertIn("primary", campaign.raw["measurement"])
                self.assertIn("acceptance", campaign.raw["comparison"])
                self.assertEqual(
                    campaign.raw["comm"]["init_timeout_seconds"],
                    expected_init_timeouts[path.name],
                )
        llama = load_campaign(REPO_ROOT / "campaigns/repro_llama3_8b_2d_32gpu.toml")
        self.assertEqual(llama.arm_names, {"tt_main_tp", "apgt"})

    def test_gate_mode_keeps_all_arms_for_interleaved_formal_phases(self) -> None:
        campaign = load_campaign(
            REPO_ROOT / "campaigns/repro_llama3_8b_seqlen_4k_32gpu.toml",
            mode="gate",
        )
        self.assertEqual(
            [phase.arms for phase in campaign.phases],
            [("fresh", "replay_2k"), ("fresh", "replay_2k")],
        )

    def test_arm_override_cannot_break_the_world_mesh(self) -> None:
        source = (REPO_ROOT / "campaigns/repro_llama3_8b_2d_32gpu.toml").read_text()
        invalid = source.replace(
            "[arms.environment]\nBENCHMARK_CONFIGURATION = \"torchtitan_baseline\"",
            "[arms.overrides]\n\"parallelism.tensor_parallel_degree\" = 1\n"
            "[arms.environment]\nBENCHMARK_CONFIGURATION = \"torchtitan_baseline\"",
            1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "campaign.toml"
            path.write_text(invalid)
            with self.assertRaisesRegex(CampaignError, "parallel mesh product"):
                load_campaign(path)

    def test_canonical_matrix_points(self) -> None:
        cases = {
            "muse_glimmer_30b_scaling.toml": ("16gpu", "32gpu", "64gpu", "128gpu"),
            "llama3_8b_2d_scaling.toml": ("8gpu", "16gpu", "32gpu", "64gpu", "128gpu"),
            "llama3_8b_seqlen.toml": ("2k", "4k", "8k", "16k", "32k"),
            "deepseek_v3_16b.toml": ("16gpu", "32gpu"),
        }
        for filename, points in cases.items():
            for point in points:
                with self.subTest(filename=filename, point=point):
                    campaign = load_campaign(REPO_ROOT / "campaigns" / filename, point=point)
                    self.assertGreater(campaign.world_size, 0)
                    self.assertTrue(campaign.phases)
        legacy = load_campaign(REPO_ROOT / "campaigns/llama3_8b_3d_legacy.toml")
        self.assertEqual(legacy.world_size, 8)

    def test_matrix_requires_an_explicit_point(self) -> None:
        with self.assertRaisesRegex(CampaignError, "requires --point"):
            load_campaign(REPO_ROOT / "campaigns/muse_glimmer_30b_scaling.toml")

    def test_gate_mode_is_functional_and_trace_only(self) -> None:
        campaign = load_campaign(
            REPO_ROOT / "campaigns/muse_glimmer_30b_scaling.toml",
            point="16gpu",
            mode="gate",
        )
        self.assertEqual(campaign.name, "muse-glimmer-30b-gt-manual-vs-apgt-16gpu-gate")
        self.assertEqual(
            [(phase.name, phase.kind) for phase in campaign.phases],
            [("functional", "correctness"), ("trace_smoke", "trace")],
        )
        for phase in campaign.phases:
            self.assertEqual(phase.overrides["training.steps"], 2)

    def test_native_argument_rendering(self) -> None:
        self.assertEqual(
            settings_to_argv(
                {
                    "training.steps": 2,
                    "profiler.enable_profiling": False,
                    "compile.components": ["model", "loss"],
                }
            ),
            [
                "--compile.components",
                "model,loss",
                "--profiler.no-enable-profiling",
                "--training.steps",
                "2",
            ],
        )

    def test_autoparallel_solver_argument_rendering(self) -> None:
        self.assertEqual(
            settings_to_argv(
                {
                    "compile.autoparallel_solver": "approx",
                    "compile.autoparallel_fast_build": False,
                    "compile.autoparallel_lazy_costs": "eager",
                    "compile.autoparallel_strategy_radius": 1,
                    "compile.autoparallel_optimality_check": True,
                    "compile.autoparallel_approx_candidate_limit": 64,
                    "compile.autoparallel_approx_bp_iters": 80,
                    "compile.autoparallel_approx_bp_tol": 0.002,
                    "compile.autoparallel_approx_max_sweeps": 6,
                    "compile.autoparallel_approx_max_time_s": 30.0,
                    "compile.autoparallel_approx_star_passes": 3,
                    "compile.autoparallel_approx_max_star_children": 16,
                    "compile.autoparallel_approx_group_domain_limit": 256,
                }
            ),
            [
                "--compile.autoparallel-approx-bp-iters",
                "80",
                "--compile.autoparallel-approx-bp-tol",
                "0.002",
                "--compile.autoparallel-approx-candidate-limit",
                "64",
                "--compile.autoparallel-approx-group-domain-limit",
                "256",
                "--compile.autoparallel-approx-max-star-children",
                "16",
                "--compile.autoparallel-approx-max-sweeps",
                "6",
                "--compile.autoparallel-approx-max-time-s",
                "30.0",
                "--compile.autoparallel-approx-star-passes",
                "3",
                "--compile.no-autoparallel-fast-build",
                "--compile.autoparallel-lazy-costs",
                "eager",
                "--compile.autoparallel-optimality-check",
                "--compile.autoparallel-solver",
                "approx",
                "--compile.autoparallel-strategy-radius",
                "1",
            ],
        )

    def test_parity_rejects_undeclared_difference(self) -> None:
        with self.assertRaisesRegex(CampaignError, "outside declared"):
            validate_pair(
                "left",
                {"training": {"seq_len": 2048}},
                "right",
                {"training": {"seq_len": 4096}},
                ["compile.enable_autoparallel"],
            )

    def test_parity_ignores_process_local_callable_addresses(self) -> None:
        result = validate_pair(
            "left",
            {"model": "<function init at 0x1234abcd>"},
            "right",
            {"model": "<function init at 0xfeed5678>"},
            [],
        )
        self.assertEqual(result["observed_differences"], [])


class SourceTests(unittest.TestCase):
    def test_dirty_source_is_fail_closed_or_snapshotted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            evidence = Path(temporary) / "evidence"
            root.mkdir()
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.com"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Harness Test"], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "remote",
                    "add",
                    "origin",
                    "https://github.com/example/fixture.git",
                ],
                check=True,
            )
            (root / "tracked.txt").write_text("clean\n")
            subprocess.run(["git", "-C", str(root), "add", "tracked.txt"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)
            head = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip()
            spec = {
                "commit": head,
                "remote": "https://github.com/example/fixture.git",
                "dirty_policy": "forbid",
            }
            self.assertFalse(inspect_source("fixture", root, spec)["dirty"])
            (root / "tracked.txt").write_text("dirty\n")
            with self.assertRaisesRegex(CampaignError, "dirty_policy"):
                inspect_source("fixture", root, spec)
            spec["dirty_policy"] = "snapshot"
            record = inspect_source("fixture", root, spec, evidence_dir=evidence)
            self.assertTrue(record["dirty"])
            self.assertTrue((evidence / "fixture/tracked.diff").is_file())


if __name__ == "__main__":
    unittest.main()
