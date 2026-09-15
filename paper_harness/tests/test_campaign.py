from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from harness.campaign import CampaignError, load_campaign, settings_to_argv
from harness.experiment_lock import load_experiment_lock
from harness.parity import validate_pair
from harness.settings import load_run_settings
from harness.sources import inspect_source

REPO_ROOT = Path(__file__).resolve().parents[1]


class CampaignTests(unittest.TestCase):
    def test_only_canonical_campaigns_are_active(self) -> None:
        self.assertEqual(
            {path.name for path in (REPO_ROOT / "campaigns").glob("*.toml")},
            {
                "deepseek_v3_16b.toml",
                "llama3_8b_2d.toml",
                "llama3_8b_3d.toml",
                "llama3_8b_seqlen.toml",
                "muse_glimmer_30b.toml",
            },
        )

    def test_all_settings_resolve_to_the_global_lock(self) -> None:
        lock = load_experiment_lock()
        for setting in load_run_settings().values():
            with self.subTest(model=setting.model, setting=setting.setting):
                campaign = load_campaign(setting.campaign, point=setting.point)
                self.assertEqual(campaign.source_specs(), lock["sources"])
                self.assertEqual(
                    campaign.raw["mast"]["conda_fbpkg"],
                    lock["runtime"]["conda_fbpkg"],
                )
                self.assertEqual(
                    campaign.raw["parallelism"]["spmd_backend"],
                    "default",
                )

    def test_authored_source_runtime_and_backend_pins_are_rejected(self) -> None:
        source = (REPO_ROOT / "campaigns/llama3_8b_2d.toml").read_text()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_pin = root / "source.toml"
            source_pin.write_text(
                source + '\n[sources.torchtitan]\nremote = "x"\ncommit = "y"\n'
            )
            with self.assertRaisesRegex(CampaignError, "source pins are forbidden"):
                load_campaign(source_pin, point="8gpu")

            runtime_pin = root / "runtime.toml"
            runtime_pin.write_text(
                source.replace(
                    'gpu_name_contains = "H100"',
                    'gpu_name_contains = "H100"\nconda_fbpkg = "other:1"',
                )
            )
            with self.assertRaisesRegex(CampaignError, "runtime pins are forbidden"):
                load_campaign(runtime_pin, point="8gpu")

            backend_pin = root / "backend.toml"
            backend_pin.write_text(
                source.replace(
                    "pipeline_parallel_degree = 1",
                    'pipeline_parallel_degree = 1\nspmd_backend = "spmd_types"',
                    1,
                )
            )
            with self.assertRaisesRegex(CampaignError, "spmd_backend is forbidden"):
                load_campaign(backend_pin, point="8gpu")

    def test_canonical_matrix_points(self) -> None:
        cases = {
            "muse_glimmer_30b.toml": ("16gpu", "32gpu", "64gpu", "128gpu"),
            "llama3_8b_2d.toml": ("8gpu", "16gpu", "32gpu", "64gpu", "128gpu"),
            "llama3_8b_seqlen.toml": ("2k", "4k", "8k", "16k", "32k"),
            "deepseek_v3_16b.toml": ("16gpu", "32gpu"),
        }
        for filename, points in cases.items():
            for point in points:
                with self.subTest(filename=filename, point=point):
                    campaign = load_campaign(
                        REPO_ROOT / "campaigns" / filename,
                        point=point,
                    )
                    self.assertGreater(campaign.world_size, 0)
                    self.assertTrue(campaign.phases)
        self.assertEqual(
            load_campaign(REPO_ROOT / "campaigns/llama3_8b_3d.toml").world_size,
            8,
        )

    def test_matrix_requires_an_explicit_point(self) -> None:
        with self.assertRaisesRegex(CampaignError, "requires --point"):
            load_campaign(REPO_ROOT / "campaigns/muse_glimmer_30b.toml")

    def test_gate_mode_is_functional_and_trace_only(self) -> None:
        campaign = load_campaign(
            REPO_ROOT / "campaigns/muse_glimmer_30b.toml",
            point="16gpu",
            mode="gate",
        )
        self.assertEqual(
            [(phase.name, phase.kind) for phase in campaign.phases],
            [("functional", "correctness"), ("trace_smoke", "trace")],
        )
        for phase in campaign.phases:
            self.assertEqual(phase.overrides["training.steps"], 2)

    def test_arm_override_cannot_break_the_world_mesh(self) -> None:
        source = (REPO_ROOT / "campaigns/llama3_8b_2d.toml").read_text()
        invalid = source.replace(
            '[arms.environment]\nBENCHMARK_CONFIGURATION = "torchtitan_baseline"',
            '[arms.overrides]\n"parallelism.tensor_parallel_degree" = 1\n'
            '[arms.environment]\nBENCHMARK_CONFIGURATION = "torchtitan_baseline"',
            1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "campaign.toml"
            path.write_text(invalid)
            with self.assertRaisesRegex(CampaignError, "parallel mesh product"):
                load_campaign(path, point="8gpu")

    def test_native_argument_rendering_keeps_sample_based_fields(self) -> None:
        self.assertEqual(
            settings_to_argv(
                {
                    "training.local_batch_size": 2,
                    "training.global_batch_size": 8,
                    "training.gradient_accumulation_steps": 1,
                    "training.seq_len": 8192,
                }
            ),
            [
                "--training.global-batch-size",
                "8",
                "--training.local-batch-size",
                "2",
                "--training.seq-len",
                "8192",
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


class SourceTests(unittest.TestCase):
    def test_dirty_source_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            root.mkdir()
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(
                ["git", "-C", str(root), "config", "user.email", "test@example.com"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(root), "config", "user.name", "Harness Test"],
                check=True,
            )
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
            subprocess.run(
                ["git", "-C", str(root), "commit", "-qm", "fixture"], check=True
            )
            head = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                text=True,
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


if __name__ == "__main__":
    unittest.main()
