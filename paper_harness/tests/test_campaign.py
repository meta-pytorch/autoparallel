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
                "llama3_8b_3d_weight_shard_order_ab.toml",
                "llama3_8b_planner.toml",
                "llama3_8b_replanning_approx.toml",
                "muse_glimmer_30b.toml",
            },
        )

    def test_all_settings_resolve_to_the_global_lock(self) -> None:
        lock = load_experiment_lock()
        settings = load_run_settings()
        self.assertEqual(len(settings), 23)
        for setting in settings.values():
            with self.subTest(model=setting.model, setting=setting.setting):
                campaign = load_campaign(setting.campaign, point=setting.point)
                self.assertEqual(campaign.source_specs(), lock["sources"])
                self.assertEqual(
                    campaign.raw["mast"]["conda_fbpkg"],
                    lock["runtime"]["conda_fbpkg"],
                )
                if not campaign.is_planner:
                    self.assertEqual(
                        campaign.raw["parallelism"]["spmd_backend"],
                        "default",
                    )
                self.assertEqual(
                    campaign.raw["mast"]["environment"]["TORCHINDUCTOR_CUDAGRAPHS"],
                    "0",
                )

    def test_every_training_autoparallel_arm_uses_approx(self) -> None:
        for setting in load_run_settings().values():
            campaign = load_campaign(setting.campaign, point=setting.point)
            if campaign.is_planner:
                continue
            for phase in campaign.phases:
                for arm_name in phase.arms:
                    arm = campaign.arm(arm_name)
                    if arm.profile != "apgt_validated_v1":
                        continue
                    self.assertEqual(
                        campaign.phase_arm_settings(phase, arm).get(
                            "compile.autoparallel_solver"
                        ),
                        "approx",
                        (setting.model, setting.setting, phase.name, arm_name),
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

            cudagraph_enabled = root / "cudagraph.toml"
            cudagraph_enabled.write_text(
                source.replace(
                    'TORCHINDUCTOR_CUDAGRAPHS = "0"',
                    'TORCHINDUCTOR_CUDAGRAPHS = "1"',
                )
            )
            with self.assertRaisesRegex(
                CampaignError, "TORCHINDUCTOR_CUDAGRAPHS must be '0'"
            ):
                load_campaign(cudagraph_enabled, point="8gpu")

    def test_canonical_matrix_points(self) -> None:
        cases = {
            "muse_glimmer_30b.toml": ("8gpu", "16gpu", "32gpu", "64gpu"),
            "llama3_8b_2d.toml": ("8gpu", "16gpu", "32gpu", "64gpu", "128gpu"),
            "llama3_8b_3d.toml": ("2x2x4", "4x2x4", "8x2x4"),
            "llama3_8b_replanning_approx.toml": (
                "seq2k-lb2",
                "seq4k-lb2",
                "seq8k-lb2",
                "seq16k-lb2",
                "seq32k-lb2",
                "seq2k-lb4",
                "seq2k-lb8",
            ),
            "deepseek_v3_16b.toml": ("2x2x4", "2x2x8", "4x2x8"),
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

    def test_weight_shard_order_campaign_changes_only_source_variant(self) -> None:
        campaign = load_campaign(
            REPO_ROOT / "campaigns/llama3_8b_3d_weight_shard_order_ab.toml"
        )
        self.assertEqual(campaign.world_size, 16)
        self.assertEqual(
            campaign.raw["comparison"]["declared_variable"],
            "autoparallel_weight_physical_shard_order_lowering",
        )
        self.assertEqual(
            [arm.name for arm in campaign.arms],
            ["apgt_baseline", "apgt_fixed"],
        )
        self.assertEqual(
            [arm.environment["BENCHMARK_AP_SOURCE_VARIANT"] for arm in campaign.arms],
            ["baseline", "candidate"],
        )
        self.assertEqual(
            [
                campaign.phase_arm_settings(campaign.phases[0], arm)
                for arm in campaign.arms
            ],
            [
                campaign.phase_arm_settings(campaign.phases[0], campaign.arms[0]),
                campaign.phase_arm_settings(campaign.phases[0], campaign.arms[0]),
            ],
        )

    def test_matrix_requires_an_explicit_point(self) -> None:
        with self.assertRaisesRegex(CampaignError, "requires --point"):
            load_campaign(REPO_ROOT / "campaigns/muse_glimmer_30b.toml")

    def test_requested_three_dimensional_meshes(self) -> None:
        deepseek = {
            "2x2x4": (4, 8, 4),
            "2x2x8": (4, 16, 8),
            "4x2x8": (8, 16, 8),
        }
        for point, expected in deepseek.items():
            campaign = load_campaign(
                REPO_ROOT / "campaigns/deepseek_v3_16b.toml", point=point
            )
            parallelism = campaign.raw["parallelism"]
            self.assertEqual(
                (
                    parallelism["data_parallel_shard_degree"],
                    parallelism["expert_parallel_degree"],
                    parallelism["tensor_parallel_degree"],
                ),
                expected,
            )

        for point, expected_dp in (("2x2x4", 2), ("4x2x4", 4), ("8x2x4", 8)):
            campaign = load_campaign(
                REPO_ROOT / "campaigns/llama3_8b_3d.toml", point=point
            )
            parallelism = campaign.raw["parallelism"]
            self.assertEqual(
                (
                    parallelism["data_parallel_shard_degree"],
                    parallelism["context_parallel_degree"],
                    parallelism["tensor_parallel_degree"],
                ),
                (expected_dp, 2, 4),
            )
            self.assertEqual(campaign.raw["training"]["seq_len"], 16384)

    def test_planner_campaign_expands_exact_approx_and_lp_runs(self) -> None:
        campaign = load_campaign(REPO_ROOT / "campaigns/llama3_8b_planner.toml")
        self.assertTrue(campaign.is_planner)
        self.assertEqual(campaign.world_size, 8)
        self.assertFalse(campaign.arms)
        self.assertFalse(campaign.phases)
        runs = campaign.planner_runs()
        self.assertEqual(len(runs), 24)
        self.assertEqual(
            [
                (point["name"], point["mesh"])
                for point in campaign.raw["planner"]["points"]
            ],
            [
                ("1d", [8]),
                ("2d", [4, 8]),
                ("3d", [4, 2, 4]),
                ("4d", [2, 2, 2, 4]),
            ],
        )
        for run in runs:
            if run["role"] == "candidate":
                self.assertEqual(run["solver"], "approx")
                self.assertTrue(run["lazy_costs"])
                self.assertTrue(run["seeded"])
            else:
                self.assertEqual(run["solver"], "lp")
                self.assertIsNone(run["lazy_costs"])
                self.assertFalse(run["seeded"])

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
