from __future__ import annotations

import unittest
from pathlib import Path

from harness.campaign import load_campaign
from harness.parity import validate_pair


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGNS = REPO_ROOT / "campaigns"


class Llama3ReplanningCampaignTests(unittest.TestCase):
    def test_canonical_generator_is_fixed_to_4x8_ilp(self) -> None:
        campaign = load_campaign(CAMPAIGNS / "llama3_8b_replanning_canonical.toml")
        self.assertEqual(campaign.world_size, 32)
        self.assertEqual(campaign.raw["mast"]["locality"], "dc;pci1")
        self.assertEqual(campaign.raw["training"]["seq_len"], 2048)
        self.assertEqual(campaign.raw["training"]["local_batch_size"], 2)
        arm = campaign.arm("canonical_fresh")
        settings = campaign.phase_arm_settings(campaign.phases[0], arm)
        self.assertEqual(settings["compile.autoparallel_solver"], "ilp")
        self.assertEqual(
            settings["compile.autoparallel_placements_save_path"],
            "{output}/canonical_2k.json",
        )

    def test_same_mesh_matrix_covers_only_approved_cases(self) -> None:
        expected = {
            "seq2k_lb2": (2048, 2, 8, "s2048_lb2"),
            "seq4k_lb2": (4096, 2, 8, "s4096_lb2"),
            "seq8k_lb2": (8192, 2, 8, "s8192_lb2"),
            "seq16k_lb2": (16384, 2, 8, "s16384_lb2"),
            "seq2k_lb4": (2048, 4, 16, "s2048_lb4"),
            "seq2k_lb8": (2048, 8, 32, "s2048_lb8"),
        }
        path = CAMPAIGNS / "llama3_8b_replanning_same_mesh.toml"
        for point, values in expected.items():
            with self.subTest(point=point):
                campaign = load_campaign(path, point=point)
                seq_len, local_batch, global_batch, replay_case = values
                self.assertEqual(campaign.world_size, 32)
                self.assertEqual(campaign.raw["mast"]["locality"], "dc;pci1")
                self.assertEqual(campaign.raw["training"]["seq_len"], seq_len)
                self.assertEqual(
                    campaign.raw["training"]["local_batch_size"], local_batch
                )
                self.assertEqual(
                    campaign.raw["training"]["global_batch_size"], global_batch
                )
                self.assertEqual(
                    campaign.raw["mast"]["environment"]["BENCHMARK_REPLAY_CASE"],
                    replay_case,
                )

    def test_fresh_and_replay_change_only_the_declared_placement_path(self) -> None:
        campaign = load_campaign(
            CAMPAIGNS / "llama3_8b_replanning_same_mesh.toml",
            point="seq4k_lb2",
        )
        phase = campaign.phases[0]
        check = validate_pair(
            "fresh",
            campaign.phase_arm_settings(phase, campaign.arm("fresh")),
            "canonical_replay",
            campaign.phase_arm_settings(phase, campaign.arm("canonical_replay")),
            list(campaign.raw["comparison"]["allowed_config_paths"]),
        )
        self.assertEqual(
            [item["path"] for item in check["observed_differences"]],
            ["compile.autoparallel_placements_load_path"],
        )

    def test_cross_scale_points_remain_user_gated(self) -> None:
        source = (CAMPAIGNS / "llama3_8b_replanning_same_mesh.toml").read_text()
        self.assertNotIn('name = "scale2x8"', source)
        self.assertNotIn('name = "scale8x8"', source)
        self.assertTrue(
            (
                REPO_ROOT / "workloads/llama3_replanning/CROSS_SCALE_USER_GATE.md"
            ).is_file()
        )


if __name__ == "__main__":
    unittest.main()
