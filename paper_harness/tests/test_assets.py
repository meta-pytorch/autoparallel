from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from harness.assets import load_asset_lock, resolve_assets
from harness.campaign import CampaignError
from harness.sources import manifest_digest, tree_manifest


class AssetLockTests(unittest.TestCase):
    def test_checked_in_lock_uses_one_internal_workspace(self) -> None:
        lock = load_asset_lock()
        self.assertEqual(
            lock["workspace_uri"],
            "ws://ws.ai.pci0ai/checkpoint/infra",
        )
        self.assertTrue(lock["assets"])

    def test_resolve_assets_checks_complete_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset = root / "locked/assets/example"
            asset.mkdir(parents=True)
            (asset / "value.txt").write_text("value\n")
            digest = manifest_digest(tree_manifest(asset))
            lock = root / "asset_lock.toml"
            lock.write_text(
                "schema_version = 1\n"
                'workspace_uri = "ws://example/workspace"\n'
                'root = "locked/assets"\n'
                "[assets.example]\n"
                'relative_path = "example"\n'
                "file_count = 1\n"
                f'tree_sha256 = "{digest}"\n'
            )
            roots, evidence = resolve_assets(root, ["example"], path=lock)
            self.assertEqual(roots["example"], asset)
            self.assertEqual(evidence["example"]["tree_sha256"], digest)

            (asset / "value.txt").write_text("changed\n")
            with self.assertRaisesRegex(CampaignError, "tree digest differs"):
                resolve_assets(root, ["example"], path=lock)


if __name__ == "__main__":
    unittest.main()
