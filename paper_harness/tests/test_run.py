from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from harness.run import _materialize_source
from harness.runtime import _allocation_fingerprint
from harness.settings import load_run_settings, resolve_run_setting


class RunTests(unittest.TestCase):
    def test_every_setting_is_unique_and_resolvable(self) -> None:
        settings = load_run_settings()
        self.assertEqual(len(settings), 32)
        for key, expected in settings.items():
            self.assertEqual(resolve_run_setting(*key), expected)

    def test_materialize_source_checks_out_exact_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            remote = root / "remote"
            source_root = root / "sources"
            records = root / "records"
            remote.mkdir()
            source_root.mkdir()
            subprocess.run(["git", "init", "-q", str(remote)], check=True)
            subprocess.run(
                ["git", "-C", str(remote), "config", "user.email", "test@example.com"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(remote), "config", "user.name", "Harness Test"],
                check=True,
            )
            (remote / "value.txt").write_text("value\n")
            subprocess.run(["git", "-C", str(remote), "add", "value.txt"], check=True)
            subprocess.run(
                ["git", "-C", str(remote), "commit", "-qm", "fixture"], check=True
            )
            commit = subprocess.check_output(
                ["git", "-C", str(remote), "rev-parse", "HEAD"],
                text=True,
            ).strip()

            checkout = _materialize_source(
                "fixture",
                {"remote": str(remote), "commit": commit},
                source_root=source_root,
                record_root=records,
            )
            self.assertEqual(
                subprocess.check_output(
                    ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                    text=True,
                ).strip(),
                commit,
            )

    def test_allocation_fingerprint_uses_the_full_rank_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "runtime/allocation_reference"
            reference.mkdir(parents=True)
            records = [
                {
                    "hostname": "host-a.pci1",
                    "rank": 0,
                    "local_rank": 0,
                    "world_size": 2,
                    "local_world_size": 1,
                    "gpu_ordinal": 0,
                    "gpu_uuid": "gpu-a",
                },
                {
                    "hostname": "host-b.pci1",
                    "rank": 1,
                    "local_rank": 0,
                    "world_size": 2,
                    "local_world_size": 1,
                    "gpu_ordinal": 0,
                    "gpu_uuid": "gpu-b",
                },
            ]
            for rank, record in enumerate(records):
                (reference / f"rank_{rank:03d}.json").write_text(json.dumps(record))
            initial = _allocation_fingerprint(root, 2, 1)
            records[1]["gpu_uuid"] = "gpu-c"
            (reference / "rank_001.json").write_text(json.dumps(records[1]))
            self.assertNotEqual(initial, _allocation_fingerprint(root, 2, 1))


if __name__ == "__main__":
    unittest.main()
