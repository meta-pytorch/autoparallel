from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from harness.run import _materialize_source
from harness.settings import load_run_settings, resolve_run_setting


class RunTests(unittest.TestCase):
    def test_every_setting_is_unique_and_resolvable(self) -> None:
        settings = load_run_settings()
        self.assertEqual(len(settings), 27)
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


if __name__ == "__main__":
    unittest.main()
