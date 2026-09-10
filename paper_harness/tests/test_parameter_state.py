from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import torch

from workloads.parameter_state import register_post_load_parameter_audit


class PostLoadParameterAuditTests(unittest.TestCase):
    def test_audit_runs_once_after_every_model_part_loads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            previous = os.environ.get("PARAMETER_AUDIT_DIR")
            previous_rank = os.environ.get("RANK")
            os.environ["PARAMETER_AUDIT_DIR"] = temporary
            os.environ["RANK"] = "0"
            try:
                parts = [
                    torch.nn.Linear(2, 2, bias=False),
                    torch.nn.Linear(2, 1, bias=False),
                ]
                register_post_load_parameter_audit(None, parts, None)
                output = Path(temporary) / "rank_00.json"

                parts[0].load_state_dict({"weight": torch.full((2, 2), 3.0)})
                self.assertFalse(output.exists())

                parts[1].load_state_dict({"weight": torch.full((1, 2), 5.0)})
                records = json.loads(output.read_text())
                self.assertEqual(
                    [record["stage"] for record in records], ["post_load"] * 2
                )
                self.assertEqual([record["sum"] for record in records], [12.0, 10.0])

                parts[0].load_state_dict({"weight": torch.zeros((2, 2))})
                self.assertEqual(json.loads(output.read_text()), records)
            finally:
                if previous is None:
                    os.environ.pop("PARAMETER_AUDIT_DIR", None)
                else:
                    os.environ["PARAMETER_AUDIT_DIR"] = previous
                if previous_rank is None:
                    os.environ.pop("RANK", None)
                else:
                    os.environ["RANK"] = previous_rank


if __name__ == "__main__":
    unittest.main()
