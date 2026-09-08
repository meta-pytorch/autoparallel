from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harness.analysis import _primary_for_structured


class PrimaryMeasurementTests(unittest.TestCase):
    def test_real_structured_log_shape_supports_both_rank_selectors(self) -> None:
        rows = {
            0: [
                {
                    "global_rank": 0,
                    "step": 7,
                    "time_us": 1787201404773584,
                    "log_type_name": "fetching_batch_end",
                    "value": 51.549804862588644,
                },
                {
                    "global_rank": 0,
                    "step": 7,
                    "time_us": 1787201405041866,
                    "log_type_name": "step_end",
                    "value": 321.88741071149707,
                },
            ],
            1: [
                {
                    "global_rank": 1,
                    "step": 7,
                    "time_us": 1787201404773622,
                    "log_type_name": "fetching_batch_end",
                    "value": 52.081394009292126,
                },
                {
                    "global_rank": 1,
                    "step": 7,
                    "time_us": 1787201405040835,
                    "log_type_name": "step_end",
                    "value": 321.76718628033996,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root / "job/structured_logs"
            logs.mkdir(parents=True)
            for rank, records in rows.items():
                (logs / f"training.global_rank_{rank}.jsonl").write_text(
                    "".join(json.dumps(record) + "\n" for record in records)
                )

            critical = _primary_for_structured(
                root,
                config={
                    "steps": [7],
                    "rank_selector": "max_step_end",
                    "value": "step_end_duration",
                },
                world_size=2,
            )
            self.assertEqual(critical["rows"][0]["selected_rank"], 0)
            self.assertAlmostEqual(critical["latency_ms"]["mean"], 321.88741071149707)

            post_data = _primary_for_structured(
                root,
                config={
                    "steps": [7],
                    "rank_selector": "latest_fetching_batch_end",
                    "value": "step_end_minus_fetching_batch_end",
                },
                world_size=2,
            )
            self.assertEqual(post_data["rows"][0]["selected_rank"], 1)
            self.assertAlmostEqual(post_data["latency_ms"]["mean"], 267.213)


if __name__ == "__main__":
    unittest.main()
