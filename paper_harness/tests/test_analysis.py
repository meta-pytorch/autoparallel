from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harness.analysis import _parameter_state_audit, _primary_for_structured


class _Phase:
    name = "performance"
    arms = ("baseline", "treatment")


class _Campaign:
    phases = (_Phase(),)
    world_size = 2


def _write_parameter_audit(
    root: Path,
    arm: str,
    rows: tuple[tuple[list[int], float, float], tuple[list[int], float, float]],
    *,
    checkpoint_wrapped: bool = False,
) -> None:
    output = root / "performance" / arm / "parameter_audit"
    output.mkdir(parents=True)
    for rank, (local_shape, total, square_total) in enumerate(rows):
        name = "layers.0.attention.wo.weight"
        if checkpoint_wrapped:
            name = "layers.0._checkpoint_wrapped_module.attention.wo.weight"
        record = {
            "part": 0,
            "name": name,
            "raw_name": "model.layers.0.attention.wo.weight",
            "global_shape": [2, 2],
            "local_shape": local_shape,
            "dtype": "torch.bfloat16",
            "sample_sha256": "unused-by-global-moment-audit",
            "sum": total,
            "square_sum": square_total,
        }
        (output / f"rank_{rank:02d}.json").write_text(
            json.dumps([record], indent=2, sort_keys=True) + "\n"
        )


class ParameterStateAuditTests(unittest.TestCase):
    def test_matching_global_moments_pass_across_different_local_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_parameter_audit(
                root,
                "baseline",
                (([2, 2], 2.0, 6.0), ([2, 2], 4.0, 8.0)),
                checkpoint_wrapped=True,
            )
            _write_parameter_audit(
                root, "treatment", (([1, 2], 2.5, 5.0), ([1, 2], 0.5, 2.0))
            )

            result = _parameter_state_audit(_Campaign(), root)

            self.assertEqual(result["status"], "passed")
            self.assertTrue(result["required"])

    def test_mismatched_global_moments_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_parameter_audit(
                root, "baseline", (([2, 1], 1.0, 3.0), ([2, 1], 2.0, 4.0))
            )
            _write_parameter_audit(
                root, "treatment", (([1, 2], 2.5, 5.0), ([1, 2], 1.5, 2.0))
            )

            result = _parameter_state_audit(_Campaign(), root)

            self.assertEqual(result["status"], "failed")
            self.assertEqual(
                result["errors"][0]["parameter"], "0:layers.0.attention.wo.weight"
            )
            self.assertIn("sum", result["errors"][0]["differences"])

    def test_partial_parameter_audit_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_parameter_audit(
                root, "baseline", (([2, 1], 1.0, 3.0), ([2, 1], 2.0, 4.0))
            )

            result = _parameter_state_audit(_Campaign(), root)

            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["errors"][-1]["run"], "performance/treatment")
            self.assertEqual(
                result["errors"][-1]["error"], "missing parameter audit files"
            )


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
