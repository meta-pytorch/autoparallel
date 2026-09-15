from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from harness.analysis import (
    PLANNER_TIMING_FIELDS,
    TIMER_EVENTS,
    _all_rank_memory,
    _input_hash_audit,
    _parameter_state_audit,
    _pair_summaries,
    _placement_audit,
    _planner_result,
    _primary_for_structured,
)


class _Phase:
    name = "performance"
    arms = ("baseline", "treatment")


class _Campaign:
    phases = (_Phase(),)
    world_size = 2
    raw = {"comparison": {"pairs": [["baseline", "treatment"]]}}


def _write_parameter_audit(
    root: Path,
    arm: str,
    rows: tuple[tuple[list[int], float, float], tuple[list[int], float, float]],
    *,
    checkpoint_wrapped: bool = False,
    stage: str | None = None,
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
        if stage is not None:
            record["stage"] = stage
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

    def test_missing_parameter_audits_fail_for_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = _parameter_state_audit(_Campaign(), Path(temporary))

            self.assertEqual(result["status"], "failed")
            self.assertTrue(result["required"])
            self.assertEqual(len(result["errors"]), 2)

    def test_parameter_audit_stage_must_match_across_arms(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = (([2, 1], 1.0, 3.0), ([2, 1], 2.0, 4.0))
            _write_parameter_audit(root, "baseline", rows)
            _write_parameter_audit(root, "treatment", rows, stage="post_load")

            result = _parameter_state_audit(_Campaign(), root)

            self.assertEqual(result["status"], "failed")
            self.assertEqual(
                result["errors"][0]["error"], "parameter audit stage differs"
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

    def test_pair_secondary_uses_only_declared_normal_steps(self) -> None:
        class Arm:
            def __init__(self, name: str) -> None:
                self.name = name

        class Phase:
            name = "performance"
            kind = "performance"
            arms = ("baseline", "treatment")

        class Campaign:
            arms = (Arm("baseline"), Arm("treatment"))
            phases = (Phase(),)
            raw = {
                "training": {"global_batch_size": 1, "seq_len": 1000},
                "comparison": {"pairs": [["baseline", "treatment"]]},
                "measurement": {"primary": {"steps": [7]}},
            }

        def metrics(values: tuple[float, float]) -> dict:
            return {
                "components_ms": {
                    name: {
                        "per_step_rank_max": {
                            "count": 2,
                            "mean": sum(values) / 2,
                            "median": sum(values) / 2,
                            "sample_sd": 0.0,
                            "population_sd": 0.0,
                            "min": min(values),
                            "max": max(values),
                            "samples": list(values),
                        },
                        "steps": [6, 7],
                    }
                    for name in TIMER_EVENTS
                }
            }

        phases = {
            "performance": {
                "arms": {
                    "baseline": {"structured_metrics": metrics((100.0, 200.0))},
                    "treatment": {"structured_metrics": metrics((80.0, 160.0))},
                }
            }
        }
        result = _pair_summaries(Campaign(), phases)[0]
        self.assertEqual(
            result["components_ms"]["step_end"]["baseline_ms"]["samples"],
            [200.0],
        )
        self.assertEqual(
            result["throughput_tokens_per_second"]["baseline"]["samples"],
            [5000.0],
        )


class RuntimeEvidenceAuditTests(unittest.TestCase):
    def test_buffered_memory_is_accepted_without_tensorboard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics_root = root / "job/benchmark_metrics"
            metrics_root.mkdir(parents=True)
            for rank in range(2):
                records = []
                for step in (5, 10, 15, 20, 25):
                    records.append(
                        {
                            "step": step,
                            "metrics": {
                                "memory/max_active(GiB)": rank + step / 10,
                                "memory/max_reserved(GiB)": rank + step / 10 + 1,
                                "memory/num_alloc_retries": 0,
                                "memory/num_ooms": 0,
                            },
                        }
                    )
                (metrics_root / f"rank_{rank:05d}.json").write_text(
                    json.dumps({"rank": rank, "records": records}) + "\n"
                )

            result = _all_rank_memory(root, 5, 2)

            self.assertEqual(result["status"], "passed")
            self.assertEqual(
                result["metrics"]["memory/max_active(GiB)"]["peak"]["step"],
                25,
            )

    def test_runtime_input_hashes_and_placements_are_paired(self) -> None:
        phases = (
            SimpleNamespace(name="canonical", arms=("canonical_fresh",)),
            SimpleNamespace(
                name="performance", arms=("target_fresh", "replay_2k")
            ),
            SimpleNamespace(name="trace", arms=("target_fresh", "replay_2k")),
        )
        campaign = SimpleNamespace(
            phases=phases,
            world_size=2,
            raw={
                "data": {"runtime_input_hash_audit": True},
                "artifacts": {"placement_audit": "replanning"},
                "comparison": {"pairs": [["target_fresh", "replay_2k"]]},
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical_digest = "a" * 64
            canonical_hash = "b" * 64
            for phase in phases:
                for arm in phase.arms:
                    placement_root = root / phase.name / arm / "placement_audit"
                    placement_root.mkdir(parents=True)
                    for rank in range(2):
                        replay = arm == "replay_2k"
                        row = {
                            "rank": rank,
                            "mode": "replay_2k" if replay else "fresh",
                            "configured_solver": "approx",
                            "solver_invoked": not replay,
                            "placement_digest": canonical_digest,
                            "placement_file_sha256": canonical_hash,
                            "canonical_placement_digest": (
                                canonical_digest if replay else None
                            ),
                            "canonical_file_sha256": canonical_hash if replay else None,
                        }
                        (placement_root / f"rank_{rank:03d}.json").write_text(
                            json.dumps(row) + "\n"
                        )
                    if phase.name != "canonical":
                        input_root = root / phase.name / arm / "input_audit"
                        input_root.mkdir(parents=True)
                        for rank in range(2):
                            (input_root / f"rank_{rank:05d}.json").write_text(
                                json.dumps(
                                    {
                                        "rank": rank,
                                        "dp_rank": rank,
                                        "batch_count": 2,
                                        "batch_sha256": ["c" * 64, "d" * 64],
                                    }
                                )
                                + "\n"
                            )

            self.assertEqual(_input_hash_audit(campaign, root)["status"], "passed")
            self.assertEqual(_placement_audit(campaign, root)["status"], "passed")


class PlannerAnalysisTests(unittest.TestCase):
    def test_planner_result_validates_the_profile_cli_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_lock = {
                "autoparallel": {
                    "head": "1" * 40,
                    "tree_sha256": "2" * 64,
                    "dirty": False,
                }
            }
            lock_path = root / "source_lock.json"
            lock_path.write_text(json.dumps(source_lock) + "\n")
            lock_sha256 = hashlib.sha256(lock_path.read_bytes()).hexdigest()
            expected = {
                "point": "2d",
                "mesh": [4, 8],
                "repeat": 1,
                "role": "candidate",
                "solver": "approx",
                "lazy_costs": True,
                "seeded": True,
            }
            output = root / "planner/2d/repeat_01/approx"
            output.mkdir(parents=True)
            for marker in ("started", "completed"):
                (output / marker).write_text(f"{marker}\n")
            (output / "exit_code").write_text("0\n")
            (output / "stdout_stderr.log").write_text("profile complete\n")
            (output / "command.json").write_text(
                json.dumps(
                    [
                        "python",
                        "search_profile.py",
                        "--source-lock",
                        "lock",
                        "--detailed-solution",
                    ]
                )
                + "\n"
            )
            result = {
                "status": "success",
                "request": {
                    "model": "llama8b",
                    "mesh": "4,8",
                    "solver": "approx",
                    "lazy_costs": "true",
                    "seeded": True,
                    "revision_label": "1" * 40,
                    "source_lock": "lock",
                },
                "source_lock": {
                    "sha256": lock_sha256,
                    "autoparallel": source_lock["autoparallel"],
                },
                "expanded_config": {"mesh_shape": [4, 8]},
                "objective": 12.0,
                "placement_sha256": "3" * 64,
                "timings": dict.fromkeys(PLANNER_TIMING_FIELDS, 1.0),
                "environment": {"torch": "test"},
            }
            (output / "result.json").write_text(json.dumps(result) + "\n")

            row, errors = _planner_result(
                root,
                expected,
                source_lock=source_lock,
                source_lock_sha256=lock_sha256,
            )

            self.assertFalse(errors)
            self.assertEqual(row["status"], "passed")
            self.assertEqual(row["timings_s"]["search_total_s"], 1.0)


if __name__ == "__main__":
    unittest.main()
