# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".agents/skills/autoparallel-optimizer/scripts"


def load_script(name):
    path = SCRIPTS / f"{name}.py"
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(f"test_{name}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


modeled = load_script("modeled_baselines")
triage = load_script("build_improvement_queue")
corpus = load_script("evaluate_optimizer_replay_corpus")


def test_activation_liveness_accounts_for_local_shards():
    plan = {
        "nodes": [
            {
                "name": "input",
                "op": "placeholder",
                "shape": [8, 4],
                "dtype": "float32",
                "placement": "S(0)",
                "inputs": [],
            },
            {
                "name": "first",
                "op": "call_function",
                "shape": [8, 4],
                "dtype": "float32",
                "placement": "S(0)",
                "inputs": [{"name": "input"}],
            },
            {
                "name": "second",
                "op": "call_function",
                "shape": [8, 4],
                "dtype": "float32",
                "placement": "R",
                "inputs": [{"name": "first"}],
            },
        ]
    }

    estimate = modeled.activation_liveness_estimate(plan, (2,))

    assert estimate["peak_bytes"] == 192
    assert not estimate["aliasing_modeled"]


def test_optimizer_update_calibration_is_bounded_by_regime_and_scale(tmp_path):
    calibration = {
        "schema_version": 1,
        "kind": "test-calibration",
        "intercept_us": 10.0,
        "slope_us_per_local_parameter_gib": 100.0,
        "calibration": {
            "local_parameter_gib_min": 0.5,
            "local_parameter_gib_max": 2.0,
            "maximum_prospective_kernel_relative_error": 0.1,
        },
        "source_run_ids": ["run"],
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(calibration))
    loaded = modeled.load_optimizer_cost_calibration(path)

    estimate = modeled.optimizer_update_estimate(
        2**30,
        loaded,
        {"platform": "matched", "kernel": "unknown"},
    )

    assert estimate["status"] == "screening_only"
    assert estimate["latency"]["expected"] == 110.0
    assert modeled.optimizer_update_estimate(3 * 2**30, loaded)["reason"] == (
        "scale_out_of_range"
    )


def test_relative_fsdp_calibration_uses_structural_features():
    calibration = {
        "schema_version": 2,
        "use": "screening_only",
        "strategies": {
            "fsdp": {
                "method": "log-structural",
                "coefficients": [0.0, 1.0, 0.0, 0.0],
                "family_holdout_winner_accuracy": 1.0,
                "family_holdout_maximum_multiplicative_error": 1.25,
            }
        },
    }
    autoparallel = {
        "unit": "us",
        "optimistic": 80.0,
        "expected": 100.0,
        "pessimistic": 120.0,
        "calibration_relative_error": 0.0,
        "schedule_model": "test",
    }
    baseline = {
        "strategy": "fsdp",
        "latency": {
            "unit": "us",
            "optimistic": 40.0,
            "expected": 50.0,
            "pessimistic": 60.0,
            "calibration_relative_error": 0.0,
            "schedule_model": "test",
        },
        "communication": {"wrap_unit_count": 4},
    }

    calibrated = modeled.calibrate_relative_baseline(
        autoparallel, baseline, 1024, calibration
    )

    assert calibrated["latency"]["expected"] == pytest.approx(50.0)
    assert calibrated["latency"]["calibration_relative_error"] == 0.25
    assert calibrated["relative_calibration"]["calibrated_ratio"] == 2.0


def test_improvement_candidate_prefers_replayable_representatives():
    records = [
        {
            "run_id": "new-no-replay",
            "created_at": "2026-01-02T00:00:00Z",
            "bundle": "new",
            "artifact_roles": [],
        },
        {
            "run_id": "old-with-replay",
            "created_at": "2026-01-01T00:00:00Z",
            "bundle": "old",
            "optimizer_replay": {"available": True},
        },
    ]

    candidate = triage._run_candidate(
        identifier="test",
        kind="finding",
        priority="P2",
        title="test",
        records=records,
        occurrence_count=2,
        max_representatives=1,
        details={},
        suggested_action="test",
    )

    assert candidate["representative_bundles"] == ["old"]
    assert candidate["replayable_representative_bundles"] == ["old"]
    assert candidate["replayable_run_count"] == 1


def test_corpus_selection_is_replay_only_and_split_is_stable():
    catalog = {
        "runs": [
            {
                "bundle": "replay",
                "optimizer_replay": {"available": True},
                "campaign": {"id": "campaign", "case_id": "case"},
            },
            {
                "bundle": "plain",
                "optimizer_replay": {"available": False},
                "campaign": {"id": "campaign", "case_id": "other"},
            },
        ]
    }

    assert corpus.bundles_from_feedback(catalog, campaign_id="campaign") == ["replay"]
    assert corpus._split_for_case("case", "seed", 0.5) == corpus._split_for_case(
        "case", "seed", 0.5
    )
    with pytest.raises(ValueError, match="between zero and one"):
        corpus.evaluate_corpus(Path("."), [], holdout_fraction=2.0)
