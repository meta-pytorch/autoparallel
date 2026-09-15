from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from harness.campaign import CampaignError
from harness.cli import build_parser
from harness.dryrun import _definition_checks
from harness.run import (
    _attempt_path,
    _materialization_root,
    _materialize_source,
    _mounted_workspace,
    _passing_json,
    _validate_materialization_ready,
    _write_materialization_ready,
    run_reproduction,
)
from harness.settings import load_run_settings, resolve_run_setting
from scripts.measurements.measurement import (
    _ensure_submitted,
    _package_evidence,
    _scheduler_summary,
)


class RunTests(unittest.TestCase):
    def test_run_cli_accepts_explicit_attempt_number(self) -> None:
        args = build_parser().parse_args(
            [
                "run",
                "--model",
                "llama3_8b",
                "--setting",
                "2d-8gpu",
                "--attempt-number",
                "7",
            ]
        )
        self.assertEqual(args.attempt_number, 7)

    def test_every_setting_is_unique_and_resolvable(self) -> None:
        settings = load_run_settings()
        self.assertTrue(settings)
        for key, expected in settings.items():
            self.assertEqual(resolve_run_setting(*key), expected)
        planner = resolve_run_setting("llama3_8b", "planning-scalability")
        self.assertEqual(planner.campaign.name, "llama3_8b_planner.toml")

    def test_attempt_path_requires_an_explicit_number(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary)
            self.assertEqual(_attempt_path(task), task / "attempts/001")
            (task / "attempts/002").mkdir(parents=True)
            self.assertEqual(_attempt_path(task, 2), task / "attempts/002")
            with self.assertRaisesRegex(CampaignError, "attempt-number"):
                _attempt_path(task, 0)

    def test_attempt_path_rejects_legacy_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary)
            (task / "attempt").mkdir()
            with self.assertRaisesRegex(CampaignError, "legacy mutable attempt"):
                _attempt_path(task)

    def test_shared_cache_is_namespaced_by_experiment_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary) / "task"
            cache = Path(temporary) / "cache"
            with mock.patch.dict("os.environ", {"HARNESS_CACHE_ROOT": str(cache)}):
                root, shared = _materialization_root(task, "abc123")
            self.assertTrue(shared)
            self.assertEqual(root, cache / "abc123")

    def test_shared_cache_requires_an_absolute_path(self) -> None:
        with mock.patch.dict("os.environ", {"HARNESS_CACHE_ROOT": "relative/cache"}):
            with self.assertRaisesRegex(CampaignError, "must be an absolute path"):
                _materialization_root(Path("/task"), "abc123")

    def test_materialization_ready_is_atomic_and_lock_specific(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            python = root / "conda/bin/python"
            python.parent.mkdir(parents=True)
            python.write_text("")
            lock = {
                "sources": {
                    "autoparallel": {"commit": "a" * 40},
                    "torchtitan": {"commit": "b" * 40},
                },
                "runtime": {"conda_fbpkg": "runtime:1"},
            }
            ready = root / "READY.json"
            _write_materialization_ready(
                ready, lock_digest="digest", lock=lock, python=python
            )
            _validate_materialization_ready(
                ready, lock_digest="digest", lock=lock, python=python
            )
            value = json.loads(ready.read_text())
            value["experiment_lock_sha256"] = "stale"
            ready.write_text(json.dumps(value))
            with self.assertRaisesRegex(CampaignError, "stale"):
                _validate_materialization_ready(
                    ready, lock_digest="digest", lock=lock, python=python
                )

    def test_only_passed_json_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "analysis.json"
            path.write_text('{"status": "incomplete"}\n')
            self.assertFalse(_passing_json(path))
            path.write_text('{"status": "passed"}\n')
            self.assertTrue(_passing_json(path))

    def test_mounted_workspace_attempts_cleanup_after_body_failure(self) -> None:
        mountpoint = Path("/absolute/mount")
        with (
            mock.patch(
                "harness.run._mount_workspace",
                return_value=(mountpoint, Path("/absolute/package")),
            ),
            mock.patch("harness.run.os.path.ismount", return_value=True),
            mock.patch("harness.run._capture") as capture,
        ):
            with self.assertRaisesRegex(RuntimeError, "asset validation"):
                with _mounted_workspace(Path("/task"), Path("/records"), name="assets"):
                    raise RuntimeError("asset validation failed")
        self.assertEqual(capture.call_count, 1)
        self.assertEqual(capture.call_args.args[0][:2], ["fusermount", "-u"])

    def test_post_submit_failure_still_monitors_remote_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            task = workspace / "model-setting-reproduction"
            attempt = task / "attempts/001"
            attempt.mkdir(parents=True)
            (attempt / "SEALED").write_text("sealed\n")
            (attempt / "dryrun_audit.json").write_text('{"status": "passed"}\n')
            (attempt / "job_id.txt").write_text("job-123\n")
            python = workspace / "python"
            python.write_text("")
            calls = []

            def measurement(action, **kwargs):
                calls.append((action, kwargs.get("extra")))
                if action == "submit":
                    raise CampaignError("post-submit audit failed")

            with (
                mock.patch.dict(
                    "os.environ",
                    {"HARNESS_WORKSPACE_ROOT": str(workspace)},
                    clear=False,
                ),
                mock.patch(
                    "harness.run.resolve_run_setting",
                    return_value=SimpleNamespace(campaign=Path("campaign"), point=None),
                ),
                mock.patch(
                    "harness.run.load_experiment_lock",
                    return_value={
                        "sources": {"torchtitan": {}, "autoparallel": {}},
                        "runtime": {"conda_fbpkg": "runtime:1"},
                    },
                ),
                mock.patch("harness.run.experiment_lock_digest", return_value="digest"),
                mock.patch("harness.run._materialize_source", return_value=workspace),
                mock.patch("harness.run._fetch_package", return_value=python),
                mock.patch("harness.run._write_materialization_ready"),
                mock.patch("harness.run.load_campaign") as load_campaign,
                mock.patch(
                    "harness.run._mount_workspace",
                    return_value=(workspace / "mounted", workspace / "oilfs"),
                ),
                mock.patch("harness.run._measurement", side_effect=measurement),
            ):
                load_campaign.return_value = SimpleNamespace(raw={}, is_planner=True)
                with self.assertRaisesRegex(CampaignError, "post-submit audit failed"):
                    run_reproduction("model", "setting")

            self.assertEqual([call[0] for call in calls], ["submit", "monitor"])
            self.assertEqual(
                calls[1][1],
                ["--run-root", str(workspace / "mounted/outputs/job-123/run")],
            )

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


class SubmissionLifecycleTests(unittest.TestCase):
    def test_existing_job_id_never_resubmits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            attempt = Path(temporary)
            (attempt / "job_id.txt").write_text("job-123\n")
            with mock.patch("scripts.measurements.measurement._capture") as capture:
                job_id, newly_submitted = _ensure_submitted(
                    attempt, Path("/unused/python")
                )
            self.assertEqual(job_id, "job-123")
            self.assertFalse(newly_submitted)
            capture.assert_not_called()

    def test_definition_checks_ports_and_force_single_region(self) -> None:
        resolved = {
            "campaign_type": "training",
            "mast": {
                "nodes": 2,
                "nproc_per_node": 8,
                "hardware": "grandteton_80g_roce",
                "locality": "dc;pci1",
                "conda_fbpkg": "runtime:1",
                "retries": 0,
                "master_port": 29500,
            },
            "experiment_lock": {"runtime": {"conda_fbpkg": "runtime:1"}},
            "resolved_phases": [{"arms": ["baseline", "treatment"]}],
        }
        definition = {
            "applicationMetadata": {"torchx/scheduler": "mast_conda"},
            "hpcClusterUuid": "MastGenAICluster",
            "name": "test-campaign-wangkj-abc123",
            "localityConstraints": {
                "locality": "Locality.DC",
                "options": ["pci1"],
            },
            "maxJobFailures": 0,
            "hpcTaskGroups": [
                {
                    "taskCount": 2,
                    "taskCountPerHost": 1,
                    "spec": {
                        "arguments": [
                            "--rdzv_backend",
                            "mast",
                            "--rdzv_conf",
                            "use_libuv=True",
                            "--rdzv_id",
                            "test-campaign-wangkj-abc123",
                            "--nnodes",
                            "2",
                            "--nproc-per-node",
                            "8",
                            "--role",
                            "training",
                            "--no-python",
                            "/packages/torchtitan_additional_packages/payload/harness_repo/launcher/run_rank.sh",
                        ],
                        "applicationPackages": [
                            {"fbpkgIdentifier": "runtime:1"},
                            {"fbpkgIdentifier": "oil.oilfs:stable"},
                            {"fbpkgIdentifier": "torchtitan_workspace:abc"},
                            {"fbpkgIdentifier": "torchtitan_additional_packages:def"},
                        ],
                        "command": "$WORKSPACE_DIR/mount.sh && /packages/conda_mast_core/tee/torchx_tee.sh command",
                        "env": {
                            "HARNESS_PAYLOAD_ROOT": "/packages/torchtitan_additional_packages/payload",
                            "DUMP_DIR": "/mnt/wsfuse/outputs/${app_id}",
                            "TITAN_STRUCT_LOGGER_HANDLERS": (
                                "torchtitan.observability.structured_logger."
                                "jsonl_handler.register_jsonl_handler"
                            ),
                            "EXPERIMENT_TASK": "training",
                        },
                        "machineConstraints": {
                            "types": {
                                "serverSubTypes": [
                                    "LogicalServerSubType.T20_GRAND_TETON_HBM3_ROCE"
                                ]
                            }
                        },
                        "ports": {"training_phase_2": 29501},
                        "resourceLimit": {"compute": {"gpu": 8}},
                        "restartPolicy": {"maxTotalFailures": 0},
                        "ttlsConfig": {"enable": True},
                        "unixUser": "root",
                    },
                }
            ],
        }
        command = [
            "torchx",
            "run",
            "--scheduler_args=conda_fbpkg_id=runtime:1,localityConstraints=dc;pci1,forceSingleRegion=False",
        ]
        launcher = Path(__file__).resolve().parents[1] / "launcher"
        resolved["name"] = "test-campaign"
        checks, _ = _definition_checks(
            definition,
            resolved=resolved,
            launcher_root=launcher,
            command=command,
            combined=None,
        )
        self.assertTrue(all(checks.values()), checks)

        resolved["mast"]["nodes"] = 1
        definition["hpcTaskGroups"][0]["taskCount"] = 1
        arguments = definition["hpcTaskGroups"][0]["spec"]["arguments"]
        arguments[arguments.index("--nnodes") + 1] = "1"
        arguments[arguments.index("--rdzv_backend") + 1] = "c10d"
        config_index = arguments.index("--rdzv_conf")
        del arguments[config_index : config_index + 2]
        insert_at = arguments.index("--rdzv_backend") + 2
        arguments[insert_at:insert_at] = [
            "--rdzv_endpoint",
            "localhost:0",
        ]
        checks, _ = _definition_checks(
            {"status": "ok", "data": definition},
            resolved=resolved,
            launcher_root=launcher,
            command=command,
            combined=None,
        )
        self.assertTrue(all(checks.values()), checks)
        command[-1] = command[-1].replace("False", "True")
        checks, _ = _definition_checks(
            definition,
            resolved=resolved,
            launcher_root=launcher,
            command=command,
            combined=None,
        )
        self.assertFalse(checks["force_single_region_false"])

        command[-1] = command[-1].replace("True", "False")
        definition["name"] = "job-123"
        definition["hpcTaskGroups"][0]["spec"]["arguments"][
            definition["hpcTaskGroups"][0]["spec"]["arguments"].index("--rdzv_id") + 1
        ] = "job-123"
        definition["hpcTaskGroups"][0]["spec"]["env"][
            "DUMP_DIR"
        ] = "/mnt/wsfuse/outputs/job-123"
        definition["localityConstraints"]["locality"] = "DC"
        definition["hpcTaskGroups"][0]["spec"]["machineConstraints"]["types"][
            "serverSubTypes"
        ] = ["T20_GRAND_TETON_HBM3_ROCE"]
        checks, _ = _definition_checks(
            {"status": "ok", "data": definition},
            resolved=resolved,
            launcher_root=launcher,
            command=command,
            combined=None,
            expected_job_id="job-123",
        )
        self.assertTrue(all(checks.values()), checks)
        definition["localityConstraints"]["locality"] = "UNKNOWN"
        checks, _ = _definition_checks(
            {"status": "ok", "data": definition},
            resolved=resolved,
            launcher_root=launcher,
            command=command,
            combined=None,
            expected_job_id="job-123",
        )
        self.assertFalse(checks["locality_scope"])

    def test_terminal_scheduler_summary_is_strict(self) -> None:
        task = {
            "attemptIndex": 0,
            "state": "COMPLETE",
            "hostname": "host.pci1",
            "exitCode": 0,
        }
        group = {
            "attemptIndex": 0,
            "attemptEpoch": 0,
            "numTasks": 1,
            "numFailedTasks": 0,
            "numShrunkTasks": 0,
            "onElasticCapacity": False,
            "state": "COMPLETE",
            "taskExecutionAttempts": {"task": [task]},
        }
        status = {
            "status": "ok",
            "data": {
                "state": "COMPLETE",
                "numRestarts": 0,
                "latestAttempt": {
                    "attemptIndex": 0,
                    "state": "COMPLETE",
                    "taskGroupExecutionAttempts": {"training": [group]},
                },
            },
        }
        summary = _scheduler_summary(status, expected_nodes=1)
        self.assertFalse(summary["continuity_errors"])
        task.pop("exitCode")
        summary = _scheduler_summary(status, expected_nodes=1)
        self.assertIn(
            "terminal task exit code is missing or nonzero",
            summary["continuity_errors"],
        )

    def test_package_evidence_records_metadata_and_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "METADATA").write_text("metadata\n")
            script = root / "run.sh"
            script.write_text("#!/bin/sh\n")
            script.chmod(0o755)
            evidence = _package_evidence(root)
            self.assertEqual(evidence["metadata_files"], ["METADATA"])
            self.assertEqual(evidence["files"]["run.sh"]["mode"], "0755")
            self.assertTrue(evidence["files"]["run.sh"]["executable"])


if __name__ == "__main__":
    unittest.main()
