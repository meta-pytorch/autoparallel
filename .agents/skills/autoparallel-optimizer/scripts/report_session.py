from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from types import TracebackType
from typing import Any, Callable, Literal, Mapping

import torch
from collect_reordered_metrics import ReorderedMetricsCollector
from run_report import (
    AutoParallelRunReport,
    ResolvedReportConfig,
    resolve_report_config,
)
from torch.distributed.tensor.placement_types import Replicate, Shard

_DETECT_TOPOLOGY = object()
_MODES = {"plan", "evaluate", "validate", "measure"}


class AutoParallelReportSession:
    """Keep reporting and reordered metrics in one failure-safe lifetime."""

    def __init__(
        self,
        mesh: Any,
        *,
        request: Mapping[str, Any],
        campaign: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
        compiler_config: Mapping[str, Any] | None = None,
        source_root: str | Path | None = None,
        external_source_roots: Mapping[str, str | Path] | None = None,
        entry_point: str | None = None,
        launch_config: Mapping[str, Any] | None = None,
        include_optimizer_state: bool = False,
        include_optimizer_replay: bool | None = None,
        parallelization: Mapping[str, Any] | None = None,
        mode: str = "evaluate",
        config: ResolvedReportConfig | None = None,
        profile_name: str | None = None,
        profiles_path: str | Path | None = None,
        report_root: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
        run_id: str | None = None,
        rank: int | None = None,
        report_rank: int = 0,
        nccl_topology: Any = _DETECT_TOPOLOGY,
        reordering_overrides: Mapping[str, Any] | None = None,
        include_placeholder_signatures: bool = False,
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"Unknown AutoParallel report mode: {mode}")
        self.mode = mode
        self.mesh = mesh
        self.rank = _resolve_rank(rank, environ)
        self.report_rank = report_rank
        self.is_reporter = self.rank == self.report_rank
        self.config = config or resolve_report_config(
            profile_name=profile_name,
            profiles_path=profiles_path,
            report_root=report_root,
            environ=environ,
        )
        self._scratch: TemporaryDirectory[str] | None = None
        self._scratch_path: Path | None = None
        self.include_optimizer_state = include_optimizer_state
        if include_optimizer_replay is None:
            include_optimizer_replay = (
                self.config.profile.get("capture_level") == "standard"
            )
        self.include_optimizer_replay = include_optimizer_replay
        self.compiler_config = dict(compiler_config or {})
        self.report: AutoParallelRunReport | None = None
        if self.is_reporter:
            self._scratch = TemporaryDirectory(prefix="autoparallel-report-")
            self._scratch_path = Path(self._scratch.name)
            merged_provenance = dict(provenance or {})
            merged_provenance.update(
                _default_provenance(
                    compiler_config=compiler_config,
                    source_root=source_root,
                    external_source_roots=external_source_roots,
                    entry_point=entry_point,
                    launch_config=launch_config,
                )
            )
            self.report = AutoParallelRunReport(
                request=_request_with_campaign(request, campaign),
                provenance=merged_provenance,
                config=self.config,
                run_id=run_id,
                source_root=source_root,
                parallelization=parallelization,
            )
        collector_kwargs: dict[str, Any] = {
            "reordering_overrides": reordering_overrides,
            "include_placeholder_signatures": include_placeholder_signatures,
        }
        if nccl_topology is not _DETECT_TOPOLOGY:
            collector_kwargs["nccl_topology"] = nccl_topology
        self.collector = ReorderedMetricsCollector(mesh, **collector_kwargs)
        self._collector_entered = False
        self._active = False
        self._metrics_recorded = False
        self._plan_recorded = False
        self._measurement_recorded = False

    @property
    def path(self) -> Path | None:
        return self.report.path if self.report is not None else None

    @property
    def cost_model(self) -> Any:
        self._ensure_active()
        return self.collector.cost_model

    def __enter__(self) -> "AutoParallelReportSession":
        if self._active:
            raise RuntimeError("AutoParallelReportSession is not reentrant")
        try:
            self.collector.__enter__()
            self._collector_entered = True
            self._active = True
            self._validate_reordering_config()
            if self.report is not None:
                self.report.record_runtime_context(
                    self.collector.to_dict()["cost_model"]
                )
            self._enforce_cost_model_policy()
            self._record_topology_findings()
        except BaseException as error:
            if self._collector_entered:
                self.collector.__exit__(type(error), error, error.__traceback__)
                self._collector_entered = False
            self._record_failure(error, error.__traceback__)
            self._cleanup()
            raise
        return self

    def _validate_reordering_config(self) -> None:
        declared_to_effective = {
            "metric_reordering": "enabled",
            "metric_collective_bucketing": "collective_bucketing",
            "metric_max_topo_span": "max_topo_span",
            "metric_bucket_all_all_reduce_groups": "bucket_all_all_reduce_groups",
        }
        for declared_name, effective_name in declared_to_effective.items():
            if declared_name not in self.compiler_config:
                continue
            declared = self.compiler_config[declared_name]
            effective = getattr(self.collector.reordering_config, effective_name)
            if declared != effective:
                raise ValueError(
                    f"Declared {declared_name}={declared!r} does not match "
                    f"effective {effective_name}={effective!r}"
                )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ) -> Literal[False]:
        collector_error = None
        if self._collector_entered:
            try:
                self.collector.__exit__(exc_type, exc_value, exc_traceback)
            except BaseException as error:
                collector_error = error
            self._collector_entered = False

        effective_error = collector_error or exc_value
        effective_traceback = (
            collector_error.__traceback__
            if collector_error is not None
            else exc_traceback
        )
        if effective_error is not None:
            try:
                self._record_failure(effective_error, effective_traceback)
            finally:
                self._cleanup()
            if collector_error is not None:
                raise collector_error
            return False

        try:
            if self.mode != "plan":
                self.record_metrics()
            if self.report is not None and not self.report.finalized:
                routing = self.report.report["parallelization"]
                if routing is not None and routing["autoparallel"] is None:
                    self.report.record_autoparallel_outcome(
                        "inconclusive",
                        reason_codes=("ACCEPTANCE_NOT_EVALUATED",),
                    )
                self.report.complete()
        except BaseException as error:
            self._record_failure(error, error.__traceback__)
            raise
        finally:
            self._cleanup()
        return False

    def record_plan(self, optimizer: Any) -> None:
        self._ensure_active()
        if self._plan_recorded:
            raise RuntimeError("This report session already contains a plan")
        if self.report is not None:
            assert self._scratch_path is not None
            placements_path = self._scratch_path / "placements.json"
            optimizer.save_placements(placements_path)
            optimizer_state_path = None
            if self.include_optimizer_state:
                optimizer_state_path = self._scratch_path / "optimizer.ap"
                optimizer.save(optimizer_state_path)
            optimizer_replay_path = None
            if self.include_optimizer_replay:
                optimizer_replay_path = self._scratch_path / "optimizer-replay.pt.gz"
                optimizer.save_replay(optimizer_replay_path)
            optimizer_json = optimizer.get_json()
            strategy_space = _strategy_space_summary(optimizer)
            if strategy_space is not None:
                optimizer_json["summary"]["strategy_space"] = strategy_space
            self.report.mark_stage("captured", evidence_level="captured")
            self.report.record_plan(
                optimizer_json,
                placements_path=placements_path,
            )
            if optimizer_state_path is not None:
                self.report.record_optimizer_state(optimizer_state_path)
            if optimizer_replay_path is not None:
                self.report.record_optimizer_replay(optimizer_replay_path)
            self._record_plan_findings(optimizer_json)
        self._plan_recorded = True

    def mark_substage(self, substage: str) -> None:
        self._ensure_active()
        if self.report is not None:
            self.report.mark_substage(substage)

    def mark_lowered(self) -> None:
        self._ensure_active()
        if self.report is not None:
            self.report.mark_stage("lowered", evidence_level="lowered")

    def record_autoparallel_outcome(
        self,
        outcome: str,
        *,
        reason_codes: list[str] | tuple[str, ...] = (),
        details: Mapping[str, Any] | None = None,
        related_run_id: str | None = None,
    ) -> None:
        self._ensure_active()
        if self.report is not None:
            self.report.record_autoparallel_outcome(
                outcome,
                reason_codes=reason_codes,
                details=details,
                related_run_id=related_run_id,
            )

    def record_fallback_outcome(
        self,
        strategy: str,
        outcome: str,
        *,
        reason_codes: list[str] | tuple[str, ...] = (),
        evidence_level: str = "none",
        details: Mapping[str, Any] | None = None,
        related_run_id: str | None = None,
    ) -> None:
        self._ensure_active()
        if self.report is not None:
            self.report.record_fallback_outcome(
                strategy,
                outcome,
                reason_codes=reason_codes,
                evidence_level=evidence_level,
                details=details,
                related_run_id=related_run_id,
            )

    def record_competitiveness_assessment(
        self,
        verdict: str,
        *,
        evidence: str,
        autoparallel_estimate: Mapping[str, Any] | None = None,
        baselines: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] = (),
        reason_codes: list[str] | tuple[str, ...] = (),
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self._ensure_active()
        if self.report is not None:
            self.report.record_competitiveness_assessment(
                verdict,
                evidence=evidence,
                autoparallel_estimate=autoparallel_estimate,
                baselines=baselines,
                reason_codes=reason_codes,
                details=details,
            )

    def record_parallelization_decision(
        self,
        selected_strategy: str,
        status: str,
        *,
        reason_codes: list[str] | tuple[str, ...] = (),
    ) -> None:
        self._ensure_active()
        if self.report is not None:
            self.report.record_parallelization_decision(
                selected_strategy,
                status,
                reason_codes=reason_codes,
            )

    def record_metrics(self) -> None:
        self._ensure_active()
        if self._metrics_recorded:
            return
        self.collector.require_forward_backward()
        if self.report is not None:
            metrics = self.collector.to_dict()
            self.report.record_metrics(metrics)
            self._record_metric_findings(metrics)
        self._metrics_recorded = True

    def record_validation(
        self,
        name: str,
        result: Mapping[str, Any],
        *,
        evidence_level: str,
    ) -> None:
        self._ensure_active()
        if self.mode != "plan":
            self.record_metrics()
        if self.report is not None:
            self.report.record_validation(
                name,
                result,
                evidence_level=evidence_level,
            )

    def measure_step(
        self,
        name: str,
        step: Callable[[], Any],
        *,
        warmup_iterations: int = 3,
        measured_iterations: int = 10,
        optimizer_update: bool = False,
        boundary_details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Measure an AutoParallel training step on the target CUDA devices.

        ``step`` must perform the declared boundary on every rank and reset
        iteration-local state itself.
        """

        self._ensure_active()
        if self.mode != "measure":
            raise RuntimeError("measure_step requires report mode='measure'")
        if self._measurement_recorded:
            raise RuntimeError("This report session already contains a measurement")
        if warmup_iterations < 0:
            raise ValueError("warmup_iterations must be non-negative")
        if measured_iterations < 1:
            raise ValueError("measured_iterations must be positive")
        if not torch.cuda.is_available():
            raise RuntimeError("measure_step requires CUDA target execution")

        device = torch.cuda.current_device()
        warmup_wall_times = [0.0, 0.0]
        if warmup_iterations:
            warmup_started = time.perf_counter()
            step()
            torch.cuda.synchronize(device)
            first_warmup_wall_time = time.perf_counter() - warmup_started
            for _ in range(warmup_iterations - 1):
                step()
            torch.cuda.synchronize(device)
            total_warmup_wall_time = time.perf_counter() - warmup_started
            warmup_wall_times = _distributed_max(
                [first_warmup_wall_time, total_warmup_wall_time], device
            )
        else:
            torch.cuda.synchronize(device)
        self.collector.require_forward_backward()
        compiled_graph_count = len(self.collector.records)
        _distributed_barrier()
        torch.cuda.synchronize(device)

        baseline_allocated = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        events: list[tuple[Any, Any]] = []
        for _ in range(measured_iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            step()
            torch.cuda.synchronize(device)
            end.record()
            events.append((start, end))
        torch.cuda.synchronize(device)
        if len(self.collector.records) != compiled_graph_count:
            raise RuntimeError(
                "Graph specialization occurred during measured iterations; "
                "increase warmup or stabilize the measured inputs"
            )

        samples_us = [start.elapsed_time(end) * 1000.0 for start, end in events]
        memory_values = [
            float(baseline_allocated),
            float(baseline_reserved),
            float(torch.cuda.max_memory_allocated(device)),
            float(torch.cuda.max_memory_reserved(device)),
        ]
        samples_us = _distributed_max(samples_us, device)
        memory_values = _distributed_max(memory_values, device)
        world_size = _distributed_world_size()
        estimate = _latest_step_estimate(self.collector.records)
        boundary = {
            "forward": True,
            "backward": True,
            "optimizer_update": optimizer_update,
        }
        for key, value in (boundary_details or {}).items():
            if key in boundary:
                raise ValueError(f"boundary_details cannot override {key}")
            boundary[str(key)] = value
        result: dict[str, Any] = {
            "kind": "autoparallel_target_step",
            "boundary": boundary,
            "hardware": {
                "device_type": "cuda",
                "device_name": torch.cuda.get_device_name(device),
                "local_device_index": device,
                "world_size": world_size,
                "rank_reduction": "max" if world_size > 1 else "local",
            },
            "timing": {
                "method": "cuda_events_with_device_synchronize",
                "warmup_iterations": warmup_iterations,
                "warmup_wall_time": {
                    "method": "host_monotonic_clock_with_device_synchronize",
                    "rank_reduction": "max" if world_size > 1 else "local",
                    "first_iteration_seconds": warmup_wall_times[0],
                    "total_seconds": warmup_wall_times[1],
                    "includes_lazy_compilation_when_applicable": True,
                },
                "measured_iterations": measured_iterations,
                "samples_us": samples_us,
                **_sample_statistics(samples_us),
            },
            "memory": {
                "method": "torch.cuda peak memory statistics",
                "baseline_allocated_bytes": int(memory_values[0]),
                "baseline_reserved_bytes": int(memory_values[1]),
                "peak_allocated_bytes": int(memory_values[2]),
                "peak_reserved_bytes": int(memory_values[3]),
            },
            "estimate": estimate,
            "comparisons": [],
        }
        configuration_hash = None
        if self.report is not None:
            configuration_hash = self.report.report["provenance"].get(
                "configuration_sha256"
            )
        if configuration_hash is not None:
            measurement_configuration = {
                "report_configuration_sha256": configuration_hash,
                "kind": result["kind"],
                "boundary": result["boundary"],
                "timing_method": result["timing"]["method"],
                "warmup_iterations": warmup_iterations,
                "measured_iterations": measured_iterations,
                "rank_reduction": result["hardware"]["rank_reduction"],
            }
            measurement_configuration_hash = _sha256_bytes(
                json.dumps(
                    measurement_configuration,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            )
            result["measurement_configuration"] = measurement_configuration
            result["measurement_configuration_sha256"] = measurement_configuration_hash
            result["comparisons"] = [
                {
                    "configuration_hash": measurement_configuration_hash,
                    "metric": "step_time",
                    "unit": "us",
                    "estimated": estimate["step_time_us"],
                    "observed": result["timing"]["mean_us"],
                },
                {
                    "configuration_hash": measurement_configuration_hash,
                    "metric": "step_peak_allocated_memory",
                    "unit": "bytes",
                    "estimated": estimate["peak_memory_bytes"],
                    "observed": result["memory"]["peak_allocated_bytes"],
                },
            ]
        self.record_validation(name, result, evidence_level="measured")
        self._measurement_recorded = True
        return result

    def add_finding(
        self,
        code: str,
        message: str,
        *,
        severity: str = "warning",
        context: Mapping[str, Any] | None = None,
        suggested_action: str | None = None,
    ) -> None:
        self._ensure_active()
        if self.report is not None:
            self.report.add_finding(
                code,
                message,
                severity=severity,
                context=context,
                suggested_action=suggested_action,
            )

    def _enforce_cost_model_policy(self) -> None:
        cost_model = self.collector.to_dict()["cost_model"]
        if not cost_model["status"].startswith("pytorch_default"):
            return
        policy = self.config.profile.get("policies", {}).get("fallback_costs", "warn")
        severity = "error" if policy == "reject" else "warning"
        if self.report is not None:
            self.report.add_finding(
                "FALLBACK_COST_PATH",
                "No NCCL topology is active; the PyTorch fallback cost model was selected",
                severity=severity,
                context=cost_model,
                suggested_action="Provide or detect an NCCL topology before planning",
            )
        if policy == "reject":
            raise RuntimeError(
                "The selected cluster profile rejects the PyTorch fallback cost model"
            )

    def _record_topology_findings(self) -> None:
        if self.report is None:
            return
        try:
            from autoparallel.cost_models.nccl_cost_model import (
                NCCLTopoConfig,
                nccl_cost_coverage,
            )

            topology = self.collector.cost_model
            if not isinstance(topology, NCCLTopoConfig):
                return
            coverage = tuple(nccl_cost_coverage(topology, tuple(self.mesh.shape)))
        except (ImportError, AttributeError, TypeError, ValueError):
            return
        uncertain = sorted(
            {term for term in coverage if term in {"algo_loop", "extrapolated"}}
        )
        if not uncertain:
            return
        self.report.add_finding(
            "EXTRAPOLATED_CRITICAL_TERM",
            "The selected topology uses uncalibrated or extrapolated collective costs",
            context={
                "mesh_shape": list(self.mesh.shape),
                "coverage": list(coverage),
                "uncertain_terms": uncertain,
                "topology": str(topology),
            },
            suggested_action=(
                "Benchmark the reported collective coverage before deployment ranking"
            ),
        )

    def _record_plan_findings(self, optimizer_json: Mapping[str, Any]) -> None:
        if self.report is None:
            return
        diagnostics = self.config.profile.get("diagnostics", {})
        summary = optimizer_json.get("summary", {})
        total = summary.get("total")
        communication = summary.get("comm")
        threshold = float(diagnostics.get("high_communication_fraction", 0.2))
        if _is_number(total) and _is_number(communication) and total > 0:
            fraction = float(communication) / float(total)
            if fraction >= threshold:
                self.report.add_finding(
                    "HIGH_COMMUNICATION_FRACTION",
                    "Communication is a large fraction of the selected ILP cost",
                    context={
                        "communication_fraction": fraction,
                        "threshold": threshold,
                        "communication_time_us": communication,
                        "total_time_us": total,
                        "largest_edges": _largest_communication_edges(optimizer_json),
                    },
                    suggested_action=(
                        "Collect reordered metrics and inspect the largest redistributions"
                    ),
                )

        storage = summary.get("parameter_storage", {})
        fraction = storage.get("local_to_global_fraction")
        replicated_threshold = float(
            diagnostics.get("replicated_parameter_fraction", 0.999)
        )
        replicate_policy = self.config.profile.get("policies", {}).get(
            "replicate_only", "warn"
        )
        if (
            _is_number(fraction)
            and fraction >= replicated_threshold
            and storage.get("global_bytes", 0) > 0
            and replicate_policy != "ignore"
        ):
            self.report.add_finding(
                "REPLICATED_PARAMETER_STORAGE",
                "The selected plan keeps persistent parameter storage replicated",
                severity="error" if replicate_policy == "reject" else "warning",
                context={
                    "local_to_global_fraction": fraction,
                    "threshold": replicated_threshold,
                    "global_bytes": storage.get("global_bytes"),
                    "local_bytes": storage.get("local_bytes"),
                },
                suggested_action=(
                    "Check memory viability and sharding coverage before target evaluation"
                ),
            )

        skipped_groups = summary.get("clustering", {}).get("skipped_groups", [])
        if skipped_groups:
            self.report.add_finding(
                "CLUSTERING_GROUPS_SKIPPED",
                "Repeated-subgraph clustering skipped malformed groups",
                context={
                    "count": len(skipped_groups),
                    "groups": skipped_groups[:10],
                },
                suggested_action="Minimize a skipped group and add a clustering regression",
            )

        strategy_space = summary.get("strategy_space", {})
        selected_with_options = strategy_space.get(
            "selected_replicated_with_sharded_output_option"
        )
        selected_fraction = strategy_space.get(
            "selected_replicated_with_sharded_output_option_fraction"
        )
        selection_threshold = float(
            diagnostics.get("selected_replication_with_sharded_options_fraction", 0.5)
        )
        if (
            _is_number(selected_with_options)
            and selected_with_options > 0
            and _is_number(selected_fraction)
            and selected_fraction >= selection_threshold
        ):
            self.report.add_finding(
                "SELECTED_REPLICATION_WITH_SHARDED_OPTIONS",
                "Most selected tensor nodes are replicated despite available sharded strategies",
                context={
                    **strategy_space,
                    "threshold": selection_threshold,
                },
                suggested_action=(
                    "Inspect workload representativeness, constraints, and cost-model saturation before adding propagation rules"
                ),
            )

    def _record_metric_findings(self, metrics: Mapping[str, Any]) -> None:
        if self.report is None:
            return
        diagnostics = self.config.profile.get("diagnostics", {})
        minimum_count = int(
            diagnostics.get("dependency_bound_collective_minimum_count", 8)
        )
        threshold = float(diagnostics.get("dependency_bound_collective_fraction", 0.5))
        phase_context = {}
        raw_count = 0
        post_bucket_count = 0
        analyzed_count = 0
        dependency_bound_count = 0
        for graph in metrics.get("graphs", []):
            bucketing = graph.get("bucketing_diagnostics")
            if not isinstance(bucketing, Mapping):
                continue
            before = bucketing.get("collective_counts_before", {})
            after = bucketing.get("collective_counts_after", {})
            phase_raw = int(before.get("all_reduce", 0))
            phase_post = int(after.get("all_reduce", 0))
            phase_analyzed = 0
            phase_dependency_bound = 0
            for call in bucketing.get("greedy_bucket_calls", []):
                type_counts = call.get("collective_type_counts", {})
                if any("all_reduce" in name for name in type_counts):
                    phase_analyzed += int(call.get("candidate_count", 0))
                    phase_dependency_bound += int(
                        call.get("dependency_blocked_count", 0)
                    )
            raw_count += phase_raw
            post_bucket_count += phase_post
            analyzed_count += phase_analyzed
            dependency_bound_count += phase_dependency_bound
            phase_context[str(graph.get("phase", "unknown"))] = {
                "raw_all_reduce_count": phase_raw,
                "post_bucket_all_reduce_count": phase_post,
                "analyzed_all_reduce_count": phase_analyzed,
                "dependency_bound_count": phase_dependency_bound,
            }
        if raw_count < minimum_count:
            return
        request = self.report.report.get("request") or {}
        mesh_shape = request.get("mesh", {}).get("shape", [])
        omitted_count = max(raw_count - analyzed_count, 0)
        omitted_fraction = omitted_count / raw_count
        coverage_threshold = float(
            diagnostics.get("multigroup_collective_omission_fraction", 0.2)
        )
        reordering = metrics.get("reordering") or {}
        expanded_group_discovery = bool(
            reordering.get("bucket_all_all_reduce_groups", False)
        )
        if (
            len(mesh_shape) > 1
            and not expanded_group_discovery
            and omitted_fraction >= coverage_threshold
        ):
            self.report.add_finding(
                "MULTIGROUP_BUCKETING_COVERAGE_GAP",
                "Many all-reduces on a multi-dimensional mesh were not analyzed for bucketing",
                context={
                    "raw_all_reduce_count": raw_count,
                    "analyzed_all_reduce_count": analyzed_count,
                    "omitted_all_reduce_count": omitted_count,
                    "omitted_all_reduce_fraction": omitted_fraction,
                    "threshold": coverage_threshold,
                    "minimum_count": minimum_count,
                    "mesh_shape": list(mesh_shape),
                    "by_phase": phase_context,
                },
                suggested_action=(
                    "Measure a bounded all-group all-reduce bucketing counterfactual; "
                    "promote it only if repeated runtime improves without liveness issues"
                ),
            )
        fraction = dependency_bound_count / raw_count
        if fraction < threshold:
            return
        self.report.add_finding(
            "DEPENDENCY_BOUND_TENSOR_PARALLEL_COLLECTIVES",
            "Most all-reduce candidates are dependency-bound and cannot be horizontally bucketed",
            context={
                "raw_all_reduce_count": raw_count,
                "post_bucket_all_reduce_count": post_bucket_count,
                "dependency_bound_count": dependency_bound_count,
                "dependency_bound_fraction": fraction,
                "threshold": threshold,
                "minimum_count": minimum_count,
                "by_phase": phase_context,
            },
            suggested_action=(
                "Inspect partial-sum matmuls and rerank a bounded placement shortlist; "
                "do not treat dependency-bound activation collectives as bucketable "
                "gradient reductions"
            ),
        )

    def _ensure_active(self) -> None:
        if not self._active:
            raise RuntimeError("AutoParallelReportSession is not active")

    def _record_failure(
        self,
        error: BaseException,
        error_traceback: TracebackType | None,
    ) -> None:
        if self.report is not None and not self.report.finalized:
            routing = self.report.report["parallelization"]
            if routing is not None and routing["autoparallel"] is None:
                substage = self.report.report["planning"].get("current_substage")
                message = str(error).lower()
                reason_code = "AUTOPARALLEL_EXCEPTION"
                inferred_capture = substage is None and (
                    type(error).__name__ == "DynamicOutputShapeException"
                    or (
                        type(error).__name__ == "Unsupported"
                        and "data-dependent branching" in message
                    )
                )
                if substage == "capture" or inferred_capture:
                    reason_code = (
                        "CAPTURE_DYNAMIC_OUTPUT_SHAPE_UNSUPPORTED"
                        if type(error).__name__ == "DynamicOutputShapeException"
                        or "data-dependent expression" in message
                        or "dynamic output shape" in message
                        else "CAPTURE_UNSUPPORTED"
                    )
                elif substage == "solve":
                    if isinstance(error, TimeoutError):
                        reason_code = "SHARDING_SOLVER_TIMEOUT"
                    elif "could not find a feasible solution" in message:
                        reason_code = "SHARDING_SOLVER_INFEASIBLE"
                elif substage == "lower":
                    reason_code = "LOWERING_FAILED"
                elif substage == "compile":
                    reason_code = "AUTOPARALLEL_COMPILE_FAILED"
                self.report.record_autoparallel_outcome(
                    "failed",
                    reason_codes=(reason_code,),
                    details={
                        "exception_type": type(error).__name__,
                        "substage": substage,
                        "inferred_substage": "capture" if inferred_capture else None,
                    },
                )
            self.report.record_failure(error, error_traceback)

    def _cleanup(self) -> None:
        self._active = False
        if self._scratch is not None:
            self._scratch.cleanup()


def _default_provenance(
    *,
    compiler_config: Mapping[str, Any] | None,
    source_root: str | Path | None,
    external_source_roots: Mapping[str, str | Path] | None,
    entry_point: str | None,
    launch_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if entry_point is not None and not entry_point.strip():
        raise ValueError("entry_point must not be empty")
    source = _source_state(source_root)
    external_sources = {
        str(name): _source_state(path)
        for name, path in sorted((external_source_roots or {}).items())
    }
    missing_reproduction_fields = []
    if entry_point is None:
        missing_reproduction_fields.append("entry_point")
    if launch_config is None:
        missing_reproduction_fields.append("launch_config")
    if source.get("git_revision") is None:
        missing_reproduction_fields.append("source.git_revision")
    for name, external_source in external_sources.items():
        if external_source.get("git_revision") is None:
            missing_reproduction_fields.append(f"external_sources.{name}.git_revision")
    provenance: dict[str, Any] = {
        "report_session_version": 5,
        "pytorch_build": str(torch.__version__),
        "python_version": platform.python_version(),
        "cuda_build": torch.version.cuda,
        "compiler_config": dict(compiler_config or {}),
        "reproduction": {
            "entry_point": entry_point,
            "launch_config": dict(launch_config or {}),
            "source": source,
            "external_sources": external_sources,
            "complete": not missing_reproduction_fields,
            "missing": missing_reproduction_fields,
        },
    }
    git_revision = source.get("git_revision")
    if git_revision is not None:
        provenance["git_revision"] = git_revision
        provenance["working_tree_dirty"] = source["working_tree_dirty"]
    return provenance


def _source_state(source_root: str | Path | None) -> dict[str, Any]:
    root_result = _run_git(["rev-parse", "--show-toplevel"], source_root)
    if root_result is None:
        return {"version_control": "unavailable"}
    root = Path(root_result.decode().strip()).resolve()
    revision_result = _run_git(["rev-parse", "HEAD"], root)
    status_result = _run_git(
        ["status", "--porcelain=v1", "-z", "--untracked-files=normal"], root
    )
    diff_result = _run_git(["diff", "--binary", "HEAD", "--"], root)
    if revision_result is None or status_result is None or diff_result is None:
        return {"version_control": "git", "repository": root.name}

    revision = revision_result.decode().strip()
    untracked_count = sum(
        entry.startswith(b"?? ") for entry in status_result.split(b"\0") if entry
    )
    status_sha256 = _sha256_bytes(status_result)
    tracked_diff_sha256 = _sha256_bytes(diff_result)
    state_material = b"\0".join(
        [revision.encode(), status_sha256.encode(), tracked_diff_sha256.encode()]
    )
    return {
        "version_control": "git",
        "repository": root.name,
        "git_revision": revision,
        "working_tree_dirty": bool(status_result),
        "working_tree_state_sha256": _sha256_bytes(state_material),
        "git_status_sha256": status_sha256,
        "tracked_diff_sha256": tracked_diff_sha256,
        "untracked_file_count": untracked_count,
        "fingerprint_scope": "tracked contents and untracked paths",
    }


def _request_with_campaign(
    request: Mapping[str, Any], campaign: Mapping[str, Any] | None
) -> dict[str, Any]:
    merged = dict(request)
    if campaign is None:
        return merged
    if "campaign" in merged:
        raise ValueError("request already contains campaign metadata")
    campaign_data = dict(campaign)
    campaign_id = campaign_data.get("id")
    if not isinstance(campaign_id, str) or not campaign_id.strip():
        raise ValueError("campaign.id must be a non-empty string")
    expected_cases = campaign_data.get("expected_case_count")
    if expected_cases is not None and (
        not isinstance(expected_cases, int)
        or isinstance(expected_cases, bool)
        or expected_cases < 1
    ):
        raise ValueError("campaign.expected_case_count must be a positive integer")
    for field in ("suite", "case_id"):
        value = campaign_data.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"campaign.{field} must be a non-empty string")
    merged["campaign"] = campaign_data
    return merged


def _strategy_space_summary(optimizer: Any) -> dict[str, Any] | None:
    selected_keys = getattr(optimizer, "selected_keys", None)
    nodes = getattr(optimizer, "nodes", None)
    strategies = getattr(optimizer, "strats", None)
    resolve = getattr(optimizer, "_resolve_decision_var", None)
    if not isinstance(selected_keys, list) or nodes is None or strategies is None:
        return None
    if not callable(resolve):
        return None

    selected: dict[Any, Any] = {}
    for key in selected_keys:
        node = nodes[key[0]]
        selected.setdefault(node, resolve(key).strategy)

    total = 0
    with_sharded_option = 0
    selected_replicated = 0
    selected_replicated_with_sharded_option = 0
    replicate_only = 0
    operations: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for node, selected_strategy in selected.items():
        selected_specs = list(_iter_output_specs(selected_strategy.output_specs))
        if not selected_specs:
            continue
        options = getattr(strategies.get(node), "strategies", ())
        if not options:
            continue
        total += 1
        has_sharded_option = any(
            _output_has_shard(option.output_specs) for option in options
        )
        is_replicate_only = all(
            _output_is_replicated(option.output_specs) for option in options
        )
        is_selected_replicated = _output_is_replicated(selected_strategy.output_specs)
        with_sharded_option += int(has_sharded_option)
        replicate_only += int(is_replicate_only)
        selected_replicated += int(is_selected_replicated)
        if is_selected_replicated and has_sharded_option:
            selected_replicated_with_sharded_option += 1
            operation = str(node.target) if node.op == "call_function" else node.op
            operations[operation] += 1
            if len(examples) < 10:
                examples.append(
                    {
                        "name": node.name,
                        "operation": operation,
                        "shape": _node_shape(node),
                    }
                )

    if total == 0:
        return None
    return {
        "selected_tensor_node_count": total,
        "nodes_with_sharded_output_option": with_sharded_option,
        "nodes_with_sharded_output_option_fraction": with_sharded_option / total,
        "selected_replicated_node_count": selected_replicated,
        "selected_replicated_node_fraction": selected_replicated / total,
        "selected_replicated_with_sharded_output_option": (
            selected_replicated_with_sharded_option
        ),
        "selected_replicated_with_sharded_output_option_fraction": (
            selected_replicated_with_sharded_option / total
        ),
        "replicate_only_option_set_count": replicate_only,
        "top_selected_replicated_operations": [
            {"operation": operation, "count": count}
            for operation, count in operations.most_common(10)
        ],
        "examples": examples,
    }


def _iter_output_specs(value: Any):
    if hasattr(value, "placements"):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_output_specs(item)


def _output_has_shard(value: Any) -> bool:
    return any(
        isinstance(placement, Shard)
        for spec in _iter_output_specs(value)
        for placement in spec.placements
    )


def _output_is_replicated(value: Any) -> bool:
    specs = list(_iter_output_specs(value))
    return bool(specs) and all(
        isinstance(placement, Replicate)
        for spec in specs
        for placement in spec.placements
    )


def _node_shape(node: Any) -> list[Any] | None:
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        return None
    return [int(dim) if isinstance(dim, int) else str(dim) for dim in value.shape]


def _largest_communication_edges(
    optimizer_json: Mapping[str, Any], limit: int = 5
) -> list[dict[str, Any]]:
    edges = []
    for node in optimizer_json.get("nodes", []):
        for input_edge in node.get("inputs", []):
            cost = input_edge.get("comm_cost")
            if not _is_number(cost) or cost <= 0:
                continue
            edges.append(
                {
                    "cost_us": cost,
                    "producer": input_edge.get("name"),
                    "consumer": node.get("name"),
                    "operation": node.get("op"),
                    "source_placement": input_edge.get("src_placement"),
                    "destination_placement": input_edge.get("dst_placement"),
                    "source": node.get("source"),
                }
            )
    return sorted(edges, key=lambda edge: edge["cost_us"], reverse=True)[:limit]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _run_git(arguments: list[str], cwd: str | Path | None) -> bytes | None:
    result = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        capture_output=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _latest_step_estimate(records: list[dict[str, Any]]) -> dict[str, Any]:
    phases: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("phase") in {"forward", "backward"}:
            phases[record["phase"]] = record
    missing = {"forward", "backward"} - phases.keys()
    if missing:
        raise RuntimeError(
            f"Cannot compare measurement without graph metrics: {sorted(missing)}"
        )
    forward = phases["forward"]
    backward = phases["backward"]
    return {
        "selection": "latest_record_per_phase",
        "forward_compile_index": forward.get("compile_index"),
        "backward_compile_index": backward.get("compile_index"),
        "forward_time_us": forward["total_time"],
        "backward_time_us": backward["total_time"],
        "step_time_us": forward["total_time"] + backward["total_time"],
        "forward_peak_memory_bytes": forward["peak_memory"],
        "backward_peak_memory_bytes": backward["peak_memory"],
        "peak_memory_bytes": max(forward["peak_memory"], backward["peak_memory"]),
    }


def _sample_statistics(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "mean_us": statistics.mean(samples),
        "median_us": statistics.median(samples),
        "p95_us": ordered[p95_index],
        "min_us": ordered[0],
        "max_us": ordered[-1],
    }


def _distributed_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def _distributed_barrier() -> None:
    if _distributed_world_size() > 1:
        torch.distributed.barrier()


def _distributed_max(values: list[float], device: int) -> list[float]:
    if _distributed_world_size() == 1:
        return values
    tensor = torch.tensor(values, dtype=torch.float64, device=f"cuda:{device}")
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return tensor.cpu().tolist()


def _resolve_rank(rank: int | None, environ: Mapping[str, str] | None) -> int:
    if rank is not None:
        return rank
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    env = os.environ if environ is None else environ
    return int(env.get("RANK", "0"))
