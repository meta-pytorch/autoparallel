from __future__ import annotations

import json
import warnings
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch

from autoparallel.cost_models.collective_runtime_estimation import (
    get_nccl_topo_config,
    set_nccl_topo_config,
)
from autoparallel.cost_models.nccl_cost_model import detect_nccl_topo_config
from autoparallel.graph_passes.auto_bucketing import (
    aten_autobucketing_config,
    aten_autobucketing_reordering_pass,
)
from autoparallel.graph_passes.debug_helpers import (
    create_execution_trace,
    make_custom_runtime_estimation,
)
from autoparallel.graph_passes.estimate_graph_metrics import estimate_graph_metrics

_DETECT_TOPOLOGY = object()
_REORDERING_FIELDS = (
    "max_in_flight_gb",
    "compute_overlap_multipler",
    "max_coll_distance",
    "max_compute_pre_fetch",
    "max_topo_span",
    "collective_bucketing",
)


def classify_graph_phase(graph: torch.fx.Graph) -> str:
    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    if any(
        node.name.startswith("tangents_") or str(node.target).startswith("tangents_")
        for node in placeholders
    ):
        return "backward"

    tags = {
        node.meta.get("partitioner_tag")
        for node in graph.nodes
        if node.meta.get("partitioner_tag") in {"is_forward", "is_backward"}
    }
    if tags == {"is_backward"}:
        return "backward"
    if "is_backward" in tags:
        raise RuntimeError(
            "Cannot classify a graph with mixed partitioner tags and no tangent "
            "placeholder"
        )
    return "forward"


class ReorderedMetricsCollector:
    """Install AutoParallel's ATen reordering pass and collect graph estimates.

    Keep this context active around AutoParallel planning, torch.compile, and the
    forward/backward invocation. This preserves one communication cost model for
    both the ILP and the scheduled graph estimates. Compile with Inductor overlap
    scheduling disabled so this collector's pass is the only scheduler.
    """

    def __init__(
        self,
        mesh: Any,
        *,
        nccl_topology: Any = _DETECT_TOPOLOGY,
        reordering_overrides: Mapping[str, Any] | None = None,
        trace_dir: str | Path | None = None,
    ) -> None:
        self.mesh = mesh
        self._requested_topology = nccl_topology
        self._reordering_overrides = dict(reordering_overrides or {})
        self.trace_dir = Path(trace_dir) if trace_dir is not None else None
        self.records: list[dict[str, Any]] = []
        self.topology_config: Any = None
        self.cost_model: Any = None
        self.cost_model_status = "not_configured"
        self._active = False
        self._config_patch = None
        self._previous_topology = None
        self._previous_max_topo_span = None
        self.reordering_config: Any = None
        self.runtime_estimator: Any = None

    def __enter__(self) -> "ReorderedMetricsCollector":
        if self._active:
            raise RuntimeError("ReorderedMetricsCollector is not reentrant")
        if torch._inductor.config.post_grad_custom_post_pass is not None:
            raise RuntimeError(
                "post_grad_custom_post_pass is already configured; compose the "
                "model-specific pass explicitly before using this collector"
            )

        unknown = set(self._reordering_overrides) - set(_REORDERING_FIELDS)
        if unknown:
            raise ValueError(f"Unknown reordering options: {sorted(unknown)}")

        detected = self._requested_topology is _DETECT_TOPOLOGY
        if detected:
            self.topology_config = detect_nccl_topo_config(self.mesh)
        else:
            self.topology_config = self._requested_topology

        if self.topology_config is None:
            self.cost_model = "default"
            self.cost_model_status = (
                "pytorch_default_fallback" if detected else "pytorch_default_explicit"
            )
            if detected:
                warnings.warn(
                    "NCCL topology detection returned None; planning and scheduled "
                    "metrics will use the PyTorch default communication cost model",
                    RuntimeWarning,
                    stacklevel=2,
                )
        else:
            self.cost_model = self.topology_config
            self.cost_model_status = "nccl_detected" if detected else "nccl_explicit"

        runtime_estimator = make_custom_runtime_estimation(self.mesh)
        reordering_values = {
            name: getattr(aten_autobucketing_config, name)
            for name in _REORDERING_FIELDS
        }
        reordering_values.update(self._reordering_overrides)
        reordering_values.update(
            custom_runtime_estimation=runtime_estimator,
            save_trace=False,
            _counter=0,
        )
        self.reordering_config = SimpleNamespace(**reordering_values)
        self.runtime_estimator = runtime_estimator

        self._previous_topology = get_nccl_topo_config()
        self._previous_max_topo_span = aten_autobucketing_config.max_topo_span
        try:
            set_nccl_topo_config(self.topology_config)
            aten_autobucketing_config.max_topo_span = (
                self.reordering_config.max_topo_span
            )
            self._config_patch = torch._inductor.config.patch(
                {
                    "reorder_for_peak_memory": False,
                    "reorder_for_compute_comm_overlap": False,
                    "post_grad_custom_post_pass": self._post_grad_pass,
                }
            )
            self._config_patch.__enter__()
        except BaseException:
            set_nccl_topo_config(self._previous_topology)
            aten_autobucketing_config.max_topo_span = self._previous_max_topo_span
            raise
        self._active = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            assert self._config_patch is not None
            self._config_patch.__exit__(exc_type, exc_value, traceback)
        finally:
            set_nccl_topo_config(self._previous_topology)
            aten_autobucketing_config.max_topo_span = self._previous_max_topo_span
            self._active = False

    def _post_grad_pass(self, graph: torch.fx.Graph) -> torch.fx.GraphModule:
        if (
            torch._inductor.config.aten_distributed_optimizations.enable_overlap_scheduling
        ):
            raise RuntimeError(
                "ReorderedMetricsCollector must be the only overlap scheduler; "
                "use autoparallel_backend(..., overlap_scheduling=False)"
            )

        phase = classify_graph_phase(graph)
        reordered = aten_autobucketing_reordering_pass(
            graph, configs=self.reordering_config
        )
        metrics = estimate_graph_metrics(reordered, self.runtime_estimator)
        phase_index = sum(record["phase"] == phase for record in self.records)
        trace_path = None
        if self.trace_dir is not None:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = self.trace_dir / f"{phase}_{phase_index}.json"
            create_execution_trace(
                reordered,
                self.runtime_estimator,
                name=f"autoparallel_{phase}",
                file_path=str(trace_path),
            )

        record = {
            "phase": phase,
            "compile_index": len(self.records),
            "phase_index": phase_index,
            "placeholders": _placeholder_signature(graph),
            **asdict(metrics),
        }
        if trace_path is not None:
            record["trace"] = str(trace_path)
        self.records.append(record)
        return reordered

    def require_forward_backward(self) -> None:
        phases = {record["phase"] for record in self.records}
        missing = {"forward", "backward"} - phases
        if missing:
            raise RuntimeError(f"Missing graph metrics for phases: {sorted(missing)}")

    def to_dict(self) -> dict[str, Any]:
        reordering = {
            name: getattr(self.reordering_config, name) for name in _REORDERING_FIELDS
        }
        return {
            "cost_model": {
                "status": self.cost_model_status,
                "topology": repr(self.topology_config),
            },
            "reordering": reordering,
            "graphs": list(self.records),
        }

    def write_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as file:
            json.dump(self.to_dict(), file, indent=2)
            file.write("\n")


def _placeholder_signature(graph: torch.fx.Graph) -> list[dict[str, Any]]:
    signature = []
    for node in graph.nodes:
        if node.op != "placeholder":
            continue
        value = node.meta.get("val")
        shape = getattr(value, "shape", None)
        signature.append(
            {
                "name": node.name,
                "shape": [str(dim) for dim in shape] if shape is not None else None,
                "dtype": str(getattr(value, "dtype", None)),
            }
        )
    return signature
