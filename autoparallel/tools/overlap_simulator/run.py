#!/usr/bin/env python3
"""
Overlap Scheduling Experiments Runner

This script runs overlap scheduling experiments with various bucketing strategies
on different model variants and configurations.
"""

# Standard library imports
import argparse
import copy
import dataclasses
import json
import logging
import os
from math import inf
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third-party imports
import torch
import torch.distributed as dist
import torch.fx as fx
from torch import device, tensor
from torch._dynamo.testing import rand_strided
from torch._inductor.fx_passes.bucketing import is_wait_tensor
from torch.fx.operator_schemas import normalize_function
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.utils._dtype_abbrs import dtype_abbrs

# Local imports
import torch._dynamo.config
import torch._functorch.config
import torch._inductor.config
import torch._inductor.inductor_prims
import torch.fx.experimental._config

# Constants
DEFAULT_VARIANT = "llama3_8b_bw_256_2d_32"

# Launch overhead in microseconds
LAUNCH_OVERHEAD_US = 1
BYTES_PER_MB = 1024 * 1024
MS_TO_US_MULTIPLIER = 1000

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def configure_torch() -> None:
    """Configure torch settings for overlap scheduling experiments."""
    torch._inductor.config.allow_buffer_reuse = False
    torch._inductor.config.reorder_for_compute_comm_overlap = False
    torch._inductor.config.reorder_for_peak_memory = False
    torch._inductor.config.max_autotune = False
    torch._inductor.config.coordinate_descent_tuning = False
    torch._inductor.config.deterministic = False
    torch._inductor.config.aten_distributed_optimizations.collective_bucketing = True
    torch._inductor.config.triton.store_cubin = False
    torch._inductor.config.test_configs.runtime_triton_dtype_assert = False
    torch._functorch.config.functionalize_rng_ops = False
    torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
    torch._functorch.config.unlift_effect_tokens = False
    torch._functorch.config.selective_decompose = False


@dataclasses.dataclass
class Stats:
    """Statistics for graph collective operations."""
    num_ag: int  # Number of all-gather operations
    num_rs: int  # Number of reduce-scatter operations
    num_ar: int  # Number of all-reduce operations
    runtime: float  # Total runtime in milliseconds

    def __str__(self) -> str:
        return f"AG:{self.num_ag}, RS:{self.num_rs}, AR:{self.num_ar}, Runtime:{self.runtime:.2f}ms"


@dataclasses.dataclass
class VariantConfig:
    """Configuration for a model variant."""
    repro_class: type
    load_args_func: Callable
    get_mesh_sizes_func: Callable
    get_colls_file_func: Callable
    get_pg_names_func: Callable


class CollectiveEstimationParser:
    """Parser for collective estimation table files."""

    @staticmethod
    def parse_table(file_path: str) -> Dict[Tuple[str, int, str], Dict[int, float]]:
        """
        Parse the collectives estimations table file.

        Args:
            file_path: Path to the table file

        Returns:
            Dict mapping (group_name, group_size, collective_name) -> {size_mb: time_ms}
        """
        result: Dict[Tuple[str, int, str], Dict[int, float]] = {}

        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            logger.error(f"Collective estimation file not found: {file_path}")
            return result
        except Exception as e:
            logger.error(f"Error reading collective estimation file {file_path}: {e}")
            return result

        if len(lines) < 2:
            logger.warning(f"Collective estimation file {file_path} has insufficient data")
            return result

        # Parse header to get size columns
        header = lines[0]
        size_columns: List[int] = []
        for part in header.split():
            if part.endswith("MB"):
                try:
                    size_mb = int(part.replace("MB", ""))
                    size_columns.append(size_mb)
                except ValueError:
                    continue

        # Process data lines (skip separator line)
        for line_num, line in enumerate(lines[2:], start=3):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 3 + len(size_columns):
                logger.warning(f"Insufficient data in line {line_num} of {file_path}")
                continue

            try:
                group_name = parts[0]
                group_size = int(parts[1])
                collective = parts[2]

                size_to_time: Dict[int, float] = {}
                for i, size_mb in enumerate(size_columns):
                    time_ms = float(parts[3 + i])
                    size_to_time[size_mb] = time_ms

                result[(group_name, group_size, collective)] = size_to_time

            except (ValueError, IndexError) as e:
                logger.warning(f"Error parsing line {line_num} in {file_path}: {e}")
                continue

        logger.info(f"Parsed {len(result)} collective entries from {file_path}")
        return result

    @staticmethod
    def interpolate_time(size_to_time: Dict[int, float], size_mb: float) -> float:
        """
        Interpolate or extrapolate time for a given size in MB.

        Args:
            size_to_time: Mapping of size (MB) to time (ms)
            size_mb: Target size in MB

        Returns:
            Estimated time in milliseconds
        """
        if not size_to_time:
            return 0.0

        sorted_sizes = sorted(size_to_time.keys())

        # For sizes less than 1MB, use 1MB value
        if size_mb < 1.0:
            return size_to_time.get(1, size_to_time[sorted_sizes[0]])

        # Exact match
        size_int = int(size_mb)
        if size_int in size_to_time:
            return size_to_time[size_int]

        # Find surrounding points for interpolation
        lower_size = None
        upper_size = None

        for s in sorted_sizes:
            if s <= size_mb:
                lower_size = s
            if s >= size_mb and upper_size is None:
                upper_size = s

        # Extrapolation cases
        if lower_size is None:
            # Below minimum - use first two points
            if len(sorted_sizes) >= 2:
                s1, s2 = sorted_sizes[0], sorted_sizes[1]
                t1, t2 = size_to_time[s1], size_to_time[s2]
                slope = (t2 - t1) / (s2 - s1)
                return max(0.0, t1 + slope * (size_mb - s1))
            return size_to_time[sorted_sizes[0]]

        if upper_size is None:
            # Above maximum - use last two points
            if len(sorted_sizes) >= 2:
                s1, s2 = sorted_sizes[-2], sorted_sizes[-1]
                t1, t2 = size_to_time[s1], size_to_time[s2]
                slope = (t2 - t1) / (s2 - s1)
                return max(0.0, t2 + slope * (size_mb - s2))
            return size_to_time[sorted_sizes[-1]]

        # Interpolation between two points
        if lower_size == upper_size:
            return size_to_time[lower_size]

        t1, t2 = size_to_time[lower_size], size_to_time[upper_size]
        fraction = (size_mb - lower_size) / (upper_size - lower_size)
        return t1 + fraction * (t2 - t1)


class NodeEstimator:
    """Handles runtime estimation for nodes in the computation graph."""

    def __init__(self,
                 nodes_estimations_dict: Dict[fx.Node, float],
                 collective_table: Dict[Tuple[str, int, str], Dict[int, float]]):
        self.node_names_ests = {n.name: est for n, est in nodes_estimations_dict.items()}
        self.collective_table = collective_table

    @staticmethod
    def get_hint(x: Union[int, torch.SymInt]) -> Optional[int]:
        """Extract concrete int from SymInt if needed."""
        if isinstance(x, int):
            return x
        if hasattr(x, 'node') and hasattr(x.node, 'hint'):
            return x.node.hint
        return None

    @staticmethod
    def get_tensor_bytes(node: fx.Node) -> Optional[int]:
        """Get the size in bytes of the tensor produced by this node."""
        if "val" not in node.meta:
            return None

        t = node.meta["val"]
        if not isinstance(t, torch.Tensor):
            return None

        shape = [NodeEstimator.get_hint(dim) for dim in t.shape]
        if any(s is None for s in shape):
            return None

        numel = 1
        for dim in shape:
            numel *= dim
        return numel * t.dtype.itemsize

    def get_collective_info(self, node: fx.Node) -> Optional[Tuple[str, int, int, str]]:
        """
        Extract collective type, group_size, tensor bytes, and group_name.

        Returns:
            (collective_name, group_size, tensor_bytes, group_name) or None
        """
        if node.op != "call_function":
            return None

        target_str = str(node.target)
        collective_name = None
        group_size = None

        # Determine collective type and extract group_size
        if "all_gather_into_tensor" in target_str:
            collective_name = "all_gather_into_tensor"
            if len(node.args) >= 2:
                group_size = self.get_hint(node.args[1]) if hasattr(node.args[1], 'node') else node.args[1]
                if isinstance(node.args[1], int):
                    group_size = node.args[1]

        elif "reduce_scatter_tensor" in target_str:
            collective_name = "reduce_scatter_tensor"
            if len(node.args) >= 3:
                group_size = self.get_hint(node.args[2]) if hasattr(node.args[2], 'node') else node.args[2]
                if isinstance(node.args[2], int):
                    group_size = node.args[2]

        elif "all_reduce" in target_str:
            collective_name = "all_reduce"
            # No explicit group_size in args for all_reduce

        else:
            return None

        # Get tensor bytes from input tensor
        input_node = node.args[0] if node.args else None
        if not isinstance(input_node, fx.Node):
            return None

        tensor_bytes = self.get_tensor_bytes(input_node)
        if tensor_bytes is None:
            return None

        # Extract group_name
        try:
            group_name = get_group_name(node)
        except Exception as e:
            logger.warning(f"Failed to extract group name from node {node.name}: {e}")
            group_name = ""

        return (collective_name, group_size, tensor_bytes, group_name)

    def estimate(self, node: fx.Node) -> float:
        """
        Estimate execution time for a node in milliseconds.

        Args:
            node: FX node to estimate

        Returns:
            Estimated runtime in milliseconds
        """
        # Check pre-computed estimation (matmul, etc.)
        if node.name in self.node_names_ests:
            return self.node_names_ests[node.name]

        # Check if this is a collective operation
        coll_info = self.get_collective_info(node)
        if coll_info is not None:
            collective_name, group_size, tensor_bytes, node_group_name = coll_info
            size_mb = tensor_bytes / BYTES_PER_MB

            # Look up in the table
            if group_size is not None:
                for (table_group, gs, cn), size_to_time in self.collective_table.items():
                    if gs == group_size and cn == collective_name and table_group in node_group_name:
                        return CollectiveEstimationParser.interpolate_time(size_to_time, size_mb)
            else:
                # For all_reduce without explicit group_size
                for (table_group, gs, cn), size_to_time in self.collective_table.items():
                    if cn == collective_name and table_group in node_group_name:
                        return CollectiveEstimationParser.interpolate_time(size_to_time, size_mb)

        return 0.0


class TraceGenerator:
    """Generates execution traces for visualization."""

    @staticmethod
    def is_communication_node(node: fx.Node) -> bool:
        """Check if node is a communication operation."""
        return (node.op == "call_function" and
                isinstance(node.target, torch._ops.OpOverload) and
                node.target.namespace == "_c10d_functional")

    @staticmethod
    def get_tid(node: fx.Node) -> Union[int, str]:
        """Get thread ID for trace visualization."""
        if TraceGenerator.is_communication_node(node):
            if node.target == torch.ops._c10d_functional.wait_tensor.default:
                return 0
            return f"group-{node.args[-1]}"
        return 0

    @staticmethod
    def get_repr(arg: Any, mode: str = "full") -> Any:
        """Get representation of argument for trace."""
        def get_dtype_repr(dtype):
            return dtype_abbrs[dtype]

        if isinstance(arg, torch.Tensor):
            return {
                "shape": tuple(arg.shape),
                "dtype": get_dtype_repr(arg.dtype)
            }

        if isinstance(arg, (int, float, str)):
            return arg

        if isinstance(arg, torch.dtype):
            return get_dtype_repr(arg)

        if isinstance(arg, torch.fx.Node):
            if mode == "name_only" or "val" not in arg.meta:
                return f"fx node {arg.name}"
            elif mode == "full":
                return {"name": arg.name, "data": TraceGenerator.get_repr(arg.meta["val"])}
            elif mode == "content_only":
                return TraceGenerator.get_repr(arg.meta["val"])
            else:
                raise ValueError(f"Unknown mode {mode}")

        if isinstance(arg, (list, tuple)):
            return [TraceGenerator.get_repr(x, mode="name_only") for x in arg]

        if isinstance(arg, dict):
            return {k: TraceGenerator.get_repr(v, mode="name_only") for k, v in arg.items()}

        return f"arg {type(arg)}"

    @classmethod
    def generate_trace(cls,
                      gm: fx.GraphModule,
                      runtime_estimator: Callable[[fx.Node], float],
                      name: str) -> Dict[str, Any]:
        """
        Generate execution trace for visualization.

        Args:
            gm: Graph module to trace
            runtime_estimator: Function to estimate node runtime
            name: Name for the trace

        Returns:
            Trace dictionary for perfetto visualization
        """
        trace_events = []
        curr_time = {0: 0}
        global_time: Dict[fx.Node, int] = {}

        for node_idx, node in enumerate(gm.graph.nodes):
            dur_ms = runtime_estimator(node)
            dur = dur_ms * MS_TO_US_MULTIPLIER  # Convert to microseconds
            tid = cls.get_tid(node)

            if tid not in curr_time:
                curr_time[tid] = curr_time[0]

            event = {
                "ph": "X",
                "cat": "kernel",
                "name": str(node),
                "pid": 0,
                "tid": tid
            }

            if cls.is_communication_node(node):
                if tid == 0 and is_wait_tensor(node) and node.args[0].op != "placeholder":
                    # Sync with compute stream for wait tensor
                    comm_end_time = global_time.pop(node.args[0])
                    curr_time[tid] = max(curr_time[tid], comm_end_time)
                else:
                    curr_time[tid] = max(curr_time[0], curr_time[tid])

            event["ts"] = curr_time[tid]
            event["dur"] = dur
            curr_time[tid] += dur + LAUNCH_OVERHEAD_US

            if tid != 0:
                curr_time[0] += LAUNCH_OVERHEAD_US
                global_time[node] = curr_time[tid]

            # Add metadata
            args = {
                "order": node_idx,
                "output": cls.get_repr(node, mode="content_only"),
                "inputs": [cls.get_repr(arg) for arg in node.args]
            }
            event["args"] = args

            if dur > 0.0:
                trace_events.append(event)

        return {
            "traceEvents": trace_events,
            "traceName": f"{name}_trace.json"
        }


class ExperimentRunner:
    """Main experiment runner for overlap scheduling."""

    def __init__(self, variant_config: VariantConfig):
        self.variant_config = variant_config
        self.setup_process_groups()

    def setup_process_groups(self) -> None:
        """Set up fake process groups for simulation."""
        from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

        store = FakeStore()
        mesh_sizes = self.variant_config.get_mesh_sizes_func()
        world_size = 1
        for size in mesh_sizes:
            world_size *= size

        self.pg = dist.init_process_group(
            backend="fake",
            rank=0,
            world_size=world_size,
            store=store
        )

        mesh = DeviceMesh("fake", torch.arange(world_size).view(*mesh_sizes))
        pgs = []
        pg_names = self.variant_config.get_pg_names_func()

        for i, size in enumerate(mesh_sizes):
            pg = mesh.get_group(i)
            pgs.append(pg)

        torch._C._distributed_c10d._unregister_all_process_groups()
        for pg, pg_name in zip(pgs, pg_names):
            torch._C._distributed_c10d._register_process_group(pg_name, pg)

    def run_experiment(self, variant_name: str) -> Tuple[Stats, Stats]:
        """
        Run overlap scheduling experiment.

        Args:
            variant_name: Name of the variant being tested

        Returns:
            Tuple of (stats_before, stats_after) optimization
        """
        try:
            # Setup model and graph
            mod = self.variant_config.repro_class()

            with torch.no_grad():
                from torch.fx.experimental.proxy_tensor import make_fx
                from torch._subclasses.fake_tensor import FakeTensorMode
                from torch._dynamo.debug_utils import InputReader

                mode = FakeTensorMode()
                reader = InputReader()
                self.variant_config.load_args_func(reader)
                args = reader.args

                gm = make_fx(mod, tracing_mode="fake")(*args)

                # Import scheduling functions
                from torch._inductor.fx_passes.overlap_scheduling import (
                    schedule_overlap_bucketing_with_estimations
                )

                gm_before = copy.deepcopy(gm)

                # Run optimization to get estimations
                gm_copy = copy.deepcopy(gm)
                gm_after, nodes_estimations_dict_before = schedule_overlap_bucketing_with_estimations(
                    gm_copy,
                    collective_bucketing=True,
                    insert_overlap_deps=False,
                    max_memory_increase_ratio=0.0,
                    collective_estimator="analytical",
                )

                # Create estimators
                colls_file_path = resolve_colls_file_path(
                    self.variant_config.get_colls_file_func()
                )
                estimator_before = self._create_estimator(
                    nodes_estimations_dict_before, colls_file_path
                )

                # Get stats before optimization
                stats_before = self.calculate_stats(
                    gm_before, estimator_before, f"{variant_name}_before"
                )

                # Run optimization again for after stats
                _, nodes_estimations_dict_after = schedule_overlap_bucketing_with_estimations(
                    gm,
                    collective_bucketing=True,
                    insert_overlap_deps=False,
                    max_memory_increase_ratio=0.0,
                    collective_estimator="analytical",
                )

                estimator_after = self._create_estimator(
                    nodes_estimations_dict_after, colls_file_path
                )

                stats_after = self.calculate_stats(
                    gm_after, estimator_after, f"{variant_name}_after"
                )

                return stats_before, stats_after

        except Exception as e:
            logger.error(f"Error running experiment for {variant_name}: {e}")
            raise

    def _create_estimator(self,
                         nodes_estimations_dict: Dict[fx.Node, float],
                         colls_file_path: str) -> NodeEstimator:
        """Create a node estimator with collective table."""
        collective_table = CollectiveEstimationParser.parse_table(colls_file_path)
        return NodeEstimator(nodes_estimations_dict, collective_table)

    def calculate_stats(self,
                       gm: fx.GraphModule,
                       estimator: NodeEstimator,
                       name: str) -> Stats:
        """Calculate statistics for a graph module."""
        num_ag = num_rs = num_ar = 0

        for node in gm.graph.nodes:
            if node.op == 'call_function':
                target_str = str(node.target)
                if 'all_gather_into_tensor' in target_str:
                    num_ag += 1
                elif 'reduce_scatter_tensor' in target_str:
                    num_rs += 1
                elif 'all_reduce' in target_str:
                    num_ar += 1

        trace = TraceGenerator.generate_trace(gm, estimator.estimate, name)

        # Calculate total runtime
        max_end_time = 0.0
        for event in trace.get("traceEvents", []):
            ts = event.get("ts", 0)
            dur = event.get("dur", 0)
            end_time = ts + dur
            max_end_time = max(max_end_time, end_time)

        runtime_ms = max_end_time / MS_TO_US_MULTIPLIER  # Convert back to ms

        return Stats(num_ag=num_ag, num_rs=num_rs, num_ar=num_ar, runtime=runtime_ms)

    def cleanup(self) -> None:
        """Clean up process groups."""
        dist.destroy_process_group()


def get_group_name(n: fx.Node) -> str:
    """Extract the group name from a collective operation node."""
    opt_args_kwargs = normalize_function(
        n.target,  # type: ignore[arg-type]
        args=n.args,
        kwargs=n.kwargs,
        normalize_to_only_use_kwargs=True,
    )
    assert opt_args_kwargs is not None
    _, kwargs = opt_args_kwargs
    return kwargs["group_name"]


def resolve_colls_file_path(filename: str) -> str:
    """Resolve collective estimations filename to full path relative to run.py."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, filename)


def get_variant_configs() -> Dict[str, VariantConfig]:
    """Get all available variant configurations."""
    # Import all repro modules
    from repro_llama3_8b_bw_256_2d_32layers import (
        Repro as Repro_bw_256_2d_32, load_args as load_args_bw_256_2d_32,
        get_mesh_sizes as get_mesh_sizes_bw_256_2d_32,
        get_colls_estimations_file as get_colls_file_bw_256_2d_32,
        get_pg_names as get_pg_names_bw_256_2d_32
    )
    from repro_llama3_8b_bw_256_1d_32layers import (
        Repro as Repro_bw_256_1d_32, load_args as load_args_bw_256_1d_32,
        get_mesh_sizes as get_mesh_sizes_bw_256_1d_32,
        get_colls_estimations_file as get_colls_file_bw_256_1d_32,
        get_pg_names as get_pg_names_bw_256_1d_32
    )
    from repro_llama3_8b_bw_64_2d_32layers import (
        Repro as Repro_bw_64_2d_32, load_args as load_args_bw_64_2d_32,
        get_mesh_sizes as get_mesh_sizes_bw_64_2d_32,
        get_colls_estimations_file as get_colls_file_bw_64_2d_32,
        get_pg_names as get_pg_names_bw_64_2d_32
    )
    from repro_llama3_8b_bw_64_1d_32layers import (
        Repro as Repro_bw_64_1d_32, load_args as load_args_bw_64_1d_32,
        get_mesh_sizes as get_mesh_sizes_bw_64_1d_32,
        get_colls_estimations_file as get_colls_file_bw_64_1d_32,
        get_pg_names as get_pg_names_bw_64_1d_32
    )
    from repro_llama3_8b_fw_256_2d_32layers import (
        Repro as Repro_fw_256_2d_32, load_args as load_args_fw_256_2d_32,
        get_mesh_sizes as get_mesh_sizes_fw_256_2d_32,
        get_colls_estimations_file as get_colls_file_fw_256_2d_32,
        get_pg_names as get_pg_names_fw_256_2d_32
    )
    from repro_llama3_8b_fw_256_1d_32layers import (
        Repro as Repro_fw_256_1d_32, load_args as load_args_fw_256_1d_32,
        get_mesh_sizes as get_mesh_sizes_fw_256_1d_32,
        get_colls_estimations_file as get_colls_file_fw_256_1d_32,
        get_pg_names as get_pg_names_fw_256_1d_32
    )
    from repro_llama3_8b_fw_64_2d_32layers import (
        Repro as Repro_fw_64_2d_32, load_args as load_args_fw_64_2d_32,
        get_mesh_sizes as get_mesh_sizes_fw_64_2d_32,
        get_colls_estimations_file as get_colls_file_fw_64_2d_32,
        get_pg_names as get_pg_names_fw_64_2d_32
    )
    from repro_llama3_8b_fw_64_1d_32layers import (
        Repro as Repro_fw_64_1d_32, load_args as load_args_fw_64_1d_32,
        get_mesh_sizes as get_mesh_sizes_fw_64_1d_32,
        get_colls_estimations_file as get_colls_file_fw_64_1d_32,
        get_pg_names as get_pg_names_fw_64_1d_32
    )

    return {
        "llama3_8b_bw_256_2d_32": VariantConfig(
            Repro_bw_256_2d_32, load_args_bw_256_2d_32, get_mesh_sizes_bw_256_2d_32,
            get_colls_file_bw_256_2d_32, get_pg_names_bw_256_2d_32
        ),
        "llama3_8b_bw_256_1d_32": VariantConfig(
            Repro_bw_256_1d_32, load_args_bw_256_1d_32, get_mesh_sizes_bw_256_1d_32,
            get_colls_file_bw_256_1d_32, get_pg_names_bw_256_1d_32
        ),
        "llama3_8b_bw_64_2d_32": VariantConfig(
            Repro_bw_64_2d_32, load_args_bw_64_2d_32, get_mesh_sizes_bw_64_2d_32,
            get_colls_file_bw_64_2d_32, get_pg_names_bw_64_2d_32
        ),
        "llama3_8b_bw_64_1d_32": VariantConfig(
            Repro_bw_64_1d_32, load_args_bw_64_1d_32, get_mesh_sizes_bw_64_1d_32,
            get_colls_file_bw_64_1d_32, get_pg_names_bw_64_1d_32
        ),
        "llama3_8b_fw_256_2d_32": VariantConfig(
            Repro_fw_256_2d_32, load_args_fw_256_2d_32, get_mesh_sizes_fw_256_2d_32,
            get_colls_file_fw_256_2d_32, get_pg_names_fw_256_2d_32
        ),
        "llama3_8b_fw_256_1d_32": VariantConfig(
            Repro_fw_256_1d_32, load_args_fw_256_1d_32, get_mesh_sizes_fw_256_1d_32,
            get_colls_file_fw_256_1d_32, get_pg_names_fw_256_1d_32
        ),
        "llama3_8b_fw_64_2d_32": VariantConfig(
            Repro_fw_64_2d_32, load_args_fw_64_2d_32, get_mesh_sizes_fw_64_2d_32,
            get_colls_file_fw_64_2d_32, get_pg_names_fw_64_2d_32
        ),
        "llama3_8b_fw_64_1d_32": VariantConfig(
            Repro_fw_64_1d_32, load_args_fw_64_1d_32, get_mesh_sizes_fw_64_1d_32,
            get_colls_file_fw_64_1d_32, get_pg_names_fw_64_1d_32
        ),
    }


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Run overlap scheduling experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_improved.py --variant llama3_8b_bw_256_2d_32
  python run_improved.py --variant llama3_8b_fw_256_2d_32
  python run_improved.py --variant llama3_8b_fw_64_1d_32
"""
    )

    variant_choices = list(get_variant_configs().keys())
    parser.add_argument(
        "--variant",
        type=str,
        default=DEFAULT_VARIANT,
        choices=variant_choices,
        help=f"Model variant (default: {DEFAULT_VARIANT})"
    )

    return parser


def main() -> None:
    """Main entry point."""
    # Configure torch before any experiments
    configure_torch()

    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Get variant configuration
    variant_configs = get_variant_configs()
    if args.variant not in variant_configs:
        logger.error(f"Unknown variant: {args.variant}")
        return

    variant_config = variant_configs[args.variant]

    # Run experiment
    try:
        logger.info(f"Running overlap scheduling experiment for variant: {args.variant}")

        runner = ExperimentRunner(variant_config)
        stats_before, stats_after = runner.run_experiment(args.variant)

        # Print results
        logger.info("Experiment completed successfully")
        print(f"\nResults for {args.variant}:")
        print(f"BEFORE: {stats_before}")
        print(f"AFTER:  {stats_after}")

        # Calculate improvement
        if stats_before.runtime > 0:
            improvement = ((stats_before.runtime - stats_after.runtime) / stats_before.runtime) * 100
            print(f"Runtime improvement: {improvement:.2f}%")

        runner.cleanup()

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


if __name__ == '__main__':
    main()