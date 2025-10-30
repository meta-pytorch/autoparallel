import functools
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import statistics

import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from torch._dynamo.testing import rand_strided
from torch._inductor import comm_analysis
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._mode_utils import no_dispatch


CACHE_DIR = "/tmp/bubble_estimator_cache"

class NCCL_COLL(IntEnum):
    ALL_REDUCE = 0
    ALL_GATHER = 1
    REDUCE_SCATTER = 2
    ALL_TO_ALL = 3

    def __str__(self) -> str:
        return self.name

dummy_graph = fx.Graph()

def get_hint(x: Union[int, torch.SymInt]) -> int | None:
    if isinstance(x, int):
        return x
    assert isinstance(x, torch.SymInt)
    if not x.node.has_hint():
        return None
    return x.node.hint

def to_real(t: torch.Tensor) -> torch.Tensor | None:
   shape = [get_hint(dim) for dim in t.shape]
   stride = [get_hint(s) for s in t.stride()]
   ret = rand_strided(shape, stride, device=t.device, dtype=t.dtype)  # type: ignore[arg-type]
   return ret

def do_bench() -> Callable[[Callable[[], Any]], float]:
    return functools.partial(
        torch._inductor.runtime.benchmarking.benchmarker.benchmark_gpu,
        warmup=5,
    )

def cache_key(func, args, kwargs):
    key = f"{func}:"
    for a in pytree.tree_leaves((args, kwargs)):
        if isinstance(a, torch.Tensor):
            key += f"T:{a.dtype}{a.shape}{a.stride()}"
        else:
            key += f"{a}"
    return key


def _make_filename(key: str) -> str:
    filename = hashlib.sha256(key.encode()).hexdigest()
    return filename


def cache_lookup(func, args, kwargs) -> Optional[float]:
    key = cache_key(func, args, kwargs)
    filename = _make_filename(key)
    filepath = Path(CACHE_DIR) / filename

    if filepath.exists():
        try:
            with open(filepath, "r") as f:
                return float(f.read().strip())
        except Exception as e:
            print(f"Warning: Failed to load cache from {filepath}: {e}")
            return None
    return None


def cache_save(func, args, kwargs, value: float) -> None:
    key = cache_key(func, args, kwargs)
    filename = _make_filename(key)
    cache_path = Path(CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)
    filepath = cache_path / filename

    try:
        with open(filepath, "w") as f:
            f.write(str(value))
    except Exception as e:
        print(f"Warning: Failed to save cache to {filepath}: {e}")


class ProfileGuidedEstimator:
    """
    Profile-guided estimator for predicting operation durations.

    Stores samples from profiling traces and provides predictions using
    linear interpolation/extrapolation. Designed to be extensible for
    different operation types (collectives, matmuls, etc.).
    """

    def __init__(self):
        # Storage: {(nccl_coll_type, group_size, pg_desc): {input_size_bytes: [durations]}}
        self.collective_samples: Dict[Tuple[NCCL_COLL, int, str], Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))

        # Storage: {(M, N, K, dtype, op_type): {flops: [durations]}}
        self.matmul_samples: Dict[Tuple[int, int, int, str, str], Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))

    def collective_add_sample(
        self,
        nccl_coll_type: NCCL_COLL,
        group_size: int,
        input_size_bytes: int,
        process_group_description: str,
        duration_ms: float,
    ) -> None:
        """
        Add a collective operation sample to the estimator.

        Args:
            nccl_coll_type: Type of NCCL collective (ALL_REDUCE, ALL_GATHER, etc.)
            group_size: Number of ranks in the process group
            input_size_bytes: Size of input data in bytes
            duration_ms: Measured duration in milliseconds
        """
        key = (nccl_coll_type, group_size)
        self.collective_samples[key][input_size_bytes].append(duration_ms)

    def collective_predict(
        self,
        nccl_coll_type: NCCL_COLL,
        group_size: int,
        input_size_bytes: int,
        process_group_description: str,
    ) -> Optional[float]:
        """
        Predict collective operation duration using linear interpolation/extrapolation.

        Args:
            nccl_coll_type: Type of NCCL collective
            group_size: Number of ranks in the process group
            input_size_bytes: Size of input data in bytes
            process_group_description: Description of the process group

        Returns:
            Predicted duration in milliseconds, or None if no matching samples found
        """
        key = (nccl_coll_type, group_size)

        if key not in self.collective_samples:
            return None

        size_to_durations = self.collective_samples[key]

        # Average durations for each size
        averaged_samples = {}
        for size_bytes, durations in size_to_durations.items():
            averaged_samples[size_bytes] = statistics.mean(durations)

        # Sort by size
        sorted_sizes = sorted(averaged_samples.keys())

        if not sorted_sizes:
            return None

        # Exact match
        if input_size_bytes in averaged_samples:
            return averaged_samples[input_size_bytes]

        # Single sample - return it
        if len(sorted_sizes) == 1:
            return averaged_samples[sorted_sizes[0]]

        # Find closest smaller and larger samples
        smaller_sizes = [s for s in sorted_sizes if s < input_size_bytes]
        larger_sizes = [s for s in sorted_sizes if s > input_size_bytes]

        if smaller_sizes and larger_sizes:
            # Interpolation between closest samples
            s1 = smaller_sizes[-1]  # Largest smaller
            s2 = larger_sizes[0]    # Smallest larger
            d1 = averaged_samples[s1]
            d2 = averaged_samples[s2]

            # Linear interpolation
            ratio = (input_size_bytes - s1) / (s2 - s1)
            return d1 + ratio * (d2 - d1)

        elif smaller_sizes:
            # Extrapolation using two largest samples
            if len(smaller_sizes) == 1:
                # Only one sample smaller, use constant extrapolation
                return averaged_samples[smaller_sizes[0]]

            s1 = smaller_sizes[-2]  # Second largest
            s2 = smaller_sizes[-1]  # Largest
            d1 = averaged_samples[s1]
            d2 = averaged_samples[s2]

            # Linear extrapolation
            slope = (d2 - d1) / (s2 - s1)
            return d2 + slope * (input_size_bytes - s2)

        else:  # only larger_sizes
            # Target is smaller than all samples - return smallest duration
            # (conservative: assume operation cannot be faster than smallest we've seen)
            return averaged_samples[larger_sizes[0]]

    def matmul_add_sample(
        self,
        M: int,
        N: int,
        K: int,
        dtype: str,
        op_type: str,
        duration_ms: float,
    ) -> None:
        """
        Add a matmul operation sample to the estimator.

        Args:
            M: First dimension (rows of A, rows of output)
            N: Second dimension (cols of B, cols of output)
            K: Inner dimension (cols of A, rows of B)
            dtype: Data type (e.g., "BFloat16", "Float", "Float16")
            op_type: Operation type (e.g., "aten::mm", "aten::addmm", "aten::bmm")
            duration_ms: Measured duration in milliseconds
        """
        key = (M, N, K, dtype, op_type)
        flops = 2 * M * N * K  # Standard GEMM FLOPs calculation
        self.matmul_samples[key][flops].append(duration_ms)

    def matmul_predict(
        self,
        M: int,
        N: int,
        K: int,
        dtype: str,
        op_type: str,
    ) -> Optional[float]:
        """
        Predict matmul operation duration using linear interpolation/extrapolation.

        Args:
            M: First dimension
            N: Second dimension
            K: Inner dimension
            dtype: Data type
            op_type: Operation type

        Returns:
            Predicted duration in milliseconds, or None if no matching samples found
        """
        key = (M, N, K, dtype, op_type)

        if key not in self.matmul_samples:
            return None

        flops_to_durations = self.matmul_samples[key]

        # Average durations for each FLOP count
        averaged_samples = {}
        for flops, durations in flops_to_durations.items():
            averaged_samples[flops] = statistics.mean(durations)

        # Sort by FLOPs
        sorted_flops = sorted(averaged_samples.keys())

        if not sorted_flops:
            return None

        target_flops = 2 * M * N * K

        # Exact match
        if target_flops in averaged_samples:
            return averaged_samples[target_flops]

        # Single sample - return it
        if len(sorted_flops) == 1:
            return averaged_samples[sorted_flops[0]]

        # Find closest smaller and larger samples
        smaller_flops = [f for f in sorted_flops if f < target_flops]
        larger_flops = [f for f in sorted_flops if f > target_flops]

        if smaller_flops and larger_flops:
            # Interpolation
            f1 = smaller_flops[-1]  # Largest smaller
            f2 = larger_flops[0]    # Smallest larger
            d1 = averaged_samples[f1]
            d2 = averaged_samples[f2]

            ratio = (target_flops - f1) / (f2 - f1)
            return d1 + ratio * (d2 - d1)

        elif smaller_flops:
            # Extrapolation using two largest samples
            if len(smaller_flops) == 1:
                return averaged_samples[smaller_flops[0]]

            f1 = smaller_flops[-2]
            f2 = smaller_flops[-1]
            d1 = averaged_samples[f1]
            d2 = averaged_samples[f2]

            slope = (d2 - d1) / (f2 - f1)
            return d2 + slope * (target_flops - f2)

        else:  # only larger_flops
            # Target is smaller than all samples - return smallest duration
            # (conservative: assume operation cannot be faster than smallest we've seen)
            return averaged_samples[larger_flops[0]]

    def print_statistics(self) -> None:
        """
        Print statistics about collected samples showing distribution (p25, p50, p75).
        Shows up to 12 representative samples per operation type.
        """
        print("\n" + "=" * 80)
        print("ProfileGuidedEstimator Statistics")
        print("=" * 80)

        # Print collective statistics
        if self.collective_samples:
            print("\nCollective Operations:")
            print("-" * 80)

            for key, size_to_durations in sorted(self.collective_samples.items()):
                nccl_coll_type, group_size = key
                print(f"\n  {nccl_coll_type} (group_size={group_size})")

                # Get all sizes and their stats
                size_stats = []
                for size_bytes, durations in sorted(size_to_durations.items()):
                    if durations:
                        p25 = statistics.quantiles(durations, n=4)[0] if len(durations) >= 2 else durations[0]
                        p50 = statistics.median(durations)
                        p75 = statistics.quantiles(durations, n=4)[2] if len(durations) >= 2 else durations[0]
                        size_stats.append((size_bytes, len(durations), p25, p50, p75))

                # Show up to 12 representative samples
                if len(size_stats) <= 12:
                    samples_to_show = size_stats
                else:
                    # Select evenly spaced samples
                    step = len(size_stats) / 12
                    indices = [int(i * step) for i in range(12)]
                    samples_to_show = [size_stats[i] for i in indices]

                for size_bytes, count, p25, p50, p75 in samples_to_show:
                    size_mb = size_bytes / (1024 * 1024)
                    print(f"    {size_mb:8.2f} MB: n={count:3d}  p25={p25:7.3f}ms  p50={p50:7.3f}ms  p75={p75:7.3f}ms")

        # Print matmul statistics
        if self.matmul_samples:
            print("\n" + "-" * 80)
            print("Matmul Operations:")
            print("-" * 80)

            for key, flops_to_durations in sorted(self.matmul_samples.items()):
                M, N, K, dtype, op_type = key
                print(f"\n  {op_type} (M={M}, N={N}, K={K}, dtype={dtype})")

                # Get all FLOP counts and their stats
                flop_stats = []
                for flops, durations in sorted(flops_to_durations.items()):
                    if durations:
                        p25 = statistics.quantiles(durations, n=4)[0] if len(durations) >= 2 else durations[0]
                        p50 = statistics.median(durations)
                        p75 = statistics.quantiles(durations, n=4)[2] if len(durations) >= 2 else durations[0]
                        flop_stats.append((flops, len(durations), p25, p50, p75))

                # Show up to 12 representative samples
                if len(flop_stats) <= 12:
                    samples_to_show = flop_stats
                else:
                    # Select evenly spaced samples
                    step = len(flop_stats) / 12
                    indices = [int(i * step) for i in range(12)]
                    samples_to_show = [flop_stats[i] for i in indices]

                for flops, count, p25, p50, p75 in samples_to_show:
                    gflops = flops / 1e9
                    print(f"    {gflops:8.2f} GFLOP: n={count:3d}  p25={p25:7.3f}ms  p50={p50:7.3f}ms  p75={p75:7.3f}ms")

        if not self.collective_samples and not self.matmul_samples:
            print("\n  No samples collected")

        print("\n" + "=" * 80)


def parse_profile_traces(
    trace_paths: List[str],
    estimator: Optional['ProfileGuidedEstimator'] = None
) -> Dict[str, float]:
    latency_map: Dict[str, List[float]] = defaultdict(list)

    # Map from External ID to metadata from record_param_comms events
    external_id_to_metadata: Dict[int, Dict[str, Any]] = {}

    for trace_path in trace_paths:
        print(f"Parsing profile trace: {trace_path}")
        try:
            with open(trace_path, 'r') as f:
                trace_data = json.load(f)

            # Extract trace events
            trace_events = trace_data.get("traceEvents", [])

            # First pass: Build External ID to metadata mapping
            for event in trace_events:
                if event.get("ph") == "X" and event.get("cat") == "cpu_op":
                    op_name = event.get("name")
                    args = event.get("args", {})
                    external_id = args.get("External id")

                    if not external_id:
                        continue

                    # Collect metadata for NCCL collectives
                    if op_name == "record_param_comms":
                        external_id_to_metadata[external_id] = {
                            "input_dims": args.get("Input Dims"),
                            "collective_name": args.get("Collective name"),
                            "group_size": args.get("Group size"),
                            "dtype": args.get("dtype"),
                            "in_msg_nelems": args.get("In msg nelems"),
                            "out_msg_nelems": args.get("Out msg nelems"),
                            "process_group_name": args.get("Process Group Name"),
                            "process_group_desc": args.get("Process Group Description"),
                        }

                    # Collect metadata for matmul operations
                    elif op_name in ("aten::mm", "aten::addmm", "aten::bmm", "_scaled_dot_product_flash_attention"):
                        external_id_to_metadata[external_id] = {
                            "op_name": op_name,
                            "input_dims": args.get("Input Dims"),
                            "input_types": args.get("Input type"),
                        }

            # Second pass: Extract latencies and NCCL metadata
            for event in trace_events:
                # Only process duration events (phase "X") with a name and duration
                if event.get("ph") == "X" and "name" in event and "dur" in event and "args" in event and "stream" in event.get("args"):
                    op_name = event["name"]
                    duration_us = event["dur"]  # Duration in microseconds
                    duration_ms = duration_us / 1000.0  # Convert to milliseconds
                    args = event.get("args", {})

                    # Check if this is an NCCL kernel
                    if "ncclDevKernel" in op_name:
                        external_id = args.get("External id")
                        collective_name = args.get("Collective name", "")
                        group_size = args.get("Group size")
                        in_msg_nelems = args.get("In msg nelems")
                        dtype = args.get("dtype", "")
                        process_group_desc = args.get("Process Group Description", "")

                        # Map collective name to NCCL_COLL enum
                        nccl_coll_type = None
                        if "AllReduce" in op_name or "allreduce" in collective_name.lower():
                            nccl_coll_type = NCCL_COLL.ALL_REDUCE
                        elif "AllGather" in op_name or "allgather" in collective_name.lower():
                            nccl_coll_type = NCCL_COLL.ALL_GATHER
                        elif "ReduceScatter" in op_name or "reduce_scatter" in collective_name.lower():
                            nccl_coll_type = NCCL_COLL.REDUCE_SCATTER
                        elif "AllToAll" in op_name or "alltoall" in collective_name.lower():
                            nccl_coll_type = NCCL_COLL.ALL_TO_ALL

                        # Calculate input size in bytes
                        input_size_bytes = None
                        if in_msg_nelems and dtype:
                            dtype_size = {
                                "Float": 4, "Float32": 4,
                                "BFloat16": 2, "Half": 2, "Float16": 2,
                                "Double": 8, "Float64": 8,
                                "Int32": 4, "Int64": 8,
                            }.get(dtype, 1)
                            input_size_bytes = in_msg_nelems * dtype_size

                        # Get Input Dims from metadata if available
                        input_dims = None
                        if external_id and external_id in external_id_to_metadata:
                            metadata = external_id_to_metadata[external_id]
                            input_dims = metadata.get("input_dims")
                            # Fallback to metadata if not in kernel args
                            if not collective_name:
                                collective_name = metadata.get("collective_name", "")
                            if not group_size:
                                group_size = metadata.get("group_size")
                            if not process_group_desc:
                                process_group_desc = metadata.get("process_group_desc", "")

                        # Add sample to estimator if provided and all required data is available
                        if (estimator and nccl_coll_type is not None and
                            group_size is not None and input_size_bytes is not None and
                            process_group_desc):
                            estimator.collective_add_sample(
                                nccl_coll_type=nccl_coll_type,
                                group_size=group_size,
                                input_size_bytes=input_size_bytes,
                                process_group_description=process_group_desc,
                                duration_ms=duration_ms
                            )

                    # Check if this is a GEMM/matmul kernel
                    if estimator and "gemm" in op_name.lower():
                        external_id = args.get("External id")

                        # Try to find corresponding CPU operation with shapes
                        if external_id and external_id in external_id_to_metadata:
                            metadata = external_id_to_metadata[external_id]
                            input_dims = metadata.get("input_dims")
                            input_types = metadata.get("input_types")
                            cpu_op_name = metadata.get("op_name")

                            if input_dims and input_types:
                                # Extract M, N, K and dtype
                                M, N, K, dtype = None, None, None, None

                                if cpu_op_name == "aten::mm":
                                    # Input Dims: [[M, K], [K, N], [M, N]]
                                    if len(input_dims) >= 2 and len(input_dims[0]) == 2 and len(input_dims[1]) == 2:
                                        M, K = input_dims[0]
                                        K_check, N = input_dims[1]
                                        if len(input_types) > 0:
                                            dtype = input_types[0].replace("c10::", "")

                                elif cpu_op_name == "aten::addmm":
                                    # Input Dims: [[bias_dims], [M, K], [K, N], [M, N]]
                                    if len(input_dims) >= 3 and len(input_dims[1]) == 2 and len(input_dims[2]) == 2:
                                        M, K = input_dims[1]
                                        K_check, N = input_dims[2]
                                        if len(input_types) > 1:
                                            dtype = input_types[1].replace("c10::", "")

                                elif cpu_op_name == "aten::bmm":
                                    # Input Dims: [[B, M, K], [B, K, N], [B, M, N]]
                                    if len(input_dims) >= 2 and len(input_dims[0]) == 3 and len(input_dims[1]) == 3:
                                        B, M, K = input_dims[0]
                                        B_check, K_check, N = input_dims[1]
                                        M = B * M  # Treat batched as larger M
                                        if len(input_types) > 0:
                                            dtype = input_types[0].replace("c10::", "")

                                # Add sample if we successfully extracted dimensions
                                if M and N and K and dtype:
                                    estimator.matmul_add_sample(
                                        M=M,
                                        N=N,
                                        K=K,
                                        dtype=dtype,
                                        op_type=cpu_op_name or "gemm",
                                        duration_ms=duration_ms
                                    )

                    latency_map[op_name].append(duration_ms)

            print(f"  Extracted {len(latency_map)} unique operation types")

        except Exception as e:
            print(f"Warning: Failed to parse trace file {trace_path}: {e}")
            continue

    for op_name, durs in latency_map.items():
        print(f"XXX OP:{op_name}:#{len(durs)}:{durs[:10]}")

    # Print statistics if estimator was provided
    if estimator:
        estimator.print_statistics()

    return latency_map


@dataclass(frozen=True)
class OpInfo:
    func: str
    start_time: float
    end_time: float
    stream: str

    def __repr__(self) -> str:
        duration = self.end_time - self.start_time
        return f"OpInfo(func={self.func}, stream={self.stream}, duration={duration:.3f}, start={self.start_time:.6f}, end={self.end_time:.6f})"


@dataclass
class BubbleInfo:
    total_time: float
    bubble_time: float

    def __repr__(self) -> str:
        return f"BubbleInfo(total_time={self.total_time:.3f}ms, bubble_time={self.bubble_time:.3f}ms)"


def is_collective(func, args, kwargs):
    return not is_wait(func, args, kwargs) and "_c10d_functional." in str(func)


def is_wait(func, args, kwargs):
    return "_c10d_functional.wait_tensor" in str(func)


def get_group_name(func, args, kwargs):
    if "group_name" in kwargs:
        return kwargs["group_name"]

    for a in args:
        if isinstance(a, str) and a not in ("sum", "avg"):
            return a

    assert False


class BubbleEstimatorMode(TorchDispatchMode):
    def __init__(
        self,
        visualize: bool = False,
        chrome_trace_path: Optional[str] = None,
        use_latencies_from_profile_traces: Optional[List[str]] = None,
    ):
        super().__init__()
        self.ops = []
        self.stream_next_available_time = defaultdict(lambda: 0)
        self.tensor_to_op = {}
        self.fake_mode = FakeTensorMode()
        self.chrome_trace_path = chrome_trace_path

        # Parse profile traces if provided
        self.profile_guided_estimator = None
        if use_latencies_from_profile_traces:
            self.profile_guided_estimator = ProfileGuidedEstimator()
            parse_profile_traces(
                use_latencies_from_profile_traces,
                self.profile_guided_estimator
            )

    def convert_args(self, args):
        return pytree.tree_map_only(torch.Tensor, self.fake_mode.from_tensor, args)

    def profile_guided_estimate_runtime_ms(self, func, args, kwargs) -> Optional[float]:
        """
        Estimate runtime using ProfileGuidedEstimator for NCCL collectives and matmul operations.

        Returns:
            Predicted duration in milliseconds, or None if not applicable
        """
        if not self.profile_guided_estimator:
            return None

        func_str = str(func)

        # Try NCCL collective estimation
        if is_collective(func, args, kwargs):
            # Determine NCCL collective type
            nccl_coll_type = None
            if "all_reduce" in func_str.lower():
                nccl_coll_type = NCCL_COLL.ALL_REDUCE
            elif "all_gather" in func_str.lower():
                nccl_coll_type = NCCL_COLL.ALL_GATHER
            elif "reduce_scatter" in func_str.lower():
                nccl_coll_type = NCCL_COLL.REDUCE_SCATTER
            elif "all_to_all" in func_str.lower():
                nccl_coll_type = NCCL_COLL.ALL_TO_ALL

            if nccl_coll_type is not None:
                # Extract process group name/description
                process_group_desc = get_group_name(func, args, kwargs)
                if process_group_desc:
                    # Try to extract tensor size and compute input_size_bytes
                    input_tensor = None
                    for arg in args:
                        if isinstance(arg, torch.Tensor):
                            input_tensor = arg
                            break

                    if input_tensor is not None:
                        # Calculate input size in bytes
                        numel = input_tensor.numel()
                        dtype_size = input_tensor.element_size()
                        input_size_bytes = numel * dtype_size

                        # Get group size from the estimator's existing samples
                        group_size = None
                        for key in self.profile_guided_estimator.collective_samples.keys():
                            if key[0] == nccl_coll_type:
                                group_size = key[1]
                                break

                        if group_size is not None:
                            # Try to predict
                            return self.profile_guided_estimator.collective_predict(
                                nccl_coll_type=nccl_coll_type,
                                group_size=group_size,
                                input_size_bytes=input_size_bytes,
                                process_group_description=process_group_desc
                            )

        # Try matmul estimation
        if "mm" in func_str.lower() or "matmul" in func_str.lower() or "scaled_dot_product" in func_str.lower():
            # Determine operation type
            op_type = None
            if "aten.mm" in func_str:
                op_type = "aten::mm"
            elif "aten.addmm" in func_str:
                op_type = "aten::addmm"
            elif "aten.bmm" in func_str:
                op_type = "aten::bmm"
            elif "scaled_dot_product" in func_str:
                op_type = "_scaled_dot_product_flash_attention"

            if op_type and len(args) >= 2:
                # Extract tensors
                tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]

                if len(tensors) >= 2:
                    M, N, K, dtype = None, None, None, None

                    if op_type == "aten::mm":
                        # args: (A, B) where A is [M, K], B is [K, N]
                        if len(tensors[0].shape) == 2 and len(tensors[1].shape) == 2:
                            M, K = tensors[0].shape
                            K_check, N = tensors[1].shape
                            dtype = str(tensors[0].dtype).replace("torch.", "").replace("bfloat16", "BFloat16").replace("float32", "Float").replace("float16", "Half")

                    elif op_type == "aten::addmm":
                        # args: (bias, A, B) where A is [M, K], B is [K, N]
                        if len(tensors) >= 3 and len(tensors[1].shape) == 2 and len(tensors[2].shape) == 2:
                            M, K = tensors[1].shape
                            K_check, N = tensors[2].shape
                            dtype = str(tensors[1].dtype).replace("torch.", "").replace("bfloat16", "BFloat16").replace("float32", "Float").replace("float16", "Half")

                    elif op_type == "aten::bmm":
                        # args: (A, B) where A is [B, M, K], B is [B, K, N]
                        if len(tensors[0].shape) == 3 and len(tensors[1].shape) == 3:
                            B, M_single, K = tensors[0].shape
                            B_check, K_check, N = tensors[1].shape
                            M = B * M_single  # Treat batch as larger M
                            dtype = str(tensors[0].dtype).replace("torch.", "").replace("bfloat16", "BFloat16").replace("float32", "Float").replace("float16", "Half")

                    if M and N and K and dtype:
                        return self.profile_guided_estimator.matmul_predict(
                            M=int(M),
                            N=int(N),
                            K=int(K),
                            dtype=dtype,
                            op_type=op_type
                        )

        return None

    def estimate_runtime_ms(self, func, args, kwargs) -> float:
        """
        Estimate runtime for an operation using multiple fallback strategies.

        Priority order:
        1. ProfileGuidedEstimator (for NCCL collectives and matmul operations)
        2. File-based cache
        3. Actual benchmarking
        """
        # Strategy 1: Try ProfileGuidedEstimator for collectives and matmuls
        predicted = self.profile_guided_estimate_runtime_ms(func, args, kwargs)
        if predicted is not None:
            return predicted
        return 0

        # Strategy 2: Check cache
        # cached = cache_lookup(func, args, kwargs)
        # if cached is not None:
        #     return cached

        # # Strategy 3: Benchmark the operation
        # if is_collective(func, args, kwargs):
        #     node = dummy_graph.call_function(func, args, kwargs)
        #     runtime_ms = comm_analysis.estimate_nccl_collective_runtime_from_fx_node(node)
        # else:
        #     with no_dispatch():
        #         real_args = pytree.tree_map_only(torch.Tensor, to_real, args)
        #         real_kwargs = pytree.tree_map_only(torch.Tensor, to_real, kwargs)
        #         runtime_ms = do_bench()(lambda: func(*real_args, **real_kwargs))

        # cache_save(func, args, kwargs, runtime_ms)
        # return runtime_ms

    def __enter__(self):
        super().__enter__()
        if self.fake_mode:
            self.fake_mode.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.analyze()
        if self.chrome_trace_path is not None:
            self.export_chrome_trace(self.chrome_trace_path)
        if self.fake_mode:
            self.fake_mode.__exit__(exc_type, exc_val, exc_tb)
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        print(f"XXX __torch_dispatch__ {func} {args} {kwargs}")
        if kwargs is None:
            kwargs = {}
        stream = "default"
        if is_collective(func, args, kwargs):
            group_name = get_group_name(func, args, kwargs)
            stream = group_name

        start_time = max(self.stream_next_available_time["default"], self.stream_next_available_time[stream])
        if is_wait(func, args, kwargs):
            collective_op = self.tensor_to_op[args[0]]
            coll_end_time = collective_op.end_time
            start_time = max(start_time, coll_end_time)
            end_time = start_time
        else:
            op_duration_time = self.estimate_runtime_ms(func, args, kwargs)
            print(f"XXX ESTIMATED_DURATION_TIME:{op_duration_time}")
            end_time = start_time + op_duration_time
        self.stream_next_available_time[stream] = end_time

        op_info = OpInfo(str(func), start_time, end_time, stream)
        self.ops.append(op_info)
        result = func(*args, **kwargs)

        def _track_tensor(t):
            self.tensor_to_op[t] = op_info

        pytree.tree_map_only(torch.Tensor, _track_tensor, result)
        return result

    def analyze(self) -> BubbleInfo:
        if not self.ops:
            return BubbleInfo(total_time=0.0, bubble_time=0.0)

        total_time = max(op.end_time for op in self.ops)

        default_ops = [op for op in self.ops if op.stream == "default"]

        default_ops_sorted = sorted(default_ops, key=lambda op: op.start_time)

        idle_intervals = []
        if default_ops_sorted:
            if default_ops_sorted[0].start_time > 0:
                idle_intervals.append((0, default_ops_sorted[0].start_time))

            for i in range(len(default_ops_sorted) - 1):
                gap_start = default_ops_sorted[i].end_time
                gap_end = default_ops_sorted[i + 1].start_time
                if gap_start < gap_end:
                    idle_intervals.append((gap_start, gap_end))

            if default_ops_sorted[-1].end_time < total_time:
                idle_intervals.append((default_ops_sorted[-1].end_time, total_time))
        else:
            idle_intervals.append((0, total_time))

        bubble_time = sum(end - start for start, end in idle_intervals)

        bubble_info = BubbleInfo(total_time=total_time, bubble_time=bubble_time)
        bubble_ratio = bubble_time / total_time if total_time > 0 else 0.0
        print(f"ANALYSIS: {bubble_info} bubble_ratio={bubble_ratio:.3f}")
        return bubble_info

    def export_chrome_trace(self, filename: str = "trace.json") -> None:
        if not self.ops:
            print("No operations to export")
            return

        stream_to_tid = {}
        tid_counter = 0
        for op in self.ops:
            if op.stream not in stream_to_tid:
                stream_to_tid[op.stream] = tid_counter
                tid_counter += 1

        trace_events = []

        for idx, op in enumerate(self.ops):
            func_name = op.func

            event = {
                "name": func_name,
                "cat": op.stream,
                "ph": "X",
                "ts": op.start_time * 1000,
                "dur": (op.end_time - op.start_time) * 1000,
                "pid": 0,
                "tid": stream_to_tid[op.stream],
                "args": {
                    "op_index": idx,
                    "stream": op.stream,
                    "start_ms": f"{op.start_time:.3f}",
                    "end_ms": f"{op.end_time:.3f}",
                    "duration_ms": f"{op.end_time - op.start_time:.3f}"
                }
            }
            trace_events.append(event)

        for stream, tid in stream_to_tid.items():
            metadata_event = {
                "name": "thread_name",
                "ph": "M",
                "pid": 0,
                "tid": tid,
                "args": {
                    "name": stream
                }
            }
            trace_events.append(metadata_event)

        trace_data = {
            "traceEvents": trace_events,
            "displayTimeUnit": "ms"
        }

        filepath = Path(filename)
        with open(filepath, "w") as f:
            json.dump(trace_data, f, indent=2)

        print(f"Chrome trace exported to: {filepath.absolute()}")
        print(f"Open in Chrome at: chrome://tracing or https://ui.perfetto.dev/")
