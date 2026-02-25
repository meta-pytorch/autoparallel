# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Independent per-mesh-dim sharding optimization.

This module solves the sharding optimization problem by running the original
enumeration-based ILP (from :class:`ShardingOptimizer`) **independently on each
mesh dimension** using 1D sub-meshes, then combines the per-dim solutions into
multi-dimensional placements.

Key advantages over the joint ILP:
- Strategy count per node is O(d+1) on a 1D mesh instead of O((d+1)^k) on a
  k-dimensional mesh (where d = tensor dims, k = mesh dims).
- Each 1D ILP has exact redistribution costs and tight LP relaxation via
  one-hot encoding (unlike the factor-based ILP which uses z-variables).
- Total solve time is roughly k × (time for 1D ILP), which is dramatically
  faster than the joint formulation for k ≥ 2.

Limitation:
- Mesh dimensions are assumed independent — cross-mesh-dim interactions (e.g.
  joint memory constraints) are approximated per-dim.
"""

from __future__ import annotations

import operator
import time
from typing import Any, Optional

import torch
import torch.fx
from torch.distributed._tensor.placement_types import TensorMeta
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import OpSpec
from torch.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.utils._pytree import tree_flatten

from .optimize_sharding import ShardingOptimizer


class IndependentShardingOptimizer:
    """Sharding optimizer that solves per mesh dimension independently.

    Runs k independent :class:`ShardingOptimizer` instances on 1D sub-meshes
    (one per mesh dimension), then combines the 1D solutions into multi-dim
    placements on the full mesh.

    Public API mirrors :class:`ShardingOptimizer` and
    :class:`FactorShardingOptimizer` so it can be used as a drop-in replacement.

    Parameters
    ----------
    gm : torch.fx.GraphModule
        Traced FX graph (joint forward + backward).
    mesh : DeviceMesh
        Target device mesh (may be multi-dimensional).
    rescale_grad_comm_cost_for_mp : float
        Scaling factor for gradient communication costs (mixed precision).
    """

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        mesh: Any,
        rescale_grad_comm_cost_for_mp: float = 1.0,
    ) -> None:
        self.gm = gm
        self.graph = gm.graph
        self.mesh = mesh
        self.rescale_grad_comm_cost_for_mp = rescale_grad_comm_cost_for_mp
        self._timings: dict[str, float] = {}
        self._solved = False

        # Create k 1D sub-solvers, one per mesh dimension.
        self._sub_solvers: list[ShardingOptimizer] = []
        for m in range(mesh.ndim):
            sub_mesh = self._create_1d_mesh(m)
            t0 = time.perf_counter()
            sub_solver = ShardingOptimizer(
                gm, sub_mesh, rescale_grad_comm_cost_for_mp
            )
            self._timings[f"build_dim{m}"] = time.perf_counter() - t0
            self._sub_solvers.append(sub_solver)

    # -----------------------------------------------------------------
    # Sub-mesh creation
    # -----------------------------------------------------------------

    def _create_1d_mesh(self, mesh_dim: int) -> Any:
        """Create a 1D DeviceMesh for the given mesh dimension.

        Uses the parent mesh's subscript operator to extract a proper 1D
        sub-mesh that reuses existing process groups.
        """
        if hasattr(self.mesh, "mesh_dim_names") and self.mesh.mesh_dim_names:
            dim_name = self.mesh.mesh_dim_names[mesh_dim]
            return self.mesh[dim_name]

        # Fallback: construct a 1D mesh directly with sequential device IDs.
        dim_size = self.mesh.shape[mesh_dim]
        return torch.distributed.device_mesh.init_device_mesh(
            self.mesh.device_type,
            (dim_size,),
        )

    # -----------------------------------------------------------------
    # Constraint methods (project multi-dim → 1D per sub-solver)
    # -----------------------------------------------------------------

    def add_input_constraints(
        self, input_placements: Optional[list[Optional[tuple[Placement, ...]]]] = None
    ) -> None:
        """Add input constraints, projecting multi-dim placements to 1D per dim."""
        for m, solver in enumerate(self._sub_solvers):
            if input_placements is None:
                solver.add_input_constraints(None)
            else:
                projected = [
                    (p[m],) if p is not None else None for p in input_placements
                ]
                solver.add_input_constraints(projected)

    def add_output_constraints(
        self, output_placements: Optional[list[Optional[tuple[Placement, ...]]]] = None
    ) -> None:
        """Add output constraints, projecting multi-dim placements to 1D per dim."""
        for m, solver in enumerate(self._sub_solvers):
            if output_placements is None:
                solver.add_output_constraints(None)
            else:
                projected = [
                    (p[m],) if p is not None else None for p in output_placements
                ]
                solver.add_output_constraints(projected)

    def add_grad_param_constraints(self) -> None:
        """Ensure parameters and their gradients have matching placements."""
        for solver in self._sub_solvers:
            solver.add_grad_param_constraints()

    def add_parameter_memory_constraint(
        self, memory_factor_low: float, memory_factor_high: float
    ) -> None:
        """Add parameter memory constraints per sub-solver.

        NOTE: This is an approximation — the true memory constraint depends on
        the joint sharding across all mesh dims (product of per-dim shard
        factors).  Here we apply the constraint independently to each 1D
        sub-solver, which may over- or under-constrain memory.
        """
        # TODO: Implement a more accurate joint memory constraint by
        # post-processing the combined solution or using a Lagrangian approach.
        for solver in self._sub_solvers:
            solver.add_parameter_memory_constraint(
                memory_factor_low, memory_factor_high
            )

    def add_node_constraint(
        self,
        node: torch.fx.Node,
        placement: Optional[tuple[Placement, ...]] = None,
        constraint_name: Optional[str] = None,
    ) -> None:
        """Pin a node to a specific multi-dim placement."""
        if placement is None:
            for solver in self._sub_solvers:
                solver.add_node_constraint(node, None, constraint_name)
        else:
            for m, solver in enumerate(self._sub_solvers):
                solver.add_node_constraint(node, (placement[m],), constraint_name)

    # -----------------------------------------------------------------
    # Solve
    # -----------------------------------------------------------------

    def get_solution(self, verbose: bool = False) -> dict[torch.fx.Node, OpSpec]:
        """Solve each 1D sub-problem and combine into multi-dim OpSpecs."""
        # 1. Solve each 1D sub-problem.
        sub_solutions: list[dict[torch.fx.Node, OpSpec]] = []
        for m, solver in enumerate(self._sub_solvers):
            t0 = time.perf_counter()
            sol = solver.get_solution(verbose=verbose)
            self._timings[f"solve_dim{m}"] = time.perf_counter() - t0
            sub_solutions.append(sol)

        self._solved = True

        # 2. Combine into multi-dim OpSpecs on the full mesh.
        result: dict[torch.fx.Node, OpSpec] = {}
        for node in self.graph.nodes:
            if node.op == "output":
                continue

            per_dim_specs = [
                sub_solutions[m].get(node) for m in range(self.mesh.ndim)
            ]
            if all(s is None for s in per_dim_specs):
                continue

            combined = self._combine_specs(node, per_dim_specs)
            if combined is not None:
                result[node] = combined

        return result

    # -----------------------------------------------------------------
    # Solution combination
    # -----------------------------------------------------------------

    def _extract_1d_placements(
        self, spec: OpSpec | None, kind: str = "output"
    ) -> Placement:
        """Extract the single 1D placement from a 1D OpSpec.

        Parameters
        ----------
        spec : OpSpec or None
            A 1D OpSpec (single placement per DTensorSpec).
        kind : str
            "output" to extract from output_specs, or "input_N" to extract
            from input_specs[N].

        Returns
        -------
        Placement
            The 1D placement (Shard, Replicate, or Partial).
        """
        if spec is None:
            return Replicate()

        if kind == "output":
            out = spec.output_specs
            if isinstance(out, DTensorSpec):
                return out.placements[0]
            elif isinstance(out, (tuple, list)):
                # Multi-output: return first non-None spec's placement
                for s in out:
                    if isinstance(s, DTensorSpec):
                        return s.placements[0]
            return Replicate()
        elif kind.startswith("input_"):
            idx = int(kind.split("_")[1])
            if spec.input_specs is not None and idx < len(spec.input_specs):
                inp = spec.input_specs[idx]
                if isinstance(inp, DTensorSpec):
                    return inp.placements[0]
            return Replicate()
        return Replicate()

    def _extract_1d_output_placements_tuple(
        self, spec: OpSpec | None
    ) -> tuple | Placement:
        """Extract all 1D output placements, handling multi-output ops.

        Returns a single Placement for single-output ops, or a tuple of
        Placements for multi-output ops.
        """
        if spec is None:
            return Replicate()

        out = spec.output_specs
        if isinstance(out, DTensorSpec):
            return out.placements[0]
        elif isinstance(out, (tuple, list)):
            return tuple(
                s.placements[0] if isinstance(s, DTensorSpec) else None
                for s in out
            )
        return Replicate()

    def _combine_specs(
        self,
        node: torch.fx.Node,
        per_dim_specs: list[OpSpec | None],
    ) -> OpSpec | None:
        """Combine k 1D OpSpecs into one multi-dim OpSpec on the full mesh."""
        val = node.meta.get("val")
        k = self.mesh.ndim

        # --- Build output_specs ---
        output_specs = self._build_combined_output_specs(node, val, per_dim_specs, k)
        if output_specs is None:
            return None

        # --- Build input_specs ---
        input_specs = self._build_combined_input_specs(node, per_dim_specs, k)

        return OpSpec(output_specs=output_specs, input_specs=input_specs)

    def _build_combined_output_specs(
        self,
        node: torch.fx.Node,
        val: Any,
        per_dim_specs: list[OpSpec | None],
        k: int,
    ) -> DTensorSpec | tuple | None:
        """Build combined multi-dim output specs from per-dim 1D specs."""
        if isinstance(val, torch.Tensor):
            placements: list[Placement] = []
            for m in range(k):
                p = self._extract_1d_placements(per_dim_specs[m], "output")
                placements.append(p)
            tensor_meta = TensorMeta(val.shape, val.stride(), val.dtype)
            return DTensorSpec(
                self.mesh, tuple(placements), tensor_meta=tensor_meta
            )
        elif isinstance(val, (tuple, list)):
            # Multi-output op: combine per-output placements across dims.
            per_dim_out_placements = []
            for m in range(k):
                p = self._extract_1d_output_placements_tuple(per_dim_specs[m])
                per_dim_out_placements.append(p)

            # Determine number of outputs from val
            num_outputs = len(val)
            specs = []
            for ri in range(num_outputs):
                v = val[ri]
                if isinstance(v, torch.Tensor):
                    plc_list: list[Placement] = []
                    for m in range(k):
                        pdp = per_dim_out_placements[m]
                        if isinstance(pdp, tuple) and ri < len(pdp):
                            p = pdp[ri] if pdp[ri] is not None else Replicate()
                        elif isinstance(pdp, Placement):
                            p = pdp
                        else:
                            p = Replicate()
                        plc_list.append(p)
                    tm = TensorMeta(v.shape, v.stride(), v.dtype)
                    specs.append(
                        DTensorSpec(self.mesh, tuple(plc_list), tensor_meta=tm)
                    )
                else:
                    specs.append(None)
            return tuple(specs)
        else:
            return None

    def _build_combined_input_specs(
        self,
        node: torch.fx.Node,
        per_dim_specs: list[OpSpec | None],
        k: int,
    ) -> list[DTensorSpec] | None:
        """Build combined multi-dim input specs from per-dim 1D specs."""
        # Determine number of inputs from any non-None sub-solver spec.
        num_inputs = 0
        for spec in per_dim_specs:
            if spec is not None and spec.input_specs is not None:
                num_inputs = len(spec.input_specs)
                break

        if num_inputs == 0:
            # Placeholders and get_attr: use output_specs as input_specs
            if node.op in ("placeholder", "get_attr"):
                out = self._build_combined_output_specs(
                    node, node.meta.get("val"), per_dim_specs, k
                )
                if isinstance(out, DTensorSpec):
                    return [out]
            return None

        # Get tensor input nodes for TensorMeta
        flat_args, _ = tree_flatten(node.args)
        tensor_args = [a for a in flat_args if isinstance(a, torch.fx.Node)]

        input_specs: list[DTensorSpec] = []
        for inp_idx in range(num_inputs):
            placements: list[Placement] = []
            for m in range(k):
                spec = per_dim_specs[m]
                if (
                    spec is not None
                    and spec.input_specs is not None
                    and inp_idx < len(spec.input_specs)
                ):
                    inp = spec.input_specs[inp_idx]
                    if isinstance(inp, DTensorSpec):
                        placements.append(inp.placements[0])
                    else:
                        placements.append(Replicate())
                else:
                    placements.append(Replicate())

            # Get TensorMeta from corresponding input node.
            inp_tm = None
            if inp_idx < len(tensor_args):
                arg_val = tensor_args[inp_idx].meta.get("val")
                if isinstance(arg_val, torch.Tensor):
                    inp_tm = TensorMeta(
                        arg_val.shape, arg_val.stride(), arg_val.dtype
                    )
                elif isinstance(arg_val, (tuple, list)):
                    if (
                        node.target is operator.getitem
                        and inp_idx == 0
                        and len(node.args) > 1
                        and isinstance(node.args[1], int)
                    ):
                        idx = node.args[1]
                        if idx < len(arg_val) and isinstance(
                            arg_val[idx], torch.Tensor
                        ):
                            v = arg_val[idx]
                            inp_tm = TensorMeta(v.shape, v.stride(), v.dtype)

            input_specs.append(
                DTensorSpec(self.mesh, tuple(placements), tensor_meta=inp_tm)
            )

        return input_specs

    # -----------------------------------------------------------------
    # Stats & logging
    # -----------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return aggregated ILP size statistics across all sub-solvers."""
        total_vars = sum(len(s.ds) for s in self._sub_solvers)
        total_constraints = sum(
            len(s.prob.constraints) for s in self._sub_solvers
        )

        stats: dict[str, Any] = {
            "num_graph_nodes": len(list(self.graph.nodes)),
            "num_ilp_variables": total_vars,
            "num_ilp_constraints": total_constraints,
            "mesh_shape": tuple(self.mesh.shape),
            "num_sub_problems": self.mesh.ndim,
        }

        per_dim: list[dict[str, Any]] = []
        for m, solver in enumerate(self._sub_solvers):
            per_dim.append(
                {
                    "mesh_dim": m,
                    "mesh_size": self.mesh.shape[m],
                    "num_ilp_variables": len(solver.ds),
                    "num_ilp_constraints": len(solver.prob.constraints),
                }
            )
        stats["per_dim"] = per_dim
        stats["timings"] = dict(self._timings)
        return stats

    def get_log(self, verbose: bool = False) -> str:
        """Human-readable summary of the independent optimizer."""
        lines: list[str] = []
        lines.append("Independent per-mesh-dim ILP optimizer")
        lines.append(f"  Mesh shape: {tuple(self.mesh.shape)}")
        lines.append(f"  Sub-problems: {self.mesh.ndim}")

        stats = self.get_stats()
        lines.append(
            f"  Total ILP variables:    {stats['num_ilp_variables']:,}"
        )
        lines.append(
            f"  Total ILP constraints:  {stats['num_ilp_constraints']:,}"
        )

        for dim_stats in stats["per_dim"]:
            m = dim_stats["mesh_dim"]
            lines.append(
                f"    Dim {m} (size={dim_stats['mesh_size']}): "
                f"{dim_stats['num_ilp_variables']:,} vars, "
                f"{dim_stats['num_ilp_constraints']:,} constraints"
            )

        timings = stats.get("timings", {})
        if timings:
            lines.append("")
            lines.append("  Timings:")
            for step, dt in timings.items():
                lines.append(f"    {step:30s} {dt:.3f}s")

        return "\n".join(lines)
