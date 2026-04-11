"""
ShardingBackend: protocol for swappable sharding strategy backends.

The AutoParallel optimizer is backend-agnostic — it operates on cost numbers
and opaque spec objects. The backend encapsulates:
1. Strategy generation (enumerate valid input/output sharding pairs per op)
2. Cost estimation (communication and computation cost per strategy)
3. Solution application (inserting redistributes into the FX graph)

Two implementations:
- DTensorBackend: wraps existing DTensor OpStrategy/OpSpec pipeline
- CuTeBackend: uses ShardedLayout + 5-primitive propagation engine
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


def _fmt_spec(spec):
    """Format a spec (DTensorSpec, ShardedLayout, or tuple) concisely."""
    if isinstance(spec, tuple):
        return "(" + ", ".join(_fmt_spec(s) for s in spec) + ")"
    # DTensorSpec: use placements' str() for concise S(0), R, P(sum)
    if hasattr(spec, 'placements'):
        return "(" + ", ".join(str(p) for p in spec.placements) + ")"
    # ShardedLayout: show mesh_dim_map concisely
    if hasattr(spec, 'mesh_dim_map'):
        parts = []
        for d, mds in sorted(spec.mesh_dim_map.items()):
            if mds:
                parts.append(f"S({d})")
            else:
                parts.append("R")
        if spec.partial:
            for md, op in spec.partial.items():
                parts.append(f"P({op})")
        return "(" + ", ".join(parts) + ")"
    return str(spec)


@dataclass
class OpOption:
    """One valid strategy for an op — backend-agnostic.

    The optimizer only sees cost numbers and opaque spec objects.
    The backend knows how to interpret the specs.

    Exposes .output_specs, .input_specs, .redistribute_cost as aliases
    for compatibility with code expecting OpSpec-like objects.

    Attributes:
        output_spec: Backend-specific output sharding (DTensorSpec or ShardedLayout)
        input_specs: Backend-specific per-input shardings
        compute_cost: Estimated computation time in microseconds
        comm_cost: Estimated communication time in microseconds (to redistribute inputs)
        redistribute_costs: Per-arg, per-source-option cost matrix.
            redistribute_costs[arg_idx][src_option_idx] = cost to redistribute
            from src_option to this strategy's required input sharding.
    """
    output_spec: Any
    input_specs: tuple
    compute_cost: float = 0.0
    comm_cost: float = 0.0
    redistribute_costs: list = field(default_factory=list)

    # Aliases for OpSpec compatibility — the optimizer accesses these
    @property
    def output_specs(self):
        return self.output_spec

    @property
    def redistribute_cost(self):
        return self.redistribute_costs

    def __repr__(self):
        return f"OpOption({_fmt_spec(self.input_specs)} -> {_fmt_spec(self.output_spec)})"


class OpOptionList(list):
    """A list of OpOption that also exposes .strategies for OpStrategy compatibility.

    The optimizer code accesses strats[node].strategies[i] — this class makes
    strats[node] a plain list where .strategies is just self.
    This eliminates the need for wrapper classes while keeping backward compat.

    Registered as a pytree leaf so tree_map doesn't descend into it.
    """

    @property
    def strategies(self):
        return self


@runtime_checkable
class ShardingBackend(Protocol):
    """Interface for sharding strategy generation, cost estimation, and application.

    The optimizer calls these methods without knowing the underlying representation.
    Both DTensor and CuTe backends implement this protocol.
    """

    def enumerate_options(
        self,
        mesh: Any,
        node: Any,
        input_options: list[list[OpOption]],
        user_args: tuple,
        user_kwargs: dict,
    ) -> list[OpOption]:
        """Generate all valid strategies for a graph node.

        Args:
            mesh: Device mesh (DeviceMesh or mesh shape tuple)
            node: FX graph node (torch.fx.Node)
            input_options: list of list[OpOption] — available options per input
            user_args: actual runtime argument values (for meta-computation)
            user_kwargs: actual runtime keyword argument values

        Returns:
            List of OpOption — all valid (input, output) combinations with costs.
            Each OpOption has redistribute_costs filled in for the optimizer's
            cost matrix construction.
        """
        ...

    def create_all_options(
        self,
        mesh: Any,
        node: Any,
    ) -> list[OpOption]:
        """Generate all possible shardings for a tensor (placeholder/parameter).

        Args:
            mesh: Device mesh
            node: FX graph node — backend extracts shape/dtype/stride from node.meta["val"]

        Returns:
            List of OpOption where each has a single output_spec and
            trivial costs (no computation, no communication).
        """
        ...

    def redistribute_cost(
        self,
        src_spec: Any,
        tgt_spec: Any,
        mesh: Any,
    ) -> float:
        """Cost of redistributing from src to tgt sharding in microseconds.

        Args:
            src_spec: Source sharding (backend-specific)
            tgt_spec: Target sharding (backend-specific)
            mesh: Device mesh

        Returns:
            Estimated time in microseconds. 0.0 if no communication needed.
        """
        ...

    def apply_solution(
        self,
        gm: Any,
        solution: dict,
        mesh: Any,
    ) -> Any:
        """Apply the chosen sharding solution to the FX graph.

        Args:
            gm: torch.fx.GraphModule
            solution: dict[Node -> OpOption] — chosen strategy per node
            mesh: Device mesh

        Returns:
            Modified GraphModule with redistribute collectives inserted.

        Inserts redistribute collectives where a node's output_spec doesn't
        match the next node's required input_spec.
        """
        ...
