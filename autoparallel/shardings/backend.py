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


@dataclass
class OpOption:
    """One valid strategy for an op — backend-agnostic.

    The optimizer only sees cost numbers and opaque spec objects.
    The backend knows how to interpret the specs.

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
        tensor_shape: tuple[int, ...],
    ) -> list[OpOption]:
        """Generate all possible shardings for a tensor (placeholder/parameter).

        Args:
            mesh: Device mesh
            tensor_shape: Global tensor shape

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
