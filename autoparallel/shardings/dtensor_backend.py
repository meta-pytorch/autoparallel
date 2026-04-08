"""
DTensor sharding backend for AutoParallel optimizer.

Wraps the existing DTensor OpStrategy/OpSpec/DTensorSpec pipeline behind
the ShardingBackend protocol. No functional changes — this is a thin adapter
that allows the optimizer to use the same interface for both DTensor and CuTe.
"""

from __future__ import annotations

from typing import Any

from .backend import OpOption, ShardingBackend


class DTensorBackend:
    """ShardingBackend implementation wrapping existing DTensor infrastructure."""

    def enumerate_options(
        self,
        mesh: Any,
        node: Any,
        input_options: list[list[OpOption]],
        user_args: tuple,
        user_kwargs: dict,
    ) -> list[OpOption]:
        """Generate all valid strategies using DTensor's placement_options pipeline."""
        from torch.distributed.tensor._op_schema import OpStrategy, OpSpec
        from .placement_options import get_placement_options_for_node

        # Convert input OpOptions back to OpStrategy objects
        input_strategies = []
        for opts in input_options:
            # Each input's options become an OpStrategy with one OpSpec per option
            op_specs = []
            for opt in opts:
                # opt.output_spec is a DTensorSpec (or OpSpec) from a previous step
                if isinstance(opt.output_spec, OpStrategy):
                    input_strategies.append(opt.output_spec)
                    break
                elif hasattr(opt, '_op_strategy'):
                    input_strategies.append(opt._op_strategy)
                    break
            else:
                # Build OpStrategy from individual OpOptions
                specs = [
                    OpSpec(
                        opt.output_spec,
                        input_specs=opt.input_specs,
                        redistribute_cost=[opt.redistribute_costs[0]] if opt.redistribute_costs else [[0.0]],
                    )
                    for opt in opts
                ]
                input_strategies.append(OpStrategy(specs))

        # Call existing DTensor pipeline
        op_strategy = get_placement_options_for_node(
            mesh, node, input_strategies, user_args, user_kwargs
        )

        # Convert OpStrategy to list[OpOption]
        return self._op_strategy_to_options(op_strategy)

    def create_all_options(
        self,
        mesh: Any,
        node: Any,
    ) -> list[OpOption]:
        """Generate all possible shardings using DTensor's _create_all_options."""
        from .propagation_rules import _create_all_options

        tensor = node.meta["val"]
        op_strategy = _create_all_options(
            mesh, tensor.shape, tensor=tensor
        )
        return self._op_strategy_to_options(op_strategy)

    def redistribute_cost(
        self,
        src_spec: Any,
        tgt_spec: Any,
        mesh: Any,
    ) -> float:
        """Cost of redistributing using DTensor's cost model."""
        from ..cost_models.collective_runtime_estimation import estimate_strategy_comms_cost

        return estimate_strategy_comms_cost(src_spec, tgt_spec)

    def apply_solution(
        self,
        gm: Any,
        solution: dict,
        mesh: Any,
        params_spec: Any = None,
        buffers_spec: Any = None,
    ) -> Any:
        """Apply solution using existing apply_sharding_to_model.

        Converts OpOption solution back to the {node: OpSpec} format
        that apply_sharding_to_model expects.
        """
        from ..apply_sharding import apply_sharding_to_model

        # Convert {node: OpOption} to {node: OpSpec}
        op_spec_solution = {}
        for node, opt in solution.items():
            if hasattr(opt, '_op_spec'):
                op_spec_solution[node] = opt._op_spec
            else:
                # opt.output_spec and opt.input_specs are DTensorSpecs
                from torch.distributed.tensor._op_schema import OpSpec
                op_spec_solution[node] = OpSpec(
                    opt.output_spec,
                    input_specs=opt.input_specs,
                    redistribute_cost=opt.redistribute_costs or [[0.0]],
                )

        return apply_sharding_to_model(
            gm, op_spec_solution,
            params_spec=params_spec,
            buffers_spec=buffers_spec,
        )

    def _op_strategy_to_options(self, op_strategy) -> list[OpOption]:
        """Convert an OpStrategy to a list of OpOption."""
        from ..cost_models.compute_estimation import estimate_strategy_runtime_cost

        results = []
        for op_spec in op_strategy.strategies:
            opt = OpOption(
                output_spec=op_spec.output_specs,
                input_specs=op_spec.input_specs,
                compute_cost=0.0,  # computed later in _build_decision_vars
                comm_cost=0.0,     # computed later via edge costs
                redistribute_costs=op_spec.redistribute_cost,
            )
            # Stash the original OpSpec for apply_solution
            opt._op_spec = op_spec
            opt._op_strategy = op_strategy
            results.append(opt)
        return results
