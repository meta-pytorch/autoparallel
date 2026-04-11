# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import collections
import itertools
import logging
import operator
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.utils._pytree as pytree
from torch.distributed._tensor.placement_types import Placement, TensorMeta
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpSpec,
    OpStrategy,
    RuntimeSchemaInfo,
    TupleStrategy,
)
from torch.distributed.tensor._ops.utils import generate_redistribute_costs
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.utils._pytree import tree_flatten, tree_map_only

from autoparallel.shardings.propagation_rules import generate_dummy_redistribute_costs

from .dtensor_sharding_helpers import get_op_strategy, with_implicit_strategies
from .propagation_rules import _op_rules, remove_invalid_configs

logger = logging.getLogger(__name__)


def _get_meta_tensors_for_op(op, user_args, user_kwargs):
    out_t = op(*user_args, **user_kwargs)

    if isinstance(out_t, torch.Tensor):
        out_tensor_meta = TensorMeta(out_t.shape, out_t.stride(), out_t.dtype)
    else:
        out_tensor_meta = tree_map_only(
            torch.Tensor, lambda x: TensorMeta(x.shape, x.stride(), x.dtype), out_t
        )

    input_tensor_metas = tree_flatten(user_args)[0]
    input_tensor_metas = tree_map_only(
        torch.Tensor,
        lambda x: TensorMeta(x.shape, x.stride(), x.dtype),
        input_tensor_metas,
    )
    input_tensor_metas = tuple(
        x for x in input_tensor_metas if isinstance(x, TensorMeta)
    )
    return out_tensor_meta, input_tensor_metas


def propagate_tensor_meta(op, user_args, user_kwargs, out_strat):
    new_tensor_meta, tensor_metas = _get_meta_tensors_for_op(op, user_args, user_kwargs)

    for strat in out_strat.strategies:
        if strat.output_specs is None:
            continue
        if isinstance(new_tensor_meta, TensorMeta):
            strat.output_spec.tensor_meta = new_tensor_meta
        else:
            # This is basically trying to workaround this behavior of DTensor
            # https://github.com/pytorch/pytorch/pull/159205#issuecomment-3121562920
            # would be good to have changed in main
            new_output_specs = []
            mesh = strat.mesh
            for ospec, tm in zip(strat.output_specs, new_tensor_meta):
                # replace None with Replicate() in the output_spec
                # as this is done by default but somewhere further
                # down the line in DTensor
                if ospec is None and isinstance(tm, TensorMeta):
                    ospec = DTensorSpec(
                        mesh=mesh, placements=(Replicate(),) * mesh.ndim
                    )
                # Some multi-output ops (e.g. SDPA backward) have optional
                # outputs that are None at runtime. DTensor's strategy still
                # creates a DTensorSpec for the position, but the actual
                # output doesn't exist (hence, tm is None). Replace with None so downstream
                # code (remove_invalid_configs, validate) skips it gracefully.
                elif ospec is not None and tm is None:
                    ospec = None
                new_output_specs.append(ospec)
            strat.output_specs = tuple(new_output_specs)

            for ospec, tm in zip(strat.output_specs, new_tensor_meta):
                if ospec is not None:
                    if ospec.tensor_meta != tm:
                        # This is overcoming some limitations of the lack of
                        # tensor_meta for sdpa which returns None
                        # we should just fix this all across the board
                        if ospec.tensor_meta is None:
                            ospec.tensor_meta = tm
                        else:
                            assert tm is None
        if strat.input_specs is None:

            supported_ops = {
                torch.ops.prims.convert_element_type.default,
                torch.ops.aten.clone.default,
                torch.ops.aten.slice.Tensor,
            }
            assert op in supported_ops, (
                f"{op} strategy doesn't have input_specs, only harcoded "
                "{supported_ops} for now"
            )
            strat.input_specs = (strat.output_specs,)
            assert strat.redistribute_cost is None
        # Factory ops and other no-input ops (e.g. iota) have 0 tensor
        # inputs but carry a dummy input_spec equal to output_spec so
        # the solver has something to wire up.
        if len(tensor_metas) == 0:
            assert len(strat.input_specs) == 1, f"{op}, {len(strat.input_specs)}"
        else:
            assert len(tensor_metas) == len(
                strat.input_specs
            ), f"{op}, {len(tensor_metas)}, {len(strat.input_specs)}"

        for tm, ispec in zip(tensor_metas, strat.input_specs):
            if ispec.tensor_meta is None:
                ispec.tensor_meta = tm


def fill_missing_redistribute_cost(op, specs, out_strat):
    for strat in out_strat.strategies:
        # TODO: check me
        if strat.redistribute_cost is None:
            # TODO: the torch.ops.aten.slice.Tensor is wrong here and in the input_spec!!!!!
            handled_ops = {
                torch.ops.aten.ones_like.default,
                torch.ops.aten.full_like.default,
                torch.ops.aten.empty_like.default,
                torch.ops.prims.convert_element_type.default,
                torch.ops.aten.slice.Tensor,
                torch.ops.aten.select.int,
            }
            assert op in handled_ops, f"got {op}, supported ops here are {handled_ops}"
            # assert len(specs) == 1, f"Expected len(specs) == 1, got {len(specs)}"
            redistribute_costs = [
                generate_redistribute_costs(specs[0], strat.output_spec)
            ]
            strat.redistribute_cost = redistribute_costs


def keep_unique_configs(op_strat: OpStrategy) -> OpStrategy:
    added = set()
    filtered_strats = []
    for strat in op_strat.strategies:
        input_specs = strat.input_specs
        output_specs = strat.output_specs
        if isinstance(input_specs, list):
            input_specs = tuple(input_specs)
        if isinstance(output_specs, list):
            output_specs = tuple(output_specs)
        try:
            key = (input_specs, output_specs)
            if key in added:
                continue

            added.add(key)
        except TypeError:
            logger.debug("Failed to hash, skipping dedup")
        filtered_strats.append(strat)
    return OpStrategy(filtered_strats)


def _fingerprint_spec(spec):
    """Hashable fingerprint for a DTensorSpec or tuple of them."""
    if spec is None:
        return None
    if isinstance(spec, DTensorSpec):
        return (spec.placements, spec.tensor_meta)
    # tuple of Optional[DTensorSpec]
    return tuple(_fingerprint_spec(s) for s in spec)


def _fingerprint_arg(arg):
    """Create a hashable fingerprint for a get_placement_options argument."""
    if isinstance(arg, OpStrategy):
        return tuple(_fingerprint_spec(s.output_specs) for s in arg.strategies)
    if isinstance(arg, torch.Tensor):
        return (arg.shape, arg.stride(), arg.dtype)
    if isinstance(arg, (list, tuple)):
        return tuple(_fingerprint_arg(a) for a in arg)
    # int, float, None, bool, dtype, etc. — already hashable
    return arg


def _copy_op_strategy(op_strategy):
    """Lightweight copy of an OpStrategy: new OpSpec wrappers, shared DTensorSpecs."""
    return OpStrategy(
        [
            OpSpec(
                output_specs=s.output_specs,
                input_specs=s.input_specs,
                redistribute_cost=(
                    [list(row) for row in s.redistribute_cost]
                    if s.redistribute_cost is not None
                    else None
                ),
            )
            for s in op_strategy.strategies
        ]
    )


_placement_options_cache: dict[tuple, OpStrategy] = {}


def reset_placement_options_cache():
    _placement_options_cache.clear()


class PlacementOptionsTimer:
    """Accumulates per-phase timing for get_placement_options calls."""

    def __init__(self):
        self.strategy_gen = 0.0
        self.propagate_meta = 0.0
        self.fill_redist_cost = 0.0
        self.filter_dedup = 0.0
        self.call_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        # Per-op breakdown: op -> (total_time, count)
        self.per_op: dict[str, tuple[float, int]] = collections.defaultdict(
            lambda: (0.0, 0)
        )

    def record_op(self, op, elapsed):
        key = str(op)
        prev_time, prev_count = self.per_op[key]
        self.per_op[key] = (prev_time + elapsed, prev_count + 1)

    def report(self):
        total = (
            self.strategy_gen
            + self.propagate_meta
            + self.fill_redist_cost
            + self.filter_dedup
        )
        logger.debug(
            "placement_options breakdown (%d calls, %.3fs total): "
            "strategy_gen=%.3fs, propagate_meta=%.3fs, "
            "fill_redist_cost=%.3fs, filter_dedup=%.3fs, "
            "cache_hits=%d, cache_misses=%d",
            self.call_count,
            total,
            self.strategy_gen,
            self.propagate_meta,
            self.fill_redist_cost,
            self.filter_dedup,
            self.cache_hits,
            self.cache_misses,
        )
        top_ops = sorted(self.per_op.items(), key=lambda kv: -kv[1][0])[:10]
        for op_name, (op_time, op_count) in top_ops:
            logger.debug("  %-60s %.3fs (%d calls)", op_name, op_time, op_count)


_placement_options_timer = PlacementOptionsTimer()


def get_placement_options_timer():
    return _placement_options_timer


def reset_placement_options_timer():
    global _placement_options_timer
    _placement_options_timer = PlacementOptionsTimer()


def get_placement_options(mesh, op, specs, user_args, user_kwargs):
    assert len(specs) == len(user_args)
    timer = _placement_options_timer
    t_start = time.perf_counter()

    try:
        cache_key = (
            op,
            tuple(_fingerprint_arg(s) for s in specs),
            tuple(_fingerprint_arg(a) for a in user_args),
            tuple(_fingerprint_arg(v) for v in user_kwargs.values())
            if user_kwargs
            else (),
        )
        hash(cache_key)  # fail fast if key contains unhashable types (e.g. SymInts)
    except TypeError:
        cache_key = None

    if cache_key is not None and cache_key in _placement_options_cache:
        out_strat = _copy_op_strategy(_placement_options_cache[cache_key])
        timer.call_count += 1
        timer.cache_hits += 1
        timer.record_op(op, time.perf_counter() - t_start)
        return out_strat

    strat = []
    needs_pytree = False
    for spec in specs:
        if isinstance(spec, OpStrategy):
            strat.append(spec)
        elif (
            isinstance(spec, list)
            and len(spec) > 0
            and any(isinstance(x, OpStrategy) for x in spec)
        ):
            strat.append(TupleStrategy(spec))
            needs_pytree = True
        else:
            strat.append(spec)
    strat = tuple(strat)

    op_schema = OpSchema(op, strat, {}, RuntimeSchemaInfo(needs_pytree=needs_pytree))

    t0 = time.perf_counter()
    if op in _op_rules:
        out_strat = _op_rules[op](mesh, op_schema)
    else:
        with with_implicit_strategies():
            out_strat = get_op_strategy(op, op_schema)
    t1 = time.perf_counter()

    # operator.getitem is self-contained: its input is a tuple of tensors
    # but input_specs tracks only the selected element, so
    # propagate_tensor_meta's input count assertion would fail.
    if op is not operator.getitem:
        propagate_tensor_meta(op, user_args, user_kwargs, out_strat)
    t2 = time.perf_counter()

    fill_missing_redistribute_cost(op, specs, out_strat)
    t3 = time.perf_counter()

    out_strat = remove_invalid_configs(out_strat, mesh)
    out_strat = keep_unique_configs(out_strat)
    t4 = time.perf_counter()

    timer.strategy_gen += t1 - t0
    timer.propagate_meta += t2 - t1
    timer.fill_redist_cost += t3 - t2
    timer.filter_dedup += t4 - t3
    timer.call_count += 1
    timer.cache_misses += 1
    timer.record_op(op, t4 - t_start)

    if cache_key is not None:
        _placement_options_cache[cache_key] = out_strat
    return out_strat


def get_local_map_placement_option(
    mesh,
    specs,
    user_args,
    node,
    local_map_kwargs,
):
    in_placements = local_map_kwargs["in_placements"]
    out_placements = local_map_kwargs["out_placements"]
    assert in_placements is not None
    assert out_placements is not None
    assert (
        local_map_kwargs.get("in_grad_placements", None) is None
    ), "Not yet implemented"
    assert local_map_kwargs.get("device_mesh", None) in (
        mesh,
        None,
    ), "Not yet implemented"
    assert "call_local_map" in str(node.target) or "call_local_map_backward" in str(
        node.target
    )
    in_specs = []
    num_activation_inputs = len(user_args) - len(in_placements)
    # activations are always replicated
    replicated = tuple(Replicate() for _ in range(mesh.ndim))
    for activation in user_args[:num_activation_inputs]:
        # we have activation inputs for the bwd hop
        if isinstance(activation, torch.SymInt):
            in_specs.append(None)
        else:
            in_specs.append(
                DTensorSpec(
                    mesh=mesh,
                    placements=replicated,
                    tensor_meta=TensorMeta(
                        shape=activation.shape,
                        stride=activation.stride(),
                        dtype=activation.dtype,
                    ),
                )
            )

    assert len(user_args) == (num_activation_inputs + len(in_placements))

    for example, placement in zip(user_args[num_activation_inputs:], in_placements):
        if placement is None:
            # not a dtensor
            in_specs.append(None)
            continue

        in_specs.append(
            DTensorSpec(
                mesh=mesh,
                placements=placement,
                tensor_meta=TensorMeta(
                    shape=example.shape,
                    stride=example.stride(),
                    dtype=example.dtype,
                ),
            )
        )

    out_specs = []
    output_val = node.meta["val"]
    assert isinstance(output_val, (torch.Tensor, list, tuple))
    outs = output_val if isinstance(output_val, (list, tuple)) else [output_val]
    for example, placement in zip(outs, out_placements):
        if example is None:
            # Due to how HOP backward is partitioned, it can return None
            out_specs.append(None)
            continue

        if placement is None:
            # not a dtensor
            out_specs.append(None)
            continue

        elif isinstance(placement, Placement):
            placement = [placement]

        assert isinstance(placement, (list, tuple)), "Not implemented"
        out_specs.append(
            DTensorSpec(
                mesh=mesh,
                placements=placement,
                tensor_meta=TensorMeta(
                    shape=example.shape,
                    stride=example.stride(),
                    dtype=example.dtype,
                ),
            )
        )

    for example in outs[len(out_placements) :]:
        if example is None or isinstance(example, torch.SymInt):
            # Due to how HOP backward is partitioned, it can return None or SymInt
            out_specs.append(None)
            continue
        # we have activation outputs for the fwd hop
        out_specs.append(
            DTensorSpec(
                mesh=mesh,
                placements=replicated,
                tensor_meta=TensorMeta(
                    shape=example.shape,
                    stride=example.stride(),
                    dtype=example.dtype,
                ),
            )
        )

    redistribute_costs = []
    for user_strategy, input_spec in zip(specs, in_specs):
        if input_spec is None:
            costs = generate_dummy_redistribute_costs(user_strategy)
        else:
            costs = generate_redistribute_costs(user_strategy, input_spec)
        redistribute_costs.append(costs)

    return OpStrategy(
        [
            OpSpec(
                output_specs=tuple(out_specs),
                input_specs=tuple(in_specs),
                redistribute_cost=redistribute_costs,
            )
        ]
    )


def _is_flex_attention_hop(node):
    target = node.target
    return isinstance(target, torch._ops.HigherOrderOperator) and target.name() in (
        "flex_attention",
        "flex_attention_backward",
    )


def get_flex_attention_placement_option(mesh, specs, user_args, node):
    """Build OpStrategy for flex_attention / flex_attention_backward HOPs.

    Attention is independent per (batch, head) pair, so we enumerate all
    combinations of {Replicate, Shard(0), Shard(1)} across mesh dimensions.
    Block-mask and other auxiliary tensors are always replicated.
    """
    flat_orig, _ = tree_flatten(node.args)
    flat_specs, _ = tree_flatten(specs)
    flat_uargs, _ = tree_flatten(user_args)

    # Keep only FX Node entries (tensor nodes AND GraphModule nodes).
    node_specs = []
    node_uargs = []
    is_tensor_input = []
    for orig, spec, uarg in zip(flat_orig, flat_specs, flat_uargs):
        if isinstance(orig, torch.fx.Node):
            node_specs.append(spec)
            node_uargs.append(uarg)
            is_tensor_input.append(isinstance(uarg, torch.Tensor))

    # Attention tensors have the same batch (dim 0) and heads (dim 1) as Q.
    q_val = node.args[0].meta["val"]
    B, H = q_val.shape[0], q_val.shape[1]

    def is_attention_tensor(t):
        return (
            isinstance(t, torch.Tensor)
            and t.ndim >= 2
            and t.shape[0] == B
            and t.shape[1] == H
        )

    replicated = tuple(Replicate() for _ in range(mesh.ndim))

    # Valid per-mesh-dim placements for attention tensors.
    per_dim_options = [Replicate(), Shard(0), Shard(1)]
    all_placements = list(itertools.product(per_dim_options, repeat=mesh.ndim))

    # Build output specs structure. flex_attention returns a tuple.
    output_val = node.meta["val"]
    assert isinstance(output_val, (tuple, list))

    strategies = []
    for placement in all_placements:
        placement = tuple(placement)

        in_specs = []
        for uarg, producer_strat, is_tensor in zip(
            node_uargs, node_specs, is_tensor_input
        ):
            if not is_tensor:
                # GraphModule submodule — no tensor to shard.
                in_specs.append(None)
            elif is_attention_tensor(uarg):
                in_specs.append(
                    DTensorSpec(
                        mesh=mesh,
                        placements=placement,
                        tensor_meta=TensorMeta(uarg.shape, uarg.stride(), uarg.dtype),
                    )
                )
            else:
                # Auxiliary tensor (block mask etc.) — always replicate.
                in_specs.append(
                    DTensorSpec(
                        mesh=mesh,
                        placements=replicated,
                        tensor_meta=TensorMeta(uarg.shape, uarg.stride(), uarg.dtype),
                    )
                )

        out_specs = []
        for out in output_val:
            if isinstance(out, torch.Tensor) and is_attention_tensor(out):
                out_specs.append(
                    DTensorSpec(
                        mesh=mesh,
                        placements=placement,
                        tensor_meta=TensorMeta(out.shape, out.stride(), out.dtype),
                    )
                )
            else:
                out_specs.append(None)

        redistribute_costs = []
        for producer_strat, spec in zip(node_specs, in_specs):
            if spec is None:
                redistribute_costs.append(
                    generate_dummy_redistribute_costs(producer_strat)
                )
            else:
                redistribute_costs.append(
                    generate_redistribute_costs(producer_strat, spec)
                )

        strategies.append(
            OpSpec(
                output_specs=tuple(out_specs),
                input_specs=tuple(in_specs),
                redistribute_cost=redistribute_costs,
            )
        )

    return OpStrategy(strategies)


def get_placement_options_for_node(mesh, node, specs, user_args, user_kwargs):
    if local_map_kwargs := node.meta.get("local_map_kwargs", {}):
        assert not user_kwargs
        return get_local_map_placement_option(
            mesh, specs, user_args, node, local_map_kwargs
        )
    if _is_flex_attention_hop(node):
        return get_flex_attention_placement_option(mesh, specs, user_args, node)
    return get_placement_options(mesh, node.target, specs, user_args, user_kwargs)


def _get_device_from_mesh(mesh):
    if mesh.device_type == "cpu":
        return torch.device("cpu")
    device_handle = _get_device_handle(mesh.device_type)
    return torch.device(mesh.device_type, device_handle.current_device())


# An FX graph interpreter that logs inputs and outputs of each node
# with a few exceptions for c10d ops
class DebugInterpreter(torch.fx.Interpreter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logs = []

    def log(self, node: str, args: Iterable[Any], inputs_or_outputs: str):
        leaves, _ = pytree.tree_flatten(args)
        for i, arg in enumerate(leaves):
            if not isinstance(arg, torch.Tensor):
                self._logs.append(f"{node=}, {inputs_or_outputs}[{i}]={arg}")
                continue

            if arg.numel() == 0:
                self._logs.append(f"{node=}, {inputs_or_outputs}[{i}].numel()=0")
                continue

            if arg.is_complex():
                real = torch.hash_tensor(arg.real)
                imag = torch.hash_tensor(arg.imag)
                self._logs.append(f"{node=}, {inputs_or_outputs}[{i}], {real=} {imag=}")
                continue

            self._logs.append(
                f"{node=}, {inputs_or_outputs}[{i}]={torch.hash_tensor(arg)} nan={torch.any(torch.isnan(arg))}"
            )

    def run_node(self, n: torch.fx.Node) -> Any:
        args, kwargs = self.fetch_args_kwargs_from_env(n)

        # reading wait_tensor inputs is undefined behavior
        if "wait_tensor" not in n.name:
            args, _ = self.fetch_args_kwargs_from_env(n)
            self.log(n.name, args, "args")

        out = super().run_node(n)

        # reading functional collectives outputs before wait_tensor is undefined behavior
        if "c10d" not in str(n.target):
            outs = out
            if isinstance(outs, torch.Tensor):
                outs = [outs]
            self.log(n.name, outs, "outs")

        return out

    def get_logs(self):
        return self._logs


# Always prints from rank 0 to rank N
def print_rank_by_rank(msg: Any):
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch.distributed.barrier()
    for i in range(world_size):
        if rank == i:
            logger.debug(f"{rank=} start")
            logger.debug(msg)
            logger.debug(f"{rank=} done")
        torch.distributed.barrier()


def hash_tensor(t: torch.Tensor) -> str:
    if isinstance(t, torch.distributed.tensor.DTensor):
        t = t.to_local()
        return f"DTensor({hash_tensor(t)})"

    if t.is_complex():
        return f"real={hash_tensor(t.real)}, imag={hash_tensor(t.imag)})"

    return f"{torch.hash_tensor(t)}"


class NumericsLogger:
    def __init__(self, base_dir: str):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.rank = torch.distributed.get_rank()
        self.dir = self._create_run_dir()

    def _create_run_dir(self) -> Path:
        """
        Find the next available integer directory name under base_dir.
        Example: base_dir/0, base_dir/1, base_dir/2, ...
        """
        existing = [
            int(p.name) for p in self.base.iterdir() if p.is_dir() and p.name.isdigit()
        ]
        next_id = (max(existing) + 1) if existing else 0
        run_dir = self.base / str(next_id)
        torch.distributed.barrier()
        if self.rank == 0:
            run_dir.mkdir()
        torch.distributed.barrier()
        return run_dir

    def log_model_weights(self, parallel_mod):
        if self.rank == 0:
            path = self.dir / "weights.log"

            logs = []
            for name, param in parallel_mod.named_parameters():
                logs.append(f"{name=} hash={hash_tensor(param)}")
            for name, buf in parallel_mod.named_buffers():
                logs.append(f"{name=} hash={hash_tensor(buf)}")

            with open(path, "a") as f:
                f.write("\n".join(logs) + "\n")

            logger.info(f"Weight hashes written to {path}")

    def log_fw_intermediates(self, logs):
        rank = torch.distributed.get_rank()
        path = self.dir / f"rank_{rank}_fw_intermediates.log"
        with open(path, "a") as f:
            f.write("\n".join(logs) + "\n")

    def log_diff(self, t, rank=0, prefix="?"):
        if self.rank == rank:
            path = self.dir / "diff.log"
            if isinstance(t, torch.distributed.tensor.DTensor):
                t = t.to_local()
            with open(path, "a") as f:
                f.write(f"[{prefix}] hash={hash_tensor(t)}, norm={torch.norm(t)}\n")

    def log_pp_model_weights(self, orig_mod, stage_mods, num_world_stages, should_log):
        path = self.dir / "pp_weights.log"

        torch.distributed.barrier()
        # First print the params of every stage
        for i in range(num_world_stages):
            if should_log and i in stage_mods:
                param_logs = []
                real_params = dict(stage_mods[i].named_parameters())
                for name, _ in orig_mod.named_parameters():
                    if name not in real_params:
                        continue
                    param = real_params[name]
                    param_logs.append(f"{name=} hash={hash_tensor(param)}")
                with open(path, "a") as f:
                    f.write("\n".join(param_logs) + "\n")
            torch.distributed.barrier()

        # Then print the buffers of every stage
        for i in range(num_world_stages):
            if should_log and i in stage_mods:
                buffer_logs = []
                real_buffers = dict(stage_mods[i].named_buffers())
                for name, _ in orig_mod.named_buffers():
                    if name not in real_buffers:
                        continue
                    buffer = real_buffers[name]
                    buffer_logs.append(f"{name=} hash={hash_tensor(buffer)}")
                with open(path, "a") as f:
                    f.write("\n".join(buffer_logs) + "\n")
            torch.distributed.barrier()

        if self.rank == 0:
            logger.info(f"Weight hashes written to {path}")

    def log_pp_grads(self, orig_mod, stage_mods, num_world_stages, should_log):
        path = self.dir / "diff.log"

        for i in range(num_world_stages):
            if should_log and i in stage_mods:
                grad_logs = []
                real_params = dict(stage_mods[i].named_parameters())
                for name, _ in orig_mod.named_parameters():
                    if name not in real_params:
                        continue
                    grad = real_params[name].grad
                    if grad is None:
                        grad_logs.append(f"[grad {name}] None")
                    else:
                        grad = grad.to_local()
                        grad_logs.append(
                            f"[grad {name}] hash={hash_tensor(grad)}, norm={torch.norm(grad)}"
                        )
                with open(path, "a") as f:
                    f.write("\n".join(grad_logs) + "\n")
            torch.distributed.barrier()


def debug_boxed_nop_preserve_node_meta(fx_g, example_inputs, numerics_logger):
    from torch._inductor.fx_passes.post_grad import view_to_reshape

    view_to_reshape(fx_g)

    def run(args):
        with torch.fx.traceback.preserve_node_meta():
            interp = DebugInterpreter(fx_g)
            out = interp.boxed_run(args)
            mylogs = interp.get_logs()
            if numerics_logger:
                numerics_logger.log_fw_intermediates(mylogs)
            return out

    run._boxed_call = True
    return run
