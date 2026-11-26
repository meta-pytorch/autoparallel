# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""``auto_functionalized_v2`` nodes: custom ops with ``mutates_args``.

``aot_export_module`` turns such a call into ``auto_functionalized_v2(op, **kwargs)``, where
every tensor is a kwarg and a mutated argument is a base in ``_all_bases``. These tests run
one-op graphs through the solver on a fake process group, and through the apply path on a
real (gloo, CPU) group, comparing with the eager op.
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch._functorch._aot_autograd.descriptors import PlainAOTInput, PlainAOTOutput
from torch._functorch.aot_autograd import aot_export_module
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor.experimental import register_sharding
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel.api import AutoParallel
from autoparallel.apply_sharding import (
    _build_physical_placements,
    _lower_to_parallel_graph,
    _make_local_args,
)
from autoparallel.cost_models.compute_estimation import (
    estimate_strategy_runtime_cost,
    override_cost_model_device_name,
)
from autoparallel.graph_passes.graph_utils import all_input_nodes
from autoparallel.optimize_sharding import ShardingOptimizer
from autoparallel.shardings.auto_functionalize import (
    decode_auto_functionalized_v2,
    is_auto_functionalized_v2,
)
from autoparallel.shardings.placement_options import reset_placement_options_cache

# ---------------------------------------------------------------------------
# Toy mutating ops. Each has a registered DTensor rule (``register_sharding``); the
# rules are the thing the solver must pick up through the HOP.
# ---------------------------------------------------------------------------


@torch.library.custom_op("ap_af_test::rowop", mutates_args=("buf",))
def rowop(x: torch.Tensor, buf: torch.Tensor, alpha: float) -> torch.Tensor:
    """``buf += alpha * x``; returns ``2 * x``. Pointwise, so any dim may be sharded."""
    buf.add_(x * alpha)
    return x * 2


@rowop.register_fake
def _rowop_fake(x, buf, alpha):
    return torch.empty_like(x)


@torch.library.custom_op("ap_af_test::rowop_", mutates_args=("buf",))
def rowop_inplace_named(
    x: torch.Tensor, buf: torch.Tensor, alpha: float
) -> torch.Tensor:
    """``rowop`` under a name DTensor treats as an in-place op (trailing underscore)."""
    buf.add_(x * alpha)
    return x * 2


@rowop_inplace_named.register_fake
def _rowop_inplace_named_fake(x, buf, alpha):
    return torch.empty_like(x)


@torch.library.custom_op("ap_af_test::pick", mutates_args=("ids", "alpha"))
def pick(scores: torch.Tensor, ids: torch.Tensor, alpha: torch.Tensor) -> None:
    """Row-wise arg-max into two mutated outputs, returning nothing (like the router ops)."""
    top, idx = scores.max(dim=-1, keepdim=True)
    ids.copy_(idx)
    alpha.copy_(top)


@pick.register_fake
def _pick_fake(scores, ids, alpha) -> None:
    return None


@register_sharding(torch.ops.ap_af_test.rowop.default)
@register_sharding(torch.ops.ap_af_test.rowop_.default)
def _rowop_rule(x, buf, alpha):
    options = [([Shard(d)], [Shard(d), Shard(d), None]) for d in range(x.ndim)]
    options.append(([Replicate()], [Replicate(), Replicate(), None]))
    return options


@register_sharding(torch.ops.ap_af_test.pick.default)
def _pick_rule(scores, ids, alpha):
    # Rows are independent; the reduced dim (-1) may not be sharded. No outputs.
    return [([], [Shard(0), Shard(0), Shard(0)]), ([], [Replicate()] * 3)]


class OneOp(torch.nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, x, buf):
        y = self.op(x, buf, 0.5)
        return (y,)


class Pick(torch.nn.Module):
    def forward(self, scores, ids, alpha):
        torch.ops.ap_af_test.pick(scores, ids, alpha)
        return (ids * 1, alpha * 1)


ROWOP = torch.ops.ap_af_test.rowop
ROWOP_UNDERSCORE = torch.ops.ap_af_test.rowop_


def _export(module, *inputs):
    """One-op graph of ``module`` as the step compiler would capture it."""
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    # As the step compiler captures: triton_ops stay opaque nodes, and a mutating one becomes
    # auto_functionalized_v2 instead of being decomposed into its kernel launch.
    with fake_mode, torch._functorch.config.patch(decompose_custom_triton_ops=False):
        fakes = tuple(torch.empty_like(t) for t in inputs)
        gm, _ = aot_export_module(module, fakes, trace_joint=False)
    for i, node in enumerate(gm.graph.find_nodes(op="placeholder")):
        node.meta["desc"] = PlainAOTInput(idx=i)
    output = gm.graph.find_nodes(op="output")[0]
    output.meta["desc"] = [PlainAOTOutput(idx=i) for i in range(len(output.args[0]))]
    return gm, fake_mode


def _af_node(gm):
    (node,) = [n for n in gm.graph.nodes if is_auto_functionalized_v2(n)]
    return node


def _optimizer(gm, fake_mode, mesh):
    reset_placement_options_cache()
    with override_cost_model_device_name("NVIDIA H200"), fake_mode:
        return ShardingOptimizer(gm, mesh)


@pytest.fixture
def mesh_1d():
    return DeviceMesh("cpu", torch.arange(4), mesh_dim_names=("dp",))


@pytest.fixture
def mesh_2d():
    return DeviceMesh("cpu", torch.arange(8).reshape(2, 4), mesh_dim_names=("dp", "tp"))


def _placements(spec):
    return tuple(spec.placements)


# ---------------------------------------------------------------------------
# Graph structure
# ---------------------------------------------------------------------------


def test_node_structure_and_decode():
    gm, _ = _export(OneOp(ROWOP), torch.randn(8, 4), torch.zeros(8, 4))
    node = _af_node(gm)
    assert node.args == (ROWOP.default,)
    assert "x" in node.kwargs and "_all_bases" in node.kwargs

    call = decode_auto_functionalized_v2(node)
    assert call.op == ROWOP.default
    assert call.base_of_arg == {"buf": 0}
    assert not call.replicate_only
    assert len(call.bases) == 1


def test_all_input_nodes_reads_kwargs():
    gm, _ = _export(OneOp(ROWOP), torch.randn(8, 4), torch.zeros(8, 4))
    node = _af_node(gm)
    x, buf = gm.graph.find_nodes(op="placeholder")
    # Duplicates preserved, kwargs order: x, then _all_bases.
    assert all_input_nodes(node) == [x, buf]


class SliceMutation(torch.nn.Module):
    def forward(self, x, buf):
        y = torch.ops.ap_af_test.rowop(x[:, :2], buf[:, :2], 0.5)
        return (y,)


def test_view_of_base_is_replicate_only():
    gm, _ = _export(SliceMutation(), torch.randn(8, 4), torch.zeros(8, 4))
    assert decode_auto_functionalized_v2(_af_node(gm)).replicate_only


# ---------------------------------------------------------------------------
# Strategies and the solve
# ---------------------------------------------------------------------------


def _node_strategies(optimizer, node):
    return optimizer.strats[optimizer._normalize_node(node)].strategies


@pytest.mark.parametrize(
    "op", [ROWOP, ROWOP_UNDERSCORE], ids=["plain", "inplace_named"]
)
def test_strategies_are_the_registered_rule(mesh_1d, op):
    gm, fake_mode = _export(OneOp(op), torch.randn(64, 8), torch.zeros(64, 8))
    opt = _optimizer(gm, fake_mode, mesh_1d)
    strategies = _node_strategies(opt, _af_node(gm))

    got = {tuple(_placements(s) for s in strat.input_specs) for strat in strategies}
    want = {
        ((Shard(0),), (Shard(0),)),
        ((Shard(1),), (Shard(1),)),
        ((Replicate(),), (Replicate(),)),
    }
    assert got == want
    for strat in strategies:
        x_spec, buf_spec = strat.input_specs
        ret_spec, base_spec = strat.output_specs
        # (op return, updated base): the base is placed like the mutated argument.
        assert _placements(ret_spec) == _placements(x_spec)
        assert _placements(base_spec) == _placements(buf_spec)
        assert base_spec.tensor_meta.shape == torch.Size([64, 8])
        assert len(strat.redistribute_cost) == 2


def test_no_return_op_with_two_mutated_args(mesh_1d):
    gm, fake_mode = _export(
        Pick(),
        torch.randn(64, 16),
        torch.zeros(64, 1, dtype=torch.int64),
        torch.zeros(64, 1),
    )
    opt = _optimizer(gm, fake_mode, mesh_1d)
    node = _af_node(gm)
    assert decode_auto_functionalized_v2(node).base_of_arg == {"ids": 0, "alpha": 1}
    for strat in _node_strategies(opt, node):
        # value = (None, updated ids, updated alpha)
        assert len(strat.output_specs) == 3
        assert strat.output_specs[0] is None
        scores, ids, alpha = strat.input_specs
        assert _placements(strat.output_specs[1]) == _placements(ids)
        assert _placements(strat.output_specs[2]) == _placements(alpha)
        assert strat.output_specs[1].tensor_meta.dtype == torch.int64


def test_unregistered_op_falls_back_to_replicate(mesh_1d):
    @torch.library.custom_op("ap_af_test::norule", mutates_args=("buf",))
    def norule(x: torch.Tensor, buf: torch.Tensor) -> None:
        buf.add_(x)

    @norule.register_fake
    def _(x, buf) -> None:
        return None

    class M(torch.nn.Module):
        def forward(self, x, buf):
            torch.ops.ap_af_test.norule(x, buf)
            return (buf * 1,)

    gm, fake_mode = _export(M(), torch.randn(64, 8), torch.zeros(64, 8))
    opt = _optimizer(gm, fake_mode, mesh_1d)
    (strat,) = _node_strategies(opt, _af_node(gm))
    assert all(_placements(s) == (Replicate(),) for s in strat.input_specs)


def test_view_of_base_gets_only_the_replicated_strategy(mesh_1d):
    gm, fake_mode = _export(SliceMutation(), torch.randn(64, 8), torch.zeros(64, 8))
    opt = _optimizer(gm, fake_mode, mesh_1d)
    (strat,) = _node_strategies(opt, _af_node(gm))
    assert all(_placements(s) == (Replicate(),) for s in strat.input_specs)
    assert all(_placements(s) == (Replicate(),) for s in strat.output_specs)


def test_solve_picks_registered_sharding(mesh_1d):
    gm, fake_mode = _export(
        OneOp(ROWOP), torch.randn(4096, 1024), torch.zeros(4096, 1024)
    )
    opt = _optimizer(gm, fake_mode, mesh_1d)
    opt.add_sharded_input_constraint([(Shard(0),), (Shard(0),)])
    opt.add_sharded_output_constraint([(Shard(0),), (Shard(0),)])
    with override_cost_model_device_name("NVIDIA H200"), fake_mode:
        sol = opt.get_solution()
    spec = sol[_af_node(gm)]
    assert [_placements(s) for s in spec.input_specs] == [(Shard(0),), (Shard(0),)]
    assert [_placements(s) for s in spec.output_specs] == [(Shard(0),), (Shard(0),)]


def test_solve_on_two_mesh_dims(mesh_2d):
    gm, fake_mode = _export(
        OneOp(ROWOP), torch.randn(4096, 1024), torch.zeros(4096, 1024)
    )
    opt = _optimizer(gm, fake_mode, mesh_2d)
    opt.add_sharded_input_constraint([(Shard(0), Shard(1)), (Shard(0), Shard(1))])
    with override_cost_model_device_name("NVIDIA H200"), fake_mode:
        sol = opt.get_solution()
    spec = sol[_af_node(gm)]
    assert [_placements(s) for s in spec.input_specs] == [(Shard(0), Shard(1))] * 2


class ScaleRowsFn(torch.autograd.Function):
    """A mutating op cannot carry an autograd formula, so training code wraps it."""

    @staticmethod
    def forward(ctx, x, buf):
        return torch.ops.ap_af_test.rowop(x, buf, 0.5)

    @staticmethod
    def backward(ctx, grad):
        return grad * 2, None


class WithMutatingOp(torch.nn.Module):
    def __init__(self, dim, rows):
        super().__init__()
        self.lin = torch.nn.Linear(dim, dim, bias=False)
        self.register_buffer("state", torch.zeros(rows, dim))

    def forward(self, x):
        return ScaleRowsFn.apply(self.lin(x), self.state) + self.state


def test_autoparallel_end_to_end_with_a_mutating_op(mesh_1d):
    """The whole pipeline: joint trace (forward + backward), solve, apply."""
    rows, dim = 4096, 1024
    with torch.device("meta"):
        model = WithMutatingOp(dim, rows)

    def input_fn():
        return torch.randn(rows, dim, requires_grad=True)

    reset_placement_options_cache()
    with override_cost_model_device_name("NVIDIA H200"):
        with AutoParallel(model, input_fn, mesh_1d, repeated_subgraphs=False) as autop:
            autop.add_input_constraints([(Shard(0),)])
            autop.add_output_constraints([(Shard(0),)])
            sol = autop.optimize_placement()
            (node,) = [n for n in autop.gm.graph.nodes if is_auto_functionalized_v2(n)]
            chosen = sol[autop.sharding_optimizer._concrete_to_orig.get(node, node)]
            # Sharded rows end to end: no all-gather around the op.
            assert [_placements(s) for s in chosen.input_specs] == [(Shard(0),)] * 2
            autop.apply_placement(sol)
            applied = [
                n for n in autop.parallel_gm.graph.nodes if is_auto_functionalized_v2(n)
            ]
    assert applied, "the applied graph keeps the functionalized op"
    # The op runs on the local shard: rows / 4.
    assert all(n.meta["val"][0].shape == (rows // 4, dim) for n in applied)


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


def test_cost_scales_with_sharding(mesh_1d):
    gm, fake_mode = _export(
        OneOp(ROWOP), torch.randn(65536, 1024), torch.zeros(65536, 1024)
    )
    opt = _optimizer(gm, fake_mode, mesh_1d)
    node = _af_node(gm)
    by_input = {
        _placements(strat.input_specs[0]): strat
        for strat in _node_strategies(opt, node)
    }
    with override_cost_model_device_name("NVIDIA H200"):
        sharded = estimate_strategy_runtime_cost(node, by_input[(Shard(0),)])
        replicated = estimate_strategy_runtime_cost(node, by_input[(Replicate(),)])
        unsharded = estimate_strategy_runtime_cost(node, None)
    assert 0 < sharded < replicated
    # reads x and the base, writes the return and the base: 4 tensors
    assert replicated == pytest.approx(unsharded, rel=1e-6)
    assert replicated == pytest.approx(4 * sharded, rel=0.05)


# ---------------------------------------------------------------------------
# Apply, against the eager op, on a real process group
# ---------------------------------------------------------------------------


def free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def solve_and_run(module, inputs, rank, world_size, input_placements, device_type):
    """Solve + apply the one-op graph of ``module`` on this rank.

    Returns the parallel graph's local outputs, their placements, the solution, the graph
    and this rank's local inputs.
    """
    mesh = init_device_mesh(device_type, (world_size,), mesh_dim_names=("dp",))
    gm, fake_mode = _export(module, *inputs)
    reset_placement_options_cache()
    with override_cost_model_device_name("NVIDIA H200"), fake_mode:
        opt = ShardingOptimizer(gm, mesh)
        opt.add_sharded_input_constraint([(p,) for p in input_placements])
        sol = opt.get_solution()
        physical = _build_physical_placements(sol, {})
        local_args = _make_local_args(gm, physical)
    with torch._functorch.config.patch(decompose_custom_triton_ops=False):
        parallel_gm = _lower_to_parallel_graph(gm, sol, local_args)

    def shard(t, p):
        if p.is_shard():
            return t.chunk(world_size, p.dim)[rank].contiguous()
        return t.clone()

    local_inputs = [shard(t, p) for t, p in zip(inputs, input_placements)]
    # Some kernels settle over their first launches (autotuning): record the second run.
    parallel_gm(*[t.clone() for t in local_inputs])
    outs = parallel_gm(*[t.clone() for t in local_inputs])
    out_node = gm.graph.find_nodes(op="output")[0]
    out_placements = [sol[n].output_specs.placements[0] for n in out_node.args[0]]
    return outs, out_placements, sol, gm, local_inputs


def check_chosen_rule(sol, gm, input_placements):
    """The registered rule was chosen (not the replicated fallback), and agrees with the
    placements the inputs were constrained to."""
    chosen = [_placements(s) for s in sol[_af_node(gm)].input_specs]
    if len(set(input_placements)) == 1:
        assert chosen == [(p,) for p in input_placements], chosen
    else:
        # The rule ties the operands together: the solver redistributes some of them.
        assert len(set(chosen)) == 1, chosen


def check_against_local_eager(
    module,
    mutated,
    local_inputs,
    outs,
    out_placements,
    rank,
    skip_mutated=False,
    tol=None,
):
    """The applied graph, on this rank's shards, equals the eager op on those shards.

    Valid whenever the op's rule makes a per-shard call equal to the shard of the global
    call; that is what a registered rule asserts. The graph returns the mutated inputs
    first, then the module's outputs.
    """
    ref_inputs = [t.clone() for t in local_inputs]
    ref_outs = module(*ref_inputs)
    expected = [ref_inputs[i] for i in mutated] + list(ref_outs)
    assert len(outs) == len(expected), (len(outs), len(expected))
    first = len(mutated) if skip_mutated else 0
    for out, exp in zip(outs[first:], expected[first:]):
        torch.testing.assert_close(out, exp, **(tol or {}))


def _toy_rowop(op):
    def case():
        return OneOp(op), (torch.randn(8, 6), torch.randn(8, 6)), [1]

    return case


def _toy_pick():
    scores = torch.randn(8, 6)
    return (
        Pick(),
        (scores, torch.zeros(8, 1, dtype=torch.int64), torch.zeros(8, 1)),
        [1, 2],
    )


def _worker(rank, world_size, port, case_name, input_placements):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        case = {
            "rowop": _toy_rowop(ROWOP),
            "rowop_": _toy_rowop(ROWOP_UNDERSCORE),
            "pick": _toy_pick,
        }[case_name]
        torch.manual_seed(0)
        module, inputs, mutated = case()
        outs, out_placements, sol, gm, local_inputs = solve_and_run(
            module, inputs, rank, world_size, input_placements, "cpu"
        )
        check_chosen_rule(sol, gm, input_placements)
        if len(set(input_placements)) == 1:
            # (Mismatched operands have no per-shard eager call; the global check covers them.)
            check_against_local_eager(
                module, mutated, local_inputs, outs, out_placements, rank
            )

        # The toy rules are also true globally: the output is the shard of the eager result.
        ref_inputs = [t.clone() for t in inputs]
        expected = [*(ref_inputs[i] for i in mutated), *module(*ref_inputs)]
        for out, exp, plc in zip(outs, expected, out_placements):
            exp_local = exp.chunk(world_size, plc.dim)[rank] if plc.is_shard() else exp
            torch.testing.assert_close(out, exp_local)
    finally:
        dist.destroy_process_group()


def _spawn(case_name, world_size, input_placements):
    mp.spawn(
        _worker,
        args=(world_size, free_port(), case_name, input_placements),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("case_name", ["rowop", "rowop_"])
def test_apply_matches_eager_rowop(case_name, world_size):
    _spawn(case_name, world_size, (Shard(0), Shard(0)))


def test_apply_matches_eager_rowop_sharded_on_dim1():
    _spawn("rowop", 2, (Shard(1), Shard(1)))


def test_apply_redistributes_a_misplaced_base():
    # x row-sharded, buf column-sharded: the op's rule needs them to agree, so apply must
    # redistribute a kwarg/base it finds in kwargs, not only positional args.
    _spawn("rowop", 2, (Shard(0), Shard(1)))


@pytest.mark.parametrize("world_size", [2, 4])
def test_apply_matches_eager_no_return_two_mutated(world_size):
    _spawn("pick", world_size, (Shard(0), Shard(0), Shard(0)))
