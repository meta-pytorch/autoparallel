# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from contextlib import contextmanager
from functools import partial
from typing import Any

import torch
import torch.fx.node
import torch.utils._pytree as pytree
from torch._functorch._aot_autograd.descriptors import AOTOutput
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs
from torch._inductor.fx_passes.bucketing import (
    is_all_gather_into_tensor,
    is_reduce_scatter_tensor,
)


@contextmanager
def exclude_from_fx_side_effectful(exclude_vals: set[Any]):
    original_val = torch.fx.node._side_effectful_functions.copy()
    try:
        torch.fx.node._side_effectful_functions -= exclude_vals
        yield
    finally:
        torch.fx.node._side_effectful_functions.clear()
        torch.fx.node._side_effectful_functions.update(original_val)


exclude_wait_from_fx_side_effectful = partial(
    exclude_from_fx_side_effectful,
    {
        torch.ops._c10d_functional.wait_tensor,
        torch.ops._c10d_functional.wait_tensor.default,
    },
)


@dataclasses.dataclass(frozen=True)
class PrefetchOutput(AOTOutput):
    pass


@dataclasses.dataclass(frozen=True)
class EpilogueInput(AOTOutput):
    pass


def split_fsdp_prefetch(g: torch.fx.Graph) -> tuple[torch.fx.Graph, torch.fx.Graph]:
    g_ins = g.find_nodes(op="placeholder")
    prefetch_g_outs_map = []

    for g_in in g_ins:
        n = g_in
        last_ag = None
        while True:
            if len(n.users) != 1:
                break
            user = next(iter(n.users))
            if len(user.all_input_nodes) > 1:
                break
            n = user
            if is_all_gather_into_tensor(n):
                last_ag = n
        if last_ag is None:
            prefetch_g_outs_map.append(g_in)
        else:
            w_n = next(iter(last_ag.users))
            prefetch_g_outs_map.append(w_n)

    prefetch_g_outs = prefetch_g_outs_map
    prefetch_g_outs_descs: list[AOTOutput] = [
        PrefetchOutput() for _ in range(len(prefetch_g_outs))
    ]
    g_outs = pytree.arg_tree_leaves(*(n.args for n in g.find_nodes(op="output")))
    g_outs_descs = pytree.arg_tree_leaves(
        next(iter(g.find_nodes(op="output"))).meta.get("desc", [None] * len(g_outs))
    )
    with exclude_wait_from_fx_side_effectful():
        prefetch_g = _extract_graph_with_inputs_outputs(
            g,
            g_ins,
            prefetch_g_outs,
            prefetch_g_outs_descs,
            ignore_must_be_in_fw_bw=True,
        )

        main_g = _extract_graph_with_inputs_outputs(
            g,
            prefetch_g_outs,
            g_outs,
            g_outs_descs,
            ignore_must_be_in_fw_bw=True,
        )
    return prefetch_g, main_g


def split_fsdp_reduce_scatters_epilogue(
    g: torch.fx.Graph,
) -> tuple[torch.fx.Graph, torch.fx.Graph]:
    g_ins = g.find_nodes(op="placeholder")
    g_outs = pytree.arg_tree_leaves(*(n.args for n in g.find_nodes(op="output")))
    g_outs_descs = pytree.arg_tree_leaves(
        next(iter(g.find_nodes(op="output"))).meta.get("desc", [None] * len(g_outs))
    )

    g_outs_map = []
    for g_out in g_outs:
        n = g_out
        last_rs = None
        while n is not None:
            if len(n.all_input_nodes) != 1:
                break
            n_in = n.all_input_nodes[0]
            if len(n_in.users) > 1:
                break
            prev_n = n
            n = n_in
            if is_reduce_scatter_tensor(prev_n):
                # In AP for mesh dim > 1
                # The reduction of gradients happen in multiple steps
                last_rs = n
        if last_rs is not None:
            g_outs_map.append(last_rs)
        else:
            g_outs_map.append(g_out)

    epi_g_ins = [n for n in g_outs_map if n is not None]
    epi_g_ins_descs: list[AOTOutput] = [EpilogueInput() for _ in range(len(epi_g_ins))]

    with exclude_wait_from_fx_side_effectful():
        main_g = _extract_graph_with_inputs_outputs(
            g,
            g_ins,
            epi_g_ins,
            epi_g_ins_descs,
            ignore_must_be_in_fw_bw=True,
        )
        epi_g = _extract_graph_with_inputs_outputs(
            g,
            epi_g_ins,
            g_outs,
            g_outs_descs,
            ignore_must_be_in_fw_bw=True,
        )

    return main_g, epi_g
