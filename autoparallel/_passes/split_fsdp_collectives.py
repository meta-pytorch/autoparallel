# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses

import torch
import torch.utils._pytree as pytree
from torch._functorch._aot_autograd.descriptors import AOTOutput
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs


@dataclasses.dataclass(frozen=True)
class PrefetchOutput(AOTOutput):
    pass


def split_fsdp_prefetch(
    gm: torch.fx.GraphModule,
) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
    g = gm.graph
    g_ins = g.find_nodes(op="placeholder")
    prefetch_g_outs_map = {}

    for g_in in g_ins:
        n = g_in
        while True:
            if len(n.users) != 1:
                break
            user = next(iter(n.users))
            if len(user.all_input_nodes) > 1:
                break
            n = user
        prefetch_g_outs_map[g_in] = n

    prefetch_g_outs = list(prefetch_g_outs_map.values())
    prefetch_g_outs_descs: list[AOTOutput] = [
        PrefetchOutput() for _ in range(len(prefetch_g_outs))
    ]

    prefetch_g = _extract_graph_with_inputs_outputs(
        g,
        g_ins,
        prefetch_g_outs,
        prefetch_g_outs_descs,
    )

    g_outs = pytree.arg_tree_leaves(*(n.args for n in g.find_nodes(op="output")))
    g_outs_descs = pytree.arg_tree_leaves(
        next(iter(g.find_nodes(op="output"))).meta.get("desc", [None] * len(g_outs))
    )
    main_g = _extract_graph_with_inputs_outputs(
        g,
        prefetch_g_outs,
        g_outs,
        g_outs_descs,
    )
    main_gm = torch.fx._lazy_graph_module._make_graph_module(gm, main_g)
    prefetch_gm = torch.fx._lazy_graph_module._make_graph_module(gm, prefetch_g)
    return prefetch_gm, main_gm
