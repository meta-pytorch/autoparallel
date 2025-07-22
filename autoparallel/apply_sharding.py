# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import operator

import torch
import torch.nn as nn
from torch._functorch._aot_autograd.descriptors import (
    GradAOTOutput,
    ParamAOTInput,
    PlainAOTInput,
    PlainAOTOutput,
    TangentAOTInput,
)
from torch._functorch._aot_autograd.fx_utils import (
    named_buffer_nodes,
    named_param_nodes,
)
from torch._subclasses.fake_tensor import FakeTensor, unset_fake_temporarily
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._redistribute import redistribute_local_tensor
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard  # noqa
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._pytree import tree_flatten, tree_map_only


def my_redistribute_local_tensor(arg, curr_spec, tgt_spec):
    # if curr_spec.placements == (Shard(0), Shard(0)) and tgt_spec.placements == (
    #     Replicate(),
    #     Shard(0),
    # ):
    #     # TODO: double-check in which cases this is valid
    #     x = curr_spec.placements[0]._to_replicate_tensor(
    #         arg, curr_spec.mesh, 0, curr_spec.shape
    #     )
    # elif curr_spec.placements == (Partial(), Shard(0)) and tgt_spec.placements == (
    #     Shard(0),
    #     Shard(0),
    # ):
    #     x = curr_spec.placements[0]._reduce_shard_value(
    #         arg, curr_spec.mesh, 0, tgt_spec.placements[0]
    #     )
    # elif curr_spec.placements == (Partial(), Shard(1)) and tgt_spec.placements == (Replicate(), Shard(1)):
    #    from IPython import embed; embed(); sys.sdf
    # else:
    x = redistribute_local_tensor(arg, curr_spec, tgt_spec)
    return x


class ApplyShardingInterpreter(torch.fx.Interpreter):
    def __init__(self, module, sharding_placement):
        super().__init__(module, garbage_collect_values=True, graph=None)
        self.sharding_placement = sharding_placement

    def run_node(self, n):
        self._curr_node = n
        return super().run_node(n)

    def redistribute_tensor(self, arg, curr_spec, tgt_spec):
        tgt_placements = tuple(
            p if not p.is_partial() else Replicate() for p in tgt_spec.placements
        )
        x = arg
        if curr_spec.placements != tgt_spec.placements:
            tgt_spec_c = DTensorSpec(
                tgt_spec.mesh, tgt_placements, tensor_meta=tgt_spec.tensor_meta
            )
            x = my_redistribute_local_tensor(arg, curr_spec, tgt_spec_c)
        return x

    def redistribute_getitem_arg(self, arg, idx):
        node = self._curr_node
        assert node.target == operator.getitem, f"Got {node.target}"
        if not isinstance(arg, torch.Tensor):
            return arg

        # use this instead of node.all_input_nodes as it handles repeated nodes
        all_input_nodes = [
            x for x in tree_flatten(node.args)[0] if isinstance(x, torch.fx.Node)
        ]
        curr_spec = self.sharding_placement[all_input_nodes[0]].output_specs[idx]
        tgt_spec = self.sharding_placement[node].input_specs[0]

        x = self.redistribute_tensor(arg, curr_spec, tgt_spec)
        self.tgt_spec = tgt_spec
        return x

    def redistribute_args(self, args):
        node = self._curr_node
        # use this instead of node.all_input_nodes as it handles repeated nodes
        all_input_nodes = [
            x for x in tree_flatten(node.args)[0] if isinstance(x, torch.fx.Node)
        ]
        num_input_nodes = len(all_input_nodes)
        curr_specs = [
            self.sharding_placement[n].output_specs for n in all_input_nodes
        ]  # FIXME ?
        tgt_specs = [
            self.sharding_placement[node].input_specs[c] for c in range(num_input_nodes)
        ]

        flat_args, treespec = tree_flatten(args)
        flat_args_t = [x for x in flat_args if isinstance(x, torch.Tensor)]
        assert len(flat_args_t) == len(curr_specs) == len(tgt_specs)
        new_flat_args_t = []
        for arg, curr_spec, tgt_spec in zip(flat_args_t, curr_specs, tgt_specs):
            x = self.redistribute_tensor(arg, curr_spec, tgt_spec)
            new_flat_args_t.append(x)
            self.tgt_spec = tgt_spec
        new_flat_args = []
        counter = 0
        for x in flat_args:
            if isinstance(x, torch.Tensor):
                x = new_flat_args_t[counter]
                counter += 1
            new_flat_args.append(x)

        new_args = treespec.unflatten(new_flat_args)
        return list(new_args)

    def call_function(self, target, args, kwargs):
        new_args = []
        node = self._curr_node

        # TODO: fix getitem propagation to have as input all the tensors
        # this will require fixing other things down the line
        # and might require removing the workaround None -> Replicate from
        # DTensor sharding ops
        if node.target == operator.getitem:
            new_args.append(list(args[0]))
            new_args[0][args[1]] = self.redistribute_getitem_arg(
                new_args[0][args[1]], args[1]
            )
            new_args[0] = tuple(new_args[0])
            new_args.append(args[1])
        else:
            new_args = self.redistribute_args(args)

        # apply sharding to constructor functions as well
        if target == torch.ops.aten.full.default:
            val = list(new_args[0])
            spec = self.sharding_placement[node].output_specs
            for mesh_size, placement in zip(spec.mesh.shape, spec.placements):
                if placement.is_shard():
                    # TODO: fix uneven cases ?
                    val[placement.dim] //= mesh_size
            new_args[0] = tuple(val)

        # use DTensor machinery to ensure the view ops are valid
        # otherwise we would end-up forcing global shapes on local tensors
        # which would yield errors
        if target in {
            torch.ops.aten._unsafe_view.default,
            torch.ops.aten.view.default,
            torch.ops.aten.expand.default,
        }:
            new_args[0] = DTensor.from_local(
                new_args[0], self.tgt_spec.mesh, self.tgt_spec.placements
            )
            # TODO: see if we can remove this contiguous properly
            new_args[0] = new_args[0].contiguous()

        out = super().call_function(target, tuple(new_args), kwargs)
        out = tree_map_only(DTensor, lambda x: x.to_local(), out)
        return out


def shard_node_given_placements(node, sharding_placement, *, meta: bool):
    # TODO: not sure if we actually guarantee sharding_placement has ever
    # input node lol
    tgt_spec = sharding_placement[node].input_specs[0]
    mesh = tgt_spec.mesh
    # all tensors start as replicated
    curr_placement = (Replicate(),) * mesh.ndim
    tensor = node.meta["val"]

    if meta:
        assert isinstance(
            tensor, FakeTensor
        ), f"only FakeTensor params supported for now, got {type(tensor)}"
        ctx = unset_fake_temporarily
        with ctx():
            tensor = torch.randn(tensor.shape, dtype=tensor.dtype, device="meta")
    else:
        ctx = contextlib.nullcontext

    with ctx():
        sharded_tensor = DTensor.from_local(tensor, mesh, curr_placement).redistribute(
            mesh, tgt_spec.placements
        )

    return sharded_tensor


def shard_nodes_given_placements(gm, sharding_placement):
    nodes = [x for x in gm.graph.find_nodes(op="placeholder")]
    sharded_tensors = []
    for node in nodes:
        sharded_tensors.append(
            shard_node_given_placements(node, sharding_placement, meta=False)
        )
    return sharded_tensors


def apply_sharding_to_model(gm, sharding_placement, params_spec, buffers_spec):
    args = shard_nodes_given_placements(gm, sharding_placement)

    # run with DTensor to apply the collectives given the graph
    interp = ApplyShardingInterpreter(gm, sharding_placement)

    args = [x.to_local() for x in args]

    # TODO: make_fx here is suspicious in case of dynamic shapes
    parallel_gm = make_fx(interp.run)(*args)

    # TODO: tlparse this
    parallel_gm.print_readable(expanded_def=True)

    # Copy descriptors over to new graph
    for n1, n2 in zip((n for n in gm.graph.nodes if n.op in ('placeholder', 'output')), (n for n in parallel_gm.graph.nodes if n.op in ('placeholder', 'output'))):
        n2.meta['desc'] = n1.meta['desc']
        if n2.op == 'placeholder':
            n2.target = n1.target
            # TODO: would be nice to also do name as well

    sharded_param_dict = {}
    sharded_buffer_dict = {}

    # NB: ok to NOT use the parallel_gm here because we will just reapply the
    # correct sharding placement via sharding_placement
    fqn_to_param = named_param_nodes(gm.graph)
    fqn_to_buffer = named_buffer_nodes(gm.graph)

    for fqn in params_spec:
        n = fqn_to_param[fqn]
        with unset_fake_temporarily():
            sharded_param_dict[fqn] = nn.Parameter(
                shard_node_given_placements(n, sharding_placement, meta=True)
            )

    for fqn in buffers_spec:
        n = fqn_to_buffer[fqn]
        sharded_buffer_dict[fqn] = shard_node_given_placements(
            n, sharding_placement, meta=True
        )

    return parallel_gm, sharded_param_dict, sharded_buffer_dict
