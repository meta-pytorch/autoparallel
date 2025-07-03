# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import copy
from collections import OrderedDict
from contextlib import suppress

import torch
import torch.utils._pytree as pytree
from torch._functorch.partitioners import default_partition
from torch._inductor.decomposition import select_decomp_table
from torch._inductor.fx_passes.joint_graph import joint_graph_passes
from torch._inductor.fx_passes.post_grad import remove_assert_ops
from torch._logging import trace_structured
from torch._subclasses import FakeTensorMode
from torch.distributed.tensor import DeviceMesh

from .apply_sharding import apply_sharding_to_model
from .export_module import aot_export_module, apply_node_renaming
from .optimize_sharding import ShardingOptimizer
from .utils import _get_device_from_mesh


def _add_alias(gm):
    """
    Helper function to add alias nodes to every node in the graph
    this gives more configuration opportunities
    """
    graph = gm.graph

    nodes = [n for n in graph.nodes if n.op == "call_function"]
    node_map = {node: idx for idx, node in enumerate(nodes)}
    inputs = graph.find_nodes(op="placeholder")
    for node in inputs:
        if len(node.users) == 0:
            # node is not used, don't add alias for it
            continue
        first_user = nodes[min(node_map[n] for n in node.users)]
        first_user = inputs[-1].next
        with graph.inserting_before(first_user):
            alias_node = graph.call_function(torch.ops.aten.alias.default, args=(node,))
            alias_node.meta.update(node.meta)

            def delete_user_cb(n):
                return n != alias_node

            node.replace_all_uses_with(alias_node, delete_user_cb=delete_user_cb)

    """
    for node in nodes:
        # skip ops which return tuple
        if not isinstance(node.meta["val"], torch.Tensor):
            continue
        with graph.inserting_after(node):
            alias_node = graph.call_function(torch.ops.aten.alias.default, args=(node,))
            alias_node.meta.update(node.meta)

            def delete_user_cb(n):
                return n != alias_node

            node.replace_all_uses_with(alias_node, delete_user_cb=delete_user_cb)

    """

    for node in graph.find_nodes(op="output")[0].all_input_nodes:
        with graph.inserting_after(node):
            alias_node = graph.call_function(torch.ops.aten.alias.default, args=(node,))
            alias_node.meta.update(node.meta)

            def delete_user_cb(n):
                return n != alias_node

            node.replace_all_uses_with(alias_node, delete_user_cb=delete_user_cb)

    gm.recompile()
    return gm


def try_convert_fake_to_real(tensors):
    out = {}
    for k, t in tensors.items():
        out[k] = torch.distributed.tensor.randn(
            t.shape, dtype=t.dtype, device_mesh=t.device_mesh, placements=t.placements
        )
    return out


class BufferDict(torch.nn.Module):
    def __init__(self, buffers):
        super().__init__()
        self._keys = {}
        for name, buffer in buffers.items():
            persistent = True  # TODO: fixme
            self.register_buffer(name, buffer, persistent=persistent)
            self._keys[name] = None

    def values(self):
        return (getattr(self, k) for k in self._keys)

    def extra_repr(self):
        lines = []
        for k in self._keys:
            b = getattr(self, k)
            size_str = "x".join(str(size) for size in b.size())
            device_str = f" ({b.device})"
            lines.append(
                f"({k}): Buffer containing [{torch.typename(b)} of size {size_str} {device_str}]"
            )
        return "\n".join(lines)


def prepare_module(parallel_gm, spec, num_fwd_outputs):
    """
    This function takes the parallelized joint graph and splits it in
    fwd + bwd, wraps it in an autograd.Function and returns a nn.Module that perform
    the computation
    TODO: need to let the user specify the weight initialization
    """
    # TODO: this should be present elsewhere in the stack, it's a hack for
    # properly splitting fwd/bwd. This seems to be an issue with aot_export_module
    # TODO: This doesn't seem needed anymore?
    # for users in parallel_gm.graph.find_nodes(op="output")[0].all_input_nodes[0].users:
    #     users.meta["partitioner_tag"] = "must_be_in_backward"

    # let's remove those otherwise we can't clean the backward graph properly
    with suppress(KeyError):
        torch.fx.node._side_effectful_functions.remove(
            torch.ops._c10d_functional.wait_tensor
        )
    with suppress(KeyError):
        torch.fx.node._side_effectful_functions.remove(
            torch.ops._c10d_functional.wait_tensor.default
        )
    fwd_gm, bwd_gm = default_partition(
        parallel_gm, None, num_fwd_outputs=num_fwd_outputs
    )

    class AutoParallelFunc(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            out = fwd_gm(*args)
            ctx.num_inputs = len(args)
            ctx.save_for_backward(*out[num_fwd_outputs:])
            ctx.set_materialize_grads(False)
            out = out[:num_fwd_outputs]
            return pytree.tree_unflatten(out, spec)

        @staticmethod
        def backward(ctx, *grad):
            flat_grad, _ = pytree.tree_flatten(grad)
            saved_tensors = ctx.saved_tensors
            # remove tensors that don't need gradient
            flat_grad = [x for x in flat_grad if x is not None]
            grads = bwd_gm(*(saved_tensors + tuple(flat_grad)))
            # TODO: handle buffers
            return grads + (None,) * ctx.num_inputs

    class AutoParallelModule(torch.nn.Module):
        def __init__(self, parameters, buffers):
            super().__init__()
            # need to use OrderedDict due to constraints from nn.ParameterDict
            # on ordering
            self.params = torch.nn.ParameterDict(OrderedDict(parameters))
            self.buffers_ = BufferDict(buffers)

        def forward(self, *x):
            x = pytree.tree_flatten(x)[0]
            params = [p.to_local() for p in self.params.values()]
            buffers = [b.to_local() for b in self.buffers_.values()]
            out = AutoParallelFunc.apply(*(params + buffers + list(x)))
            return out

    return AutoParallelModule, fwd_gm, bwd_gm


def _get_decomp_table():
    decomp_table = copy.copy(select_decomp_table())
    # TODO: removing those as they cause missing DTensor propagation rules
    decomp_table.pop(torch.ops.aten.full_like.default)
    decomp_table.pop(torch.ops.aten.empty_like.default)
    decomp_table.pop(torch.ops.aten.threshold_backward.default)
    decomp_table.pop(torch.ops.aten.native_layer_norm.default)
    decomp_table.pop(torch.ops.aten.embedding_dense_backward.default)
    decomp_table.pop(torch.ops.aten.native_layer_norm_backward.default)

    # decompose addmm to allow for TP on mm
    decomp_table.pop(torch.ops.aten.addmm.default)

    def addmm_decomp(self, mat1, mat2, beta=1, alpha=1):
        return self + mat1 @ mat2

    decomp_table[torch.ops.aten.addmm.default] = addmm_decomp
    # decomp_table = None

    return decomp_table


def get_graph_clusters(gm):
    """
    Super hacky, only works for the case where we have a single sdpa per block
    Don't handle backward, just a POC to see if the solver will have speedups
    with those constraints added
    """
    the_ops = [
        torch.ops.aten._scaled_dot_product_efficient_attention.default,
        torch.ops.aten._scaled_dot_product_efficient_attention_backward.default,
    ]
    clusters = []
    for op in the_ops:
        clusters += get_graph_clusters_one_op(gm, op)
    return clusters


def get_graph_clusters_one_op(gm, the_op):
    from collections import defaultdict

    picked_group = [
        n for n in gm.graph.nodes if n.op == "call_function" and n.target == the_op
    ]

    node_map = {n: i for i, n in enumerate(gm.graph.nodes)}
    node_locs = [node_map[n] for n in picked_group]
    diff = node_locs[1] - node_locs[0]
    assert all(
        diff == (node_locs[i + 1] - node_locs[i]) for i in range(len(picked_group) - 1)
    ), "Very basic graph clustering for now, this model isn't handled by it"

    g = [[p] for p in picked_group]

    def _get_node_hash(node):
        if "val" not in node.meta:
            return (node.op,)
        if isinstance(node.meta["val"], torch.Tensor):
            shape = node.meta["val"].shape
        else:
            shape = tuple(
                x.shape for x in node.meta["val"] if isinstance(x, torch.Tensor)
            )
        return (str(node.target), shape)

    def get_sequence_hash(seq):
        return hash(tuple(_get_node_hash(x) for x in seq))

    next_elm = True
    for i in range(diff):
        if next_elm:
            new_seq = [p + [p[-1].next] if p[-1].op != "output" else p for p in g]
        else:
            new_seq = [[p[0].prev] + p if p[0].op != "placeholder" else p for p in g]

        def get_elm(s):
            if next_elm:
                return s[-1]
            return s[0]

        hashes = defaultdict(list)
        for i, s in enumerate(new_seq):
            hashes[get_sequence_hash(s)].append((i, get_elm(s)))

        if len(hashes) != 1:
            next_elm = False
            continue

        for cls in hashes.values():
            for cl_id, cl in cls:
                if next_elm:
                    g[cl_id].append(cl)
                else:
                    g[cl_id].insert(0, cl)

    clusters = defaultdict(list)
    for cl in g:
        clusters[get_sequence_hash(cl)].append(cl)

    clusters = [v for v in clusters.values() if len(v) > 1]
    assert (
        len(clusters) == 1
    ), "Very basic graph clustering, this model isn't handled by it"
    return clusters


def move_to_fake(model: torch.nn.Module, mode: FakeTensorMode, device: torch.device):
    """
    Move the model to the fake mode and move the weights to the fake device
    """

    def assert_is_meta_tensor(name, t):
        assert isinstance(t, torch.Tensor) and t.device == torch.device(
            "meta"
        ), f"tensor {name} must be on meta device, not {t.device}"

    def _move_to_fake(module, k, device, parameter=True):
        # lots of ways you might try to swap params with fake params do not work, but this one does
        submod = module
        while len(k.split(".")) > 1:
            submod_name, k = k.split(".", 1)
            submod = getattr(submod, submod_name)

        fake_tensor = mode.from_tensor(getattr(submod, k)).to(device)
        if parameter:
            fake_tensor = torch.nn.Parameter(fake_tensor)

        setattr(submod, k, fake_tensor)

    with mode:
        for k, p in model.named_parameters():
            assert_is_meta_tensor(k, p)
            _move_to_fake(model, k, device, parameter=True)
        for k, b in model.named_buffers():
            assert_is_meta_tensor(k, b)
            _move_to_fake(model, k, device, parameter=False)

    return model


class AutoParallel:
    """
    Args:
        mesh: Defines placement options.
        The meta model is moved to a fake device based on mesh.device_type.
    """

    def __init__(self, model, input_fn, mesh: DeviceMesh, **kwargs):
        self.fake_mode = FakeTensorMode()
        device = _get_device_from_mesh(mesh)
        self.model = move_to_fake(model, self.fake_mode, device)
        self.input_fn = input_fn
        self.mesh = mesh
        self.build_model_graph()

        clusters = None
        if kwargs.get("repeated_subgraphs", False):
            clusters = get_graph_clusters(self.gm)
        sharding_optimizer = ShardingOptimizer(self.gm, self.mesh, clusters)
        # makes sharding of params and gradients the same
        sharding_optimizer.add_grad_param_constraints()
        self.sharding_optimizer = sharding_optimizer

        self.input_constraints = None
        self.output_constraints = None

    def build_model_graph(self):
        decomp_table = _get_decomp_table()
        # needed because of https://github.com/pytorch/pytorch/issues/148977
        torch.__future__.set_swap_module_params_on_conversion(True)
        with self.fake_mode:
            inputs = self.input_fn()
            if not isinstance(inputs, tuple):
                inputs = (inputs,)

            (
                gm,
                self.spec,
                self.params_len,
                self.buffer_len,
                self.metadata,
            ) = aot_export_module(
                self.model, inputs, decompositions=decomp_table, trace_joint=True
            )

        # cleanup graph
        gm.graph.eliminate_dead_code()
        gm.recompile()
        # disable pattern_matcher as it gets on our way
        # we basically want to remove noops in here
        prev = torch._inductor.config.pattern_matcher
        torch._inductor.config.pattern_matcher = False
        gm = joint_graph_passes(gm)
        torch._inductor.config.pattern_matcher = prev
        remove_assert_ops(gm.graph)
        gm.graph.eliminate_dead_code()
        gm.recompile()
        # now add aliases nodes to the graph to
        # give more room for optimizations
        _add_alias(gm)
        apply_node_renaming(gm, self.params_len, self.buffer_len, self.metadata)
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "autoparallel_joint_graph",
                "encoding": "string",
            },
            payload_fn=lambda: str(gm.graph),
        )

        self.gm = gm

    def add_parameter_memory_constraint(self, low=None, high=None):
        # by default, divide the parameters by the world size
        if low is None:
            low = 0.0
        if high is None:
            high = 1.0 / self.mesh.size()

        assert low <= high, f"low should be <= high, got low{low}, high={high}"

        self.sharding_optimizer.add_parameter_memory_constraint(low, high)

    def add_input_constraints(self, constraints):
        assert self.input_constraints is None, "Input constraints have already been set"
        self.sharding_optimizer.add_sharded_input_constraint(constraints)
        self.input_constraints = constraints

    def add_output_constraints(self, constraints):
        assert (
            self.output_constraints is None
        ), "Output constraints have already been set"
        # forces sharding of fwd output to be S(0) on first dimension and R on others
        self.sharding_optimizer.add_sharded_output_constraint(constraints)
        self.output_constraints = constraints

    def optimize_placement(self, verbose=True):
        if self.input_constraints is None:
            # forces sharding of input to be S(0) on first dimension and R on others
            self.add_input_constraints(
                [None] * len(self.sharding_optimizer.get_input_nodes())
            )

        if self.output_constraints is None:
            # forces sharding of fwd output to be S(0) on first dimension and R on others
            self.add_output_constraints(
                [None] * len(self.sharding_optimizer.get_fn_output_nodes())
            )

        self.sharding_placement = self.sharding_optimizer.get_solution(verbose=False)

        if verbose:
            print(self.sharding_optimizer.get_log())

        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "autoparallel_sharding_optimizer_log",
                "encoding": "string",
            },
            payload_fn=lambda: self.sharding_optimizer.get_log(colored=False),
        )

        if self.sharding_optimizer.prob.status == -1:
            raise RuntimeError("Didn't find solution")

        return self.sharding_placement

    def apply_placement(self, sharding_placement=None):
        if sharding_placement is None:
            sharding_placement = self.sharding_placement
        with self.fake_mode:
            parallel_gm, sharded_weights, sharded_buffers = apply_sharding_to_model(
                self.gm, sharding_placement
            )
        # clean it up by removing the added aliases from previous pass
        # as well as redundant views
        parallel_gm = joint_graph_passes(parallel_gm)
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "autoparallel_parallel_graph",
                "encoding": "string",
            },
            payload_fn=lambda: str(parallel_gm.graph),
        )
        # now rename input/param/tangent/output/grad_param/grad_input nodes following
        # our convention
        apply_node_renaming(
            parallel_gm, self.params_len, self.buffer_len, self.metadata
        )
        self.parallel_gm = parallel_gm

        param_names = [k.replace(".", "/") for k, _ in self.model.named_parameters()]
        buffer_names = [k.replace(".", "/") for k, _ in self.model.named_buffers()]
        assert len(param_names) == len(sharded_weights)
        assert len(buffer_names) == len(sharded_buffers)
        sharded_weights = {k: v for k, v in zip(param_names, sharded_weights)}
        sharded_buffers = {k: v for k, v in zip(buffer_names, sharded_buffers)}

        self.sharded_weights = sharded_weights
        self.sharded_buffers = sharded_buffers
        self.parallel_model_fn, self.fwd_gm, self.bwd_gm = prepare_module(
            parallel_gm, self.spec, self.metadata.num_outputs
        )

        sharded_weights = try_convert_fake_to_real(sharded_weights)
        sharded_buffers = try_convert_fake_to_real(sharded_buffers)
        self.parallel_model = self.parallel_model_fn(sharded_weights, sharded_buffers)

        return self.parallel_model
