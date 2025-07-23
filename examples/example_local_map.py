# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Callable, Optional, Union

import torch
import torch.utils._pytree as pytree
from torch import Tensor, nn
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.distributed._tensor.experimental import local_map
from torch.distributed.device_mesh import _mesh_resources
from torch.distributed.tensor import DeviceMesh, DTensor, Placement, Replicate
from torch.distributed.tensor.experimental._func_map import (
    InputPlacements,
    OutputPlacements,
)
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.fx.experimental.proxy_tensor import (
    ProxyTorchDispatchMode,
    disable_proxy_modes_tracing,
    get_proxy_slot,
    track_tensor_tree,
)
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel.api import AutoParallel

# just to dump tlparse
torch.compile(lambda x: x + 1, backend="eager")(torch.rand(10))


class LocalMapAOTExportModule(HigherOrderOperator):
    """
    A HOP that integrates with autoparallel's current frontend (aot_export_module).
    This HOP exists starting the pre-solver graph and lives until we apply sharding.
    During which, runtime_func will be inlined into the post-solver graph.
    """

    def __init__(self):
        super().__init__("local_map_hop")

    def __call__(self, runtime_func, *args, **kwargs):
        return super().__call__(runtime_func, *args, **kwargs)


local_map_hop = LocalMapAOTExportModule()

# def fn(x):
#     return x

# local_map_hop(fn, torch.randn(10, 10))
# breakpoint()

from torch._higher_order_ops.utils import (
    _maybe_run_with_interpreter,
    _set_compilation_env,
    materialize_as_graph,
    reenter_make_fx,
    save_tensors_and_symints_for_backward,
    saved_tensors_and_symints,
    unique_graph_id,
    validate_subgraph_args_types,
)


def create_fw_bw_graph(
    fw_func,
    *_args,
):
    # See Note:[HOP create fw_bw graph]

    # All of these imports need to be here in order to avoid circular dependencies
    from torch._dispatch.python import suspend_functionalization
    from torch._functorch.aot_autograd import AOTConfig, create_joint
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
    from torch._subclasses.functional_tensor import disable_functional_mode
    from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing

    dummy_aot_config = AOTConfig(
        fw_compiler=None,  # type: ignore[arg-type]
        bw_compiler=None,  # type: ignore[arg-type]
        partition_fn=None,  # type: ignore[arg-type]
        decompositions={},
        num_params_buffers=0,
        aot_id=0,
        keep_inference_input_mutations=False,
    )

    with suspend_functionalization(), disable_functional_mode():
        with disable_proxy_modes_tracing():

            def _from_fun(
                t: Union[Tensor, torch.SymInt, int],
            ) -> Union[Tensor, torch.SymInt, int]:
                if isinstance(t, torch.Tensor):
                    return torch.empty_strided(
                        t.size(),
                        t.stride(),
                        device=t.device,
                        dtype=t.dtype,
                        requires_grad=t.requires_grad,
                    )
                return t

            # If someone runs this hop under the default compiler backend ("eager")
            # Then this path will be run with the actual user inputs. We convert them
            # to fake tensors in order to not perform any actual compute.
            from torch._guards import detect_fake_mode

            fake_mode = detect_fake_mode(_args)
            if fake_mode is None:
                fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

            with fake_mode:
                fw_inputs = pytree.tree_map(_from_fun, _args)

            assert all(
                isinstance(t, (FakeTensor, int, torch.SymInt)) for t in fw_inputs
            )

            # redundant? we already _from_fun'd the inputs
            example_flat_out = pytree.tree_map(
                _from_fun,
                fw_func(*fw_inputs),
            )
            example_grad = _from_fun(example_flat_out)

        from torch.fx.experimental.proxy_tensor import make_fx

        def joint_f(
            example_grad,
            *fw_inputs,
        ):
            def run_fwd(*fw_inputs):
                outs = fw_func(*fw_inputs)
                if not isinstance(outs, (list, tuple)):
                    outs = (outs,)
                masks = [o.requires_grad for o in outs]
                return (outs, masks)

            joint = create_joint(run_fwd, aot_config=dummy_aot_config)
            optional_grad = [example_grad] if example_grad.requires_grad else []
            _, grads = joint(fw_inputs, optional_grad)

            return grads

        joint_graph = make_fx(joint_f)(example_grad, *fw_inputs)
        # do i need to return fw_graph here? by definition it is traceable, so should be fine to run again with runtime_func
        return None, joint_graph


# def create_fw_bw_graph(
#     runtime_wrapper,
#     *args,
# ):
#     from torch._dispatch.python import suspend_functionalization
#     from torch._functorch.aot_autograd import AOTConfig, create_joint
#     from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
#     from torch._subclasses.functional_tensor import disable_functional_mode
#     from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing
#     from typing import Union

#     with suspend_functionalization(), disable_functional_mode():
#         with disable_proxy_modes_tracing():
#             def _from_fun(
#                 t: Union[torch.Tensor, torch.SymInt, int],
#             ) -> Union[torch.Tensor, torch.SymInt, int]:
#                 if isinstance(t, torch.Tensor):
#                     return torch.empty_strided(
#                         t.size(),
#                         t.stride(),
#                         device=t.device,
#                         dtype=t.dtype,
#                         requires_grad=t.requires_grad,
#                     )
#                 return t


#             fw_inputs = pytree.tree_map(_from_fun, args)
#             assert all(
#                 isinstance(t, (FakeTensor, int, torch.SymInt))
#                 for t in fw_inputs
#             )

#             out = runtime_wrapper(*fw_inputs)
#             example_flat_out = pytree.tree_map(
#                 _from_fun,
#                 out,
#             )
#             example_grad = _from_fun(example_flat_out)
#             breakpoint()

#         return None


class LocalMapAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, runtime_func, bwd_func, *args, **kwargs):
        ctx.save_for_backward(*args)

        save_tensors_and_symints_for_backward(ctx, args)
        ctx._bwd_func = bwd_func

        with torch._C._AutoDispatchBelowAutograd():  # why
            return local_map_hop(runtime_func, *args, **kwargs)

    @staticmethod
    def backward(ctx, *grads):
        args = saved_tensors_and_symints(ctx)
        grad_ins = ctx._bwd_func(*grads, *args)
        # TODO: hopify to make opaque
        # grad_ins = local_map_backward_hop(ctx._bwd_func, *grads, *args)
        return None, None, *grad_ins


@local_map_hop.py_impl(torch._C.DispatchKey.Autograd)
def autograd_key(
    runtime_func,
    *args,
    **kwargs,
):
    if "_inline" in kwargs:
        del kwargs["_inline"]
        return runtime_func(*args, **kwargs)

    # else trace
    # trace joint, pass to .apply
    _, bw_graph = create_fw_bw_graph(runtime_func, *args)
    return LocalMapAutogradOp.apply(runtime_func, bw_graph, *args, **kwargs)


@local_map_hop.py_functionalize_impl
def functional_mode_key(ctx, runtime_func, *args, **kwargs):
    assert not kwargs

    unwrapped_inputs = ctx.unwrap_tensors(args)
    with ctx.redispatch_to_next():
        # TODO: local_map mutation checks
        out = local_map_hop(runtime_func, *unwrapped_inputs)
        return ctx.wrap_tensors(out)


@local_map_hop.py_impl(FakeTensorMode)
def fake_mode_key(
    mode,
    runtime_func,
    *args,
    **kwargs,
):
    with mode:
        return runtime_func(*args, **kwargs)


@local_map_hop.py_impl(ProxyTorchDispatchMode)
def proxy_mode_key(
    proxy_mode,
    runtime_func,
    *args,
    **kwargs,
):
    assert (
        proxy_mode is not None
    ), "Mode should always be enabled for python fallback key"
    assert len(kwargs) == 0

    example_out = local_map_hop(runtime_func, *args, **kwargs)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)

    def another_wrapper(*another_args, **another_kwargs):
        return functools.partial(local_map_hop, runtime_func)(
            *another_args, **another_kwargs
        )

    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", another_wrapper, proxy_args, {}
    )
    out_proxy.node.meta["custom"] = {
        "local_map_kwargs": runtime_func.local_map_kwargs,
    }
    return track_tensor_tree(
        example_out, out_proxy, constant=None, tracer=proxy_mode.tracer
    )


def apply_local_map(*local_map_args, **local_map_kwargs):
    assert local_map_kwargs[
        "redistribute_inputs"
    ], "Autoparallel should always be allowed to redistribute inputs"

    # manually issue the hop, which will not be not necessary with a dynamo frontend
    def decorator(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            def runtime_func(*runtime_args, **runtime_kwargs):
                # hop doesn't like the functools.partial created by local_map
                return local_map(
                    fn,
                    *local_map_args,
                    **local_map_kwargs,
                )(*runtime_args, **runtime_kwargs)

            runtime_func.local_map_kwargs = local_map_kwargs
            return local_map_hop(runtime_func, *args, **kwargs)

        return wrapped

    return decorator


@apply_local_map(
    out_placements=[
        [Replicate(), Replicate()],
    ],
    in_placements=(
        [Replicate(), Replicate()],
        [Replicate(), Replicate()],
    ),  # intentionally suboptimal, just to test
    redistribute_inputs=True,
)
def boosted(w, x):
    return torch.matmul(x, w.t()) * 12345


class Block(nn.Module):
    def __init__(self, nheads, dim1, dim2):
        super().__init__()
        self.nheads = nheads
        bias = False
        self.wq = nn.Linear(dim1, dim1, bias=bias)
        self.wk = nn.Linear(dim1, dim1, bias=bias)
        self.wv = nn.Linear(dim1, dim1, bias=bias)
        self.wo = nn.Linear(dim1, dim1, bias=bias)
        self.w1 = nn.Linear(dim1, dim2, bias=bias)
        self.w2 = nn.Linear(dim2, dim1, bias=bias)

    def init_weights(self):
        for lin in [self.wq, self.wk, self.wv, self.wo, self.w1, self.w2]:
            torch.nn.init.normal_(lin.weight)
            if lin.bias is not None:
                torch.nn.init.normal_(lin.bias)

    def forward(self, x):
        q = self.wq(x)
        k = boosted(self.wk.weight, x)
        # k = self.wk(x)
        v = self.wv(x)

        q = q.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
        k = k.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
        v = v.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)

        o = nn.functional.scaled_dot_product_attention(q, k, v)
        o = o.permute(0, 2, 1, 3).flatten(-2)

        o = self.wo(o)

        o0 = o + x

        o = self.w1(o0)
        o = torch.nn.functional.relu(o)
        o = self.w2(o)

        o = o0 + o

        return o


world_size = 256

fake_store = FakeStore()
torch.distributed.init_process_group(
    "fake", store=fake_store, rank=0, world_size=world_size
)
# mesh = torch.distributed.device_mesh.init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
mesh = torch.distributed.device_mesh.init_device_mesh(
    "cuda",
    (world_size // 8, 8),
    mesh_dim_names=(
        "dp",
        "tp",
    ),
)

bs = 8 * mesh.shape[0]
seq_len = 256
nheads = 48
dim1 = 6144
dim2 = dim1 * 4


def input_fn():
    return torch.rand(bs, seq_len, dim1, device="cuda")


# parallelize the model
with torch.device("meta"):
    model = Block(nheads, dim1, dim2)
autop = AutoParallel(model, input_fn, mesh)
autop.add_parameter_memory_constraint(low=None, high=None)

x_sharding = (Shard(0), Replicate())

autop.add_input_constraints([x_sharding])
autop.add_output_constraints([x_sharding])

sharding_placement = autop.optimize_placement()

# AutoParallel produces a module with meta-DTensor parameters that need to be initialized
parallel_mod = autop.apply_placement(sharding_placement)
parallel_mod.to_empty(device="cuda")
parallel_mod.init_weights()

# now let's run it
x = (torch.rand(bs // mesh.shape[0], seq_len, dim1, device="cuda"),)
out = parallel_mod(*x)
out.backward(torch.randn_like(out))

print("All good!")
