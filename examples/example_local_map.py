# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.distributed._tensor.experimental import local_map
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch._ops import HigherOrderOperator
import functools

from autoparallel.api import AutoParallel

from torch.fx.experimental.proxy_tensor import (
    get_proxy_slot, 
    ProxyTorchDispatchMode,
    disable_proxy_modes_tracing,
    track_tensor_tree
)
import torch.utils._pytree as pytree


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

class LocalMapAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, runtime_func, *args, **kwargs):
        ctx.save_for_backward(*args)

        with torch._C._AutoDispatchBelowAutograd():  # why
            return local_map_hop(runtime_func, *args, **kwargs)
        #     out = runtime_func(*args, **kwargs)
        #     return out
        # out = runtime_func(*args, **kwargs)
        # breakpoint()
        # return out

    @staticmethod
    def backward(ctx, *grads):
        # mmmmm could really use the backward graph here
        fwd_inputs = ctx.saved_tensors
        return None, *[torch.ones_like(i) * 12345 for i in fwd_inputs]

@local_map_hop.py_impl(torch._C.DispatchKey.Autograd)
def autograd_key(
    runtime_func,
    *args,
    **kwargs,
):
    return LocalMapAutogradOp.apply(runtime_func, *args, **kwargs)

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
    assert proxy_mode is not None, "Mode should always be enabled for python fallback key"
    assert len(kwargs) == 0

    example_out = local_map_hop(runtime_func, *args, **kwargs)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)
    def another_wrapper(*another_args, **another_kwargs):
        return functools.partial(local_map_hop, runtime_func)(*another_args, **another_kwargs)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", another_wrapper, proxy_args, {}
    )
    out_proxy.node.meta["custom"] = {
        "local_map_kwargs": runtime_func.local_map_kwargs,
    }
    return track_tensor_tree(
        example_out, out_proxy, constant=None, tracer=proxy_mode.tracer
    )


from typing import Callable, Optional

from torch.distributed.tensor import DeviceMesh, DTensor, Placement, Replicate
from torch.distributed.tensor.experimental._func_map import InputPlacements, OutputPlacements
from torch.distributed.device_mesh import _mesh_resources

def apply_local_map(*local_map_args, **local_map_kwargs):
    assert local_map_kwargs["redistribute_inputs"], "Autoparallel should always be allowed to redistribute inputs"

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
    out_placements=[Replicate(),],
    in_placements=([Replicate()], [Replicate()]),  # intentionally suboptimal, just to test
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
