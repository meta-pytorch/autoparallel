# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: this file may be removed once we move to a dynamo frontend

import functools

import torch
import torch.utils._pytree as pytree
from torch._higher_order_ops.utils import (
    save_tensors_and_symints_for_backward,
    saved_tensors_and_symints,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tensor.experimental import local_map
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree


class LocalMapAOTExportModule(HigherOrderOperator):
    """
    A HOP that integrates with autoparallel's current frontend (aot_export_module).
    This HOP exists starting the pre-solver graph and lives until we apply sharding.
    During which, orig_fwd will be inlined into the post-solver graph.
    """

    def __init__(self):
        super().__init__("local_map_hop")

    def __call__(self, orig_fwd, *args, **kwargs):
        return super().__call__(orig_fwd, *args, **kwargs)


local_map_hop = LocalMapAOTExportModule()


def create_hop_joint_graph(
    fw_func,
    *_args,
):
    # Keeping these imports here
    # Avoid circular dependencies once we upstream with dynamo frontend
    from torch._dispatch.python import suspend_functionalization
    from torch._functorch.aot_autograd import AOTConfig, create_joint
    from torch._guards import detect_fake_mode
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
    from torch._subclasses.functional_tensor import disable_functional_mode
    from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing, make_fx

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

            # create a tensor (fake) from a compiler wrapped FunctionalTensor
            def _from_fun(t):
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
            example_grads = _from_fun(example_flat_out)
            if not isinstance(example_grads, (list, tuple)):
                example_grads = [example_grads]

        def joint_f(
            *primals_and_tangents,
        ):
            fw_inputs = primals_and_tangents[: len(_args)]
            example_grads = primals_and_tangents[len(_args) :]

            def run_fwd(*fw_inputs):
                outs = fw_func(*fw_inputs)
                if not isinstance(outs, (list, tuple)):
                    outs = (outs,)
                masks = [o.requires_grad for o in outs]
                return (outs, masks)

            joint = create_joint(run_fwd, aot_config=dummy_aot_config)
            optional_grads = []
            for example_grad in example_grads:
                if example_grad.requires_grad:
                    optional_grads.append(example_grad)
            _, grads = joint(fw_inputs, optional_grads)
            return grads

        primals_and_tangents = [*fw_inputs, *example_grads]
        joint_graph = make_fx(joint_f)(*primals_and_tangents)
        return None, joint_graph


class LocalMapAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, orig_fwd, joint_graph, *args, **kwargs):
        ctx.save_for_backward(*args)

        save_tensors_and_symints_for_backward(ctx, args)
        ctx.joint_graph = joint_graph

        with torch._C._AutoDispatchBelowAutograd():
            return local_map_hop(orig_fwd, *args, **kwargs)

    @staticmethod
    def backward(ctx, *grads):
        args = saved_tensors_and_symints(ctx)
        grad_ins = ctx.joint_graph(*args, *grads)
        # TODO: hopify to support local_map'd function containing custom autograd.Function
        return None, None, *grad_ins


@local_map_hop.py_impl(torch._C.DispatchKey.Autograd)
def autograd_key(
    orig_fwd,
    *args,
    **kwargs,
):
    if "_inline" in kwargs:
        # Solver pass adds a _inline kwarg, which tells this hop to desugar on the next trace
        del kwargs["_inline"]
        return orig_fwd(*args, **kwargs)

    _, joint_graph = create_hop_joint_graph(orig_fwd, *args)
    return LocalMapAutogradOp.apply(orig_fwd, joint_graph, *args, **kwargs)


@local_map_hop.py_functionalize_impl
def functional_mode_key(ctx, orig_fwd, *args, **kwargs):
    assert not kwargs

    unwrapped_inputs = ctx.unwrap_tensors(args)
    with ctx.redispatch_to_next():
        # TODO: dynamo safety checks on orig_fwd
        out = local_map_hop(orig_fwd, *unwrapped_inputs)
        return ctx.wrap_tensors(out)


@local_map_hop.py_impl(FakeTensorMode)
def fake_mode_key(
    mode,
    orig_fwd,
    *args,
    **kwargs,
):
    with mode:
        return orig_fwd(*args, **kwargs)


@local_map_hop.py_impl(ProxyTorchDispatchMode)
def proxy_mode_key(
    proxy_mode,
    orig_fwd,
    *args,
    **kwargs,
):
    assert (
        proxy_mode is not None
    ), "Mode should always be enabled for python fallback key"
    assert len(kwargs) == 0

    example_out = local_map_hop(orig_fwd, *args, **kwargs)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)

    def call_local_map(*another_args, **another_kwargs):
        return functools.partial(local_map_hop, orig_fwd)(
            *another_args, **another_kwargs
        )

    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", call_local_map, proxy_args, {}
    )
    out_proxy.node.meta["custom"] = {
        "dtensor_local_map_kwargs": orig_fwd.local_map_kwargs,
    }
    return track_tensor_tree(
        example_out, out_proxy, constant=None, tracer=proxy_mode.tracer
    )


# Running HOP in eager with real tensors
@local_map_hop.py_impl(torch._C.DispatchKey.CPU)
@local_map_hop.py_impl(torch._C.DispatchKey.CUDA)
def real_impl(
    orig_fwd,
    *args,
    **kwargs,
):
    return orig_fwd(*args, **kwargs)


def apply_local_map(*local_map_args, **local_map_kwargs):
    # NOTE: We manually issue the hop, which will not be not necessary with a dynamo frontend.
    # 1. Same as local_map, must be applied on a function, not a method.
    # 2. the local_map'd function must be make_fx traceable. Otherwise, we may
    # inline the wrong graph. In a dynamo frontend, speculate_subgraph will handle this.
    # 3. All inputs to the local_map'd function must be Tensor types. Otherwise, we won't
    # know which tensors to apply _from_fun to. For instance, don't pass nn.Modules to local_map.
    # In dynamo frontend, tensors will be lifted, and will modify the wrapped function's signature.

    assert local_map_kwargs[
        "redistribute_inputs"
    ], "Autoparallel should always be allowed to redistribute inputs"
    assert local_map_kwargs["in_grad_placements"] is None, "Not yet implemented"
    assert local_map_kwargs["device_mesh"] is None, "Must be provided by Autoparallel"

    def decorator(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            def orig_fwd(*runtime_args, **runtime_kwargs):
                # wrap the functools.partial for hop utils to work out of box
                return local_map(
                    fn,
                    *local_map_args,
                    **local_map_kwargs,
                )(*runtime_args, **runtime_kwargs)

            orig_fwd.local_map_kwargs = local_map_kwargs
            return local_map_hop(orig_fwd, *args, **kwargs)

        return wrapped

    return decorator
