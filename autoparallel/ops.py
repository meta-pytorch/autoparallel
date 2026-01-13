# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.distributed.tensor.experimental._context_parallel._attention import (
    _scaled_dot_product_ring_cudnn_attention,
    _scaled_dot_product_ring_cudnn_attention_backward,
    _scaled_dot_product_ring_efficient_attention,
    _scaled_dot_product_ring_efficient_attention_backward,
    _scaled_dot_product_ring_flash_attention,
    _scaled_dot_product_ring_flash_attention_backward,
)
from torch.distributed.tensor.placement_types import Shard
from torch.nn.attention import SDPBackend

from autoparallel.collectives import get_mesh_from_global, local_map


# Backend-specific backward wrappers to handle signature differences
def _flash_backward_wrapper(mesh, grad_out, q, k, v, out, forward_outputs, kwargs):
    """Handle flash attention backward with correct argument order."""
    # Forward outputs: lse, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, debug_attn_mask
    # Backward expects: mesh, grad_out, query, key, value, out, logsumexp, cum_seq_q, cum_seq_k,
    #                   max_q, max_k, dropout_p, is_causal, philox_seed, philox_offset, *, scale
    (
        lse,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        philox_seed,
        philox_offset,
    ) = forward_outputs[:7]
    return _scaled_dot_product_ring_flash_attention_backward(
        mesh,
        grad_out,
        q,
        k,
        v,
        out,
        lse,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        kwargs.get("dropout_p", 0.0),
        kwargs.get("is_causal", False),
        philox_seed,
        philox_offset,
        scale=kwargs.get("scale", None),
    )


def _efficient_backward_wrapper(mesh, grad_out, q, k, v, out, forward_outputs, kwargs):
    """Handle efficient attention backward with correct argument order."""
    # Forward outputs: lse, philox_seed, philox_offset
    # Backward expects: mesh, grad_out, query, key, value, bias, out, logsumexp,
    #                   philox_seed, philox_offset, dropout_p, grad_input_mask, is_causal, *, scale
    lse, philox_seed, philox_offset = forward_outputs[:3]
    # Build grad_input_mask based on which inputs require gradients
    attn_bias = kwargs.get("attn_bias", None)
    grad_input_mask = (
        q.requires_grad,
        k.requires_grad,
        v.requires_grad,
        attn_bias.requires_grad if attn_bias is not None else False,
    )
    return _scaled_dot_product_ring_efficient_attention_backward(
        mesh,
        grad_out,
        q,
        k,
        v,
        attn_bias,
        out,
        lse,
        philox_seed,
        philox_offset,
        kwargs.get("dropout_p", 0.0),
        grad_input_mask,
        kwargs.get("is_causal", False),
        scale=kwargs.get("scale", None),
    )


def _cudnn_backward_wrapper(mesh, grad_out, q, k, v, out, forward_outputs, kwargs):
    """Handle cudnn attention backward with correct argument order."""
    # Forward outputs: lse, philox_seed, philox_offset, softmax_stats(?), bias(?), cum_seq_q, cum_seq_k, max_q, max_k, debug_attn_mask
    # Backward expects: mesh, grad_out, query, key, value, out, logsumexp,
    #                   philox_seed, philox_offset, attn_bias, cum_seq_q, cum_seq_k,
    #                   max_q, max_k, dropout_p, is_causal, *, scale
    lse, philox_seed, philox_offset = forward_outputs[:3]
    # CuDNN may have additional outputs; extract what we need
    if len(forward_outputs) >= 9:
        cum_seq_q, cum_seq_k, max_q, max_k = forward_outputs[5:9]
    else:
        # Fallback if structure is different
        cum_seq_q, cum_seq_k, max_q, max_k = forward_outputs[3:7]

    return _scaled_dot_product_ring_cudnn_attention_backward(
        mesh,
        grad_out,
        q,
        k,
        v,
        out,
        lse,
        philox_seed,
        philox_offset,
        kwargs.get("attn_bias", None),
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        kwargs.get("dropout_p", 0.0),
        kwargs.get("is_causal", False),
        scale=kwargs.get("scale", None),
    )


# Mapping of backward functions to their wrappers
_CP_BACKWARD_WRAPPERS = {
    _scaled_dot_product_ring_flash_attention_backward: _flash_backward_wrapper,
    _scaled_dot_product_ring_efficient_attention_backward: _efficient_backward_wrapper,
    _scaled_dot_product_ring_cudnn_attention_backward: _cudnn_backward_wrapper,
}


class _ContextParallelAttention(torch.autograd.Function):
    """
    Generic context parallel attention that supports multiple backends.
    Uses **kwargs to be future-proof against signature changes.
    """

    @staticmethod
    def forward(ctx, op_forward, op_backward, q, k, v, kwargs_keys_str, *kwargs_values):
        """
        Args:
            op_forward: Forward operation (e.g., _scaled_dot_product_ring_flash_attention)
            op_backward: Backward operation (e.g., _scaled_dot_product_ring_flash_attention_backward)
            q, k, v: Query, key, value tensors
            kwargs_keys_str: Comma-separated string of kwarg names (e.g., 'dropout_p,is_causal,scale')
            *kwargs_values: Values corresponding to kwargs_keys
        """
        # Get mesh from global context (avoids passing it through local_map which would flatten it)
        mesh = get_mesh_from_global()["tp"]

        ctx.op_backward = op_backward
        ctx.mesh = mesh

        # Reconstruct kwargs dict from keys string and values
        kwargs_keys = kwargs_keys_str.split(",") if kwargs_keys_str else []
        kwargs_dict = dict(zip(kwargs_keys, kwargs_values))
        ctx.kwargs = kwargs_dict

        # Call the forward operation with all kwargs
        outputs = op_forward(mesh, q, k, v, **kwargs_dict)

        # outputs is a tuple: (out, lse, ...) where the rest varies by backend
        out = outputs[0]
        forward_outputs = outputs[1:]

        # Separate tensors from non-tensors for proper saving
        # Tensors must be saved via save_for_backward for proper memory management
        tensors_to_save = [q, k, v, out]
        non_tensor_outputs = []

        for i, item in enumerate(forward_outputs):
            if isinstance(item, torch.Tensor):
                tensors_to_save.append(item)
                non_tensor_outputs.append(("tensor", len(tensors_to_save) - 1))
            else:
                non_tensor_outputs.append(("value", item))

        ctx.save_for_backward(*tensors_to_save)
        ctx.non_tensor_outputs = non_tensor_outputs
        ctx.num_forward_outputs = len(forward_outputs)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        # Retrieve saved tensors
        saved_tensors = ctx.saved_tensors
        q, k, v, out = saved_tensors[:4]
        saved_forward_tensors = saved_tensors[4:]

        # Reconstruct forward_outputs from saved tensors and non-tensor values
        forward_outputs = []
        tensor_idx = 0
        for output_type, output_value in ctx.non_tensor_outputs:
            if output_type == "tensor":
                forward_outputs.append(saved_forward_tensors[tensor_idx])
                tensor_idx += 1
            else:
                forward_outputs.append(output_value)
        forward_outputs = tuple(forward_outputs)

        # Use the backend-specific wrapper to handle argument ordering
        wrapper_fn = _CP_BACKWARD_WRAPPERS.get(ctx.op_backward)
        if wrapper_fn is None:
            raise RuntimeError(
                f"No backward wrapper found for {ctx.op_backward}. "
                "This backend may not be supported yet."
            )

        grads = wrapper_fn(
            ctx.mesh,
            grad_out,
            q,
            k,
            v,
            out,
            forward_outputs,
            ctx.kwargs,
        )

        # Return gradients:
        # (None for op_forward, None for op_backward, grad_q, grad_k, grad_v, None for kwargs_keys_str, None for each kwargs_value)
        num_kwargs = len(ctx.kwargs)
        return (None, None) + grads[:3] + (None,) + (None,) * num_kwargs


# Backend registry for context parallel attention
_CP_ATTENTION_BACKENDS = {
    SDPBackend.FLASH_ATTENTION: (
        _scaled_dot_product_ring_flash_attention,
        _scaled_dot_product_ring_flash_attention_backward,
    ),
    SDPBackend.EFFICIENT_ATTENTION: (
        _scaled_dot_product_ring_efficient_attention,
        _scaled_dot_product_ring_efficient_attention_backward,
    ),
    SDPBackend.CUDNN_ATTENTION: (
        _scaled_dot_product_ring_cudnn_attention,
        _scaled_dot_product_ring_cudnn_attention_backward,
    ),
}


def context_parallel_attention(
    q, k, v, *, backend=SDPBackend.FLASH_ATTENTION, **kwargs
):
    """
    Generic context parallel attention supporting multiple backends.

    Args:
        q, k, v: Query, key, value tensors
        backend: SDPBackend to use (FLASH_ATTENTION, EFFICIENT_ATTENTION, or CUDNN_ATTENTION)
        **kwargs: Additional arguments passed to the attention operation (e.g., dropout_p, is_causal, scale, attn_bias)

    Returns:
        Attention output tensor

    This function is future-proof as it uses **kwargs to pass arguments, so changes
    to backend signatures won't require updating this function.
    """
    if backend not in _CP_ATTENTION_BACKENDS:
        raise ValueError(
            f"Unsupported backend: {backend}. Supported backends: {list(_CP_ATTENTION_BACKENDS.keys())}"
        )

    op_forward, op_backward = _CP_ATTENTION_BACKENDS[backend]

    mesh = get_mesh_from_global()
    plc = (Shard(0), Shard(2))
    out_placements = (plc,)

    # Convert kwargs to a comma-separated string of keys and a tuple of values
    # Using a string prevents pytree from flattening it
    kwargs_keys_str = ",".join(kwargs.keys()) if kwargs else ""
    kwargs_values = tuple(kwargs.values())

    # in_placements for: op_forward, op_backward, q, k, v, kwargs_keys_str, *kwargs_values
    # Note: mesh is NOT passed through local_map (it would be flattened by pytree)
    # Instead, we retrieve it inside the autograd function using get_mesh_from_global()
    num_kwargs = len(kwargs)
    in_placements = (None, None, plc, plc, plc, None) + (None,) * num_kwargs

    return local_map(
        _ContextParallelAttention.apply,
        out_placements=out_placements,
        in_placements=in_placements,
        redistribute_inputs=True,
        in_grad_placements=None,
        device_mesh=mesh,
    )(op_forward, op_backward, q, k, v, kwargs_keys_str, *kwargs_values)
