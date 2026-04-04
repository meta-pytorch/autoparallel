"""
Op registry: maps ATen operator names to CuTe sharding propagation functions.

Each entry maps an ATen op string (e.g., "aten.mm.default") to a propagation
function from propagation.py. The optimizer looks up the right function given
an op from the computation graph.

Many ATen ops map to the same propagation function (e.g., all pointwise ops
map to propagate_pointwise, all reduction variants map to propagate_reduction).
"""

from .propagation import (
    propagate_addmm,
    propagate_argmax,
    propagate_argmin,
    propagate_baddbmm,
    propagate_bmm,
    propagate_broadcast,
    propagate_cat,
    propagate_convolution,
    propagate_cumsum,
    propagate_cumprod,
    propagate_dot,
    propagate_dropout,
    propagate_einsum,
    propagate_embedding,
    propagate_expand,
    propagate_flatten,
    propagate_flip,
    propagate_gather,
    propagate_identity,
    propagate_index_select,
    propagate_layer_norm,
    propagate_mm,
    propagate_movedim,
    propagate_permute,
    propagate_pointwise,
    propagate_reduction,
    propagate_repeat,
    propagate_roll,
    propagate_scatter,
    propagate_select,
    propagate_slice,
    propagate_softmax,
    propagate_sort,
    propagate_split,
    propagate_squeeze,
    propagate_stack,
    propagate_t,
    propagate_topk,
    propagate_transpose,
    propagate_unbind,
    propagate_unflatten,
    propagate_unsqueeze,
    propagate_view,
)


# =============================================================================
# Registry: ATen op name -> propagation function
# =============================================================================

# Identity ops: all dims Carry from single input
_IDENTITY_OPS = [
    "aten.alias.default",
    "aten.clone.default",
    "aten.contiguous.default",
    "aten.detach.default",
    "aten.empty_like.default",
    "aten.fill_.Scalar",
    "aten.full_like.default",
    "aten.ones_like.default",
    "aten.rand_like.default",
    "aten.randint_like.default",
    "aten.randint_like.low_dtype",
    "aten.randn_like.default",
    "aten.view.dtype",
    "aten.zero_.default",
    "aten.zeros_like.default",
    "prims.view_of.default",
]

# Pointwise ops: broadcast-compatible Carry across inputs
_POINTWISE_OPS = [
    # Arithmetic
    "aten.add.Scalar",
    "aten.add.Tensor",
    "aten.add.out",
    "aten.add_.Scalar",
    "aten.add_.Tensor",
    "aten.sub.Scalar",
    "aten.sub.Tensor",
    "aten.sub.out",
    "aten.sub_.Scalar",
    "aten.sub_.Tensor",
    "aten.mul.Scalar",
    "aten.mul.Tensor",
    "aten.mul.out",
    "aten.mul_.Scalar",
    "aten.mul_.Tensor",
    "aten.div.Scalar",
    "aten.div.Tensor",
    "aten.div.Tensor_mode",
    "aten.div.out",
    "aten.div.out_mode",
    "aten.div_.Scalar",
    "aten.div_.Tensor",
    "aten.div_.Tensor_mode",
    "aten.true_divide.Tensor",
    "aten.true_divide.Scalar",
    "aten.rsub.Scalar",
    "aten.rsub.Tensor",
    "aten.pow.Tensor_Tensor",
    "aten.pow.Tensor_Scalar",
    "aten.pow.Scalar",
    "aten.pow_.Scalar",
    "aten.floor_divide.Tensor",
    "aten.floor_divide.Scalar",
    "aten.fmod.Tensor",
    "aten.fmod.Scalar",
    "aten.fmod_.Tensor",
    "aten.fmod_.Scalar",
    "aten.remainder.Tensor",
    "aten.remainder.Scalar",
    "aten.remainder_.Tensor",
    "aten.remainder_.Scalar",
    "aten.addcmul.default",
    "aten.addcmul_.default",
    "aten.addcdiv.default",
    "aten.addcdiv_.default",
    # Comparison
    "aten.eq.Scalar",
    "aten.eq.Tensor",
    "aten.ne.Scalar",
    "aten.ne.Tensor",
    "aten.lt.Scalar",
    "aten.lt.Tensor",
    "aten.le.Scalar",
    "aten.le.Tensor",
    "aten.gt.Scalar",
    "aten.gt.Tensor",
    "aten.ge.Scalar",
    "aten.ge.Tensor",
    # Unary math
    "aten.abs.default",
    "aten.abs_.default",
    "aten.neg.default",
    "aten.neg_.default",
    "aten.sqrt.default",
    "aten.sqrt_.default",
    "aten.rsqrt.default",
    "aten.rsqrt_.default",
    "aten.reciprocal.default",
    "aten.reciprocal_.default",
    "aten.exp.default",
    "aten.exp_.default",
    "aten.exp2.default",
    "aten.expm1.default",
    "aten.log.default",
    "aten.log_.default",
    "aten.log2.default",
    "aten.log10.default",
    "aten.log1p.default",
    "aten.sin.default",
    "aten.sin_.default",
    "aten.cos.default",
    "aten.cos_.default",
    "aten.tan.default",
    "aten.tan_.default",
    "aten.asin.default",
    "aten.acos.default",
    "aten.atan.default",
    "aten.sinh.default",
    "aten.cosh.default",
    "aten.tanh.default",
    "aten.tanh_.default",
    "aten.asinh.default",
    "aten.acosh.default",
    "aten.atanh.default",
    "aten.erf.default",
    "aten.erf_.default",
    "aten.erfc.default",
    "aten.erfinv.default",
    "aten.lgamma.default",
    "aten.digamma.default",
    "aten.ceil.default",
    "aten.ceil_.default",
    "aten.floor.default",
    "aten.floor_.default",
    "aten.round.default",
    "aten.round_.default",
    "aten.trunc.default",
    "aten.trunc_.default",
    "aten.sign.default",
    "aten.sign_.default",
    "aten.sgn.default",
    "aten.frac.default",
    "aten.frac_.default",
    "aten.positive.default",
    "aten.isnan.default",
    "aten.isinf.default",
    "aten.signbit.default",
    "aten.conj_physical.default",
    "aten.conj_physical_.default",
    "aten._conj.default",
    "aten.angle.default",
    # Activations
    "aten.relu.default",
    "aten.relu_.default",
    "aten.gelu.default",
    "aten.gelu_backward.default",
    "aten.silu.default",
    "aten.silu_.default",
    "aten.silu_backward.default",
    "aten.sigmoid.default",
    "aten.sigmoid_.default",
    "aten.sigmoid_backward.default",
    "aten.tanh_backward.default",
    "aten.threshold_backward.default",
    "aten.hardtanh.default",
    "aten.hardtanh_.default",
    "aten.logit.default",
    "aten.logit_.default",
    # Clamp
    "aten.clamp.default",
    "aten.clamp.Tensor",
    "aten.clamp_.default",
    "aten.clamp_.Tensor",
    "aten.clamp_min.default",
    "aten.clamp_max.default",
    "aten.clip.default",
    "aten.clip_.default",
    # Binary element-wise
    "aten.maximum.default",
    "aten.minimum.default",
    "aten.fmax.default",
    "aten.fmin.default",
    "aten.copysign.Tensor",
    "aten.copysign.Scalar",
    "aten.hypot.default",
    "aten.nextafter.default",
    "aten.xlogy.Tensor",
    "aten.xlogy.Scalar",
    "aten.lerp.Tensor",
    "aten.lerp.Scalar",
    "aten.lerp_.Tensor",
    "aten.lerp_.Scalar",
    "aten.heaviside.default",
    "aten.logaddexp.default",
    "aten.float_power.Tensor_Tensor",
    "aten.float_power.Tensor_Scalar",
    "aten.float_power.Scalar",
    # Ternary
    "aten.where.self",
    "aten.where.self_out",
    # Masking
    "aten.masked_fill.Scalar",
    "aten.masked_fill_.Scalar",
    "aten.masked_fill.Tensor",
    "aten.masked_fill_.Tensor",
    "aten.nan_to_num.default",
    "aten.nan_to_num_.default",
    # Bitwise
    "aten.bitwise_and.Scalar",
    "aten.bitwise_and.Tensor",
    "aten.bitwise_and_.Scalar",
    "aten.bitwise_and_.Tensor",
    "aten.bitwise_or.Scalar",
    "aten.bitwise_or.Tensor",
    "aten.bitwise_or_.Scalar",
    "aten.bitwise_or_.Tensor",
    "aten.bitwise_xor.Scalar",
    "aten.bitwise_xor.Tensor",
    "aten.bitwise_xor_.Scalar",
    "aten.bitwise_xor_.Tensor",
    "aten.bitwise_not.default",
    "aten.bitwise_not_.default",
    # Logical
    "aten.logical_and.default",
    "aten.logical_or.default",
    "aten.logical_xor.default",
    "aten.logical_not.default",
    # Copy / cast
    "aten.copy_.default",
    "aten.to.dtype",
    # Dropout backward
    "aten.native_dropout_backward.default",
]

# Reduction ops: Remove dim with Partial
_REDUCTION_OPS = {
    # (op_name, reduce_op) — reduce_op determines the Partial type
    "aten.sum.default": "sum",
    "aten.sum.dim_IntList": "sum",
    "aten.mean.default": "sum",  # mean is sum / count
    "aten.mean.dim": "sum",
    "aten.mean.out": "sum",
    "aten.prod.default": "sum",  # prod uses log-sum-exp or custom partial
    "aten.prod.dim_int": "sum",
    "aten.max.default": "max",
    "aten.max.dim": "max",
    "aten.max.out": "max",
    "aten.min.default": "min",
    "aten.min.dim": "min",
    "aten.min.out": "min",
    "aten.amax.default": "max",
    "aten.amax.out": "max",
    "aten.amin.default": "min",
    "aten.amin.out": "min",
    "aten.all.default": "sum",  # all = (sum == count)
    "aten.all.dim": "sum",
    "aten.any.default": "sum",  # any = (sum > 0)
    "aten.any.dim": "sum",
    "aten.any.dims": "sum",
    "aten.nansum.default": "sum",
    "aten.logsumexp.default": "sum",
}

# Ops that require replicate on affected dims (non-linear reductions, order-dependent)
_REPLICATE_AFFECTED_OPS = [
    "aten.argmax.default",
    "aten.argmin.default",
    "aten.cumsum.default",
    "aten.cumprod.default",
    "aten.cummax.default",
    "aten.cummin.default",
    "aten.logcumsumexp.default",
    "aten.sort.default",
    "aten.sort.stable",
    "aten.topk.default",
    "aten.kthvalue.default",
    "aten.median.default",
    "aten.median.dim",
    "aten.mode.default",
    "aten.nanmedian.default",
    "aten.nanmedian.dim",
    "aten._softmax.default",
    "aten._softmax_backward_data.default",
    "aten._log_softmax.default",
    "aten._log_softmax_backward_data.default",
    "aten._safe_softmax.default",
    "aten.native_layer_norm.default",
    "aten.native_layer_norm_backward.default",
    "aten._fused_rms_norm.default",
    "aten._fused_rms_norm_backward.default",
    "aten.std.correction",
    "aten.var.correction",
    "aten.var_mean.correction",
]

# Random ops: identity but reject Partial
_RANDOM_OPS = [
    "aten.normal_.default",
    "aten.uniform_.default",
    "aten.native_dropout.default",
    "aten.bernoulli_.float",
    "aten.bernoulli.default",
]


# =============================================================================
# Build the registry
# =============================================================================


OP_REGISTRY = {}

# Identity
for op in _IDENTITY_OPS:
    OP_REGISTRY[op] = propagate_identity

# Pointwise
for op in _POINTWISE_OPS:
    OP_REGISTRY[op] = propagate_pointwise

# Reduction
for op, reduce_op in _REDUCTION_OPS.items():
    OP_REGISTRY[op] = propagate_reduction

# Replicate-affected
for op in _REPLICATE_AFFECTED_OPS:
    # These map to various specific functions depending on the op
    # For the registry, we use a generic marker; the caller extracts the dim arg
    OP_REGISTRY[op] = propagate_softmax  # placeholder — caller dispatches by op

# Random
for op in _RANDOM_OPS:
    OP_REGISTRY[op] = propagate_dropout

# Named ops (one-to-one)
OP_REGISTRY.update({
    # Matrix ops
    "aten.mm.default": propagate_mm,
    "aten.bmm.default": propagate_bmm,
    "aten.addmm.default": propagate_addmm,
    "aten.baddbmm.default": propagate_baddbmm,
    "aten.dot.default": propagate_dot,
    "aten.t.default": propagate_t,
    # View ops
    "aten.view.default": propagate_view,
    "aten.view_copy.default": propagate_view,
    "aten._unsafe_view.default": propagate_view,
    "aten.reshape.default": propagate_view,
    "aten.permute.default": propagate_permute,
    "aten.transpose.int": propagate_transpose,
    "aten.unsqueeze.default": propagate_unsqueeze,
    "aten.squeeze.default": propagate_squeeze,
    "aten.squeeze.dim": propagate_squeeze,
    "aten.squeeze.dims": propagate_squeeze,
    "aten.squeeze_.dim": propagate_squeeze,
    "aten.squeeze_.dims": propagate_squeeze,
    "aten.expand.default": propagate_expand,
    "aten.expand_copy.default": propagate_expand,
    "aten.repeat.default": propagate_repeat,
    # Tensor ops
    "aten.select.int": propagate_select,
    "aten.slice.Tensor": propagate_slice,
    "aten.cat.default": propagate_cat,
    "aten.stack.default": propagate_stack,
    "aten.split.Tensor": propagate_split,
    "aten.split_with_sizes.default": propagate_split,
    "aten.unbind.int": propagate_unbind,
    "aten.flip.default": propagate_flip,
    "aten.roll.default": propagate_roll,
    "aten.gather.default": propagate_gather,
    "aten.index_select.default": propagate_index_select,
    "aten.scatter.src": propagate_scatter,
    "aten.scatter.value": propagate_scatter,
    "aten.scatter_.src": propagate_scatter,
    "aten.scatter_.value": propagate_scatter,
    # Embedding
    "aten.embedding.default": propagate_embedding,
    # Convolution
    "aten.convolution.default": propagate_convolution,
})


def get_propagation_rule(op_name):
    """Look up the propagation function for an ATen operator.

    Returns the propagation function, or None if the op is not registered.
    """
    return OP_REGISTRY.get(op_name)
