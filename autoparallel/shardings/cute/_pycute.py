"""
Vendored subset of pycute (NVIDIA CUTLASS CuTe layouts).

This module contains the minimal set of CuTe layout utilities needed for
CuTe-based sharding placement and propagation. Copied from:
    cutlass/python/pycute/ (typing.py, int_tuple.py, layout.py)

Original copyright:
    Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    SPDX-License-Identifier: BSD-3-Clause
"""

from abc import ABC
from functools import reduce
from itertools import chain

# =============================================================================
# typing.py — Integer ABC
# =============================================================================


class Integer(ABC):
    @classmethod
    def __subclasshook__(cls, c):
        if c in [bool, float]:
            return False
        return issubclass(c, int)


# =============================================================================
# int_tuple.py — Functions for manipulating IntTuples
# =============================================================================


def is_int(x):
    return isinstance(x, Integer)


def is_tuple(x):
    return isinstance(x, tuple)


def flatten(t):
    if is_tuple(t):
        if len(t) == 0:
            return ()
        else:
            return tuple(i for a in t for i in flatten(a))
    else:
        return (t,)


def signum(a):
    return bool(a > 0) - bool(a < 0)


def product(a):
    if is_tuple(a):
        return reduce(lambda val, elem: val * product(elem), a, 1)
    else:
        return a


def inner_product(a, b):
    if is_tuple(a):
        assert len(a) == len(b)
        return sum(inner_product(x, y) for x, y in zip(a, b))
    else:
        assert not is_tuple(b)
        return a * b


def shape_div(a, b):
    """Inclusive prefix ceil div with output congruent to input a."""
    if is_tuple(a):
        if is_tuple(b):
            assert len(a) == len(b)
            return tuple(shape_div(x, y) for x, y in zip(a, b))
        else:
            r = []
            for v in a:
                r.append(shape_div(v, b))
                b = shape_div(b, product(v))
            return tuple(r)
    else:
        if is_tuple(b):
            return shape_div(a, product(b))
        else:
            assert a % b == 0 or b % a == 0
            return (a + b - 1) // b


def prefix_product(a, init=1):
    """Exclusive prefix product with output congruent to input a."""
    if is_tuple(a):
        if is_tuple(init):
            assert len(a) == len(init)
            return tuple(prefix_product(x, i) for x, i in zip(a, init))
        else:
            r = []
            for v in a:
                r.append(prefix_product(v, init))
                init = init * product(v)
            return tuple(r)
    else:
        if is_tuple(init):
            assert False
        else:
            return init


def idx2crd(idx, shape, stride=None):
    if stride is None:
        stride = prefix_product(shape)

    if is_tuple(idx):
        if is_tuple(shape):
            assert len(idx) == len(shape) and len(idx) == len(stride)
            return tuple(idx2crd(i, s, d) for i, s, d in zip(idx, shape, stride))
        else:
            assert False
    else:
        if is_tuple(shape):
            assert len(shape) == len(stride)
            return tuple(idx2crd(idx, s, d) for s, d in zip(shape, stride))
        else:
            return (idx // stride) % shape


def crd2idx(crd, shape, stride=None):
    if stride is None:
        stride = prefix_product(shape)

    if is_tuple(crd):
        if is_tuple(shape):
            assert len(crd) == len(shape) and len(crd) == len(stride)
            return sum(crd2idx(c, s, d) for c, s, d in zip(crd, shape, stride))
        else:
            assert False, f"crd={crd}, shape={shape}"
    else:
        if crd is None:
            crd = 0

        if is_tuple(shape):
            assert len(shape) == len(stride)
            result = 0
            for i in range(len(shape) - 1):
                result += crd2idx(crd % product(shape[i]), shape[i], stride[i])
                crd = crd // product(shape[i])
            return result + crd2idx(crd, shape[-1], stride[-1])
        else:
            return crd * stride


def slice_(crd, trg):
    """Filter trg according to crd: keep only elements paired with None."""
    if is_tuple(crd):
        if is_tuple(trg):
            assert len(crd) == len(trg)
            return tuple(
                chain(
                    *filter(
                        lambda x: x != (),
                        [slice_(c, s) for c, s in zip(crd, trg)],
                    )
                )
            )
        else:
            assert False
    elif crd is None:
        return (trg,)
    else:
        return ()


def has_none(a):
    """Determine if None appears at any terminal of an int_tuple."""
    if is_tuple(a):
        return any(has_none(v) for v in a)
    else:
        return a is None


# =============================================================================
# layout.py — CuTe Layout class and algebra
# =============================================================================


class LayoutBase:
    pass


def is_layout(x):
    return isinstance(x, LayoutBase)


class Layout(LayoutBase):
    def __init__(self, _shape, _stride=None):
        self.shape = _shape
        if _stride is None:
            self.stride = prefix_product(self.shape)
        else:
            self.stride = _stride

    def __eq__(self, other):
        return self.shape == other.shape and self.stride == other.stride

    def __len__(self):
        if is_tuple(self.shape):
            return len(self.shape)
        else:
            return 1

    def __call__(self, *args):
        """Map a logical coordinate to a linear index, or slice the layout."""
        if has_none(args):
            if len(args) == 1:
                return Layout(slice_(args[0], self.shape), slice_(args[0], self.stride))
            else:
                return Layout(slice_(args, self.shape), slice_(args, self.stride))
        else:
            if len(args) == 1:
                return crd2idx(args[0], self.shape, self.stride)
            else:
                return crd2idx(args, self.shape, self.stride)

    def __getitem__(self, i):
        if is_tuple(self.shape):
            return Layout(self.shape[i], self.stride[i])
        else:
            assert i == 0
            return Layout(self.shape, self.stride)

    def size(self):
        """Size of the domain."""
        return product(self.shape)

    def cosize(self):
        """Size of the codomain."""
        return self(self.size() - 1) + 1

    def __str__(self):
        return f"{self.shape}:{self.stride}"

    def __repr__(self):
        return f"Layout({self.shape},{self.stride})"

    def __hash__(self):
        def _make_hashable(x):
            if is_tuple(x):
                return tuple(_make_hashable(e) for e in x)
            return x

        return hash((_make_hashable(self.shape), _make_hashable(self.stride)))


def make_layout(*layouts):
    """Make a Layout from a list of layouts (each becomes a mode)."""
    if len(layouts) == 1 and not is_layout(layouts[0]):
        layouts = layouts[0]

    shape, stride = zip(*((a.shape, a.stride) for a in layouts))
    return Layout(shape, stride)


def size(layout):
    if is_layout(layout):
        return layout.size()
    return product(layout)


def cosize(layout):
    return layout.cosize()


def coalesce(layout, profile=None):
    """Flatten and combine modes while preserving the int-to-int function."""
    if is_tuple(profile):
        assert len(layout) >= len(profile)
        return make_layout(
            chain(
                (coalesce(layout[i], profile[i]) for i in range(0, len(profile))),
                (layout[i] for i in range(len(profile), len(layout))),
            )
        )

    result_shape = [1]
    result_stride = [0]
    for shape, stride in zip(flatten(layout.shape), flatten(layout.stride)):
        if shape == 1:
            continue
        elif result_shape[-1] == 1:
            result_shape[-1] = shape
            result_stride[-1] = stride
        elif result_shape[-1] * result_stride[-1] == stride:
            result_shape[-1] = result_shape[-1] * shape
        else:
            result_shape.append(shape)
            result_stride.append(stride)

    if len(result_shape) == 1:
        return Layout(result_shape[0], result_stride[0])
    else:
        return Layout(tuple(result_shape), tuple(result_stride))


def layout_filter(layout, profile=None):
    """Replace stride-0 modes with size-1, then coalesce to remove them."""
    if is_tuple(profile):
        assert len(layout) >= len(profile)
        return make_layout(
            chain(
                (
                    layout_filter(layout[i], profile[i])
                    for i in range(0, len(profile))
                ),
                (layout[i] for i in range(len(profile), len(layout))),
            )
        )

    result_shape = []
    result_stride = []
    for shape, stride in zip(flatten(layout.shape), flatten(layout.stride)):
        if not (shape == 1 or stride == 0):
            result_shape.append(shape)
            result_stride.append(stride)

    if len(result_shape) == 0:
        return Layout(1, 0)
    else:
        return coalesce(Layout(tuple(result_shape), tuple(result_stride)))


def composition(layout_a, layout_b):
    """Layout composition: R(c) = A(B(c))."""
    if layout_b is None:
        return layout_a
    elif is_int(layout_b):
        return composition(layout_a, Layout(layout_b))
    elif is_tuple(layout_b):
        assert len(layout_a) >= len(layout_b)
        return make_layout(
            chain(
                (
                    composition(layout_a[i], layout_b[i])
                    for i in range(0, len(layout_b))
                ),
                (layout_a[i] for i in range(len(layout_b), len(layout_a))),
            )
        )
    elif is_tuple(layout_b.shape):
        return make_layout(
            composition(layout_a, b_i) for b_i in layout_b
        )

    if layout_b.stride == 0:
        return Layout(layout_b.shape, 0)
    else:
        result_shape = []
        result_stride = []
        rest_shape = layout_b.shape
        rest_stride = layout_b.stride
        flat_a = coalesce(layout_a)
        for curr_shape, curr_stride in zip(
            flatten(flat_a.shape)[:-1], flatten(flat_a.stride)[:-1]
        ):
            assert curr_shape % rest_stride == 0 or rest_stride % curr_shape == 0
            new_shape = min(max(1, curr_shape // rest_stride), rest_shape)

            if new_shape != 1:
                result_shape.append(new_shape)
                result_stride.append(rest_stride * curr_stride)

            rest_shape = rest_shape // new_shape
            # ceil_div(abs(rest_stride), curr_shape) * signum(rest_stride)
            rest_stride = -(-rest_stride // curr_shape)

        if rest_shape != 1 or len(result_shape) == 0:
            result_shape.append(rest_shape)
            result_stride.append(rest_stride * flatten(flat_a.stride)[-1])

        if len(result_shape) == 1:
            return Layout(result_shape[0], result_stride[0])
        else:
            return Layout(tuple(result_shape), tuple(result_stride))


def complement(layout, max_idx=1):
    """Layout complement: a layout with disjoint image from layout."""
    if is_int(layout):
        return complement(Layout(layout))

    result_shape = []
    result_stride = []
    current_idx = 1

    sorted_ds = sorted(zip(flatten(layout.stride), flatten(layout.shape)))
    for stride, shape in sorted_ds:
        if stride == 0 or shape == 1:
            continue

        in_bound = current_idx <= shape * stride
        assert (type(in_bound) is not bool) or in_bound

        result_shape.append(stride // current_idx)
        result_stride.append(current_idx)
        current_idx = shape * stride

    result_shape.append((max_idx + current_idx - 1) // current_idx)  # ceil_div
    result_stride.append(current_idx)

    return coalesce(Layout(tuple(result_shape), tuple(result_stride)))


def right_inverse(layout):
    """Layout right inverse: L(L^‡(k)) = k."""
    if layout is None:
        return None
    elif is_int(layout):
        return Layout(layout)

    result_shape = []
    result_stride = []
    current_idx = 1

    flat_shape = flatten(layout.shape)
    flat_stride = flatten(layout.stride)
    sorted_dsa = sorted(zip(flat_stride, flat_shape, prefix_product(flat_shape)))
    for stride, shape, rstride in sorted_dsa:
        if shape == 1:
            continue
        if current_idx != stride:
            break

        result_shape.append(shape)
        result_stride.append(rstride)
        current_idx = shape * stride

    return coalesce(Layout(tuple(result_shape), tuple(result_stride)))


def left_inverse(layout):
    """Layout left inverse: L^†(L(k)) = k."""
    if layout is None:
        return None
    elif is_int(layout):
        return Layout(layout)
    return right_inverse(make_layout(layout, complement(layout)))


def logical_divide(layout_a, layout_b):
    """Split layout_a by composition of layout_b and its complement."""
    if layout_b is None:
        return layout_a
    elif is_int(layout_b):
        return logical_divide(layout_a, Layout(layout_b))
    elif is_tuple(layout_b):
        assert len(layout_a) >= len(layout_b)
        return make_layout(
            chain(
                (
                    logical_divide(layout_a[i], layout_b[i])
                    for i in range(0, len(layout_b))
                ),
                (layout_a[i] for i in range(len(layout_b), len(layout_a))),
            )
        )

    return composition(
        layout_a, make_layout(layout_b, complement(layout_b, size(layout_a)))
    )
