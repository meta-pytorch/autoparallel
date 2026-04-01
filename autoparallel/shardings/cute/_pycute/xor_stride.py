"""
XorStride: XOR-based strides for CuTe layouts.

This module implements an integer-semimodule where group addition is replaced
with binary XOR, as described in the CuTe paper's discussion of generalized
stride types.

In the XOR semimodule over (Z, +):
- coord * XorStride(v) = XorStride(v) if coord is odd, XorStride(0) if even
  (because s ^ s = 0, so repeated XOR self-cancels)
- XorStride(a) + XorStride(b) = XorStride(a ^ b)
- int + XorStride(v) resolves to int ^ v (plain int)

This means XorStride modes must have shape 2 (binary) for injectivity,
since 2*s = s ^ s = 0 in the XOR semimodule.

Use cases:
- Ring attention zigzag (context parallelism):
    Layout((2,2,2), (1, 2, XorStride(7)))  # GPU g gets chunks g and 7-g
- Shared memory bank conflict swizzle patterns:
    Layout((2,2), (1, XorStride(3)))  # equivalent to Swizzle(1,0,1)

Examples:
    XorStride(7) * 1  -> XorStride(7)
    XorStride(7) * 0  -> XorStride(0)
    XorStride(3) + XorStride(5)  -> XorStride(6)  (3 ^ 5 = 6)
    4 + XorStride(3)  -> 7  (4 ^ 3 = 7)
"""

from __future__ import annotations


class XorStride:
    """
    A stride whose contributions combine via XOR instead of addition.

    When used as a stride in a Layout, the inner_product XORs this stride's
    contribution into the accumulated index rather than adding it.
    """

    __slots__ = ("value",)

    def __init__(self, value: int):
        self.value = value

    def __mul__(self, other: int) -> XorStride:
        if isinstance(other, int):
            return XorStride(self.value if other % 2 == 1 else 0)
        return NotImplemented

    def __rmul__(self, other: int) -> XorStride:
        return self.__mul__(other)

    def __add__(self, other: XorStride | int) -> XorStride | int:
        if isinstance(other, XorStride):
            return XorStride(self.value ^ other.value)
        if isinstance(other, int):
            return other ^ self.value
        return NotImplemented

    def __radd__(self, other: XorStride | int) -> XorStride | int:
        return self.__add__(other)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, XorStride):
            return self.value == other.value
        if isinstance(other, int):
            return self.value == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(("XorStride", self.value))

    def __repr__(self) -> str:
        return f"XorStride({self.value})"

    def __str__(self) -> str:
        return self.__repr__()


def is_xor_stride(x: object) -> bool:
    return isinstance(x, XorStride)
