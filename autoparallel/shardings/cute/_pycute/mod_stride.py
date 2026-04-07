"""
ModStride: modular arithmetic strides for CuTe layouts.

This module implements an integer-semimodule where arithmetic is performed
modulo a fixed modulus, enabling cyclic/rotational patterns in CuTe layouts.

In the modular semimodule over Z/nZ:
- coord * ModStride(v, n) = ModStride((coord * v) % n, n)
- ModStride(a, n) + ModStride(b, n) = ModStride((a + b) % n, n)
- ModStride(a, n) + int = (a + int) % n  (resolves to plain int)
- int + ModStride(a, n) = (int + a) % n  (resolves to plain int)
- ModStride(a, n) + XorStride(v) = a ^ v  (resolves to plain int, enables mixed layouts)

For injectivity, gcd(value, modulus) must equal 1 when used as a stride
on a mode of size modulus. Otherwise the mapping is not one-to-one.

Use cases:
- Ring attention step rotation:
    Layout((4, 4), (ModStride(1, 4), ModStride(3, 4)))
    # (gpu, step) -> source_gpu = (gpu + 3*step) % 4 = (gpu - step) % 4
- Combined with XorStride for full ring attention schedule:
    Layout((2, 2, 2, 4), (ModStride(1,4), ModStride(2,4), XorStride(7), ModStride(3,4)))
    # (b0, b1, pair, step) -> chunk assignment with zigzag + rotation

Examples:
    ModStride(3, 4) * 2  -> ModStride(2, 4)    (3*2 % 4 = 6 % 4 = 2)
    ModStride(3, 4) * 0  -> ModStride(0, 4)
    ModStride(1, 4) + ModStride(3, 4) -> ModStride(0, 4)  (1+3 % 4 = 0)
    5 + ModStride(3, 4)  -> 0  ((5+3) % 4 = 0)
"""

from __future__ import annotations

from .xor_stride import XorStride


class ModStride:
    """
    A stride whose contributions combine via modular arithmetic.

    When used as a stride in a Layout, the inner_product computes
    contributions modulo the given modulus.
    """

    __slots__ = ("value", "modulus")

    def __init__(self, value: int, modulus: int):
        self.value = value % modulus
        self.modulus = modulus

    def __mul__(self, other: int) -> ModStride:
        if isinstance(other, int):
            return ModStride((self.value * other) % self.modulus, self.modulus)
        return NotImplemented

    def __rmul__(self, other: int) -> ModStride:
        return self.__mul__(other)

    def __add__(self, other: ModStride | XorStride | int) -> ModStride | int:
        if isinstance(other, ModStride):
            if self.modulus != other.modulus:
                raise ValueError(
                    f"Cannot add ModStrides with different moduli: {self.modulus} vs {other.modulus}"
                )
            return ModStride((self.value + other.value) % self.modulus, self.modulus)
        if isinstance(other, XorStride):
            # ModStride resolves to int, then XOR
            return self.value ^ other.value
        if isinstance(other, int):
            return (self.value + other) % self.modulus
        return NotImplemented

    def __radd__(self, other: ModStride | XorStride | int) -> ModStride | int:
        if isinstance(other, int):
            return (other + self.value) % self.modulus
        if isinstance(other, XorStride):
            # XorStride + ModStride: resolve ModStride to int, then XOR
            return other.value ^ self.value
        return self.__add__(other)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ModStride):
            return self.value == other.value and self.modulus == other.modulus
        if isinstance(other, int):
            return self.value == other % self.modulus
        return NotImplemented

    def __hash__(self) -> int:
        return hash(("ModStride", self.value, self.modulus))

    def __repr__(self) -> str:
        return f"ModStride({self.value}, {self.modulus})"

    def __str__(self) -> str:
        return self.__repr__()


def is_mod_stride(x: object) -> bool:
    return isinstance(x, ModStride)
