"""
ScaledBasis and ArithmeticTuple: Coordinate-producing strides for CuTe layouts.

This module implements the integer-semimodule basis vectors from the CuTe paper,
ported from CUTLASS C++ (cute/numeric/arithmetic_tuple.hpp).

A ScaledBasis(value, *indices) represents a scaled basis vector: value * e_{indices}.
When used as a stride in a Layout, the inner_product produces coordinate tuples
instead of integers, enabling coordinate-producing layouts like:
    Layout((4, 8), (E(0), E(1)))  maps (i, j) -> (i, j)

ArithmeticTuple is a tuple that supports element-wise addition, used to accumulate
coordinate contributions from multiple basis vectors.

Examples:
    E(0) * 3 + E(1) * 5  -> ArithmeticTuple(3, 5) = (3, 5) as coordinates
    Layout((4, 8), (E(0), E(1)))(2, 3) -> (2, 3)
    Layout((4, 8), (E(0), 2*E(1)))(2, 3) -> (2, 6)
"""

from __future__ import annotations


class ScaledBasis:
    """
    A scaled basis vector: value * e_{indices}.

    Represents a unit contribution to a specific position in a coordinate tuple.
    ScaledBasis(1, 0) = e_0 = unit vector in dimension 0.
    ScaledBasis(3, 1) = 3 * e_1 = 3 in dimension 1.

    When multiplied by an integer (coordinate), produces a ScaledBasis with
    scaled value. When added to another ScaledBasis or ArithmeticTuple,
    produces an ArithmeticTuple with values at the right positions.
    """

    __slots__ = ("value", "index")

    def __init__(self, value: int = 1, index: int = 0):
        self.value = value
        self.index = index

    def __mul__(self, other: int) -> ScaledBasis:
        if isinstance(other, int):
            return ScaledBasis(self.value * other, self.index)
        return NotImplemented

    def __rmul__(self, other: int) -> ScaledBasis:
        if isinstance(other, int):
            return ScaledBasis(other * self.value, self.index)
        return NotImplemented

    def __add__(self, other: ScaledBasis | ArithmeticTuple | int) -> ArithmeticTuple:
        if isinstance(other, int):
            if other == 0:
                return _basis_to_atuple(self)
            # int + ScaledBasis: treat int as value at no specific index
            # This produces an ArithmeticTuple with the int added to position 0
            # Actually, for mixed stride addition (e.g., 16 + 3*E(1)):
            # return an ArithmeticTuple that tracks both contributions
            return _basis_to_atuple(self) + ArithmeticTuple(other)
        if isinstance(other, ScaledBasis):
            return _basis_to_atuple(self) + _basis_to_atuple(other)
        if isinstance(other, ArithmeticTuple):
            return _basis_to_atuple(self) + other
        return NotImplemented

    def __radd__(self, other: int | ArithmeticTuple) -> ArithmeticTuple:
        if isinstance(other, int):
            if other == 0:
                return _basis_to_atuple(self)
            return ArithmeticTuple(other) + _basis_to_atuple(self)
        if isinstance(other, ArithmeticTuple):
            return other + _basis_to_atuple(self)
        return NotImplemented

    def __floordiv__(self, other: int) -> ScaledBasis:
        """Floor divide the value, keeping the basis index."""
        if isinstance(other, int):
            return ScaledBasis(self.value // other, self.index)
        return NotImplemented

    def __mod__(self, other: int) -> ScaledBasis:
        """Modulo the value, keeping the basis index."""
        if isinstance(other, int):
            return ScaledBasis(self.value % other, self.index)
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ScaledBasis):
            return self.value == other.value and self.index == other.index
        if isinstance(other, int):
            return False
        return NotImplemented

    def __lt__(self, other: object) -> bool:
        if isinstance(other, ScaledBasis):
            return (self.index, self.value) < (other.index, other.value)
        if isinstance(other, int):
            return False  # ScaledBasis sorts after integers
        return NotImplemented

    def __le__(self, other: object) -> bool:
        if isinstance(other, ScaledBasis):
            return (self.index, self.value) <= (other.index, other.value)
        if isinstance(other, int):
            return False
        return NotImplemented

    def __gt__(self, other: object) -> bool:
        if isinstance(other, ScaledBasis):
            return (self.index, self.value) > (other.index, other.value)
        if isinstance(other, int):
            return True  # ScaledBasis sorts after integers
        return NotImplemented

    def __ge__(self, other: object) -> bool:
        if isinstance(other, ScaledBasis):
            return (self.index, self.value) >= (other.index, other.value)
        if isinstance(other, int):
            return True
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.value, self.index))

    def __repr__(self) -> str:
        if self.value == 1:
            return f"E({self.index})"
        return f"{self.value}*E({self.index})"

    def __str__(self) -> str:
        return self.__repr__()


def E(index: int) -> ScaledBasis:
    """Create a unit basis vector e_{index}."""
    return ScaledBasis(1, index)


class ArithmeticTuple:
    """
    A tuple supporting element-wise arithmetic, used for coordinate accumulation.

    Produced by adding ScaledBasis vectors together:
        E(0) * 3 + E(1) * 5 -> ArithmeticTuple(3, 5)

    Supports addition with other ArithmeticTuples and ScaledBasis vectors.
    """

    __slots__ = ("values",)

    def __init__(self, *values: int):
        self.values = values

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, i: int) -> int:
        return self.values[i]

    def __add__(self, other: ArithmeticTuple | ScaledBasis | int) -> ArithmeticTuple:
        if isinstance(other, int) and other == 0:
            return self
        if isinstance(other, ScaledBasis):
            return self + _basis_to_atuple(other)
        if isinstance(other, ArithmeticTuple):
            # Extend to the max length, padding with 0
            n = max(len(self), len(other))
            a = self.values + (0,) * (n - len(self))
            b = other.values + (0,) * (n - len(other))
            return ArithmeticTuple(*(x + y for x, y in zip(a, b)))
        return NotImplemented

    def __radd__(self, other: int) -> ArithmeticTuple:
        if isinstance(other, int) and other == 0:
            return self
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ArithmeticTuple):
            return self.values == other.values
        if isinstance(other, tuple):
            return self.values == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.values)

    def __repr__(self) -> str:
        return f"ArithmeticTuple{self.values}"

    def __str__(self) -> str:
        return str(self.values)

    def to_tuple(self) -> tuple[int, ...]:
        return self.values


def _basis_to_atuple(b: ScaledBasis) -> ArithmeticTuple:
    """Convert a ScaledBasis to an ArithmeticTuple with the value at the right position."""
    values = [0] * (b.index + 1)
    values[b.index] = b.value
    return ArithmeticTuple(*values)


def is_scaled_basis(x: object) -> bool:
    return isinstance(x, ScaledBasis)


def is_arithmetic_tuple(x: object) -> bool:
    return isinstance(x, ArithmeticTuple)


def is_coord_stride(x: object) -> bool:
    """Check if x is a coordinate-producing stride (ScaledBasis or tuple containing them)."""
    if isinstance(x, (ScaledBasis, ArithmeticTuple)):
        return True
    if isinstance(x, tuple):
        return any(is_coord_stride(e) for e in x)
    return False


def make_basis_like(shape, depth=0):
    """
    Create coordinate-producing strides matching the given shape.

    For shape (M, N), returns (E(0), E(1)).
    For shape ((M, N), K), returns ((E(0), E(1)), E(2)).

    Uses lexicographic ordering: leftmost index is outermost (slowest-varying).
    """
    if isinstance(shape, tuple):
        results = []
        idx = [depth]
        for s in shape:
            result, next_idx = _make_basis_like_rec(s, idx[0])
            results.append(result)
            idx[0] = next_idx
        return tuple(results)
    else:
        return E(depth)


def _make_basis_like_rec(shape, start_idx):
    """Recursive helper returning (basis, next_index)."""
    if isinstance(shape, tuple):
        results = []
        idx = start_idx
        for s in shape:
            result, idx = _make_basis_like_rec(s, idx)
            results.append(result)
        return tuple(results), idx
    else:
        return E(start_idx), start_idx + 1
