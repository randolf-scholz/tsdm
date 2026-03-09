r"""Protocols for operator overloading support."""

# ruff: noqa: D101
__all__ = [
    # Boolean operations
    "SupportsBoolOP",
    "SupportsMutBoolOP",
    # Arithmetic operations
    "SupportsAdd",
    "SupportsSub",
    "SupportsMul",
    "SupportsDiv",
    "SupportsModulo",
    "SupportsPow",
    "SupportsFloorDiv",
    "SupportsMatmul",
    # Mutable operations
    "SupportsMutAdd",
    "SupportsMutSub",
    "SupportsMutMul",
    "SupportsMutDiv",
    "SupportsMutModulo",
    "SupportsMutPow",
    "SupportsMutFloorDiv",
    "SupportsMutMatmul",
]

from typing import Protocol, Self


# region Boolean OPS -------------------------------------------------------------------
class SupportsBoolOP[Scalar](Protocol):
    # AND &
    def __and__(self, other: Self | Scalar, /) -> Self: ...
    def __rand__(self, other: Self | Scalar, /) -> Self: ...
    # OR |
    def __or__(self, other: Self | Scalar, /) -> Self: ...
    def __ror__(self, other: Self | Scalar, /) -> Self: ...
    # XOR ^
    def __xor__(self, other: Self | Scalar, /) -> Self: ...
    def __rxor__(self, other: Self | Scalar, /) -> Self: ...


class SupportsMutBoolOP[Scalar](SupportsBoolOP[Scalar], Protocol):
    # inplace boolean operations
    # AND &=
    def __iand__(self, other: Self | Scalar, /) -> Self: ...
    # OR |=
    def __ior__(self, other: Self | Scalar, /) -> Self: ...
    # XOR ^=
    def __ixor__(self, other: Self | Scalar, /) -> Self: ...


# endregion Boolean OPS ----------------------------------------------------------------


# region Arithmetic OPS ----------------------------------------------------------------
class SupportsAdd[Scalar](Protocol):
    # addition +
    def __add__(self, other: Self | Scalar, /) -> Self: ...
    def __radd__(self, other: Self | Scalar, /) -> Self: ...


class SupportsSub[Scalar](Protocol):
    # subtraction -
    def __sub__(self, other: Self | Scalar, /) -> Self: ...
    def __rsub__(self, other: Self | Scalar, /) -> Self: ...


class SupportsMul[Scalar](Protocol):
    # multiplication *
    def __mul__(self, other: Self | Scalar, /) -> Self: ...
    def __rmul__(self, other: Self | Scalar, /) -> Self: ...


class SupportsDiv[Scalar](Protocol):
    # true division /
    def __truediv__(self, other: Self | Scalar, /) -> Self: ...
    def __rtruediv__(self, other: Self | Scalar, /) -> Self: ...


class SupportsModulo[Scalar](Protocol):
    # modulo %
    def __mod__(self, other: Self | Scalar, /) -> Self: ...
    def __rmod__(self, other: Self | Scalar, /) -> Self: ...


class SupportsPow[Scalar](Protocol):
    # power **
    def __pow__(self, exponent: Self | Scalar, /) -> Self: ...
    def __rpow__(self, base: Self | Scalar, /) -> Self: ...


class SupportsFloorDiv[Scalar](Protocol):
    # floor division //
    def __floordiv__(self, other: Self | Scalar, /) -> Self: ...
    def __rfloordiv__(self, other: Self | Scalar, /) -> Self: ...


class SupportsMatmul(Protocol):
    # matrix multiplication @
    def __matmul__(self, other: Self, /) -> Self: ...
    def __rmatmul__(self, other: Self, /) -> Self: ...


# endregion Arithmetic OPS -------------------------------------------------------------


# region Mutable Arithmetic OPS --------------------------------------------------------
class SupportsMutAdd[Scalar](SupportsAdd[Scalar], Protocol):
    # addition +=
    def __iadd__(self, other: Self | Scalar, /) -> Self: ...


class SupportsMutSub[Scalar](SupportsSub[Scalar], Protocol):
    # subtraction -=
    def __isub__(self, other: Self | Scalar, /) -> Self: ...


class SupportsMutMul[Scalar](SupportsMul[Scalar], Protocol):
    # multiplication *=
    def __imul__(self, other: Self | Scalar, /) -> Self: ...


class SupportsMutDiv[Scalar](SupportsDiv[Scalar], Protocol):
    # inplace true division /=
    def __itruediv__(self, other: Self | Scalar, /) -> Self: ...


class SupportsMutModulo[Scalar](SupportsModulo[Scalar], Protocol):
    # inplace modulo %=
    def __imod__(self, other: Self | Scalar, /) -> Self: ...


class SupportsMutPow[Scalar](SupportsPow[Scalar], Protocol):
    # inplace power **=
    def __ipow__(self, exponent: Self | Scalar, /) -> Self: ...


class SupportsMutFloorDiv[Scalar](SupportsFloorDiv[Scalar], Protocol):
    # inplace floor division //=
    def __ifloordiv__(self, other: Self | Scalar, /) -> Self: ...


class SupportsMutMatmul(SupportsMatmul, Protocol):
    # inplace matrix multiplication @=
    def __imatmul__(self, other: Self, /) -> Self: ...


# endregion Mutable Arithmetic OPS -----------------------------------------------------
