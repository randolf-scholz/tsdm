r"""Protocols for operator overloading support."""

# ruff: noqa: D101
__all__ = [
    # Unary arithmetic
    "SupportsUnaryArithmetic",
    # Comparison operations
    "SupportsEquality",
    "SupportsComparison",
    # Mixins
    "SupportsArray",
    "SupportsArrayUfunc",
    "SupportsDataFrame",
    "SupportsDevice",
    "SupportsDtype",
    "SupportsItem",
    "SupportsNdim",
    "SupportsRound",
    "SupportsShape",
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

from abc import abstractmethod
from typing import Any, Literal, Protocol, Self, overload, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class SupportsShape(Protocol):
    r"""Protocol for objects that support shape."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]: ...


@runtime_checkable
class SupportsNdim(Protocol):
    r"""We just test for ndim, since e.g. tf.Tensor does not have ndim."""

    @property
    @abstractmethod
    def ndim(self) -> int: ...


@runtime_checkable
class SupportsDevice[DeviceT = Any](Protocol):
    r"""Protocol for objects that support `device`."""

    @property
    @abstractmethod
    def device(self) -> DeviceT: ...


@runtime_checkable
class SupportsDtype[DTypeT = Any](Protocol):
    r"""We just test for dtype, since e.g. tf.Tensor does not have ndim.

    Examples:
        - `numpy.ndarray`
        - `pandas.Series`
        - `polars.Series`
        - `torch.Tensor`

    Examples: (Counter-Examples)
        - `pandas.DataFrame`
        - `polars.DataFrame`
        - `pyarrow.Array`
    """

    @property
    @abstractmethod
    def dtype(self) -> DTypeT: ...


@runtime_checkable
class SupportsArray(Protocol):
    r"""Protocol for objects that support `__array__`.

    References:
        https://numpy.org/doc/stable/reference/c-api/array.html
    """

    @abstractmethod
    def __array__(self) -> NDArray[np.object_]:
        r"""Return the array of the tensor."""
        ...


@runtime_checkable
class SupportsArrayUfunc(SupportsArray, Protocol):
    r"""Protocol for objects that support `__array_ufunc__`.

    Notably, numpy functions like `numpy.exp` can be directly applied to such objects.
    The main example are `pandas.Series` and `pandas.DataFrame`.

    Examples:
        - `numpy.ndarray`
        - `pandas.DataFrame`
        - `pandas.Index`
        - `pandas.Series`
        - `polars.Series`

    Examples: (Couter-Examples)
        - `polars.DataFrame`
        - `pyarrow.Array`
        - `pyarrow.Table`
        - `torch.Tensor`

    References:
        https://numpy.org/doc/stable/reference/ufuncs.html
    """

    @abstractmethod
    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: Literal["__call__", "reduce", "reduceat", "accumulate", "outer", "at"],
        /,
        *inputs: Any,
        **kwargs: Any,
    ) -> Self:
        r"""Return the array resulting from applying the ufunc."""
        ...


@runtime_checkable
class SupportsDataFrame[DataFrameT = Any](Protocol):
    r"""Protocol for objects that support `__dataframe__`.

    References:
        https://data-apis.org/dataframe-protocol/latest/index.html
    """

    @abstractmethod
    def __dataframe__(self) -> DataFrameT: ...


@runtime_checkable
class SupportsItem[Scalar](Protocol):  # +T
    r"""Protocol for objects that support `.item()`.

    Return the scalar value the tensor if it only has a single element.

    If the tensor has more than one element, raise an error.
    """

    @abstractmethod
    def item(self) -> Scalar: ...


@runtime_checkable
class SupportsRound(Protocol):
    r"""Protocol for objects that support `round`."""

    # FIXME: https://github.com/python/typing/discussions/1782
    @overload
    @abstractmethod
    def round(self) -> Self: ...
    @overload
    @abstractmethod
    def round(self, *, decimals: int) -> Self: ...


class SupportsEquality[ComparableT = Any, ResultT = Any](Protocol):
    # Note: really no good choice for the argument type here,
    #   as most built-ins will require object argument, but DSLs will restrict to their own type.
    #   so just use Any.
    # equality ==
    def __eq__(self, other: ComparableT, /) -> ResultT: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    # inequality !=
    def __ne__(self, other: ComparableT, /) -> ResultT: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]


class SupportsComparison[ComparableT = Any, ResultT = Any](Protocol):
    # comparisons (element-wise)
    def __le__(self, other: Self | ComparableT, /) -> ResultT: ...
    def __ge__(self, other: Self | ComparableT, /) -> ResultT: ...
    def __lt__(self, other: Self | ComparableT, /) -> ResultT: ...
    def __gt__(self, other: Self | ComparableT, /) -> ResultT: ...

    # @overload
    # def __le__(self, other: ComparableT, /) -> ResultT: ...
    # @overload
    # def __le__(self, other: Self, /) -> ResultT: ...
    # @overload
    # def __ge__(self, other: ComparableT, /) -> ResultT: ...
    # @overload
    # def __ge__(self, other: Self, /) -> ResultT: ...
    # @overload
    # def __lt__(self, other: ComparableT, /) -> ResultT: ...
    # @overload
    # def __lt__(self, other: Self, /) -> ResultT: ...
    # @overload
    # def __gt__(self, other: ComparableT, /) -> ResultT: ...
    # @overload
    # def __gt__(self, other: Self, /) -> ResultT: ...


class SupportsUnaryArithmetic(Protocol):
    def __abs__(self) -> Self: ...
    def __pos__(self) -> Self: ...
    def __neg__(self) -> Self: ...


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
