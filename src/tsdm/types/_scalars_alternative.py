r"""Protocol for scalars.

Note:
    We do not provide protocols for string and byte types, since `np.str_` has some
    unintuitive behavior. For instance `np.string_("x") + np.string_("y")` returns
    the plain python string `"xy"`, which is not a numpy string.

Note:
    Boolean Scalars lack the `__invert__` method, because `~True` is `-2`, which is not a boolean.
    This method will get deprecated anyways: https://discuss.python.org/t/bool-deprecation/62232
"""

__all__ = [
    # extras
    "BoolOP",
    "IntOP",
    "FloatOP",
    "ComplexOP",
    # Generic Scalars
    "BaseScalar",
    "OrderedScalar",
    "AdditiveScalar",
    # concrete data types
    "BoolScalar",
    "ComplexScalar",
    "FloatScalar",
    "IntScalar",
    "TimeDelta",
    "TimeStamp",
]

from typing import (
    Protocol,
    Self,
    SupportsFloat,
    SupportsInt,
    overload,
    runtime_checkable,
    type_check_only,
)


@type_check_only
class BoolOP[T](Protocol):
    # similar to numpy._FloatOP
    @overload
    def __call__(self, other: bool, /) -> T: ...  # noqa: FBT001
    @overload
    def __call__(self, other: T, /) -> T: ...


@type_check_only
class IntOP[T](Protocol):
    # similar to numpy._FloatOP
    @overload
    def __call__(self, other: int, /) -> T: ...
    @overload
    def __call__(self, other: T, /) -> T: ...


@type_check_only
class FloatOP[T](Protocol):
    # similar to numpy._FloatOP
    @overload
    def __call__(self, other: float, /) -> T: ...
    @overload
    def __call__(self, other: T, /) -> T: ...


@type_check_only
class ComplexOP[T](Protocol):
    # similar to numpy._FloatOP
    @overload
    def __call__(self, other: complex, /) -> T: ...
    @overload
    def __call__(self, other: T, /) -> T: ...


# region generic scalars ---------------------------------------------------------------
@runtime_checkable
class BaseScalar(Protocol):
    r"""Protocol for scalars.

    Note:
        All scalar types must be hashable.
    """

    def __hash__(self) -> int: ...
    def __eq__(self, other: object, /) -> BoolScalar: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    def __ne__(self, other: object, /) -> BoolScalar: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]


@runtime_checkable
class OrderedScalar(BaseScalar, Protocol):
    r"""Protocol for scalars that support inequality comparisons.

    Examples:
        - `bool`, `int`, `float`, `datetime`, `timedelta`, `tuple`, `str`

    Counter-Examples:
        - `complex` (not ordered)
        - `list` (not hashable)
    """

    def __ge__(self, other: Self, /) -> BoolScalar: ...
    def __gt__(self, other: Self, /) -> BoolScalar: ...
    def __le__(self, other: Self, /) -> BoolScalar: ...
    def __lt__(self, other: Self, /) -> BoolScalar: ...


@runtime_checkable
class AdditiveScalar(BaseScalar, Protocol):
    r"""Protocol for scalars that support addition and subtraction.

    Examples:
        - `int`, `float`, `complex`, `timedelta`

    Counter-Examples:
        - `bool` (does not support `__pos__` and `__neg__`)
        - `datetime.datetime` (does not support addition with datetime)
    """

    # unary operations
    # NOTE: __abs__ disabled due to complex numbers
    # def __abs__(self) -> Self: ...
    def __neg__(self) -> Self: ...
    def __pos__(self) -> Self: ...

    # binary operations
    # + (addition)
    def __add__(self, other: Self, /) -> Self: ...
    def __radd__(self, other: Self, /) -> Self: ...
    # - (subtraction)
    def __sub__(self, other: Self, /) -> Self: ...
    def __rsub__(self, other: Self, /) -> Self: ...


class CompatScalar[T](Protocol):
    r"""Scalar type that is compatible with the given type `T`."""


# endregion generic scalars ------------------------------------------------------------


# region concrete data types -----------------------------------------------------------
@runtime_checkable
class SupportsBool(Protocol):
    r"""Protocol for types that support boolean operations."""

    def __bool__(self) -> bool: ...


@runtime_checkable
class BoolScalar(OrderedScalar, Protocol):
    r"""Protocol for boolean scalars.

    Note:
        Crucially, `BooleanScalar` objects must be convertible to `bool` via the `__bool__` method.
        The `__invert__` method is not included, as `~True` is `-2`, which is not a boolean.
    """

    # conversion to python scalar
    def __bool__(self) -> bool: ...
    def __int__(self) -> int: ...

    # unary operations
    # NOTE: __invert__ disabled, because ~True == -2
    # def __invert__(self) -> Self: ...

    # binary operations
    # and `&`
    __and__: BoolOP[Self]
    __rand__: BoolOP[Self]
    # or `|`
    __or__: BoolOP[Self]
    __ror__: BoolOP[Self]
    # xor `^`
    __xor__: BoolOP[Self]
    __rxor__: BoolOP[Self]


@runtime_checkable
class IntScalar(OrderedScalar, Protocol):
    r"""Protocol for integer scalars."""

    # conversion to python scalar
    def __bool__(self) -> bool: ...
    def __int__(self) -> int: ...
    def __index__(self) -> int: ...

    # unary operations
    def __abs__(self) -> Self: ...
    def __neg__(self) -> Self: ...
    def __pos__(self) -> Self: ...

    # binary operations
    # + (addition)
    __add__: IntOP[Self]
    __radd__: IntOP[Self]
    # - (subtraction)
    __sub__: IntOP[Self]
    __rsub__: IntOP[Self]
    # * (multiplication)
    __mul__: IntOP[Self]
    __rmul__: IntOP[Self]
    # ** (power)
    __pow__: IntOP[Self]
    __rpow__: IntOP[Self]
    # % (modulo)
    __mod__: IntOP[Self]
    __rmod__: IntOP[Self]
    # // (floor division)
    __floordiv__: IntOP[Self]
    __rfloordiv__: IntOP[Self]

    # / (division)
    def __truediv__(self, other: Self | int, /) -> SupportsFloat: ...
    def __rtruediv__(self, other: Self | int, /) -> SupportsFloat: ...


@runtime_checkable
class FloatScalar(OrderedScalar, Protocol):
    r"""Protocol for floating point scalars."""

    # conversion to python scalar
    def __float__(self) -> float: ...

    # unary operations
    # abs() (absolute value)
    def __abs__(self) -> Self: ...
    # - (negation)
    def __neg__(self) -> Self: ...
    # + (positive)
    def __pos__(self) -> Self: ...

    # + (addition)
    __add__: FloatOP[Self]
    __radd__: FloatOP[Self]
    # - (subtraction)
    __sub__: FloatOP[Self]
    __rsub__: FloatOP[Self]
    # * (multiplication)
    __mul__: FloatOP[Self]
    __rmul__: FloatOP[Self]
    # / (division)
    __truediv__: FloatOP[Self]
    __rtruediv__: FloatOP[Self]
    # ** (power)
    __pow__: FloatOP[Self]
    __rpow__: FloatOP[Self]
    # // (floor division)
    __floordiv__: FloatOP[Self]
    __rfloordiv__: FloatOP[Self]
    # % (modulo)
    __mod__: FloatOP[Self]
    __rmod__: FloatOP[Self]


@runtime_checkable
class ComplexScalar(BaseScalar, Protocol):
    r"""Protocol for complex scalars."""

    # @property
    # def imag(self) -> Self: ...
    # @property
    # def real(self) -> Self: ...

    # conversion to python scalar
    def __complex__(self) -> complex: ...

    # unary operations
    def __abs__(self) -> FloatScalar: ...
    def __neg__(self) -> Self: ...
    def __pos__(self) -> Self: ...

    # binary operations
    # + (addition)
    __add__: ComplexOP[Self]
    __radd__: ComplexOP[Self]
    # - (subtraction)
    __sub__: ComplexOP[Self]
    __rsub__: ComplexOP[Self]
    # * (multiplication)
    __mul__: ComplexOP[Self]
    __rmul__: ComplexOP[Self]
    # / (division)
    __truediv__: ComplexOP[Self]
    __rtruediv__: ComplexOP[Self]
    # ** (power)
    __pow__: ComplexOP[Self]
    __rpow__: ComplexOP[Self]


@runtime_checkable
class TimeDelta(OrderedScalar, Protocol):
    r"""Time delta provides several arithmetical operations."""

    # unary operations
    # def __bool__(self) -> bool: ...
    def __abs__(self) -> Self: ...
    def __pos__(self) -> Self: ...
    def __neg__(self) -> Self: ...

    # arithmetic
    # addition +
    def __add__(self, other: Self, /) -> Self: ...
    def __radd__(self, other: Self, /) -> Self: ...

    # subtraction -
    def __sub__(self, other: Self, /) -> Self: ...
    def __rsub__(self, other: Self, /) -> Self: ...

    # multiplication *
    def __mul__(self, other: int, /) -> Self: ...
    def __rmul__(self, other: int, /) -> Self: ...

    # division /
    def __truediv__(self, other: Self, /) -> SupportsFloat: ...

    # @overload
    # def __truediv__(self, other: Self, /) -> float: ...
    # @overload
    # def __truediv__(self, other: float, /) -> Self: ...

    # floor division //
    def __floordiv__(self, other: Self, /) -> SupportsInt: ...

    # @overload
    # def __floordiv__(self, other: Self, /) -> int: ...
    # @overload
    # def __floordiv__(self, other: int, /) -> Self: ...

    # modulo %
    def __mod__(self, other: Self, /) -> Self: ...

    # NOTE: __rmod__ missing on fallback pydatetime
    # def __rmod__(self, other: Self, /) -> Self: ...

    # divmod
    def __divmod__(self, other: Self, /) -> tuple[SupportsInt, Self]: ...

    # NOTE: __rdivmod__ missing on fallback pydatetime
    # def __rdivmod__(self, other: Self, /) -> tuple[SupportsInt, Self]: ...


@runtime_checkable
class TimeStamp[TD: TimeDelta](OrderedScalar, Protocol):
    r"""TimeStamps can be compared and subtracted.

    Note: Due to typing issues, we only support the signature `__sub__(self, other: Self) -> TD`.
    """

    def __add__(self, other: TD, /) -> Self: ...
    def __radd__(self, other: TD, /) -> Self: ...

    # FIXME: https://github.com/python/mypy/issues/18101
    def __sub__(self, other: Self, /) -> TD: ...

    # @overload
    # def __sub__(self, other: Self, /) -> TD: ...
    # @overload
    # def __sub__(self, other: TD, /) -> Self: ...

    # NOTE: __rsub__ missing on fallback pydatetime
    # def __rsub__(self, other: Self, /) -> TD: ...
    # @overload
    # def __rsub__(self, other: TD, /) -> Self: ...
    # @overload
    # def __rsub__(self, other: Self, /) -> TD: ...


# endregion concrete data types --------------------------------------------------------
