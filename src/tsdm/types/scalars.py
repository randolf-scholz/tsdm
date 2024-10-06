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
    "BaseScalar",
    "OrderedScalar",
    "AdditiveScalar",
    # concrete data types
    "BoolScalar",
    "ComplexScalar",
    "DateTime",
    "FloatScalar",
    "IntScalar",
    "TimeDelta",
]

from typing import (
    Protocol,
    Self,
    SupportsFloat,
    SupportsInt,
    runtime_checkable,
)


# region generic scalars ---------------------------------------------------------------
@runtime_checkable
class BaseScalar(Protocol):
    r"""Protocol for scalars.

    Note:
        All scalar types must be hashable.
    """

    def __hash__(self) -> int: ...
    def __eq__(self, other: object, /) -> "BoolScalar": ...  # type: ignore[override]
    def __ne__(self, other: object, /) -> "BoolScalar": ...  # type: ignore[override]


@runtime_checkable
class OrderedScalar(BaseScalar, Protocol):
    r"""Protocol for scalars that support inequality comparisons.

    Examples:
        - `bool`, `int`, `float`, `datetime`, `timedelta`, `tuple`, `str`

    Counter-Examples:
        - `complex` (not ordered)
        - `list` (not hashable)
    """

    def __ge__(self, other: Self, /) -> "BoolScalar": ...
    def __gt__(self, other: Self, /) -> "BoolScalar": ...
    def __le__(self, other: Self, /) -> "BoolScalar": ...
    def __lt__(self, other: Self, /) -> "BoolScalar": ...


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


# endregion generic scalars ------------------------------------------------------------


# region concrete data types -----------------------------------------------------------


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
    def __and__(self, other: Self, /) -> Self: ...
    def __rand__(self, other: Self, /) -> Self: ...
    # or `|`
    def __or__(self, other: Self, /) -> Self: ...
    def __ror__(self, other: Self, /) -> Self: ...
    # xor `^`
    def __xor__(self, other: Self, /) -> Self: ...
    def __rxor__(self, other: Self, /) -> Self: ...


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
    def __add__(self, other: Self, /) -> Self: ...
    def __radd__(self, other: Self, /) -> Self: ...
    # - (subtraction)
    def __sub__(self, other: Self, /) -> Self: ...
    def __rsub__(self, other: Self, /) -> Self: ...
    # * (multiplication)
    def __mul__(self, other: Self, /) -> Self: ...
    def __rmul__(self, other: Self, /) -> Self: ...
    # ** (power)
    def __pow__(self, exponent: Self, /) -> Self: ...
    def __rpow__(self, base: Self, /) -> Self: ...
    # % (modulo)
    def __mod__(self, other: Self, /) -> Self: ...
    def __rmod__(self, other: Self, /) -> Self: ...
    # / (division)
    def __truediv__(self, other: Self, /) -> SupportsFloat: ...
    def __rtruediv__(self, other: Self, /) -> SupportsFloat: ...
    # // (floor division)
    def __floordiv__(self, other: Self, /) -> SupportsInt: ...
    def __rfloordiv__(self, other: Self, /) -> SupportsInt: ...


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

    # binary operations
    # + (addition)
    def __add__(self, other: Self, /) -> Self: ...
    def __radd__(self, other: Self, /) -> Self: ...
    # - (subtraction)
    def __sub__(self, other: Self, /) -> Self: ...
    def __rsub__(self, other: Self, /) -> Self: ...
    # * (multiplication)
    def __mul__(self, other: Self, /) -> Self: ...
    def __rmul__(self, other: Self, /) -> Self: ...
    # / (division)
    def __truediv__(self, other: Self, /) -> Self: ...
    def __rtruediv__(self, other: Self, /) -> Self: ...
    # ** (power)
    def __pow__(self, exponent: Self, /) -> Self: ...
    def __rpow__(self, base: Self, /) -> Self: ...
    # // (floor division)
    def __floordiv__(self, other: Self, /) -> Self: ...
    def __rfloordiv__(self, other: Self, /) -> Self: ...
    # % (modulo)
    def __mod__(self, other: Self, /) -> Self: ...
    def __rmod__(self, other: Self, /) -> Self: ...


@runtime_checkable
class ComplexScalar(BaseScalar, Protocol):
    r"""Protocol for complex scalars."""

    # conversion to python scalar
    def __complex__(self) -> complex: ...

    # unary operations
    def __abs__(self) -> FloatScalar: ...
    def __neg__(self) -> Self: ...
    def __pos__(self) -> Self: ...

    # binary operations
    # + (addition)
    def __add__(self, other: Self, /) -> Self: ...
    def __radd__(self, other: Self, /) -> Self: ...
    # - (subtraction)
    def __sub__(self, other: Self, /) -> Self: ...
    def __rsub__(self, other: Self, /) -> Self: ...
    # * (multiplication)
    def __mul__(self, other: Self, /) -> Self: ...
    def __rmul__(self, other: Self, /) -> Self: ...
    # / (division)
    def __truediv__(self, other: Self, /) -> Self: ...
    def __rtruediv__(self, other: Self, /) -> Self: ...
    # ** (power)
    def __pow__(self, exponent: Self, /) -> Self: ...
    def __rpow__(self, base: Self, /) -> Self: ...

    # @property
    # def imag(self) -> Self: ...
    # @property
    # def real(self) -> Self: ...


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
class DateTime[TD: TimeDelta](OrderedScalar, Protocol):
    r"""Datetime can be compared and subtracted.

    Note: Due to typing issues, we only support the signature `__sub__(self, other: Self) -> TD`.
    """

    def __add__(self, other: TD, /) -> Self: ...
    def __radd__(self, other: TD, /) -> Self: ...

    # NOTE: we only keep this overload, the others are fragile.
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