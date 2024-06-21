r"""Types related to time."""

__all__ = [
    # protocol
    "TimeDelta",
    "DateTime",
    # new types
    "TD",
    "TD_co",
    "TD_contra",
    "DT",
    "DT_co",
    "DT_contra",
]

from typing_extensions import (
    Any,
    Protocol,
    Self,
    SupportsFloat,
    SupportsInt,
    TypeVar,
    runtime_checkable,
)

# region datetime and timedelta protocols ----------------------------------------------
TD = TypeVar("TD", bound="TimeDelta")
TD_co = TypeVar("TD_co", bound="TimeDelta", covariant=True)
TD_contra = TypeVar("TD_contra", bound="TimeDelta", contravariant=True)
DT = TypeVar("DT", bound="DateTime[Any]")
DT_co = TypeVar("DT_co", bound="DateTime[Any]", covariant=True)
DT_contra = TypeVar("DT_contra", bound="DateTime[Any]", contravariant=True)


@runtime_checkable
class TimeDelta(Protocol):
    r"""Time delta provides several arithmetical operations."""

    # unary operations
    def __pos__(self) -> Self: ...
    def __neg__(self) -> Self: ...
    def __abs__(self) -> Self: ...
    def __bool__(self) -> bool: ...

    # comparisons
    def __le__(self, other: Self, /) -> bool: ...
    def __lt__(self, other: Self, /) -> bool: ...
    def __ge__(self, other: Self, /) -> bool: ...
    def __gt__(self, other: Self, /) -> bool: ...

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
class DateTime(Protocol[TD]):  # bind appropriate TimeDelta type
    r"""Datetime can be compared and subtracted."""

    def __le__(self, other: Self, /) -> bool: ...
    def __lt__(self, other: Self, /) -> bool: ...
    def __ge__(self, other: Self, /) -> bool: ...
    def __gt__(self, other: Self, /) -> bool: ...

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


# endregion datetime and timedelta protocols -------------------------------------------
