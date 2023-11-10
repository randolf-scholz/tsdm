#!/usr/bin/env python


from datetime import datetime, timedelta
from typing import (
    Protocol,
    Self,
    SupportsFloat,
    SupportsInt,
    TypeVar,
    overload,
    runtime_checkable,
)

from numpy import datetime64, float64, int64, timedelta64
from pandas import Timedelta, Timestamp

TD_Type = TypeVar("TD_Type", bound="TimeDelta")
DT_Type = TypeVar("DT_Type", bound="DateTime[TimeDelta]")


@runtime_checkable
class TimeDelta(Protocol):
    """Time delta provides several arithmetical operations."""

    # unary operations
    def __pos__(self: TD_Type) -> TD_Type: ...
    def __neg__(self: TD_Type) -> TD_Type: ...
    def __abs__(self: TD_Type) -> TD_Type: ...
    def __bool__(self) -> bool: ...

    # comparisons
    def __le__(self: TD_Type, other: TD_Type, /) -> bool: ...
    def __lt__(self: TD_Type, other: TD_Type, /) -> bool: ...
    def __ge__(self: TD_Type, other: TD_Type, /) -> bool: ...
    def __gt__(self: TD_Type, other: TD_Type, /) -> bool: ...

    # arithmetic
    # addition +
    def __add__(self: TD_Type, other: TD_Type, /) -> TD_Type: ...
    def __radd__(self: TD_Type, other: TD_Type, /) -> TD_Type: ...

    # subtraction -
    def __sub__(self: TD_Type, other: TD_Type, /) -> TD_Type: ...
    def __rsub__(self: TD_Type, other: TD_Type, /) -> TD_Type: ...

    # multiplication *
    def __mul__(self: TD_Type, other: int, /) -> TD_Type: ...
    def __rmul__(self: TD_Type, other: int, /) -> TD_Type: ...

    # division /
    def __truediv__(self: TD_Type, other: TD_Type, /) -> SupportsFloat: ...

    # @overload
    # def __truediv__(self, other: Self, /) -> float: ...
    # @overload
    # def __truediv__(self, other: float, /) -> Self: ...

    # floor division //
    def __floordiv__(self: TD_Type, other: TD_Type, /) -> SupportsInt: ...

    # @overload
    # def __floordiv__(self, other: Self, /) -> int: ...
    # @overload
    # def __floordiv__(self, other: int, /) -> Self: ...

    # modulo %
    def __mod__(self: TD_Type, other: TD_Type, /) -> TD_Type: ...

    # NOTE: __rmod__ missing on fallback pydatetime
    # def __rmod__(self, other: Self, /) -> Self: ...

    # divmod
    def __divmod__(self: TD_Type, other: TD_Type, /) -> tuple[SupportsInt, TD_Type]: ...

    # NOTE: __rdivmod__ missing on fallback pydatetime
    # def __rdivmod__(self, other: Self, /) -> tuple[SupportsInt, Self]: ...


@runtime_checkable
class DateTime(Protocol[TD_Type]):  # bind appropriate TimeDelta type
    """Datetime can be compared and subtracted."""

    def __le__(self: DT_Type, other: DT_Type, /) -> bool: ...
    def __lt__(self: DT_Type, other: DT_Type, /) -> bool: ...
    def __ge__(self: DT_Type, other: DT_Type, /) -> bool: ...
    def __gt__(self: DT_Type, other: DT_Type, /) -> bool: ...

    def __add__(self: DT_Type, other: TD_Type, /) -> DT_Type: ...
    def __radd__(self: DT_Type, other: TD_Type, /) -> DT_Type: ...

    # Fallback: no overloads
    # def __sub__(self, other: Self, /) -> TD_Type: ...

    # order A
    # @overload
    # def __sub__(self, other: Self, /) -> TD_Type: ...
    # @overload
    # def __sub__(self, other: TD_Type, /) -> Self: ...

    # order B
    @overload
    def __sub__(self, other: TD_Type, /) -> Self: ...
    @overload
    def __sub__(self, other: Self, /) -> TD_Type: ...


# fmt: off
python_dt: DateTime[timedelta] = datetime.fromisoformat("2021-01-01") # incompatible with A
numpy_dt: DateTime[timedelta64] = datetime64("2021-01-01")  # incompatible with B
pandas_dt: DateTime[Timedelta] = Timestamp("2021-01-01") # incompatible with B


python_float_dt: DateTime[float] = float(1.0) # incompatible with B
python_int_dt: DateTime[int] = int(1) # incompatible with B
numpy_float_dt: DateTime[float64] = float64(1.0) # incompatible with B
numpy_int_dt: DateTime[int64] = int64(1) # incompatible with B
# fmt: on
