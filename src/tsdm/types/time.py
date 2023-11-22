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
    # Old types
    "DTVar",
    "TDVar",
    "RealDTVar",
    "RealTDVar",
    "NumpyTDVar",
    "NumpyDTVar",
]


from datetime import datetime as py_dt, timedelta as py_td

from numpy import (
    datetime64 as np_dt,
    floating as np_float,
    integer as np_int,
    timedelta64 as np_td,
)
from pandas import Timedelta as pd_td, Timestamp as pd_dt
from typing_extensions import (
    Any,
    Protocol,
    Self,
    SupportsFloat,
    SupportsInt,
    TypeVar,
    overload,
    runtime_checkable,
)

# region datetime and timedelta protocols ----------------------------------------------
TD = TypeVar("TD", bound="TimeDelta")
TD_co = TypeVar("TD_co", bound="TimeDelta", covariant=True)
TD_contra = TypeVar("TD_contra", bound="TimeDelta", contravariant=True)

# FIXME: https://github.com/python/typing/issues/548
DT = TypeVar("DT", bound="DateTime")
DT_co = TypeVar("DT_co", bound="DateTime", covariant=True)
DT_contra = TypeVar("DT_contra", bound="DateTime", contravariant=True)


@runtime_checkable
class TimeDelta(Protocol):
    """Time delta provides several arithmetical operations."""

    # unary operations
    def __pos__(self: TD) -> TD: ...
    def __neg__(self: TD) -> TD: ...
    def __abs__(self: TD) -> TD: ...
    def __bool__(self) -> bool: ...

    # comparisons
    def __le__(self: TD, other: TD, /) -> bool: ...
    def __lt__(self: TD, other: TD, /) -> bool: ...
    def __ge__(self: TD, other: TD, /) -> bool: ...
    def __gt__(self: TD, other: TD, /) -> bool: ...

    # arithmetic
    # addition +
    def __add__(self: TD, other: TD, /) -> TD: ...
    def __radd__(self: TD, other: TD, /) -> TD: ...

    # subtraction -
    def __sub__(self: TD, other: TD, /) -> TD: ...
    def __rsub__(self: TD, other: TD, /) -> TD: ...

    # multiplication *
    def __mul__(self: TD, other: int, /) -> TD: ...
    def __rmul__(self: TD, other: int, /) -> TD: ...

    # division /
    def __truediv__(self: TD, other: TD, /) -> SupportsFloat: ...

    # @overload
    # def __truediv__(self, other: Self, /) -> float: ...
    # @overload
    # def __truediv__(self, other: float, /) -> Self: ...

    # floor division //
    def __floordiv__(self: TD, other: TD, /) -> SupportsInt: ...

    # @overload
    # def __floordiv__(self, other: Self, /) -> int: ...
    # @overload
    # def __floordiv__(self, other: int, /) -> Self: ...

    # modulo %
    def __mod__(self: TD, other: TD, /) -> TD: ...

    # NOTE: __rmod__ missing on fallback pydatetime
    # def __rmod__(self, other: Self, /) -> Self: ...

    # divmod
    def __divmod__(self: TD, other: TD, /) -> tuple[SupportsInt, TD]: ...

    # NOTE: __rdivmod__ missing on fallback pydatetime
    # def __rdivmod__(self, other: Self, /) -> tuple[SupportsInt, Self]: ...


@runtime_checkable
class DateTime(Protocol[TD]):  # bind appropriate TimeDelta type
    """Datetime can be compared and subtracted."""

    def __le__(self: DT, other: DT, /) -> bool: ...
    def __lt__(self: DT, other: DT, /) -> bool: ...
    def __ge__(self: DT, other: DT, /) -> bool: ...
    def __gt__(self: DT, other: DT, /) -> bool: ...

    def __add__(self: DT, other: TD, /) -> DT: ...
    def __radd__(self: DT, other: TD, /) -> DT: ...

    # NOTE: we only keep this overload, the others are fragile.
    def __sub__(self: DT, other: DT, /) -> TD: ...

    # @overload
    # def __sub__(self, other: TD, /) -> Self: ...
    # @overload
    # def __sub__(self, other: Self, /) -> TD: ...

    # NOTE: __rsub__ missing on fallback pydatetime
    # @overload
    # def __rsub__(self, other: TD, /) -> Self: ...
    # @overload
    # def __rsub__(self, other: Self, /) -> TD: ...


# endregion datetime and timedelta protocols -------------------------------------------


# Time-Type-Variables
DTVar = TypeVar("DTVar", int, float, np_int, np_float, np_dt, pd_dt)
r"""TypeVar for `Timestamp` values."""
TDVar = TypeVar("TDVar", int, float, np_int, np_float, np_td, pd_td)
r"""TypeVar for `Timedelta` values."""

# Real-Time-Type-Variables
RealDTVar = TypeVar("RealDTVar", py_dt, np_dt, pd_dt)
r"""TypeVar for `Timestamp` values."""
RealTDVar = TypeVar("RealTDVar", py_td, np_td, pd_td)
r"""TypeVar for `Timedelta` values."""

# Numpy-Time-Type-Variables
NumpyDTVar = TypeVar("NumpyDTVar", np_int, np_float, np_dt)
r"""TypeVar for `Timestamp` values."""
NumpyTDVar = TypeVar("NumpyTDVar", np_int, np_float, np_td)
r"""TypeVar for `Timedelta` values."""
