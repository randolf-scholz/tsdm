#! /usr/bin/env python3

from typing import (
    Any,
    Protocol,
    Self,
    SupportsFloat,
    SupportsInt,
    TypeAlias,
    TypeVar,
    overload,
    reveal_type,
)


class TimeDelta(Protocol):
    """Time delta provides several arithmetical operations."""

    # unary operations
    def __pos__(self) -> Self: ...
    def __neg__(self) -> Self: ...
    def __abs__(self) -> Self: ...

    # comparisons
    def __eq__(self, other: object, /) -> bool: ...
    def __ne__(self, other: object, /) -> bool: ...
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


TD = TypeVar("TD", bound=TimeDelta, contravariant=True)
# TD: TypeAlias = TimeDelta


class DateTime(Protocol[TD]):  # bind appropriate TimeDelta type
    """Datetime can be compared and subtracted."""

    def __hash__(self) -> int: ...

    def __eq__(self, other: object, /) -> bool: ...

    def __le__(self, other: Self, /) -> bool: ...
    def __lt__(self, other: Self, /) -> bool: ...
    def __ge__(self, other: Self, /) -> bool: ...
    def __gt__(self, other: Self, /) -> bool: ...

    def __add__(self, other: TD, /) -> Self: ...
    def __radd__(self, other: TD, /) -> Self: ...
    def __sub__(self, other: Self, /) -> Any: ...


def main():
    from datetime import datetime, timedelta

    import numpy
    import pandas

    ISO_DATE = "2021-01-01"

    TIMEDELTAS: dict[str, TimeDelta] = {
        "float": 10.0,
        "int": 10,
        "numpy": numpy.timedelta64(1, "D"),
        "numpy_float": numpy.float64(10.0),
        "numpy_int": numpy.int64(10),
        "pandas": pandas.Timedelta(days=1),
        "python": timedelta(days=1),
        # "arrow": pyarrow.scalar(TD, type=pyarrow_td_type),
    }

    DATETIMES: dict[str, DateTime] = {
        "float": 10.0,
        "int": 10,
        "numpy": numpy.datetime64(ISO_DATE),
        "numpy_float": numpy.float64(10.0),
        "numpy_int": numpy.int64(10),
        "pandas": pandas.Timestamp(ISO_DATE),
        "python": datetime.fromisoformat(ISO_DATE),
        # "arrow": pyarrow.scalar(DT, type=pyarrow.timestamp("s")),
    }

    d = datetime.fromisoformat(ISO_DATE)
    e: DateTime = datetime.fromisoformat(ISO_DATE)
    f: DateTime = numpy.float64(10.0)
    g: DateTime = numpy.int64(10)
    w: DateTime = numpy.datetime64(ISO_DATE)
    x: DateTime = 1.0
    y: DateTime = 10
    z: DateTime = datetime.fromisoformat(ISO_DATE)
    reveal_type(d - d)
    reveal_type(e - e)
    reveal_type(d + (d - d))
    reveal_type(e + (e - e))
    reveal_type(w - w)


if __name__ == "__main__":
    main()
