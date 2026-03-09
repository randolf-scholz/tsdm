r"""Protocols for series-like numerical data types."""

__all__ = [
    # Comparison Protocols
    "SupportsSeriesComparison",
    "SupportsSeriesEquality",
    # Series Protocols
    "SeriesType",
    "SeriesSupportsComparison",
    "SpanLikeSeries",
    "TimeLikeSeries",
    # Specialized Series Protocols
    "BooleanSeries",
    "IntegerSeries",
    "FloatSeries",
    "ComplexSeries",
    "TimedeltaSeries",
    "DatetimeSeries",
]

from collections.abc import Iterator
from typing import Any, Protocol, Self, overload, runtime_checkable

from numpy.typing import NDArray

from numerical_types.arrays import (
    BooleanArray,
    ComplexArray,
    DatetimeArray,
    FloatArray,
    IntegerArray,
    SpanLikeArray,
    TimedeltaArray,
    TimeLikeArray,
)
from numerical_types.scalars import (
    BoolScalar,
    ComplexScalar,
    FloatScalar,
    IntScalar,
)


class SupportsSeriesEquality[ComparableT](Protocol):  # noqa: D101
    # equality ==
    def __eq__(self, other: ComparableT, /) -> BooleanSeries: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    # inequality !=
    def __ne__(self, other: ComparableT, /) -> BooleanSeries: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]


class SupportsSeriesComparison[ComparableT](Protocol):  # noqa: D101
    # comparisons (element-wise)
    def __le__(self, other: ComparableT, /) -> BooleanSeries: ...
    def __ge__(self, other: ComparableT, /) -> BooleanSeries: ...
    def __lt__(self, other: ComparableT, /) -> BooleanSeries: ...
    def __gt__(self, other: ComparableT, /) -> BooleanSeries: ...


@runtime_checkable
class SeriesType[V](Protocol):
    r"""A 1d-array of homogeneous data type.

    Examples:
        - `pandas.Index`
        - `pandas.Series`
        - `polars.Series`
        - `pandas.extensions.ExtensionArray`
        - `pyarrow.Array`

    Counter-Examples:
        - `numpy.ndarray`     lacks `equals`
        - `pandas.DataFrame`  lacks `equals`
        - `polars.DataFrame`  lacks `equals`
        - `pyarrow.Table`     lacks `equals`
        - `torch.Tensor`      lacks `equals`

    NOTE: Many methods have subtle differences between backends:
     - `diff`: gives discrete differences for polars and pandas, but not for pyarrow
     - `value_counts`: polars returns a DataFrame, pandas a Series, pyarrow a StructArray
     - `unique`: polars and pyarrow return `Self`, pandas returns `np.ndarray` or ExtensionArray.
     - `to_numpy`: is superfluous.

    References:
        - https://numpy.org/devdocs/user/basics.interoperability.html
    """

    def __array__(self) -> NDArray: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[V]: ...
    @overload
    def __getitem__(self, key: int, /) -> V: ...
    @overload
    def __getitem__(self, key: slice, /) -> Self: ...

    def equals(self, other: Self, /) -> bool:
        r"""Check if the series is equal to another series."""
        ...


@runtime_checkable
class SeriesSupportsComparison[ComparableT](
    SupportsSeriesEquality[ComparableT],
    SupportsSeriesComparison[ComparableT],
    Protocol,
):
    r"""Protocol for series-like types supporting comparison operations."""


@runtime_checkable
class BooleanSeries[BoolT](
    SupportsSeriesEquality,
    BooleanArray[BoolT],
    Protocol,
):
    r"""Protocol for boolean series-like types supporting standard boolean operations."""

    def __iter__(self) -> Iterator[BoolT]: ...
    def all(self) -> BoolScalar | Any: ...
    def any(self) -> BoolScalar | Any: ...


@runtime_checkable
class IntegerSeries[IntT](
    SupportsSeriesEquality,
    SupportsSeriesComparison,
    IntegerArray[IntT],
    Protocol,
):
    r"""Protocol for integer series-like types supporting standard arithmetic operations."""

    def __iter__(self) -> Iterator[IntT]: ...
    def min(self) -> IntScalar | Any: ...
    def max(self) -> IntScalar | Any: ...
    def sum(self) -> IntScalar | Any: ...


@runtime_checkable
class FloatSeries[FloatT](
    SupportsSeriesEquality,
    SupportsSeriesComparison,
    FloatArray[FloatT],
    Protocol,
):
    r"""Protocol for floating-point series-like types supporting standard arithmetic operations."""

    def __iter__(self) -> Iterator[FloatT]: ...
    # aggregations
    def min(self) -> FloatScalar | Any: ...
    def max(self) -> FloatScalar | Any: ...
    def mean(self) -> FloatScalar | Any: ...
    def sum(self) -> FloatScalar | Any: ...
    def std(self) -> FloatScalar | Any: ...
    def var(self) -> FloatScalar | Any: ...


@runtime_checkable
class ComplexSeries[ComplexT](
    SupportsSeriesEquality,
    ComplexArray[ComplexT],
    Protocol,
):
    r"""Protocol for complex series-like types supporting standard arithmetic operations."""

    def __iter__(self) -> Iterator[ComplexT]: ...
    def sum(self) -> ComplexScalar | Any: ...
    def mean(self) -> ComplexScalar | Any: ...
    def std(self) -> FloatScalar: ...
    def var(self) -> FloatScalar: ...


@runtime_checkable
class SpanLikeSeries[SpanT](
    SupportsSeriesEquality,
    SupportsSeriesComparison,
    SpanLikeArray[SpanT],
    Protocol,
):
    r"""Protocol for duration series-like types supporting standard duration operations."""

    def __iter__(self) -> Iterator[SpanT]: ...
    # / (division)
    @overload
    def __truediv__(self, other: Self | SpanT, /) -> FloatSeries: ...
    @overload  # NOTE: 'Self | Any' to fix numpy overload issue
    def __truediv__(self, other: int, /) -> Self | Any: ...


@runtime_checkable
class TimedeltaSeries[SpanT](
    SupportsSeriesEquality,
    SupportsSeriesComparison,
    TimedeltaArray[SpanT],
    Protocol,
):
    r"""Protocol for timedelta series-like types supporting standard duration operations."""

    def __iter__(self) -> Iterator[SpanT]: ...


@runtime_checkable
class TimeLikeSeries[
    TimeT,
    SpanT = Any,
    DualT: SpanLikeSeries = Any,
](
    SupportsSeriesEquality,
    SupportsSeriesComparison,
    TimeLikeArray[TimeT, SpanT, DualT],
    Protocol,
):
    r"""Protocol for timestamp series-like types supporting standard datetime operations."""

    def __iter__(self) -> Iterator[TimeT]: ...


@runtime_checkable
class DatetimeSeries[
    TimeT,
    SpanT = Any,
    DualT: TimedeltaSeries = Any,
](
    SupportsSeriesEquality,
    SupportsSeriesComparison,
    DatetimeArray[TimeT, SpanT, DualT],
    Protocol,
):
    r"""Protocol for datetime series-like types supporting standard datetime operations."""

    def __iter__(self) -> Iterator[TimeT]: ...
