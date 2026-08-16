r"""Timeseries data structures and functions."""

__all__ = [
    # Protocols
    "TimeSeries",
    "TimeSeriesCollection",
]

from abc import abstractmethod
from collections.abc import Iterator
from typing import Any, ClassVar, Protocol, ReadOnly, Self, overload

# TODO: Use generic slice in 3.15
type RangeSelector[T] = slice | list[T] | list[bool]


class TimeSeries[TableT, TimeT = Any](Protocol):
    r"""Protocol for time series objects.

    Describes a single time series implemented as a Table-like object, indexed by time.
    """

    FIELDS: ClassVar[frozenset[str]]
    r"""The fields of the time series."""

    timeseries: ReadOnly[TableT]  # type: ignore
    r"""The time series data."""
    timeseries_metadata: ReadOnly[TableT | None]  # type: ignore
    r"""Data associated with the time such as measurement device, unit, etc."""
    static_covariates: ReadOnly[TableT | None]  # type: ignore
    r"""The metadata of the dataset."""
    static_covariates_metadata: ReadOnly[TableT | None]  # type: ignore
    r"""Data associated with each metadata such as measurement device, unit,  etc."""

    timeindex: ReadOnly[Any]  # type: ignore
    r"""The time index of the time series."""

    @abstractmethod
    def __len__(self) -> int: ...
    @abstractmethod
    def __iter__(self) -> Iterator[TimeT]: ...
    @abstractmethod
    def __contains__(self, key: TimeT, /) -> object: ...
    @abstractmethod
    def __getitem__(self, key: TimeT | RangeSelector[TimeT], /) -> Self: ...


class TimeSeriesCollection[KeyT, TableT](Protocol):
    r"""Protocol for time series collection objects."""

    FIELDS: ClassVar[frozenset[str]]
    r"""The fields of the time series collection."""

    timeseries: ReadOnly[TableT]  # type: ignore
    r"""The collection of time series data."""
    timeseries_metadata: ReadOnly[TableT | None]  # type: ignore
    r"""Data associated with each channel such as measurement device, unit, etc."""
    static_covariates: ReadOnly[TableT | None]  # type: ignore
    r"""The static covariates associated with each t  # pyrefly: ignore[invalid-annotation]imeseries."""
    static_covariates_metadata: ReadOnly[TableT | None]  # type: ignore
    r"""Data associated with each metadata such as measurement device, unit, etc."""
    constants: ReadOnly[TableT | None]  # type: ignore
    r"""Additional data that is independent of the metaindex."""
    constants_metadata: ReadOnly[TableT | None]  # type: ignore
    r"""Data associated with each global metadata such as measurement device, unit, etc."""

    timeindex: ReadOnly[Any]  # type: ignore
    r"""The time index of the time series."""
    metaindex: ReadOnly[Any]  # type: ignore
    r"""The meta index of the time series collection."""

    @abstractmethod
    def __len__(self) -> int: ...
    @abstractmethod
    def __iter__(self) -> Iterator[KeyT]: ...
    @abstractmethod
    def __contains__(self, item: KeyT, /) -> object: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: RangeSelector[KeyT], /) -> Self: ...
    @overload
    @abstractmethod
    def __getitem__[TimeT = Any](self, key: KeyT, /) -> TimeSeries[TableT, TimeT]: ...
