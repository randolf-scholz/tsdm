r"""Timeseries data structures and functions."""

__all__ = [
    # Protocols
    "Metadata",
    "TimeSeries",
    "TimeSeriesCollection",
]

from abc import abstractmethod
from collections.abc import Iterator, Mapping
from typing import Any, ClassVar, Protocol, Self, TypedDict, overload


class Metadata(TypedDict, total=False):
    r"""Protocol for metadata objects."""

    name: str
    r"""The name of the dataset."""
    tags: frozenset[str]
    r"""Tags associated with the dataset."""
    lables: Mapping[str, Any]
    r"""Labels associated with the dataset."""


class TimeSeries[TableT, TimeT = Any](Protocol):
    r"""Protocol for time series objects.

    Describes a single time series implemented as a Table-like object, indexed by time.
    """

    FIELDS: ClassVar[frozenset[str]]
    r"""The fields of the time series."""

    timeseries: TableT
    r"""The time series data."""
    timeseries_metadata: TableT | None
    r"""Data associated with the time such as measurement device, unit, etc."""
    static_covariates: TableT | None
    r"""The metadata of the dataset."""
    static_covariates_metadata: TableT | None
    r"""Data associated with each metadata such as measurement device, unit,  etc."""
    metadata: TableT | None
    r"""The metadata of the dataset."""

    @abstractmethod
    def __len__(self) -> int: ...
    @abstractmethod
    def __iter__(self) -> Iterator[TimeT]: ...
    @abstractmethod
    def __contains__(self, key: TimeT, /) -> object: ...
    @abstractmethod
    def __getitem__(self, key: TimeT, /) -> Self: ...


class TimeSeriesCollection[KeyT, TableT](Protocol):
    r"""Protocol for time series collection objects."""

    FIELDS: ClassVar[frozenset[str]]
    r"""The fields of the time series collection."""

    timeseries: TableT
    r"""The collection of time series data."""
    timeseries_metadata: TableT | None
    r"""Data associated with each channel such as measurement device, unit, etc."""
    static_covariates: TableT | None
    r"""The static covariates associated with each timeseries."""
    static_covariates_metadata: TableT | None
    r"""Data associated with each metadata such as measurement device, unit,  etc."""
    constants: TableT | None
    r"""Additional data that is independent of the metaindex."""
    constants_metadata: TableT | None
    r"""Data associated with each global metadata such as measurement device, unit,  etc."""
    metadata: TableT | None
    r"""The metadata of the dataset."""

    @abstractmethod
    def __len__(self) -> int: ...
    @abstractmethod
    def __iter__(self) -> Iterator[KeyT]: ...
    @abstractmethod
    def __contains__(self, item: KeyT, /) -> object: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: slice | list[KeyT], /) -> Self: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: KeyT, /) -> TimeSeries[TableT]: ...
