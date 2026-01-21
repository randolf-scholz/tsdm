r"""Timeseries data structures and functions."""

__all__ = [
    # Protocols
    "Metadata",
    "TimeSeries",
    "TimeSeriesCollection",
    # Classes
    # NamedTuples
]

from abc import abstractmethod
from collections.abc import Hashable, Iterator, Mapping
from typing import Any, ClassVar, Optional, Protocol, Self, overload


class Metadata(Protocol):
    r"""Protocol for metadata objects."""

    name: Optional[str]
    r"""The name of the dataset."""
    tags: frozenset[str]
    r"""Tags associated with the dataset."""
    lables: Mapping[str, Any]
    r"""Labels associated with the dataset."""


class TimeSeries[T](Protocol):
    r"""Protocol for time series objects."""

    FIELDS: ClassVar[frozenset[str]]
    r"""The fields of the time series."""

    timeseries: T
    r"""The time series data."""
    timeseries_metadata: Optional[T]
    r"""Data associated with the time such as measurement device, unit, etc."""
    static_covariates: Optional[T]
    r"""The metadata of the dataset."""
    static_covariates_metadata: Optional[T]
    r"""Data associated with each metadata such as measurement device, unit,  etc."""
    metadata: Optional[Metadata]
    r"""The metadata of the dataset."""

    @abstractmethod
    def __len__(self) -> int: ...
    @abstractmethod
    def __iter__(self) -> Iterator: ...
    @abstractmethod
    def __contains__(self, key: Hashable, /) -> bool: ...
    @abstractmethod
    def __getitem__(self, key: Any, /) -> Self: ...


class TimeSeriesCollection[Key, T](Protocol):
    r"""Protocol for time series collection objects."""

    FIELDS: ClassVar[frozenset[str]]
    r"""The fields of the time series collection."""

    timeseries: T
    r"""The collection of time series data."""
    timeseries_metadata: Optional[T] = None
    r"""Data associated with each channel such as measurement device, unit, etc."""
    static_covariates: Optional[T] = None
    r"""The static covariates associated with each timeseries."""
    static_covariates_metadata: Optional[T] = None
    r"""Data associated with each metadata such as measurement device, unit,  etc."""
    constants: Optional[T] = None
    r"""Additional data that is independent of the metaindex."""
    constants_metadata: Optional[T] = None
    r"""Data associated with each global metadata such as measurement device, unit,  etc."""
    metadata: Optional[Metadata]
    r"""The metadata of the dataset."""

    @abstractmethod
    def __len__(self) -> int: ...
    @abstractmethod
    def __iter__(self) -> Iterator[Any]: ...
    @abstractmethod
    def __contains__(self, item: Any, /) -> bool: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: slice | list[Key], /) -> Self: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: Key, /) -> TimeSeries[T]: ...
