r"""Timeseries data structures and functions."""

__all__ = [
    # Protocols
    "TimeSeries",
    "TimeSeriesCollection",
    "RangeSelector",
]

from abc import abstractmethod
from collections.abc import Iterator
from typing import Any, ClassVar, Protocol, Self, overload

# TODO: Use generic slice in 3.15
type RangeSelector[T] = slice | list[T] | list[bool]


class TimeSeries[TimeT, TableT](Protocol):
    r"""Protocol for time series objects.

    Describes a single time series implemented as a Table-like object, indexed by time.

    Attributes:
        timeseries: The time series data.
        timeseries_metadata: Data associated with the time such as measurement device, unit, etc.
        static_covariates: The metadata of the dataset.
        static_covariates_metadata: Data associated with each metadata such as measurement device, unit, etc.
        timeindex: The time index of the time series.
    """

    FIELDS: ClassVar[frozenset[str]]
    r"""The fields of the time series."""

    # TODO: Use typing.ReadOnly (PEP 767)
    @property
    @abstractmethod
    def timeseries(self) -> TableT: ...
    @property
    @abstractmethod
    def timeseries_metadata(self) -> TableT | None: ...
    @property
    @abstractmethod
    def static_covariates(self) -> TableT | None: ...
    @property
    @abstractmethod
    def static_covariates_metadata(self) -> TableT | None: ...
    @property
    @abstractmethod
    def timeindex(self) -> Any: ...

    @abstractmethod
    def __len__(self) -> int: ...
    @abstractmethod
    def __iter__(self) -> Iterator[TimeT]: ...
    @abstractmethod
    def __contains__(self, key: TimeT, /) -> object: ...
    @abstractmethod
    def __getitem__(self, key: TimeT | RangeSelector[TimeT], /) -> Self: ...


class TimeSeriesCollection[KeyT, TableT](Protocol):
    r"""Protocol for time series collection objects.

    Attributes:
        timeseries: The collection of time series data.
        timeseries_metadata: Data associated with each channel such as measurement device, unit, etc.
        static_covariates: The static covariates associated with each time series.
        static_covariates_metadata: Data associated with each metadata such as measurement device, unit, etc.
        constants: Additional data that is independent of the metaindex.
        constants_metadata: Data associated with each global metadata such as measurement device, unit, etc.
        timeindex: The time index of the time series.
        metaindex: The meta index of the time series collection.

    """

    FIELDS: ClassVar[frozenset[str]]
    r"""The fields of the time series collection."""

    # TODO: Use typing.ReadOnly (PEP 767)
    @property
    @abstractmethod
    def timeseries(self) -> TableT: ...
    @property
    @abstractmethod
    def timeseries_metadata(self) -> TableT | None: ...
    @property
    @abstractmethod
    def static_covariates(self) -> TableT | None: ...
    @property
    @abstractmethod
    def static_covariates_metadata(self) -> TableT | None: ...
    @property
    @abstractmethod
    def constants(self) -> TableT | None: ...
    @property
    @abstractmethod
    def constants_metadata(self) -> TableT | None: ...
    @property
    @abstractmethod
    def timeindex(self) -> Any: ...
    @property
    @abstractmethod
    def metaindex(self) -> Any: ...

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
    def __getitem__[TimeT = Any](self, key: KeyT, /) -> TimeSeries[TimeT, TableT]: ...
