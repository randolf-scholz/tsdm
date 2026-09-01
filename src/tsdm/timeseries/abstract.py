r"""Timeseries data structures and functions."""

__all__ = [
    # Protocols
    "TimeSeries",
    "TimeSeriesCollection",
    "RangeSelector",
    "SplitTimeData",
    "MergedTimeData",
    "TripletTimeData",
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


class SplitTimeData[ArrayT](Protocol):
    r"""Protocol for forecasting requests.

    Attributes:
        context_times:     Float[..., $N], padded NaN, non-decreasing
        context_values:    Float[..., $N, D], padded NaN
        context_mask:      Bool[..., $N, D], padded False
        query_times:       Float[..., $K], padded NaN, non-decreasing
        query_mask:        Bool[..., $K, F],  padded False
        target_values:     Float[..., $K, F],  padded NaN
        static_covariates: Float[..., M],  padded NaN
    """

    # TODO: Use typing.ReadOnly (PEP 767)

    @property
    def context_times(self) -> ArrayT: ...
    @property
    def context_values(self) -> ArrayT: ...
    @property
    def context_mask(self) -> ArrayT: ...

    @property
    def query_times(self) -> ArrayT: ...
    @property
    def query_mask(self) -> ArrayT: ...
    @property
    def target_values(self) -> ArrayT | None: ...

    @property
    def static_covariates(self) -> ArrayT | None: ...


class MergedTimeData[ArrayT](Protocol):
    r"""Protocol for joint time representation.

    Attributes:
        timestamps:        Float[..., $T], padded NaN, non-decreasing
        context_mask:      Bool[..., $T, D], padded False
        context_values:    Float[..., $T, D], padded NaN
        query_mask:        Bool[..., $T, E], padded False
        target_values:     Float[..., $T, E], padded NaN
        static_covariates: Float[..., M], padded NaN
    """

    # TODO: Use typing.ReadOnly (PEP 767)

    @property
    def timestamps(self) -> ArrayT: ...

    @property
    def context_mask(self) -> ArrayT: ...
    @property
    def context_values(self) -> ArrayT: ...

    @property
    def query_mask(self) -> ArrayT: ...
    @property
    def target_values(self) -> ArrayT | None: ...

    @property
    def static_covariates(self) -> ArrayT | None: ...


class TripletTimeData[ArrayT](Protocol):
    r"""Protocol for triplet representation.

    Tall data format that stacks context and query data into a 3 column representation of
    (time, channel, value) triplets.

    Attributes:
        context_times:     Float[..., $X], padded NaN, non-decreasing
        context_channels:  Long[..., $X], padded -1
        context_values:    Float[..., $X], padded NaN
        query_times:       Float[..., $Q], padded NaN, non-decreasing
        query_channels:    Long[..., $Q], padded -1
        target_values:     Float[..., $Q], padded NaN
        static_covariates: Float[..., M], padded NaN
    """

    # TODO: Use typing.ReadOnly (PEP 767)

    @property
    def context_times(self) -> ArrayT: ...
    @property
    def context_channels(self) -> ArrayT: ...
    @property
    def context_values(self) -> ArrayT: ...

    @property
    def query_times(self) -> ArrayT: ...
    @property
    def query_channels(self) -> ArrayT: ...
    @property
    def target_values(self) -> ArrayT | None: ...

    @property
    def static_covariates(self) -> ArrayT | None: ...
