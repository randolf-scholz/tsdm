r"""Backend-independent interfaces for time-series containers and samples.

The protocols in this module specify the common structure of a single time
series, a collection of time series, and prediction samples. Concrete
backends use these contracts while choosing their own table, index, and tensor
types.
"""

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

type RangeSelector[T] = slice[T] | list[T] | list[bool]


class TimeSeries[TimeT, TableT](Protocol):
    r"""Protocol for time series objects.

    Describes a single time series implemented as a table indexed by time.

    Attributes:
        timeseries: The time series data.
        timeseries_metadata: Metadata for each measurement channel, such as its device or unit.
        static_covariates: Static covariates associated with the time series.
        static_covariates_metadata: Metadata describing each static covariate.
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
        timeseries_metadata: Metadata for each measurement channel, such as its device or unit.
        static_covariates: The static covariates associated with each time series.
        static_covariates_metadata: Metadata describing each static covariate.
        constants: Additional data that is independent of the metaindex.
        constants_metadata: Metadata describing the collection-wide constants.
        timeindex: The time index of the time series.
        metaindex: The index of series identifiers in the collection.
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
    r"""Protocol for a prediction sample with separate context and query time axes.

    Leading dimensions represent optional batch axes. A `$` prefix marks a
    dynamic dimension that may vary between samples. `$N` and `$K` are the
    context and query lengths; `D` and `F` are their channel counts, and `M`
    is the number of static covariates. Within each sample, timestamps are
    non-decreasing. Padded values and timestamps use `NaN`; masks use `False`.

    Attributes:
        context_times: Timestamps of type `Float[..., $N]`.
        context_values: Context values of type `Float[..., $N, D]`.
        context_mask: Availability mask of type `Bool[..., $N, D]`.
        query_times: Query timestamps of type `Float[..., $K]`.
        query_mask: Target mask of type `Bool[..., $K, F]`.
        target_values: Targets of type `Float[..., $K, F]`, or `None`.
        static_covariates: Time-independent variables of type `Float[..., M]`, or `None`.
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
    r"""Protocol for a prediction sample with a shared context and query time axis.

    Leading dimensions represent optional batch axes. A `$` prefix marks a
    dynamic dimension that may vary between samples. `$T` is the shared
    sequence length, `D` and `E` are the context and target channel counts,
    and `M` is the number of static covariates. Within each sample, timestamps
    are non-decreasing. Padded values and timestamps use `NaN`; masks use `False`.

    Attributes:
        timestamps: Timestamps of type `Float[..., $T]`.
        context_mask: Availability mask of type `Bool[..., $T, D]`.
        context_values: Context values of type `Float[..., $T, D]`.
        query_mask: Target mask of type `Bool[..., $T, E]`.
        target_values: Targets of type `Float[..., $T, E]`, or `None`.
        static_covariates: Time-independent variables of type `Float[..., M]`, or `None`.
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
    r"""Protocol for a prediction sample represented by time, channel, and value triplets.

    Leading dimensions represent optional batch axes. A `$` prefix marks a
    dynamic dimension that may vary between samples. `$X` and `$Q` are the
    numbers of context and query triplets, and `M` is the number of static
    covariates. Within each sample, timestamps are non-decreasing. Padded
    values and timestamps use `NaN`; channel indices use `-1`.

    Attributes:
        context_times: Timestamps of type `Float[..., $X]`.
        context_channels: Channel indices of type `Long[..., $X]`.
        context_values: Context values of type `Float[..., $X]`.
        query_times: Query timestamps of type `Float[..., $Q]`.
        query_channels: Channel indices of type `Long[..., $Q]`.
        target_values: Targets of type `Float[..., $Q]`, or `None`.
        static_covariates: Time-independent variables of type `Float[..., M]`, or `None`.
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
