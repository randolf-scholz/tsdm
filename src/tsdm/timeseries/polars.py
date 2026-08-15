r"""Implementations of time series containers backed by Polars."""

__all__ = [
    "PolarsTS",
    "PolarsTSC",
]

from collections.abc import Iterator
from dataclasses import KW_ONLY, dataclass, fields
from functools import cached_property
from typing import Any, ClassVar, Self, cast, overload

import polars as pl

from tsdm.constants import UNDEFINED
from tsdm.pprint import pprint_repr

from .base import RangeSelector, TimeSeries, TimeSeriesCollection


@pprint_repr
@dataclass(frozen=True)
class PolarsTS[TimeT = Any](TimeSeries[pl.DataFrame, TimeT]):
    r"""A single time series backed by a Polars DataFrame.

    Polars does not have a dedicated row index. ``timeindex`` is therefore a
    row-aligned Series that carries the timestamps for ``timeseries``. It is
    intentionally kept separate from the data frame, although applications may
    also retain a timestamp column in ``timeseries``.
    """

    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "timeseries",
            "timeseries_metadata",
            "static_covariates",
            "static_covariates_metadata",
        }
    )
    r"""The essential time series fields."""

    name: str | None = None
    r"""The name of the time series."""

    _: KW_ONLY

    timeseries: pl.DataFrame
    r"""The time series values."""
    timeindex: pl.Series
    r"""The row-aligned time index."""
    timeseries_metadata: pl.DataFrame | None = None
    r"""Data associated with the time series variables."""
    static_covariates: pl.DataFrame | None = None
    r"""Static covariates associated with the time series."""
    static_covariates_metadata: pl.DataFrame | None = None
    r"""Metadata associated with the static covariates."""

    @cached_property
    def unique_timeindex(self) -> pl.Series:
        r"""Return distinct timestamps in order of appearance."""
        return self.timeindex.unique(maintain_order=True)

    def __post_init__(self) -> None:
        r"""Validate that the data and time index are aligned."""
        if not isinstance(self.timeseries, pl.DataFrame):
            raise TypeError(
                "Expected timeseries to be a polars DataFrame,"
                f" got {type(self.timeseries)}."
            )
        if not isinstance(self.timeindex, pl.Series):
            raise TypeError(
                f"Expected timeindex to be a polars Series, got {type(self.timeindex)}."
            )
        if self.timeseries.height != self.timeindex.len():
            raise ValueError(
                "Expected timeseries and timeindex to have the same length,"
                f" got {self.timeseries.height} and {self.timeindex.len()}."
            )

        for f in fields(self):
            if getattr(self, f.name, UNDEFINED) is UNDEFINED:
                raise ValueError(f"The field '{f.name}' is undefined.")

    def __len__(self) -> int:
        r"""Return the number of distinct timestamps."""
        return self.unique_timeindex.len()

    def __iter__(self) -> Iterator[TimeT]:
        r"""Iterate over distinct timestamps in order of appearance."""
        return iter(self.unique_timeindex)

    def __contains__(self, key: object, /) -> bool:
        r"""Check whether a timestamp is present."""
        return key in self.unique_timeindex

    def __getitem__(self, key: TimeT | RangeSelector[TimeT], /) -> Self:
        r"""Return the subset selected by timestamp labels.

        A scalar selects one timestamp, a list selects multiple timestamps, a
        boolean list selects rows, and a slice is resolved against the ordered
        distinct timestamps. Slice stop labels are included, matching
        ``pandas.DataFrame.loc``.
        """
        match key:
            case slice() as s:
                labels = self._select_slice(s)
                self._check_keys(labels)
                mask = self.timeindex.is_in(labels)

            case list(items):
                if items and all(isinstance(k, bool) for k in items):
                    if len(items) != self.timeseries.height:
                        raise ValueError(
                            "Expected boolean mask to have one entry per timeseries row,"
                            f" got {len(items)} and {self.timeseries.height}."
                        )
                    mask = pl.Series("mask", items)
                else:
                    labels = cast("list[TimeT]", items)
                    self._check_keys(labels)
                    mask = self.timeindex.is_in(labels)

            case scalar:
                labels = [scalar]
                self._check_keys(labels)
                mask = self.timeindex.is_in(labels)

        sliced = {name: getattr(self, name) for name in self.FIELDS - {"timeseries"}}
        return self.__class__(
            name=self.name,
            timeseries=self.timeseries.filter(mask),
            timeindex=self.timeindex.filter(mask),
            **sliced,
        )

    def _check_keys(self, keys: list[TimeT] | pl.Series, /) -> None:
        r"""Raise ``KeyError`` when a label is not present in an index."""
        if missing_keys := [key for key in keys if key not in self.unique_timeindex]:
            raise KeyError(missing_keys)

    def _select_slice(self, key: slice, /) -> pl.Series:
        r"""Resolve a label slice against an index, including its stop label."""
        index = self.unique_timeindex
        start = 0 if key.start is None else index.index_of(key.start)
        stop = len(index) if key.stop is None else index.index_of(key.stop)
        if start is None or stop is None:
            raise KeyError(key)
        return index[slice(start, stop + 1, key.step)]


@pprint_repr
@dataclass(frozen=True)
class PolarsTSC[KeyT](TimeSeriesCollection[KeyT, pl.DataFrame]):
    r"""A collection of time series backed by a Polars DataFrame.

    ``timeindex`` and ``metaindex`` are row-aligned with ``timeseries``. The
    former contains timestamps and the latter contains collection identifiers.
    Their pairing replaces Pandas' two-level ``MultiIndex``.
    """

    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "timeseries",
            "timeseries_metadata",
            "static_covariates",
            "static_covariates_metadata",
            "constants",
            "constants_metadata",
        }
    )
    r"""The essential time series collection fields."""

    name: str | None = None
    r"""The name of the collection."""

    _: KW_ONLY

    timeseries: pl.DataFrame
    r"""The collection's time series values."""
    timeindex: pl.Series
    r"""The row-aligned timestamps."""
    metaindex: pl.Series
    r"""The row-aligned collection identifiers."""
    timeseries_metadata: pl.DataFrame | None = None
    r"""Data associated with the time series variables."""
    static_covariates: pl.DataFrame | None = None
    r"""Static covariates, ordered like the distinct metaindex values."""
    static_covariates_metadata: pl.DataFrame | None = None
    r"""Metadata associated with the static covariates."""
    constants: pl.DataFrame | None = None
    r"""Data independent of the collection identifiers."""
    constants_metadata: pl.DataFrame | None = None
    r"""Metadata associated with the constants."""

    @cached_property
    def unique_timeindex(self) -> pl.Series:
        r"""Return distinct timestamps in order of appearance."""
        return self.timeindex.unique(maintain_order=True)

    @cached_property
    def unique_metaindex(self) -> pl.Series:
        r"""Return distinct collection identifiers in order of appearance."""
        return self.metaindex.unique(maintain_order=True)

    def __post_init__(self) -> None:
        r"""Validate that the data and its two index Series are aligned."""
        if not isinstance(self.timeseries, pl.DataFrame):
            raise TypeError(
                "Expected timeseries to be a polars DataFrame,"
                f" got {type(self.timeseries)}."
            )
        if not isinstance(self.timeindex, pl.Series):
            raise TypeError(
                f"Expected timeindex to be a polars Series, got {type(self.timeindex)}."
            )
        if not isinstance(self.metaindex, pl.Series):
            raise TypeError(
                f"Expected metaindex to be a polars Series, got {type(self.metaindex)}."
            )

        lengths = {
            "timeseries": self.timeseries.height,
            "timeindex": self.timeindex.len(),
            "metaindex": self.metaindex.len(),
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(
                "Expected timeseries, timeindex, and metaindex to have the same"
                f" length, got {lengths}."
            )

        if self.static_covariates is not None:
            if not isinstance(self.static_covariates, pl.DataFrame):
                raise TypeError(
                    "Expected static_covariates to be a polars DataFrame or None,"
                    f" got {type(self.static_covariates)}."
                )
            if self.static_covariates.height != len(self):
                raise ValueError(
                    "Expected static_covariates to have one row per collection,"
                    f" got {self.static_covariates.height} and {len(self)}."
                )

        for f in fields(self):
            if getattr(self, f.name, UNDEFINED) is UNDEFINED:
                raise ValueError(f"The field '{f.name}' is undefined.")

    def __len__(self) -> int:
        r"""Return the number of distinct collection identifiers."""
        return self.unique_metaindex.len()

    def __iter__(self) -> Iterator[KeyT]:
        r"""Iterate over distinct collection identifiers in order of appearance."""
        return iter(self.unique_metaindex)

    def __contains__(self, key: object, /) -> bool:
        r"""Check whether a collection identifier is present."""
        return key in self.unique_metaindex

    @overload
    def __getitem__(self, key: RangeSelector[KeyT], /) -> Self: ...
    @overload
    def __getitem__[TimeT = Any](self, key: KeyT, /) -> PolarsTS[TimeT]: ...
    def __getitem__(self, key: KeyT | RangeSelector[KeyT], /) -> PolarsTS[Any] | Self:
        r"""Select one time series or a subset of the collection.

        Scalar keys return a :class:`PolarsTS`; lists and slices return a
        :class:`PolarsTSC`. A boolean list selects collection identifiers by
        their order in the metaindex.
        """
        match key:
            case slice() as s:
                labels = self._select_slice(s)
                return self._subset(labels)

            case list(items):
                if items and all(isinstance(k, bool) for k in items):
                    # mask branch.
                    if len(items) != self.unique_metaindex.len():
                        raise ValueError(
                            "Expected boolean mask to have one entry per collection,"
                            f" got {len(items)} and {self.unique_metaindex.len()}."
                        )
                    return self._subset(
                        self.unique_metaindex.filter(pl.Series("mask", items))
                    )
                labels = cast("list[KeyT]", items)
                return self._subset(labels)

            case scalar:
                labels = [scalar]
                self._check_keys(labels)
                mask = self.metaindex.is_in(labels)
                return PolarsTS(
                    name=self.name,
                    timeseries=self.timeseries.filter(mask),
                    timeindex=self.timeindex.filter(mask),
                    timeseries_metadata=self.timeseries_metadata,
                    static_covariates=self._static_covariates_for(labels),
                    static_covariates_metadata=self.static_covariates_metadata,
                )

    def _check_keys(self, keys: list[Any] | pl.Series, /) -> None:
        r"""Raise ``KeyError`` when a label is not present in an index."""
        if missing_keys := [key for key in keys if key not in self.unique_metaindex]:
            raise KeyError(missing_keys)

    def _static_covariates_for(
        self, keys: list[KeyT] | pl.Series, /
    ) -> pl.DataFrame | None:
        r"""Select static covariates corresponding to collection identifiers."""
        if self.static_covariates is None:
            return None
        mask = self.unique_metaindex.is_in(pl.Series(keys).implode())
        return self.static_covariates.filter(mask)

    def _subset(self, keys: list[KeyT] | pl.Series, /) -> Self:
        r"""Return the collection restricted to ``keys``."""
        self._check_keys(keys)
        mask = self.metaindex.is_in(pl.Series(keys).implode())
        sliced = {
            name: getattr(self, name)
            for name in self.FIELDS - {"timeseries", "static_covariates"}
        }
        return self.__class__(
            name=self.name,
            timeseries=self.timeseries.filter(mask),
            timeindex=self.timeindex.filter(mask),
            metaindex=self.metaindex.filter(mask),
            static_covariates=self._static_covariates_for(keys),
            **sliced,
        )

    def _select_slice(self, key: slice, /) -> pl.Series:
        r"""Resolve a label slice against an index, including its stop label."""
        index = self.unique_metaindex
        start = 0 if key.start is None else index.index_of(key.start)
        stop = len(index) if key.stop is None else index.index_of(key.stop)
        if start is None or stop is None:
            raise KeyError(key)
        return index[slice(start, stop + 1, key.step)]
