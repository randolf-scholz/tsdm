r"""Implementations of time series containers backed by Polars."""

__all__ = [
    "PolarsTS",
    "PolarsTSC",
]

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field, fields
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

    Polars does not have a dedicated row index. ``time_column`` identifies the
    timestamp column contained in ``timeseries``; ``timeindex`` is inferred
    from that row-aligned column.
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
    r"""The time series table, including its timestamp columns."""
    time_column: str
    r"""The timestamp column in ``timeseries``."""
    timeseries_metadata: pl.DataFrame | None = None
    r"""Data associated with the time series variables."""
    static_covariates: pl.DataFrame | None = None
    r"""Static covariates associated with the time series."""
    static_covariates_metadata: pl.DataFrame | None = None
    r"""Metadata associated with the static covariates."""

    timeindex: pl.Series = field(init=False)
    r"""The row-aligned timestamp column inferred from ``timeseries``."""

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

        object.__setattr__(self, "timeindex", self._infer_timeindex())

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
                mask = self.timeindex.is_in(labels.implode())

            case list(items):
                if items and all(isinstance(k, bool) for k in items):
                    if len(items) != self.timeseries.height:
                        raise ValueError(
                            "Expected boolean mask to have one entry per timeseries row,"
                            f" got {len(items)} and {self.timeseries.height}."
                        )
                    mask = pl.Series("mask", items)
                else:
                    labels = pl.Series(items)
                    self._check_keys(labels)
                    mask = self.timeindex.is_in(labels.implode())

            case scalar:
                labels = pl.Series([scalar])
                self._check_keys(labels)
                mask = self.timeindex.is_in(labels.implode())

        sliced = {name: getattr(self, name) for name in self.FIELDS - {"timeseries"}}
        return self.__class__(
            name=self.name,
            timeseries=self.timeseries.filter(mask),
            time_column=self.time_column,
            **sliced,
        )

    def _infer_timeindex(self) -> pl.Series:
        r"""Infer the row-aligned timestamp column from ``timeseries``."""
        return self.timeseries.get_column(self.time_column)

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
class PolarsTSC[KeyT](
    TimeSeriesCollection[KeyT, pl.DataFrame], Mapping[KeyT, PolarsTS[Any]]
):
    r"""A collection of time series backed by a Polars DataFrame.

    Polars does not have a dedicated row index. ``time_column`` and
    ``meta_columns`` identify the row-aligned timestamp and collection-key
    columns contained in ``timeseries``. Their pairing replaces Pandas'
    two-level ``MultiIndex``.
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
    r"""The collection's time series table, including its index columns."""
    time_column: str
    r"""The timestamp column in ``timeseries``."""
    meta_columns: list[str]
    r"""The collection-key columns in ``timeseries``."""
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

    timeindex: pl.Series = field(init=False)
    r"""The row-aligned timestamps inferred from ``timeseries``."""
    metaindex: pl.DataFrame = field(init=False)
    r"""The distinct collection identifiers inferred from ``timeseries``."""

    @cached_property
    def unique_timeindex(self) -> pl.Series:
        r"""Return distinct timestamps in order of appearance."""
        return self.timeindex.unique(maintain_order=True)

    def __post_init__(self) -> None:
        r"""Validate collection columns and infer its derived indices."""
        if not isinstance(self.timeseries, pl.DataFrame):
            raise TypeError(
                "Expected timeseries to be a polars DataFrame,"
                f" got {type(self.timeseries)}."
            )

        if not isinstance(self.time_column, str):
            raise TypeError(
                f"Expected time_column to be a string, got {self.time_column!r}."
            )

        if (
            not isinstance(self.meta_columns, list)
            or not self.meta_columns
            or not all(isinstance(column, str) for column in self.meta_columns)
            or len(set(self.meta_columns)) != len(self.meta_columns)
            or set(self.meta_columns).intersection({self.time_column})
        ):
            raise TypeError(
                f"Expected meta_columns to be a non-empty list of unique strings,"
                f" disjoint from time_column {self.time_column!r},"
                f" got {self.meta_columns!r}."
            )

        object.__setattr__(
            self, "timeindex", self.timeseries.get_column(self.time_column)
        )
        object.__setattr__(
            self,
            "metaindex",
            self.timeseries.select(*self.meta_columns).unique(maintain_order=True),
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
            if any(
                col not in self.static_covariates.columns for col in self.meta_columns
            ):
                raise ValueError(
                    "Expected static_covariates to contain all meta_columns"
                    f" {self.meta_columns}, got {self.static_covariates.columns}."
                )

        for f in fields(self):
            if getattr(self, f.name, UNDEFINED) is UNDEFINED:
                raise ValueError(f"The field '{f.name}' is undefined.")

    def __len__(self) -> int:
        r"""Return the number of distinct collection identifiers."""
        return self.metaindex.height

    def __iter__(self) -> Iterator[KeyT]:
        r"""Iterate over distinct collection identifiers in order of appearance."""
        match self.metaindex.width:
            case 1:
                return iter(self.metaindex.to_series())
            case _:
                return cast("Iterator[KeyT]", iter(self.metaindex.iter_rows()))

    def __contains__(self, key: object, /) -> bool:
        r"""Check whether a collection identifier is present."""
        match self.metaindex.width:
            case 1:
                return key in self.metaindex.to_series()
            case _:
                return any(key == value for value in self.metaindex.iter_rows())

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
                    if len(items) != len(self):
                        raise ValueError(
                            "Expected boolean mask to have one entry per collection,"
                            f" got {len(items)} and {len(self)}."
                        )
                    return self._subset(
                        [
                            label
                            for label, selected in zip(self, items, strict=True)
                            if selected
                        ]
                    )
                labels = cast("list[KeyT]", items)
                return self._subset(labels)

            case scalar:
                labels = [scalar]
                self._check_keys(labels)
                keys = self._key_frame(labels)
                return PolarsTS(
                    name=self.name,
                    timeseries=self.timeseries.join(
                        keys, on=self.meta_columns, how="semi"
                    ),
                    time_column=self.time_column,
                    timeseries_metadata=self.timeseries_metadata,
                    static_covariates=self._static_covariates_for(labels),
                    static_covariates_metadata=self.static_covariates_metadata,
                )

    def _check_keys(self, keys: Sequence[KeyT] | pl.Series, /) -> None:
        r"""Raise ``KeyError`` when a label is not present in an index."""
        if missing_keys := [key for key in keys if key not in self]:
            raise KeyError(missing_keys)

    def _static_covariates_for(
        self, keys: Sequence[KeyT] | pl.Series, /
    ) -> pl.DataFrame | None:
        r"""Select static covariates corresponding to collection identifiers."""
        if self.static_covariates is None:
            return None
        mask = pl.Series("mask", [key in keys for key in self])
        return self.static_covariates.filter(mask)

    def _subset(self, keys: Sequence[KeyT] | pl.Series, /) -> Self:
        r"""Return the collection restricted to ``keys``."""
        self._check_keys(keys)
        key_frame = self._key_frame(keys)
        sliced = {
            name: getattr(self, name)
            for name in self.FIELDS - {"timeseries", "static_covariates"}
        }
        return self.__class__(
            name=self.name,
            timeseries=self.timeseries.join(
                key_frame, on=self.meta_columns, how="semi"
            ),
            time_column=self.time_column,
            meta_columns=self.meta_columns,
            static_covariates=self._static_covariates_for(keys),
            **sliced,
        )

    def _key_frame(self, keys: Sequence[KeyT] | pl.Series, /) -> pl.DataFrame:
        r"""Convert collection keys to a table indexed by ``meta_columns``."""
        if not len(keys):
            return self.metaindex.head(0)
        match self.metaindex.width:
            case 1:
                return pl.DataFrame(
                    {self.meta_columns[0]: keys}, schema=self.metaindex.schema
                )
            case _:
                return pl.DataFrame(keys, schema=self.metaindex.schema, orient="row")

    def _select_slice(self, key: slice, /) -> pl.Series:
        r"""Resolve a label slice against an index, including its stop label."""
        if self.metaindex.width != 1:
            raise ValueError(
                "Cannot slice a multi-column metaindex. Use a list of keys instead."
            )

        index = self.metaindex.to_series()
        start = 0 if key.start is None else index.index_of(key.start)
        stop = len(index) if key.stop is None else index.index_of(key.stop)
        if start is None or stop is None:
            raise KeyError(key)
        return index[slice(start, stop + 1, key.step)]
