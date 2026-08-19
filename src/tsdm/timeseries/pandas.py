r"""Implementation of TimeSeries and TimeSeriesCollection using Pandas DataFrames."""

__all__ = [
    # Constants
    "TIMESERIES",
    "TIMESERIES_COLLECTIONS",
    # Classes
    "PandasTS",
    "PandasTSC",
    # Concrete classes
    "beijing_air_quality",
    "damped_pendulum_ansari2023",
    "electricity",
    "etth1",
    "etth2",
    "ettm1",
    "ettm2",
    "in_silico",
    "kiwi_benchmark",
    "mimic_iv_bilos2021",
    "physionet2012",
    "physionet2019",
    "traffic",
    "ushcn",
    "ushcn_de_brouwer2019",
]

import warnings
from collections.abc import Callable as Fn, Iterator, Mapping
from dataclasses import KW_ONLY, asdict, dataclass, field, fields
from typing import Any, ClassVar, Self, cast, overload

from pandas import DataFrame, Index, MultiIndex, Series

from tsdm import datasets
from tsdm.constants import UNDEFINED
from tsdm.datasets import Dataset
from tsdm.pprint import pprint_repr

from .base import RangeSelector, TimeSeries, TimeSeriesCollection


@pprint_repr
@dataclass(frozen=True)
class PandasTS[TimeT = Any](TimeSeries[DataFrame, TimeT]):
    r"""Abstract Base Class for TimeSeriesDatasets.

    A TimeSeriesDataset is a dataset that contains time series data and metadata.
    More specifically, it is a tuple (TS, M) where TS is a time series and M is the metadata.

    For a given time-index, the time series data is a vector of measurements.
    """

    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "timeseries",
            "timeseries_metadata",
            "static_covariates",
            "static_covariates_metadata",
        }
    )
    r"""The essential fields of the time series collection."""

    name: str | None = None
    r"""The name of the dataset."""

    _: KW_ONLY

    # Main Attributes
    timeseries: DataFrame
    r"""The time series data."""
    timeseries_metadata: DataFrame | None = None
    r"""Data associated with the time such as measurement device, unit, etc."""
    static_covariates: DataFrame | None = None
    r"""The metadata of the dataset."""
    static_covariates_metadata: DataFrame | None = None
    r"""Data associated with each metadata such as measurement device, unit,  etc."""

    # derived fields
    timeindex: Series = field(init=False)
    r"""The timestamps that index the time series."""

    @classmethod
    def from_dataset(cls, arg: Dataset | type[Dataset], /) -> Self:
        r"""Create a TimeSeries from a Dataset."""
        ds = arg() if isinstance(arg, type) else arg

        if unknown_fields := set(ds.table_names) - cls.FIELDS:
            raise ValueError(f"The following tables: {unknown_fields}")

        return cls(
            **{k: ds.tables.get(k, None) for k in cls.FIELDS},
            name=ds.__class__.__name__,
        )

    def __post_init__(self) -> None:
        r"""Post init."""
        if not isinstance(self.timeseries, DataFrame):
            raise TypeError(
                f"Expected timeseries to be a pandas DataFrame,"
                f" got {type(self.timeseries)}."
            )

        object.__setattr__(self, "timeindex", self._infer_timeindex())

        # ensure no dataclass fields are undefined
        for f in fields(self):
            if getattr(self, f.name, UNDEFINED) is UNDEFINED:
                raise ValueError(f"The field '{f.name}' is undefined.")

    def __len__(self) -> int:
        r"""Return the number of timestamps."""
        return len(self.timeindex)

    def __iter__(self) -> Iterator[TimeT]:
        r"""Iterate over the timestamps."""
        return iter(self.timeindex)

    def __contains__(self, key: object, /) -> bool:
        r"""Check if the key is in the timeindex."""
        return key in self.timeindex

    def __getitem__(self, key: TimeT | RangeSelector[TimeT], /) -> Self:
        r"""Return the subset of the timeseries at index `key`."""
        match key:
            case slice() as s:
                labels = self._select_slice(s)
                self._check_keys(labels)
                timeseries = self.timeseries.loc[labels]

            case list(items):
                if items and all(isinstance(k, bool) for k in items):
                    if len(items) != self.timeseries.shape[0]:
                        raise ValueError(
                            "Expected boolean mask to have one entry per timeseries row,"
                            f" got {len(items)} and {self.timeseries.shape[0]}."
                        )
                    timeseries = self.timeseries.loc[items]
                else:
                    labels = cast("list[TimeT]", items)
                    self._check_keys(labels)
                    timeseries = self.timeseries.loc[labels]

            case scalar:
                labels = [scalar]
                self._check_keys(labels)
                timeseries = self.timeseries.loc[labels]

        sliced = (  # formatting
            {k: v for k, v in asdict(self).items() if k in self.FIELDS}
            | {"timeseries": timeseries}
        )
        return self.__class__(**sliced)

    def _check_keys(self, keys: list[TimeT] | Index, /) -> None:
        r"""Raise ``KeyError`` when a label is not present in the time index."""
        if missing_keys := [key for key in keys if key not in self.timeindex]:
            raise KeyError(missing_keys)

    def _select_slice(self, key: slice, /) -> Index:
        r"""Resolve a label slice against an index, including its stop label."""
        index = self.timeindex.index
        start = 0 if key.start is None else index.get_indexer_for([key.start]).item()
        stop = (
            len(index) - 1
            if key.stop is None
            else index.get_indexer_for([key.stop]).item()
        )
        if start == -1 or stop == -1:
            raise KeyError(key)
        return index[slice(start, stop + 1, key.step)]

    def _infer_timeindex(self) -> Series:
        r"""Get timestamps indexed by the canonical time index."""
        index = self.timeseries.index.copy()
        if isinstance(index, MultiIndex):
            raise TypeError(
                "Tried to create a TimeSeries from a DataFrame with MultiIndex."
                "\n    Are you sure this is not a TimeSeriesCollection?"
            )
        return Series(index, index=index, name=index.name)


@pprint_repr
@dataclass(frozen=True)
class PandasTSC[KeyT](
    TimeSeriesCollection[KeyT, DataFrame], Mapping[KeyT, PandasTS[Any]]
):
    r"""Class for **equimodal** TimeSeriesCollections.

    A `TimeSeriesCollection` is a collection of `TimeSeries` objects.
    `Equimodal` means that all time series share the same schema (i.e. subset of variables).
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
    r"""The essential fields of the time series collection."""

    name: str | None = None
    r"""The name of the collection."""

    _: KW_ONLY

    # Main attributes
    timeseries: DataFrame
    r"""The collection of time series data."""
    timeseries_metadata: DataFrame | None = None
    r"""Data associated with each channel such as measurement device, unit, etc."""
    static_covariates: DataFrame | None = None
    r"""The static covariates associated with each timeseries."""
    static_covariates_metadata: DataFrame | None = None
    r"""Data associated with each metadata such as measurement device, unit,  etc."""
    constants: DataFrame | None = None
    r"""Additional data that is independent of the metaindex."""
    constants_metadata: DataFrame | None = None
    r"""Data associated with each global metadata such as measurement device, unit,  etc."""

    # derived fields
    timeindex: Series = field(init=False)
    r"""The row-aligned timestamps indexed by their collection keys."""
    metaindex: Index = field(init=False)
    r"""The index of the collection."""

    @classmethod
    def from_dataset(cls, arg: Dataset | type[Dataset], /) -> Self:
        r"""Create a TimeSeries from a Dataset."""
        ds = arg() if isinstance(arg, type) else arg

        if superfluous_names := (set(ds.table_names) - cls.FIELDS):
            warnings.warn(
                f"The following tables are skipped: {superfluous_names}",
                UserWarning,
                stacklevel=2,
            )

        return cls(ds.name, **{k: ds.tables.get(k, None) for k in cls.FIELDS})

    def __post_init__(self) -> None:
        r"""Post init."""
        if not isinstance(self.timeseries, DataFrame):
            raise TypeError(
                f"Expected timeseries to be a pandas DataFrame,"
                f" got {type(self.timeseries)}."
            )

        object.__setattr__(self, "timeindex", self._infer_timeindex())
        object.__setattr__(self, "metaindex", self._infer_metaindex())

        # ensure that the index of the static covariates is a subset of the metaindex
        self._validate_static_covariates()

        # ensure no dataclass fields are undefined
        for f in fields(self):
            if getattr(self, f.name, UNDEFINED) is UNDEFINED:
                raise ValueError(f"The field '{f.name}' is undefined.")

    def __len__(self) -> int:
        r"""Get the number of timeseries in the collection."""
        return len(self.metaindex)

    def __iter__(self) -> Iterator[Any]:
        r"""Iterate over the timeseries in the collection."""
        return iter(self.metaindex)

    def __contains__(self, key: object, /) -> bool:
        r"""Check if the key is in the metaindex."""
        return key in self.metaindex

    def _infer_timeindex(self) -> Series:
        r"""Get timestamps indexed by row-aligned collection keys."""
        index = self.timeseries.index.copy()
        if not isinstance(index, MultiIndex):
            raise TypeError("Expected a timeseries with MultiIndex.")
        return Series(
            index.get_level_values(-1),
            index=index.droplevel(-1),
            name=index.names[-1],
        )

    def _infer_metaindex(self) -> Index:
        r"""Get the metaindex."""
        return self.timeindex.index.unique()

    def _validate_static_covariates(self) -> None:
        r"""Ensure that the static covariates index is a subset of the metaindex."""
        match self.static_covariates:
            case None: ...  # fmt: skip
            case DataFrame() as static_cov:
                superfluous_keys = static_cov.index.difference(self.metaindex)
                missing_keys = self.metaindex.difference(static_cov.index)
                if not superfluous_keys.empty:
                    warnings.warn(
                        f"The static_covariates contains unused keys:"
                        f" {superfluous_keys.tolist()}",
                        UserWarning,
                        stacklevel=2,
                    )
                if not missing_keys.empty:
                    warnings.warn(
                        f"No static covariates present for some keys:"
                        f" {missing_keys.tolist()}",
                        UserWarning,
                        stacklevel=2,
                    )
            case _:
                raise TypeError(
                    f"Expected static_covariates to be a pandas DataFrame or None,"
                    f" got {type(self.static_covariates)}."
                )

    @overload
    def __getitem__(self, key: RangeSelector[KeyT], /) -> Self: ...
    @overload
    def __getitem__[TimeT = Any](self, key: KeyT, /) -> PandasTS[TimeT]: ...
    def __getitem__(self, key: KeyT | RangeSelector[KeyT], /) -> PandasTS[Any] | Self:
        r"""Get the timeseries and metadata of the dataset at index `key`."""
        match key:
            case slice() as s:
                labels = self._select_slice(s)
                return self._subset(labels)

            case list(items):
                if items and all(isinstance(k, bool) for k in items):
                    if len(items) != len(self.metaindex):
                        raise ValueError(
                            "Expected boolean mask to have one entry per collection,"
                            f" got {len(items)} and {len(self.metaindex)}."
                        )
                    return self._subset(self.metaindex[items])
                labels = cast("list[KeyT]", items)
                return self._subset(labels)

            case scalar:
                labels = [scalar]
                self._check_keys(labels)
                timeseries = self.timeseries.loc[scalar]
                return PandasTS(
                    name=self.name,
                    timeseries=timeseries,
                    timeseries_metadata=self.timeseries_metadata,
                    static_covariates=self._static_covariates_for(labels),
                    static_covariates_metadata=self.static_covariates_metadata,
                )

    def _check_keys(self, keys: list[Any] | Index, /) -> None:
        r"""Raise ``KeyError`` when a label is not present in the meta index."""
        if missing_keys := [key for key in keys if key not in self.metaindex]:
            raise KeyError(missing_keys)

    def _static_covariates_for(self, keys: list[KeyT] | Index, /) -> DataFrame | None:
        r"""Select static covariates corresponding to collection identifiers."""
        if self.static_covariates is None:
            return None
        return self.static_covariates.loc[keys]

    def _subset(self, keys: list[KeyT] | Index, /) -> Self:
        r"""Return the collection restricted to ``keys``."""
        self._check_keys(keys)
        sliced = (  # formatting
            {k: v for k, v in asdict(self).items() if k in self.FIELDS}
            | {
                "timeseries": self.timeseries.loc[keys],
                "static_covariates": self._static_covariates_for(keys),
            }
        )
        return self.__class__(**sliced)

    def _select_slice(self, key: slice, /) -> Index:
        r"""Resolve a label slice against an index, including its stop label."""
        index = self.metaindex
        start = 0 if key.start is None else index.get_indexer_for([key.start]).item()
        stop = (
            len(index) - 1
            if key.stop is None
            else index.get_indexer_for([key.stop]).item()
        )
        if start == -1 or stop == -1:
            raise KeyError(key)
        return index[slice(start, stop + 1, key.step)]


class electricity(PandasTS[Any]):
    r"""The Electricity dataset wrapped as a pandas time series."""

    timeseries: DataFrame
    timeseries_metadata: None
    static_covariates: None
    static_covariates_metadata: None

    def __init__(self) -> None:
        ds = datasets.Electricity(initialize=False)
        super().__init__(
            ds.name, timeseries=ds.timeseries.to_pandas().set_index("time")
        )


class traffic(PandasTS[Any]):
    r"""The Traffic dataset wrapped as a pandas time series."""

    timeseries: DataFrame
    timeseries_metadata: None
    static_covariates: None
    static_covariates_metadata: None

    def __init__(self) -> None:
        ds = datasets.Traffic(initialize=False)
        super().__init__(
            ds.name, timeseries=ds.timeseries.to_pandas().set_index("time")
        )


class etth1(PandasTS[Any]):
    r"""The ETTh1 dataset wrapped as a pandas time series."""

    timeseries: DataFrame
    timeseries_metadata: None
    static_covariates: None
    static_covariates_metadata: None

    def __init__(self) -> None:
        ds = datasets.ETT(initialize=False)
        super().__init__("ETTh1", timeseries=ds["ETTh1"].to_pandas().set_index("date"))


class etth2(PandasTS[Any]):
    r"""The ETTh2 dataset wrapped as a pandas time series."""

    timeseries: DataFrame
    timeseries_metadata: None
    static_covariates: None
    static_covariates_metadata: None

    def __init__(self) -> None:
        ds = datasets.ETT(initialize=False)
        super().__init__("ETTh2", timeseries=ds["ETTh2"].to_pandas().set_index("date"))


class ettm1(PandasTS[Any]):
    r"""The ETTm1 dataset wrapped as a pandas time series."""

    timeseries: DataFrame
    timeseries_metadata: None
    static_covariates: None
    static_covariates_metadata: None

    def __init__(self) -> None:
        ds = datasets.ETT(initialize=False)
        super().__init__("ETTm1", timeseries=ds["ETTm1"].to_pandas().set_index("date"))


class ettm2(PandasTS[Any]):
    r"""The ETTm2 dataset wrapped as a pandas time series."""

    timeseries: DataFrame
    timeseries_metadata: None
    static_covariates: None
    static_covariates_metadata: None

    def __init__(self) -> None:
        ds = datasets.ETT(initialize=False)
        super().__init__("ETTm2", timeseries=ds["ETTm2"].to_pandas().set_index("date"))


class beijing_air_quality(PandasTSC[str]):
    r"""The Beijing Air Quality dataset wrapped as a pandas collection."""

    timeseries: DataFrame
    timeseries_metadata: DataFrame
    static_covariates: None
    static_covariates_metadata: None
    constants: None
    constants_metadata: None

    def __init__(self) -> None:
        ds = datasets.BeijingAirQuality(initialize=False)
        super().__init__(
            ds.name,
            timeseries=ds.timeseries.to_pandas().set_index(["station", "time"]),
            timeseries_metadata=(
                ds.timeseries_metadata.to_pandas()
                .set_index("variable")
                .drop(index=["station", "time"])
            ),
        )


class in_silico(PandasTSC[int]):
    r"""The in silico dataset wrapped as a pandas collection."""

    timeseries: DataFrame
    timeseries_metadata: DataFrame
    static_covariates: None
    static_covariates_metadata: None
    constants: None
    constants_metadata: None

    def __init__(self) -> None:
        ds = datasets.InSilico(initialize=False)
        super().__init__(
            ds.name,
            timeseries=ds.timeseries.to_pandas().set_index(["run_id", "time"]),
            timeseries_metadata=(
                ds.timeseries_metadata.to_pandas()
                .set_index("variable")
                .drop(index=["run_id", "time"])
            ),
        )


class kiwi_benchmark(PandasTSC[tuple[int, int]]):
    r"""The KIWI dataset wrapped as a pandas collection."""

    timeseries: DataFrame
    timeseries_metadata: DataFrame
    static_covariates: DataFrame
    static_covariates_metadata: DataFrame
    constants: None
    constants_metadata: None

    def __init__(self) -> None:
        ds = datasets.KiwiBenchmark(initialize=False)
        super().__init__(
            ds.name,
            timeseries=ds.timeseries.to_pandas().set_index(
                ["run_id", "experiment_id", "elapsed_time"]
            ),
            timeseries_metadata=ds.timeseries_metadata.to_pandas().set_index("name"),
            static_covariates=ds.static_covariates.to_pandas().set_index(
                ["run_id", "experiment_id"]
            ),
            static_covariates_metadata=(
                ds.static_covariates_metadata.to_pandas().set_index("name")
            ),
        )


class ushcn(PandasTSC[int]):
    r"""The USHCN dataset wrapped as a pandas collection."""

    timeseries: DataFrame
    timeseries_metadata: DataFrame
    static_covariates: DataFrame
    static_covariates_metadata: DataFrame
    constants: None
    constants_metadata: None

    def __init__(self) -> None:
        ds = datasets.USHCN(initialize=False)
        super().__init__(
            ds.name,
            timeseries=ds.timeseries.to_pandas().set_index(["COOP_ID", "DATE"]),
            timeseries_metadata=(
                ds.timeseries_metadata.to_pandas()
                .set_index("variable")
                .drop(index=["COOP_ID", "DATE"])
            ),
            static_covariates=ds.static_covariates.to_pandas().set_index("COOP_ID"),
            static_covariates_metadata=(
                ds.static_covariates_metadata.to_pandas()
                .set_index("variable")
                .drop(index="COOP_ID")
            ),
        )


class ushcn_de_brouwer2019(PandasTSC[int]):
    r"""The USHCN_DeBrouwer2019 dataset wrapped as a pandas collection."""

    timeseries: DataFrame
    timeseries_metadata: None
    static_covariates: None
    static_covariates_metadata: None
    constants: None
    constants_metadata: None

    def __init__(self) -> None:
        ds = datasets.USHCN_DeBrouwer2019(initialize=False)
        super().__init__(
            ds.name,
            timeseries=ds.timeseries.to_pandas().set_index(["ID", "Time"]),
        )


class physionet2012(PandasTSC[int]):
    r"""The PhysioNet2012 dataset wrapped as a pandas collection."""

    timeseries: DataFrame
    timeseries_metadata: DataFrame
    static_covariates: DataFrame
    static_covariates_metadata: DataFrame
    constants: None
    constants_metadata: None

    def __init__(self) -> None:
        ds = datasets.PhysioNet2012(initialize=False)
        super().__init__(
            ds.name,
            timeseries=ds.timeseries.to_pandas().set_index(["RecordID", "Time"]),
            timeseries_metadata=(
                ds.timeseries_metadata.to_pandas()
                .set_index("variable")
                .drop(index=["RecordID", "Time"])
            ),
            static_covariates=ds.static_covariates.to_pandas().set_index("RecordID"),
            static_covariates_metadata=(
                ds.static_covariates_metadata.to_pandas()
                .set_index("variable")
                .drop(index="RecordID")
            ),
        )


class physionet2019(PandasTSC[int]):
    r"""The PhysioNet2019 dataset wrapped as a pandas collection."""

    timeseries: DataFrame
    timeseries_metadata: DataFrame
    static_covariates: DataFrame
    static_covariates_metadata: DataFrame
    constants: None
    constants_metadata: None

    def __init__(self) -> None:
        ds = datasets.PhysioNet2019(initialize=False)
        super().__init__(
            ds.name,
            timeseries=ds.timeseries.to_pandas().set_index(["patient", "time"]),
            timeseries_metadata=(
                ds.timeseries_metadata.to_pandas()
                .set_index("variable")
                .drop(index=["patient", "time"])
            ),
            static_covariates=ds.static_covariates.to_pandas().set_index("patient"),
            static_covariates_metadata=(
                ds.static_covariates_metadata.to_pandas()
                .set_index("variable")
                .drop(index="patient")
            ),
        )


class mimic_iv_bilos2021(PandasTSC[int]):
    r"""The MIMIC_IV_Bilos2021 dataset wrapped as a pandas collection."""

    timeseries: DataFrame
    timeseries_metadata: None
    static_covariates: None
    static_covariates_metadata: None
    constants: None
    constants_metadata: None

    def __init__(self) -> None:
        ds = datasets.MIMIC_IV_Bilos2021(initialize=False)
        super().__init__(
            ds.name,
            timeseries=ds.timeseries.to_pandas().set_index(["hadm_id", "time_stamp"]),
        )


class damped_pendulum_ansari2023(PandasTSC[int]):
    r"""The DampedPendulum_Ansari2023 dataset wrapped as a pandas collection."""

    timeseries: DataFrame
    timeseries_metadata: DataFrame
    static_covariates: None
    static_covariates_metadata: None
    constants: None
    constants_metadata: None

    def __init__(self) -> None:
        ds = datasets.DampedPendulum_Ansari2023(initialize=False)
        super().__init__(
            ds.name,
            timeseries=ds.timeseries.to_pandas().set_index(["sequence_id", "time"]),
            timeseries_metadata=(
                ds.timeseries_metadata.to_pandas()
                .set_index("variable")
                .drop(index=["sequence_id", "time"])
            ),
        )


TIMESERIES: dict[str, Fn[[], TimeSeries[DataFrame]]] = {
    "ETTh1"       : etth1,
    "ETTh2"       : etth2,
    "ETTm1"       : ettm1,
    "ETTm2"       : ettm2,
    "Electricity" : electricity,
    "Traffic"     : traffic,
}  # fmt: skip
r"""Dictionary of all available time series datasets."""

TIMESERIES_COLLECTIONS: dict[str, Fn[[], TimeSeriesCollection[Any, DataFrame]]] = {
    "DampedPendulum_Ansari2023" : damped_pendulum_ansari2023,
    "InSilico"                  : in_silico,
    "KiwiBenchmark"             : kiwi_benchmark,
    "MIMIC_IV_Bilos2021"        : mimic_iv_bilos2021,
    "PhysioNet2012"             : physionet2012,
    "PhysioNet2019"             : physionet2019,
    "USHCN"                     : ushcn,
    "BeijingAirQuality"         : beijing_air_quality,
    "USHCN_DeBrouwer2019"       : ushcn_de_brouwer2019,
}  # fmt: skip
r"""Dictionary of all available time series collections."""
