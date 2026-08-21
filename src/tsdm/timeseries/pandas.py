r"""Implementation of TimeSeries and TimeSeriesCollection using Pandas DataFrames."""

# mypy: disable-error-code="overload-overlap"

__all__ = [
    # Constants
    "TIMESERIES",
    "TIMESERIES_COLLECTIONS",
    # Classes
    "PandasTS",
    "PandasTSC",
    "Sample",
    "PandasForecastingDataset",
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

import math
import warnings
from collections.abc import Callable as Fn, Iterator, Mapping
from dataclasses import KW_ONLY, asdict, dataclass, field, fields, replace
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Self, cast, overload

from pandas import DataFrame, Index, MultiIndex, Series

from tsdm import datasets
from tsdm.constants import UNDEFINED
from tsdm.datasets import Dataset
from tsdm.pprint import pprint_repr
from tsdm.types import SupportsGetItem

from .base import RangeSelector, TimeSeries, TimeSeriesCollection
from .samples import SeparateTimeSample


@pprint_repr
@dataclass(frozen=True)
class PandasTS[TimeT = Any]:
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
        return cast("Self", PandasTS(**sliced))

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
class PandasTSC[KeyT, TimeT = Any](Mapping[KeyT, PandasTS[TimeT]]):
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
    def __getitem__(self, key: KeyT, /) -> PandasTS[TimeT]: ...
    def __getitem__(self, key: KeyT | RangeSelector[KeyT], /) -> PandasTS[TimeT] | Self:
        r"""Get the timeseries and metadata of the dataset at index `key`."""
        match key:
            case slice() as s:
                labels = self._select_slice(s)
                return self._subset(labels)

            case Series(dtype=dtype) as series if dtype.kind == "b":
                if len(series) != len(self.metaindex):
                    raise ValueError(
                        "Expected boolean mask to have one entry per collection,"
                        f" got {len(series)} and {len(self.metaindex)}."
                    )
                mask = series.to_numpy(dtype=bool, na_value=False)
                return self._subset(self.metaindex[mask])

            case Series() as series:
                return self._subset(series.tolist())

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
        return cast("Self", PandasTSC(**sliced))

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


@pprint_repr
@dataclass(frozen=True, slots=True)
class Sample(SeparateTimeSample[Series | DataFrame]):
    r"""A pandas-backed forecasting sample with separate context and query times."""

    _: KW_ONLY

    context_times: Series
    r"""Timestamps associated with the context values."""
    context_values: DataFrame
    r"""Values available to the model as forecasting context."""
    context_mask: DataFrame
    r"""Mask indicating which context values are available."""
    query_times: Series
    r"""Timestamps at which predictions are requested."""
    query_mask: DataFrame
    r"""Mask indicating which target values should be predicted."""
    target_values: Optional[DataFrame] = None
    r"""Target values at the query times."""
    static_covariates: Optional[Series | DataFrame] = None
    r"""Static covariates associated with the time series."""

    def drop_null_rows(self) -> Self:
        r"""Drop rows that contain only NAN values."""
        context_values = self.context_values.dropna(how="all")

        if self.target_values is None:
            return replace(
                self,
                context_times=self.context_times.loc[context_values.index],
                context_values=context_values,
                context_mask=self.context_mask.loc[context_values.index],
            )

        target_values = self.target_values.dropna(how="all")
        return replace(
            self,
            context_times=self.context_times.loc[context_values.index],
            context_values=context_values,
            context_mask=self.context_mask.loc[context_values.index],
            query_times=self.query_times.loc[target_values.index],
            query_mask=self.query_mask.loc[target_values.index],
            target_values=target_values,
        )


@pprint_repr
@dataclass
class PandasForecastingDataset[KeyT](SupportsGetItem[KeyT, Sample]):
    r"""Creates sample from a TimeSeriesCollection.

    This class is responsible for creating samples from a TimeSeriesCollection.
    It acts as a Map-Stype Dataset, keys should be created from an appriopriate Sampler instance.

    There are different modus operandi for creating samples from a TimeSeriesCollection.

    Note:
        - column-sparse
        - separate context and query times
        - masked: In this format, two equimodal copies of the data are stored with appropriate masking.
            - inputs = (t, s, m)
            - targets = (s', m')
        - dense: Here, the data is split into groups of equal length. (x, u, y) share the same time index.
            - inputs = (t, x, u, m_x)
            - targets = (y, m_y)
        - sparse: Here, the data is split into groups sparse tensors. ALl NAN-only rows are dropped.
            - inputs = (t_y, (t_x, x), (t_u, u), m_x)
            - targets = (y, m_y)

    This class is used inside DataLoader.

    +---------------+------------------+------------------+
    | variable      | observation-mask | forecasting-mask |
    +===============+==================+==================+
    | observables X | ✔                | ✘                |
    +---------------+------------------+------------------+
    | controls U    | ✔                | ✔                |
    +---------------+------------------+------------------+
    | targets Y     | ✘                | ✔                |
    +---------------+------------------+------------------+

    Examples:
        - time series classification task: empty forecasting horizon.
        - time series imputation task: observation horizon and forecasting horizon overlap
        - time series forecasting task: observation horizon and forecasting horizon
        - time series forecasting task (autoregressive): observables = targets
        - time series event forecasting: predict both event and time of event (tᵢ, yᵢ)_{i=1:n} given n
        - time series event forecasting++: predict both event and time of event (tᵢ, yᵢ)_{i=1:n} and n

    Notes:
        Samples are frozen dataclasses that structurally implement
        :class:`SeparateTimeSample`.
    """

    dataset: PandasTS | PandasTSC
    r"""The dataset to sample from."""

    _: KW_ONLY

    targets: Index | list = UNDEFINED
    r"""Columns of the data that are used as targets."""
    observables: Index | list = UNDEFINED
    r"""Columns of the data that are used as inputs."""
    covariates: Index | list = UNDEFINED
    r"""Columns of the data that are used as controls."""
    sparse_index: bool = False
    r"""Whether to drop sparse rows from the index."""
    sparse_columns: bool = False
    r"""Whether to drop sparse cols from the data."""

    def __post_init__(self) -> None:
        r"""Post init."""
        self.targets = Index([]) if self.targets is UNDEFINED else Index(self.targets)
        self.observables = (
            self.dataset.timeseries.columns.copy()
            if self.observables is UNDEFINED
            else Index(self.observables)
        )
        self.covariates = (
            Index([]) if self.covariates is UNDEFINED else Index(self.covariates)
        )
        self.validate()

    def __getitem__(self, key: KeyT, /) -> Sample:
        return self.make_sample(
            key,
            sparse_index=self.sparse_index,
            sparse_columns=self.sparse_columns,
        )

    def make_sample(
        self,
        key: KeyT,
        *,
        sparse_index: bool = False,
        sparse_columns: bool = False,
    ) -> Sample:
        r"""Create a sample from a TimeSeriesCollection.

        Args:
            key: The key of the sample - e.g. tuple[outer_index, (obs_rane, forecasting_range)].
            sparse_index: Whether to drop rows that contain only NAN values.
            sparse_columns: Whether to drop columns that contain only NAN values.
        """
        match self.dataset:
            case PandasTS() as tsd:
                horizons = cast("Any", key)
                static_covariates = tsd.static_covariates
            case PandasTSC() as tsc:
                try:
                    outer_key, horizons = cast("Any", key)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid key: {key!r}") from exc
                tsd = tsc[outer_key]
                static_covariates = (
                    None
                    if tsc.static_covariates is None
                    else tsc.static_covariates.loc[outer_key]
                )
            case _:
                raise TypeError(f"Invalid dataset type: {type(self.dataset)=}")

        try:
            observation_horizon, forecasting_horizon = horizons
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid key: {key!r}") from exc

        observables = Index(self.observables)
        targets = Index(self.targets)
        covariates = Index(self.covariates)

        # extract the relevant time series and their joint index
        ts_observed: DataFrame = tsd.timeseries.loc[observation_horizon]
        ts_forecast: DataFrame = tsd.timeseries.loc[forecasting_horizon]
        joint_horizon_index = ts_observed.index.union(ts_forecast.index)

        joint_horizon_mask = tsd.timeseries.index.isin(joint_horizon_index)
        ts = tsd.timeseries.loc[joint_horizon_mask]
        ts_observed_mask = ts.index.isin(ts_observed.index)
        ts_forecast_mask = ts.index.isin(ts_forecast.index)

        context_columns = (
            ts.columns.intersection(observables.union(covariates), sort=False)
            if sparse_columns
            else ts.columns
        )
        target_columns = targets if sparse_columns else ts.columns
        context_values = DataFrame(math.nan, index=ts.index, columns=context_columns)
        target_values = DataFrame(
            math.nan,
            index=ts.index[ts_forecast_mask],
            columns=target_columns,
        )

        # Observables are available in the observation horizon. Time-varying
        # covariates are available throughout the joint horizon.
        context_values.loc[ts_observed_mask, observables] = ts.loc[
            ts_observed_mask, observables
        ]
        context_values.loc[:, covariates] = ts.loc[:, covariates]

        # Targets are available in the forecasting horizon. Assigning from the
        # unmasked data preserves values where observation and forecast overlap.
        target_values.loc[:, targets] = ts.loc[ts_forecast_mask, targets]

        context_times = context_values.index.to_series().copy()
        context_mask = context_values.notna()
        query_times = target_values.index.to_series().copy()
        query_mask = target_values.notna()

        # assemble sample
        sample = Sample(
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
            query_times=query_times,
            query_mask=query_mask,
            target_values=target_values,
            static_covariates=static_covariates,
        )

        if sparse_index:
            sample = sample.drop_null_rows()

        return sample

    def validate(self) -> None:
        r"""Validate that chosen columns are present."""
        ts = self.dataset.timeseries
        observables = set(self.observables)
        targets = set(self.targets)
        covariates = set() if self.covariates is None else set(self.covariates)
        ts_columns = set(ts.columns)

        if cols := covariates - ts_columns:
            raise ValueError(f"Covariates {cols} not in found timeseries columns!")
        if cols := observables - ts_columns:
            raise ValueError(f"Observables {cols} not in found timeseries columns!")
        if cols := targets - ts_columns:
            raise ValueError(f"Targets {cols} not found in timeseries columns!")
        if cols := covariates & observables:
            raise ValueError(f"Covariates and observables not disjoint! {cols}.")
        if cols := ts_columns - (observables | targets | covariates):
            warnings.warn(f"Unused columns in timeseries: {cols}", stacklevel=2)


if TYPE_CHECKING:
    # ensure base classes are compatible with protocols
    # TODO: subclass protocols when PEP 767 (ReadOnly attributes) is accepted.

    def _upcast_ts[KeyT](arg: PandasTS[KeyT], /) -> TimeSeries[KeyT, DataFrame]:
        return arg

    def _upcast_tsc[KeyT](
        arg: PandasTSC[KeyT], /
    ) -> TimeSeriesCollection[KeyT, DataFrame]:
        return arg
