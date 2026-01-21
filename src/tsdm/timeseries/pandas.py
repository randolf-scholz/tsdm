r"""Implementation of TimeSeries and TimeSeriesCollection using Pandas DataFrames."""

__all__ = [
    # Classes
    "Metadata",
    "PandasTS",
    "PandasTSC",
    # Functions
    "damped_pendulum_ansari2023",
    "etth1",
    "etth2",
    "ettm1",
    "ettm2",
    "electricity",
    "in_silico",
    "kiwi_benchmark",
    "mimic_iii_de_brouwer2019",
    "mimic_iv_bilos2021",
    "physio_net2012",
    "physio_net2019",
    "traffic",
    "ushcn",
    "ushcn_de_brouwer2019",
]

from collections.abc import Hashable, Iterator, Mapping
from dataclasses import KW_ONLY, asdict, dataclass, fields
from typing import Any, ClassVar, Optional, Self, overload

from pandas import DataFrame, Index, MultiIndex, Series

from tsdm import datasets
from tsdm.constants import UNDEFINED
from tsdm.datasets import Dataset
from tsdm.timeseries.base import Metadata, TimeSeries, TimeSeriesCollection
from tsdm.types.scalars import TimestampScalar
from tsdm.utils.decorators import pprint_repr


@pprint_repr
@dataclass
class PandasTS(TimeSeries[DataFrame]):
    r"""Abstract Base Class for TimeSeriesDatasets.

    A TimeSeriesDataset is a dataset that contains time series data and metadata.
    More specifically, it is a tuple (TS, M) where TS is a time series and M is the metadata.

    For a given time-index, the time series data is a vector of measurements.
    """

    FIELDS: ClassVar[frozenset[str]] = frozenset({
        "timeseries",
        "timeseries_metadata",
        "static_covariates",
        "static_covariates_metadata",
    })
    r"""The essential fields of the time series collection."""

    _: KW_ONLY

    # Header
    name: Optional[str] = UNDEFINED
    r"""The name of the dataset."""

    # Main Attributes
    timeseries: DataFrame
    r"""The time series data."""
    timeseries_metadata: Optional[DataFrame] = None
    r"""Data associated with the time such as measurement device, unit, etc."""
    static_covariates: Optional[DataFrame] = None
    r"""The metadata of the dataset."""
    static_covariates_metadata: Optional[DataFrame] = None
    r"""Data associated with each metadata such as measurement device, unit,  etc."""
    timeindex: Index = UNDEFINED  # derived field
    r"""The time-index of the dataset."""
    timeindex_metadata: Optional[DataFrame] = None
    r"""Data associated with the time such as measurement device, unit, etc."""
    metadata: Optional[Metadata] = None
    r"""The metadata of the dataset."""

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

        if self.name is UNDEFINED:
            self.name = self._infer_name()

        if self.timeindex is UNDEFINED:
            self.timeindex = self._infer_timeindex()

        # ensure no dataclass fields are undefined
        for field in fields(self):
            if getattr(self, field.name, UNDEFINED) is UNDEFINED:
                raise ValueError(f"The field '{field.name}' is undefined.")

    def __len__(self) -> int:
        r"""Return the number of timestamps."""
        return len(self.timeindex)

    def __iter__(self) -> Iterator[TimestampScalar]:
        r"""Iterate over the timestamps."""
        return iter(self.timeindex)

    def __contains__(self, key: Hashable, /) -> bool:
        r"""Check if the key is in the timeindex."""
        return key in self.timeindex

    def __getitem__(self, key: Any, /) -> "PandasTS":
        r"""Return the subset of the timeseries at index `key`."""
        fields = {k: v for k, v in asdict(self).items() if k in self.FIELDS}
        fields.update(timeseries=self.timeseries.loc[key])
        return PandasTS(**fields)

    def _infer_name(self) -> str | None:
        r"""Get the name of the collection."""
        if self.metadata is not None:
            return self.metadata.name
        if (name := getattr(self.timeseries, "name", None)) is not None:
            return str(name)
        return None

    def _infer_timeindex(self) -> Index:
        r"""Get the timeindex."""
        index = self.timeseries.index.copy()
        if isinstance(index, MultiIndex):
            raise TypeError(
                "Tried to create a TimeSeries from a DataFrame with MultiIndex."
                "\n    Are you sure this is not a TimeSeriesCollection?"
            )
        return index.unique()


@pprint_repr
@dataclass
class PandasTSC[Key](TimeSeriesCollection[Key, PandasTS], Mapping[Key, PandasTS]):
    r"""Class for **equimodal** TimeSeriesCollections.

    A `TimeSeriesCollection` is a collection of `TimeSeries` objects.
    `Equimodal` means that all time series share the same schema (i.e. subset of variables).
    """

    FIELDS: ClassVar[frozenset[str]] = frozenset({
        "timeseries",
        "timeseries_metadata",
        "static_covariates",
        "static_covariates_metadata",
        "constants",
        "constants_metadata",
    })
    r"""The essential fields of the time series collection."""

    _: KW_ONLY

    # Header
    name: Optional[str] = UNDEFINED
    r"""The name of the collection."""

    # Main attributes
    timeseries: DataFrame
    r"""The collection of time series data."""
    timeseries_metadata: Optional[DataFrame] = None
    r"""Data associated with each channel such as measurement device, unit, etc."""
    static_covariates: Optional[DataFrame] = None
    r"""The static covariates associated with each timeseries."""
    static_covariates_metadata: Optional[DataFrame] = None
    r"""Data associated with each metadata such as measurement device, unit,  etc."""
    constants: Optional[DataFrame] = None
    r"""Additional data that is independent of the metaindex."""
    constants_metadata: Optional[DataFrame] = None
    r"""Data associated with each global metadata such as measurement device, unit,  etc."""
    timeindex: MultiIndex = UNDEFINED  # derived field
    r"""The time-index of the collection."""
    metaindex: Index = UNDEFINED  # derived field
    r"""The index of the collection."""
    metadata: Optional[Metadata] = None
    r"""The metadata of the dataset."""

    @classmethod
    def from_dataset(cls, arg: Dataset | type[Dataset], /) -> Self:
        r"""Create a TimeSeries from a Dataset."""
        ds = arg() if isinstance(arg, type) else arg

        if bad_names := set(ds.table_names) - cls.FIELDS:
            raise ValueError(f"The following table names: {bad_names}")

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

        if self.name is UNDEFINED:
            self.name = self._infer_name()

        if self.timeindex is UNDEFINED:
            self.timeindex = self._infer_timeindex()

        if self.metaindex is UNDEFINED:
            self.metaindex = self._infer_metaindex()

        # ensure that the index of the static covariates is a subset of the metaindex
        self._validate_static_covariates()

        # ensure no dataclass fields are undefined
        for field in fields(self):
            if getattr(self, field.name, UNDEFINED) is UNDEFINED:
                raise ValueError(f"The field '{field.name}' is undefined.")

    def __len__(self) -> int:
        r"""Get the number of timeseries in the collection."""
        return len(self.metaindex)

    def __iter__(self) -> Iterator[Any]:
        r"""Iterate over the timeseries in the collection."""
        return iter(self.metaindex)

    def __contains__(self, key: object, /) -> bool:
        r"""Check if the key is in the metaindex."""
        return key in self.metaindex

    def _infer_name(self) -> str | None:
        r"""Get the name of the collection."""
        if self.metadata is not None:
            return self.metadata.name
        if (name := getattr(self.timeseries, "name", None)) is not None:
            return str(name)
        return None

    def _infer_timeindex(self) -> MultiIndex:
        r"""Get the timeindex."""
        index = self.timeseries.index.copy()
        if not isinstance(index, MultiIndex):
            raise TypeError("Expected a timeseries with MultiIndex.")
        return index.unique()

    def _infer_metaindex(self) -> Index:
        r"""Get the metaindex."""
        return self.timeindex.copy().droplevel(-1).unique()

    def _validate_static_covariates(self) -> None:
        r"""Ensure that the static covariates index is a subset of the metaindex."""
        match self.static_covariates:
            case None: ...  # fmt: skip
            case DataFrame() as static_cov:
                if not static_cov.index.difference(self.metaindex).empty:
                    raise ValueError(
                        "Static covariates index is not a subset of the metaindex."
                    )
            case _:
                raise TypeError(
                    f"Expected static_covariates to be a pandas DataFrame or None,"
                    f" got {type(self.static_covariates)}."
                )

    # fmt: off
    @overload
    def __getitem__(self, key: Index | Series | slice | list[Key], /) -> Self: ...
    @overload
    def __getitem__(self, key: Key, /) -> PandasTS: ...  # pyright: ignore[reportOverlappingOverload]
    # fmt: on
    def __getitem__(self, key: Any, /) -> PandasTS | Self:
        r"""Get the timeseries and metadata of the dataset at index `key`."""
        # only pass non-derived fields
        fields = {k: v for k, v in asdict(self).items() if k in self.FIELDS}
        ts = self.timeseries.loc[key]
        cov = self.static_covariates
        cov = cov if cov is None else cov.loc[key]
        fields.update(timeseries=ts, static_covariates=cov)

        if isinstance(ts.index, MultiIndex):
            return self.__class__(**fields)
        return PandasTS(**{k: v for k, v in fields.items() if k in PandasTS.FIELDS})


def electricity() -> PandasTS:
    r"""The Electricity dataset wrapped as TimeSeriesCollection."""
    return PandasTS.from_dataset(datasets.Electricity)


def traffic() -> PandasTS:
    r"""The Traffic dataset wrapped as TimeSeriesCollection."""
    return PandasTS.from_dataset(datasets.Traffic)


def etth1() -> PandasTS:
    r"""The ETTh1 dataset wrapped as TimeSeriesCollection."""
    ds = datasets.ETT()
    return PandasTS(timeseries=ds["ETTh1"], name="ETTh1")


def etth2() -> PandasTS:
    r"""The ETTh2 dataset wrapped as TimeSeriesCollection."""
    ds = datasets.ETT()
    return PandasTS(timeseries=ds["ETTh2"], name="ETTh2")


def ettm1() -> PandasTS:
    r"""The ETTm1 dataset wrapped as TimeSeriesCollection."""
    ds = datasets.ETT()
    return PandasTS(timeseries=ds["ETTm1"], name="ETTm1")


def ettm2() -> PandasTS:
    r"""The ETTm2 dataset wrapped as TimeSeriesCollection."""
    ds = datasets.ETT()
    return PandasTS(timeseries=ds["ETTm2"], name="ETTm2")


def in_silico() -> PandasTSC:
    r"""The in silico dataset wrapped as TimeSeriesCollection."""
    return PandasTSC.from_dataset(datasets.InSilico)


def kiwi_benchmark() -> PandasTSC:
    r"""The KIWI dataset wrapped as TimeSeriesCollection."""
    return PandasTSC.from_dataset(datasets.KiwiBenchmark)


def ushcn() -> PandasTSC:
    r"""The USHCN dataset wrapped as TimeSeriesCollection."""
    return PandasTSC.from_dataset(datasets.USHCN)


def ushcn_de_brouwer2019() -> PandasTSC:
    r"""The USHCN_DeBrouwer2019 dataset wrapped as TimeSeriesCollection."""
    return PandasTSC.from_dataset(datasets.USHCN_DeBrouwer2019)


def physio_net2012() -> PandasTSC:
    r"""The PhysioNet2012 dataset wrapped as TimeSeriesCollection."""
    return PandasTSC.from_dataset(datasets.PhysioNet2012)


def physio_net2019() -> PandasTSC:
    r"""The PhysioNet2019 dataset wrapped as TimeSeriesCollection."""
    return PandasTSC.from_dataset(datasets.PhysioNet2019)


def mimic_iv_bilos2021() -> PandasTSC:
    r"""The MIMIC_IV_Bilos2021 dataset wrapped as TimeSeriesCollection."""
    return PandasTSC.from_dataset(datasets.MIMIC_IV_Bilos2021)


def mimic_iii_de_brouwer2019() -> PandasTSC:
    r"""The MIMIC_III_DeBrouwer2019 dataset wrapped as TimeSeriesCollection."""
    return PandasTSC.from_dataset(datasets.MIMIC_III_DeBrouwer2019)


def damped_pendulum_ansari2023() -> PandasTSC:
    r"""The DampedPendulum_Ansari2023 dataset wrapped as TimeSeriesCollection."""
    return PandasTSC.from_dataset(datasets.DampedPendulum_Ansari2023)
