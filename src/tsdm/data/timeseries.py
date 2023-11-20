"""Timeseries data structures and functions."""

__all__ = [
    "TimeSeriesCollection",
    "TimeSeriesDataset",
]

from collections.abc import Iterator, Mapping
from dataclasses import KW_ONLY, dataclass

import pandas
from pandas import DataFrame, Index, MultiIndex, Series
from torch.utils.data import Dataset as TorchDataset
from typing_extensions import Any, Optional, Self, overload

from tsdm.types.variables import key_var as K
from tsdm.utils.strings import repr_dataclass


@dataclass
class TimeSeriesDataset(TorchDataset[Series]):  # Q: Should this be a Mapping?
    r"""Abstract Base Class for TimeSeriesDatasets.

    A TimeSeriesDataset is a dataset that contains time series data and metadata.
    More specifically, it is a tuple (TS, M) where TS is a time series and M is metadata.

    For a given time-index, the time series data is a vector of measurements.
    """

    timeseries: DataFrame
    r"""The time series data."""

    _: KW_ONLY

    # Main Attributes
    metadata: Optional[DataFrame] = None
    r"""The metadata of the dataset."""
    timeindex: Index = NotImplemented
    r"""The time-index of the dataset."""
    name: str = NotImplemented
    r"""The name of the dataset."""

    # Space Descriptors
    index_description: Optional[DataFrame] = None
    r"""Data associated with the time such as measurement device, unit, etc."""
    timeseries_description: Optional[DataFrame] = None
    r"""Data associated with each channel such as measurement device, unit, etc."""
    metadata_description: Optional[DataFrame] = None
    r"""Data associated with each metadata such as measurement device, unit,  etc."""

    def __post_init__(self) -> None:
        r"""Post init."""
        if self.name is NotImplemented:
            self.name = self.__class__.__name__
        if self.timeindex is NotImplemented:
            self.timeindex = self.timeseries.index.copy().unqiue()

    def __iter__(self) -> Iterator[Series]:
        r"""Iterate over the timestamps."""
        return iter(self.timeindex)

    def __len__(self) -> int:
        r"""Return the number of timestamps."""
        return len(self.timeindex)

    @overload
    def __getitem__(self, key: K, /) -> Series: ...
    @overload
    def __getitem__(self, key: Index | slice | list[K], /) -> DataFrame: ...
    def __getitem__(self, key, /):
        r"""Get item from timeseries."""
        # we might get an index object, or a slice, or boolean mask...
        return self.timeseries.loc[key]

    def __repr__(self) -> str:
        r"""Get the representation of the collection."""
        return repr_dataclass(self, title=self.name)


@dataclass
class TimeSeriesCollection(Mapping[Any, TimeSeriesDataset]):
    r"""Abstract Base Class for **equimodal** TimeSeriesCollections.

    A TimeSeriesCollection is a tuple (I, D, G) consiting of

    - index $I$
    - indexed TimeSeriesDatasets $D = { (TS_i, M_i) ∣ i ∈ I }$
    - global variables $G∈𝓖$
    """

    timeseries: DataFrame
    r"""The time series data."""

    _: KW_ONLY

    # Main attributes
    metadata: Optional[DataFrame] = None
    r"""The metadata of the dataset."""
    timeindex: MultiIndex = NotImplemented
    r"""The time-index of the collection."""
    metaindex: Index = NotImplemented
    r"""The index of the collection."""
    global_metadata: Optional[DataFrame] = None
    r"""Metaindex-independent data."""

    # Space descriptors
    timeseries_description: Optional[DataFrame] = None
    r"""Data associated with each channel such as measurement device, unit, etc."""
    metadata_description: Optional[DataFrame] = None
    r"""Data associated with each metadata such as measurement device, unit,  etc."""
    timeindex_description: Optional[DataFrame] = None
    r"""Data associated with the time such as measurement device, unit, etc."""
    metaindex_description: Optional[DataFrame] = None
    r"""Data associated with each index such as measurement device, unit, etc."""
    global_metadata_description: Optional[DataFrame] = None
    r"""Data associated with each global metadata such as measurement device, unit,  etc."""

    # other
    name: str = NotImplemented
    r"""The name of the collection."""

    def __post_init__(self) -> None:
        r"""Post init."""
        if self.name is NotImplemented:
            if hasattr(self.timeseries, "name") and self.timeseries.name is not None:
                self.name = str(self.timeseries.name)
            else:
                self.name = self.__class__.__name__

        if self.timeindex is NotImplemented:
            self.timeindex = self.timeseries.index.copy()

        if self.metaindex is NotImplemented:
            if self.metadata is not None:
                self.metaindex = self.metadata.index.copy().unique()
            elif isinstance(self.timeseries.index, MultiIndex):
                self.metaindex = self.timeseries.index.copy().droplevel(-1).unique()
                # self.timeseries = self.timeseries.droplevel(0)
            else:
                self.metaindex = self.timeseries.index.copy().unique()

    @overload
    def __getitem__(self, key: K, /) -> TimeSeriesDataset: ...
    @overload
    def __getitem__(self, key: slice, /) -> Self: ...
    def __getitem__(self, key, /):
        r"""Get the timeseries and metadata of the dataset at index `key`."""
        # TODO: There must be a better way to slice this

        match key:
            case Series() as s if isinstance(s.index, MultiIndex):
                ts = self.timeseries.loc[s]
            case Series() as s:
                assert pandas.api.types.is_bool_dtype(s)
                # NOTE: loc[s] would not work here?!
                ts = self.timeseries.loc[s[s].index]
            case _:
                ts = self.timeseries.loc[key]

        # make sure metadata is always DataFrame.
        match self.metadata:
            case DataFrame() as df:
                md = df.loc[key]
                if isinstance(md, Series):
                    md = df.loc[[key]]
            case _:
                md = self.metadata

        # slice the timeindex-descriptions
        match self.timeindex_description:
            case DataFrame() as desc if desc.index.equals(self.metaindex):
                tidx_desc = desc.loc[key]
            case _:
                tidx_desc = self.timeindex_description

        # slice the ts-descriptions
        match self.timeseries_description:
            case DataFrame() as desc if desc.index.equals(self.metaindex):
                ts_desc = desc.loc[key]
            case _:
                ts_desc = self.timeseries_description

        # slice the metadata-descriptions
        match self.metadata_description:
            case DataFrame() as desc if desc.index.equals(self.metaindex):
                md_desc = desc.loc[key]
            case _:
                md_desc = self.metadata_description

        if isinstance(ts.index, MultiIndex):
            # ~~index = ts.index.droplevel(-1).unique()~~
            return TimeSeriesCollection(
                name=self.name,
                # metaindex=index,  # NOTE: regenerate metaindex.
                timeseries=ts,
                metadata=md,
                timeindex_description=tidx_desc,
                timeseries_description=ts_desc,
                metadata_description=md_desc,
                global_metadata=self.global_metadata,
                global_metadata_description=self.global_metadata_description,
                metaindex_description=self.metaindex_description,
            )

        return TimeSeriesDataset(
            name=self.name,
            timeindex=ts.index,
            timeseries=ts,
            metadata=md,
            index_description=tidx_desc,
            timeseries_description=ts_desc,
            metadata_description=md_desc,
        )

    def __len__(self) -> int:
        r"""Get the length of the collection."""
        return len(self.metaindex)

    def __iter__(self) -> Iterator[Any]:
        r"""Iterate over the collection."""
        return iter(self.metaindex)

    def __repr__(self) -> str:
        r"""Get the representation of the collection."""
        return repr_dataclass(self, title=self.name)


# TIMESERIES: dict[str, type[TimeSeriesCollection]] = {
#     "InSilicoTSC": InSilicoTSC,
#     "KiwiBenchmarkTSC": KiwiBenchmarkTSC,
# }
# """Dictionary of all available timseries classes."""
#
# OLD_DATASETS: dict[str, type[Dataset]] = {
#     "KiwiRuns": KiwiRuns,
# }
# """Deprecated dataset classes."""
#
# OLD_TIMESERIES: dict[str, type[TimeSeriesCollection]] = {
#     "KiwiRunsTSC": KiwiRunsTSC,
# }
# """Deprecated timeseries classes."""
