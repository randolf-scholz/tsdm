"""Time series dataset."""
from dataclasses import KW_ONLY, dataclass
from typing import Any, Iterator, Mapping, Optional, overload

from pandas import DataFrame, Index, MultiIndex, Series
from torch.utils.data import Dataset as TorchDataset
from typing_extensions import Self

from tsdm.types.variables import key_var as K
from tsdm.utils.strings import repr_dataclass


@dataclass
class TimeSeriesDataset(TorchDataset[Series]):
    r"""Abstract Base Class for TimeSeriesDatasets.

    A TimeSeriesDataset is a dataset that contains time series data and MetaData.
    More specifically, it is a tuple (TS, M) where TS is a time series and M is metdata.
    """

    timeseries: DataFrame
    r"""The time series data."""

    _: KW_ONLY

    # Main Attributes
    name: str = NotImplemented
    r"""The name of the dataset."""
    index: Index = NotImplemented
    r"""The time-index of the dataset."""
    metadata: Optional[DataFrame] = None
    r"""The metadata of the dataset."""

    # Space Descriptors
    time_features: Optional[DataFrame] = None
    r"""Data associated with the time such as measurement device, unit, etc."""
    value_features: Optional[DataFrame] = None
    r"""Data associated with each channel such as measurement device, unit, etc."""
    metadata_features: Optional[DataFrame] = None
    r"""Data associated with each metadata such as measurement device, unit,  etc."""

    def __post_init__(self) -> None:
        r"""Post init."""
        if self.name is NotImplemented:
            self.name = self.__class__.__name__
        if self.index is NotImplemented:
            self.index = self.timeseries.index.copy().unqiue()

    def __len__(self) -> int:
        r"""Return the number of timestamps."""
        return len(self.index)

    @overload
    def __getitem__(self, key: K) -> Series:
        ...

    @overload
    def __getitem__(self, key: Index | slice | list[K]) -> DataFrame:
        ...

    def __getitem__(self, key):
        r"""Get item from timeseries."""
        return self.timeseries.loc[key]

    def __iter__(self) -> Iterator[Series]:
        r"""Iterate over the timestamps."""
        return iter(self.index)

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
    name: str = NotImplemented
    r"""The name of the collection."""
    index: Index = NotImplemented
    r"""The index of the collection."""

    metadata: Optional[DataFrame] = None
    r"""The metadata of the dataset."""
    global_metadata: Optional[DataFrame] = None
    r"""The global data of the dataset."""

    # Space descriptors
    index_features: Optional[DataFrame] = None
    r"""Data associated with each index such as measurement device, unit, etc."""
    time_features: Optional[DataFrame] = None
    r"""Data associated with the time such as measurement device, unit, etc."""
    value_features: Optional[DataFrame] = None
    r"""Data associated with each channel such as measurement device, unit, etc."""
    metadata_features: Optional[DataFrame] = None
    r"""Data associated with each metadata such as measurement device, unit,  etc."""
    global_features: Optional[DataFrame] = None
    r"""Data associated with each global metadata such as measurement device, unit,  etc."""

    def __post_init__(self) -> None:
        r"""Post init."""
        if self.name is NotImplemented:
            if hasattr(self.timeseries, "name") and self.timeseries.name is not None:
                self.name = str(self.timeseries.name)
            else:
                self.name = self.__class__.__name__
        if self.index is NotImplemented:
            if self.metadata is not None:
                self.index = self.metadata.index.copy().unique()
            elif isinstance(self.timeseries.index, MultiIndex):
                self.index = self.timeseries.index.copy().droplevel(-1).unique()
                # self.timeseries = self.timeseries.droplevel(0)
            else:
                self.index = self.timeseries.index.copy().unique()

    @overload
    def __getitem__(self, key: K) -> TimeSeriesDataset:
        ...

    @overload
    def __getitem__(self, key: slice) -> Self:
        ...

    def __getitem__(self, key):
        r"""Get the timeseries and metadata of the dataset at index `key`."""
        # TODO: There must be a better way to slice this
        if (
            isinstance(key, Series)
            and isinstance(key.index, Index)
            and not isinstance(key.index, MultiIndex)
        ):
            ts = self.timeseries.loc[key[key].index]
        else:
            ts = self.timeseries.loc[key]

        # make sure metadata is always DataFrame.
        if self.metadata is None:
            md = None
        elif isinstance(_md := self.metadata.loc[key], Series):
            md = self.metadata.loc[[key]]
        else:
            md = _md

        if self.time_features is None:
            tf = None
        elif self.time_features.index.equals(self.index):
            tf = self.time_features.loc[key]
        else:
            tf = self.time_features

        if self.value_features is None:
            vf = None
        elif self.value_features.index.equals(self.index):
            vf = self.value_features.loc[key]
        else:
            vf = self.value_features

        if self.metadata_features is None:
            mf = None
        elif self.metadata_features.index.equals(self.index):
            mf = self.metadata_features.loc[key]
        else:
            mf = self.metadata_features

        if isinstance(ts.index, MultiIndex):
            index = ts.index.droplevel(-1).unique()
            return TimeSeriesCollection(
                name=self.name,
                index=index,
                timeseries=ts,
                metadata=md,
                time_features=tf,
                value_features=vf,
                metadata_features=mf,
                global_metadata=self.global_metadata,
                global_features=self.global_features,
                index_features=self.index_features,
            )

        return TimeSeriesDataset(
            name=self.name,
            index=ts.index,
            timeseries=ts,
            metadata=md,
            time_features=tf,
            value_features=vf,
            metadata_features=mf,
        )

    def __len__(self) -> int:
        r"""Get the length of the collection."""
        return len(self.index)

    def __iter__(self) -> Iterator[K]:
        r"""Iterate over the collection."""
        return iter(self.index)
        # for key in self.index:
        #     yield self[key]

    def __repr__(self) -> str:
        r"""Get the representation of the collection."""
        return repr_dataclass(self, title=self.name)
