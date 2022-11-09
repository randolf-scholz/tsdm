r"""Representation of Time Series Datasets."""

from __future__ import annotations

__all__ = [
    # Classes
    "TimeTensor",
    "TimeSeriesDataset",
    "TimeSeriesTuple",
    "TimeSeriesBatch",
    # Types
    "IndexedArray",
    # Functions
]

from collections.abc import Iterator, Sized
from typing import Any, NamedTuple, Optional, TypeAlias

import numpy as np
from pandas import DataFrame, Index, Series, Timedelta
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset

from tsdm.utils.strings import repr_array, repr_sequence


class _IndexMethodClone:
    r"""Clone .loc and similar methods to tensor-like object."""

    def __init__(self, data: Tensor, index: Index, method: str = "loc") -> None:
        self.data = data
        self.index = index
        self.index_method = getattr(self.index, method)

    def __getitem__(self, key):
        idx = self.index_method[key]

        if isinstance(idx, Series):
            idx = idx.values

        return self.data[idx]


class _TupleIndexMethodClone:
    r"""Clone .loc and similar methods to tensor-like object."""

    def __init__(
        self, data: tuple[Tensor, ...], index: tuple[Index, ...], method: str = "loc"
    ):
        self.data = data
        self.index = index
        self.method = tuple(getattr(idx, method) for idx in self.index)

    def __getitem__(self, item):
        indices = tuple(method[item] for method in self.method)
        return tuple(data[indices] for data in self.data)


class TimeTensor(Tensor):
    r"""Subclass of torch that holds an index.

    Use `TimeTensor.loc` and `TimeTensor.iloc` just like with DataFrames.
    """

    def __new__(
        cls,
        x: Sized,
        *args: Any,
        index: Optional[Index] = None,
        **kwargs: Any,
    ) -> TimeTensor:  # need delayed annotation here!
        r"""Create new TimeTensor.

        If `index` is not provided, then `range(len(x))` will be used as the index.
        """
        if isinstance(x, (DataFrame, Series)):
            assert index is None, "Index given, but x is DataFrame/Series"
            x = x.values
        return super().__new__(cls, *(x, *args), **kwargs)

    def __init__(
        self,
        x: Sized,
        index: Optional[Index] = None,
    ):
        super().__init__()  # optional
        if isinstance(x, (DataFrame, Series)):
            index = x.index
        else:
            index = Index(np.arange(len(x))) if index is None else index

        self.index = Series(np.arange(len(x)), index=index)
        # self.loc = self.index.loc
        self.loc = _IndexMethodClone(self, self.index, "loc")
        self.iloc = _IndexMethodClone(self, self.index, "iloc")
        self.at = _IndexMethodClone(self, self.index, "at")
        self.iat = _IndexMethodClone(self, self.index, "iat")


IndexedArray: TypeAlias = Series | DataFrame | TimeTensor
r"""Type Hint for IndexedArrays."""


def is_indexed_array(x: Any) -> bool:
    r"""Test if Union[Series, DataFrame, TimeTensor]."""
    return isinstance(x, (DataFrame, Series, TimeTensor))


class TimeSeriesTuple(NamedTuple):
    r"""A tuple of Tensors describing a slice of a multivariate timeseries."""

    timestamps: Tensor
    r"""Timestamps of the data."""

    observables: Tensor
    r"""The observables."""

    covariates: Tensor
    r"""The controls."""

    targets: Tensor
    r"""The targets."""


class TimeSeriesBatch(NamedTuple):
    r"""Inputs for the model."""

    inputs: TimeSeriesTuple
    r"""The inputs."""

    future: TimeSeriesTuple
    r"""The future."""

    metadata: Tensor
    r"""The metadata."""


class TimeSeriesDataset(TorchDataset):
    r"""A general Time Series Dataset.

    Consists of 2 things
    - timeseries: single TimeTensor or tuple[TimeTensor]
    - metadata: single Tensor or tuple[Tensor]

    in the case of a tuple, the elements are allowed to be NamedTuples.

    When retrieving items, we generally use slices:

    - ds[timestamp] = ds[timestamp:timestamp]
    - ds[t₀:t₁] = tuple[X[t₀:t₁] for X in self.timeseries], metadata
    """

    timeseries: IndexedArray
    metadata: Optional[IndexedArray] = None
    ts_type: type[tuple] = tuple
    r"""The type of the timeseries."""
    md_type: type[tuple] = tuple
    r"""The type of the metadata."""

    def __init__(
        self,
        timeseries: IndexedArray,
        metadata: Optional[IndexedArray] = None,
    ):
        super().__init__()

        self.timeseries = timeseries
        self.metadata = metadata

        # # Set up the timeseries
        # if isinstance(timeseries, Mapping):
        #     self.ts_type = namedtuple("timeseries", timeseries.keys())  # type: ignore[misc]
        #     self.timeseries = self.ts_type(**timeseries)
        # # test for namedtuple
        # elif isinstance(timeseries, tuple):
        #     if hasattr(timeseries, "_fields"):  # check namedtuple
        #         self.ts_type = type(timeseries)
        #         self.timeseries = timeseries
        #     else:
        #         self.ts_type = tuple
        #         self.timeseries = tuple(timeseries)
        # else:
        #     self.timeseries = timeseries
        #
        # # Set up the metadata
        # if isinstance(metadata, Mapping):
        #     self.md_type = namedtuple("metadata", metadata.keys())  # type: ignore[misc]
        #     self.timeseries = self.md_type(**timeseries)
        # elif isinstance(metadata, tuple):
        #     if hasattr(metadata, "_fields"):  # check namedtuple
        #         self.md_type = type(metadata)
        #         self.metadata = metadata
        #     else:
        #         self.md_type = tuple
        #         self.metadata = tuple(metadata)
        # else:
        #     self.metadata = metadata

    def __repr__(self) -> str:
        r"""Pretty print."""
        title = self.__class__.__name__
        pad = 2

        if isinstance(self.timeseries, tuple):
            ts_lines = repr_sequence(
                self.timeseries,
                linebreaks=False,
                padding=pad,
                repr_fun=repr_array,
                title="timeseries=",
            )
        else:
            ts_lines = repr_array(self.timeseries)

        if self.metadata is None:
            md_lines = f"{None}"
        elif isinstance(self.metadata, tuple):
            md_lines = repr_sequence(
                self.metadata,
                linebreaks=False,
                padding=pad,
                repr_fun=repr_array,
                title="metadata=",
            )
        else:
            md_lines = "metadata=" + repr_array(self.metadata)

        return f"{title}[{ts_lines}, {md_lines}]"

    def __len__(self) -> int:
        r"""Return the total number of observations across all timeseries."""
        if isinstance(self.timeseries, tuple):
            return sum(len(ts) for ts in self.timeseries)
        return len(self.timeseries)

    def timespan(self) -> Timedelta:
        r"""Return the timespan of the dataset."""
        if isinstance(self.timeseries, tuple):
            tmax = max(max(ts.index) for ts in self.timeseries)
            tmin = min(min(ts.index) for ts in self.timeseries)
            return tmax - tmin
        return max(self.timeseries.index) - min(self.timeseries.index)

    def __getitem__(self, item: Any) -> TimeSeriesDataset:
        r"""Return corresponding slice from each tensor."""
        if isinstance(self.timeseries, tuple):
            if hasattr(self.timeseries, "_fields"):  # namedtuple
                timeseries = self.ts_type(*(ts[item] for ts in self.timeseries))
            else:
                timeseries = tuple(ts[item] for ts in self.timeseries)
        # if isinstance(item, list):
        #     return [self[x] for x in item]
        else:
            timeseries = self.timeseries.loc[item]

        return TimeSeriesDataset(timeseries, metadata=self.metadata)

    def __iter__(self) -> Iterator:
        r"""Iterate over each timeseries."""
        if self.metadata is None:
            return iter(self.timeseries)
        return iter(zip(self.timeseries, self.metadata))
