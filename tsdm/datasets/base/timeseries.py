"""Representation of Time Series Datasets."""

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
    "tensor_info",
]

from collections.abc import Collection, Iterable, Mapping, Sized
from typing import Any, NamedTuple, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from pandas import DataFrame, Index, Series
from torch import Tensor
from torch.utils.data import TensorDataset


class _IndexMethodClone:
    r"""Clone .loc and similar methods to tensor-like object."""

    def __init__(self, data: ArrayLike, index: Index, method: str = "loc"):
        self.data = data
        self.index = index
        self.index_method = getattr(self.index, method)

    def __getitem__(self, item):
        idx = self.index_method[item].values
        return self.data[idx]


class _TupleIndexMethodClone:
    r"""Clone .loc and similar methods to tensor-like object."""

    def __init__(
        self, data: tuple[ArrayLike, ...], index: tuple[Index, ...], method: str = "loc"
    ):
        self.data = data
        self.index = index
        self.method = tuple(getattr(idx, method) for idx in self.index)

    def __getitem__(self, item):
        indices = tuple(method[item] for method in self.method)
        return tuple(data[indices] for data in self.data)


class TimeTensor(Tensor):
    r"""Subclass of torch that holds an index.

    Use :meth:`TimeTensor.loc` and meth:`TimeTensor.iloc` just like with DataFrames.
    """

    def __new__(
        cls,
        x: Sized,
        *args: Any,
        index: Optional[Index] = None,
        **kwargs: Any,
    ) -> TimeTensor:  # need delayed annotation here!
        r"""Create new TimeTensor.

        Parameters
        ----------
        x: Sized
            The input data.
        args: Any
        index: Optional[Index] = None,
            If :class:`None`, then ``range(len(x))`` will be used as the index.
        kwargs: Any
        """
        if isinstance(x, (DataFrame, Series)):
            assert index is None, "Index given, but x is DataFrame/Series"
            x = x.values
        return super().__new__(cls, x, *args, **kwargs)  # type: ignore[call-arg]

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


IndexedArray = Union[Series, DataFrame, TimeTensor]
r"""Type Hint for IndexedArrays."""


def is_indexed_array(x: Any) -> bool:
    r"""Test if Union[Series, DataFrame, TimeTensor]."""
    return isinstance(x, (DataFrame, Series, TimeTensor))


def tensor_info(x: Tensor) -> str:
    r"""Print useful information about Tensor."""
    return f"{x.__class__.__name__}[{tuple(x.shape)}, {x.dtype}, {x.device.type}]"


class AlignedTimeSeriesDataset(TensorDataset):
    r"""Dataset of Sequences with shared index.

    This represents a special case of a time-series dataset in which all `TimeTensor` s
    share the same index.
    """


class TimeSeriesDataset(TensorDataset):
    """A general Time Series Dataset.

    Consists of 2 things
    - timeseries: TimeTensor / tuple[TimeTensor]
    - metadata: Tensor / tuple[Tensor]

    When retrieving items, we generally use slices:

    - ds[timestamp] = ds[timestamp:timestamp]
    - ds[t₀:t₁] = tuple[X[t₀:t₁] for X in self.timeseries], metadata
    """

    timeseries: Union[TimeTensor, tuple[TimeTensor, ...]]
    metadata: Optional[Union[Tensor, tuple[Tensor, ...]]] = None

    def __init__(
        self,
        *tensors: IndexedArray,
        timeseries: Optional[
            Union[
                IndexedArray,
                tuple[Index, Sized],
                dict[Index, Sized],
                Collection[IndexedArray],
                Collection[tuple[Index, Sized]],
            ]
        ] = None,
        metadata: Optional[Union[Tensor, tuple[Tensor]]] = None,
    ):
        super().__init__()

        ts_tensors = []
        for tensor in tensors:
            ts_tensors.append(TimeTensor(tensor))

        if timeseries is not None:
            # Case: IndexedArray
            if is_indexed_array(timeseries):
                ts_tensors.append(TimeTensor(timeseries))
            # Case: tuple[Index, ArrayLike],
            elif (
                isinstance(timeseries, tuple)
                and len(timeseries) == 2
                and isinstance(timeseries[0], Index)
            ):
                index, tensor = timeseries
                ts_tensors.append(TimeTensor(tensor, index=index))
            # Case: dict[Index, ArrayLike],
            elif isinstance(timeseries, Mapping):
                for index, tensor in timeseries.items():
                    ts_tensors.append(TimeTensor(tensor, index=index))
            # Case: Iterable
            elif isinstance(timeseries, Iterable):
                timeseries = list(timeseries)
                firstobs = timeseries[0]
                # Case: Iterable[IndexedArray]
                if is_indexed_array(firstobs):
                    for tensor in timeseries:
                        ts_tensors.append(TimeTensor(tensor))
                # Case: Iterable[tuple(Index, ArrayLike)]
                elif (
                    isinstance(firstobs, tuple)
                    and len(firstobs) == 2
                    and isinstance(firstobs[0], Index)
                ):
                    for index, tensor in timeseries:
                        ts_tensors.append(TimeTensor(tensor, index=index))
                else:
                    raise ValueError(f"{timeseries=} not undertstood")
            else:
                raise ValueError(f"{timeseries=} not undertstood")

        self.timeseries = tuple(ts_tensors)

        if metadata is not None:
            if isinstance(metadata, tuple):
                self.metadata = tuple(Tensor(tensor) for tensor in metadata)
            else:
                self.metadata = Tensor(metadata)

    def __repr__(self) -> str:
        r"""Pretty print."""
        pad = r"  "

        if isinstance(self.timeseries, tuple):
            ts_lines = [tensor_info(tensor) for tensor in self.timeseries]
        else:
            ts_lines = [tensor_info(self.timeseries)]

        if self.metadata is None:
            md_lines = [f"{None}"]
        elif isinstance(self.metadata, tuple):
            md_lines = [tensor_info(tensor) for tensor in self.metadata]
        else:
            md_lines = [tensor_info(self.metadata)]

        return (
            f"{self.__class__.__name__}("
            + "".join(["\n" + 2 * pad + line for line in ts_lines])
            + "\n"
            + pad
            + "metadata:"
            + "".join(["\n" + 2 * pad + line for line in md_lines])
            + "\n"
            + ")"
        )

    def __len__(self) -> int:
        r"""Return the length of the longest timeseries."""
        if isinstance(self.timeseries, tuple):
            return max(len(tensor) for tensor in self.timeseries)
        return len(self.timeseries)

    def __getitem__(self, item):
        r"""Return corresponding slice from each tensor."""
        return tuple(tensor.loc[item] for tensor in self.timeseries)


class TimeSeriesTuple(NamedTuple):
    r"""A tuple of Tensors describing a slice of a multivariate timeseries."""

    timestamps: Tensor
    """Timestamps of the data."""

    observables: Tensor
    """The observables."""

    covariates: Tensor
    """The controls."""

    targets: Tensor
    """The targets."""


class TimeSeriesBatch(NamedTuple):
    """Inputs for the model."""

    inputs: TimeSeriesTuple
    """The inputs."""

    future: TimeSeriesTuple
    """The future."""

    metadata: Tensor
    """The metadata."""
