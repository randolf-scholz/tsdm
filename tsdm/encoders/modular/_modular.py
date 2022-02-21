r"""Implementation of encoders.

Notes
-----
Contains encoders in modular form.
  - See :mod:`tsdm.encoders.functional` for functional implementations.
"""

from __future__ import annotations

__all__ = [
    # ABC
    "BaseEncoder",
    # Classes
    "ChainedEncoder",
    "ConcatEncoder",
    "DataFrameEncoder",
    "DateTimeEncoder",
    "FloatEncoder",
    "FrameEncoder",
    "FrameSplitter",
    "IdentityEncoder",
    "IntEncoder",
    "MinMaxScaler",
    "PositionalEncoder",
    "ProductEncoder",
    "Standardizer",
    "TensorEncoder",
    "Time2Float",
    "TimeDeltaEncoder",
    "TripletEncoder",
    "LogEncoder",
]

import logging
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from functools import singledispatchmethod
from typing import (
    Any,
    Final,
    Generic,
    Literal,
    NamedTuple,
    Optional,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import pandas as pd
import pandas.api.types
import torch
from numpy.typing import NDArray
from pandas import (
    NA,
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    Timedelta,
    Timestamp,
)
from pandas.core.indexes.frozen import FrozenList
from torch import Tensor

from tsdm.datasets import TimeTensor
from tsdm.util.decorators import post_hook, pre_hook
from tsdm.util.strings import repr_mapping, repr_namedtuple, repr_sequence
from tsdm.util.types import PathType

__logger__ = logging.getLogger(__name__)

PandasObject = Union[Index, Series, DataFrame]
r"""Type Hint for pandas objects."""
TensorLike = Union[Tensor, NDArray, DataFrame, Series]
r"""Type Hint for tensor-like objects."""
TensorType = TypeVar("TensorType", Tensor, NDArray, DataFrame, Series)
r"""TypeVar for tensor-like objects."""


def get_broadcast(data: Any, axis: tuple[int, ...]) -> tuple[Union[None, slice], ...]:
    r"""Get the broadcast transform.

    Example
    -------
    data is (2,3,4,5,6,7)
    axis is (0,2,-1)
    broadcast is (:, None, :, None, None, :)
    then, given a tensor x of shape (2, 4, 7), we can perform
    element-wise operations via data + x[broadcast]
    """
    rank = len(data.shape)
    axes = list(range(rank))
    axis = tuple(a % rank for a in axis)
    broadcast = tuple(slice(None) if a in axis else None for a in axes)
    return broadcast


def apply_along_axes(
    a: Tensor, b: Tensor, op: Callable, axes: tuple[int, ...]
) -> Tensor:
    r"""Apply a function to multiple axes of a tensor.

    Parameters
    ----------
    a: Tensor
    b: Tensor
    op: Callable
    axes: tuple[int, ...]

    Returns
    -------
    Tensor
    """
    axes = tuple(axes)
    rank = len(a.shape)
    source = tuple(range(rank))
    inverse_permutation = axes + tuple(ax for ax in range(rank) if ax not in axes)
    perm = tuple(np.argsort(inverse_permutation))
    if isinstance(a, Tensor):
        a = torch.moveaxis(a, source, perm)
        a = op(a, b)
        a = torch.moveaxis(a, source, inverse_permutation)
    else:
        a = np.moveaxis(a, source, perm)
        a = op(a, b)
        a = np.moveaxis(a, source, inverse_permutation)
    return a


class BaseEncoder(ABC):
    r"""Base class that all encoders must subclass."""

    is_fitted: bool = False
    """Whether the encoder has been fitted."""

    def __init__(self):
        super().__init__()
        self.fit = post_hook(self.fit, self._post_fit_hook)
        self.encode = pre_hook(self.encode, self._pre_encode_hook)
        self.decode = pre_hook(self.decode, self._pre_decode_hook)
        self.transform = self.encode
        self.inverse_transform = self.decode

    def __matmul__(self, other: BaseEncoder) -> ChainedEncoder:
        r"""Return chained encoders."""
        return ChainedEncoder(self, other)

    def __or__(self, other: BaseEncoder) -> ProductEncoder:
        r"""Return product encoders."""
        return ProductEncoder(self, other)

    def __repr__(self) -> str:
        r"""Return a string representation of the encoder."""
        return f"{self.__class__.__name__}()"

    def fit(self, data: Any, /) -> None:  # pylint: disable=method-hidden
        r"""By default does nothing."""

    @abstractmethod
    def encode(self, data, /):  # pylint: disable=method-hidden
        r"""Transform the data."""

    @abstractmethod
    def decode(self, data, /):  # pylint: disable=method-hidden
        r"""Reverse the applied transformation."""

    def _post_fit_hook(
        self, *args: Any, **kwargs: Any  # pylint: disable=unused-argument
    ) -> None:
        self.is_fitted = True

    def _pre_encode_hook(
        self, *args: Any, **kwargs: Any  # pylint: disable=unused-argument
    ) -> None:
        if not self.is_fitted:
            raise RuntimeError("Encoder has not been fitted.")

    def _pre_decode_hook(
        self, *args: Any, **kwargs: Any  # pylint: disable=unused-argument
    ) -> None:
        if not self.is_fitted:
            raise RuntimeError("Encoder has not been fitted.")


class ChainedEncoder(BaseEncoder, Sequence[BaseEncoder]):
    r"""Represents function composition of encoders."""

    def __init__(self, *encoders: BaseEncoder):
        super().__init__()

        self.chained = []

        for encoder in encoders:
            if isinstance(encoder, ChainedEncoder):
                for enc in encoder:
                    self.chained.append(enc)
            else:
                self.chained.append(encoder)

        self.is_fitted = all(e.is_fitted for e in self.chained)

    def fit(self, data: Any, /) -> None:
        r"""Fit to the data."""
        for encoder in reversed(self.chained):
            encoder.fit(data)
            data = encoder.encode(data)

    def encode(self, data, /):
        r"""Encode the input."""
        for encoder in reversed(self.chained):
            data = encoder.encode(data)
        return data

    def decode(self, data, /):
        r"""Decode tne input."""
        for encoder in self.chained:
            data = encoder.decode(data)
        return data

    def __len__(self):
        r"""Return number of chained encoders."""
        return self.chained.__len__()

    def __getitem__(self, item):
        r"""Return given item."""
        return self.chained.__getitem__(item)

    def __repr__(self):
        r"""Pretty print."""
        return repr_sequence(self)


class Time2Float(BaseEncoder):
    r"""Convert ``Series`` encoded as ``datetime64`` or ``timedelta64`` to ``floating``.

    By default, the data is mapped onto the unit interval `[0,1]`
    """

    original_dtype: np.dtype
    offset: Any
    common_interval: Any
    scale: Any

    def __init__(self, normalization: Literal["gcd", "max", "none"] = "max"):
        r"""Choose the normalizations scheme.

        Parameters
        ----------
        normalization: Literal["gcd", "max", "none"], default "max"
        """
        super().__init__()
        self.normalization = normalization

    def fit(self, data: Series, /) -> None:
        r"""Fit to the data.

        Parameters
        ----------
        data: Series
        """
        ds = data
        self.original_dtype = ds.dtype
        self.offset = ds[0].copy()
        assert (
            ds.is_monotonic_increasing
        ), "Time-Values must be monotonically increasing!"

        assert not (
            np.issubdtype(self.original_dtype, np.floating)
            and self.normalization == "gcd"
        ), f"{self.normalization=} illegal when original dtype is floating."

        if np.issubdtype(ds.dtype, np.datetime64):
            ds = ds.view("datetime64[ns]")
            self.offset = ds[0].copy()
            timedeltas = ds - self.offset
        elif np.issubdtype(ds.dtype, np.timedelta64):
            timedeltas = ds.view("timedelta64[ns]")
        elif np.issubdtype(ds.dtype, np.integer):
            timedeltas = ds.view("timedelta64[ns]")
        elif np.issubdtype(ds.dtype, np.floating):
            __logger__.warning("Array is already floating dtype.")
            timedeltas = ds
        else:
            raise ValueError(f"{ds.dtype=} not supported")

        if self.normalization == "none":
            self.scale = np.array(1).view("timedelta64[ns]")
        elif self.normalization == "gcd":
            self.scale = np.gcd.reduce(timedeltas.view(int)).view("timedelta64[ns]")
        elif self.normalization == "max":
            self.scale = ds[-1]
        else:
            raise ValueError(f"{self.normalization=} not understood")

    def encode(self, data: Series, /) -> Series:
        """Encode the data.

        Roughly equal to::
            ds ⟶ ((ds - offset)/scale).astype(float)

        Parameters
        ----------
        data: Series

        Returns
        -------
        Series
        """
        if np.issubdtype(data.dtype, np.integer):
            return data
        if np.issubdtype(data.dtype, np.datetime64):
            data = data.view("datetime64[ns]")
            self.offset = data[0].copy()
            timedeltas = data - self.offset
        elif np.issubdtype(data.dtype, np.timedelta64):
            timedeltas = data.view("timedelta64[ns]")
        elif np.issubdtype(data.dtype, np.floating):
            __logger__.warning("Array is already floating dtype.")
            return data
        else:
            raise ValueError(f"{data.dtype=} not supported")

        common_interval = np.gcd.reduce(timedeltas.view(int)).view("timedelta64[ns]")

        return (timedeltas / common_interval).astype(float)

    def decode(self, data: Series, /) -> Series:
        """Apply the inverse transformation.

        Roughly equal to:
        ``ds ⟶ (scale*ds + offset).astype(original_dtype)``
        """


class DateTimeEncoder(BaseEncoder):
    r"""Encode DateTime as Float."""

    unit: str = "s"
    r"""The base frequency to convert timedeltas to."""
    base_freq: str = "s"
    r"""The frequency the decoding should be rounded to."""
    offset: Timestamp
    r"""The starting point of the timeseries."""
    kind: Union[type[Series], type[DatetimeIndex]]
    r"""Whether to encode as index of Series."""
    name: Optional[str] = None
    r"""The name of the original Series."""
    dtype: np.dtype
    r"""The original dtype of the Series."""
    freq: Optional[Any] = None
    r"""The frequency attribute in case of DatetimeIndex."""

    def __init__(self, unit: str = "s", base_freq: str = "s"):
        r"""Initialize the parameters.

        Parameters
        ----------
        unit: Optional[str]=None
        """
        super().__init__()
        self.unit = unit
        self.base_freq = base_freq

    def fit(self, data: Union[Series, DatetimeIndex], /) -> None:
        r"""Store the offset."""
        if isinstance(data, Series):
            self.kind = Series
        elif isinstance(data, DatetimeIndex):
            self.kind = DatetimeIndex
        else:
            raise ValueError(f"Incompatible {type(data)=}")

        self.offset = Timestamp(Series(data).iloc[0])
        self.name = str(data.name) if data.name is not None else None
        self.dtype = data.dtype
        if isinstance(data, DatetimeIndex):
            self.freq = data.freq

    def encode(self, data: Union[Series, DatetimeIndex], /) -> Series:
        r"""Encode the input."""
        return (data - self.offset) / Timedelta(1, unit=self.unit)

    def decode(self, data: Series, /) -> Union[Series, DatetimeIndex]:
        r"""Decode the input."""
        __logger__.debug("Decoding %s", type(data))
        converted = pandas.to_timedelta(data, unit=self.unit)
        datetimes = Series(converted + self.offset, name=self.name, dtype=self.dtype)
        if self.kind == Series:
            return datetimes.round(self.base_freq)
        return DatetimeIndex(  # pylint: disable=no-member
            datetimes, freq=self.freq, name=self.name, dtype=self.dtype
        ).round(self.base_freq)

    def __repr__(self) -> str:
        r"""Return a string representation."""
        return f"{self.__class__.__name__}(unit={self.unit!r})"


class TimeDeltaEncoder(BaseEncoder):
    r"""Encode TimeDelta as Float."""

    unit: str = "s"
    r"""The base frequency to convert timedeltas to."""
    base_freq: str = "s"
    r"""The frequency the decoding should be rounded to."""

    def __init__(self, unit: str = "s", base_freq: str = "s"):
        r"""Initialize the parameters.

        Parameters
        ----------
        unit: Optional[str]=None
        """
        super().__init__()
        self.unit = unit
        self.base_freq = base_freq

    def encode(self, data, /):
        r"""Encode the input."""
        return data / Timedelta(1, unit=self.unit)

    def decode(self, data, /):
        r"""Decode the input."""
        result = data * Timedelta(1, unit=self.unit)
        if hasattr(result, "apply"):
            return result.apply(lambda x: x.round(self.base_freq))
        if hasattr(result, "map"):
            return result.map(lambda x: x.round(self.base_freq))
        return result

    def __repr__(self) -> str:
        r"""Return a string representation."""
        return f"{self.__class__.__name__}(unit={self.unit!r})"


class IdentityEncoder(BaseEncoder):
    r"""Dummy class that performs identity function."""

    def encode(self, data, /):
        r"""Encode the input."""
        return data

    def decode(self, data, /):
        r"""Decode the input."""
        return data


class DataFrameEncoder(BaseEncoder):
    r"""Combine multiple encoders into a single one.

    It is assumed that the DataFrame Modality doesn't change.
    """

    column_encoders: Union[BaseEncoder, Mapping[Any, BaseEncoder]]
    r"""Encoders for the columns."""
    index_encoders: Optional[Union[BaseEncoder, Mapping[Any, BaseEncoder]]] = None
    r"""Optional Encoder for the index."""
    colspec: Series
    r"""The columns-specification of the DataFrame."""
    encode_index: bool
    r"""Whether to encode the index."""
    column_wise: bool
    r"""Whether to encode column-wise."""
    partitions: Optional[dict] = None
    r"""Contains partitions if used column wise."""

    def __init__(
        self,
        column_encoders: Union[BaseEncoder, Mapping[Any, BaseEncoder]],
        *,
        index_encoders: Optional[Union[BaseEncoder, Mapping[Any, BaseEncoder]]] = None,
    ):
        r"""Set up the individual encoders.

        Note: the same encoder instance can be used for multiple columns.

        Parameters
        ----------
        column_encoders
        index_encoders
        """
        super().__init__()
        self.column_encoders = column_encoders

        if isinstance(index_encoders, Mapping):
            raise NotImplementedError("Multi-Index encoders not yet supported")

        self.index_encoders = index_encoders
        self.column_wise: bool = isinstance(self.column_encoders, Mapping)
        self.encode_index: bool = index_encoders is not None

        index_spec = DataFrame(
            columns=["col", "encoder"],
            index=Index([], name="partition"),
        )

        if self.encode_index:
            if not isinstance(self.index_encoders, Mapping):
                _idxenc_spec = Series(
                    {
                        "col": NA,
                        "encoder": self.index_encoders,
                    },
                    name=0,
                )
                # index_spec = index_spec.append(_idxenc_spec)
                index_spec.loc[0] = _idxenc_spec
            else:
                raise NotImplementedError(
                    "Multiple Index encoders are not supported yet."
                )

        if not isinstance(self.column_encoders, Mapping):
            colenc_spec = DataFrame(
                columns=["col", "encoder"],
                index=Index([], name="partition"),
            )

            _colenc_spec = Series(
                {
                    "col": NA,
                    "encoder": self.column_encoders,
                },
                name=0,
            )
            # colenc_spec = colenc_spec.append(_colenc_spec)
            # colenc_spec = pandas.concat([colenc_spec, _colenc_spec])
            colenc_spec.loc[0] = _colenc_spec
        else:
            keys = self.column_encoders.keys()
            assert len(set(keys)) == len(keys), "Some index are duplicates!"

            encoders = Series(self.column_encoders.values(), name="encoder")
            partitions = Series(range(len(encoders)), name="partition")

            _columns: dict = defaultdict(list)
            for key, encoder in self.column_encoders.items():
                if isinstance(key, str):
                    _columns[encoder] = key
                else:
                    _columns[encoder].extend(key)

            columns = Series(_columns, name="col")
            colenc_spec = DataFrame(encoders, index=partitions)
            colenc_spec = colenc_spec.join(columns, on="encoder")

        self.spec = pandas.concat(
            [index_spec, colenc_spec],
            keys=["index", "columns"],
            names=["section", "partition"],
        ).astype({"col": object})

        self.spec.name = self.__class__.__name__

        # add extra repr options by cloning from spec.
        # for x in [
        #     "_repr_data_resource_",
        #     "_repr_fits_horizontal_",
        #     "_repr_fits_vertical_",
        #     "_repr_html_",
        #     "_repr_latex_",
        # ]:
        #     setattr(self, x, getattr(self.spec, x))

    def fit(self, data: DataFrame, /) -> None:
        r"""Fit to the data."""
        self.colspec = data.dtypes

        if self.index_encoders is not None:
            if isinstance(self.index_encoders, Mapping):
                raise NotImplementedError("Multiple index encoders not yet supported")
            self.index_encoders.fit(data.index)

        if isinstance(self.column_encoders, Mapping):
            # check if cols are a proper partition.
            # keys = set(df.columns)
            # _keys = set(self.column_encoders.keys())
            # assert keys <= _keys, f"Missing encoders for columns {keys - _keys}!"
            # assert (
            #     keys >= _keys
            # ), f"Encoder given for non-existent columns {_keys- keys}!"

            for _, series in self.spec.loc["columns"].iterrows():
                encoder = series["encoder"]
                cols = series["col"]
                encoder.fit(data[cols])
        else:
            cols = list(data.columns)
            self.spec.loc["columns"].iloc[0]["col"] = cols
            encoder = self.spec.loc["columns", "encoder"].item()
            encoder.fit(data)

    def encode(self, data: DataFrame, /) -> tuple:
        r"""Encode the input."""
        tensors = []
        for _, series in self.spec.loc["columns"].iterrows():
            encoder = series["encoder"]
            cols = series["col"]
            tensors.append(encoder.encode(data[cols]))
        encoded_columns = tuple(tensors)

        if self.index_encoders is not None:
            if isinstance(self.index_encoders, Mapping):
                raise NotImplementedError("Multiple index encoders not yet supported")
            encoded_index = self.index_encoders.encode(data.index)
            return encoded_index, *encoded_columns
        return encoded_columns

    def decode(self, data: tuple, /) -> DataFrame:
        r"""Decode the input."""
        if self.encode_index:
            if isinstance(self.index_encoders, Mapping):
                raise NotImplementedError("Multiple index encoders not yet supported")
            encoder = self.spec.loc["index", "encoder"].item()
            index = encoder.decode(data[0])
            data = data[1:]
        else:
            index = None

        columns = []
        col_names = []
        for partition, (col_name, encoder) in self.spec.loc["columns"].iterrows():
            tensor = data[partition]
            columns.append(encoder.decode(tensor))
            if isinstance(col_name, str):
                col_names.append(col_name)
            else:
                col_names.extend(col_name)

        columns = [
            np.expand_dims(arr, axis=1) if arr.ndim < 2 else arr for arr in columns
        ]
        values = np.concatenate(columns, axis=1)
        df = DataFrame(values, index=index, columns=col_names)
        return df[self.colspec.index].astype(self.colspec)  # bring cols in right order

    def __repr__(self) -> str:
        """Pretty print."""
        return f"{self.__class__.__name__}(" + self.spec.__repr__() + "\n)"

    def _repr_html_(self) -> str:
        """HTML representation."""
        html_repr = self.spec._repr_html_()  # pylint: disable=protected-access
        return f"<h3>{self.__class__.__name__}</h3> {html_repr}"


class Standardizer(BaseEncoder, Generic[TensorType]):
    r"""A StandardScalar that works with batch dims."""

    mean: Any
    r"""The mean value."""
    stdv: Any
    r"""The standard-deviation."""
    ignore_nan: bool = True
    r"""Whether to ignore nan-values while fitting."""
    axis: tuple[int, ...]
    r"""The axis to perform the scaling. If None, automatically select the axis."""

    class Parameters(NamedTuple):
        r"""The parameters of the StandardScalar."""

        mean: TensorLike
        stdv: TensorLike
        axis: tuple[int, ...]

        def __repr__(self) -> str:
            r"""Pretty print."""
            return repr_namedtuple(self)

    def __init__(
        self,
        /,
        mean: Optional[Tensor] = None,
        stdv: Optional[Tensor] = None,
        *,
        ignore_nan: bool = True,
        axis: Optional[Union[int, tuple[int, ...]]] = None,
    ):
        super().__init__()
        self.ignore_nan = ignore_nan
        self.axis = (axis,) if isinstance(axis, int) else axis  # type: ignore
        self.mean = mean
        self.stdv = stdv

    def __repr__(self) -> str:
        r"""Pretty print."""
        return f"{self.__class__.__name__}(axis={self.axis})"

    def __getitem__(self, item: Any) -> Standardizer:
        r"""Return a slice of the Standardizer."""
        encoder = Standardizer(
            mean=self.mean[item], stdv=self.stdv[item], axis=self.axis[1:]
        )
        encoder.is_fitted = self.is_fitted
        return encoder

    @property
    def param(self) -> Parameters:
        r"""Parameters of the Standardizer."""
        return self.Parameters(self.mean, self.stdv, self.axis)

    def fit(self, data: TensorType, /) -> None:
        r"""Compute the mean and stdv."""
        rank = len(data.shape)
        if self.axis is None:
            self.axis = () if rank == 1 else (-1,)

        selection = [(a % rank) for a in self.axis]
        axes = tuple(k for k in range(rank) if k not in selection)

        if isinstance(data, Tensor):
            if self.ignore_nan:
                self.mean = torch.nanmean(data, dim=axes)
                self.stdv = torch.sqrt(torch.nanmean((data - self.mean) ** 2, dim=axes))
            else:
                self.mean = torch.mean(data, dim=axes)
                self.stdv = torch.std(data, dim=axes)
        else:
            if self.ignore_nan:
                self.mean = np.nanmean(data, axis=axes)
                self.stdv = np.nanstd(data, axis=axes)
            else:
                self.mean = np.mean(data, axis=axes)
                self.stdv = np.std(data, axis=axes)

        # if isinstance(data, Tensor):
        #     mask = ~torch.isnan(data) if self.ignore_nan else torch.full_like(data, True)
        #     count = torch.maximum(torch.tensor(1), mask.sum(dim=axes))
        #     masked = torch.where(mask, data, torch.tensor(0.0))
        #     self.mean = masked.sum(dim=axes)/count
        #     residual = apply_along_axes(masked, -self.mean, torch.add, axes=axes)
        #     self.stdv = torch.sqrt((residual**2).sum(dim=axes)/count)
        # else:
        #     if isinstance(data, DataFrame) or isinstance(data, Series):
        #         data = data.values
        #     mask = ~np.isnan(data) if self.ignore_nan else np.full_like(data, True)
        #     count = np.maximum(1, mask.sum(axis=axes))
        #     masked = np.where(mask, data, 0)
        #     self.mean = masked.sum(axis=axes)/count
        #     residual = apply_along_axes(masked, -self.mean, np.add, axes=axes)
        #     self.stdv = np.sqrt((residual**2).sum(axis=axes)/count)

    def encode(self, data: TensorType, /) -> TensorType:
        r"""Encode the input."""
        if self.mean is None:
            raise RuntimeError("Needs to be fitted first!")

        __logger__.debug("Encoding data %s", data)
        broadcast = get_broadcast(data, self.axis)
        __logger__.debug("Broadcasting to %s", broadcast)

        return (data - self.mean[broadcast]) / self.stdv[broadcast]

    def decode(self, data: TensorType, /) -> TensorType:
        r"""Decode the input."""
        if self.mean is None:
            raise RuntimeError("Needs to be fitted first!")

        __logger__.debug("Encoding data %s", data)
        broadcast = get_broadcast(data, self.axis)
        __logger__.debug("Broadcasting to %s", broadcast)

        return data * self.stdv[broadcast] + self.mean[broadcast]


class MinMaxScaler(BaseEncoder, Generic[TensorType]):
    r"""A MinMaxScaler that works with batch dims and both numpy/torch."""

    xmin: Union[NDArray, Tensor]
    xmax: Union[NDArray, Tensor]
    ymin: Union[NDArray, Tensor]
    ymax: Union[NDArray, Tensor]
    scale: Union[NDArray, Tensor]
    """The scaling factor."""
    axis: tuple[int, ...]
    r"""Over which axis to perform the scaling."""

    class Parameters(NamedTuple):
        r"""The parameters of the MinMaxScaler."""

        xmin: TensorLike
        xmax: TensorLike
        ymin: TensorLike
        ymax: TensorLike
        scale: TensorLike
        axis: tuple[int, ...]

        def __repr__(self) -> str:
            r"""Pretty print."""
            return repr_namedtuple(self)

    def __init__(
        self,
        /,
        ymin: Optional[Union[float, TensorType]] = None,
        ymax: Optional[Union[float, TensorType]] = None,
        xmin: Optional[Union[float, TensorType]] = None,
        xmax: Optional[Union[float, TensorType]] = None,
        *,
        axis: Optional[Union[int, tuple[int, ...]]] = None,
    ):
        r"""Initialize the MinMaxScaler.

        Parameters
        ----------
        ymin
        ymax
        axis
        """
        super().__init__()
        self.xmin = np.array(0.0 if xmin is None else xmin)
        self.xmax = np.array(1.0 if xmax is None else xmax)
        self.ymin = np.array(0.0 if ymin is None else ymin)
        self.ymax = np.array(1.0 if ymax is None else ymax)
        self.scale = (self.ymax - self.ymin) / (self.xmax - self.xmin)
        self.axis = (axis,) if isinstance(axis, int) else axis  # type: ignore

    def __getitem__(self, item: Any) -> MinMaxScaler:
        r"""Return a slice of the MinMaxScaler."""
        xmin = self.xmin if self.xmin.ndim == 0 else self.xmin[item]
        xmax = self.xmax if self.xmax.ndim == 0 else self.xmax[item]
        ymin = self.ymin if self.ymin.ndim == 0 else self.ymin[item]
        ymax = self.ymax if self.ymax.ndim == 0 else self.ymax[item]

        oldvals = (self.xmin, self.xmax, self.ymin, self.ymax)
        newvals = (xmin, xmax, ymin, ymax)
        assert not all(x.ndim == 0 for x in oldvals)
        lost_ranks = max(x.ndim for x in oldvals) - max(x.ndim for x in newvals)

        encoder = MinMaxScaler(
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, axis=self.axis[lost_ranks:]
        )

        encoder.is_fitted = self.is_fitted
        return encoder

    def __repr__(self) -> str:
        r"""Pretty print."""
        return f"{self.__class__.__name__}(axis={self.axis})"

    @property
    def param(self) -> Parameters:
        r"""Parameters of the MinMaxScaler."""
        return self.Parameters(
            self.xmin, self.xmax, self.ymin, self.ymax, self.scale, self.axis
        )

    @singledispatchmethod
    def fit(self, data: TensorType, /) -> None:
        r"""Compute the min and max."""
        data = np.asarray(data)
        rank = len(data.shape)
        if self.axis is None:
            self.axis = () if rank == 1 else (-1,)

        selection = [(a % rank) for a in self.axis]
        axes = tuple(k for k in range(rank) if k not in selection)
        # axes = self.axis
        self.ymin = np.array(self.ymin, dtype=data.dtype)
        self.ymax = np.array(self.ymax, dtype=data.dtype)
        self.xmin = np.nanmin(data, axis=axes)
        self.xmax = np.nanmax(data, axis=axes)
        self.scale = (self.ymax - self.ymin) / (self.xmax - self.xmin)

    @fit.register(torch.Tensor)
    def _(self, data: Tensor, /) -> None:
        r"""Compute the min and max."""
        mask = torch.isnan(data)
        self.ymin = torch.tensor(self.ymin, device=data.device, dtype=data.dtype)
        self.ymax = torch.tensor(self.ymax, device=data.device, dtype=data.dtype)
        neginf = torch.tensor(float("-inf"), device=data.device, dtype=data.dtype)
        posinf = torch.tensor(float("+inf"), device=data.device, dtype=data.dtype)

        self.xmin = torch.amin(torch.where(mask, posinf, data), dim=self.axis)
        self.xmax = torch.amax(torch.where(mask, neginf, data), dim=self.axis)
        self.scale = (self.ymax - self.ymin) / (self.xmax - self.xmin)

    def encode(self, data: TensorType, /) -> TensorType:
        r"""Encode the input."""
        __logger__.debug("Encoding data %s", data)
        broadcast = get_broadcast(data, self.axis)
        __logger__.debug("Broadcasting to %s", broadcast)

        xmin = self.xmin[broadcast] if self.xmin.ndim > 1 else self.xmin
        scale = self.scale[broadcast] if self.scale.ndim > 1 else self.scale
        ymin = self.ymin[broadcast] if self.ymin.ndim > 1 else self.ymin

        return (data - xmin) * scale + ymin

    def decode(self, data: TensorType, /) -> TensorType:
        r"""Decode the input."""
        __logger__.debug("Decoding data %s", data)
        broadcast = get_broadcast(data, self.axis)
        __logger__.debug("Broadcasting to %s", broadcast)

        xmin = self.xmin[broadcast] if self.xmin.ndim > 1 else self.xmin
        scale = self.scale[broadcast] if self.scale.ndim > 1 else self.scale
        ymin = self.ymin[broadcast] if self.ymin.ndim > 1 else self.ymin

        return (data - ymin) / scale + xmin

    # @singledispatchmethod
    # def fit(self, data: Union[Tensor, np.ndarray], /) -> None:
    #     r"""Compute the min and max."""
    #     return self.fit(np.asarray(data))
    #
    # @fit.register(Tensor)
    # def _(self, data: Tensor) -> Tensor:
    #     r"""Compute the min and max."""
    #     mask = torch.isnan(data)
    #     self.xmin = torch.min(
    #         torch.where(mask, torch.tensor(float("+inf")), data), dim=self.axis
    #     )
    #     self.xmax = torch.max(
    #         torch.where(mask, torch.tensor(float("-inf")), data), dim=self.axis
    #     )
    #     self.scale = (self.ymax - self.ymin) / (self.xmax - self.xmin)

    # @encode.register(Tensor)  # type: ignore[no-redef]
    # def _(self, data: Tensor) -> Tensor:
    #     r"""Encode the input."""
    #     return self.scale * (data - self.xmin) + self.ymin
    #
    # @encode.register(Tensor)  # type: ignore[no-redef]
    # def _(self, data: NDArray) -> NDArray:
    #     r"""Encode the input."""
    #     return self.scale * (data - self.xmin) + self.ymin

    # @singledispatchmethod

    # @decode.register(Tensor)  # type: ignore[no-redef]
    # def _(self, data, /):
    #     r"""Decode the input."""
    #     return (1 / self.scale) * (data - self.ymin) + self.xmin
    #
    # @decode.register(Tensor)  # type: ignore[no-redef]
    # def _(self, data, /):
    #     r"""Decode the input."""
    #     return (1 / self.scale) * (data - self.ymin) + self.xmin

    # def fit(self, data, /) -> None:
    #     r"""Compute the min and max."""
    #     data = np.asarray(data)
    #     self.xmax = np.nanmax(data, axis=self.axis)
    #     self.xmin = np.nanmin(data, axis=self.axis)
    #     print(self.xmax, self.xmin)
    #     self.scale = (self.ymax - self.ymin) / (self.xmax - self.xmin)
    #
    # def encode(self, data, /):
    #     r"""Encode the input."""
    #     return self.scale * (data - self.xmin) + self.ymin
    #
    # def decode(self, data, /):
    #     r"""Decode the input."""
    #     return (1 / self.scale) * (data - self.ymin) + self.xmin


class ConcatEncoder(BaseEncoder):
    r"""Concatenate multiple encoders."""

    lengths: list[int]
    numdims: list[int]
    axis: int
    maxdim: int

    def __init__(self, axis: int = 0) -> None:
        r"""Concatenate tensors along the specified axis.

        Parameters
        ----------
        axis: int
        """
        super().__init__()
        self.axis = axis

    def fit(self, data: tuple[Tensor, ...], /) -> None:
        r"""Fit to the data."""
        self.numdims = [d.ndim for d in data]
        self.maxdim = max(self.numdims)
        # pad dimensions if necessary
        data = [d[(...,) + (None,) * (self.maxdim - d.ndim)] for d in data]
        # store the lengths of the slices
        self.lengths = [x.shape[self.axis] for x in data]

    def encode(self, data: tuple[Tensor, ...], /) -> Tensor:
        r"""Encode the input."""
        return torch.cat(
            [d[(...,) + (None,) * (self.maxdim - d.ndim)] for d in data], dim=self.axis
        )

    def decode(self, data: Tensor, /) -> tuple[Tensor, ...]:
        r"""Decode the input."""
        result = torch.split(data, self.lengths, dim=self.axis)
        return tuple(x.squeeze() for x in result)


class TensorEncoder(BaseEncoder):
    r"""Converts objects to Tensor."""

    dtype: torch.dtype
    r"""The default dtype."""
    device: torch.device
    r"""The device the tensors are stored in."""
    names: Optional[list[str]] = None
    # colspecs: list[Series]
    # """The data types/column names of all the tensors"""
    return_type: type = tuple

    def __init__(
        self,
        names: Optional[list[str]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.names = names
        self.dtype = torch.float32 if dtype is None else dtype
        # default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu") if device is None else device

        if names is not None:
            self.return_type = namedtuple("namedtuple", names)  # type: ignore[misc]

    def fit(self, data: PandasObject, /) -> None:
        r"""Fit to the data."""
        # for item in data:
        #     _type = type(item)
        #     if isinstance(item, DataFrame):
        #         columns = item.columns
        #         dtypes = item.dtypes
        #     elif isinstance(item, Series) or isinstance(item, Index):
        #         name = item.name
        #         dtype = item.dtype

    @overload
    def encode(self, data: PandasObject, /) -> Tensor:  # type: ignore[misc]
        ...

    @overload
    def encode(self, data: tuple[PandasObject, ...], /) -> tuple[Tensor, ...]:
        ...

    def encode(self, data, /):
        r"""Convert each inputs to tensor."""
        if isinstance(data, tuple):
            return tuple(self.encode(x) for x in data)
        return torch.tensor(data.values, device=self.device, dtype=self.dtype)

    @overload
    def decode(self, data: Tensor, /) -> PandasObject:
        ...

    @overload
    def decode(self, data: tuple[Tensor, ...], /) -> tuple[PandasObject, ...]:
        ...

    def decode(self, data, /):
        r"""Convert each input from tensor to numpy."""
        if isinstance(data, tuple):
            return tuple(self.decode(x) for x in data)
        return data.cpu().numpy()

    def __repr__(self):
        r"""Pretty print."""
        return f"{self.__class__.__name__}()"


class FloatEncoder(BaseEncoder):
    r"""Converts all columns of DataFrame to float32."""

    dtypes: Series = None
    r"""The original dtypes."""

    def __init__(self, dtype: str = "float32"):
        self.target_dtype = dtype
        super().__init__()

    def fit(self, data: PandasObject, /) -> None:
        r"""Remember the original dtypes."""
        self.dtypes = data.dtypes if hasattr(data, "dtypes") else data.dtype

    def encode(self, data: PandasObject, /) -> PandasObject:
        r"""Make everything float32."""
        return data.astype(self.target_dtype)

    def decode(self, data: PandasObject, /) -> PandasObject:
        r"""Restore original dtypes."""
        return data.astype(self.dtypes)

    def __repr__(self):
        r"""Pretty print."""
        return f"{self.__class__.__name__}()"


class IntEncoder(BaseEncoder):
    r"""Converts all columns of DataFrame to int32."""

    dtypes: Series = None
    r"""The original dtypes."""

    def fit(self, data: PandasObject, /) -> None:
        r"""Remember the original dtypes."""
        self.dtypes = data.dtypes

    def encode(self, data: PandasObject, /) -> PandasObject:
        r"""Make everything int32."""
        return data.astype("int32")

    def decode(self, data, /):
        r"""Restore original dtypes."""
        return data.astype(self.dtypes)

    def __repr__(self):
        r"""Pretty print."""
        return f"{self.__class__.__name__}()"


class TimeSlicer(BaseEncoder):
    r"""Reorganizes the data by slicing."""

    # TODO: multiple horizons

    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon

    @staticmethod
    def is_tensor_pair(data: Any) -> bool:
        r"""Check if the data is a pair of tensors."""
        if isinstance(data, Sequence) and len(data) == 2:
            if isinstance(data[0], torch.Tensor) and isinstance(data[1], torch.Tensor):
                return True
        return False

    @overload
    def encode(self, data: TimeTensor, /) -> Sequence[TimeTensor]:
        ...

    @overload
    def encode(self, data: Sequence[TimeTensor], /) -> Sequence[Sequence[TimeTensor]]:
        ...

    @overload
    def encode(
        self,
        data: Sequence[tuple[Tensor, Tensor]],
        /,
    ) -> Sequence[Sequence[tuple[Tensor, Tensor]]]:
        ...

    def encode(self, data, /):
        """Slice the data.

        Provide pairs of tensors (T, X) and return a list of pairs (T_sliced, X_sliced).
        """
        if isinstance(data, TimeTensor):
            return data[: self.horizon], data[self.horizon]
        if self.is_tensor_pair(data):
            T, X = data
            idx = T <= self.horizon
            return (T[idx], X[idx]), (T[~idx], X[~idx])
        return tuple(self.encode(item) for item in data)

    @overload
    def decode(self, data: Sequence[TimeTensor], /) -> TimeTensor:
        ...

    @overload
    def decode(self, data: Sequence[Sequence[TimeTensor]], /) -> Sequence[TimeTensor]:
        ...

    @overload
    def decode(
        self,
        data: Sequence[Sequence[tuple[Tensor, Tensor]]],
        /,
    ) -> Sequence[tuple[Tensor, Tensor]]:
        ...

    def decode(self, data, /):
        """Restores the original data."""
        if isinstance(data[0], TimeTensor) or self.is_tensor_pair(data[0]):
            return torch.cat(data, dim=0)
        return tuple(self.decode(item) for item in data)


class PositionalEncoder(BaseEncoder):
    r"""Positional encoding.

    .. math::
        x_{2 k}(t)   &:=\sin \left(\frac{t}{t^{2 k / τ}}\right) \\
        x_{2 k+1}(t) &:=\cos \left(\frac{t}{t^{2 k / τ}}\right)

    """

    # Constants
    num_dim: Final[int]
    r"""Number of dimensions."""

    # Buffers
    scale: Final[float]
    r"""Scale factor for positional encoding."""
    scales: Final[np.ndarray]
    r"""Scale factors for positional encoding."""

    def __init__(self, num_dim: int, scale: float) -> None:
        super().__init__()

        self.num_dim = num_dim
        self.scale = float(scale)
        self.scales = self.scale ** (-np.arange(0, num_dim + 2, 2) / num_dim)
        assert self.scales[0] == 1.0, "Something went wrong."

    def encode(self, data: np.ndarray, /) -> np.ndarray:
        r"""Signature: [...] -> [...2d].

        Note: we simple concatenate the sin and cosine terms without interleaving them.

        Parameters
        ----------
        data: Tensor

        Returns
        -------
        Tensor
        """
        z = np.einsum("..., d -> ...d", data, self.scales)
        return np.concatenate([np.sin(z), np.cos(z)], axis=-1)

    def decode(self, data: np.ndarray, /) -> np.ndarray:
        r"""Signature: [...2d] -> [...].

        Parameters
        ----------
        data: Tensor

        Returns
        -------
        Tensor
        """
        return np.arcsin(data[..., 0])


class TripletEncoder(BaseEncoder):
    r"""Encode the data into triplets."""

    categories: pd.CategoricalDtype
    r"""The stored categories."""
    dtypes: Series
    r"""The original dtypes."""

    def __init__(self, sparse: bool = True) -> None:
        r"""Initialize the encoder.

        Parameters
        ----------
        sparse: bool = True
        """
        super().__init__()
        self.sparse = sparse

    def fit(self, data: DataFrame, /) -> None:
        r"""Fit the encoder.

        Parameters
        ----------
        data
        """
        self.categories = pd.CategoricalDtype(data.columns)
        self.dtypes = data.dtypes
        # result = data.melt(ignore_index=False)
        # # observed = result["value"].notna()
        # # result = result[observed]
        # variable = result.columns[0]
        # result[variable] = result[variable].astype(pd.StringDtype())
        # self.categories = pd.CategoricalDtype(result[variable].unique())

    def encode(self, data: DataFrame, /) -> DataFrame:
        r"""Encode the data."""
        result = data.melt(ignore_index=False).dropna()
        # observed = result["value"].notna()
        # result = result[observed]
        variable = result.columns[0]
        result[variable] = result[variable].astype(pd.StringDtype())
        result[variable] = result[variable].astype(self.categories)
        result.rename(columns={variable: "variable"}, inplace=True)
        # result.index.rename("time", inplace=True)
        # result.sort_values(by=["time", "variable"], inplace=True)
        result = result.sort_index()

        if not self.sparse:
            return result
        return pd.get_dummies(
            result, columns=["variable"], sparse=True, prefix="", prefix_sep=""
        )

    def decode(self, data: DataFrame, /) -> DataFrame:
        r"""Decode the data."""
        if self.sparse:
            df = data.iloc[:, 1:].stack()
            df = df[df == 1]
            df.index = df.index.rename("variable", level=-1)
            df = df.reset_index(level=-1)
            df["value"] = data["value"]
        else:
            df = data
        df = df.pivot_table(
            # TODO: FIX with https://github.com/pandas-dev/pandas/pull/45994
            # simply use df.index.names instead then.
            index=df.index,
            columns="variable",
            values="value",
            dropna=False,
        )
        if isinstance(data.index, MultiIndex):
            df.index = MultiIndex.from_tuples(df.index, names=data.index.names)

        # re-add missing columns
        for cat in self.categories.categories:
            if cat not in df.columns:
                df[cat] = float("nan")  # TODO: replace with pd.NA when supported

        result = df[self.categories.categories]  # fix column order
        result = result.astype(self.dtypes)
        return result


class CSVEncoder(BaseEncoder):
    r"""Encode the data into a CSV file."""

    filename: PathType
    r"""The filename of the CSV file."""
    dtypes: Series
    r"""The original dtypes."""
    read_csv_kwargs: dict[str, Any]
    r"""The kwargs for the read_csv function."""
    to_csv_kwargs: dict[str, Any]
    r"""The kwargs for the to_csv function."""

    def __init__(
        self,
        filename: PathType,
        to_csv_kwargs: Optional[dict[str, Any]] = None,
        read_csv_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        r"""Initialize the encoder.

        Parameters
        ----------
        filename: str
        to_csv_kwargs: Optional[dict[str, Any]]
        read_csv_kwargs: Optional[dict[str, Any]]
        """
        super().__init__()
        self.filename = filename
        self.read_csv_kwargs = read_csv_kwargs or {}
        self.to_csv_kwargs = to_csv_kwargs or {}

    def fit(self, data: DataFrame, /) -> None:
        r"""Fit the encoder.

        Parameters
        ----------
        data
        """
        self.dtypes = data.dtypes

    def encode(self, data: DataFrame, /) -> PathType:
        r"""Encode the data."""
        data.to_csv(self.filename, **self.to_csv_kwargs)
        return self.filename

    def decode(self, data: Optional[PathType] = None, /) -> DataFrame:
        r"""Decode the data."""
        if data is None:
            data = self.filename
        frame = pd.read_csv(data, **self.read_csv_kwargs)
        return DataFrame(frame).astype(self.dtypes)


class ProductEncoder(BaseEncoder, Sequence[BaseEncoder]):
    r"""Product-Type for Encoders."""

    encoders: tuple[BaseEncoder, ...]

    def __init__(self, *encoders: BaseEncoder) -> None:
        super().__init__()
        self.encoders = encoders
        self.is_fitted = all(e.is_fitted for e in self.encoders)

    def __len__(self) -> int:
        r"""Return the number of the encoders."""
        return len(self.encoders)

    @singledispatchmethod
    def __getitem__(self, index):
        r"""Get the encoder at the given index."""
        raise ValueError(f"Index {index} not supported.")

    @__getitem__.register(int)
    def _(self, index: int) -> ProductEncoder:
        encoder = ProductEncoder(self.encoders[index])
        encoder.is_fitted = self.is_fitted
        return encoder

    @__getitem__.register(slice)
    def _(self, index: slice) -> ProductEncoder:
        encoder = ProductEncoder(*self.encoders[index])
        encoder.is_fitted = self.is_fitted
        return encoder

    def __repr__(self):
        r"""Pretty print."""
        return repr_sequence(self)

    def fit(self, data: Sequence[Any], /) -> None:
        r"""Fit the encoder."""
        # print(data)
        for encoder, x in zip(self.encoders, data):
            encoder.fit(x)

    def encode(self, data: Sequence[Any], /) -> Sequence[Any]:
        r"""Encode the data."""
        return tuple(encoder.encode(x) for encoder, x in zip(self.encoders, data))

    def decode(self, data: Sequence[Any], /) -> Sequence[Any]:
        r"""Decode the data."""
        return tuple(encoder.decode(x) for encoder, x in zip(self.encoders, data))


# class FrameSplitter(BaseEncoder):
#     r"""Split a DataFrame into multiple groups."""
#
#     columns: Index
#     dtypes: Series
#     groups: dict[Any, Sequence[Any]]
#
#     @staticmethod
#     def _pairwise_disjoint(groups: Iterable[Sequence[HashableType]]) -> bool:
#         union: set[HashableType] = set().union(*(set(obj) for obj in groups))
#         n = sum(len(u) for u in groups)
#         return n == len(union)
#
#     def __init__(self, groups: dict[HashableType, Sequence[HashableType]]) -> None:
#         super().__init__()
#         self.groups = groups
#         assert self._pairwise_disjoint(self.groups.values())
#
#     def fit(self, data: DataFrame, /) -> None:
#         r"""Fit the encoder."""
#         self.columns = data.columns
#         self.dtypes = data.dtypes
#
#     def encode(self, data: DataFrame, /) -> tuple[DataFrame, ...]:
#         r"""Encode the data."""
#         encoded = []
#         for columns in self.groups.values():
#             encoded.append(data[columns].dropna(how="all"))
#         return tuple(encoded)
#
#     def decode(self, data: tuple[DataFrame, ...], /) -> DataFrame:
#         r"""Decode the data."""
#         decoded = pd.concat(data, axis="columns")
#         decoded = decoded.astype(self.dtypes)
#         decoded = decoded[self.columns]
#         return decoded


class FrameSplitter(BaseEncoder, Mapping):
    r"""Split a DataFrame into multiple groups.

    The special value ``...`` (:class:`Ellipsis`) can be used to indicate
    that all other columns belong to this group.

    This function can be used on index columns as well.
    """

    column_columns: Index
    column_dtypes: Series
    column_indices: list[int]

    index_columns: Index
    index_dtypes = Series
    index_indices: list[int]

    # FIXME: Union[types.EllipsisType, set[Hashable]] in 3.10
    groups: dict[Hashable, Union[Hashable, list[Hashable]]]
    group_indices: dict[Hashable, list[int]]

    indices: dict[Hashable, list[int]]
    has_ellipsis: bool = False
    ellipsis: Optional[Hashable] = None

    permutation: list[int]
    inverse_permutation: list[int]

    # @property
    # def names(self) -> set[Hashable]:
    #     r"""Return the union of all groups."""
    #     sets: list[set] = [
    #         set(obj) if isinstance(obj, Iterable) else {Ellipsis}
    #         for obj in self.groups.values()
    #     ]
    #     union: set[Hashable] = set.union(*sets)
    #     assert sum(len(u) for u in sets) == len(union), "Duplicate columns!"
    #     return union

    def __init__(self, groups: Iterable[Hashable]) -> None:
        super().__init__()

        if not isinstance(groups, Mapping):
            groups = dict(enumerate(groups))

        self.groups = {}
        for key, obj in groups.items():
            if obj is Ellipsis:
                self.groups[key] = obj
                self.ellipsis = key
                self.has_ellipsis = True
            elif isinstance(obj, str) or not isinstance(obj, Iterable):
                self.groups[key] = [obj]
            else:
                self.groups[key] = list(obj)

    def __repr__(self):
        r"""Return a string representation of the object."""
        return repr_mapping(self)

    def __len__(self):
        r"""Return the number of groups."""
        return len(self.groups)

    def __iter__(self):
        r"""Iterate over the groups."""
        return iter(self.groups)

    def __getitem__(self, item):
        r"""Return the group."""
        return self.groups[item]

    def fit(self, data: DataFrame, /) -> None:
        r"""Fit the encoder."""
        index = data.index.to_frame()
        self.column_dtypes = data.dtypes
        self.column_columns = data.columns
        self.index_columns = index.columns
        self.index_dtypes = index.dtypes

        assert not (
            j := set(self.index_columns) & set(self.column_columns)
        ), f"index columns and data columns must be disjoint {j}!"

        data = data.copy().reset_index()

        def get_idx(cols: Any) -> list[int]:
            return [data.columns.get_loc(i) for i in cols]

        self.indices: dict[Hashable, int] = dict(enumerate(data.columns))
        self.group_indices: dict[Hashable, list[int]] = {}
        self.column_indices = get_idx(self.column_columns)
        self.index_indices = get_idx(self.index_columns)

        # replace ellipsis indices
        if self.has_ellipsis:
            # FIXME EllipsisType in 3.10
            fixed_cols = set().union(
                *(
                    set(cols) for cols in self.groups.values() if cols is not Ellipsis  # type: ignore[arg-type]
                )
            )
            ellipsis_columns = [c for c in data.columns if c not in fixed_cols]
            self.groups[self.ellipsis] = ellipsis_columns

        # set column indices
        self.permutation = []
        for group, columns in self.groups.items():
            if columns is Ellipsis:
                continue
            self.group_indices[group] = get_idx(columns)
            self.permutation += self.group_indices[group]
        self.inverse_permutation = np.argsort(self.permutation).tolist()
        # sorted(p.copy(), key=p.__getitem__)

    def encode(self, data: DataFrame, /) -> tuple[DataFrame, ...]:
        r"""Encode the data."""
        # copy the frame and add index as columns.
        data = data.copy()
        data = data.reset_index()  # prepend index as columns!
        data_columns = set(data.columns)

        assert data_columns <= set(self.indices.values()), (
            f"Unknown columns {data_columns - set(self.indices)}."
            "If you want to encode unknown columns add a group ``...`` (Ellipsis)."
        )

        encoded = []
        for columns in self.groups.values():
            encoded.append(data[columns].dropna(how="all").squeeze(axis="columns"))
        return tuple(encoded)

    def decode(self, data: tuple[DataFrame, ...], /) -> DataFrame:
        r"""Decode the data."""
        data = tuple(DataFrame(x) for x in data)
        joined = pd.concat(data, axis="columns")

        # unshuffle the columns, restoring original order
        joined = joined.iloc[..., self.inverse_permutation]

        # Assemble the columns
        columns = joined.iloc[..., self.column_indices]
        columns.columns = self.column_columns
        columns = columns.astype(self.column_dtypes)
        columns = columns.squeeze(axis="columns")

        # assemble the index
        index = joined.iloc[..., self.index_indices]
        index.columns = self.index_columns
        index = index.astype(self.index_dtypes)
        index = index.squeeze(axis="columns")

        if isinstance(index, Series):
            decoded = columns.set_index(index)
        else:
            decoded = columns.set_index(MultiIndex.from_frame(index))
        return decoded


class FrameEncoder(BaseEncoder):
    r"""Encode a DataFrame by group-wise transformations."""

    columns: Index
    dtypes: Series
    index_columns: Index
    index_dtypes: Series

    column_encoders: Optional[Union[BaseEncoder, Mapping[Hashable, BaseEncoder]]]
    r"""Encoders for the columns."""
    index_encoders: Optional[Union[BaseEncoder, Mapping[Hashable, BaseEncoder]]]
    r"""Optional Encoder for the index."""
    column_decoders: Optional[Union[BaseEncoder, Mapping[Hashable, BaseEncoder]]]
    r"""Reverse Dictionary from encoded column name -> encoder"""
    index_decoders: Optional[Union[BaseEncoder, Mapping[Hashable, BaseEncoder]]]
    r"""Reverse Dictionary from encoded index name -> encoder"""

    @staticmethod
    def _names(
        obj: Union[Index, Series, DataFrame]
    ) -> Union[Hashable, FrozenList[Hashable]]:
        if isinstance(obj, MultiIndex):
            return FrozenList(obj.names)
        if isinstance(obj, (Series, Index)):
            return obj.name
        if isinstance(obj, DataFrame):
            return FrozenList(obj.columns)
        raise ValueError

    def __init__(
        self,
        column_encoders: Optional[
            Union[BaseEncoder, Mapping[Hashable, BaseEncoder]]
        ] = None,
        *,
        index_encoders: Optional[
            Union[BaseEncoder, Mapping[Hashable, BaseEncoder]]
        ] = None,
    ):
        super().__init__()
        self.column_encoders = column_encoders
        self.index_encoders = index_encoders

    def fit(self, data: DataFrame, /) -> None:
        r"""Fit the encoder."""
        data = data.copy()
        index = data.index.to_frame()
        self.columns = data.columns
        self.dtypes = data.dtypes
        self.index_columns = index.columns
        self.index_dtypes = index.dtypes

        if self.column_encoders is None:
            self.column_decoders = None
        elif isinstance(self.column_encoders, BaseEncoder):
            self.column_encoders.fit(data)
            self.column_decoders = self.column_encoders
        else:
            self.column_decoders = {}
            for group, encoder in self.column_encoders.items():
                encoder.fit(data[group])
                encoded = encoder.encode(data[group])
                self.column_decoders[self._names(encoded)] = encoder

        if self.index_encoders is None:
            self.index_decoders = None
        elif isinstance(self.index_encoders, BaseEncoder):
            self.index_encoders.fit(index)
            self.index_decoders = self.index_encoders
        else:
            self.index_decoders = {}
            for group, encoder in self.index_encoders.items():
                encoder.fit(index[group])
                encoded = encoder.encode(index[group])
                self.index_decoders[self._names(encoded)] = encoder

    def encode(self, data: DataFrame, /) -> DataFrame:
        r"""Encode the data."""
        data = data.copy(deep=True)
        index = data.index.to_frame()
        encoded_cols = data
        encoded_inds = encoded_cols.index.to_frame()

        if self.column_encoders is None:
            pass
        elif isinstance(self.column_encoders, BaseEncoder):
            encoded = self.column_encoders.encode(data)
            encoded_cols = encoded_cols.drop(columns=data.columns)
            encoded_cols[self._names(encoded)] = encoded
        else:
            for group, encoder in self.column_encoders.items():
                encoded = encoder.encode(data[group])
                encoded_cols = encoded_cols.drop(columns=group)
                encoded_cols[self._names(encoded)] = encoded

        if self.index_encoders is None:
            pass
        elif isinstance(self.index_encoders, BaseEncoder):
            encoded = self.index_encoders.encode(index)
            encoded_inds = encoded_inds.drop(columns=index.columns)
            encoded_inds[self._names(encoded)] = encoded
        else:
            for group, encoder in self.index_encoders.items():
                encoded = encoder.encode(index[group])
                encoded_inds = encoded_inds.drop(columns=group)
                encoded_inds[self._names(encoded)] = encoded

        # Assemble DataFrame
        encoded = DataFrame(encoded_cols)
        encoded[self._names(encoded_inds)] = encoded_inds
        encoded = encoded.set_index(self._names(encoded_inds))
        return encoded

    def decode(self, data: DataFrame, /) -> DataFrame:
        r"""Decode the data."""
        data = data.copy(deep=True)
        index = data.index.to_frame()
        decoded_cols = data
        decoded_inds = decoded_cols.index.to_frame()

        if self.column_decoders is None:
            pass
        elif isinstance(self.column_decoders, BaseEncoder):
            decoded = self.column_decoders.decode(data)
            decoded_cols = decoded_cols.drop(columns=data.columns)
            decoded_cols[self._names(decoded)] = decoded
        else:
            for group, encoder in self.column_decoders.items():
                decoded = encoder.decode(data[group])
                decoded_cols = decoded_cols.drop(columns=group)
                decoded_cols[self._names(decoded)] = decoded

        if self.index_decoders is None:
            pass
        elif isinstance(self.index_decoders, BaseEncoder):
            decoded = self.index_decoders.decode(index)
            decoded_inds = decoded_inds.drop(columns=index.columns)
            decoded_inds[self._names(decoded)] = decoded
        else:
            for group, encoder in self.index_decoders.items():
                decoded = encoder.decode(index[group])
                decoded_inds = decoded_inds.drop(columns=group)
                decoded_inds[self._names(decoded)] = decoded

        # Restore index order + dtypes
        decoded_inds = decoded_inds[self.index_columns]
        decoded_inds = decoded_inds.astype(self.index_dtypes)

        # Assemble DataFrame
        decoded = DataFrame(decoded_cols)
        decoded[self._names(decoded_inds)] = decoded_inds
        decoded = decoded.set_index(self._names(decoded_inds))
        decoded = decoded[self.columns]
        decoded = decoded.astype(self.dtypes)

        return decoded

    def __repr__(self) -> str:
        r"""Return a string representation of the encoder."""
        items = {
            "column_encoders": self.column_encoders,
            "index_encoders": self.index_encoders,
        }
        return repr_mapping(items, title=self.__class__.__name__)


class LogEncoder(BaseEncoder):
    r"""Encode data on logarithmic scale.

    Uses base 2 by default for lower numerical error and fast computation.
    """

    threshold: NDArray
    replacement: NDArray

    def fit(self, data: NDArray, /) -> None:
        r"""Fit the encoder to the data."""
        assert np.all(data >= 0)

        mask = data == 0
        self.threshold = data[~mask].min()
        self.replacement = np.log2(self.threshold / 2)

    def encode(self, data: NDArray, /) -> NDArray:
        """Encode data on logarithmic scale."""
        result = data.copy()
        mask = data <= 0
        result[:] = np.where(mask, self.replacement, np.log2(data))
        return result

    def decode(self, data: NDArray, /) -> NDArray:
        r"""Decode data on logarithmic scale."""
        result = 2**data
        mask = result < self.threshold
        result[:] = np.where(mask, 0, result)
        return result
