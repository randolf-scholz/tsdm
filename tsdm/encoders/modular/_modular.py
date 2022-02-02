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
    "ConcatEncoder",
    "ChainedEncoder",
    "DataFrameEncoder",
    "DateTimeEncoder",
    "FloatEncoder",
    "IdentityEncoder",
    "MinMaxScaler",
    "PositionalEncoder",
    "Standardizer",
    "TensorEncoder",
    "Time2Float",
    "IntEncoder",
    "TripletEncoder",
]

import logging
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import singledispatchmethod
from typing import Any, Final, Literal, Optional, Union, overload

import numpy as np
import pandas as pd
import pandas.api.types
import torch
from pandas import NA, DataFrame, DatetimeIndex, Index, Series, Timedelta, Timestamp
from torch import Tensor

from tsdm.datasets import TimeTensor

__logger__ = logging.getLogger(__name__)

PandasObject = Union[Index, Series, DataFrame]
r"""Type Hint for pandas objects."""


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
    iperm = axes + tuple(ax for ax in range(rank) if ax not in axes)
    perm = tuple(np.argsort(iperm))
    if isinstance(a, Tensor):
        a = torch.moveaxis(a, source, perm)
        a = op(a, b)
        a = torch.moveaxis(a, source, iperm)
    else:
        a = np.moveaxis(a, source, perm)
        a = op(a, b)
        a = np.moveaxis(a, source, iperm)
    return a


class BaseEncoder(ABC):
    """Base class that all encoders must subclass."""

    # TODO: Implement __matmul__ @ for composition of encoders.

    def __init__(self):
        super().__init__()
        self.transform = self.encode
        self.inverse_transform = self.decode

    def fit(self, data):
        r"""By default does nothing."""

    @abstractmethod
    def encode(self, data, /):
        r"""Transform the data."""

    @abstractmethod
    def decode(self, data, /):
        r"""Reverse the applied transformation."""

    def __matmul__(self, other):
        r"""Return chained encoders."""
        return ChainedEncoder(self, other)

    def __repr__(self):
        """Return a string representation of the encoder."""
        return f"{self.__class__.__name__}"


class ChainedEncoder(BaseEncoder):
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

    def fit(self, data):
        r"""Fit to the data."""
        for encoder in reversed(self.chained):
            encoder.fit(data)
            data = encoder.encode(data)

    def encode(self, data):
        r"""Encode the input."""
        for encoder in reversed(self.chained):
            data = encoder.encode(data)
        return data

    def decode(self, data):
        r"""Decode tne input."""
        for encoder in self.chained:
            data = encoder.decode(data)
        return data

    def __iter__(self):
        r"""Yield the encoders 1 by 1."""
        for encoder in self.chained:
            yield encoder

    def __len__(self):
        r"""Return number of chained encoders."""
        return self.chained.__len__()

    def __getitem__(self, item):
        r"""Return given item."""
        return self.chained.__getitem__(item)

    def __contains__(self, item):
        r"""Return contains."""
        return self.chained.__contains__(item)

    def __repr__(self):
        r"""Pretty print."""
        return f"{self.__class__.__name__}{self.chained}"


class Time2Float(BaseEncoder):
    r"""Convert ``Series`` encoded as ``datetime64`` or ``timedelta64`` to ``floating``.

    By default, the data is mapped onto the unit interval `[0,1]`
    """

    original_dtype: np.dtype
    offset: Any
    common_interval: Any
    scale: Any

    def __init__(self, normalization: Literal["gcd", "max", "none"] = "max"):
        """Choose the normalizations scheme.

        Parameters
        ----------
        normalization: Literal["gcd", "max", "none"], default="max"
        """
        super().__init__()
        self.normalization = normalization

    def fit(self, data: Series):
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

    def encode(self, ds: Series) -> Series:
        """Encode the data.

        Roughly equal to::
            ds ⟶ ((ds - offset)/scale).astype(float)

        Parameters
        ----------
        ds: Series

        Returns
        -------
        Series
        """
        if np.issubdtype(ds.dtype, np.integer):
            return ds
        if np.issubdtype(ds.dtype, np.datetime64):
            ds = ds.view("datetime64[ns]")
            self.offset = ds[0].copy()
            timedeltas = ds - self.offset
        elif np.issubdtype(ds.dtype, np.timedelta64):
            timedeltas = ds.view("timedelta64[ns]")
        elif np.issubdtype(ds.dtype, np.floating):
            __logger__.warning("Array is already floating dtype.")
            return ds
        else:
            raise ValueError(f"{ds.dtype=} not supported")

        common_interval = np.gcd.reduce(timedeltas.view(int)).view("timedelta64[ns]")

        return (timedeltas / common_interval).astype(float)

    def decode(self, ds: Series, /) -> Series:
        """Apply the inverse transformation.

        Roughly equal to:
        ``ds ⟶ (scale*ds + offset).astype(original_dtype)``
        """


class DateTimeEncoder(BaseEncoder):
    r"""Encode DateTime as Float."""

    unit: str = "s"
    r"""The base frequency to convert timedeltas to."""
    basefreq: str = "s"
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

    def __init__(self, unit: str = "s", basefreq: str = "s"):
        r"""Initialize the parameters.

        Parameters
        ----------
        unit: Optional[str]=None
        """
        super().__init__()
        self.unit = unit
        self.basefreq = basefreq

    def fit(self, data: Union[Series, DatetimeIndex]):
        r"""Store the offset."""
        if isinstance(data, Series):
            self.kind = Series
        elif isinstance(data, DatetimeIndex):
            self.kind = DatetimeIndex
        else:
            raise ValueError(f"Incompatible {type(data)=}")

        self.offset = Timestamp(data[0])
        self.name = str(data.name) if data.name is not None else None
        self.dtype = data.dtype
        if isinstance(data, DatetimeIndex):
            self.freq = data.freq

    def encode(self, datetimes: Union[Series, DatetimeIndex]) -> Series:
        r"""Encode the input."""
        return (datetimes - self.offset) / Timedelta(1, unit=self.unit)

    def decode(self, timedeltas: Series) -> Union[Series, DatetimeIndex]:
        r"""Decode the input."""
        __logger__.debug("Decoding %s", type(timedeltas))
        converted = pandas.to_timedelta(timedeltas, unit=self.unit)
        datetimes = Series(converted + self.offset, name=self.name, dtype=self.dtype)
        if self.kind == Series:
            return datetimes.round(self.basefreq)
        return DatetimeIndex(  # pylint: disable=no-member
            datetimes, freq=self.freq, name=self.name, dtype=self.dtype
        ).round(self.basefreq)


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

    def fit(self, df: DataFrame, /):
        r"""Fit to the data."""
        self.colspec = df.dtypes

        if self.index_encoders is not None:
            if isinstance(self.index_encoders, Mapping):
                raise NotImplementedError("Multiple index encoders not yet supported")
            self.index_encoders.fit(df.index)

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
                encoder.fit(df[cols])
        else:
            cols = list(df.columns)
            self.spec.loc["columns"].iloc[0]["col"] = cols
            encoder = self.spec.loc["columns", "encoder"].item()
            encoder.fit(df)

    def encode(self, df: DataFrame, /) -> tuple:
        r"""Encode the input."""
        tensors = []
        for _, series in self.spec.loc["columns"].iterrows():
            encoder = series["encoder"]
            cols = series["col"]
            tensors.append(encoder.encode(df[cols]))
        encoded_columns = tuple(tensors)

        if self.index_encoders is not None:
            if isinstance(self.index_encoders, Mapping):
                raise NotImplementedError("Multiple index encoders not yet supported")
            encoded_index = self.index_encoders.encode(df.index)
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


class Standardizer(BaseEncoder):
    r"""A StandardScalar that works with batch dims."""

    mean: Any
    r"""The mean value."""
    stdv: Any
    r"""The standard-deviation."""
    ignore_nan: bool = True
    r"""Whether to ignore nan-values while fitting."""
    axis: tuple[int, ...]
    r"""Over which axis to perform the scaling."""

    def __init__(
        self,
        /,
        mean: Optional[Tensor] = None,
        stdv: Optional[Tensor] = None,
        *,
        ignore_nan: bool = True,
        axis: Union[int, tuple[int, ...]] = -1,
    ):
        super().__init__()
        self.ignore_nan = ignore_nan
        self.axis = axis if isinstance(axis, tuple) else (axis,)
        self.mean = mean
        self.stdv = stdv

    def fit(self, data):
        r"""Compute the mean and stdv."""
        rank = len(data.shape)
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

    def encode(self, data):
        r"""Encode the input."""
        if self.mean is None:
            raise RuntimeError("Needs to be fitted first!")
        return (data - self.mean) / self.stdv

    def decode(self, data):
        r"""Decode the input."""
        if self.mean is None:
            raise RuntimeError("Needs to be fitted first!")
        return data * self.stdv + self.mean

    def __repr__(self) -> str:
        r"""Pretty print."""
        return f"{self.__class__.__name__}(" + f"{list(self.axis)}" + ")"

    def __getitem__(self, item: Any) -> Standardizer:
        r"""Return a slice of the Standarsizer."""
        return Standardizer(
            mean=self.mean[item], stdv=self.stdv[item], axis=self.axis[1:]
        )


class MinMaxScaler(BaseEncoder):
    r"""A MinMaxScaler that works with batch dims and both numpy/torch."""

    xmin: Union[np.ndarray, Tensor]
    xmax: Union[np.ndarray, Tensor]
    ymin: Union[np.ndarray, Tensor]
    ymax: Union[np.ndarray, Tensor]
    scale: Union[np.ndarray, Tensor]
    """The scaling factor."""

    def __init__(
        self,
        /,
        ymin: Optional[Union[float, np.ndarray]] = None,
        ymax: Optional[Union[float, np.ndarray]] = None,
        xmin: Optional[Union[float, np.ndarray]] = None,
        xmax: Optional[Union[float, np.ndarray]] = None,
        *,
        axis: Union[int, tuple[int, ...]] = -1,
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

        if isinstance(axis, Iterable):
            self.axis = tuple(axis)
        else:
            self.axis = (axis,)

    @singledispatchmethod
    def fit(self, data: np.ndarray) -> None:
        r"""Compute the min and max."""
        data = np.asarray(data)
        self.ymin = np.array(self.ymin, dtype=data.dtype)
        self.ymax = np.array(self.ymax, dtype=data.dtype)
        self.xmin = np.nanmin(data, axis=self.axis)
        self.xmax = np.nanmax(data, axis=self.axis)
        self.scale = (self.ymax - self.ymin) / (self.xmax - self.xmin)

    @fit.register(torch.Tensor)
    def _(self, data: Tensor) -> None:
        r"""Compute the min and max."""
        mask = torch.isnan(data)
        self.ymin = torch.tensor(self.ymin, device=data.device, dtype=data.dtype)
        self.ymax = torch.tensor(self.ymax, device=data.device, dtype=data.dtype)
        neginf = torch.tensor(float("-inf"), device=data.device, dtype=data.dtype)
        posinf = torch.tensor(float("+inf"), device=data.device, dtype=data.dtype)

        self.xmin = torch.amin(
            torch.where(mask, posinf, data), dim=self.axis, keepdim=True
        )
        self.xmax = torch.amax(
            torch.where(mask, neginf, data), dim=self.axis, keepdim=True
        )
        self.scale = (self.ymax - self.ymin) / (self.xmax - self.xmin)

    def encode(self, data, /):
        r"""Encode the input."""
        __logger__.debug("Encoding data %s", data)
        return (data - self.xmin) * self.scale + self.ymin

    def decode(self, data, /):
        r"""Decode the input."""
        __logger__.debug("Decoding data %s", data)
        return (data - self.ymin) / self.scale + self.xmin

    # @singledispatchmethod
    # def fit(self, data: Union[Tensor, np.ndarray]):
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

    # def fit(self, data):
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

    def fit(self, data):
        r"""Fit to the data."""
        self.numdims = [d.ndim for d in data]
        self.maxdim = max(self.numdims)
        # pad dimensions if necessary
        data = [d[(...,) + (None,) * (self.maxdim - d.ndim)] for d in data]
        # store the lengths of the slices
        self.lengths = [x.shape[self.axis] for x in data]

    def encode(self, data, /):
        r"""Encode the input."""
        return torch.cat(
            [d[(...,) + (None,) * (self.maxdim - d.ndim)] for d in data], axis=self.axis
        )

    def decode(self, data, /):
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

    def fit(self, *data: PandasObject):
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
    def encode(self, data: PandasObject) -> Tensor:  # type: ignore[misc]
        ...

    @overload
    def encode(self, data: tuple[PandasObject, ...]) -> tuple[Tensor, ...]:
        ...

    def encode(self, data):
        r"""Convert each inputs to tensor."""
        if isinstance(data, tuple):
            if self.return_type == tuple:
                return self.return_type(self.encode(x) for x in data)
            return self.return_type(*(self.encode(x) for x in data))
        return torch.tensor(data.values, device=self.device, dtype=self.dtype)

    @overload
    def decode(self, data: Tensor) -> PandasObject:
        ...

    @overload
    def decode(self, data: tuple[Tensor, ...]) -> tuple[PandasObject, ...]:
        ...

    def decode(self, data):
        r"""Convert each input from tensor to numpy."""
        if isinstance(data, tuple):
            return self.return_type(*(self.decode(x) for x in data))
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

    def fit(self, data: PandasObject):
        r"""Remember the original dtypes."""
        self.dtypes = data.dtypes if hasattr(data, "dtypes") else data.dtype

    def encode(self, data: PandasObject) -> PandasObject:
        r"""Make everything float32."""
        return data.astype(self.target_dtype)

    def decode(self, data, /):
        r"""Restore original dtypes."""
        return data.astype(self.dtypes)

    def __repr__(self):
        r"""Pretty print."""
        return f"{self.__class__.__name__}()"


class IntEncoder(BaseEncoder):
    """Converts all columns of DataFrame to int32."""

    dtypes: Series = None
    r"""The original dtypes."""

    def fit(self, data: PandasObject):
        r"""Remember the original dtypes."""
        self.dtypes = data.dtypes

    def encode(self, data: PandasObject) -> PandasObject:
        r"""Make everything int32."""
        return data.astype("int32")

    def decode(self, data, /):
        r"""Restore original dtypes."""
        return data.astype(self.dtypes)

    def __repr__(self):
        """Pretty print."""
        return f"{self.__class__.__name__}()"


class TimeSlicer(BaseEncoder):
    """Reorganizes the data by slicing."""

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
    def encode(self, data: TimeTensor) -> Sequence[TimeTensor]:
        ...

    @overload
    def encode(self, data: Sequence[TimeTensor]) -> Sequence[Sequence[TimeTensor]]:
        ...

    @overload
    def encode(
        self, data: Sequence[tuple[Tensor, Tensor]]
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
    def decode(self, data: Sequence[TimeTensor]) -> TimeTensor:
        ...

    @overload
    def decode(self, data: Sequence[Sequence[TimeTensor]]) -> Sequence[TimeTensor]:
        ...

    @overload
    def decode(
        self, data: Sequence[Sequence[tuple[Tensor, Tensor]]]
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
        x: Tensor

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
        x: Tensor

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

    def fit(self, data: DataFrame) -> None:
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

    def encode(self, df: DataFrame) -> DataFrame:
        r"""Encode the data."""
        result = df.melt(ignore_index=False).dropna()
        # observed = result["value"].notna()
        # result = result[observed]
        variable = result.columns[0]
        result[variable] = result[variable].astype(pd.StringDtype())
        result[variable] = result[variable].astype(self.categories)
        result.rename(columns={variable: "variable"}, inplace=True)
        result.index.rename("time", inplace=True)
        result.sort_values(by=["time", "variable"], inplace=True)

        if not self.sparse:
            return result
        return pd.get_dummies(
            result, columns=["variable"], sparse=True, prefix="", prefix_sep=""
        )

    def decode(self, data: DataFrame, /) -> DataFrame:
        r"""Decode the data."""
        if self.sparse:
            df = data.iloc[:, 1:]
            df = df[df == 1].stack().reset_index(level=-1)
            df["value"] = data["value"]
            df = df.rename(columns={"level_1": "variable"})
        else:
            df = data
        df = df.pivot_table(
            index="time", columns="variable", values="value", dropna=False
        )

        # re-add missing columns
        for cat in self.categories.categories:
            if cat not in df.columns:
                df[cat] = float("nan")  # TODO: replace with pd.NA when supported

        result = df[self.categories.categories]  # fix column order
        result = result.astype(self.dtypes)
        return result


class CSVEncoder(BaseEncoder):
    r"""Encode the data into a CSV file."""

    def __init__(self, filename: str, header: bool = True) -> None:
        r"""Initialize the encoder.

        Parameters
        ----------
        filename: str
        header: bool = True
        """
        super().__init__()
        self.filename = filename
        self.header = header

    def fit(self, data: DataFrame) -> None:
        r"""Fit the encoder.

        Parameters
        ----------
        data
        """
        self.dtypes = data.dtypes
        self.columns = data.columns

    def encode(self, df: DataFrame) -> DataFrame:
        r"""Encode the data."""
        df.to_csv(self.filename, index=False, header=self.header)
        return df

    def decode(self, fname: Optional[str] = None, /) -> DataFrame:
        r"""Decode the data."""
        if fname is None:
            fname = self.filename
        frame = DataFrame(pd.read_csv(fname)).astype(self.dtypes)
        return frame
