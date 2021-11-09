r"""#TODO add module summary line.

#TODO add module description.
"""

from __future__ import annotations

__all__ = [
    # Classes
    "BaseEncoder",
    "Time2Float",
    "DateTimeEncoder",
    "Standardizer",
    "DataFrameEncoder",
    "ChainedEncoder",
]

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Literal, Optional, Union

import numpy
import numpy as np
import pandas.api.types
import torch
from pandas import (
    NA,
    DataFrame,
    DatetimeIndex,
    Index,
    Series,
    Timedelta,
    Timestamp,
)
from torch import Tensor

__logger__ = logging.getLogger(__name__)


class BaseEncoder(ABC):
    """Base class that all encoders must subclass."""

    # TODO: Implement __matmul__ @ for composition of encoders.

    def __init__(self):
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

    def fit(self, data, *args, **kwargs):
        r"""Fit to the data."""
        for encoder in reversed(self.chained):
            encoder.fit(data)
            data = encoder.encode(data)

    def encode(self, data, *args, **kwargs):
        r"""Encode the input."""
        for encoder in reversed(self.chained):
            data = encoder.encode(data)
        return data

    def decode(self, data, *args, **kwargs):
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

    Parameters
    ----------
    ds: Series

    Returns
    -------
    Series
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

    def fit(self, ds: Series):
        r"""Fit to the data.

        Parameters
        ----------
        ds: Series
        """
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
    offset: Timestamp
    r"""The starting point of the timeseries."""
    kind: Union[Series, DatetimeIndex] = None
    r"""Whether to encode as index of Series."""
    name: Optional[str] = None
    r"""The name of the original Series."""
    dtype: np.dtype
    r"""The original dtype of the Series."""

    def __init__(self, unit: str = "s"):
        """Initialize the parameters.

        Parameters
        ----------
        unit: Optional[str]=None
        """
        super().__init__()
        self.unit = unit

    def fit(self, datetimes: Union[Series, DatetimeIndex]):
        r"""Store the offset."""
        if isinstance(datetimes, Series):
            self.kind = Series
        elif isinstance(datetimes, DatetimeIndex):
            self.kind = DatetimeIndex
        else:
            raise ValueError(f"Incompatible {type(datetimes)=}")

        self.offset = Timestamp(datetimes[0])
        self.name = datetimes.name
        self.dtype = datetimes.dtype

    def encode(self, datetimes: Union[Series, DatetimeIndex]) -> Series:
        r"""Encode the input."""
        return (datetimes - self.offset) / Timedelta(1, unit=self.unit)

    def decode(self, timedeltas: Series) -> Union[Series, DatetimeIndex]:
        r"""Decode the input."""
        converted = pandas.to_timedelta(timedeltas, unit=self.unit)
        return self.kind(converted + self.offset, name=self.name, dtype=self.dtype)


class IdentityEncoder(BaseEncoder):
    r"""Dummy class that performs identity function."""

    def __init__(self):
        r"""Initialize the encoder."""
        super().__init__()

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

    column_encoder: Union[BaseEncoder, dict[Any, BaseEncoder]]
    r"""Encoders for the columns."""
    index_encoder: Optional[BaseEncoder] = None
    r"""Optional Encoder for the index."""
    colspec: Series = None
    r"""The columns-specification of the DataFrame."""
    encode_index: bool
    r"""Whether to encode the index."""
    column_wise: bool
    r"""Whether to encode column-wise"""
    partitions: Optional[dict] = None
    r"""Contains partitions if used column wise"""

    def __init__(
        self,
        encoders: Union[BaseEncoder, dict[Any, BaseEncoder]],
        *,
        index_encoder: Optional[BaseEncoder] = None,
    ):
        r"""Set up the individual encoders.

        Note: the same encoder instance can be used for multiple columns.

        Parameters
        ----------
        encoders
        index_encoder
        """
        super().__init__()
        self.column_encoder = encoders
        self.index_encoder = index_encoder
        self.column_wise: bool = isinstance(self.column_encoder, dict)
        self.encode_index: bool = index_encoder is not None

        if self.encode_index:
            _idxenc_spec = {
                "col": NA,
                "encoder": self.index_encoder,
            }
            idxenc_spec = DataFrame.from_records(
                _idxenc_spec, index=Index([NA], name="partition")
            )
        else:
            idxenc_spec = DataFrame(
                columns=["col", "encoder"],
                index=Index([], name="partition"),
            )

        if not isinstance(self.column_encoder, dict):
            _colenc_spec = {
                "col": NA,
                "encoder": self.column_encoder,
            }
            colenc_spec = DataFrame.from_records(
                _colenc_spec, index=Index([0], name="partition")
            )
        else:
            keys = self.column_encoder.keys()
            assert len(set(keys)) == len(keys), "Some keys are duplicates!"

            _encoders = tuple(set(self.column_encoder.values()))
            encoders = Series(_encoders, name="encoder")
            partitions = Series(range(len(_encoders)), name="partition")

            _columns = defaultdict(list)
            for key, encoder in self.column_encoder.items():
                _columns[encoder].append(key)

            columns = Series(_columns, name="col")

            colenc_spec = DataFrame(encoders, index=partitions)
            colenc_spec = colenc_spec.join(columns, on="encoder")

        self.spec = pandas.concat(
            [idxenc_spec, colenc_spec],
            keys=["index", "columns"],
            names=["section", "partition"],
        )

        # add extra repr options by cloning from spec.
        for x in [
            "_repr_data_resource_",
            "_repr_fits_horizontal_",
            "_repr_fits_vertical_",
            "_repr_html_",
            "_repr_latex_",
        ]:
            setattr(self, x, getattr(self.spec, x))

    def fit(self, df: DataFrame, /):
        r"""Fit to the data."""
        self.colspec = df.dtypes

        if self.index_encoder is not None:
            self.index_encoder.fit(df.index)

        if isinstance(self.column_encoder, dict):
            # check if cols are a proper partition.
            keys = set(df.columns)
            _keys = set(self.column_encoder.keys())
            assert keys <= _keys, f"Missing encoders for columns {keys - _keys}!"
            assert (
                keys >= _keys
            ), f"Encoder given for non-existent columns {_keys- keys}!"

            for _, series in self.spec.loc["columns"].iterrows():
                encoder = series["encoder"]
                cols = series["col"]
                encoder.fit(df[cols])
        else:
            self.spec.loc["columns", "col"] = cols = list(df.columns)
            encoder = self.spec.loc["columns", "encoder"].item()
            encoder.fit(df.values)

    def encode(self, df: DataFrame, /) -> tuple:
        r"""Encode the input."""
        tensors = []
        for _, series in self.spec.loc["columns"].iterrows():
            encoder = series["encoder"]
            cols = series["col"]
            tensors.append(encoder.encode(df[cols]))
        encoded_columns = tuple(tensors)

        if self.index_encoder is not None:
            encoded_index = self.index_encoder.encode(df.index)
            return encoded_index, *encoded_columns
        return encoded_columns

    def decode(self, data: tuple, /) -> DataFrame:
        r"""Decode the input."""
        if self.encode_index:
            encoder = self.spec.loc["index", "encoder"].item()
            index = encoder.decode(data[0])
            data = data[1:]
        else:
            index = None

        columns = []
        col_names = []
        for partition, series in self.spec.loc["columns"].iterrows():
            tensor = data[partition]
            encoder = series["encoder"]
            columns.append(encoder.decode(tensor))
            col_names += series["col"]
        # return columns, col_names
        values = np.hstack(columns)
        df = DataFrame(values, index=index, columns=col_names)
        return df[self.colspec.index].astype(self.colspec)  # bring cols in right order

    def __repr__(self) -> str:
        """Pretty print."""
        return repr(self.spec)


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

    def __init__(self, ignore_nan: bool = True, axis: Union[int, tuple[int, ...]] = -1):
        super().__init__()
        self.ignore_nan = ignore_nan
        self.axis = axis if isinstance(axis, tuple) else (axis,)

    def fit(self, data):
        r"""Compute the mean and stdv."""
        rank = len(data.shape)
        selection = [(a % rank) for a in self.axis]
        axes = tuple(k for k in range(rank) if k not in selection)

        if isinstance(data, Tensor):
            if self.ignore_nan:
                self.mean = torch.nanmean(data, dim=axes)
                self.stdv = torch.nanmean((data - self.mean) ** 2, dim=axes)
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


# class ConcatEncoder(BaseEncoder):
#     def __init__(self, axis, framework: Literal[torch, numpy, pandas]):
#         """
#
#         Parameters
#         ----------
#         axis
#         framework
#         """
#         super().__init__()
#
#     def fit(self, *data):
#         ...
#         # if isinstance(data[0])
#
#     def encode(self, data, /):
#         pass
#
#     def decode(self, data, /):
#         pass

# class TensorEncoder(BaseEncoder):
#     r"""Converts objects to Tensor."""
#
#     colspecs: list[Series]
#     """The data types/column names of all the tensors"""
#
#     def __init__(self, dtype: Optional[torch.dtype]=None, device: Optional[torch.device] = None):
#         super().__init__()
#         self.colspecs = []
#
#     def fit(self, *data: tuple):
#         r"""Fit to the data."""
#         for ndarray in data:
#             self.
#
#
#     def encode(self, *data: tuple[Union[Series, DataFrame]]) -> tuple[Tensor, ...]:
#         r"""Converts each input from ? to tensor"""
#
#
#     def decode(self, *data: tuple[Tensor, ...]) -> tuple[Union[Series, DataFrame]]:
#         r"""Converts each input from tensor to original"""
