r"""Implementation of encoders.

Notes
-----
Contains encoders in modular form.
  - See :mod:`tsdm.encoders.functional` for functional implementations.
"""

__all__ = [
    # ABC
    "BaseEncoder",
    # Classes
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
]

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import singledispatchmethod
from typing import Any, Final, Literal, Optional, Union, overload

import numpy as np
import pandas.api.types
import torch
from pandas import NA, DataFrame, DatetimeIndex, Index, Series, Timedelta, Timestamp
from torch import Tensor

from tsdm.datasets import TimeTensor

__logger__ = logging.getLogger(__name__)


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

    def __init__(self, unit: str = "s"):
        """Initialize the parameters.

        Parameters
        ----------
        unit: Optional[str]=None
        """
        super().__init__()
        self.unit = unit

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
        print(self.unit)
        converted = pandas.to_timedelta(timedeltas, unit=self.unit)
        datetimes = Series(converted + self.offset, name=self.name, dtype=self.dtype)
        if self.kind == Series:
            return datetimes
        return DatetimeIndex(
            datetimes, freq=self.freq, name=self.name, dtype=self.dtype
        )


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
    colspec: Series = None
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
            raise NotImplementedError("Index encoders not yet supported")

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

            _encoders = tuple(set(self.column_encoders.values()))
            encoders = Series(_encoders, name="encoder")
            partitions = Series(range(len(_encoders)), name="partition")

            _columns = defaultdict(list)
            for key, encoder in self.column_encoders.items():
                _columns[encoder].append(key)

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
            keys = set(df.columns)
            _keys = set(self.column_encoders.keys())
            assert keys <= _keys, f"Missing encoders for columns {keys - _keys}!"
            assert (
                keys >= _keys
            ), f"Encoder given for non-existent columns {_keys- keys}!"

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


PandasObject = Union[Index, Series, DataFrame]


class TensorEncoder(BaseEncoder):
    r"""Converts objects to Tensor."""

    dtype: torch.dtype
    r"""The default dtype."""
    device: torch.device
    r"""The device the tensors are stored in."""
    # colspecs: list[Series]
    # """The data types/column names of all the tensors"""

    def __init__(
        self, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None
    ):
        super().__init__()

        self.dtype = torch.float32 if dtype is None else dtype
        # default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu") if device is None else device

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

    def encode(self, data: tuple[PandasObject, ...]) -> tuple[Tensor, ...]:
        r"""Convert each inputs to tensor."""
        return tuple(
            torch.tensor(ndarray.values, device=self.device, dtype=self.dtype)
            for ndarray in data
        )

    def decode(self, data: tuple[Tensor, ...]) -> tuple[PandasObject, ...]:
        r"""Convert each input from tensor to numpy."""
        return tuple(tensor.cpu().numpy() for tensor in data)

    def __repr__(self):
        r"""Pretty print."""
        return f"{self.__class__.__name__}()"


class FloatEncoder(BaseEncoder):
    """Converts all columns of DataFrame to float32."""

    dtypes: Series = None
    r"""The original dtypes."""

    def fit(self, data: PandasObject):
        r"""Remember the original dtypes."""
        self.dtypes = data.dtypes

    def encode(self, data: PandasObject) -> PandasObject:
        r"""Make everything float32."""
        return data.astype("float32")

    def decode(self, data, /):
        r"""Restore original dtypes."""
        return data.astype(self.dtypes)

    def __repr__(self):
        """Pretty print."""
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
