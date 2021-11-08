r"""#TODO add module summary line.

#TODO add module description.
"""

from __future__ import annotations

__all__ = [
    # Classes
    "BaseEncoder",
    "Time2Float",
    "DateTimeEncoder",
]

import logging
from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas.api.types
from pandas import DatetimeIndex, Series, Timedelta, Timestamp
from torch import Tensor

__logger__ = logging.getLogger(__name__)


class BaseEncoder(ABC):
    """Base class that all encoders must subclass."""

    @abstractmethod
    def fit(self, data):
        r"""Fit parameters to given data."""

    @abstractmethod
    def encode(self, data):
        r"""Transform the data."""

    @abstractmethod
    def decode(self, data):
        r"""Reverse the applied transformation."""

    # transform = encode
    # fit_transform = encode
    # inverse_transform = decode


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

    def decode(self, ds: Series) -> Series:
        """Apply the inverse transformation.

        Roughly equal to:
        ``ds ⟶ (scale*ds + offset).astype(original_dtype)``
        """


class DateTimeEncoder(BaseEncoder):
    r"""Encode TimeSeries as TimeTensor."""

    unit: str
    r"""The base frequency to convert timedeltas to."""
    offset: Timestamp
    r"""The starting point of the timeseries."""
    kind: Union[Series, DatetimeIndex] = None
    r"""Whether to encode as index of Series."""
    name: Optional[str] = None
    r"""The name of the original Series."""
    dtype: np.dtype
    r"""The original dtype of the Series."""

    def __init__(self, unit: Optional[str] = None):
        """Initialize the parameters.

        Parameters
        ----------
        unit: Optional[str]=None
        """
        self.unit = "s" if unit is None else unit

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

    def encode(self, datetimes: Union[Series, DatetimeIndex]) -> Tensor:
        r"""Encode the input."""
        timedeltas = (datetimes - self.offset) / Timedelta(1, unit=self.unit)
        return Tensor(timedeltas)

    def decode(self, timedeltas: Tensor) -> Union[Series, DatetimeIndex]:
        r"""Decode the input."""
        converted = pandas.to_timedelta(timedeltas.cpu(), unit=self.unit)
        return self.kind(converted + self.offset, name=self.name, dtype=self.dtype)


# class DataFrameEncoder(BaseEncoder):
#     r"""Combine multiple encoders into a single one."""
#
#     def __init__(
#         self,
#         index_encoder: BaseEncoder,
#         column_encoders: dict[Union[str, tuple[str, ...]], BaseEncoder],
#     ):
#         """
#
#         Parameters
#         ----------
#         index_encoder
#         column_encoders
#         """
#         self.index_encoder = index_encoder
#         self.column_encoders = column_encoders
#
#     def fit(self, df: DataFrame):
#         """
#
#         Parameters
#         ----------
#         df
#         """
#
#     def encode(self, df: DataFrame) -> DataFrame:
#         ...
#
#     def decode(self, df: DataFrame) -> DataFrame:
#         ...
