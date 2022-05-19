r"""#TODO add module summary line.

#TODO add module description.
"""
from __future__ import annotations

__all__ = [
    # Classes
    "Time2Float",
    "DateTimeEncoder",
    "TimeDeltaEncoder",
    "FloatEncoder",
    "IntEncoder",
]


import logging
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas
from pandas import DatetimeIndex, Series, Timedelta, Timestamp

from tsdm.encoders.modular.generic import BaseEncoder
from tsdm.util.types._types import PandasObject

__logger__ = logging.getLogger(__name__)


class Time2Float(BaseEncoder):
    r"""Convert `Series` encoded as `datetime64` or `timedelta64` to `floating`.

    By default, the data is mapped onto the unit interval $[0,1]$.
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
        r"""Encode the data.

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
        r"""Apply the inverse transformation.

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
