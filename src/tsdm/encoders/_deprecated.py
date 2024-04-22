from copy import deepcopy
from typing import Any, ClassVar, Optional

import numpy as np
import pandas as pd
from pandas import DatetimeIndex, Series
from typing_extensions import Literal, deprecated

from tsdm.encoders.base import BaseEncoder


@deprecated("decode unimplemented")
class Time2Float(BaseEncoder):
    r"""Convert `Series` encoded as `datetime64` or `timedelta64` to `floating`.

    By default, the data is mapped onto the unit interval $[0,1]$.
    """

    requires_fit: ClassVar[bool] = True

    original_dtype: np.dtype
    offset: Any
    common_interval: Any
    scale: Any

    def __init__(self, normalization: Literal["gcd", "max", "none"] = "max") -> None:
        r"""Choose the normalization scheme."""
        self.normalization = normalization

    def fit(self, data: Series, /) -> None:
        ds = data
        self.original_dtype = ds.dtype
        self.offset = deepcopy(ds[0])
        assert (
            ds.is_monotonic_increasing
        ), "Time-Values must be monotonically increasing!"

        assert not (
            pd.api.types.is_float_dtype(self.original_dtype)
            and self.normalization == "gcd"
        ), f"{self.normalization=} illegal when original dtype is floating."

        if pd.api.types.is_datetime64_dtype(ds):
            ds = ds.view("datetime64[ns]")
            self.offset = deepcopy(ds[0])
            timedeltas = ds - self.offset
        elif pd.api.types.is_timedelta64_dtype(ds) or pd.api.types.is_integer_dtype(ds):
            timedeltas = ds.view("timedelta64[ns]")
        elif pd.api.types.is_float_dtype(ds):
            self.LOGGER.warning("Array is already floating dtype.")
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
        r"""Roughly equal to `ds ⟶ ((ds - offset)/scale).astype(float)``."""
        if pd.api.types.is_integer_dtype(data):
            return data
        if pd.api.types.is_datetime64_dtype(data):
            data = data.view("datetime64[ns]")
            self.offset = data[0].copy()
            timedeltas = data - self.offset
        elif pd.api.types.is_timedelta64_dtype(data):
            timedeltas = data.view("timedelta64[ns]")
        elif pd.api.types.is_float_dtype(data):
            self.LOGGER.warning("Array is already floating dtype.")
            return data
        else:
            raise ValueError(f"{data.dtype=} not supported")

        common_interval = np.gcd.reduce(timedeltas.view(int)).view("timedelta64[ns]")

        return (timedeltas / common_interval).astype(float)

    def decode(self, data: Series, /) -> Series:
        r"""Roughly equal to: ``ds ⟶ (scale*ds + offset).astype(original_dtype)``."""
        raise NotImplementedError


@deprecated("use DateTimeEncoder")
class OldDateTimeEncoder(BaseEncoder):
    r"""Encode DateTime as Float."""

    requires_fit: ClassVar[bool] = True

    unit: str = "s"
    r"""The base frequency to convert timedeltas to."""
    base_freq: str = "s"
    r"""The frequency the decoding should be rounded to."""
    offset: pd.Timestamp
    r"""The starting point of the timeseries."""
    kind: type[Series] | type[DatetimeIndex]
    r"""Whether to encode as index of Series."""
    name: Optional[str] = None
    r"""The name of the original Series."""
    original_dtype: np.dtype
    r"""The original dtype of the Series."""
    freq: Optional[Any] = None
    r"""The frequency attribute in case of DatetimeIndex."""

    def __init__(self, unit: str = "s", base_freq: str = "s") -> None:
        self.unit = unit
        self.base_freq = base_freq

    def fit(self, data: Series | DatetimeIndex, /) -> None:
        match data:
            case Series():
                self.kind = Series
            case DatetimeIndex():
                self.kind = DatetimeIndex
            case _:
                raise TypeError(f"Incompatible {type(data)=}")

        self.offset = pd.Timestamp(Series(data).iloc[0])
        self.name = None if data.name is None else str(data.name)
        self.original_dtype = data.dtype
        if isinstance(data, DatetimeIndex):
            self.freq = data.freq

    def encode(self, data: Series | DatetimeIndex, /) -> Series:
        # FIXME: remove this patch with pyarrow 14
        # https://github.com/apache/arrow/issues/36789
        if isinstance(data.dtype, pd.ArrowDtype):
            data = data.astype("datetime64[ns]")

        return (data - self.offset) / pd.Timedelta(1, unit=self.unit)

    def decode(self, data: Series, /) -> Series | DatetimeIndex:
        converted = pd.to_timedelta(data, unit=self.unit)
        datetimes = Series(
            converted + self.offset, name=self.name, dtype=self.original_dtype
        )
        if self.kind == Series:
            return datetimes.round(self.base_freq)
        return DatetimeIndex(
            datetimes, freq=self.freq, name=self.name, dtype=self.original_dtype
        ).round(self.base_freq)

    def __repr__(self) -> str:
        r"""Return a string representation."""
        return f"{self.__class__.__name__}(unit={self.unit!r})"
