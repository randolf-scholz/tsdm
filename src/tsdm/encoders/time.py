r"""Encoders for timedelta and datetime types."""

__all__ = [
    # Classes
    "OldDateTimeEncoder",
    "PeriodicEncoder",
    "PositionalEncoder",
    "SocialTimeEncoder",
    "Time2Float",
    "TimeDeltaEncoder",
]

from collections.abc import Hashable
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas import DataFrame, DatetimeIndex, Series
from typing_extensions import (
    Any,
    ClassVar,
    Final,
    Literal,
    Optional,
    Self,
    cast,
    deprecated,
)

from tsdm.encoders.base import BaseEncoder
from tsdm.encoders.dataframe import FrameEncoder
from tsdm.types.aliases import DType, PandasObject
from tsdm.types.protocols import NumericalArray
from tsdm.types.time import DT, TD, DateTime, TimeDelta


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
        elif pd.api.types.is_timedelta64_dtype(ds):
            timedeltas = ds.view("timedelta64[ns]")
        elif pd.api.types.is_integer_dtype(ds):
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


@dataclass(init=False)
class DateTimeEncoder(BaseEncoder[NumericalArray, NumericalArray]):
    unit: TimeDelta
    r"""The base frequency to convert timedeltas to."""
    base_freq: TimeDelta
    r"""The frequency the decoding should be rounded to."""
    offset: DateTime
    r"""The starting point of the timeseries."""

    original_dtype: ...
    r"""The original dtype of the Series."""

    def __init__(
        self, *, unit: str | TimeDelta = "s", base_freq: str | TimeDelta = "s"
    ) -> None: ...

    def fit(self, data: NumericalArray[DT], /) -> None: ...

    def encode(self, data: NumericalArray[DT], /) -> NumericalArray[TD]:
        # FIXME: remove this patch with pyarrow 14
        # https://github.com/apache/arrow/issues/36789
        if isinstance(data.dtype, pd.ArrowDtype):
            data = data.astype("datetime64[ns]")

        return (data - self.offset) / pd.Timedelta(1, unit=self.unit)

    def decode(self, data: NumericalArray[TD], /) -> NumericalArray[DT]:
        (data * pd.Timedelta(1, unit=self.unit) + self.offset).round(self.base_freq)

        converted = pd.to_timedelta(data, unit=self.unit)
        datetimes = Series(
            converted + self.offset, name=self.name, dtype=self.original_dtype
        )
        if self.kind == Series:
            return datetimes.round(self.base_freq)
        return DatetimeIndex(
            datetimes, freq=self.freq, name=self.name, dtype=self.original_dtype
        ).round(self.base_freq)


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


class TimeDeltaEncoder(BaseEncoder):
    r"""Encode TimeDelta as Float."""

    requires_fit: ClassVar[bool] = True

    unit: str = "s"
    r"""The base frequency to convert timedeltas to."""
    base_freq: str = "s"
    r"""The frequency the decoding should be rounded to."""
    original_dtype: Any

    def __init__(self, *, unit: str = "s", base_freq: str = "s") -> None:
        self.unit = unit
        self.base_freq = base_freq
        self.timedelta = pd.Timedelta(1, unit=self.unit)

    def encode(self, data: PandasObject, /) -> PandasObject:
        return data.astype("timedelta64[ns]") / self.timedelta

    def decode(self, data: PandasObject, /) -> PandasObject:
        result = data * self.timedelta
        if hasattr(result, "apply"):
            return result.apply(lambda x: x.round(self.base_freq))
        if hasattr(result, "map"):
            return result.map(lambda x: x.round(self.base_freq))
        return result


class PositionalEncoder(BaseEncoder):
    r"""Positional encoding.

    .. math::
        x_{2 k}(t)   &:=\sin \left(\frac{t}{t^{2 k / τ}}\right) \\
        x_{2 k+1}(t) &:=\cos \left(\frac{t}{t^{2 k / τ}}\right)
    """

    requires_fit: ClassVar[bool] = False

    # Constants
    num_dim: Final[int]
    r"""Number of dimensions."""

    # Buffers
    scale: Final[float]
    r"""Scale factor for positional encoding."""
    scales: Final[np.ndarray]
    r"""Scale factors for positional encoding."""

    def __init__(self, num_dim: int, scale: float) -> None:
        self.num_dim = num_dim
        self.scale = float(scale)
        self.scales = self.scale ** (-np.arange(0, num_dim + 2, 2) / num_dim)
        assert self.scales[0] == 1.0, "Something went wrong."

    def encode(self, data: np.ndarray, /) -> np.ndarray:
        r""".. Signature: ``... -> (..., 2d)``.

        Note: we simply concatenate the sin and cosine terms without interleaving them.
        """
        z = np.einsum("..., d -> ...d", data, self.scales)
        return np.concatenate([np.sin(z), np.cos(z)], axis=-1)

    def decode(self, data: np.ndarray, /) -> np.ndarray:
        r""".. Signature:: ``(..., 2d) -> ...``."""
        return np.arcsin(data[..., 0])


class PeriodicEncoder(BaseEncoder):
    r"""Encode periodic data as sin/cos waves."""

    requires_fit: ClassVar[bool] = True

    period: float
    freq: float
    dtype: DType
    colname: Hashable

    def __init__(self, period: Optional[float] = None) -> None:
        self._period = period

    def fit(self, x: Series) -> None:
        r"""Fit the encoder."""
        self.dtype = x.dtype
        self.colname = x.name
        self.period = x.max() + 1 if self._period is None else self._period
        self._period = self.period
        self.freq = 2 * np.pi / self.period

    def encode(self, x: Series) -> DataFrame:
        r"""Encode the data."""
        x = self.freq * (x % self.period)  # ensure 0...N-1
        return DataFrame(
            np.stack([np.cos(x), np.sin(x)]).T,
            columns=[f"cos_{self.colname}", f"sin_{self.colname}"],
        )

    def decode(self, x: DataFrame) -> Series:
        r"""Decode the data."""
        x = np.arctan2(x[f"sin_{self.colname}"], x[f"cos_{self.colname}"])
        x = (x / self.freq) % self.period
        return Series(x, dtype=self.dtype, name=self.colname)

    def __repr__(self) -> str:
        r"""Pretty-print."""
        return f"{self.__class__.__name__}({self._period})"


class SocialTimeEncoder(BaseEncoder):
    r"""Social time encoding."""

    requires_fit: ClassVar[bool] = True

    level_codes = {
        "Y": "year",
        "M": "month",
        "W": "weekday",
        "D": "day",
        "h": "hour",
        "m": "minute",
        "s": "second",
        "µ": "microsecond",
        "n": "nanosecond",
    }
    original_name: Hashable
    original_dtype: DType
    original_type: type
    rev_cols: list[str]
    level_code: str
    levels: list[str]

    def __init__(self, levels: str = "YMWDhms") -> None:
        self.level_code = levels
        self.levels = [self.level_codes[k] for k in levels]

    def fit(self, x: Series, /) -> None:
        r"""Fit the encoder."""
        self.original_type = type(x)
        self.original_name = x.name
        self.original_dtype = x.dtype
        self.rev_cols = [level for level in self.levels if level != "weekday"]

    def encode(self, x: Series, /) -> DataFrame:
        r"""Encode the data."""
        if isinstance(x, DatetimeIndex):
            res = {level: getattr(x, level) for level in self.levels}
        else:
            res = {level: getattr(x, level) for level in self.levels}
        return DataFrame.from_dict(res)

    def decode(self, x: DataFrame, /) -> Series:
        r"""Decode the data."""
        x = x[self.rev_cols]
        s = pd.to_datetime(x)
        return self.original_type(s, name=self.original_name, dtype=self.original_dtype)

    def __repr__(self) -> str:
        r"""Pretty print."""
        return f"SocialTimeEncoder({self.level_code!r})"


class PeriodicSocialTimeEncoder(SocialTimeEncoder):
    r"""Combines SocialTimeEncoder with PeriodicEncoder using the right frequencies."""

    requires_fit: ClassVar[bool] = True

    frequencies = {
        "year": 1,
        "month": 12,
        "weekday": 7,
        "day": 365,
        "hour": 24,
        "minute": 60,
        "second": 60,
        "microsecond": 1000,
        "nanosecond": 1000,
    }
    r"""The frequencies of the used `PeriodicEncoder`."""

    def __new__(cls, levels: str = "YMWDhms") -> Self:
        r"""Construct a new encoder object."""
        self = super().__new__(cls)
        self.__init__(levels)  # type: ignore[misc]
        column_encoders = {
            level: PeriodicEncoder(period=self.frequencies[level])
            for level in self.levels
        }
        obj = FrameEncoder(column_encoders=column_encoders) @ self
        return cast(Self, obj)
