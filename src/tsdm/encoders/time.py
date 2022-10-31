r"""Encoders for timedelta and datetime types."""

from __future__ import annotations

__all__ = [
    # Classes
    "Time2Float",
    "DateTimeEncoder",
    "TimeDeltaEncoder",
    "PeriodicEncoder",
    "SocialTimeEncoder",
    "PeriodicEncoder",
    "PositionalEncoder",
    "TimeSlicer",
]

from typing import (
    Any,
    ClassVar,
    Final,
    Hashable,
    Literal,
    Optional,
    Sequence,
    cast,
    overload,
)

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame, DatetimeIndex, Series, Timedelta, Timestamp
from torch import Tensor

from tsdm.encoders._modular import FrameEncoder
from tsdm.encoders.base import BaseEncoder
from tsdm.utils.data import TimeTensor


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
        if np.issubdtype(data.dtype, np.integer):
            return data
        if np.issubdtype(data.dtype, np.datetime64):
            data = data.view("datetime64[ns]")
            self.offset = data[0].copy()
            timedeltas = data - self.offset
        elif np.issubdtype(data.dtype, np.timedelta64):
            timedeltas = data.view("timedelta64[ns]")
        elif np.issubdtype(data.dtype, np.floating):
            self.LOGGER.warning("Array is already floating dtype.")
            return data
        else:
            raise ValueError(f"{data.dtype=} not supported")

        common_interval = np.gcd.reduce(timedeltas.view(int)).view("timedelta64[ns]")

        return (timedeltas / common_interval).astype(float)

    def decode(self, data: Series, /) -> Series:
        r"""Roughly equal to: ``ds ⟶ (scale*ds + offset).astype(original_dtype)``."""


class DateTimeEncoder(BaseEncoder):
    r"""Encode DateTime as Float."""

    unit: str = "s"
    r"""The base frequency to convert timedeltas to."""
    base_freq: str = "s"
    r"""The frequency the decoding should be rounded to."""
    offset: Timestamp
    r"""The starting point of the timeseries."""
    kind: type[Series] | type[DatetimeIndex]
    r"""Whether to encode as index of Series."""
    name: Optional[str] = None
    r"""The name of the original Series."""
    dtype: np.dtype
    r"""The original dtype of the Series."""
    freq: Optional[Any] = None
    r"""The frequency attribute in case of DatetimeIndex."""

    def __init__(self, unit: str = "s", base_freq: str = "s"):
        super().__init__()
        self.unit = unit
        self.base_freq = base_freq

    def fit(self, data: Series | DatetimeIndex, /) -> None:
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

    def encode(self, data: Series | DatetimeIndex, /) -> Series:
        return (data - self.offset) / Timedelta(1, unit=self.unit)

    def decode(self, data: Series, /) -> Series | DatetimeIndex:
        self.LOGGER.debug("Decoding %s", type(data))
        converted = pd.to_timedelta(data, unit=self.unit)
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

    requires_fit: ClassVar[bool] = False

    unit: str = "s"
    r"""The base frequency to convert timedeltas to."""
    base_freq: str = "s"
    r"""The frequency the decoding should be rounded to."""

    def __init__(self, *, unit: str = "s", base_freq: str = "s"):
        super().__init__()
        self.unit = unit
        self.base_freq = base_freq
        self.timedelta = Timedelta(1, unit=self.unit)

    def encode(self, data, /):
        return data / self.timedelta

    def decode(self, data, /):
        result = data * self.timedelta
        if hasattr(result, "apply"):
            return result.apply(lambda x: x.round(self.base_freq))
        if hasattr(result, "map"):
            return result.map(lambda x: x.round(self.base_freq))
        return result


class PeriodicEncoder(BaseEncoder):
    r"""Encode periodic data as sin/cos waves."""

    period: float
    freq: float
    dtype: pd.dtype
    colname: Hashable

    def __init__(self, period: Optional[float] = None) -> None:
        super().__init__()
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

    def __repr__(self):
        r"""Pretty-print."""
        return f"{self.__class__.__name__}({self._period})"


class SocialTimeEncoder(BaseEncoder):
    r"""Social time encoding."""

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
    original_dtype: pd.dtype
    original_type: type
    rev_cols: list[str]
    level_code: str
    levels: list[str]

    def __init__(self, levels: str = "YMWDhms") -> None:
        super().__init__()
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
        return f"SocialTimeEncoder('{self.level_code}')"


class PeriodicSocialTimeEncoder(SocialTimeEncoder):
    r"""Combines SocialTimeEncoder with PeriodicEncoder using the right frequencies."""

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

    def __new__(cls, levels: str = "YMWDhms") -> PeriodicSocialTimeEncoder:
        r"""Construct a new encoder object."""
        self = super().__new__(cls)
        self.__init__(levels)  # type: ignore[misc]
        column_encoders = {
            level: PeriodicEncoder(period=self.frequencies[level])
            for level in self.levels
        }
        obj = FrameEncoder(column_encoders) @ self
        return cast(PeriodicSocialTimeEncoder, obj)


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
        super().__init__()

        self.num_dim = num_dim
        self.scale = float(scale)
        self.scales = self.scale ** (-np.arange(0, num_dim + 2, 2) / num_dim)
        assert self.scales[0] == 1.0, "Something went wrong."

    def encode(self, data: np.ndarray, /) -> np.ndarray:
        r""".. Signature: ``... -> (..., 2d)``.

        Note: we simple concatenate the sin and cosine terms without interleaving them.
        """
        z = np.einsum("..., d -> ...d", data, self.scales)
        return np.concatenate([np.sin(z), np.cos(z)], axis=-1)

    def decode(self, data: np.ndarray, /) -> np.ndarray:
        r""".. Signature:: ``(..., 2d) -> ...``."""
        return np.arcsin(data[..., 0])


class TimeSlicer(BaseEncoder):
    r"""Reorganizes the data by slicing."""

    requires_fit: ClassVar[bool] = False

    horizon: Any
    r"""The horizon of the data."""

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
        r"""Slice the data.

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
        r"""Restores the original data."""
        if isinstance(data[0], TimeTensor) or self.is_tensor_pair(data[0]):
            return torch.cat(data, dim=0)
        return tuple(self.decode(item) for item in data)
