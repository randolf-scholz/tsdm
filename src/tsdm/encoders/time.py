r"""Encoders for timedelta and datetime types."""

__all__ = [
    # Classes
    "DateTimeEncoder",
    "PeriodicEncoder",
    "PeriodicSocialTimeEncoder",
    "PositionalEncoder",
    "SocialTimeEncoder",
    "TimeDeltaEncoder",
]

from collections.abc import Hashable
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import pandas as pd
import pyarrow as pa
from numpy.typing import NDArray
from pandas import DataFrame, Index, Series
from typing_extensions import Any, ClassVar, Final, Optional

from tsdm.encoders.base import BaseEncoder
from tsdm.encoders.dataframe import FrameEncoder
from tsdm.types.aliases import DType, PandasDtype
from tsdm.types.time import DateTime, TimeDelta
from tsdm.utils import timedelta, timestamp

SeriesOrIndex = TypeVar("SeriesOrIndex", Index, Series)


class TimeDeltaEncoder(BaseEncoder[SeriesOrIndex, SeriesOrIndex]):
    r"""Encode TimeDelta as Float."""

    unit: pd.Timedelta = NotImplemented
    r"""The base frequency to convert timedeltas to."""
    original_dtype: PandasDtype = NotImplemented
    r"""The original dtype of the Series."""

    @property
    def params(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "original_dtype": self.original_dtype,
        }

    def __init__(self, *, unit: str | TimeDelta = NotImplemented) -> None:
        self.unit = NotImplemented if unit is NotImplemented else timedelta(unit)

    def fit(self, data: SeriesOrIndex, /) -> None:
        self.original_dtype = data.dtype

        if self.unit is NotImplemented:
            # FIXME: https://github.com/pandas-dev/pandas/issues/58403
            # This looks awkward but is robust.
            base_freq = np.gcd.reduce(data.dropna().astype(int))
            self.unit = Index([base_freq]).astype(self.original_dtype).item()

    def encode(self, data: SeriesOrIndex, /) -> SeriesOrIndex:
        return data / self.unit

    def decode(self, data: SeriesOrIndex, /) -> SeriesOrIndex:
        # FIXME: https://github.com/apache/arrow/issues/39233#issuecomment-2070756267
        try:
            return (data * self.unit).astype(self.original_dtype)
        except pa.lib.ArrowNotImplementedError:
            return data.astype(float).__mul__(self.unit).astype(self.original_dtype)


@dataclass(init=False)
class DateTimeEncoder(BaseEncoder[SeriesOrIndex, SeriesOrIndex]):
    r"""Encode Datetime as Float."""

    offset: DateTime = NotImplemented
    r"""The starting point of the timeseries."""
    unit: pd.Timedelta = NotImplemented
    r"""The base frequency to convert timedeltas to."""
    original_dtype: PandasDtype = NotImplemented
    r"""The original dtype of the Series."""

    @property
    def params(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "offset": self.offset,
            "original_dtype": self.original_dtype,
        }

    def __init__(
        self,
        *,
        unit: str | TimeDelta = NotImplemented,
        offset: str | TimeDelta = NotImplemented,
    ) -> None:
        self.unit = NotImplemented if unit is NotImplemented else timedelta(unit)
        self.offset = NotImplemented if offset is NotImplemented else timestamp(offset)

    def fit(self, data: SeriesOrIndex, /) -> None:
        self.original_dtype = data.dtype

        if self.offset is NotImplemented:
            self.offset = data.min()
        if self.unit is NotImplemented:
            # FIXME: https://github.com/pandas-dev/pandas/issues/58403
            deltas = data - self.offset
            # This looks awkward but is robust.
            base_freq = np.gcd.reduce(deltas.dropna().astype(int))
            self.unit = Index([base_freq]).astype(deltas.dtype).item()

    def encode(self, data: SeriesOrIndex, /) -> SeriesOrIndex:
        return (data - self.offset) / self.unit

    def decode(self, data: SeriesOrIndex, /) -> SeriesOrIndex:
        # FIXME: https://github.com/apache/arrow/issues/39233#issuecomment-2070756267
        try:
            return (data * self.unit + self.offset).astype(self.original_dtype)
        except pa.lib.ArrowNotImplementedError:
            return (
                data.astype(float)
                .__mul__(self.unit)
                .__add__(self.offset)
                .astype(self.original_dtype)
            )


class PositionalEncoder(BaseEncoder[NDArray, NDArray]):
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
    scales: Final[NDArray]
    r"""Scale factors for positional encoding."""

    @property
    def params(self) -> dict[str, Any]:
        return {
            "num_dim": self.num_dim,
            "scale": self.scale,
            "scaled": self.scales,
        }

    def __init__(self, num_dim: int, scale: float) -> None:
        self.num_dim = num_dim
        self.scale = float(scale)
        self.scales = self.scale ** (-np.arange(0, num_dim + 2, 2) / num_dim)
        assert self.scales[0] == 1.0, "Something went wrong."

    def encode(self, data: NDArray, /) -> NDArray:
        r""".. Signature: ``... -> (..., 2d)``.

        Note: we simply concatenate the sin and cosine terms without interleaving them.
        """
        z = np.einsum("..., d -> ...d", data, self.scales)
        return np.concatenate([np.sin(z), np.cos(z)], axis=-1)

    def decode(self, data: NDArray, /) -> NDArray:
        r""".. signature:: ``(..., 2d) -> ...``."""
        return np.arcsin(data[..., 0])


class PeriodicEncoder(BaseEncoder[Series, DataFrame]):
    r"""Encode periodic data as sin/cos waves."""

    period: float = NotImplemented
    freq: float = NotImplemented
    dtype: DType = NotImplemented
    colname: Hashable = NotImplemented

    @property
    def params(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "freq": self.freq,
            "dtype": self.dtype,
            "colname": self.colname,
        }

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


class SocialTimeEncoder(BaseEncoder[Series, DataFrame]):
    r"""Social time encoding."""

    LEVEL_CODES: ClassVar[dict[str, str]] = {
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

    @property
    def params(self) -> dict[str, Any]:
        return {
            "level_code": self.level_code,
            "levels": self.levels,
            "original_name": self.original_name,
            "original_dtype": self.original_dtype,
            "original_type": self.original_type,
            "rev_cols": self.rev_cols,
        }

    def __init__(self, levels: str = "YMWDhms") -> None:
        self.level_code = levels
        self.levels = [self.LEVEL_CODES[k] for k in levels]

    def fit(self, x: Series, /) -> None:
        r"""Fit the encoder."""
        self.original_type = type(x)
        self.original_name = x.name
        self.original_dtype = x.dtype
        self.rev_cols = [level for level in self.levels if level != "weekday"]

    def encode(self, x: Series, /) -> DataFrame:
        r"""Encode the data."""
        return DataFrame.from_dict({level: getattr(x, level) for level in self.levels})

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

    frequencies = {
        "year"        : 1,
        "month"       : 12,
        "weekday"     : 7,
        "day"         : 365,
        "hour"        : 24,
        "minute"      : 60,
        "second"      : 60,
        "microsecond" : 1000,
        "nanosecond"  : 1000,
    }  # fmt: skip
    r"""The frequencies of the used `PeriodicEncoder`."""

    def __new__(cls, levels: str = "YMWDhms") -> BaseEncoder[Series, DataFrame]:
        r"""Construct a new encoder object."""
        self = super().__new__(cls)
        self.__init__(levels)  # type: ignore[misc]
        column_encoders = {
            level: PeriodicEncoder(period=self.frequencies[level])
            for level in self.levels
        }
        return self >> FrameEncoder(column_encoders=column_encoders)
