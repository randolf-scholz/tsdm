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

from collections.abc import Hashable, Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd
import pyarrow as pa
from numpy.typing import NDArray
from pandas import DataFrame, Series
from typing_extensions import Any, ClassVar, Final, Optional, TypeVar, cast

from tsdm.encoders.base import BackendEncoder, BaseEncoder, WrappedEncoder
from tsdm.encoders.dataframe import FrameEncoder
from tsdm.types.aliases import DType, PandasDtype
from tsdm.types.protocols import NumericalTensor
from tsdm.types.time import DateTime, TimeDelta
from tsdm.utils import timedelta, timestamp
from tsdm.utils.decorators import pprint_repr

X = TypeVar("X")
Y = TypeVar("Y")
Arr = TypeVar("Arr", bound=NumericalTensor)
r"""TypeVar for tensor-like objects."""


@pprint_repr
@dataclass(init=False, slots=True)
class TimeDeltaEncoder(BackendEncoder[Arr, Arr]):
    r"""Encode TimeDelta as Float."""

    unit: TimeDelta = NotImplemented
    r"""The base frequency to convert timedeltas to."""
    timedelta_dtype: PandasDtype = NotImplemented
    r"""The original dtype of the Series."""
    round: bool = True
    r"""Whether to round to the next unit."""

    @property
    def params(self) -> dict[str, Any]:
        return asdict(self)

    def __init__(
        self,
        *,
        unit: str | TimeDelta = NotImplemented,
        rounding: bool = True,
    ) -> None:
        self.unit = NotImplemented if unit is NotImplemented else timedelta(unit)
        self.round = rounding

    def fit(self, data: Arr, /) -> None:
        self.timedelta_dtype = data.dtype

        if self.unit is NotImplemented:
            # FIXME: https://github.com/pandas-dev/pandas/issues/58403
            # This looks awkward but is robust.
            data = self.backend.drop_null(data)
            diffs = np.array(self.backend.cast(data, int))
            base_freq = int(np.gcd.reduce(diffs))

            # convert base_freq back to time delta in the original dtype
            self.unit = self.backend.scalar(base_freq, dtype=self.timedelta_dtype)

    def encode(self, x: Arr, /) -> Arr:
        try:
            return x / self.unit
        except TypeError:
            # FIXME: pyarrow: "first cast to integer before dividing date-like dtypes"
            return self.backend.cast(x, int) / self.backend.scalar(self.unit, int)

    def decode(self, y: Arr, /) -> Arr:
        if self.round:
            y = y.round()

        try:
            return self.backend.cast(y * self.unit, self.timedelta_dtype)
        except pa.lib.ArrowNotImplementedError:
            # Function 'multiply_checked' has no kernel matching input types (double, duration[ms])
            # FIXME: https://github.com/apache/arrow/issues/39233#issuecomment-2070756267
            y = self.backend.cast(y, float) * self.unit
            return self.backend.cast(y, self.timedelta_dtype)


@pprint_repr
@dataclass(init=False)
class DateTimeEncoder(BackendEncoder[Arr, Arr]):
    r"""Encode Datetime as Float."""

    offset: DateTime = NotImplemented
    r"""The starting point of the timeseries."""
    unit: TimeDelta = NotImplemented
    r"""The base frequency to convert timedeltas to."""
    datetime_dtype: Any = NotImplemented
    r"""The original dtype of the Series."""
    timedelta_dtype: Any = NotImplemented
    r"""The dtype of the timedelta."""

    @property
    def params(self) -> dict[str, Any]:
        return asdict(self)

    def __init__(
        self,
        *,
        unit: str | TimeDelta = NotImplemented,
        offset: str | TimeDelta = NotImplemented,
        rounding: bool = True,
    ) -> None:
        self.unit = NotImplemented if unit is NotImplemented else timedelta(unit)
        self.offset = NotImplemented if offset is NotImplemented else timestamp(offset)
        self.round = rounding

    def fit(self, data: Arr, /) -> None:
        # get the datetime dtype
        self.datetime_dtype = data.dtype

        # set the offset
        offset = (
            cast(DateTime, self.backend.nanmin(data))
            if self.offset is NotImplemented
            else self.offset
        )
        self.offset = self.backend.scalar(offset, dtype=self.datetime_dtype)

        # get the timedelta dtype
        deltas = self.backend.drop_null(data - self.offset)
        self.timedelta_dtype = deltas.dtype

        if self.unit is NotImplemented:
            # FIXME: https://github.com/pandas-dev/pandas/issues/58403
            # This looks awkward but is robust.
            deltas = self.backend.drop_null(deltas)
            diffs = np.array(self.backend.cast(deltas, int))
            unit: TimeDelta = int(np.gcd.reduce(diffs))
        else:
            unit = self.unit
        self.unit = self.backend.scalar(unit, dtype=self.timedelta_dtype)

    def encode(self, x: Arr, /) -> Arr:
        delta = x - self.offset

        try:
            return delta / self.unit
        except TypeError:
            # FIXME: pyarrow: "first cast to integer before dividing date-like dtypes"
            return self.backend.cast(delta, int) / self.backend.scalar(self.unit, int)

    def decode(self, y: Arr, /) -> Arr:
        if self.round:
            y = y.round()

        try:
            return self.backend.cast(y * self.unit + self.offset, self.datetime_dtype)
        except pa.lib.ArrowNotImplementedError:
            # Function 'multiply_checked' has no kernel matching input types (double, duration[ms])
            # FIXME: https://github.com/apache/arrow/issues/39233#issuecomment-2070756267
            z = self.backend.cast(y, float) * self.unit + self.offset
            return self.backend.cast(z, self.datetime_dtype)


@pprint_repr
@dataclass(init=False)
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
        return asdict(self)

    def __init__(self, num_dim: int, scale: float) -> None:
        self.num_dim = num_dim
        self.scale = float(scale)
        self.scales = self.scale ** (-np.arange(0, num_dim + 2, 2) / num_dim)
        if self.scales[0] != 1.0:
            raise ValueError("Initial scale must be 1.0")

    def encode(self, x: NDArray, /) -> NDArray:
        r""".. Signature: ``... -> (..., 2d)``.

        Note: we simply concatenate the sin and cosine terms without interleaving them.
        """
        z = np.einsum("..., d -> ...d", x, self.scales)
        return np.concatenate([np.sin(z), np.cos(z)], axis=-1)

    def decode(self, y: NDArray, /) -> NDArray:
        r""".. signature:: ``(..., 2d) -> ...``."""
        return np.arcsin(y[..., 0])


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
        z = self.freq * (x % self.period)  # ensure 0...N-1
        return DataFrame(
            np.stack([np.cos(z), np.sin(z)]).T,
            columns=[f"cos_{self.colname}", f"sin_{self.colname}"],
        )

    def decode(self, y: DataFrame) -> Series:
        r"""Decode the data."""
        z = np.arctan2(y[f"sin_{self.colname}"], y[f"cos_{self.colname}"])
        z = (z / self.freq) % self.period
        return Series(z, dtype=self.dtype, name=self.colname)


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

    level_code: str
    levels: list[str]
    original_dtype: DType
    original_name: Hashable
    original_type: type
    rev_cols: list[str]

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


@pprint_repr
@dataclass(init=False)
class PeriodicSocialTimeEncoder(WrappedEncoder[Series, DataFrame]):
    r"""Combines `SocialTimeEncoder` with `PeriodicEncoder` using the right frequencies."""

    levels: str
    r"""The levels to encode."""
    FREQUENCIES: dict[str, int]
    r"""The frequencies of the used `PeriodicEncoder`."""

    def __init__(
        self,
        *,
        levels: str = "YMWDhms",
        frequencies: Mapping[str, int] = MappingProxyType({
            "year": 1,
            "month": 12,
            "weekday": 7,
            "day": 365,
            "hour": 24,
            "minute": 60,
            "second": 60,
            "microsecond": 1000,
            "nanosecond": 1000,
        }),
    ) -> None:
        self.levels = levels
        self.encoder = SocialTimeEncoder(levels) >> FrameEncoder({
            level: PeriodicEncoder(period=frequencies[level]) for level in levels
        })
