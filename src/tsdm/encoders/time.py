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
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Final, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame, Index, Series
from pyarrow.lib import ArrowNotImplementedError

from tsdm.backend import Backend, generic, get_backend
from tsdm.backend.dtypes import DType
from tsdm.backend.pandas import PandasDtype
from tsdm.constants import UNDEFINED
from tsdm.encoders.base import FittableEncoder, StaticEncoder, WrappedEncoder
from tsdm.encoders.pandas import FrameEncoder
from tsdm.types.linalg import NumericalSeries
from tsdm.types.scalars import DurationScalar, TimestampScalar
from tsdm.utils import timedelta, timestamp
from tsdm.utils.decorators import pprint_repr


@pprint_repr
@dataclass(init=False, slots=True)
class TimeDeltaEncoder[Arr: NumericalSeries](FittableEncoder[Arr, Arr]):
    r"""Encode TimeDelta as Float."""

    unit: DurationScalar = UNDEFINED
    r"""The base frequency to convert timedeltas to."""
    timedelta_dtype: PandasDtype = UNDEFINED
    r"""The original dtype of the Series."""
    round: bool = True
    r"""Whether to round to the next unit."""

    backend: Backend = field(init=False, default=UNDEFINED)

    def __init__(
        self,
        *,
        unit: str | DurationScalar = UNDEFINED,
        rounding: bool = True,
    ) -> None:
        self.unit = UNDEFINED if unit is UNDEFINED else timedelta(unit)
        self.round = rounding

    def fit(self, data: Arr, /) -> None:
        self.backend = get_backend(data)
        self.timedelta_dtype = data.dtype

        if self.unit is UNDEFINED:
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
            y = generic.round(y)

        try:
            return self.backend.cast(y * self.unit, self.timedelta_dtype)
        except ArrowNotImplementedError:
            # Function 'multiply_checked' has no kernel matching input types (double, duration[ms])
            # FIXME: https://github.com/apache/arrow/issues/39233#issuecomment-2070756267
            y = self.backend.cast(y, float) * self.unit
            return self.backend.cast(y, self.timedelta_dtype)


@pprint_repr
@dataclass(init=False)
class DateTimeEncoder[Arr: NumericalSeries](FittableEncoder[Arr, Arr]):
    r"""Encode Datetime as Float."""

    offset: TimestampScalar = UNDEFINED
    r"""The starting point of the timeseries."""
    unit: DurationScalar = UNDEFINED
    r"""The base frequency to convert timedeltas to."""
    datetime_dtype: Any = UNDEFINED
    r"""The original dtype of the Series."""
    timedelta_dtype: Any = UNDEFINED
    r"""The dtype of the timedelta."""

    backend: Backend = field(init=False, default=UNDEFINED)

    def __init__(
        self,
        *,
        unit: str | DurationScalar = UNDEFINED,
        offset: str | DurationScalar = UNDEFINED,
        rounding: bool = True,
    ) -> None:
        self.unit = UNDEFINED if unit is UNDEFINED else timedelta(unit)
        self.offset = UNDEFINED if offset is UNDEFINED else timestamp(offset)
        self.round = rounding

    def fit(self, data: Arr, /) -> None:
        # get the datetime dtype
        self.backend = get_backend(data)
        self.datetime_dtype = data.dtype

        # set the offset
        offset = (
            cast("TimestampScalar", self.backend.nanmin(data))
            if self.offset is UNDEFINED
            else self.offset
        )
        self.offset = self.backend.scalar(offset, dtype=self.datetime_dtype)

        # get the timedelta dtype
        deltas = self.backend.drop_null(data - self.offset)
        self.timedelta_dtype = deltas.dtype

        if self.unit is UNDEFINED:
            # FIXME: https://github.com/pandas-dev/pandas/issues/58403
            # This looks awkward but is robust.
            deltas = self.backend.drop_null(deltas)
            diffs = np.array(self.backend.cast(deltas, int))
            unit: DurationScalar = int(np.gcd.reduce(diffs))
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
            y = generic.round(y)

        try:
            return self.backend.cast(y * self.unit + self.offset, self.datetime_dtype)
        except ArrowNotImplementedError:
            # Function 'multiply_checked' has no kernel matching input types (double, duration[ms])
            # FIXME: https://github.com/apache/arrow/issues/39233#issuecomment-2070756267
            z = self.backend.cast(y, float) * self.unit + self.offset
            return self.backend.cast(z, self.datetime_dtype)


# TODO: this should have signature [T: SupportsArrayUFunc](T -> T)
@pprint_repr
@dataclass(init=False)
class PositionalEncoder[T: (NDArray, DataFrame, Series, Index)](StaticEncoder[T, T]):
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

    def __init__(self, num_dim: int, scale: float) -> None:
        self.num_dim = num_dim
        self.scale = float(scale)
        self.scales = self.scale ** (-np.arange(0, num_dim + 2, 2) / num_dim)
        if self.scales[0] != 1.0:
            raise ValueError("Initial scale must be 1.0")

    def encode(self, x: T, /) -> T:
        r""".. Signature: ``... -> (..., 2d)``.

        Note: we simply concatenate the sin and cosine terms without interleaving them.
        """
        z = np.einsum("..., d -> ...d", x, self.scales)
        return np.concatenate([np.sin(z), np.cos(z)], axis=-1)

    def decode(self, y: T, /) -> T:
        r""".. signature:: ``(..., 2d) -> ...``."""
        return np.arcsin(y[..., 0])


@pprint_repr
@dataclass
class PeriodicEncoder(FittableEncoder[Series, DataFrame]):
    r"""Encode periodic data as sin/cos waves."""

    period: float = UNDEFINED

    # fitted fields
    freq: float = field(init=False, default=UNDEFINED)
    original_dtype: DType = field(init=False, default=UNDEFINED)
    original_name: Hashable = field(init=False, default=UNDEFINED)

    def fit(self, x: Series, /) -> None:
        r"""Fit the encoder."""
        self.original_dtype = x.dtype
        self.original_name = x.name

        if self.period is UNDEFINED:
            self.period = x.max() + 1

        self.freq = 2 * np.pi / self.period

    def encode(self, x: Series, /) -> DataFrame:
        r"""Encode the data."""
        z = self.freq * (x % self.period)  # ensure 0...N-1
        columns = [f"cos_{self.original_name}", f"sin_{self.original_name}"]
        return DataFrame(np.stack([np.cos(z), np.sin(z)]).T, columns=columns)

    def decode(self, y: DataFrame, /) -> Series:
        r"""Decode the data."""
        columns = [f"cos_{self.original_name}", f"sin_{self.original_name}"]
        z = np.arctan2(y[columns[1]], y[columns[0]])
        z = (z / self.freq) % self.period
        return Series(z, dtype=self.original_dtype, name=self.original_name)


@pprint_repr
@dataclass
class SocialTimeEncoder(FittableEncoder[Series, DataFrame]):
    r"""Social time encoding."""

    LEVEL_CODES: ClassVar[Mapping[str, str]] = {
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

    level_codes: str = "YMWDhms"

    # computed attributes
    original_dtype: DType = field(init=False, default=UNDEFINED)
    original_name: Hashable = field(init=False, default=UNDEFINED)
    original_type: type = field(init=False, default=UNDEFINED)

    levels: list[str] = field(init=False, default=UNDEFINED)
    rev_cols: list[str] = field(init=False, default=UNDEFINED)

    def fit(self, x: Series, /) -> None:
        r"""Fit the encoder."""
        self.levels = [self.LEVEL_CODES[k] for k in self.level_codes]
        self.rev_cols = [level for level in self.levels if level != "weekday"]
        self.original_type = type(x)
        self.original_name = x.name
        self.original_dtype = x.dtype

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

    DEFAULT_FREQUENCIES: ClassVar[Mapping[str, int]] = MappingProxyType({
        "year"        : 1,
        "month"       : 12,
        "weekday"     : 7,
        "day"         : 365,
        "hour"        : 24,
        "minute"      : 60,
        "second"      : 60,
        "microsecond" : 1000,
        "nanosecond"  : 1000,
    })  # fmt: skip
    r"""The frequencies of the used `PeriodicEncoder`."""

    levels: str
    r"""The levels to encode."""

    def __init__(
        self,
        *,
        levels: str = "YMWDhms",
        frequencies: Mapping[str, int] = DEFAULT_FREQUENCIES,
    ) -> None:
        self.levels = levels
        self.encoder = SocialTimeEncoder(levels) >> FrameEncoder({
            level: PeriodicEncoder(period=frequencies[level]) for level in levels
        })
