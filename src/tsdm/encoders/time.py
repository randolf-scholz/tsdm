r"""Encoders for timedelta and datetime types."""

__all__ = [
    # Classes
    "PeriodicEncoder",
    "PeriodicSocialTimeEncoder",
    "PositionalEncoder",
    "SocialTimeEncoder",
]

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import ClassVar, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame, Series
from pandas._typing import DtypeObj

from tsdm.constants import UNDEFINED
from tsdm.types.extra import SupportsArrayUfunc
from tsdm.utils.decorators import pprint_repr

from .base import FittableEncoder, StaticEncoder, WrappedEncoder
from .pandas import FrameEncoder


@pprint_repr
@dataclass(init=False)
class PositionalEncoder(StaticEncoder[SupportsArrayUfunc, SupportsArrayUfunc]):
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

    def encode[T: SupportsArrayUfunc](self, x: T, /) -> T:
        r""".. Signature: ``... -> (..., 2d)``.

        Note: we simply concatenate the sin and cosine terms without interleaving them.
        """
        z = np.einsum("..., d -> ...d", x, self.scales)
        return np.concatenate([np.sin(z), np.cos(z)], axis=-1)  # type: ignore

    def decode[T: SupportsArrayUfunc](self, y: T, /) -> T:
        r""".. signature:: ``(..., 2d) -> ...``."""
        return np.arcsin(y[..., 0])  # type: ignore


@pprint_repr
@dataclass
class PeriodicEncoder(FittableEncoder[Series, DataFrame]):
    r"""Encode periodic data as sin/cos waves."""

    period: float = UNDEFINED

    # fitted fields
    freq: float = field(init=False, default=UNDEFINED)
    original_dtype: DtypeObj = field(init=False, default=UNDEFINED)
    original_name: str = field(init=False, default=UNDEFINED)

    def fit(self, x: Series, /) -> None:
        r"""Fit the encoder."""
        self.original_dtype = x.dtype
        self.original_name = str(x.name)

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
    original_dtype: DtypeObj = field(init=False, default=UNDEFINED)
    original_name: str = field(init=False, default=UNDEFINED)
    original_type: type = field(init=False, default=UNDEFINED)

    levels: list[str] = field(init=False, default=UNDEFINED)
    level_columns: list[str] = field(init=False, default=UNDEFINED)

    def fit(self, x: Series, /) -> None:
        r"""Fit the encoder."""
        self.levels = [self.LEVEL_CODES[k] for k in self.level_codes]
        self.level_columns = [level for level in self.levels if level != "weekday"]
        self.original_type = type(x)
        self.original_name = str(x.name)
        self.original_dtype = x.dtype

    def encode(self, x: Series, /) -> DataFrame:
        r"""Encode the data."""
        return DataFrame.from_dict({level: getattr(x, level) for level in self.levels})

    def decode(self, x: DataFrame, /) -> Series:
        r"""Decode the data."""
        s = pd.to_datetime(x[self.level_columns])
        return Series(s, name=self.original_name, dtype=self.original_dtype)


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

    levels: str = "YMWDhms"
    r"""The levels to encode."""

    def __init__(
        self,
        *,
        levels: str = "YMWDhms",
        frequencies: Mapping[str, int] = DEFAULT_FREQUENCIES,
    ) -> None:
        self.levels = levels
        self.frequencies = frequencies
        encoder = SocialTimeEncoder(levels) >> FrameEncoder(
            {level: PeriodicEncoder(period=frequencies[level]) for level in levels}
        )
        super().__init__(encoder=encoder)
