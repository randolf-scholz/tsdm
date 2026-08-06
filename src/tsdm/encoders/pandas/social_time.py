r"""Social time encoder."""

__all__ = ["SocialTimeEncoder", "PeriodicSocialTimeEncoder"]

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import ClassVar

import pandas as pd
from pandas import DataFrame, Series
from pandas._typing import DtypeObj

from tsdm.constants import UNDEFINED
from tsdm.encoders.base import FittableEncoder, WrappedEncoder
from tsdm.pprint import pprint_repr

from .frame_encoder import FrameEncoder
from .positional import PeriodicEncoder


@pprint_repr
@dataclass(slots=True)
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
@dataclass(init=False, slots=True)
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
