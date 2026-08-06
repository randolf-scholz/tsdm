r"""Positional encoding for float data."""

__all__ = ["PeriodicEncoder"]

from dataclasses import dataclass, field

import numpy as np
from pandas import DataFrame, Series
from pandas._typing import DtypeObj

from tsdm.constants import UNDEFINED
from tsdm.encoders.base import FittableEncoder
from tsdm.pprint import pprint_repr


@pprint_repr
@dataclass(slots=True)
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
