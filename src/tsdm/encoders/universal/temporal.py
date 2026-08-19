r"""Universal temporal encoders."""

__all__ = [
    "TimeDeltaEncoder",
    "DateTimeEncoder",
]


from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from pyarrow import ArrowNotImplementedError

from tsdm.backend import Backend, get_backend
from tsdm.constants import UNDEFINED
from tsdm.datatools import timedelta, timestamp
from tsdm.encoders.base import FittableEncoder
from tsdm.pprint import pprint_repr

type DateTimeArray = Any
type TimeDeltaArray = Any
type FloatArray = Any


@pprint_repr
@dataclass(init=False, slots=True)
class TimeDeltaEncoder[X: TimeDeltaArray, Y: FloatArray](FittableEncoder[X, Y]):
    r"""Encode TimeDelta as Float."""

    unit: Any = UNDEFINED
    r"""The base frequency to convert timedeltas to."""
    timedelta_dtype: Any = UNDEFINED
    r"""The original dtype of the Series."""
    round: bool = True
    r"""Whether to round to the next unit."""

    backend: Backend = field(init=False, default=UNDEFINED)

    def __init__(
        self,
        *,
        unit: Any = UNDEFINED,
        rounding: bool = True,
    ) -> None:
        self.unit = UNDEFINED if unit is UNDEFINED else timedelta(unit)
        self.round = rounding

    def fit(self, data: X, /) -> None:
        self.backend = get_backend(data)
        self.timedelta_dtype = data.dtype  # pyrefly: ignore[missing-attribute]

        if self.unit is UNDEFINED:
            # This looks awkward but is robust.
            data = self.backend.drop_null(data)
            diffs = np.array(self.backend.cast(data, int))
            base_freq = int(np.gcd.reduce(diffs))

            # convert base_freq back to time delta in the original dtype
            self.unit = self.backend.scalar(base_freq, dtype=self.timedelta_dtype)

    def encode(self, x: X, /) -> Y:
        return cast("Y", x / self.unit)

    def decode(self, y: Y, /) -> X:
        if self.round:
            y = y.round()  # pyrefly: ignore[missing-attribute]

        try:
            return cast("X", y * self.unit)
        except TypeError, ArrowNotImplementedError:
            # Function 'multiply_checked' has no kernel matching input types (double, duration[ms])
            # FIXME: https://github.com/apache/arrow/issues/39233#issuecomment-2070756267
            y = self.backend.cast(y, float) * self.unit
            return self.backend.cast(y, self.timedelta_dtype)


@pprint_repr
@dataclass(init=False, slots=True)
class DateTimeEncoder[X: DateTimeArray, Y: FloatArray](FittableEncoder[X, Y]):
    r"""Encode Datetime as Float."""

    offset: Any = UNDEFINED
    r"""The starting point of the timeseries."""
    unit: Any = UNDEFINED
    r"""The base frequency to convert timedeltas to."""
    datetime_dtype: Any = UNDEFINED
    r"""The original dtype of the Series."""
    timedelta_dtype: Any = UNDEFINED
    r"""The dtype of the timedelta."""

    def __init__(
        self,
        *,
        unit: Any = UNDEFINED,
        offset: Any = UNDEFINED,
        rounding: bool = True,
    ) -> None:
        self.unit = UNDEFINED if unit is UNDEFINED else timedelta(unit)
        self.offset = UNDEFINED if offset is UNDEFINED else timestamp(offset)
        self.round = rounding

    def fit(self, data: X, /) -> None:
        # get the datetime dtype
        self.backend: Backend[Any] = get_backend(data)
        self.datetime_dtype = data.dtype  # pyrefly: ignore[missing-attribute]

        # set the offset
        offset = self.backend.nanmin(data) if self.offset is UNDEFINED else self.offset
        self.offset = self.backend.scalar(offset, dtype=self.datetime_dtype)

        # get the timedelta dtype
        deltas = self.backend.drop_null(data - self.offset)
        self.timedelta_dtype = deltas.dtype  # pyrefly: ignore[missing-attribute]

        if self.unit is UNDEFINED:
            # FIXME: https://github.com/pandas-dev/pandas/issues/58403
            # This looks awkward but is robust.
            deltas = self.backend.drop_null(deltas)
            diffs = np.array(self.backend.cast(deltas, int))
            unit = int(np.gcd.reduce(diffs))
        else:
            unit = self.unit

        self.unit = self.backend.scalar(unit, dtype=self.timedelta_dtype)

    def encode(self, x: X, /) -> Y:
        return cast("Y", (x - self.offset) / self.unit)

    def decode(self, y: Y, /) -> X:
        if self.round:
            y = y.round()  # pyrefly: ignore[missing-attribute]

        try:
            return cast("X", y * self.unit + self.offset)
        except ArrowNotImplementedError, TypeError:
            # Function 'multiply_checked' has no kernel matching input types (double, duration[ms])
            # FIXME: https://github.com/apache/arrow/issues/39233#issuecomment-2070756267
            z = self.backend.cast(y, float) * self.unit + self.offset
            return self.backend.cast(z, self.datetime_dtype)
