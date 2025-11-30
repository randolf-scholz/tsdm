r"""Universal temporal encoders."""

__all__ = [
    "TimeDeltaEncoder",
    "DateTimeEncoder",
]


from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from pyarrow import ArrowNotImplementedError

from tsdm.backend import Backend, generic, get_backend
from tsdm.backend.pandas import PandasDtype
from tsdm.backend.types import NumericalSeries
from tsdm.constants import UNDEFINED
from tsdm.encoders import FittableEncoder
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
