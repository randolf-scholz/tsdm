#!/usr/bin/env python

from typing import Protocol, Self, overload

import numpy as np


class TimeDelta(Protocol):
    def __sub__(self, other: Self, /) -> Self: ...


class TimeStamp[TD: TimeDelta](Protocol):
    @overload
    def __sub__(self, other: Self, /) -> TD: ...
    @overload
    def __sub__(self, other: TD, /) -> Self: ...


x: TimeStamp[np.timedelta64] = np.datetime64("2021-01-01")  # ✅
y: TimeStamp[np.float64] = np.float64(
    10.0
)  # expected overloaded function, got "_FloatOp[_64Bit]"
