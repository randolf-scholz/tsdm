import datetime as dt
from typing import Protocol, Self, overload

import numpy as np


class Timestamp[TD](Protocol):
    @overload
    def __sub__(self, other: Self, /) -> TD: ...
    @overload
    def __sub__(self, other: TD, /) -> Self: ...


py_dt = dt.datetime(year=2025, month=1, day=31)
foo: Timestamp = py_dt  # ✅
bar: Timestamp = np.datetime64(py_dt)  # ❌
