from datetime import timedelta as TD
from typing import assert_type, reveal_type

import numpy as np

td = np.timedelta64(1, "D")
assert_type(td, np.timedelta64[TD])  # ✅

n, remainder = divmod(td, td)
assert_type(remainder, np.timedelta64[TD])  # ❌ timedelta64[timedelta | int | None]


reveal_type(td + td)


import datetime
from datetime import timedelta as TD
from typing import Protocol, Self, overload

import numpy as np


class Timestamp(Protocol):
    def __add__(self, other: TD, /) -> Self: ...
    def __radd__(self, other: TD, /) -> Self: ...

    @overload
    def __sub__(self, other: Self, /) -> TD: ...
    @overload
    def __sub__(self, other: TD, /) -> Self: ...


py_dt: Timestamp = datetime.datetime(year=2025, month=1, day=31)
np_dt: Timestamp = np.datetime64(py_dt)
