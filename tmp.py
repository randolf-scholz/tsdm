# mypy: enable="warn-unused-ignore"
from enum import StrEnum
from typing import Literal, assert_type, overload, reveal_type


class PolyMorphicDict[T]:
    def __getitem__(self, item: type[T]) -> T: ...
    def __setitem__(self, key: type[T], value: T) -> None: ...


type S = Literal["slices"]  # slice
type M = Literal["masks"]  # bool
type B = Literal["bounds"]  # tuple
type W = Literal["windows"]  # windows
type U = str  # unknown (not statically known)
type Mode = S | M | B | W


class MODE(StrEnum):
    r"""Valid modes for the sampler."""

    B = "bounds"
    M = "masks"
    S = "slices"
    W = "windows"


class Foo[ModeVar: (S, M, B, W, U)]:
    def __new__[T: (S, M, B, W, U)](cls, val: T) -> "Foo[T]": ...
    def __init__[T: (S, M, B, W, U)](self: "Foo[T]", val: T) -> None: ...


reveal_type(Foo("foo"))  # prints "It's a foo!"
reveal_type(Foo("bar"))  # prints "It's a bar!"
reveal_type(Foo("masks"))
reveal_type(Foo(MODE.B))
reveal_type(Foo(MODE.W))
reveal_type(Foo(MODE("foo")))


class Bar[ModeVar: Mode | MODE]:
    val: MODE

    @overload
    def __new__[T: Mode](cls, val: T) -> "Bar[T]": ...
    @overload
    def __new__[T: MODE](cls, val: T) -> "Bar[T]": ...
    def __new__[T: Mode | MODE](cls, val: T) -> "Bar[T]":
        return super().__new__(cls)

    # @overload
    # def __init__[T: Mode](self: "Bar[T]", val: T) -> None: ...
    # @overload
    # def __init__[T: MODE](self: "Bar[T]", val: T) -> None: ...
    # def __init__[T: Mode | MODE](self: "Bar[T]", val: T) -> None:
    #     self.val = MODE(val)


reveal_type(Bar("foo"))  # prints "It's a foo!"
reveal_type(Bar("bar"))  # prints "It's a bar!"

assert_type(Bar("masks"), Bar[M])
assert_type(Bar(MODE.B), Bar[B])
assert_type(Bar(MODE.W), Bar[W])
assert_type(Bar(MODE("foo")), Bar[MODE])

assert_type(Bar("masks"), Bar[Literal[MODE.M]])
assert_type(Bar(MODE.B), Bar[Literal[MODE.B]])
assert_type(Bar(MODE.W), Bar[Literal[MODE.W]])
assert_type(Bar(MODE("foo")), Bar[MODE])


import datetime as dt
from typing import Protocol, Self, overload, reveal_type

import numpy as np


class Timedelta(Protocol):
    def __add__(self, other: Self, /) -> Self: ...
    def __radd__(self, other: Self, /) -> Self: ...
    def __sub__(self, other: Self, /) -> Self: ...
    def __rsub__(self, other: Self, /) -> Self: ...


class Timestamp[TD: Timedelta](Protocol):
    @overload
    def __sub__(self, other: Self, /) -> TD: ...
    @overload
    def __sub__(self, other: TD, /) -> Self: ...


class SupportsSubSelf[TD: Timedelta](Protocol):
    def __sub__(self, other: Self, /) -> TD: ...


class SupportsSubTD[TD: Timedelta](Protocol):
    def __sub__(self, other: TD, /) -> Self: ...


py_date = dt.date(year=2025, month=1, day=31)
py_dt = dt.datetime(year=2025, month=1, day=31, hour=1, minute=23, second=45)
py_td = dt.timedelta(seconds=37)

np_date = np.datetime64(py_date)
np_dt = np.datetime64(py_dt)
np_td = np.timedelta64(py_td)

assert_type(py_date, dt.date)
assert_type(py_dt, dt.datetime)
assert_type(py_td, dt.timedelta)

assert_type(np_date, np.datetime64[dt.date])
assert_type(np_dt, np.datetime64[dt.datetime])
assert_type(np_td, np.timedelta64[dt.timedelta])

# runtime checks
assert isinstance(py_date - py_date, dt.timedelta)
assert isinstance(py_date - py_dt, dt.timedelta)
assert isinstance(py_date - py_td, dt.datetime)

assert isinstance(py_dt - py_td, dt.datetime)
assert isinstance(py_dt - py_date, dt.timedelta)
assert isinstance(py_dt - py_dt, dt.timedelta)


assert isinstance(np_date - np_date, np.timedelta64)
assert isinstance(np_date - np_dt, np.timedelta64)
assert isinstance(np_dt - np_date, np.timedelta64)
assert isinstance(np_dt - np_dt, np.timedelta64)

for lhs in [np_dt, np_dt_from_py]:
    for rhs in [py_dt]:
        assert isinstance(lhs - rhs, dt.timedelta), type(lhs - rhs)
for lhs in [np_dt, np_dt_from_py]:
    for rhs in [np_td, np_td_from_py]:
        assert isinstance(lhs - rhs, np.datetime64), type(lhs - rhs)
for lhs in [np_dt, np_dt_from_py]:
    for rhs in [py_td]:
        assert isinstance(lhs - rhs, dt.datetime), type(lhs - rhs)

#
reveal_type(np_dt - py_dt)  # dt.timedelta
reveal_type(np_dt - np_dt)  # np.timedelta64[dt.timedelta | int | None]
reveal_type(np_dt - np_dt_from_py)  # np.timedelta64[dt.timedelta | int | None]
#
reveal_type(np_dt_from_py - py_dt)  # dt.timedelta
reveal_type(np_dt_from_py - np_dt)  # np.timedelta64[dt.timedelta | int | None]
reveal_type(np_dt_from_py - np_dt_from_py)  # np.timedelta64[dt.timedelta]
#
reveal_type(np_dt - py_td)  # E: NO MATCHING OVERLOAD
reveal_type(np_dt - np_td)  # np.datetime64[dt.date | int | None]
reveal_type(np_dt - np_td_from_py)  # np.datetime64[dt.date | int | None]
#
reveal_type(np_dt_from_py - py_td)  # E: NO MATCHING OVERLOAD
reveal_type(np_dt_from_py - np_td)  # np.datetime64[dt.datetime]
reveal_type(np_dt_from_py - np_td_from_py)  # np.datetime64[dt.datetime]
