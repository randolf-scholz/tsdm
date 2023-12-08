#!/usr/bin/env python
# FIXME: https://github.com/python/mypy/issues/16552

from typing import (
    Generic,
    Iterable,
    Iterator,
    Literal,
    TypeAlias,
    TypeVar,
    assert_type,
    overload,
)

T = TypeVar("T")
S: TypeAlias = Literal["slice"]
W: TypeAlias = Literal["window"]
Mode = TypeVar("Mode", S, W)


class Sliding(Generic[T, Mode]):
    data: list[T]
    size: int
    mode: Mode

    @overload
    def __init__(self: "Sliding[T, S]", data: list[T], size: int, mode: S) -> None: ...
    @overload
    def __init__(self: "Sliding[T, W]", data: list[T], size: int, mode: W) -> None: ...
    def __init__(self, data, size, mode):
        self.data = data
        self.size = size
        self.mode = mode

    @overload
    def __iter__(self: "Sliding[T, S]") -> Iterator[slice]: ...
    @overload
    def __iter__(self: "Sliding[T, W]") -> Iterator[list[T]]: ...
    def __iter__(self):
        match self.mode:
            case "slice":
                for i in range(len(self.data) - self.size + 1):
                    yield slice(i, i + self.size)
            case "window":
                for i in range(len(self.data) - self.size + 1):
                    yield self.data[i : i + self.size]
            case _:
                raise TypeError(f"Unknown mode: {self.mode}")


x = Sliding([1, 2, 3, 4, 5], 3, mode="slice")
assert_type(x, Sliding[int, Literal["slice"]])  # ✅
assert_type(x.__iter__(), Iterator[slice])  # ✅
assert_type(iter(x), Iterator[slice])  # ✅
assert_type(list(x), list[slice])  # ✅
x1: Iterable[slice] = x  # ✅
x2: Iterable[list[int]] = x  # ❌ false positive
x3: Iterable[str] = x  # ✅ true negative


y = Sliding([1, 2, 3, 4, 5], 3, mode="window")
assert_type(y, Sliding[int, Literal["window"]])  #  ✅
assert_type(y.__iter__(), Iterator[list[int]])  #  ✅
assert_type(iter(y), Iterator[list[int]])  #  ❌ Iterator[slice] (false negative)
assert_type(list(y), list[list[int]])  #  ❌ list[slice]  (false negative)
y1: Iterable[list[int]] = y  # ❌ false positive
y2: Iterable[slice] = y  # ✅
y3: Iterable[str] = y  # ✅ true negative
