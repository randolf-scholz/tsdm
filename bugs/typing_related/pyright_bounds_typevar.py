#!/usr/bin/env python

from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Iterable,
    Iterator,
    Literal,
    Never,
    TypeAlias,
    TypeVar,
    assert_type,
    overload,
)

T = TypeVar("T")  # pyright: ignore
S: TypeAlias = Literal["slices"]
B: TypeAlias = Literal["bounds"]
M: TypeAlias = Literal["masks"]
W: TypeAlias = Literal["windows"]
# U: TypeAlias = S | B | M | W  # statically unknown
U: TypeAlias = str  # S | B | M | W  # statically unknown
# U: TypeAlias = Any
Mode = TypeVar("Mode", S, B, M, W, U)
ModeWithUnknown = TypeVar("ModeWithUnknown", S, B, M, W, U)


class Sliding(Generic[T, Mode]):
    data: list[T]
    size: int
    mode: Mode

    # @overload
    # def __init__(
    #     self: "Sliding[T, Mode]", data: list[T], size: int, mode: Mode
    # ) -> None: ...
    @overload
    def __init__(self: "Sliding[T, S]", data: list[T], size: int, mode: S) -> None: ...
    @overload
    def __init__(self: "Sliding[T, B]", data: list[T], size: int, mode: B) -> None: ...
    @overload
    def __init__(self: "Sliding[T, M]", data: list[T], size: int, mode: M) -> None: ...
    @overload
    def __init__(self: "Sliding[T, W]", data: list[T], size: int, mode: W) -> None: ...
    @overload  # fallback
    def __init__(
        self: "Sliding[T, U]", data: list[T], size: int, mode: str
    ) -> None: ...
    def __init__(self, data, size, mode):
        self.data = data
        self.size = size
        self.mode = mode

    # @overload  # early fallback
    # def __iter__(self: "Sliding[T, Never]") -> Iterator[Any]: ...
    @overload
    def __iter__(self: "Sliding[T, S]") -> Iterator[slice]: ...
    @overload
    def __iter__(self: "Sliding[T, B]") -> Iterator[tuple[T, T]]: ...
    @overload
    def __iter__(self: "Sliding[T, M]") -> Iterator[list[bool]]: ...
    @overload
    def __iter__(self: "Sliding[T, W]") -> Iterator[list[T]]: ...
    @overload  # late fallback
    def __iter__(self: "Sliding[T, U]") -> Iterator[Any]: ...
    def __iter__(self):  # pyright: ignore[reportGeneralTypeIssues]
        num = len(self.data) - self.size + 1

        match self.mode:
            case "windows":
                for i in range(num):
                    yield self.data[i : i + self.size]
            case "bounds":
                for i in range(num):
                    yield i, i + self.size
            case "slices":
                for i in range(num):
                    yield slice(i, i + self.size)
            case "masks":
                for i in range(num):
                    yield [i <= j < i + self.size for j in range(len(self.data))]
            case _:
                raise TypeError(f"Unknown mode: {self.mode}")


def test_slice() -> None:
    sampler = Sliding([1, 2, 3, 4, 5], 3, mode="slices")

    if TYPE_CHECKING:
        assert_type(sampler, Sliding[int, S])
        assert_type(sampler.__iter__(), Iterator[slice])
        assert_type(iter(sampler), Iterator[slice])
        assert_type(list(sampler), list[slice])
        s1: Iterable[slice] = sampler

    assert list(sampler) == [slice(0, 3), slice(1, 4), slice(2, 5)]


def test_window() -> None:
    sampler = Sliding([1, 2, 3, 4, 5], 3, mode="windows")

    if TYPE_CHECKING:
        assert_type(sampler, Sliding[int, W])
        assert_type(sampler.__iter__(), Iterator[list[int]])
        assert_type(iter(sampler), Iterator[list[int]])
        assert_type(list(sampler), list[list[int]])
        w1: Iterable[list[int]] = sampler

    assert list(sampler) == [[1, 2, 3], [2, 3, 4], [3, 4, 5]]


def test_bounds() -> None:
    sampler = Sliding([1, 2, 3, 4, 5], 3, mode="bounds")

    if TYPE_CHECKING:
        assert_type(sampler, Sliding[int, B])
        assert_type(sampler.__iter__(), Iterator[tuple[int, int]])
        assert_type(iter(sampler), Iterator[tuple[int, int]])
        assert_type(list(sampler), list[tuple[int, int]])
        b1: Iterable[tuple[int, int]] = sampler

    assert list(sampler) == [(0, 3), (1, 4), (2, 5)]


def test_masks() -> None:
    sampler = Sliding([1, 2, 3, 4, 5], 3, mode="masks")

    if TYPE_CHECKING:
        assert_type(sampler, Sliding[int, M])
        assert_type(sampler.__iter__(), Iterator[list[bool]])
        assert_type(iter(sampler), Iterator[list[bool]])
        assert_type(list(sampler), list[list[bool]])
        m1: Iterable[list[bool]] = sampler

    assert list(sampler) == [
        [True, True, True, False, False],
        [False, True, True, True, False],
        [False, False, True, True, True],
    ]


def test_unknown() -> None:
    mode: str = "..."
    sampler = Sliding([1, 2, 3, 4, 5], 3, mode=mode)

    if TYPE_CHECKING:
        assert_type(sampler, Sliding[int, U])
        assert_type(sampler.__iter__(), Iterator[Any])
        assert_type(iter(sampler), Iterator[Any])
        assert_type(list(sampler), list[Any])
        u1: Iterable[Any] = sampler
