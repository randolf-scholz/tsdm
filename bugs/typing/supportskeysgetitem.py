from collections.abc import Iterable, Mapping
from typing import Protocol, TypeVar, assert_type, reveal_type, runtime_checkable

K = TypeVar("K")
V_co = TypeVar("V_co", covariant=True)
T = TypeVar("T")


@runtime_checkable
class SupportsKeysAndGetItem(Protocol[K, V_co]):
    r"""Protocol for objects that support `__getitem__` and `keys`."""

    def keys(self) -> Iterable[K]: ...
    def __getitem__(self, key: K, /) -> V_co: ...


def foo(x: SupportsKeysAndGetItem[K, T] | Iterable[tuple[K, T]]) -> dict[K, T]:
    match x:
        case SupportsKeysAndGetItem() as mapping:
            return dict(mapping)
        case Iterable() as iterable:
            assert_type(iterable, Iterable[tuple[K, T]])  # ✅
            return dict(iterable)
        case _:
            raise TypeError("Unsupported type")


def bar(x: Mapping[K, T] | Iterable[tuple[K, T]]) -> dict[K, T]:
    match x:
        case SupportsKeysAndGetItem() as mapping:
            return dict(mapping)
        case Iterable() as iterable:
            assert_type(
                x, Iterable[tuple[K, T]]
            )  # ❌ Mapping[K, T] | Iterable[tuple[K, T]]
            return dict(iterable)
        case _:
            raise TypeError("Unsupported type")


def baz(x: Mapping[K, T] | Iterable[tuple[K, T]]) -> dict[K, T]:
    if isinstance(x, SupportsKeysAndGetItem):
        return dict(x)
    if isinstance(x, Iterable):
        assert_type(x, Iterable[tuple[K, T]])  # ✅
        return dict(x)
    raise TypeError("Unsupported type")
