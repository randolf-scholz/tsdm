r"""Frozen mapping type."""
# FIXME: https://peps.python.org/pep-0603/

__all__ = ["FrozenMap"]

from collections.abc import Iterable, Iterator, Mapping
from typing import Any, overload


class FrozenMap[K = Any, V = Any](Mapping[K, V]):
    r"""A frozen mapping type."""

    # if TYPE_CHECKING:
    # fmt: off
    @overload  # mapping only
    def __init__(
        self: "FrozenMap[K, V]",  # pyright: ignore[reportInvalidTypeVarUse]
        items: Mapping[K, V] | Iterable[tuple[K, V]] = ..., /
    ) -> None: ...
    @overload  # mapping and kwargs
    def __init__(
        self: "FrozenMap[K | str, V]",  # pyright: ignore[reportInvalidTypeVarUse]
        items: Mapping[K, V] | Iterable[tuple[K, V]] = ..., /,
        **kwargs: V
    ) -> None: ...
    # fmt: on
    def __init__(
        self, items: Mapping[K, V] | Iterable[tuple[K, V]] = (), /, **kwargs: V
    ) -> None:
        self._values: dict[K, V] = dict(items, **kwargs)

    def __getitem__(self, key: K, /) -> V:
        return self._values[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __hash__(self) -> int:
        if getattr(self, "_hash", None) is None:
            self._hash = hash(frozenset(self._values.items()))
        return self._hash
