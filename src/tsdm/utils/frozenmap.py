r"""Frozen mapping type."""
# FIXME: https://peps.python.org/pep-0603/

__all__ = ["FrozenMap"]

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, overload

from tsdm.constants import EMPTY_MAP


class FrozenMap[K, V](Mapping[K, V]):
    r"""A frozen mapping type."""

    if TYPE_CHECKING:
        # fmt: off
        @overload
        def __new__(cls, /) -> "FrozenMap": ...
        @overload  # mapping only
        def __new__(cls, items: Mapping[K, V], /) -> "FrozenMap[K, V]": ...
        @overload  # mapping and kwargs
        def __new__(cls, items: Mapping[str, V] = ..., /, **kwargs: V) -> "FrozenMap[str, V]": ...
        # fmt: on

    def __init__(self, mapping: Mapping[K, V] = EMPTY_MAP, /, **kwargs: V) -> None:
        self._values: dict[K, V] = dict(mapping, **kwargs)

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
