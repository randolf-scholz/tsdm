r"""Lookup protocol definition."""

from abc import abstractmethod
from typing import Protocol, runtime_checkable


@runtime_checkable
class Lookup[K, V](Protocol):  # -K, +V
    r"""Mapping/Sequence like generic that is contravariant in Keys."""

    @abstractmethod
    def __contains__(self, key: K, /) -> bool:
        # Here, any Hashable input is accepted.
        r"""Return True if the map contains the given key."""
        ...

    @abstractmethod
    def __getitem__(self, key: K, /) -> V:
        r"""Return the value associated with the given key."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        r"""Return the number of items in the map."""
        ...
