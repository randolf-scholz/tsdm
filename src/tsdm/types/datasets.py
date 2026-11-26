r"""Generic dataset types, protocols, and adapters."""

__all__ = [
    "IndexableDataset",
    "MapDataset",
    "CallableDataset",
]

from abc import abstractmethod
from collections.abc import Callable, Collection, Iterator, Mapping
from typing import Protocol, overload, runtime_checkable

from .protocols import protocol_exclude


@runtime_checkable
class MapDataset[K, V](Protocol):  # +V
    r"""Protocol for a map-style dataset with an explicit `keys()` method.

    Note:
        The `keys()` method exposes valid keys for datasets whose keys need not
        be consecutive integers, such as dictionaries. This extends the interface
        of `torch.utils.data.Dataset`, which does not require `keys()`.
    """

    @abstractmethod
    def __len__(self) -> int: ...
    @abstractmethod
    def __getitem__(self, key: K, /) -> V: ...
    @abstractmethod
    # NOTE: We want keys to support __reverse__, which requires either
    #   - SupportsLenAndGetItem -> Use IndexableDataset instead
    #   - Reversible -> Use KeysView instead
    def keys(self) -> Collection[K]:  # supertype of ``KeysView[K]``
        r"""Access the keys."""
        ...


@runtime_checkable
class IndexableDataset[T](
    Protocol, metaclass=protocol_exclude, excluded=(str, bytes, Mapping)
):  # +T
    r"""Protocol for a dataset indexed by consecutive integer positions.

    Every index in `range(len(dataset))` must be valid, and iteration must yield
    the corresponding samples in that order. Slicing returns another indexable
    dataset. An empty dataset has no valid nonnegative integer indices.

    Unlike `collections.abc.Sequence`, this protocol does not require
    `__reversed__`, `index`, or `count`.


    Note:
        Runtime `isinstance` and `issubclass` checks explicitly exclude
        `str`, `bytes` and `collections.abc.Mapping`.
        However, at type checking time, `str` and `bytes` are accepted.

    Examples:
        - `list`, `tuple`
        - `numpy.ndarray`, `torch.Tensor`
        - `pandas.Index`, `pandas.Series`, `polars.Array`, `polars.Series`

    Counter-Examples:
        - `dict`, `set`, `collections.abc.Mapping`
        - `str`, `bytes` (at runtime)
    """

    @abstractmethod
    def __len__(self) -> int: ...

    # Mixin methods
    def __iter__(self) -> Iterator[T]:
        for i in range(len(self)):
            yield self[i]

    # Note: excluded to allow matching pyarrow.array.
    # def __contains__(self, value: object, /) -> bool:
    #     return any(x == value for x in self)

    @overload
    @abstractmethod
    def __getitem__(self, index: int, /) -> T: ...
    @overload
    @abstractmethod
    # NOTE: not "-> Self" to ensure compatibility with tuple.
    def __getitem__(self, index: slice, /) -> IndexableDataset[T]: ...


class CallableDataset[K, V]:
    r"""Adapt a key-to-value callable to a map-style dataset.

    The wrapped function remains publicly available through the `function`
    attribute. This is useful when a sampler supplies arbitrary keys to a
    PyTorch `DataLoader`.
    """

    def __init__(self, function: Callable[[K], V], /) -> None:
        self.function = function

    def __getitem__(self, key: K, /) -> V:
        return self.function(key)
