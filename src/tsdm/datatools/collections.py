r"""Generic dataset types, protocols, and access utilities."""

__all__ = [
    # ABCs & Protocols
    "Dataset",
    "IndexableDataset",
    "MapDataset",
    # Classes
    "CallableDataset",
    # Functions
    "get_first_sample",
    "get_index",
    "get_last_sample",
]

from abc import abstractmethod
from collections.abc import Callable, Collection, Iterator, Mapping
from typing import (
    Any,
    Protocol,
    TypeIs,
    _ProtocolMeta as ProtocolMeta,
    overload,
    runtime_checkable,
)

type Dataset[V] = IndexableDataset[V] | MapDataset[Any, V]  # +V
r"""Type alias for a generic dataset."""


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


class _IndexableDatasetMeta(ProtocolMeta):
    def __instancecheck__(self, other: object, /) -> TypeIs[IndexableDataset]:
        if isinstance(other, str | bytes | Mapping):
            return False
        return super().__instancecheck__(other)

    def __subclasscheck__(self, other: type, /) -> TypeIs[type[IndexableDataset]]:
        if issubclass(other, str | bytes | Mapping):
            return False
        return super().__subclasscheck__(other)


@runtime_checkable
class IndexableDataset[T](Protocol, metaclass=_IndexableDatasetMeta):  # +T
    r"""Protocol for a dataset indexed by consecutive integer positions.

    Every index in `range(len(dataset))` must be valid, and iteration must yield
    the corresponding samples in that order. Slicing returns another indexable
    dataset. An empty dataset has no valid nonnegative integer indices.

    Unlike `collections.abc.Sequence`, this protocol does not require
    `__reversed__`, `index`, or `count`.

    Membership accepts arbitrary objects, which excludes `str` during static
    type checking. The default implementation checks identity before equality,
    like `list`. Equality comparisons that return arrays or tensors may raise
    an error when converted to `bool`.

    Note:
        Runtime instance and subclass checks exclude `str`, `bytes`, and
        `Mapping` subclasses, including `dict`. They check method presence, but
        cannot verify the indexing or iteration semantics.

    Examples:
        - `list`
        - `tuple`
        - `numpy.ndarray`
        - `pandas.Index`
        - `pandas.Series` (with index labels matching `range(len(dataset))`)

        Counterexamples include `str`, `bytes`, and `dict`, which are excluded
        by the runtime checks.
    """

    @abstractmethod
    def __len__(self) -> int: ...

    # Mixin methods
    def __iter__(self) -> Iterator[T]:
        for i in range(len(self)):
            yield self[i]

    # excluded to allow matching pyarrow.array.
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


@overload
def get_index[K, V](dataset: MapDataset[K, V], /) -> list[K]: ...  # pyright: ignore[reportOverlappingOverload]
@overload
def get_index[V](dataset: IndexableDataset[V], /) -> list[int]: ...
def get_index(dataset: Dataset, /) -> list:
    r"""Return the dataset's keys as a list.

    - For pandas-like datasets, return the row index labels.
    - For map-style datasets, return the keys in iteration order.
    - For indexable datasets, return `list(range(len(dataset)))`.

    Raises:
        TypeError: If the dataset does not implement a supported protocol.
    """
    match dataset:
        # NOTE: Series and DataFrame satisfy the MapDataset protocol.
        case MapDataset() as map_dataset:
            return list(map_dataset.keys())
        case IndexableDataset() as indexable_dataset:
            return list(range(len(indexable_dataset)))
        case _:
            raise TypeError(f"Got unsupported data type {type(dataset)}.")


def get_first_sample[T](dataset: Dataset[T], /) -> T:
    r"""Return the first element of the dataset."""
    match dataset:
        case MapDataset() as map_dataset:
            return map_dataset[next(iter(map_dataset.keys()))]
        case IndexableDataset() as indexable_dataset:
            return next(iter(indexable_dataset))
        case _:
            raise TypeError(f"Got unsupported data type {type(dataset)}.")


def get_last_sample[T](dataset: Dataset[T], /) -> T:
    r"""Return the last element of the dataset."""
    match dataset:
        case MapDataset() as map_dataset:
            *_, last_key = map_dataset.keys()
            return map_dataset[last_key]
        case IndexableDataset() as indexable_dataset:
            return next(reversed(indexable_dataset))
        case _:
            raise TypeError(f"Got unsupported data type {type(dataset)}.")
