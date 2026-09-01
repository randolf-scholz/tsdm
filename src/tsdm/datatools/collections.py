r"""Generic Dataset classes."""

__all__ = [
    # ABCs & Protocols
    "Dataset",
    "MapDataset",
    "PandasDataset",
    "SequentialDataset",
    "TabularDataset",
    # Classes
    "CallableDataset",
    # Functions
    "get_first_sample",
    "get_index",
    "get_last_sample",
]

from abc import abstractmethod
from collections.abc import Callable, Collection
from typing import Any, Protocol, overload, runtime_checkable

from numpy.typing import NDArray

from tsdm.types.abc import Vec
from tsdm.types.protocols import SupportsGetItem, SupportsSlicing

type TabularDataset[K, V] = MapDataset[K, V] | PandasDataset[K, V]  # K, +V
r"""Type alias for a "tabular" dataset."""

type SequentialDataset[V] = Vec[V] | PandasDataset[Any, V]  # +V
r"""Type alias for a sequential dataset."""

type Dataset[V] = Vec[V] | MapDataset[Any, V] | PandasDataset[Any, V]  # +V
r"""Type alias for a generic dataset."""


@runtime_checkable
class MapDataset[K, V](Protocol):  # +V
    r"""Protocol version of `torch.utils.data.Dataset` with a `keys()` method.

    Note:
        We deviate from the original in that we require a `keys()` method.
        Otherwise, it is unclear how to iterate over the dataset. `torch.utils.data.Dataset`
        simply makes the assumption that the dataset is indexed by integers.
        But this is simply wrong for many use cases such as dictionaries or DataFrames.
    """

    @abstractmethod
    def __len__(self) -> int: ...
    @abstractmethod
    def __getitem__(self, key: K, /) -> V: ...
    @abstractmethod
    # NOTE: We want keys to support __reverse__, which requires either
    #   - SupportsLenAndGetItem -> Use IndexableDataset instead
    #   - Reversible -> Use KeysView instead
    def keys(self) -> Collection[K]:  # supertype of `KeysView[K]`
        r"""Access the keys."""
        ...


@runtime_checkable
class PandasDataset[K, V](Protocol):  # K, +V
    r"""Protocol version of `pandas.DataFrame`/`Series`.

    Note that in particular, `__getitem__` is not present, as it returns columns,
    but we are usually interested in the rows.
    """

    def __array__(self) -> NDArray[Any]: ...
    def __len__(self) -> int: ...

    @property
    def index(self) -> Vec[K]: ...
    @property
    def loc(self) -> SupportsGetItem[K, V]: ...
    @property
    def iloc(self) -> SupportsSlicing[V]: ...


class CallableDataset[K, V]:
    r"""Adapt a key-to-value callable to a map-style dataset.

    The wrapped function remains publicly available through the function
    attribute. This is useful when a sampler supplies arbitrary keys to a
    PyTorch DataLoader.
    """

    def __init__(self, function: Callable[[K], V], /) -> None:
        self.function = function

    def __getitem__(self, key: K, /) -> V:
        return self.function(key)


@overload
def get_index[K, V](dataset: TabularDataset[K, V], /) -> list[K]: ...  # pyright: ignore[reportOverlappingOverload]
@overload
def get_index[V](dataset: SequentialDataset[V], /) -> list[int]: ...
def get_index(dataset: Dataset, /) -> list:
    r"""Return an index object for the dataset.

    We support the following data types:
        - Series, DataFrame.
        - Mapping Types
        - Iterable Types
    """
    match dataset:
        # NOTE: Series and DataFrame satisfy the MapDataset protocol.
        case PandasDataset() as pandas_dataset:
            return list(pandas_dataset.index)
        case MapDataset() as map_dataset:
            return list(map_dataset.keys())
        case Vec() as iterable_dataset:
            return list(range(len(iterable_dataset)))
        case _:
            raise TypeError(f"Got unsupported data type {type(dataset)}.")


def get_first_sample[T](dataset: Dataset[T], /) -> T:
    r"""Return the first element of the dataset."""
    match dataset:
        case PandasDataset() as pandas_dataset:
            return pandas_dataset.iloc[0]
        case MapDataset() as map_dataset:
            return map_dataset[next(iter(map_dataset.keys()))]
        case Vec() as iterable_dataset:
            return next(iter(iterable_dataset))
        case _:
            raise TypeError(f"Got unsupported data type {type(dataset)}.")


def get_last_sample[T](dataset: Dataset[T], /) -> T:
    r"""Return the last element of the dataset."""
    match dataset:
        case PandasDataset() as pandas_dataset:
            return pandas_dataset.iloc[-1]
        case MapDataset() as map_dataset:
            *_, last_key = map_dataset.keys()
            return map_dataset[last_key]
        case Vec() as iterable_dataset:
            return next(reversed(iterable_dataset))
        case _:
            raise TypeError(f"Got unsupported data type {type(dataset)}.")
