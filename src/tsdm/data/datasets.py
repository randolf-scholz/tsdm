r"""Generic Dataset classes."""

__all__ = [
    # ABCs & Protocols
    "Dataset",
    "Indexable",
    "IterableDataset",
    "MapDataset",
    "PandasDataset",
    "SequentialDataset",
    "TabularDataset",
    "TorchDataset",
    # Classes
    "DataFrame2Dataset",
    "MappingDataset",
    # Functions
    "get_first_sample",
    "get_index",
    "get_last_sample",
]

from abc import abstractmethod
from collections.abc import Collection, Iterator, Mapping
from dataclasses import KW_ONLY, dataclass
from typing import Any, Optional, Protocol, Self, cast, overload, runtime_checkable

from numpy.typing import NDArray
from pandas import DataFrame, Index, MultiIndex

from tsdm.types.arrays import ArrayLike
from tsdm.types.mixins import SupportsGetItem, SupportsSlicing
from tsdm.utils.decorators import pprint_repr

# region Protocols ---------------------------------------------------------------------


@runtime_checkable
class TorchDataset[K, V](Protocol):  # -K, +V
    r"""Protocol version of `torch.utils.data.Dataset`."""

    @abstractmethod
    def __getitem__(self, key: K, /) -> V: ...


@runtime_checkable
class IterableDataset[V](Protocol):  # +V
    r"""Protocol version of `torch.utils.data.IterableDataset`."""

    @abstractmethod
    def __iter__(self) -> Iterator[V]: ...


@runtime_checkable
class Indexable[V](Protocol):  # +V
    r"""Protocol version of `torch.utils.data.IterableDataset` with len and getitem.

    Note:
        - We deviate from the original in that we require a `len()` method.
        - We deviate from the original in that we require a `__getitem__` method.
          Otherwise, the whole dataset needs to be wrapped in order to allow
          random access. For iterable datasets, we assume that the dataset is
          indexed by integers 0...n-1.

    Important:
        Always test for MapDataset first!

    Examples:
        - `list`, `pandas.Series`, `torch.Tensor`, `numpy.ndarray`
    """

    @abstractmethod
    def __len__(self) -> int: ...
    @abstractmethod
    def __iter__(self) -> Iterator[V]: ...
    @overload
    @abstractmethod
    def __getitem__(self, index: int, /) -> V: ...
    @overload
    @abstractmethod
    def __getitem__(self, index: slice, /) -> "Indexable[V]": ...


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
class SeriesDataset[K, V](Protocol):  # -K, +V
    r"""Protocol version of `pandas.Series`.

    Similar to a `Mapping`, but with an `__iter__` method that returns the values.
    """

    @abstractmethod
    def __array__(self) -> NDArray: ...
    @abstractmethod
    def __len__(self) -> int: ...
    @abstractmethod
    def __iter__(self) -> Iterator[V]: ...
    @abstractmethod
    def __getitem__(self, key: K, /) -> V: ...


class PandasDataset[K, V](Protocol):  # K, +V
    r"""Protocol version of `pandas.DataFrame`/`Series`.

    Note that in particular, `__getitem__` is not present, as it returns columns,
    but we are usually interested in the rows.
    """

    @property
    def index(self) -> ArrayLike[K]: ...
    @property
    def loc(self) -> SupportsGetItem[K, V]: ...
    @property
    def iloc(self) -> SupportsSlicing[V]: ...

    def __array__(self) -> NDArray[Any]: ...
    def __len__(self) -> int: ...


type TabularDataset[K, V] = MapDataset[K, V] | PandasDataset[K, V]  # K, +V
r"""Type alias for a "tabular" dataset."""

type SequentialDataset[V] = Indexable[V] | PandasDataset[Any, V]  # +V
r"""Type alias for a sequential dataset."""

type Dataset[V] = MapDataset[Any, V] | Indexable[V] | PandasDataset[Any, V]  # +V
r"""Type alias for a generic dataset."""
# endregion Protocol -------------------------------------------------------------------


@pprint_repr
@dataclass
class DataFrame2Dataset[K](MapDataset[K, DataFrame]):
    r"""Interpretes a `DataFrame` as a `torch.utils.data.Dataset` by redirecting ``.loc``.

    It is assumed that the DataFrame has a MultiIndex.
    """

    data: DataFrame

    _: KW_ONLY

    def __post_init__(self) -> None:
        self.index = self.data.index.copy().droplevel(-1).unique()

    def __len__(self) -> int:
        return len(self.index)

    def keys(self) -> Index:
        return self.index

    def __getitem__(self, key: K, /) -> DataFrame:
        return self.data.loc[key]


@pprint_repr
class MappingDataset[K, DS: TorchDataset](Mapping[K, DS]):
    r"""Represents a ``Mapping[Key, Dataset]``.

    ``ds[key]`` returns the dataset for the given key.
    If the key is a tuple, try to divert to the nested dataset.

    ``ds[(key, subkey)]=ds[key][subkey]``
    """

    datasets: Mapping[K, DS]
    index: list[K]

    def __init__(self, datasets: Mapping[K, DS], /) -> None:
        super().__init__()
        self.index = list(datasets.keys())
        self.datasets = datasets

    def __iter__(self) -> Iterator[K]:
        r"""Iterate over the keys."""
        return iter(self.index)

    def __len__(self) -> int:
        r"""Length of the dataset."""
        return len(self.index)

    @overload
    def __getitem__(self, key: K, /) -> DS: ...
    @overload
    def __getitem__(self, key: tuple[K, Any], /) -> Any: ...
    def __getitem__(self, key: K | tuple[K, Any], /) -> Any:
        r"""Get the dataset for the given key.

        If the key is a tuple, try to divert to the nested dataset.
        """
        match key:
            case k if key in self:
                return self.datasets[cast(K, k)]
            case [outer, inner]:
                return self.datasets[outer][inner]
            case _:
                raise KeyError(key)

    @classmethod
    def from_dataframe(
        cls, df: DataFrame, /, *, levels: Optional[list[str]] = None
    ) -> Self:
        r"""Create a `MappingDataset` from a `DataFrame`.

        If `levels` are given, the selected levels from the `DataFrame`'s `MultiIndex` are used as keys.
        """
        if levels is not None:
            min_index = df.index.to_frame()
            sub_index = MultiIndex.from_frame(min_index[levels])
            index = sub_index.unique()
        else:
            index = df.index

        return cls({idx: df.loc[idx] for idx in index})


def get_index(dataset: Dataset, /) -> Index:
    r"""Return an index object for the dataset.

    We support the following data types:
        - Series, DataFrame.
        - Mapping Types
        - Iterable Types
    """
    match dataset:
        # NOTE: Series and DataFrame satisfy the MapDataset protocol.
        case PandasDataset() as pandas_dataset:
            return pandas_dataset.index
        case MapDataset() as map_dataset:
            return Index(map_dataset.keys())
        case Indexable() as iterable_dataset:
            return Index(range(len(iterable_dataset)))
        case _:
            raise TypeError(f"Got unsupported data type {type(dataset)}.")


def get_first_sample[T](dataset: Dataset[T], /) -> T:
    r"""Return the first element of the dataset."""
    match dataset:
        case PandasDataset() as pandas_dataset:
            return pandas_dataset.iloc[0]
        case MapDataset() as map_dataset:
            return map_dataset[next(iter(map_dataset.keys()))]
        case Indexable() as iterable_dataset:
            return next(iter(iterable_dataset))
        case _:
            raise TypeError(f"Got unsupported data type {type(dataset)}.")


def get_last_sample[T](dataset: Dataset[T], /) -> T:
    r"""Return the last element of the dataset."""
    match dataset:
        case PandasDataset() as pandas_dataset:
            return pandas_dataset.iloc[-1]
        case MapDataset() as map_dataset:
            return map_dataset[next(reversed(map_dataset.keys()))]
        case Indexable() as iterable_dataset:
            return next(reversed(iterable_dataset))
        case _:
            raise TypeError(f"Got unsupported data type {type(dataset)}.")
