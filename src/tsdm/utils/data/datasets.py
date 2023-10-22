r"""Generic Dataset classes."""

__all__ = [
    # Type Aliases
    "Dataset",
    # Protocols
    "IterableDataset",
    "MapDataset",
    "PandasDataset",
    # Classes
    "DataFrame2Dataset",
    "DatasetCollection",
    "MappingDataset",
]

from abc import abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from typing import (
    Any,
    Optional,
    Protocol,
    TypeAlias,
    TypeVar,
    overload,
    runtime_checkable,
)

from pandas import DataFrame, MultiIndex
from torch.utils.data import Dataset as TorchDataset
from typing_extensions import Self

from tsdm.types.protocols import SupportsGetItem
from tsdm.types.variables import (
    any_var as T,
    key_contra,
    key_var as K,
    nested_key_var as K2,
    value_co as V_co,
)
from tsdm.utils.strings import repr_array, repr_mapping

TorchDatasetVar = TypeVar("TorchDatasetVar", bound=TorchDataset)


# class DataFrame2Dataset(TorchDataset[T]):
#     """Convert a DataFrame to a Dataset."""
#
#     def __getitem__(self):
# MapStyleDataset: TypeAlias = MappingProtocol


@runtime_checkable
class PandasDataset(Protocol[K, V_co]):
    """Protocol version of `pandas.DataFrame`."""

    @property
    def index(self) -> Sequence[K]: ...

    @property
    def loc(self) -> SupportsGetItem[K, V_co]: ...


@runtime_checkable
class MapDataset(Protocol[K, V_co]):
    """Protocol version of `torch.utils.data.Dataset`.

    Note:
        - We additionally require a `len()` method, since we only consider non-streaming datasets.
        - We deviate from the original in that we require a `keys()` method.
          Otherwise, it is unclear how to iterate over the dataset. `torch.utils.data.Dataset`
          simply makes the assumption that the dataset is indexed by integers.
          But this is simply wrong for many use cases such as dictionaries or DataFrames.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Length of the dataset."""
        ...

    @abstractmethod
    def __getitem__(self, key: K, /) -> V_co:
        """Map key to sample."""
        ...

    @abstractmethod
    def keys(self) -> Iterable[K]:
        """Iterate over the keys."""
        ...


@runtime_checkable
class IterableDataset(Protocol[V_co]):
    """Protocol version of `torch.utils.data.IterableDataset`.

    Note:
        - We deviate from the original in that we require a `len()` method.
        - We deviate from the original in that we require a `__getitem__` method.
          Otherwise, the whole dataset needs to be wrapped in order to allow
          random access. For iterable datasets, we assume that the dataset is
          indexed by integers 0...n-1.

    Important:
        Always test for MapDataset first!
    """

    @abstractmethod
    def __len__(self) -> int:
        """Length of the dataset."""
        ...

    @abstractmethod
    def __getitem__(self, key: int, /) -> V_co:
        """Map key to sample."""
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[V_co]:
        """Iterate over the dataset."""
        ...


Dataset: TypeAlias = MapDataset[K, V_co] | IterableDataset[V_co]
"""Type alias for a generic dataset."""


@dataclass
class DataFrame2Dataset(MapDataset[K, DataFrame], TorchDataset[DataFrame]):
    """Interpretes a `DataFrame` as a `torch.utils.data.Dataset` by redirecting `.loc`.

    It is assumed that the DataFrame has a MultiIndex.
    """

    data: DataFrame

    _: KW_ONLY

    def __post_init__(self):
        self.index = self.data.index.copy().droplevel(-1).unique()

    def __len__(self) -> int:
        return len(self.index)

    def keys(self) -> Iterator[K]:
        return iter(self.index)

    def __getitem__(self, item: key_contra, /) -> DataFrame:
        return self.data.loc[item]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(groups: {len(self.index)}, data:"
            f" {repr_array(self.data)})"
        )


class DatasetCollection(TorchDataset[TorchDataset[T]], Mapping[K, TorchDataset[T]]):
    r"""Represents a ``Mapping[index, torch.Datasets]``.

    All tensors must have a shared index, in the sense that ``index.unique()`` is identical for all inputs.
    """

    dataset: Mapping[K, TorchDataset[T]]
    r"""The dataset."""

    def __init__(self, indexed_datasets: Mapping[K, TorchDataset[T]]) -> None:
        super().__init__()
        self.dataset = dict(indexed_datasets)
        self.index = list(self.dataset.keys())
        self.keys = self.dataset.keys  # type: ignore[method-assign]
        self.values = self.dataset.values  # type: ignore[method-assign]
        self.items = self.dataset.items  # type: ignore[method-assign]

    def __len__(self) -> int:
        r"""Length of the dataset."""
        return len(self.dataset)

    @overload
    def __getitem__(self, key: Sequence[K] | slice) -> T: ...

    @overload
    def __getitem__(self, key: K) -> TorchDataset[T]: ...

    def __getitem__(self, key):
        r"""Hierarchical lookup."""
        # test for hierarchical indexing
        if isinstance(key, Sequence):
            first, rest = key[0], key[1:]
            if isinstance(first, (Iterable, slice)):
                # pass remaining indices to sub-object
                value = self.dataset[first]  # type: ignore[index]
                return value[rest]
        # no hierarchical indexing
        return self.dataset[key]

    def __iter__(self) -> Iterator[TorchDataset[T]]:  # type: ignore[override]
        r"""Iterate over the dataset."""
        for key in self.index:
            yield self.dataset[key]

    def __repr__(self) -> str:
        r"""Representation of the dataset."""
        return repr_mapping(self)


class MappingDataset(Mapping[K, TorchDatasetVar]):
    r"""Represents a ``Mapping[Key, Dataset]``.

    ``ds[key]`` returns the dataset for the given key.
    If the key is a tuple, try to divert to the nested dataset.

    ``ds[(key, subkey)]=ds[key][subkey]``
    """

    datasets: Mapping[K, TorchDatasetVar]
    index: list[K]

    def __init__(self, datasets: Mapping[K, TorchDatasetVar], /) -> None:
        super().__init__()
        assert isinstance(datasets, Mapping)
        self.index = list(datasets.keys())
        self.datasets = datasets

    def __iter__(self) -> Iterator[K]:
        r"""Iterate over the keys."""
        return iter(self.index)

    def __len__(self) -> int:
        r"""Length of the dataset."""
        return len(self.index)

    @overload
    def __getitem__(self, key: K) -> TorchDatasetVar: ...

    @overload
    def __getitem__(self, key: tuple[K, K2]) -> Any: ...

    def __getitem__(self, key):
        r"""Get the dataset for the given key.

        If the key is a tuple, try to divert to the nested dataset.
        """
        if not isinstance(key, tuple):
            return self.datasets[key]
        try:
            outer = self.datasets[key[0]]
            if len(key) == 2:
                return outer[key[1]]
            return outer[key[1:]]
        except KeyError:
            return self.datasets[key]  # type: ignore[index]

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

    def __repr__(self) -> str:
        r"""Representation of the dataset."""
        return repr_mapping(self)
