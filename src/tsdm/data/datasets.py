r"""Generic Dataset classes."""

__all__ = [
    # Type Aliases
    "Dataset",
    "TabularDataset",
    "SequentialDataset",
    # Protocols
    "TorchDataset",
    "IterableDataset",
    "MapDataset",
    "PandasDataset",
    # Classes
    "DataFrame2Dataset",
    "MappingDataset",
]

from abc import abstractmethod
from collections.abc import Iterator, Mapping, Reversible
from dataclasses import KW_ONLY, dataclass

from pandas import DataFrame, Index, MultiIndex
from typing_extensions import (
    Any,
    Optional,
    Protocol,
    Self,
    TypeAlias,
    TypeVar,
    overload,
    runtime_checkable,
)

from tsdm.types.protocols import ArrayKind, SupportsGetItem
from tsdm.types.variables import (
    key_contra as K_contra,
    key_var as K,
    nested_key_var as K2,
    value_co as V_co,
)
from tsdm.utils.strings import pprint_repr, repr_array

# class DataFrame2Dataset(TorchDataset[T]):
#     """Convert a DataFrame to a Dataset."""
#
#     def __getitem__(self):
# MapStyleDataset: TypeAlias = MappingProtocol


# region Protocols ---------------------------------------------------------------------


@runtime_checkable
class TorchDataset(Protocol[K_contra, V_co]):
    """Protocol version of `torch.utils.data.Dataset`."""

    @abstractmethod
    def __getitem__(self, key: K_contra, /) -> V_co:
        """Map key to sample."""
        ...


TorchDatasetVar = TypeVar("TorchDatasetVar", bound=TorchDataset)


@runtime_checkable
class PandasDataset(Protocol[K, V_co]):
    """Protocol version of `pandas.DataFrame`/`Series`.

    Note that in particular, `__getitem__` is not present, as it returns columns,
    but we are usually interested in the rows.
    """

    @property
    def index(self) -> ArrayKind[K]:
        """Returns the row labels of the DataFrame."""
        ...

    @property
    def loc(self) -> SupportsGetItem[K, V_co]:
        """Access a group of rows and columns by label(s) or a boolean array."""
        ...

    @property
    def iloc(self) -> SupportsGetItem[int, V_co]:
        """Purely integer location-based indexing for selection by position."""
        ...


@runtime_checkable
class IterableDataset(Protocol[V_co]):
    """Protocol version of `torch.utils.data.IterableDataset`."""

    @abstractmethod
    def __iter__(self) -> Iterator[V_co]:
        """Iterate over the dataset."""
        ...


@runtime_checkable
class IndexableDataset(Protocol[V_co]):
    """Protocol version of `torch.utils.data.IterableDataset` with len and getitem.

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
    def __iter__(self) -> Iterator[V_co]:
        """Iterate over the dataset."""
        ...

    @abstractmethod
    def __getitem__(self, key: int, /) -> V_co:
        """Map key to sample."""
        ...


@runtime_checkable
class MapDataset(Protocol[K, V_co]):
    """Protocol version of `torch.utils.data.Dataset` with a `keys()` method.

    Note:
        We deviate from the original in that we require a `keys()` method.
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
    def keys(self) -> Reversible[K] | IndexableDataset[K]:
        """Iterate over the keys."""
        ...


TabularDataset: TypeAlias = MapDataset[K, V_co] | PandasDataset[K, V_co]
"""Type alias for a "tabular" dataset."""

SequentialDataset: TypeAlias = IterableDataset[V_co] | PandasDataset[Any, V_co]
"""Type alias for a sequential dataset."""

Dataset: TypeAlias = IndexableDataset[V_co] | MapDataset[Any, V_co]
"""Type alias for a generic dataset."""
# endregion Protocol -------------------------------------------------------------------


@dataclass
class DataFrame2Dataset(MapDataset[K, DataFrame]):
    """Interpretes a `DataFrame` as a `torch.utils.data.Dataset` by redirecting `.loc`.

    It is assumed that the DataFrame has a MultiIndex.
    """

    data: DataFrame

    _: KW_ONLY

    def __post_init__(self):
        self.index = self.data.index.copy().droplevel(-1).unique()

    def __len__(self) -> int:
        return len(self.index)

    def keys(self) -> Index:
        return self.index

    def __getitem__(self, key: K) -> DataFrame:
        return self.data.loc[key]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(groups: {len(self.index)}, data:"
            f" {repr_array(self.data)})"
        )


@pprint_repr
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
