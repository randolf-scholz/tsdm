r"""Generic Dataset classes."""

__all__ = [
    # Classes
    "MappingDataset",
    "DatasetCollection",
]

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, Optional, TypeVar, overload

from pandas import DataFrame, MultiIndex
from torch.utils.data import Dataset as TorchDataset
from typing_extensions import Self

from tsdm.types.variables import KeyVar as K
from tsdm.types.variables import NestedKeyVar as K2
from tsdm.types.variables import ObjectVar as O
from tsdm.utils.strings import repr_mapping

TorchDatasetVar = TypeVar("TorchDatasetVar", bound=TorchDataset)


class DatasetCollection(TorchDataset[TorchDataset[O]], Mapping[K, TorchDataset[O]]):
    r"""Represents a ``Mapping[index, torch.Datasets]``.

    All tensors must have a shared index, in the sense that ``index.unique()`` is identical for all inputs.
    """

    dataset: Mapping[K, TorchDataset[O]]
    r"""The dataset."""

    def __init__(self, indexed_datasets: Mapping[K, TorchDataset[O]]) -> None:
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
    def __getitem__(self, key: Sequence[K] | slice) -> O:
        ...

    @overload
    def __getitem__(self, key: K) -> TorchDataset[O]:
        ...

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

    def __iter__(self) -> Iterator[TorchDataset[O]]:  # type: ignore[override]
        r"""Iterate over the dataset."""
        for key in self.index:
            yield self.dataset[key]

    def __repr__(self) -> str:
        r"""Representation of the dataset."""
        return repr_mapping(self)


class MappingDataset(Mapping[K, TorchDatasetVar]):
    r"""Represents a ``Mapping[Key, Dataset]``.

    ``ds[key]`` returns the dataset for the given key.
    If the key is a tuple, tries to divert to the nested dataset.

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
    def __getitem__(self, key: K) -> TorchDatasetVar:
        ...

    @overload
    def __getitem__(self, key: tuple[K, K2]) -> Any:
        ...

    def __getitem__(self, key):
        r"""Get the dataset for the given key.

        If the key is a tuple, tries to divert to the nested dataset.
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
    def from_dataframe(cls, df: DataFrame, levels: Optional[list[str]] = None) -> Self:
        r"""Create a `MappingDataset` from a `DataFrame`.

        If ``levels`` is given, the selected levels from the `DataFrame`'s `MultiIndex` are used as keys.
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
