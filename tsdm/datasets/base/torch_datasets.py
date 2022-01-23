r"""Datasets that subclass :class:`torch.utils.data.Dataset`."""

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from torch import Tensor
from torch.utils.data import Dataset as Torch_Dataset

from tsdm.util.strings import repr_mapping

__logger__ = logging.getLogger(__name__)


class SequenceDataset(Torch_Dataset):
    r"""Sequential Dataset."""

    def __init__(self, tensors: list[Tensor]):
        assert all(len(x) == len(tensors[0]) for x in tensors)
        self.tensors = tensors

    def __len__(self):
        r"""Length of the dataset."""
        return len(self.tensors[0])

    def __getitem__(self, idx):
        r"""Get the same slice from each tensor."""
        return [x[idx] for x in self.tensors]


class DatasetCollection(Mapping, Torch_Dataset):
    r"""Represents a ``mapping[index → torch.Datasets]``.

    All tensors must have a shared index,
    in the sense that index.unique() is identical for all inputs.
    """

    dataset: dict[Any, Torch_Dataset]
    """The dataset"""

    def __init__(self, indexed_datasets: Mapping[Any, Torch_Dataset]):
        super().__init__()
        self.dataset = dict(indexed_datasets)
        self.index = self.dataset.keys()
        self.keys = self.dataset.keys  # type: ignore[assignment]
        self.values = self.dataset.values  # type: ignore[assignment]
        self.items = self.dataset.items  # type: ignore[assignment]

    def __len__(self):
        r"""Length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, item):
        r"""Hierarchical lookup."""
        # test for hierarchical indexing
        if isinstance(item, Sequence):
            first, rest = item[0], item[1:]
            if isinstance(first, (Iterable, slice)):
                # pass remaining indices to sub-object
                value = self.dataset[first]
                return value[rest]

        # no hierarchical indexing
        return self.dataset[item]

    def __iter__(self):
        r"""Iterate over the dataset."""
        for key in self.index:
            yield self.dataset[key]

    def __repr__(self):
        r"""Representation of the dataset."""
        return repr_mapping(self)
