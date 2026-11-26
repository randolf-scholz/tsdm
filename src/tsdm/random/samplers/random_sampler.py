r"""Sampler for iterating over dataset items in random order."""

__all__ = ["RandomSampler"]

from collections.abc import Iterator, Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import Final

from numpy.random import Generator

from tsdm.constants import RNG
from tsdm.types.datasets import Dataset, IndexableDataset, MapDataset
from tsdm.utils import pretty_repr

from .base import BaseSampler


@pretty_repr
@dataclass(slots=True)
class RandomSampler[T](BaseSampler[T]):  # +T
    r"""Yield the items of a dataset once, optionally in a random order."""

    data: Final[Dataset[T]]

    _: KW_ONLY

    shuffle: bool = False
    r"""Whether to permute the dataset before yielding its items."""
    rng: Generator = RNG
    r"""Generator used to create the optional permutation."""

    index: Sequence = field(init=False)

    def __post_init__(self) -> None:
        match self.data:
            case MapDataset() as dataset:
                index = list(dataset.keys())
            case IndexableDataset() as dataset:
                index = list(range(len(dataset)))
            case _:
                raise TypeError(f"Got unsupported data type {type(self.data)}.")

        object.__setattr__(self, "index", index)

    @property
    def size(self) -> int:
        r"""Number of items yielded by the sampler."""
        return len(self)

    def __len__(self) -> int:
        return len(self.index)

    def __iter__(self) -> Iterator[T]:
        data = self.data
        index = self.index
        size = len(index)
        perm = self.rng.permutation(size) if self.shuffle else range(size)

        for n in perm:
            yield data[index[n]]
