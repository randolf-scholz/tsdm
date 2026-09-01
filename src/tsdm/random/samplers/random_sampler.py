__all__ = ["RandomSampler"]

from collections.abc import Iterator, Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import Final

from numpy.random import Generator

from tsdm.constants import RNG
from tsdm.datatools import Dataset, PandasDataset, get_index
from tsdm.pprint import pprint_repr

from .base import BaseSampler


@pprint_repr
@dataclass(slots=True)
class RandomSampler[T](BaseSampler[T]):  # +T
    r"""Sample randomly from the data source.

    Note:
        In contrast to torch.utils.data.RandomSampler, this sampler also works for map-style datasets.
        In this case, the sampler will return random values of the mapping.
        For Iterable-style datasets, the sampler will return random values of the iterable.
    """

    data: Final[Dataset[T]]

    _: KW_ONLY

    shuffle: bool = False
    r"""Whether to randomize sampling."""
    rng: Generator = RNG
    r"""The random number generator."""

    index: Sequence = field(init=False)
    size: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", get_index(self.data))
        object.__setattr__(self, "size", len(self.index))

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[T]:
        perm = self.rng.permutation(self.size) if self.shuffle else range(self.size)

        # avoids attribute lookup in the loop
        data = self.data.loc if isinstance(self.data, PandasDataset) else self.data
        index = self.index

        for n in perm:
            yield data[index[n]]
