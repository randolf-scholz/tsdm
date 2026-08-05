r"""Implementation of hierarchical sampler."""

__all__ = ["HierarchicalSampler"]

from collections.abc import Collection, Iterator, Mapping
from dataclasses import KW_ONLY, dataclass
from itertools import chain

from numpy.random import Generator
from pandas import Series

from tsdm.constants import EMPTY_MAP, RNG
from tsdm.datatools import Dataset, MapDataset
from tsdm.datatools.collections import SeriesDataset, get_index
from tsdm.pprint import pprint_repr

from .base import BaseSampler, RandomSampler, Sampler


@pprint_repr
@dataclass(init=False)
class HierarchicalSampler[K, K2](BaseSampler[tuple[K, K2]]):
    r"""Draw samples from a hierarchical data source.

    Example:
        >>> from tsdm.random.samplers import HierarchicalSampler, RandomSampler
        >>> sampler = HierarchicalSampler(
        ...     {"A": [1, 2, 3], "B": [4, 5, 6]}, shuffle=False
        ... )
        >>> list(sampler)
        [('A', 1), ('A', 2), ('A', 3), ('B', 4), ('B', 5), ('B', 6)]

    Args:
        shuffle: Whether to sample in random order.
        rng: The random number generator.
        early_stop: Ensure each group is sampled the same number of times.
    """

    data: MapDataset[K, Dataset[K2]]
    r"""The shared index."""

    _: KW_ONLY

    subsamplers: dict[K, Sampler[K2]]
    r"""The subsamplers to sample from the collection."""
    early_stop: bool = False
    r"""Whether to stop sampling when the index is exhausted."""
    shuffle: bool = False
    r"""Whether to sample in random order."""
    rng: Generator = RNG
    r"""The random number generator."""

    def __init__(
        self,
        data: MapDataset[K, Dataset[K2]],
        /,
        subsamplers: Mapping[K, Sampler[K2]] = EMPTY_MAP,
        *,
        early_stop: bool = False,
        shuffle: bool = False,
        rng: Generator = RNG,
    ) -> None:
        super().__init__(shuffle=shuffle, rng=rng)
        self.data = data
        self.early_stop = early_stop

        self.subsamplers = (
            dict(subsamplers)
            if subsamplers is not EMPTY_MAP
            else {
                key: RandomSampler(self.data[key], shuffle=self.shuffle)
                for key in self.data.keys()  # noqa: SIM118
            }
        )

        self.index: Collection[K] = get_index(self.data)

        # get the sizes of the subsamplers
        self.sizes: SeriesDataset[K, int] = Series(
            {key: len(self.subsamplers[key]) for key in self.index}
        )

        # duplicate the outer keys according to the sizes of the subsamplers
        self.partition: list[K] = list(
            chain(*([key] * min(self.sizes) for key in self.index))
            if self.early_stop
            else chain(*([key] * self.sizes[key] for key in self.index))
        )

    def __len__(self) -> int:
        r"""Return the maximum allowed index."""
        if self.early_stop:
            return min(self.sizes) * len(self.subsamplers)
        return sum(self.sizes)

    def __getitem__(self, key: K, /) -> Sampler[K2]:
        r"""Return the subsampler for the given key."""
        return self.subsamplers[key]

    def __iter__(self) -> Iterator[tuple[K, K2]]:
        r"""Yield indices of the samples.

        When ``early_stop=True``, it will sample precisely ``min() * len(subsamplers)`` samples.
        When ``early_stop=False``, it will sample all samples.
        """
        iterators = {key: iter(sampler) for key, sampler in self.subsamplers.items()}
        n = len(self.partition)
        index = self.rng.permutation(n) if self.shuffle else range(n)
        for i in index:
            key = self.partition[i]
            # This won't raise `StopIteration`, because the length is matched.
            yield key, next(iterators[key])
