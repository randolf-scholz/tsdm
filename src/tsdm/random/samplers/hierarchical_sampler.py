r"""Sampling and dataset adapters for two-level collections.

These utilities treat a mapping of datasets as a flat collection of
``(outer_key, inner_key)`` pairs while retaining access to its nested layout.
"""

__all__ = ["HierarchicalSampler", "HierarchicalDataset"]

from collections.abc import Collection, Iterator, Mapping
from dataclasses import KW_ONLY, dataclass, field
from itertools import chain
from typing import Any, cast, overload

from numpy.random import Generator

from tsdm.constants import EMPTY_MAP, RNG
from tsdm.datatools.collections import Dataset, MapDataset, get_index
from tsdm.pprint import pprint_repr
from tsdm.types import SupportsGetItem

from .base import BaseSampler, Sampler
from .random_sampler import RandomSampler


@pprint_repr
@dataclass(slots=True, init=False)
class HierarchicalSampler[K, K2](BaseSampler[tuple[K, K2]]):
    r"""Flatten a mapping of datasets into pairs of outer and inner keys.

    Each nested dataset has its own sampler. With ``early_stop=False`` every
    nested sampler is exhausted; with ``early_stop=True`` all groups contribute
    the same number of samples.

    Example:
        >>> from tsdm.random.samplers import HierarchicalSampler, RandomSampler
        >>> sampler = HierarchicalSampler(
        ...     {"A": [1, 2, 3], "B": [4, 5, 6]}, shuffle=False
        ... )
        >>> list(sampler)
        [('A', 1), ('A', 2), ('A', 3), ('B', 4), ('B', 5), ('B', 6)]

    Args:
        data: Mapping from outer keys to datasets.
        subsamplers: Per-dataset samplers, created automatically when omitted.
        early_stop: Stop after every group has produced the smallest group size.
        shuffle: Whether to shuffle the order of the emitted key pairs.
        rng: Generator used to shuffle the combined sampling schedule.
    """

    data: MapDataset[K, Dataset[K2]]
    r"""Mapping of outer keys to the datasets being sampled."""

    _: KW_ONLY

    subsamplers: Mapping[K, Sampler[K2]]
    r"""Sampler assigned to each nested dataset."""
    early_stop: bool = False
    r"""Whether each group is limited to the smallest sampler length."""
    shuffle: bool = False
    r"""Whether the combined key-pair schedule is shuffled."""
    rng: Generator = RNG
    r"""Generator used to shuffle the combined schedule."""

    # derived fields
    index: Collection[K] = field(init=False)
    r"""Outer keys available in the source mapping."""
    sizes: Mapping[K, int] = field(init=False)
    r"""Number of available inner keys for each outer key."""
    partition: list[K] = field(init=False)
    r"""Outer-key schedule aligned with the nested sampler lengths."""

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
                for key in self.data.keys()  # ruff: ignore[SIM118]
            }
        )

        self.index: Collection[K] = get_index(self.data)

        # get the sizes of the subsamplers
        self.sizes: Mapping[K, int] = {
            key: len(self.subsamplers[key]) for key in self.index
        }

        # duplicate the outer keys according to the sizes of the subsamplers
        self.partition: list[K] = list(
            chain(*([key] * min(self.sizes.values()) for key in self.index))
            if self.early_stop
            else chain(*([key] * self.sizes[key] for key in self.index))
        )

    def __len__(self) -> int:
        r"""Return the number of key pairs yielded during iteration."""
        if self.early_stop:
            return min(self.sizes.values()) * len(self.subsamplers)
        return sum(self.sizes.values())

    def __getitem__(self, key: K, /) -> Sampler[K2]:
        r"""Return the sampler associated with an outer key."""
        return self.subsamplers[key]

    def __iter__(self) -> Iterator[tuple[K, K2]]:
        r"""Yield outer and inner keys selected by the nested samplers.

        The outer-key schedule is optionally shuffled, while each inner key is
        obtained from the corresponding nested sampler in its own iteration order.
        """
        iterators = {key: iter(sampler) for key, sampler in self.subsamplers.items()}
        n = len(self.partition)
        index = self.rng.permutation(n) if self.shuffle else range(n)
        for i in index:
            key = self.partition[i]
            # This won't raise `StopIteration`, because the length is matched.
            yield key, next(iterators[key])


@pprint_repr
class HierarchicalDataset[OuterKeyT, InnerKeyT, SampleT](
    Mapping[OuterKeyT, SupportsGetItem[InnerKeyT, SampleT]]
):
    r"""Expose a mapping of datasets with convenient nested-key lookup.

    An outer key returns its complete nested dataset. A pair of outer and inner
    keys returns the sample from that dataset, equivalent to
    ``datasets[outer_key][inner_key]``.
    """

    datasets: Mapping[OuterKeyT, SupportsGetItem[InnerKeyT, SampleT]]
    index: list[OuterKeyT]

    def __init__(
        self, datasets: Mapping[OuterKeyT, SupportsGetItem[InnerKeyT, SampleT]], /
    ) -> None:
        super().__init__()
        self.index = list(datasets.keys())
        self.datasets = datasets

    def __iter__(self) -> Iterator[OuterKeyT]:
        r"""Iterate over the outer dataset keys in insertion order."""
        return iter(self.index)

    def __len__(self) -> int:
        r"""Return the number of nested datasets."""
        return len(self.index)

    @overload
    def __getitem__(self, key: OuterKeyT, /) -> SupportsGetItem[InnerKeyT, SampleT]: ...
    @overload
    def __getitem__(self, key: tuple[OuterKeyT, InnerKeyT], /) -> SampleT: ...
    def __getitem__(self, key: OuterKeyT | tuple[OuterKeyT, InnerKeyT], /) -> Any:
        r"""Return a nested dataset or one of its samples by key.

        A single key returns the corresponding dataset; an ``(outer, inner)``
        tuple returns the nested sample.
        """
        match key:
            case k if key in self:
                return self.datasets[cast("OuterKeyT", k)]
            case [outer_key, inner_key]:
                dataset = self.datasets[outer_key]
                return dataset[inner_key]
            case _:
                raise KeyError(key)
