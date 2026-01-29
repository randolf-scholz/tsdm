r"""Samplers for randomly selecting data.

Note:
    For Mapping-style datasets, the sampler will return the keys of the mapping.
"""

__all__ = [
    # ABCs & Protocols
    "BaseSampler",
    "Sampler",
    # Classes
    "RandomSampler",
]

from abc import abstractmethod
from collections.abc import Iterator
from dataclasses import KW_ONLY, dataclass, field
from typing import Final, Protocol, runtime_checkable

from numpy.random import Generator
from pandas import Index

from tsdm.constants import RNG
from tsdm.datatools.datasets import Dataset, PandasDataset, get_index
from tsdm.utils.decorators import pprint_repr


@runtime_checkable
class Sampler[T](Protocol):  # +T
    r"""Protocol for `Sampler` classes.

    Plug-in replacement for `torch.utils.data.Sampler`.
    In contrast, each Sampler must additionally have a `shuffle` attribute.
    """

    @abstractmethod
    def __len__(self) -> int:
        r"""The number of indices that can be drawn by __iter__."""
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        r"""Return an iterator over the indices of the data source."""
        ...

    @property
    @abstractmethod
    def shuffle(self) -> bool:  # pyright: ignore[reportRedeclaration]
        r"""Whether to shuffle the indices."""
        ...

    shuffle: bool  # type: ignore[no-redef]
    # SEE: https://github.com/microsoft/pyright/issues/2601#issuecomment-1545609020

    @property
    @abstractmethod
    def rng(self) -> Generator:  # pyright: ignore[reportRedeclaration]
        r"""The random number generator."""
        ...

    rng: Generator  # type: ignore[no-redef]
    # SEE: https://github.com/microsoft/pyright/issues/2601#issuecomment-1545609020


@dataclass
class BaseSampler[T](Sampler[T]):  # +T
    r"""Abstract Base Class for all Samplers."""

    _: KW_ONLY

    shuffle: bool = False
    r"""Whether to randomize sampling."""
    rng: Generator = RNG
    r"""The random number generator."""

    @abstractmethod
    def __len__(self) -> int:
        r"""Return the length of the sampler."""
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        r"""Return an iterator over the indices of the data source."""
        ...


@pprint_repr
@dataclass
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

    index: Index = field(init=False)
    size: int = field(init=False)

    def __post_init__(self) -> None:
        self.index = get_index(self.data)
        self.size = len(self.index)

    def __iter__(self) -> Iterator[T]:
        n = self.size
        index = self.index[self.rng.permutation(n)] if self.shuffle else self.index
        # avoids attribute lookup in the loop
        data = self.data.loc if isinstance(self.data, PandasDataset) else self.data
        for key in index:
            yield data[key]

    def __len__(self) -> int:
        return self.size
