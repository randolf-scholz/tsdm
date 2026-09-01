r"""Samplers for randomly selecting data.

Note:
    For Mapping-style datasets, the sampler will return the keys of the mapping.
"""

__all__ = ["BaseSampler", "Sampler"]

from abc import abstractmethod
from collections.abc import Iterator
from dataclasses import KW_ONLY, dataclass
from typing import Protocol, runtime_checkable

from numpy.random import Generator

from tsdm.constants import RNG


@runtime_checkable
class Sampler[T](Protocol):  # +T
    r"""Protocol for `Sampler` classes.

    Plug-in replacement for `torch.utils.data.Sampler`.
    In contrast, each Sampler must additionally have a `shuffle` attribute.
    """

    # TODO: Use typing.ReadOnly (PEP 767)
    @property
    @abstractmethod
    def shuffle(self) -> bool: ...
    @property
    @abstractmethod
    def rng(self) -> Generator: ...

    @abstractmethod
    def __len__(self) -> int:
        r"""The number of indices that can be drawn by __iter__."""
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        r"""Return an iterator over the indices of the data source."""
        ...


@dataclass(slots=True)
class BaseSampler[T](metaclass=type(Protocol)):  # pyrefly: ignore[invalid-inheritance]
    r"""Abstract Base Class for all Samplers."""

    _: KW_ONLY

    shuffle: bool = False
    r"""Whether to randomize sampling."""
    rng: Generator = RNG
    r"""The random number generator."""

    # def __init__(self, *, shuffle: bool = False, rng: Generator = RNG) -> None:
    #     self.shuffle = shuffle
    #     self.rng = rng

    @abstractmethod
    def __len__(self) -> int:
        r"""Return the length of the sampler."""
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        r"""Return an iterator over the indices of the data source."""
        ...
