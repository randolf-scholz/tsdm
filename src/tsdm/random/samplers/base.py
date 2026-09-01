r"""Common interfaces and base implementation for samplers.

Samplers produce a finite sequence that defines a traversal of a data source.
Depending on the implementation, those values may be samples, keys, positions,
or time-window descriptions.
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
    r"""Define the interface shared by all sampler implementations.

    This protocol is compatible in spirit with ``torch.utils.data.Sampler`` and
    additionally requires controls for shuffling and random-number generation.
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
        r"""Return the number of values yielded by this sampler."""
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        r"""Iterate over the values selected from the data source."""
        ...


@dataclass(slots=True)
class BaseSampler[T](metaclass=type(Protocol)):  # pyrefly: ignore[invalid-inheritance]
    r"""Provide common shuffle and random-generator state for samplers."""

    _: KW_ONLY

    shuffle: bool = False
    r"""Whether iteration order is randomized."""
    rng: Generator = RNG
    r"""Generator used when randomized iteration is enabled."""

    # def __init__(self, *, shuffle: bool = False, rng: Generator = RNG) -> None:
    #     self.shuffle = shuffle
    #     self.rng = rng

    @abstractmethod
    def __len__(self) -> int:
        r"""Return the number of values this sampler can yield."""
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        r"""Iterate over the values selected from the data source."""
        ...
