"""Deprecated samplers."""

__all__ = ["SliceSampler", "CollectionSampler"]

from collections.abc import Callable, Iterator, Mapping, Sequence
from itertools import chain
from typing import Optional

import numpy as np
from numpy._typing import NDArray
from pandas import Index, Series
from typing_extensions import deprecated

from tsdm.random.samplers import BaseSampler
from tsdm.types.variables import any_co as T_co, key_var as K
from tsdm.utils.data import DatasetCollection


@deprecated("Use SlidingWindowSampler instead.")
class SliceSampler(BaseSampler[Sequence[T_co]]):
    r"""Sample by index.

    Default modus operandi:

    - Use fixed window size
    - Sample starting index uniformly from [0:-window]

    Should you want to sample windows of varying size, you may supply a

    Alternatives:

    - sample with fixed horizon and start/stop between bounds
      - [sₖ, tₖ], sᵢ = t₀ + k⋅Δt, tᵢ = t₀ + (k+1)⋅Δt
    - sample with a fixed start location and varying length.
      - [sₖ, tₖ], sᵢ = t₀, tᵢ= t₀ + k⋅Δt
    - sample with a fixed final location and varying length.
      - [sₖ, tₖ], sᵢ = tₗ - k⋅Δt, tᵢ= tₗ
    - sample with varying start and final location and varying length.
      - all slices of length k⋅Δt such that 0 < k⋅Δt < max_length
      - start stop location within bounds [t_min, t_max]
      - start stop locations from the set t_offset + [t_min, t_max] ∩ Δtℤ
      - [sₖ, tⱼ], sᵢ = t₀ + k⋅Δt, tⱼ = t₀ + k⋅Δt

    Attributes:
        data:
        idx: range(len(data))
        rng: a numpy random Generator
    """

    data: Sequence[T_co]
    index: NDArray
    generator: np.random.Generator
    shuffle: bool = False

    def __init__(
        self,
        data_source: Sequence[T_co],
        /,
        *,
        slice_sampler: Optional[int | Callable[[], int]] = None,
        sampler: Optional[Callable[[], tuple[int, int]]] = None,
        generator: Optional[np.random.Generator] = None,
        shuffle: bool = False,
    ) -> None:
        super().__init__(shuffle=shuffle)
        self.data = data_source
        self.index = np.arange(len(data_source))
        self.generator = np.random.default_rng() if generator is None else generator

        match slice_sampler:
            case None:
                self._slice_sampler = lambda: max(1, len(data_source) // 10)
            case int() as number:
                self._slice_sampler = lambda: number
            case Callable() as sampler:  # type: ignore[misc]
                self._slice_sampler = sampler  # type: ignore[unreachable]
            case _:
                raise TypeError("slice_sampler not compatible.")

        def _default_sampler() -> tuple[int, int]:
            window_size: int = self._slice_sampler()
            start_index: int = self.generator.choice(
                self.index[: -1 * window_size]
            )  # -1*w silences pylint.
            return window_size, start_index

        self._sampler = _default_sampler if sampler is None else sampler

    def slice_sampler(self) -> int:
        r"""Return random window size."""
        return self._slice_sampler()

    def sampler(self) -> tuple[int, int]:
        r"""Return random start_index and window_size."""
        return self._sampler()

    def __len__(self) -> int:
        r"""Return the maximum allowed index."""
        return float("inf")  # type: ignore[return-value]

    def __iter__(self) -> Iterator[Sequence[T_co]]:
        r"""Yield random slice from dataset."""
        while True:
            # sample len and index
            window_size, start_index = self.sampler()
            # return slice
            yield self.data[start_index : start_index + window_size]


@deprecated("Use HierarchicalSampler instead.")
class CollectionSampler(BaseSampler[tuple[K, T_co]]):
    r"""Samples a single random dataset from a collection of datasets.

    Optionally, we can delegate a subsampler to then sample from the randomly drawn dataset.
    """

    idx: Index
    r"""The shared index."""
    subsamplers: Mapping[K, BaseSampler[T_co]]
    r"""The subsamplers to sample from the collection."""

    sizes: Series
    r"""The sizes of the subsamplers."""
    partition: Series
    r"""Contains each key a number of times equal to the size of the subsampler."""

    early_stop: bool = False
    r"""Whether to stop sampling when the index is exhausted."""
    shuffle: bool = False
    r"""Whether to sample in random order."""

    def __init__(
        self,
        data_source: DatasetCollection,
        /,
        subsamplers: Mapping[K, BaseSampler[T_co]],
        *,
        shuffle: bool = False,
        early_stop: bool = False,
    ) -> None:
        super().__init__(shuffle=shuffle)
        self.data = data_source
        self.idx = data_source.keys()
        self.subsamplers = dict(subsamplers)
        self.early_stop = early_stop
        self.sizes = Series({key: len(self.subsamplers[key]) for key in self.idx})

        if early_stop:
            partition = list(chain(*([key] * min(self.sizes) for key in self.idx)))
        else:
            partition = list(chain(*([key] * self.sizes[key] for key in self.idx)))
        self.partition = Series(partition)

    def __len__(self) -> int:
        r"""Return the maximum allowed index."""
        if self.early_stop:
            return min(self.sizes) * len(self.subsamplers)
        return sum(self.sizes)

    def __iter__(self) -> Iterator[tuple[K, T_co]]:
        r"""Return indices of the samples.

        When `early_stop=True`, it will sample precisely `min() * len(subsamplers)` samples.
        When `early_stop=False`, it will sample all samples.
        """
        activate_iterators = {
            key: iter(sampler) for key, sampler in self.subsamplers.items()
        }
        perm = np.random.permutation(self.partition)

        for key in perm:
            # This won't raise StopIteration, because the length is matched.
            # value = yield from activate_iterators[key]
            try:
                value = next(activate_iterators[key])
            except StopIteration as exc:
                raise RuntimeError(
                    f"Iterator of {key=} exhausted prematurely."
                ) from exc
            yield key, value

    def __getitem__(self, key: K) -> BaseSampler[T_co]:
        r"""Return the subsampler for the given key."""
        return self.subsamplers[key]
