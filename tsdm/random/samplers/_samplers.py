r"""Random Samplers."""

__all__ = [
    # Classes
    "SliceSampler",
    # "TimeSliceSampler",
    "SequenceSampler",
    "CollectionSampler",
    "IntervalSampler",
    "HierarchicalSampler",
]

import logging
from collections.abc import Callable, Iterator, Mapping, Sequence, Sized
from itertools import chain, count
from typing import Any, Generic, Optional, TypeVar, Union

import numpy as np
from numpy.random import permutation
from numpy.typing import NDArray
from pandas import DataFrame, Index, Interval, Series, Timedelta, Timestamp
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Sampler

from tsdm.datasets.torch.generic import DatasetCollection
from tsdm.util.strings import repr_mapping
from tsdm.util.types import ValueType

__logger__ = logging.getLogger(__name__)


TimedeltaLike = TypeVar("TimedeltaLike", int, float, Timedelta)
TimestampLike = TypeVar("TimestampLike", int, float, Timestamp)

Boxed = Union[
    Sequence[ValueType],
    Mapping[int, ValueType],
    Callable[[int], ValueType],
]

dt_type = Union[
    TimedeltaLike,
    Sequence[TimedeltaLike],
    Mapping[int, TimedeltaLike],
    Callable[[int], TimedeltaLike],
]


# class TimeSliceSampler(Sampler):
#     """TODO: add class."""
#
#     def __init__(self, data_source: Optional[Sized]):
#         """TODO: Add method."""
#         super().__init__(data_source)
#
#     def __iter__(self) -> Iterator:
#         """TODO: Add method."""
#         return super().__iter__()


class SliceSampler(Sampler):
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

    Attributes
    ----------
    data:
    idx: range(len(data))
    rng: a numpy random Generator
    """

    data: Sequence
    idx: NDArray
    rng: np.random.Generator

    def __init__(
        self,
        data_source: Sequence,
        slice_sampler: Optional[Union[int, Callable[[], int]]] = None,
        sampler: Optional[Callable[[], tuple[int, int]]] = None,
        generator: Optional[np.random.Generator] = None,
    ):
        super().__init__(data_source)
        self.data = data_source
        self.idx = np.arange(len(data_source))
        self.rng = np.random.default_rng() if generator is None else generator

        def slicesampler_dispatch() -> Callable[[], int]:
            # use default if None is provided
            if slice_sampler is None:
                return lambda: max(1, len(data_source) // 10)
            # convert int to constant function
            if isinstance(slice_sampler, int):
                return lambda: slice_sampler  # type: ignore
            if callable(slice_sampler):
                return slice_sampler
            raise NotImplementedError("slice_sampler not compatible.")

        self._slice_sampler = slicesampler_dispatch()

        def default_sampler() -> tuple[int, int]:
            window_size: int = self._slice_sampler()
            start_index: int = self.rng.choice(
                self.idx[: -1 * window_size]
            )  # -1*w silences pylint.
            return window_size, start_index

        self._sampler = default_sampler if sampler is None else sampler

    def slice_sampler(self) -> int:
        r"""Return random window size."""
        return self._slice_sampler()

    def sampler(self) -> tuple[int, int]:
        r"""Return random start_index and window_size."""
        return self._sampler()

    def __iter__(self) -> Iterator:
        r"""Yield random slice from dataset.

        Returns
        -------
        Iterator
        """
        while True:
            # sample len and index
            window_size, start_index = self.sampler()
            # return slice
            yield self.data[start_index : start_index + window_size]


class SequenceSampler(Sampler):
    r"""Samples sequences of length seq_len."""

    data: Sized
    r"""The dataset."""
    idx: NDArray
    r"""A list of all valid starting indices."""
    seq_len: int
    r"""The static sequence length."""
    shuffle: bool
    r"""Whether to sample in random order."""

    def __init__(self, data_source: Sized, seq_len: int, shuffle: bool = True):
        r"""Initialize the Sampler.

        Parameters
        ----------
        data_source: Sized
        seq_len: int
        shuffle: bool = True
        """
        super().__init__(data_source)
        self.data = data_source
        self.seq_len = seq_len
        self.idx = np.arange(len(self.data) - self.seq_len)
        self.shuffle = shuffle

    def __len__(self):
        r"""Return the maximum allowed index."""
        return len(self.idx)

    def __iter__(self):
        r"""Return Indices of the Samples."""
        indices = self.idx[permutation(len(self))] if self.shuffle else self.idx

        for i in indices:
            yield np.arange(i, i + self.seq_len)


# class CollectionSampler(Sampler):
#     r"""Samples a single random  object from"""
#
#     def __init__(self, data_source: Sized, shuffle: bool = True):
#         super().__init__(data_source)
#         self.data = data_source
#         self.shuffle = shuffle
#         assert hasattr(data_source, "index"), "Data must have index."
#         assert isinstance(data_source.index, Index), "Index must be ``pandas.Index``."
#         self.idx = data_source.index
#
#     def __len__(self):
#         r"""Return the maximum allowed index."""
#         return len(self.idx)
#
#     def __iter__(self):
#         r"""Return Indices of the Samples."""
#         indices = self.idx[permutation(len(self))] if self.shuffle else self.idx
#
#         for i in indices:
#             yield i


class CollectionSampler(Sampler):
    r"""Samples a single random dataset from a collection of dataset.

    Optionally, we can delegate a subsampler to then sample from the randomly drawn dataset.
    """

    idx: Index
    r"""The shared index."""
    subsamplers: Mapping[Any, Sampler]
    r"""The subsamplers to sample from the collection."""
    early_stop: bool = False
    r"""Whether to stop sampling when the index is exhausted."""
    shuffle: bool = True
    r"""Whether to sample in random order."""
    sizes: Series
    r"""The sizes of the subsamplers."""
    partition: Series
    r"""Contains each key a number of times equal to the size of the subsampler."""

    def __init__(
        self,
        data_source: DatasetCollection,
        subsamplers: Mapping[Any, Sampler],
        shuffle: bool = True,
        early_stop: bool = False,
    ):
        super().__init__(data_source)
        self.data = data_source
        self.shuffle = shuffle
        self.idx = data_source.keys()
        self.subsamplers = dict(subsamplers)
        self.early_stop = early_stop
        self.sizes = Series({key: len(self.subsamplers[key]) for key in self.idx})

        if early_stop:
            partition = list(chain(*([key] * min(self.sizes) for key in self.idx)))
        else:
            partition = list(chain(*([key] * self.sizes[key] for key in self.idx)))
        self.partition = Series(partition)

    def __len__(self):
        r"""Return the maximum allowed index."""
        if self.early_stop:
            return min(self.sizes) * len(self.subsamplers)
        return sum(self.sizes)

    def __iter__(self):
        r"""Return indices of the samples.

        When ``early_stop=True``, it will sample precisely ``min() * len(subsamplers)`` samples.
        When ``early_stop=False``, it will sample all samples.
        """
        activate_iterators = {
            key: iter(sampler) for key, sampler in self.subsamplers.items()
        }
        perm = np.random.permutation(self.partition)

        for key in perm:
            yield key, next(activate_iterators[key])

    def __getitem__(self, key: Any) -> Sampler:
        r"""Return the subsampler for the given key."""
        return self.subsamplers[key]


# class MappingSampler(Sampler):
#     r"""Sample from a Mapping of Datasets.
#
#     To be used in conjunction with :class:`tsdm.datasets.torch.MappingDataset`.
#     """
#
#     def __init__(self, data_source: Mapping[Any, TorchDataset], shuffle: bool = True):
#         super().__init__(data_source)
#         self.data = data_source
#         self.shuffle = shuffle
#         self.index = list(data_source.keys())
#
#     def __len__(self) -> int:
#         r"""Return the maximum allowed index."""
#         return len(self.data)
#
#     def __iter__(self) -> Iterator[TorchDataset]:
#         r"""Sample from the dataset."""
#         if self.shuffle:
#             perm = np.random.permutation(self.index)
#         else:
#             perm = self.index
#
#         for k in perm:
#             yield self.data[k]


class HierarchicalSampler(Sampler):
    r"""Samples a single random dataset from a collection of dataset.

    Optionally, we can delegate a subsampler to then sample from the randomly drawn dataset.
    """

    idx: Index
    r"""The shared index."""
    subsamplers: Mapping[Any, Sampler]
    r"""The subsamplers to sample from the collection."""
    early_stop: bool = False
    r"""Whether to stop sampling when the index is exhausted."""
    shuffle: bool = True
    r"""Whether to sample in random order."""
    sizes: Series
    r"""The sizes of the subsamplers."""
    partition: Series
    r"""Contains each key a number of times equal to the size of the subsampler."""

    def __init__(
        self,
        data_source: Mapping[Any, TorchDataset],
        subsamplers: Mapping[Any, Sampler],
        shuffle: bool = True,
        early_stop: bool = False,
    ):
        super().__init__(data_source)
        self.data = data_source
        self.idx = data_source.keys()
        self.subsamplers = dict(subsamplers)
        self.sizes = Series({key: len(self.subsamplers[key]) for key in self.idx})
        self.shuffle = shuffle
        self.early_stop = early_stop

        if early_stop:
            partition = list(chain(*([key] * min(self.sizes) for key in self.idx)))
        else:
            partition = list(chain(*([key] * self.sizes[key] for key in self.idx)))
        self.partition = Series(partition)

    def __len__(self):
        r"""Return the maximum allowed index."""
        if self.early_stop:
            return min(self.sizes) * len(self.subsamplers)
        return sum(self.sizes)

    def __iter__(self):
        r"""Return indices of the samples.

        When ``early_stop=True``, it will sample precisely min() * len(subsamplers) samples.
        When ``early_stop=False``, it will sample all samples.
        """
        activate_iterators = {
            key: iter(sampler) for key, sampler in self.subsamplers.items()
        }

        if self.shuffle:
            perm = np.random.permutation(self.partition)
        else:
            perm = self.partition

        for key in perm:
            yield key, next(activate_iterators[key])

    def __getitem__(self, key: Any) -> Sampler:
        r"""Return the subsampler for the given key."""
        return self.subsamplers[key]

    def __repr__(self) -> str:
        r"""Pretty print."""
        return repr_mapping(self.subsamplers)


class IntervalSampler(Sampler, Generic[TimedeltaLike]):
    r"""Returns all intervals `[a, b]`.

    The intervals must satisfy:

    - `a = t₀ + i⋅sₖ`
    - `b = t₀ + i⋅sₖ + Δtₖ`
    - `i, k ∈ ℤ`
    - `a ≥ t_min`
    - `b ≤ t_max`
    - `sₖ` is the stride corresponding to intervals of size `Δtₖ`.
    """

    offset: TimedeltaLike
    deltax: dt_type
    stride: dt_type
    shuffle: bool
    intervals: DataFrame

    @staticmethod
    def _get_value(
        obj: Union[TimedeltaLike, Boxed[TimedeltaLike]], k: int
    ) -> TimedeltaLike:
        if callable(obj):
            return obj(k)
        if isinstance(obj, Sequence):
            return obj[k]
        if isinstance(obj, Mapping):
            return obj[k]
        # Fallback: multiple!
        return obj

    def __init__(
        self,
        xmin: TimedeltaLike,
        xmax: TimedeltaLike,
        deltax: dt_type,
        stride: Optional[dt_type] = None,
        levels: Optional[Sequence[int]] = None,
        offset: Optional[TimedeltaLike] = None,
        multiples: bool = True,
        shuffle: bool = True,
    ) -> None:
        super().__init__(None)

        # set stride and offset
        zero = 0 * (xmax - xmin)
        stride = zero if stride is None else stride
        offset = xmin if offset is None else offset

        # validate bounds
        assert xmin <= offset <= xmax, "Assumption: xmin≤xoffset≤xmax violated!"

        # determine delta_max
        delta_max = max(offset - xmin, xmax - offset)

        # determine levels
        if levels is None:
            if isinstance(deltax, Mapping):
                levels = [k for k in deltax.keys() if deltax[k] <= delta_max]
            elif isinstance(deltax, Sequence):
                levels = [k for k in range(len(deltax)) if deltax[k] <= delta_max]
            elif callable(deltax):
                levels = []
                for k in count():
                    dt = self._get_value(deltax, k)
                    if dt == zero:
                        continue
                    if dt > delta_max:
                        break
                    levels.append(k)
            else:
                levels = [0]
        else:
            levels = [k for k in levels if self._get_value(deltax, k) <= delta_max]

        # validate levels
        assert all(self._get_value(deltax, k) <= delta_max for k in levels)

        # compute valid intervals
        intervals: list[Interval] = []

        # for each level, get all intervals
        for k in levels:
            dt = self._get_value(deltax, k)
            st = self._get_value(stride, k)
            x0 = self._get_value(offset, k)

            # get valid interval bounds, probably there is an easier way to do it...
            stridesa = grid(xmin, xmax, st, x0)
            stridesb = grid(xmin, xmax, st, x0 + dt)
            valid_strides = set.intersection(set(stridesa), set(stridesb))

            if not valid_strides:
                break

            intervals.extend(
                [(x0 + i * st, x0 + i * st + dt, dt, st) for i in valid_strides]
            )

        # set variables
        self.offset = offset
        self.deltax = deltax
        self.stride = stride
        self.shuffle = shuffle
        self.intervals = DataFrame(
            intervals, columns=["left", "right", "delta", "stride"]
        )

    def __iter__(self) -> Iterator[slice]:
        r"""Return an iterator over the intervals."""
        if self.shuffle:
            perm = np.random.permutation(len(self))
        else:
            perm = np.arange(len(self))

        for k in perm:
            yield slice(self.loc[k, "left"], self.loc[k, "right"])

    def __len__(self) -> int:
        r"""Length of the sampler."""
        return len(self.intervals)

    def __getattr__(self, key: str) -> Any:
        r"""Forward all other attributes to the interval frame."""
        return self.intervals.__getattr__(key)

    def __getitem__(self, key: Any) -> slice:
        r"""Return a slice from the sampler."""
        return self.intervals[key]


def grid(
    xmin: TimestampLike,
    xmax: TimestampLike,
    delta: TimedeltaLike,
    xoffset: Optional[TimestampLike] = None,
) -> list[int]:
    r"""Compute `\{k∈ℤ∣ xₘᵢₙ ≤ x₀+k⋅Δ ≤ xₘₐₓ\}`.

    That is, a list of all integers such that `x₀+k⋅Δ` is in the interval `[x₀, xₘₐₓ]`.
    Special case: if `Δ=0`, returns ``[0]``

    Parameters
    ----------
    xmin
    xmax
    delta
    xoffset

    Returns
    -------
    list[int]
    """
    xoffset = xmin if xoffset is None else xoffset
    zero = type(delta)(0)

    if delta == zero:
        return [0]

    assert delta > zero, "Assumption delta>0 violated!"
    assert xmin <= xoffset <= xmax, "Assumption: xmin≤xoffset≤xmax violated!"

    a = xmin - xoffset
    b = xmax - xoffset
    kmax = int(b // delta)  # need int() in case both are floats
    kmin = int(a // delta)

    assert xmin <= xoffset + kmin * delta
    assert xmin > xoffset + (kmin - 1) * delta
    assert xmax >= xoffset + kmax * delta
    assert xmax < xoffset + (kmax + 1) * delta

    return list(range(kmin, kmax + 1))


# class BatchSampler(Sampler[list[int]]):
#     r"""Wraps another sampler to yield a mini-batch of indices.
#
#     Args:
#         sampler (Sampler or Iterable): Base sampler. Can be any iterable object
#         batch_size (int): Size of mini-batch.
#         drop_last (bool): If ``True``, the sampler will drop the last batch if
#             its size would be less than ``batch_size``
#
#     Example:
#         >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
#         [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
#         >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
#         [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
#     """
#
#     def __init__(self, sampler: Sampler[int], batch_size: int, drop_last: bool) -> None:
#         # Since collections.abc.Iterable does not check for `__getitem__`, which
#         # is one way for an object to be an iterable, we don't do an `isinstance`
#         # check here.
#         super().__init__(sampler)
#         if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
#                 batch_size <= 0:
#             raise ValueError("batch_size should be a positive integer value, "
#                              "but got batch_size={}".format(batch_size))
#         if not isinstance(drop_last, bool):
#             raise ValueError("drop_last should be a boolean value, but got "
#                              "drop_last={}".format(drop_last))
#         self.sampler = sampler
#         self.batch_size = batch_size
#         self.drop_last = drop_last
#
#     def __iter__(self) -> Iterator[list[int]]:
#         batch = []
#         for idx in self.sampler:
#             batch.append(idx)
#             if len(batch) == self.batch_size:
#                 yield batch
#                 batch = []
#         if len(batch) > 0 and not self.drop_last:
#             yield batch
#
#     def __len__(self) -> int:
#         # Can only be called if self.sampler has __len__ implemented
#         # We cannot enforce this condition, so we turn off typechecking for the
#         # implementation below.
#         # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
#         if self.drop_last:
#             return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
#         else:
#             return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]
