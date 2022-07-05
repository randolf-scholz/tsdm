r"""Random Samplers."""

__all__ = [
    # ABCs
    "BaseSampler",
    # Classes
    "SliceSampler",
    # "TimeSliceSampler",
    "SequenceSampler",
    "CollectionSampler",
    "IntervalSampler",
    "HierarchicalSampler",
    "SlidingWindowSampler",
    # Functions
    "grid",
]

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence, Sized
from itertools import chain, count
from typing import Any, Generic, Literal, Optional, Union, cast

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from pandas import DataFrame, Index, Series, Timedelta, Timestamp
from torch.utils.data import Sampler

from tsdm.util.strings import repr_mapping
from tsdm.util.torch.generic import DatasetCollection
from tsdm.util.types import ObjectType, ValueType
from tsdm.util.types.time import NumpyDTVar, NumpyTDVar, TDVar

__logger__ = logging.getLogger(__name__)


Boxed = Union[
    Sequence[ValueType],
    Mapping[int, ValueType],
    Callable[[int], ValueType],
]

Nested = Union[
    ObjectType,
    Sequence[ObjectType],
    Mapping[int, ObjectType],
    Callable[[int], ObjectType],
]


# class TimeSliceSampler(Sampler):
#     r"""TODO: add class."""
#
#     def __init__(self, data_source: Optional[Sized]):
#         r"""TODO: Add method."""
#         super().__init__(data_source)
#
#     def __iter__(self) -> Iterator:
#         r"""TODO: Add method."""
#         return super().__iter__()


class BaseSampler(Sampler, Sized, ABC):
    r"""Abstract Base Class for all Samplers."""

    data: Sized
    r"""Copy of the original Data source."""

    def __init__(self, data_source: Sized, /) -> None:
        r"""Initialize the sampler."""
        super().__init__(data_source)
        self.data = data_source

    @abstractmethod
    def __len__(self) -> int:
        r"""Return the length of the sampler."""

    @abstractmethod
    def __iter__(self) -> Iterator:
        r"""Iterate over random indices."""


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
        /,
        *,
        slice_sampler: Optional[Union[int, Callable[[], int]]] = None,
        sampler: Optional[Callable[[], tuple[int, int]]] = None,
        generator: Optional[np.random.Generator] = None,
    ):
        super().__init__(data_source)
        self.data = data_source
        self.idx = np.arange(len(data_source))
        self.rng = np.random.default_rng() if generator is None else generator

        def _slicesampler_dispatch() -> Callable[[], int]:
            # use default if None is provided
            if slice_sampler is None:
                return lambda: max(1, len(data_source) // 10)
            # convert int to constant function
            if callable(slice_sampler):
                return slice_sampler
            if isinstance(slice_sampler, int):
                return lambda: slice_sampler  # type: ignore[return-value]
            raise NotImplementedError("slice_sampler not compatible.")

        self._slice_sampler = _slicesampler_dispatch()

        def _default_sampler() -> tuple[int, int]:
            window_size: int = self._slice_sampler()
            start_index: int = self.rng.choice(
                self.idx[: -1 * window_size]
            )  # -1*w silences pylint.
            return window_size, start_index

        self._sampler = _default_sampler if sampler is None else sampler

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


# class SequenceSampler(Sampler):
#     r"""Samples sequences of length seq_len."""
#
#     data: Sized
#     r"""The dataset."""
#     idx: NDArray
#     r"""A list of all valid starting indices."""
#     seq_len: int
#     r"""The static sequence length."""
#     shuffle: bool
#     r"""Whether to sample in random order."""
#
#     def __init__(self, data_source: Sized, /, *, seq_len: int, shuffle: bool = True):
#         r"""Initialize the Sampler.
#
#         Parameters
#         ----------
#         data_source: Sized
#         seq_len: int
#         shuffle: bool = True
#         """
#         super().__init__(data_source)
#         self.data = data_source
#         self.seq_len = seq_len
#         self.idx = np.arange(len(self.data) - self.seq_len)
#         self.shuffle = shuffle
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
#             yield np.arange(i, i + self.seq_len)


# class CollectionSampler(Sampler):
#     r"""Samples a single random  object from."""
#
#     def __init__(self, data_source: Sized, shuffle: bool = True):
#         super().__init__(data_source)
#         self.data = data_source
#         self.shuffle = shuffle
#         assert hasattr(data_source, "index"), "Data must have index."
#         assert isinstance(data_source.index, Index), "Index must be `pandas.Index`."
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
        /,
        subsamplers: Mapping[Any, Sampler],
        *,
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
            except StopIteration as E:
                raise RuntimeError(f"Iterator of {key=} exhausted prematurely.") from E
            else:
                yield key, value

    def __getitem__(self, key: Any) -> Sampler:
        r"""Return the subsampler for the given key."""
        return self.subsamplers[key]


# class MappingSampler(Sampler):
#     r"""Sample from a Mapping of Datasets.
#
#     To be used in conjunction with `tsdm.datasets.torch.MappingDataset`.
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
        data_source: Mapping[Any, Any],
        /,
        subsamplers: Mapping[Any, Sampler],
        *,
        shuffle: bool = True,
        early_stop: bool = False,
    ):
        super().__init__(data_source)
        self.data = data_source
        self.idx = Index(data_source.keys())
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

        When ``early_stop=True``, it will sample precisely ``min() * len(subsamplers)`` samples.
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
            # This won't raise StopIteration, because the length is matched.
            try:
                value = next(activate_iterators[key])
            except StopIteration as E:
                raise RuntimeError(f"Iterator of {key=} exhausted prematurely.") from E
            else:
                yield key, value

    def __getitem__(self, key: Any) -> Sampler:
        r"""Return the subsampler for the given key."""
        return self.subsamplers[key]

    def __repr__(self) -> str:
        r"""Pretty print."""
        return repr_mapping(self.subsamplers)


class IntervalSampler(Sampler, Generic[TDVar]):
    r"""Returns all intervals `[a, b]`.

    The intervals must satisfy:

    - `a = t₀ + i⋅sₖ`
    - `b = t₀ + i⋅sₖ + Δtₖ`
    - `i, k ∈ ℤ`
    - `a ≥ t_min`
    - `b ≤ t_max`
    - `sₖ` is the stride corresponding to intervals of size `Δtₖ`.
    """

    offset: TDVar
    deltax: Nested[TDVar]
    stride: Nested[TDVar]
    shuffle: bool
    intervals: DataFrame

    @staticmethod
    def _get_value(obj: Union[TDVar, Boxed[TDVar]], k: int) -> TDVar:
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
        *,
        xmin: TDVar,
        xmax: TDVar,
        deltax: Nested[TDVar],
        stride: Optional[Nested[TDVar]] = None,
        levels: Optional[Sequence[int]] = None,
        offset: Optional[TDVar] = None,
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
        delta_max = max(offset - xmin, xmax - offset)  # type: ignore[call-overload]

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
        intervals: list[
            tuple[Nested[TDVar], Nested[TDVar], Nested[TDVar], Nested[TDVar]]
        ] = []

        # for each level, get all intervals
        for k in levels:
            dt = self._get_value(deltax, k)
            st = self._get_value(stride, k)
            x0 = self._get_value(offset, k)

            # get valid interval bounds, probably there is an easier way to do it...
            stride_left: list[int] = grid(xmin, xmax, st, x0)
            stride_right: list[int] = grid(xmin, xmax, st, x0 + dt)
            valid_strides: set[int] = set.intersection(
                set(stride_left), set(stride_right)
            )

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
    tmin: Union[str, TDVar],
    tmax: Union[str, TDVar],
    timedelta: Union[str, TDVar],
    offset: Union[None, str, TDVar] = None,
) -> list[int]:
    r"""Compute $\{k∈ℤ∣ tₘᵢₙ ≤ t₀+k⋅Δt ≤ tₘₐₓ\}$.

    That is, a list of all integers such that $t₀+k⋅Δ$ is in the interval $[tₘᵢₙ, tₘₐₓ]$.
    Special case: if $Δt=0$, returns $[0]$.

    Parameters
    ----------
    tmin
    tmax
    timedelta
    offset

    Returns
    -------
    list[int]
    """
    # cast strings to timestamp/timedelta
    tmin = Timestamp(tmin) if isinstance(tmin, str) else tmin
    tmax = Timestamp(tmax) if isinstance(tmax, str) else tmax
    timedelta = Timedelta(timedelta) if isinstance(timedelta, str) else timedelta
    offset = Timestamp(offset) if isinstance(offset, str) else offset

    offset = tmin if offset is None else offset
    zero_dt = tmin - tmin  # generates zero variable of correct type

    if timedelta == zero_dt:
        return [0]

    assert timedelta > zero_dt, "Assumption delta>0 violated!"
    assert tmin <= offset <= tmax, "Assumption: xmin≤xoffset≤xmax violated!"

    a = tmin - offset
    b = tmax - offset
    kmax = int(b // timedelta)  # need int() in case both are floats
    kmin = int(a // timedelta)

    assert tmin <= offset + kmin * timedelta
    assert tmin > offset + (kmin - 1) * timedelta
    assert tmax >= offset + kmax * timedelta
    assert tmax < offset + (kmax + 1) * timedelta

    return list(range(kmin, kmax + 1))


# class BatchSampler(Sampler[list[int]]):
#     r"""Wraps another sampler to yield a mini-batch of indices.
#
#     Args:
#         sampler (Sampler or Iterable): Base sampler. Can be any iterable object
#         batch_size (int): Size of mini-batch.
#         drop_last (bool): If `True`, the sampler will drop the last batch if
#             its size would be less than `batch_size`
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
#             return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class SequenceSampler(BaseSampler):
    r"""Samples sequences of length seq_len."""

    def __init__(
        self,
        data_source: Sequence[Any],
        *,
        xmin: Optional[int] = None,
        xmax: Optional[int] = None,
        stride: int = 1,
        seq_len: int,
        return_mask: bool = False,
        shuffle: bool = False,
    ) -> None:
        super().__init__(data_source)
        self.data_source = data_source

        xmin = xmin if xmin is not None else data_source[0]
        xmax = xmax if xmax is not None else data_source[-1]

        self.xmin = xmin if not isinstance(xmin, str) else Timestamp(xmin)
        self.xmax = xmax if not isinstance(xmax, str) else Timestamp(xmax)

        self.stride = stride if not isinstance(stride, str) else Timedelta(stride)
        self.seq_len = seq_len if not isinstance(seq_len, str) else Timedelta(seq_len)
        # k_max = max {k∈ℕ ∣ x_min + seq_len + k⋅stride ≤ x_max}
        self.k_max = int((xmax - xmin - seq_len) // stride)
        self.return_mask = return_mask
        self.shuffle = shuffle

        self.samples = np.array(
            [
                (x <= self.data_source) & (self.data_source < y)
                if self.return_mask
                else [x, y]
                for x, y in self._iter_tuples()
            ]
        )

    def _iter_tuples(self) -> Iterator[tuple[Any, Any]]:
        x = self.xmin
        y = x + self.seq_len
        x, y = min(x, y), max(x, y)  # allows nice handling of negative seq_len
        yield x, y

        for _ in range(len(self)):
            x += self.stride
            y += self.stride
            yield x, y

    def __len__(self) -> int:
        r"""Return the number of samples."""
        return int((self.xmax - self.xmin - self.seq_len) // self.stride)

    def __iter__(self) -> Iterator:
        r"""Return an iterator over the samples."""
        if self.shuffle:
            perm = np.random.permutation(len(self))
        else:
            perm = np.arange(len(self))

        return iter(self.samples[perm])

    def __repr__(self) -> str:
        r"""Return a string representation of the object."""
        return f"{self.__class__.__name__}[{self.stride}, {self.seq_len}]"


class SlidingWindowSampler(BaseSampler, Generic[NumpyDTVar, NumpyTDVar]):
    r"""Sampler that generates sliding windows over an interval.

    The `SlidingWindowSampler` generates tuples.

    Inputs:
    - Ordered timestamps T
    - Starting time t_0
    - Final time t_f
    - stride ∆t (how much the sampler advances at each step)
        default, depending on data type of T:
           - integer: GCD(∆T)
           - float: max(⌊AVG(∆T)⌋, ε)
           - timestamp: resolution dependent.
    - horizons: TimeDelta or Tuple[TimeDelta]

    The sampler will return tuples of len(horizons)+1.
    """

    data: NDArray[NumpyDTVar]
    grid: NDArray[np.integer]
    horizons: NDArray[NumpyTDVar]
    mode: Literal["masks", "slices", "points"]
    shuffle: bool
    start_values: NDArray[NumpyDTVar]
    stride: NumpyTDVar
    tmax: NumpyDTVar
    tmin: NumpyDTVar
    offset: NumpyDTVar
    total_horizon: NumpyTDVar
    zero_td: NumpyTDVar

    def __init__(
        self,
        data_source: Sequence[NumpyDTVar],
        /,
        *,
        stride: Union[str, NumpyTDVar],
        horizons: Union[str, Sequence[str], NumpyTDVar, Sequence[NumpyTDVar]],
        tmin: Union[None, str, NumpyDTVar] = None,
        tmax: Union[None, str, NumpyDTVar] = None,
        mode: Literal["masks", "slices", "points"] = "masks",
        shuffle: bool = False,
    ):
        if isinstance(data_source, (Index, Series, DataFrame)):
            data_source = data_source.values
        super().__init__(data_source)

        self.shuffle = shuffle
        self.mode = mode

        if isinstance(horizons, str):
            self.horizons = np.array([Timedelta(horizons)])
        elif isinstance(horizons, Sequence):
            if isinstance(horizons[0], str):
                self.horizons = np.array([Timedelta(h) for h in horizons])
            else:
                self.horizons = np.array(horizons)
        else:
            self.horizons = np.array([horizons])

        self.total_horizon = self.horizons.sum()
        self.stride = Timedelta(stride) if isinstance(stride, str) else stride

        if tmin is None:
            if isinstance(self.data, (Series, DataFrame)):
                self.tmin = self.data.iloc[0]
            else:
                self.tmin = self.data[0]
        elif isinstance(tmin, str):
            self.tmin = Timestamp(tmin)
        else:
            self.tmin = tmin

        if tmax is None:
            if isinstance(self.data, (Series, DataFrame)):
                self.tmax = self.data.iloc[-1]
            else:
                self.tmax = self.data[-1]
        elif isinstance(tmax, str):
            self.tmax = Timestamp(tmax)
        else:
            self.tmax = tmax

        # this gives us the correct zero, depending on the dtype
        self.zero_td = cast(NumpyTDVar, self.tmin - self.tmin)

        assert self.stride > self.zero_td, "stride cannot be zero."

        cumulative_horizons: NDArray[NumpyTDVar] = np.concatenate(
            [np.array([self.zero_td]), self.horizons]
        )
        cumulative_horizons = np.cumsum(cumulative_horizons)

        self.start_values = cast(
            NDArray[NumpyDTVar],
            self.tmin + cumulative_horizons,  # type: ignore[call-overload, operator]
        )

        self.offset = cast(
            NumpyDTVar,
            self.tmin + self.total_horizon,  # type: ignore[call-overload, operator]
        )

        # precompute the possible slices
        self.grid = np.array(grid(self.tmin, self.tmax, self.stride))

    def __len__(self):
        r"""Return the number of samples."""
        return len(self.data)

    @staticmethod
    def __make__points__(bounds: NDArray[NumpyDTVar]) -> NDArray[NumpyDTVar]:
        """Return the points as-is."""
        return bounds

    @staticmethod
    def __make__slice__(window: NDArray[NumpyDTVar]) -> slice:
        """Return a tuple of slices."""
        return slice(window[0], window[-1])

    @staticmethod
    def __make__slices__(bounds: NDArray[NumpyDTVar]) -> tuple[slice, ...]:
        """Return a tuple of slices."""
        return tuple(
            slice(start, stop) for start, stop in sliding_window_view(bounds, 2)
        )

    def __make__mask__(self, window: NDArray[NumpyDTVar]) -> NDArray[np.bool_]:
        r"""Return a tuple of masks."""
        return (window[0] <= self.data) & (self.data < window[-1])

    def __make__masks__(
        self, bounds: NDArray[NumpyDTVar]
    ) -> tuple[NDArray[np.bool_], ...]:
        r"""Return a tuple of masks."""
        return tuple(
            (start <= self.data) & (self.data < stop)
            for start, stop in sliding_window_view(bounds, 2)
        )

    def __iter__(self) -> Iterator:
        r"""Iterate through.

        For each k, we return either:

        - mode=points: $(x₀ + k⋅∆t, x₁+k⋅∆t, …, xₘ+k⋅∆t)$
        - mode=slices: $(slice(x₀ + k⋅∆t, x₁+k⋅∆t), …, slice(xₘ₋₁+k⋅∆t, xₘ+k⋅∆t))$
        - mode=masks: $(mask₁, …, maskₘ)$
        """
        yield_fn: Callable[[NDArray[NumpyDTVar]], Any]
        if self.mode == "points":
            yield_fn = self.__make__points__
        else:
            single_horizon = len(self.horizons) == 1
            yield_fn = {
                ("masks", True): self.__make__mask__,
                ("masks", False): self.__make__masks__,
                ("slices", True): self.__make__slice__,
                ("slices", False): self.__make__slices__,
            }[(self.mode, single_horizon)]

        if self.shuffle:
            perm = np.random.permutation(len(self.grid))
            for k in self.grid[perm]:
                vals = self.start_values + k * self.stride
                yield yield_fn(vals)
            return

        # faster non-shuffle code path
        vals = self.start_values
        for _ in self.grid:
            vals += self.stride
            yield yield_fn(vals)
