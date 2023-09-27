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
    "compute_grid",
]

import logging
import math
from abc import abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence, Sized
from datetime import timedelta as py_td
from itertools import chain, count
from typing import (
    Any,
    Final,
    Generic,
    Literal,
    Optional,
    Protocol,
    TypeVar,
    cast,
    overload,
    runtime_checkable,
)

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from pandas import DataFrame, Index, Series, Timedelta, Timestamp

from tsdm.types.protocols import Lookup
from tsdm.types.time import DTVar, NumpyDTVar, NumpyTDVar, TDVar
from tsdm.types.variables import any_co as T_co, any_var as T, key_var as K
from tsdm.utils.data.datasets import DatasetCollection
from tsdm.utils.strings import repr_mapping


def compute_grid(
    tmin: str | DTVar,
    tmax: str | DTVar,
    timedelta: str | TDVar,
    /,
    *,
    offset: Optional[str | DTVar] = None,
) -> Sequence[int]:
    r"""Compute $\{k∈ℤ ∣ tₘᵢₙ ≤ t₀+k⋅Δt ≤ tₘₐₓ\}$.

    That is, a list of all integers such that $t₀+k⋅Δ$ is in the interval $[tₘᵢₙ, tₘₐₓ]$.
    Special case: if $Δt=0$, returns $[0]$.
    """
    # cast strings to timestamp/timedelta
    tmin = cast(DTVar, Timestamp(tmin) if isinstance(tmin, str) else tmin)
    tmax = cast(DTVar, Timestamp(tmax) if isinstance(tmax, str) else tmax)
    td = Timedelta(timedelta) if isinstance(timedelta, str) else timedelta

    offset = cast(
        DTVar,
        (
            tmin
            if offset is None
            else Timestamp(offset) if isinstance(offset, str) else offset
        ),
    )

    # generates zero variable of correct type
    zero_dt = tmin - tmin

    if td == zero_dt:
        raise ValueError("Δt=0 is not allowed!")
    if tmin > offset or offset > tmax:
        raise ValueError("tₘᵢₙ ≤ t₀ ≤ tₘₐₓ violated!")

    kmax = math.floor((tmax - offset) / td)
    kmin = math.ceil((tmin - offset) / td)

    return cast(Sequence[int], np.arange(kmin, kmax + 1))


@runtime_checkable
class Sampler(Protocol[T_co]):
    r"""Protocol for `Sampler`."""

    def __iter__(self) -> Iterator[T_co]: ...

    def __len__(self) -> int: ...


class BaseSamplerMetaClass(type(Protocol)):  # type: ignore[misc]
    r"""Metaclass for BaseDataset."""

    def __init__(
        cls, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwds: Any
    ) -> None:
        """When a new class/subclass is created, this method is called."""
        super().__init__(name, bases, namespace, **kwds)

        if "LOGGER" not in namespace:
            cls.LOGGER = logging.getLogger(f"{cls.__module__}.{cls.__name__}")


class BaseSampler(Sampler[T_co], metaclass=BaseSamplerMetaClass):
    r"""Abstract Base Class for all Samplers."""

    LOGGER: logging.Logger
    r"""Logger for the sampler."""
    data: Sized
    r"""Copy of the original Data source."""

    def __init__(self, data_source: Sized, /) -> None:
        r"""Initialize the sampler."""
        self.data = data_source

    @abstractmethod
    def __len__(self) -> int:
        r"""Return the length of the sampler."""

    @abstractmethod
    def __iter__(self) -> Iterator[T_co]:
        r"""Iterate over random indices."""


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
    idx: NDArray
    rng: np.random.Generator

    def __init__(
        self,
        data_source: Sequence[T_co],
        /,
        *,
        slice_sampler: Optional[int | Callable[[], int]] = None,
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
                return lambda: slice_sampler
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


class CollectionSampler(BaseSampler[tuple[K, T_co]]):
    r"""Samples a single random dataset from a collection of dataset.

    Optionally, we can delegate a subsampler to then sample from the randomly drawn dataset.
    """

    idx: Index
    r"""The shared index."""
    subsamplers: Mapping[K, BaseSampler[T_co]]
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
        subsamplers: Mapping[K, BaseSampler[T_co]],
        /,
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


class HierarchicalSampler(BaseSampler[tuple[K, T_co]]):
    r"""Samples a single random dataset from a collection of dataset.

    Optionally, we can delegate a subsampler to then sample from the randomly drawn dataset.
    """

    idx: Index
    r"""The shared index."""
    subsamplers: Mapping[K, Sampler[T_co]]
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
        data_source: Mapping[K, T],
        subsamplers: Mapping[K, Sampler[T_co]],
        /,
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

    def __len__(self) -> int:
        r"""Return the maximum allowed index."""
        if self.early_stop:
            return min(self.sizes) * len(self.subsamplers)
        return sum(self.sizes)

    def __iter__(self) -> Iterator[tuple[K, T_co]]:
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
            except StopIteration as exc:
                raise RuntimeError(
                    f"Iterator of {key=} exhausted prematurely."
                ) from exc
            yield key, value

    def __getitem__(self, key: K) -> Sampler[T_co]:
        r"""Return the subsampler for the given key."""
        return self.subsamplers[key]

    def __repr__(self) -> str:
        r"""Pretty print."""
        return repr_mapping(self.subsamplers, title=self.__class__.__name__)


class IntervalSampler(BaseSampler[slice], Generic[TDVar]):
    r"""Return all intervals `[a, b]`.

    The intervals must satisfy:

    - `a = t₀ + i⋅sₖ`
    - `b = t₀ + i⋅sₖ + Δtₖ`
    - `i, k ∈ ℤ`
    - `a ≥ t_min`
    - `b ≤ t_max`
    - `sₖ` is the stride corresponding to intervals of size `Δtₖ`.
    """

    offset: TDVar
    deltax: TDVar | Lookup[int, TDVar] | Callable[[int], TDVar]
    stride: TDVar | Lookup[int, TDVar] | Callable[[int], TDVar]
    shuffle: bool
    intervals: DataFrame

    @staticmethod
    def _get_value(
        obj: TDVar | Lookup[int, TDVar] | Callable[[int], TDVar], k: int, /
    ) -> TDVar:
        if callable(obj):
            return obj(k)
        if isinstance(obj, Lookup):  # Mapping/Sequence
            return obj[k]
        # Fallback: multiple!
        return obj

    def __init__(
        self,
        *,
        xmin: TDVar,
        xmax: TDVar,
        deltax: TDVar | Lookup[int, TDVar] | Callable[[int], TDVar],
        stride: Optional[TDVar | Lookup[int, TDVar] | Callable[[int], TDVar]] = None,
        levels: Optional[Sequence[int]] = None,
        offset: Optional[TDVar] = None,
        shuffle: bool = True,
    ) -> None:
        super().__init__([])
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
        intervals: list[tuple[TDVar, TDVar, TDVar, TDVar]] = []

        # for each level, get all intervals
        for k in levels:
            dt = self._get_value(deltax, k)
            st = self._get_value(stride, k)
            x0 = self._get_value(offset, k)

            # get valid interval bounds, probably there is an easier way to do it...
            stride_left: Sequence[int] = compute_grid(xmin, xmax, st, offset=x0)
            stride_right: Sequence[int] = compute_grid(xmin, xmax, st, offset=x0 + dt)
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

    def __getitem__(self, key: int) -> slice:
        r"""Return a slice from the sampler."""
        return self.intervals[key]


class SequenceSampler(BaseSampler, Generic[DTVar, TDVar]):
    r"""Samples sequences of length seq_len."""

    data_source: NDArray[DTVar]  # type: ignore[type-var]
    k_max: int
    return_mask: bool
    seq_len: TDVar
    shuffle: bool
    stride: TDVar
    xmax: DTVar
    xmin: DTVar
    # total_delta: TDVar

    def __init__(
        self,
        data_source: NDArray[DTVar],
        /,
        *,
        return_mask: bool = False,
        seq_len: str | TDVar,
        shuffle: bool = False,
        stride: str | TDVar,
        tmax: Optional[DTVar] = None,
        tmin: Optional[DTVar] = None,
    ) -> None:
        super().__init__(data_source)

        self.xmin = (
            self.data_source[0]
            if tmin is None
            else (Timestamp(tmin) if isinstance(tmin, str) else tmin)
        )
        self.xmax = (
            self.data_source[-1]
            if tmax is None
            else (Timestamp(tmax) if isinstance(tmax, str) else tmax)
        )

        total_delta = cast(TDVar, self.xmax - self.xmin)  # type: ignore[redundant-cast]
        self.stride = cast(
            TDVar, Timedelta(stride) if isinstance(stride, str) else stride
        )
        self.seq_len = cast(
            TDVar, Timedelta(seq_len) if isinstance(seq_len, str) else seq_len
        )

        # k_max = max {k∈ℕ ∣ x_min + seq_len + k⋅stride ≤ x_max}
        self.k_max = int((total_delta - self.seq_len) // self.stride)
        self.return_mask = return_mask
        self.shuffle = shuffle

        self.samples = np.array(
            [
                (
                    (x <= self.data_source) & (self.data_source < y)  # type: ignore[operator]
                    if self.return_mask
                    else [x, y]
                )
                for x, y in self._iter_tuples()
            ]
        )

    def _iter_tuples(self) -> Iterator[tuple[DTVar, DTVar]]:
        x = self.xmin
        y = cast(DTVar, x + self.seq_len)  # type: ignore[operator, call-overload, redundant-cast]
        x, y = min(x, y), max(x, y)  # allows nice handling of negative seq_len
        yield x, y

        for _ in range(len(self)):
            x = x + self.stride  # type: ignore[assignment, operator, call-overload]
            y = y + self.stride  # type: ignore[assignment, operator, call-overload]
            yield x, y

    def __len__(self) -> int:
        r"""Return the number of samples."""
        return self.k_max

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


MODE = TypeVar("MODE", Literal["masks"], Literal["slices"], Literal["points"])
Modes = TypeVar("Modes", bound=Literal["masks", "slices", "points"])


class SlidingWindowSampler(BaseSampler, Generic[MODE, NumpyDTVar, NumpyTDVar]):
    r"""Sampler that generates sliding windows over an interval.

    The `SlidingWindowSampler` generates tuples.

    Inputs:

    - Ordered timestamps T
    - Starting time t_0
    - Final time t_f
    - stride ∆t (how much the sampler advances at each step) default, depending on data type of T:
        - integer: GCD(∆T)
        - float: max(⌊AVG(∆T)⌋, ε)
        - timestamp: resolution dependent.
    - horizons: TimeDelta or Tuple[TimeDelta]

    The sampler will return tuples of ``len(horizons)+1``.
    """

    data: NDArray[NumpyDTVar]

    mode: MODE
    shuffle: Final[bool]
    stride: NumpyTDVar
    tmax: NumpyDTVar
    tmin: NumpyDTVar
    grid: Final[NDArray[np.integer]]

    horizons: NumpyTDVar | NDArray[NumpyTDVar]
    start_values: NDArray[NumpyDTVar]
    offset: NumpyDTVar
    total_horizon: NumpyTDVar
    zero_td: NumpyTDVar
    multi_horizon: bool
    cumulative_horizons: NDArray[NumpyTDVar]

    def __init__(
        self,
        data_source: Sequence[NumpyDTVar],
        /,
        *,
        horizons: str | Sequence[str] | NumpyTDVar | Sequence[NumpyTDVar],
        mode: MODE = "masks",  # type: ignore[assignment]
        shuffle: bool = False,
        stride: str | NumpyTDVar,
        tmax: Optional[str | NumpyDTVar] = None,
        tmin: Optional[str | NumpyDTVar] = None,
    ) -> None:
        super().__init__(data_source)

        # coerce non-numpy types to numpy.
        horizons = Timedelta(horizons) if isinstance(horizons, str) else horizons
        stride = Timedelta(stride) if isinstance(stride, str) else stride
        tmin = Timestamp(tmin) if isinstance(tmin, str) else tmin
        tmax = Timestamp(tmax) if isinstance(tmax, str) else tmax

        self.shuffle = shuffle
        self.mode = mode
        self.stride = stride

        if tmin is None:
            if isinstance(self.data, Series | DataFrame):
                self.tmin = self.data.iloc[0]
            else:
                self.tmin = self.data[0]
        else:
            self.tmin = tmin

        if tmax is None:
            if isinstance(self.data, Series | DataFrame):
                self.tmax = self.data.iloc[-1]
            else:
                self.tmax = self.data[-1]
        else:
            self.tmax = tmax

        # this gives us the correct zero, depending on the dtype
        self.zero_td = self.tmin - self.tmin  # type: ignore[assignment]
        assert self.stride > self.zero_td, "stride cannot be zero."

        if isinstance(horizons, Sequence):
            self.multi_horizon = True
            if isinstance(horizons[0], str | Timedelta | py_td):
                self.horizons = pd.to_timedelta(horizons)
                concat_horizons = self.horizons.insert(0, self.zero_td)  # type: ignore[union-attr]
            else:
                self.horizons = np.array(horizons)
                concat_horizons = np.concatenate(([self.zero_td], self.horizons))  # type: ignore[arg-type]

            self.cumulative_horizons = np.cumsum(concat_horizons)
            self.total_horizon = self.cumulative_horizons[-1]
        else:
            self.multi_horizon = False
            self.horizons = horizons
            self.total_horizon = self.horizons
            self.cumulative_horizons = np.cumsum([self.zero_td, self.horizons])

        self.start_values = self.tmin + self.cumulative_horizons  # type: ignore[assignment, call-overload, operator]

        self.offset = self.tmin + self.total_horizon  # type: ignore[assignment, call-overload, operator]

        # precompute the possible slices
        grid = compute_grid(self.tmin, self.tmax, self.stride, offset=self.offset)
        self.grid = grid[grid >= 0]  # type: ignore[assignment, operator]

    def __len__(self) -> int:
        r"""Return the number of samples."""
        return len(self.grid)

    @staticmethod
    def __make__points__(bounds: NDArray[NumpyDTVar]) -> NDArray[NumpyDTVar]:
        r"""Return the points as-is."""
        return bounds

    @staticmethod
    def __make__slice__(window: NDArray[NumpyDTVar]) -> slice:
        r"""Return a tuple of slices."""
        return slice(window[0], window[-1])

    @staticmethod
    def __make__slices__(bounds: NDArray[NumpyDTVar]) -> tuple[slice, ...]:
        r"""Return a tuple of slices."""
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

    @overload
    def __iter__(
        self: "SlidingWindowSampler[Literal['slices'],NumpyDTVar, NumpyTDVar]",
    ) -> Iterator[slice] | Iterator[tuple[slice, ...]]: ...
    @overload
    def __iter__(
        self: "SlidingWindowSampler[Literal['masks'],NumpyDTVar, NumpyTDVar]",
    ) -> Iterator[NDArray[np.bool_]] | Iterator[tuple[NDArray[np.bool_], ...]]: ...
    @overload
    def __iter__(
        self: "SlidingWindowSampler[Literal['points'],NumpyDTVar, NumpyTDVar]",
    ) -> Iterator[NDArray[NumpyDTVar]]: ...
    def __iter__(self):
        r"""Iterate through.

        For each k, we return either:

        - mode=points: $(x₀ + k⋅∆t, x₁+k⋅∆t, …, xₘ+k⋅∆t)$
        - mode=slices: $(slice(x₀ + k⋅∆t, x₁+k⋅∆t), …, slice(xₘ₋₁+k⋅∆t, xₘ+k⋅∆t))$
        - mode=masks: $(mask_1, …, mask_m)$
        """
        funcs = {
            ("points", True): self.__make__points__,
            ("points", False): self.__make__points__,
            ("masks", False): self.__make__mask__,
            ("masks", True): self.__make__masks__,
            ("slices", False): self.__make__slice__,
            ("slices", True): self.__make__slices__,
        }
        yield_fn = funcs[self.mode, self.multi_horizon]

        if self.shuffle:
            perm = np.random.permutation(len(self.grid))
            grid = self.grid[perm]
        else:
            grid = self.grid

        for k in grid:
            vals = self.start_values + k * self.stride
            yield yield_fn(vals)
