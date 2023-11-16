"""Deprecated samplers."""

__all__ = ["SliceSampler", "CollectionSampler"]

from collections.abc import Callable, Iterator, Mapping, Sequence
from itertools import chain, count

import numpy as np
from numpy._typing import NDArray
from pandas import DataFrame, Index, Series, Timedelta, Timestamp
from typing_extensions import Any, Generic, Optional, cast, deprecated

from tsdm.random.samplers import BaseSampler, compute_grid
from tsdm.types.protocols import Lookup
from tsdm.types.time import DTVar, TDVar
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


@deprecated("Use SlidingWindowSampler instead.")
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
    intervals: DataFrame
    shuffle: bool = False

    @staticmethod
    def _get_value(
        obj: TDVar | Lookup[int, TDVar] | Callable[[int], TDVar], k: int, /
    ) -> TDVar:
        match obj:
            case Callable() as func:  # type: ignore[misc]
                return func(k)
            case Lookup() as mapping:
                return mapping[k]
            case _:
                return obj  # type: ignore[return-value]

    def __init__(
        self,
        *,
        xmin: TDVar,
        xmax: TDVar,
        deltax: TDVar | Lookup[int, TDVar] | Callable[[int], TDVar],
        stride: Optional[TDVar | Lookup[int, TDVar] | Callable[[int], TDVar]] = None,
        levels: Optional[Sequence[int]] = None,
        offset: Optional[TDVar] = None,
        shuffle: bool = False,
    ) -> None:
        super().__init__(shuffle=shuffle)
        # set stride and offset
        zero = 0 * (xmax - xmin)
        stride = zero if stride is None else stride
        offset = xmin if offset is None else offset
        delta_max = max(offset - xmin, xmax - offset)

        # validate bounds
        assert xmin <= offset <= xmax, "Assumption: xmin≤xoffset≤xmax violated!"

        # determine levels

        match levels, deltax:
            case None, Mapping() as mapping:
                levels = [k for k, v in mapping.items() if v <= delta_max]
            case None, Sequence() as sequence:
                levels = [k for k, v in enumerate(sequence) if v <= delta_max]
            case None, Callable() as func:  # type: ignore[misc]
                levels = []
                for k in count():
                    dt = self._get_value(func, k)
                    if dt == zero:
                        continue
                    if dt > delta_max:
                        break
                    levels.append(k)
            case None, _:
                levels = [0]
            case Sequence() as seq, _:
                levels = [k for k in seq if self._get_value(deltax, k) <= delta_max]
            case _:
                raise TypeError("levels not compatible.")

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

            intervals.extend([
                (x0 + i * st, x0 + i * st + dt, dt, st) for i in valid_strides
            ])

        # set variables
        self.offset = cast(TDVar, offset)  # type: ignore[redundant-cast]
        self.deltax = deltax
        self.stride = stride
        self.intervals = DataFrame(
            intervals,
            columns=["lower_bound", "upper_bound", "delta", "stride"],
        )

    def __iter__(self) -> Iterator[slice]:
        r"""Return an iterator over the intervals."""
        # assign to avoid attribute lookup in loop
        lower_bounds = self.intervals["lower_bound"]
        upper_bounds = self.intervals["upper_bound"]

        permutation = (
            np.arange(len(self))
            if not self.shuffle
            else np.random.permutation(len(self))
        )

        for k in permutation:
            yield slice(lower_bounds[k], upper_bounds[k])

    def __len__(self) -> int:
        r"""Length of the sampler."""
        return len(self.intervals)

    def __getattr__(self, key: str) -> Any:
        r"""Forward all other attributes to the interval frame."""
        return getattr(self.intervals, key)

    def __getitem__(self, key: int) -> slice:
        r"""Return a slice from the sampler."""
        return self.intervals[key]


@deprecated("Use SlidingWindowSampler instead.")
class SequenceSampler(BaseSampler, Generic[DTVar, TDVar]):
    r"""Samples sequences of fixed length."""

    data: NDArray[DTVar]  # type: ignore[type-var]
    seq_len: TDVar
    """The length of the sequences."""
    stride: TDVar
    """The stride at which to sample."""
    xmax: DTVar
    """The maximum value at which to stop sampling."""
    xmin: DTVar
    """The minimum value at which to start sampling."""
    # total_delta: TDVar
    return_mask: bool = False
    """Whether to return masks instead of indices."""
    shuffle: bool = False
    """Whether to shuffle the data."""

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
        super().__init__(shuffle=shuffle)
        self.data = np.asarray(data_source)

        match tmin:
            case None:
                self.xmin = self.data[0]
            case str() as time_str:
                self.xmin = Timestamp(time_str)
            case _:
                self.xmin = tmin

        match tmax:
            case None:
                self.xmax = self.data[-1]
            case str() as time_str:
                self.xmax = Timestamp(time_str)
            case _:
                self.xmax = tmax

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

        self.samples = np.array([
            (
                (x <= self.data) & (self.data < y)  # type: ignore[operator]
                if self.return_mask
                else [x, y]
            )
            for x, y in self._iter_tuples()
        ])

    def _iter_tuples(self) -> Iterator[tuple[DTVar, DTVar]]:
        x = self.xmin
        y = cast(DTVar, x + self.seq_len)  # type: ignore[operator, call-overload, redundant-cast]
        # allows nice handling of negative seq_len
        x, y = min(x, y), max(x, y)  # pyright: ignore[reportGeneralTypeIssues]
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
        permutation = (
            np.arange(len(self))
            if not self.shuffle
            else np.random.permutation(len(self))
        )
        return iter(self.samples[permutation])

    def __repr__(self) -> str:
        r"""Return a string representation of the object."""
        return f"{self.__class__.__name__}[{self.stride}, {self.seq_len}]"


#
# class MultipleSlidingWindowSampler(BaseSampler, Generic[MODE, NumpyDTVar]):
#     r"""Sampler that generates sliding windows over an interval.
#
#     The `SlidingWindowSampler` generates tuples.
#
#     Inputs:
#     - Ordered timestamps $T$
#     - Starting time $t_0$
#     - Final time $t_f$
#     - stride ∆t (how much the sampler advances at each step) default,
#       depending on the data type of $T$:
#         - integer: $GCD(∆T)$
#         - float: $\max(⌊AVG(∆T)⌋, ε)$
#         - timestamp: resolution dependent.
#     - horizons: `TimeDelta` or `tuple[TimeDelta, ...]`
#
#     The sampler will return tuples of `len(horizons)+1`.
#     """
#
#     data: NDArray[NumpyDTVar]
#
#     horizons: Sequence[T]
#     stride: NumpyTDVar
#     tmin: NumpyDTVar
#     tmax: NumpyDTVar
#
#     mode: MODE
#     shuffle: bool = False
#
#     grid: Final[NDArray[np.integer]]
#     start_values: NDArray[NumpyDTVar]
#     offset: NumpyDTVar
#     zero_td: T
#     cumulative_horizons: NDArray[T]
#     total_horizon: T
#
#     def __init__(
#         self,
#         data_source: Sequence[NumpyDTVar],
#         /,
#         *,
#         horizons: Sequence[str | T],
#         stride: str | T,
#         tmin: Optional[str | NumpyDTVar] = None,
#         tmax: Optional[str | NumpyDTVar] = None,
#         mode: MODE = "masks",  # type: ignore[assignment]
#         shuffle: bool = False,
#     ) -> None:
#         super().__init__(shuffle=shuffle)
#         self.data = np.asarray(data_source)
#         self.mode = mode
#         self.stride = Timedelta(stride) if isinstance(stride, str) else stride
#
#         match tmin:
#             case None:
#                 self.tmin = (
#                     self.data.iloc[0] if isinstance(self.data, Series) else self.data[0]
#                 )
#             case str() as time_str:
#                 self.tmin = Timestamp(time_str)
#             case _:
#                 self.tmin = tmin
#
#         match tmax:
#             case None:
#                 self.tmax = (
#                     self.data.iloc[-1]
#                     if isinstance(self.data, Series)
#                     else self.data[-1]
#                 )
#             case str() as time_str:
#                 self.tmax = Timestamp(time_str)
#             case _:
#                 self.tmax = tmax
#
#         # this gives us the correct zero, depending on the dtype
#         self.zero_td = cast(NumpyTDVar, self.tmin - self.tmin)
#         assert self.stride > self.zero_td, "stride cannot be zero."
#
#         # convert horizons to timedelta
#         if isinstance(horizons[0], str | Timedelta | py_td):
#             self.horizons = pd.to_timedelta(horizons)
#             concat_horizons = self.horizons.insert(0, self.zero_td)
#         else:
#             self.horizons = np.array(horizons)
#             concat_horizons = np.concatenate(([self.zero_td], self.horizons))
#
#         self.cumulative_horizons = np.cumsum(concat_horizons)
#         self.total_horizon = self.cumulative_horizons[-1]
#
#         self.start_values = self.tmin + self.cumulative_horizons
#
#         self.offset = self.tmin + self.total_horizon
#
#         # precompute the possible slices
#         grid = compute_grid(self.tmin, self.tmax, self.stride, offset=self.offset)
#         self.grid = grid[grid >= 0]  # type: ignore[assignment, operator]
#
#     def __len__(self) -> int:
#         r"""Return the number of samples."""
#         return len(self.grid)
#
#     @staticmethod
#     def __make__points__(bounds: NDArray[NumpyDTVar]) -> NDArray[NumpyDTVar]:
#         r"""Return the points as-is."""
#         return bounds
#
#     @staticmethod
#     def __make__slices__(bounds: NDArray[NumpyDTVar]) -> tuple[slice, ...]:
#         r"""Return a tuple of slices."""
#         return tuple(
#             slice(start, stop) for start, stop in sliding_window_view(bounds, 2)
#         )
#
#     def __make__masks__(
#         self, bounds: NDArray[NumpyDTVar]
#     ) -> tuple[NDArray[np.bool_], ...]:
#         r"""Return a tuple of masks."""
#         return tuple(
#             (start <= self.data) & (self.data < stop)
#             for start, stop in sliding_window_view(bounds, 2)
#         )
#
#     @property
#     def make_key(self) -> Callable[[NDArray], Any]:
#         r"""Return the correct yield function."""
#         match self.mode:
#             case "points", _:
#                 return self.__make__points__
#             case "masks", True:
#                 return self.__make__masks__
#             case "slices", True:
#                 return self.__make__slices__
#             case _:
#                 raise ValueError(f"Invalid mode {self.mode=}")
#
#     @overload
#     def __iter__(
#         self: "SlidingWindowSampler[Literal['slices'], NumpyDTVar]",
#     ) -> Iterator[tuple[slice, ...]]: ...
#     @overload
#     def __iter__(
#         self: "SlidingWindowSampler[Literal['masks'], NumpyDTVar]",
#     ) -> Iterator[tuple[NDArray[np.bool_], ...]]: ...
#     @overload
#     def __iter__(
#         self: "SlidingWindowSampler[Literal['points'], NumpyDTVar]",
#     ) -> Iterator[tuple[NDArray[NumpyDTVar], ...]]: ...
#     def __iter__(self):  # pyright: ignore[reportGeneralTypeIssues]
#         r"""Iterate through.
#
#         For each k, we return either:
#
#         - mode=points: $(x₀ + k⋅∆t, x₁+k⋅∆t, …, xₘ+k⋅∆t)$
#         - mode=slices: $(slice(x₀ + k⋅∆t, x₁+k⋅∆t), …, slice(xₘ₋₁+k⋅∆t, xₘ+k⋅∆t))$
#         - mode=masks: $(mask_1, …, mask_m)$
#         """
#         if self.shuffle:
#             perm = np.random.permutation(len(self.grid))
#             grid = self.grid[perm]
#         else:
#             grid = self.grid
#
#         # unpack variables (avoid repeated lookups)
#         t0 = self.start_values
#         stride = self.stride
#         make_key = self.make_key
#
#         for k in grid:
#             yield make_key(t0 + k * stride)  # shifted window
