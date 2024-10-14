r"""Samplers for randomly selecting data.

Note:
    For Mapping-style datasets, the sampler will return the keys of the mapping.
"""

__all__ = [
    # CONSTANTS
    "RNG",
    # ABCs & Protocols
    "BaseSampler",
    "Sampler",
    # Classes
    "RandomSampler",
    "DiscreteSlidingWindowSampler",
    "HierarchicalSampler",
    "RandomWindowSampler",
    "SlidingSampler",
    # Functions
    "compute_grid",
]

from abc import abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import KW_ONLY, dataclass, field
from enum import StrEnum
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    Optional,
    Protocol,
    cast,
    overload,
    runtime_checkable,
)

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from numpy.random import Generator
from numpy.typing import NDArray
from pandas import Index, Series
from typing_extensions import TypeVar

from tsdm.constants import EMPTY_MAP, RNG
from tsdm.data.datasets import (
    Dataset,
    MapDataset,
    PandasDataset,
    SequentialDataset,
    get_first_sample,
    get_index,
    get_last_sample,
)
from tsdm.types.protocols import Array
from tsdm.types.scalars import TimeDelta, TimeStamp
from tsdm.utils import timedelta, timestamp
from tsdm.utils.decorators import pprint_repr


# region helper functions --------------------------------------------------------------
def compute_grid[TD: TimeDelta](
    tmin: str | TimeStamp[TD],
    tmax: str | TimeStamp[TD],
    step: str | TD,
    /,
    *,
    offset: Optional[str | TimeStamp[TD]] = None,
) -> list[int]:
    r"""Compute $\{k∈ℤ ∣ tₘᵢₙ ≤ t₀+k⋅Δt ≤ tₘₐₓ\}$.

    That is, a list of all integers such that $t₀+k⋅Δ$ is in the interval $[tₘᵢₙ, tₘₐₓ]$.
    Special case: if $Δt=0$, returns $[0]$.

    .. math::
        if ∆t > 0
            tₘᵢₙ ≤ t₀+k⋅Δt ⟺ (tₘᵢₙ-t₀)/Δt ≤ k ⟺ k ≥ ⌈(tₘᵢₙ-t₀)/Δt⌉
            t₀+k⋅Δt ≤ tₘₐₓ ⟺ (tₘₐₓ-t₀)/Δt ≥ ⟺ k ≤ ⌊(tₘₐₓ-t₀)/Δt⌋
            ⟹ ⌈(tₘᵢₙ-t₀)/Δt⌉ ≤ k ≤ ⌊(tₘₐₓ-t₀)/Δt⌋
        if ∆t < 0
            tₘᵢₙ ≤ t₀+k⋅Δt ⟺ (tₘᵢₙ-t₀)/Δt ≥ k ⟺ k ≤ ⌊(tₘᵢₙ-t₀)/Δt⌋
            t₀+k⋅Δt ≤ tₘₐₓ ⟺ (tₘₐₓ-t₀)/Δt ≤ k ⟺ k ≥ ⌈(tₘₐₓ-t₀)/Δt⌉
            ⟹ ⌈(tₘₐₓ-t₀)/Δt⌉ ≤ k ≤ ⌊(tₘᵢₙ-t₀)/Δt⌋

    Note:
        This function is used to compute the strides for the sliding window sampler.
        given a window ∆s<tₘₐₓ-tₘᵢₙ, we want to find all k≥0 such that
        tₘᵢₙ ≤ [tₗ+k⋅Δt, tᵣ+k∆t] ≤ tₘₐₓ. This is equivalent to finding all k such that
    """
    # cast strings to timestamp/timedelta
    if offset is None:
        offset = tmin

    # I gave up trying to properly type hint this function.
    # Python just lacks some critical abilities like
    #  typeof https://github.com/python/typing/issues/769
    #  or generic bounds https://github.com/python/typing/issues/548
    t_min = cast(Any, timestamp(tmin) if isinstance(tmin, str) else tmin)
    t_max = cast(Any, timestamp(tmax) if isinstance(tmax, str) else tmax)
    t_0 = cast(Any, timestamp(offset) if isinstance(offset, str) else offset)
    delta = timedelta(step) if isinstance(step, str) else step

    # validate inputs
    if (t_min > t_0) or (t_0 > t_max):
        raise ValueError("tₘᵢₙ ≤ t₀ ≤ tₘₐₓ violated!")

    # NOTE: time-delta types should support div-mod / floordiv!
    #  Importantly, floordiv always rounds down, even for negative numbers.
    #  We use this formula for ceil-div: https://stackoverflow.com/a/17511341
    zero_td = t_min - t_min
    if delta > zero_td:
        kmin = -int((t_0 - t_min) // delta)  # ⌈a/b⌉ = -(-a//b)
        kmax = int((t_max - t_0) // delta)  # ⌊a/b⌋ = a//b
    elif delta < zero_td:
        kmin = -int((t_0 - t_max) // delta)  # ⌈a/b⌉ = -(-a//b)
        kmax = int((t_min - t_0) // delta)  # ⌊a/b⌋ = a//b
    else:
        raise ValueError(f"Δt={delta} is not allowed!")

    return list(range(kmin, kmax + 1))


# endregion helper functions -----------------------------------------------------------


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

    data: Dataset[T]

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


@pprint_repr
@dataclass(init=False)
class HierarchicalSampler[K, K2](BaseSampler[tuple[K, K2]]):
    r"""Draw samples from a hierarchical data source."""

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
        *,
        subsamplers: Mapping[K, Sampler[K2]] = EMPTY_MAP,
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

        self.index: Index = get_index(self.data)
        self.sizes: Series = Series({
            key: len(self.subsamplers[key]) for key in self.index
        })

        self.partition: Series = (
            Series(chain(*([key] * min(self.sizes) for key in self.index)))
            if self.early_stop
            else Series(chain(*([key] * self.sizes[key] for key in self.index)))
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
        r"""Return indices of the samples.

        When ``early_stop=True``, it will sample precisely ``min() * len(subsamplers)`` samples.
        When ``early_stop=False``, it will sample all samples.
        """
        index = self.rng.permutation(self.partition) if self.shuffle else self.partition

        activate_iterators = {
            key: iter(sampler) for key, sampler in self.subsamplers.items()
        }

        # This won't raise `StopIteration`, because the length is matched.
        for key in index:
            yield key, next(activate_iterators[key])


# mode types
type S = Literal["slices"]  # slice
type M = Literal["masks"]  # bool
type B = Literal["bounds"]  # tuple
type W = Literal["windows"]  # windows
type U = str  # unknown (not statically known)

# horizon types
type ONE = Literal["one"]
type MULTI = Literal["multi"]

# FIXME: python==3.13 use PEP695 with default values
DT = TypeVar("DT", bound=TimeStamp)
ModeVar = TypeVar("ModeVar", S, M, B, W, U, default=U)  # type: ignore[misc]
HorizonVar = TypeVar("HorizonVar", ONE, MULTI, default=ONE)  # type: ignore[misc]


# FIXME: Allow ±∞ as bounds for timedelta types? This would allow "growing" windows.
class SlidingSampler(BaseSampler, Generic[DT, ModeVar, HorizonVar]):
    r"""Sampler that generates a single sliding window over an interval.

    Note:
        This sampler is intended to be used with continuous time series data types,
        such as `float`, `numpy.timedelta64`, `datetime.timedelta`, `pandas.Timestamp`, etc.
        For discrete time series, particularly integer types, use `DiscreteSlidingWindowSampler`.
        Otherwise, off-by-one errors may occur, for example,
        for `horizons=(3, 1)` and `stride=2`, given the data `np.arange(10)`,
        this sampler will produce 3 windows.

    Args:
        data_source: A dataset that contains the ordered timestamps.
        stride: How much the window(s) advances at each step.
        horizons: The size of the window.
            Note: The size is specified as a timedelta, not as the number of data points.
            When sampling discrete data, this may lead to off-by-one errors.
            Consider using `DiscreteSlidingWindowSampler` instead.
            Multiple horizons can be given, in which case the sampler will return a list.
        mode: There are 4 modes, determining the output of the sampler (default: 'masks').
            - `tuple` / 'bounds': return the bounds of the window(s) as a tuple.
            - `slice` / 'slice': return the slice of the lower and upper bounds of the window.
            - `bool` / 'mask': return the boolean mask of the data points inside the window.
            - `list` / 'window': return the actual data points inside the window(s).
        shuffle: Whether to shuffle the indices (default: False).
        drop_last: Whether to drop the last incomplete window (default: False).
            If true, it is guaranteed that each window is completely contained in the data.
            If false, the last window may only partially overlap with the data.
            If multiple horizons are given, these rules apply to the last horizon.

    The window is considered to be closed on the left and open on the right. Moreover,
    the sampler can return multiple subsequent horizons if `horizons` is a sequence of
    `TimeDelta` objects. In this case, lists of the above objects are returned.
    """

    class MODE(StrEnum):
        r"""Valid modes for the sampler."""

        B = "bounds"
        M = "masks"
        S = "slices"
        W = "windows"

    type Mode = Literal["slices", "masks", "bounds", "windows"]
    r"""Type hint for the mode."""
    type Horizon = Literal["one", "multi"]
    r"""Type hint for the horizon."""

    data: NDArray[DT]  # type: ignore[type-var]

    horizons: TimeDelta | NDArray[TimeDelta]  # type: ignore[type-var]
    stride: TimeDelta
    mode: MODE
    multi_horizon: bool
    shuffle: bool
    drop_last: bool
    rng: Generator

    # dependent variables
    tmin: DT
    tmax: DT
    cumulative_horizons: NDArray[TimeDelta]  # type: ignore[type-var,unused-ignore]

    # region __new__ overloads ---------------------------------------------------------
    # TODO: simplify in the future when
    #   1. PEP 696 is implemented
    #   2. enums are considered subtypes of literals.
    if TYPE_CHECKING:

        @overload  # multi-horizon -----------------------------------------------------
        def __new__[TD: TimeDelta](  # pyright: ignore[reportOverlappingOverload]
            cls,
            data_source: SequentialDataset[DT],
            /,
            *,
            mode: Literal["slices", MODE.S],
            horizons: Array[str | TD],
            stride: str | TD,
            shuffle: bool = ...,
            drop_last: bool = ...,
            rng: Generator = ...,
        ) -> "SlidingSampler[DT, S, MULTI]": ...
        @overload
        def __new__[TD: TimeDelta](
            cls,
            data_source: SequentialDataset[DT],
            /,
            *,
            mode: Literal["bounds", MODE.B],
            horizons: Array[str | TD],
            stride: str | TD,
            shuffle: bool = ...,
            drop_last: bool = ...,
            rng: Generator = ...,
        ) -> "SlidingSampler[DT, B, MULTI]": ...
        @overload
        def __new__[TD: TimeDelta](
            cls,
            data_source: SequentialDataset[DT],
            /,
            *,
            mode: Literal["masks", MODE.M],
            horizons: Array[str | TD],
            stride: str | TD,
            shuffle: bool = ...,
            drop_last: bool = ...,
            rng: Generator = ...,
        ) -> "SlidingSampler[DT, M, MULTI]": ...
        @overload
        def __new__[TD: TimeDelta](
            cls,
            data_source: SequentialDataset[DT],
            /,
            *,
            mode: Literal["windows", MODE.W],
            horizons: Array[str | TD],
            stride: str | TD,
            shuffle: bool = ...,
            drop_last: bool = ...,
            rng: Generator = ...,
        ) -> "SlidingSampler[DT, W, MULTI]": ...
        @overload  # unknown mode
        def __new__[TD: TimeDelta](
            cls,
            data_source: SequentialDataset[DT],
            /,
            *,
            mode: MODE | Mode | str,
            horizons: Array[str | TD],
            stride: str | TD,
            shuffle: bool = ...,
            drop_last: bool = ...,
            rng: Generator = ...,
        ) -> "SlidingSampler[DT, U, MULTI]": ...
        @overload  # single horizon ----------------------------------------------------
        def __new__[TD: TimeDelta](  # pyright: ignore[reportOverlappingOverload]
            cls,
            data_source: SequentialDataset[DT],
            /,
            *,
            mode: Literal["slices", MODE.S],
            horizons: str | TD,
            stride: str | TD,
            shuffle: bool = ...,
            drop_last: bool = ...,
            rng: Generator = ...,
        ) -> "SlidingSampler[DT, S, ONE]": ...
        @overload
        def __new__[TD: TimeDelta](
            cls,
            data_source: SequentialDataset[DT],
            /,
            *,
            mode: Literal["bounds", MODE.B],
            horizons: str | TD,
            stride: str | TD,
            shuffle: bool = ...,
            drop_last: bool = ...,
            rng: Generator = ...,
        ) -> "SlidingSampler[DT, B, ONE]": ...
        @overload
        def __new__[TD: TimeDelta](
            cls,
            data_source: SequentialDataset[DT],
            /,
            *,
            mode: Literal["masks", MODE.M],
            horizons: str | TD,
            stride: str | TD,
            shuffle: bool = ...,
            drop_last: bool = ...,
            rng: Generator = ...,
        ) -> "SlidingSampler[DT, M, ONE]": ...
        @overload
        def __new__[TD: TimeDelta](
            cls,
            data_source: SequentialDataset[DT],
            /,
            *,
            mode: Literal["windows", MODE.W],
            horizons: str | TD,
            stride: str | TD,
            shuffle: bool = ...,
            drop_last: bool = ...,
            rng: Generator = ...,
        ) -> "SlidingSampler[DT, W, ONE]": ...
        @overload
        def __new__[TD: TimeDelta](
            cls,
            data_source: SequentialDataset[DT],
            /,
            *,
            mode: MODE | Mode | str,
            horizons: str | TD,
            stride: str | TD,
            shuffle: bool = ...,
            drop_last: bool = ...,
            rng: Generator = ...,
        ) -> "SlidingSampler[DT, U, ONE]": ...
    # endregion __new__ overloads --------------------------------------------------------

    def __init__[TD: TimeDelta](
        self,
        data_source: SequentialDataset[DT],
        /,
        *,
        mode: Mode | ModeVar | str,
        horizons: str | TD | Array[str | TD],
        stride: str | TD,
        drop_last: bool = False,
        shuffle: bool = False,
        rng: Generator = RNG,
    ) -> None:
        super().__init__(shuffle=shuffle, rng=rng)

        # region set basic attributes --------------------------------------------------
        self.tmin = get_first_sample(data_source)
        self.tmax = get_last_sample(data_source)
        zero_td = cast(Any, self.tmin - self.tmin)  # timedelta of the correct type
        dt_type: type[DT] = type(self.tmin)
        td_type: type[Any] = type(zero_td)
        self.data = np.array(data_source, dtype=dt_type)
        self.mode = self.MODE(mode)
        self.drop_last = drop_last
        self.stride = timedelta(stride) if isinstance(stride, str) else stride

        if self.stride <= zero_td:
            raise ValueError("stride must be positive.")
        # endregion set basic attributes -----------------------------------------------

        # region set horizon(s) --------------------------------------------------------
        match horizons:
            case str(string):
                self.multi_horizon = False
                self.horizons = np.array([timedelta(string)], dtype=td_type)
            case Iterable() as iterable:
                values = list(iterable)
                self.multi_horizon = True
                self.horizons = (
                    pd.to_timedelta(values).to_numpy(dtype=td_type)
                    if isinstance(values[0], str)
                    else np.array(values, dtype=td_type)
                )
            case TimeDelta() as td:
                self.multi_horizon = False
                self.horizons = np.array([td], dtype=td_type)
            case _:
                raise TypeError(f"Invalid type {type(horizons)} for {horizons=}")

        with_zero = np.concatenate([
            np.array([zero_td], dtype=td_type),
            self.horizons,
        ])
        self.cumulative_horizons = np.cumsum(with_zero, dtype=td_type)
        # endregion set horizon(s) -----------------------------------------------------

    @property
    def grid(self) -> NDArray[np.integer]:
        r"""Return the grid of indices."""
        # NOTE: we use a property so that if drop_last is changed, the grid is recomputed correctly...
        return np.array(
            compute_grid(
                self.tmin,
                self.tmax - self.cumulative_horizons[-1 if self.drop_last else -2],
                self.stride,
            )
        )

    def __len__(self) -> int:
        r"""Return the number of samples."""
        return len(self.grid)  # - self.drop_last

    # region __iter__ overloads --------------------------------------------------------
    # fmt: off
    @overload
    def __iter__(self: "SlidingSampler[DT, S, MULTI]", /) -> Iterator[list[slice]]: ...
    @overload
    def __iter__(self: "SlidingSampler[DT, B, MULTI]", /) -> Iterator[list[tuple[DT, DT]]]: ...
    @overload
    def __iter__(self: "SlidingSampler[DT, M, MULTI]", /) -> Iterator[list[NDArray[np.bool_]]]: ...
    @overload
    def __iter__(self: "SlidingSampler[DT, W, MULTI]", /) -> Iterator[list[NDArray[DT]]]: ...  # type: ignore[type-var,unused-ignore]
    @overload  # fallback mode=str
    def __iter__(self: "SlidingSampler[DT, U, MULTI]", /) -> Iterator[list]: ...
    @overload
    def __iter__(self: "SlidingSampler[DT, S, ONE]", /) -> Iterator[slice]: ...
    @overload
    def __iter__(self: "SlidingSampler[DT, B, ONE]", /) -> Iterator[tuple[DT, DT]]: ...
    @overload
    def __iter__(self: "SlidingSampler[DT, M, ONE]", /) -> Iterator[NDArray[np.bool_]]: ...
    @overload
    def __iter__(self: "SlidingSampler[DT, W, ONE]", /) -> Iterator[NDArray[DT]]: ...  # type: ignore[type-var,unused-ignore]
    @overload  # fallback mode=str
    def __iter__(self: "SlidingSampler[DT, U, ONE]", /) -> Iterator: ...
    # fmt: on
    # endregion __iter__ overloads -----------------------------------------------------
    def __iter__(self, /) -> Iterator:
        r"""Iterate through.

        For each k, we return either:

        - mode=points: $(x₀ + k⋅∆t, x₁+k⋅∆t, …, xₘ+k⋅∆t)$
        - mode=slices: $(slice(x₀ + k⋅∆t, x₁+k⋅∆t), …, slice(xₘ₋₁+k⋅∆t, xₘ+k⋅∆t))$
        - mode=masks: $(mask_1, …, mask_m)$
        """
        # unpack variables (avoids attribute lookup in loop)
        window = self.tmin + self.cumulative_horizons
        stride = self.stride
        grid = self.grid

        if self.shuffle:
            grid = grid[self.rng.permutation(len(grid))]

        # create generator expression for the windows
        iter_horizons = (window + k * stride for k in grid)

        match self.mode, self.multi_horizon:
            case "horizons", bool():
                yield from iter_horizons
            case "bounds", False:
                for horizons in iter_horizons:
                    yield horizons[0], horizons[-1]
            case "bounds", True:
                for horizons in iter_horizons:
                    yield [  # noqa: C416
                        (start, stop)
                        for start, stop in sliding_window_view(horizons, 2)
                    ]
            case "slices", False:
                for horizons in iter_horizons:
                    yield slice(horizons[0], horizons[-1])
            case "slices", True:
                for horizons in iter_horizons:
                    yield [
                        slice(start, stop)
                        for start, stop in sliding_window_view(horizons, 2)
                    ]
            case "masks", False:
                for horizons in iter_horizons:
                    yield (horizons[0] <= self.data) & (self.data < horizons[-1])
            case "masks", True:
                for horizons in iter_horizons:
                    yield [
                        (start <= self.data) & (self.data < stop)
                        for start, stop in sliding_window_view(horizons, 2)
                    ]
            case "windows", False:
                for horizons in iter_horizons:
                    yield self.data[
                        (horizons[0] <= self.data) & (self.data < horizons[-1])
                    ]
            case "windows", True:
                for horizons in iter_horizons:
                    yield [
                        self.data[(start <= self.data) & (self.data < stop)]
                        for start, stop in sliding_window_view(horizons, 2)
                    ]
            case _:
                raise TypeError(f"Invalid mode {self.mode=}")


class DiscreteSlidingWindowSampler(BaseSampler):
    r"""Sample a sliding window from the data source."""


class RandomWindowSampler(BaseSampler):
    r"""Sample a random window from the data source.

    Args:
        mode: There are 4 modes, determining the output of the sampler (default: 'masks').
            - `bounds`: return the bounds of the window(s) as a tuple.
            - `slices`: return the slice of the lower and upper bounds of the window.
            - `masks`: return the boolean mask of the data points inside the window.
            - `points`: return the actual data points inside the window(s).
        horizons: The size of the windows.
            - Timedelta ∆t: random sample window of size ∆t
            - list[Timedelta]: random sample subsequent windows of size ∆tₖ.
            - tuple[low, high]: random sample window of size ∆t ∈ [low, high]
            - list[tuple[low, high]]: random sample subsequent windows of size ∆tₖ ∈ [low, high]
            - callable: random sample window of size ∆t = f()
        base_freq: The minimal time resolution to consider. (default: ∆tₘᵢₙ)
            - will draw ∆t ∈ [low, high] such that ∆t is a multiple of base_freq.
            - will draw tₛₜₐᵣₜ ∈ [tₘᵢₙ, tₘₐₓ] such that tₛₜₐᵣₜ-tₘᵢₙ is a multiple of base_freq.
        max_samples: The maximum number of samples to draw (optional).
            - If set to None, the sampler will draw indefinitely.
            - If not given, the sampler will draw all possible samples (O(freq²)).
    """
