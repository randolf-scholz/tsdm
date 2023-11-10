r"""Samplers for randomly selecting data.


NOTE:
    For Mapping-style datasets, the sampler will return the keys of the mapping.
"""

__all__ = [
    # ABCs
    "BaseSampler",
    # Classes
    "RandomSampler",
    # "TimeSliceSampler",
    "HierarchicalSampler",
    "SlidingWindowSampler",
    # Functions
    "compute_grid",
]

import logging
import math
from abc import abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from datetime import timedelta as py_td
from itertools import chain
from typing import (
    Any,
    ClassVar,
    Final,
    Generic,
    Literal,
    Optional,
    Protocol,
    Self,
    TypeAlias,
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

from tsdm.types.protocols import DateTime as DTLike, TimeDelta as TDLike
from tsdm.types.time import DTVar, NumpyDTVar, NumpyTDVar, TDVar
from tsdm.types.variables import (
    any_co as T_co,
    any_other_var as T2,
    any_var as T,
    key_other_var as K2,
    key_var as K,
)
from tsdm.utils.data.datasets import (
    Dataset,
    IterableDataset,
    MapDataset,
    SequentialDataset,
)
from tsdm.utils.strings import pprint_repr


@runtime_checkable
class Sampler(Protocol[T_co]):
    r"""Protocol for `Sampler` classes."""

    @property
    @abstractmethod
    def shuffle(self) -> bool:
        """Whether to shuffle the indices."""
        ...

    def __iter__(self) -> Iterator[T_co]:
        """Return an iterator over the indices of the data source."""
        ...

    def __len__(self) -> int:
        """The number of indices that can be drawn by __iter__."""
        ...


class BaseSamplerMetaClass(type(Protocol)):  # type: ignore[misc]
    r"""Metaclass for BaseDataset."""

    # def __new__(
    #     cls,
    #     name: str,
    #     bases: tuple[type, ...],
    #     namespace: dict[str, Any],
    #     /,
    #     **kwds: Any,
    # ) -> Self:
    #     # NOTE: https://stackoverflow.com/a/73677355/9318372
    #     if "__slots__" not in namespace:
    #         namespace["__slots__"] = tuple()
    #     return super(BaseSamplerMetaClass, cls).__new__(
    #         cls, name, bases, namespace, **kwds
    #     )

    def __init__(
        cls, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwds: Any
    ) -> None:
        """When a new class/subclass is created, this method is called."""
        super().__init__(name, bases, namespace, **kwds)

        if "LOGGER" not in namespace:
            cls.LOGGER = logging.getLogger(f"{cls.__module__}.{cls.__name__}")


@dataclass
class BaseSampler(Sampler[T_co], metaclass=BaseSamplerMetaClass):
    r"""Abstract Base Class for all Samplers."""

    LOGGER: ClassVar[logging.Logger]
    r"""Logger for the sampler."""

    _: KW_ONLY

    shuffle: bool = False
    r"""Whether to randomize sampling."""

    # def __init__(self, *, shuffle: bool) -> None:
    #     r"""Initialize the sampler."""
    #     self.shuffle = shuffle

    @abstractmethod
    def __len__(self) -> int:
        r"""Return the length of the sampler."""
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[T_co]:
        r"""Return an iterator over the indices of the data source."""
        ...


@pprint_repr
@dataclass(init=False, slots=True)
class RandomSampler(BaseSampler[T_co]):
    """Sample randomly from the data source."""

    data: Dataset[T_co]  # map style or iterable style
    shuffle: bool = False

    index: Index = field(init=False)
    size: int = field(init=False)

    def __init__(self, data_source: Dataset[T_co], /, *, shuffle: bool = False) -> None:
        """Initialize the sampler."""
        super(RandomSampler, self).__init__(shuffle=shuffle)
        self.data = data_source
        self.index = get_index(self.data)
        self.size = len(self.index)

    def __iter__(self) -> Iterator[T_co]:
        index = (
            self.index
            if not self.shuffle
            else self.index[np.random.permutation(self.size)]
        )
        data = self.data  # avoids attribute lookup in the loop
        for key in index:
            yield data[key]

    def __len__(self) -> int:
        return self.size


@pprint_repr
@dataclass
class HierarchicalSampler(BaseSampler[tuple[K, K2]]):
    r"""Draw samples from a hierarchical data source."""

    data_source: MapDataset[K, Dataset[K2]]
    r"""The shared index."""
    subsamplers: Mapping[K, Sampler[K2]] = NotImplemented
    r"""The subsamplers to sample from the collection."""

    _: KW_ONLY

    early_stop: bool = False
    r"""Whether to stop sampling when the index is exhausted."""
    shuffle: bool = False
    r"""Whether to sample in random order."""

    def __post_init__(self) -> None:
        self.subsamplers = (
            {
                key: RandomSampler(self.data_source[key], shuffle=self.shuffle)
                for key in self.data_source.keys()
            }
            if self.subsamplers is NotImplemented
            else dict(self.subsamplers)
        )

        self.index: Index = get_index(self.data_source)

        self.sizes: Series = Series(
            {key: len(self.subsamplers[key]) for key in self.index}
        )

        self.partition: Series = (
            Series(chain(*([key] * min(self.sizes) for key in self.index)))
            if self.early_stop
            else Series(chain(*([key] * self.sizes[key] for key in self.index)))
        )

        # if self.early_stop:  # duplicate keys to match the minimum subsampler size
        #     self.partition =
        # else:  # duplicate keys to match each sub-sampler's size
        #     self.partition = Series(
        #         chain(*([key] * self.sizes[key] for key in self.index))
        #     )

    def __len__(self) -> int:
        r"""Return the maximum allowed index."""
        if self.early_stop:
            return min(self.sizes) * len(self.subsamplers)
        return sum(self.sizes)

    def __getitem__(self, key: K) -> Sampler[K2]:
        r"""Return the subsampler for the given key."""
        return self.subsamplers[key]

    def __iter__(self) -> Iterator[tuple[K, K2]]:
        r"""Return indices of the samples.

        When ``early_stop=True``, it will sample precisely ``min() * len(subsamplers)`` samples.
        When ``early_stop=False``, it will sample all samples.
        """
        permutation = (
            self.partition
            if not self.shuffle
            else np.random.permutation(self.partition)
        )

        activate_iterators = {
            key: iter(sampler) for key, sampler in self.subsamplers.items()
        }

        # This won't raise StopIteration, because the length is matched.
        for key in permutation:
            yield key, next(activate_iterators[key])

            # for-break faster than try-next-except
            # for value in activate_iterators[key]:
            #     yield key, value
            #     break
            # else:  # activate_iterators[key] is exhausted
            #     raise RuntimeError(f"Sampler of {key=} exhausted prematurely.")


class HierarchicalMappingSampler: ...  # subsamplers for MapDataset


class HierarchicalSequenceSampler: ...  # subsamplers for IterableDataset


# TODO: Hierarchical sampler for Sequence

S: Final = "slices"  # slice
M: Final = "masks"  # bool
B: Final = "bounds"  # tuple
W: Final = "windows"  #

ONE: Final = False
MANY: Final = True

MODE = TypeVar(
    "MODE",
    Literal["masks"],
    Literal["slices"],
    Literal["points"],
    Literal["bounds"],
)
MODES: TypeAlias = Literal["bounds", "slices", "masks", "windows"]
MULTI = TypeVar("MULTI", Literal[True], Literal[False])


# FIXME: Allow ±∞ as bounds for timedelta types? This would allow "growing" windows.
class SlidingWindowSampler(BaseSampler, Generic[MODE, NumpyDTVar, NumpyTDVar]):
    r"""Sampler that generates a single sliding window over an interval.

    Note:
        This sampler is intended to be used with continuous time series data types,
        such as `float`, `numpy.timedelta64`, `datetime.timedelta`, `pandas.Timestamp`, etc.
        For discrete time series, particularly integer types, use `DiscreteSlidingWindowSampler`.
        Otherwise, off-by-one errors may occur, for example,
        for `horizons=(3, 1)` and `stride=2`, given the data `np.arange(10)`,
        this sampler will produce 3 windows.

    Note:
        drop_last and shuffle are mutable,

    Args:
        data_source: A dataset that contains the ordered timestamps.
        stride: How much the window(s) advances at each step.
        horizons: The size of the window. Multiple can be given, in which case
            the sampler will return a list of windows.
        mode: There are 4 modes, determining the output of the sampler (default: 'masks').
            - `tuple` / 'bounds': return the bounds of the window(s) as a tuple.
            - `slice` / 'slice': return the slice of the lower and upper bounds of the window.
            - `bool` / 'mask': return the boolean mask of the data points inside the window.
            - `list` / 'window': return the actual data points inside the window(s).
        closed: Whether the window is considered closed on the left or right (default: 'left').
            - `left`: the window is closed on the left and open on the right.
            - `right`: the window is open on the left and closed on the right.
            - `both`: the window is closed on both sides.
            - `neither`: the window is open on both sides.
        shuffle: Whether to shuffle the indices (default: False).
        drop_last: Whether to drop the last incomplete window (default: False).
            If multiple horizons are given, then this considers only the final horizon.

    The window is considered to be closed on the left and open on the right, but this
    can be changed by setting 'closed'

    Moreover, the sampler can return multiple subsequent horizons,
    if `horizons` is a sequence of `TimeDelta` objects. In this case,
    lists of the above objects are returned.

    Inputs:
    - Ordered timestamps $T$
    - Starting time $t_0$
    - Final time $t_f$
    - stride ∆t (how much the sampler advances at each step) default,
      depending on the data type of $T$:
        - integer: $GCD(∆T)$
        - float: $\max(⌊AVG(∆T)⌋, ε)$
        - timestamp: resolution dependent.
    - horizons: `TimeDelta` or `tuple[TimeDelta, ...]`

    The sampler will return tuples of `len(horizons)+1`.
    """

    data: NDArray[NumpyDTVar]

    horizons: NumpyTDVar | NDArray[NumpyTDVar]
    stride: NumpyTDVar
    mode: MODE
    multi_horizon: bool
    shuffle: bool
    drop_last: bool

    # dependent variables
    tmin: NumpyDTVar
    tmax: NumpyDTVar
    cumulative_horizons: NDArray[NumpyTDVar]
    grid: Final[NDArray[np.integer]]

    # region overloads
    @overload
    def __init__(
        self: "SlidingWindowSampler[W, ONE]",
        data_source: SequentialDataset[NumpyDTVar],
        /,
        *,
        stride: str | NumpyTDVar,
        horizons: str | NumpyTDVar,
        mode: W,
        closed: Literal["left", "right", "both", "neither"] = "left",
        shuffle: bool = False,
    ) -> None: ...
    @overload
    def __init__(
        self: "SlidingWindowSampler[W, MANY]",
        data_source: SequentialDataset[NumpyDTVar],
        /,
        *,
        stride: str | NumpyTDVar,
        horizons: Sequence[str | NumpyTDVar],
        mode: W,
        closed: Literal["left", "right", "both", "neither"] = "left",
        shuffle: bool = False,
    ) -> None: ...
    @overload
    def __init__(
        self: "SlidingWindowSampler[S, ONE]",
        data_source: SequentialDataset[NumpyDTVar],
        /,
        *,
        stride: str | NumpyTDVar,
        horizons: str | NumpyTDVar,
        mode: S,
        closed: Literal["left", "right", "both", "neither"] = "left",
        shuffle: bool = False,
    ) -> None: ...
    @overload
    def __init__(
        self: "SlidingWindowSampler[S, MANY]",
        data_source: SequentialDataset[NumpyDTVar],
        /,
        *,
        stride: str | NumpyTDVar,
        horizons: Sequence[str | NumpyTDVar],
        mode: S,
        closed: Literal["left", "right", "both", "neither"] = "left",
        shuffle: bool = False,
    ) -> None: ...
    @overload
    def __init__(
        self: "SlidingWindowSampler[M, ONE]",
        data_source: SequentialDataset[NumpyDTVar],
        /,
        *,
        stride: str | NumpyTDVar,
        horizons: str | NumpyTDVar,
        mode: M,
        closed: Literal["left", "right", "both", "neither"] = "left",
        shuffle: bool = False,
    ) -> None: ...
    @overload
    def __init__(
        self: "SlidingWindowSampler[M, MANY]",
        data_source: SequentialDataset[NumpyDTVar],
        /,
        *,
        stride: str | NumpyTDVar,
        horizons: Sequence[str | NumpyTDVar],
        mode: M,
        closed: Literal["left", "right", "both", "neither"] = "left",
        shuffle: bool = False,
    ) -> None: ...
    @overload
    def __init__(
        self: "SlidingWindowSampler[B, ONE]",
        data_source: SequentialDataset[NumpyDTVar],
        /,
        *,
        stride: str | NumpyTDVar,
        horizons: str | NumpyTDVar,
        mode: B,
        closed: Literal["left", "right", "both", "neither"] = "left",
        shuffle: bool = False,
    ) -> None: ...
    @overload
    def __init__(
        self: "SlidingWindowSampler[B, MANY]",
        data_source: SequentialDataset[NumpyDTVar],
        /,
        *,
        stride: str | NumpyTDVar,
        horizons: Sequence[str | NumpyTDVar],
        mode: B,
        closed: Literal["left", "right", "both", "neither"] = "left",
        shuffle: bool = False,
    ) -> None: ...

    # endregion overloads
    def __init__(
        self,
        data_source: SequentialDataset[NumpyDTVar],
        /,
        *,
        horizons: str | NumpyTDVar | Sequence[str] | Sequence[NumpyTDVar],
        stride: str | NumpyTDVar,
        mode: MODE = "masks",  # type: ignore[assignment]
        closed: Literal["left", "right", "both", "neither"] = "left",
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> None:
        super().__init__(shuffle=shuffle)
        self.data = np.asarray(data_source)
        self.mode = mode
        self.closed = closed
        self.drop_last = drop_last
        self.stride = Timedelta(stride) if isinstance(stride, str) else stride
        zero_td = 0 * self.stride  # timedelta of the correct type.
        assert self.stride > zero_td, "stride must be positive."

        # region set horizon(s)
        match horizons:
            case str() as string:
                self.horizons = Timedelta(string)
                self.multi_horizon = False
                self.horizons = np.array([self.horizons])
            case Iterable() as iterable:
                self.multi_horizon = True
                self.horizons = pd.to_timedelta(iterable)  # Series or TimeDeltaIndex
            case TDLike() as td:
                self.multi_horizon = False
                self.horizons = np.array([td])
            case _:
                raise TypeError(f"Invalid type {type(horizons)} for {horizons=}")

        concat_horizons = np.concatenate([[zero_td], self.horizons])  # type: ignore[arg-type]
        self.cumulative_horizons = np.cumsum(concat_horizons)
        # endregion set horizon(s)

        # region set tmin, tmax, offset
        self.tmin = get_first(data_source)
        self.tmax = get_last(data_source)
        # endregion set tmin, tmax, offset

        # precompute the possible slices
        # NOTE: we compute the grid assuming drop_last=False,
        #  dropping the last slice is done in __iter__ and __len__
        #  Thus, we select the grid based of cumulative_horizons[-2]
        # Q: What if only one horizon with tmin=-∞?
        self.grid = compute_grid(
            self.tmin,
            self.tmax - self.cumulative_horizons[-2],
            self.stride,
        )

        # offset = self.tmin + self.cumulative_horizons[-1]  # type: ignore[assignment, call-overload, operator]
        # grid = compute_grid(self.tmin, self.tmax, self.stride, offset=offset)
        # # Q: Why drop negative k?? If tmin is unbounded, then there is a problem...
        # self.grid = grid[grid >= 0]  # type: ignore[assignment, operator]

        # NOTE: append single value to grid

    def __len__(self) -> int:
        r"""Return the number of samples."""
        return len(self.grid) - self.drop_last

    # region make functions ------------------------------------------------------------
    @staticmethod
    def make_slice(bounds: NDArray[NumpyDTVar]) -> slice:
        r"""Return a tuple of slices."""
        return slice(bounds[0], bounds[-1])

    @staticmethod
    def make_bound(bounds: NDArray[NumpyDTVar]) -> tuple[NumpyDTVar, NumpyTDVar]:
        r"""Return the boundaries of the window."""
        return bounds[0], bounds[-1]

    def make_index(self, bounds: NDArray[NumpyDTVar]) -> NDArray[np.integer]:
        r"""Return indices of the data points inside the window."""
        raise NotImplementedError

    def make_mask(self, bounds: NDArray[NumpyDTVar]) -> NDArray[np.bool_]:
        r"""Return a tuple of masks."""
        return (bounds[0] <= self.data) & (self.data < bounds[-1])

    def make_window(self, bounds: NDArray[NumpyDTVar]) -> NDArray[NumpyDTVar]:
        r"""Return the actual data points inside the window."""
        return self.data[(bounds[0] <= self.data) & (self.data < bounds[-1])]

    @staticmethod
    def make_bounds(bounds: NDArray[NumpyDTVar]) -> list[tuple[NumpyDTVar, NumpyTDVar]]:
        r"""Return the boundaries of the windows."""
        return [(start, stop) for start, stop in sliding_window_view(bounds, 2)]

    @staticmethod
    def make_slices(bounds: NDArray[NumpyDTVar]) -> list[slice]:
        r"""Return a tuple of slices."""
        return [slice(start, stop) for start, stop in sliding_window_view(bounds, 2)]

    def make_masks(self, bounds: NDArray[NumpyDTVar]) -> list[NDArray[np.bool_]]:
        r"""Return a tuple of masks."""
        return [
            (start <= self.data) & (self.data < stop)
            for start, stop in sliding_window_view(bounds, 2)
        ]

    def make_indices(self, bounds: NDArray[NumpyDTVar]) -> list[NDArray[np.integer]]:
        r"""Return indices of the data points inside the windows."""
        raise NotImplementedError

    def make_windows(self, bounds: NDArray[NumpyDTVar]) -> list[NDArray[NumpyDTVar]]:
        r"""Return the actual data points inside the windows."""
        return [
            self.data[(start <= self.data) & (self.data < stop)]
            for start, stop in sliding_window_view(bounds, 2)
        ]

    # endregion make functions ---------------------------------------------------------

    @property
    def make_output(self) -> Callable[[NDArray[NumpyDTVar]], Any]:
        r"""Return the correct yield function."""
        match self.mode, self.multi_horizon:
            case "bounds", False:
                return self.make_bound
            case "bounds", True:
                return self.make_bounds
            case "masks", False:
                return self.make_mask
            case "masks", True:
                return self.make_masks
            case "slices", False:
                return self.make_slice
            case "slices", True:
                return self.make_slices
            case "windows", False:
                return self.make_window
            case "windows", True:
                return self.make_windows
            case _:
                raise ValueError(f"Invalid mode {self.mode=}")

    # region overloads
    @overload
    def __iter__(self: "SlidingWindowSampler[S, ONE, Any, Any]") -> Iterator[slice]: ...
    @overload
    def __iter__(
        self: "SlidingWindowSampler[S, MANY, Any, Any]",
    ) -> Iterator[list[slice]]: ...
    @overload
    def __iter__(
        self: "SlidingWindowSampler[M, ONE, Any, Any]",
    ) -> Iterator[tuple[NumpyDTVar, NumpyDTVar]]: ...
    @overload
    def __iter__(
        self: "SlidingWindowSampler[M, MANY, Any, Any]",
    ) -> Iterator[list[tuple[NumpyDTVar, NumpyDTVar]]]: ...
    @overload
    def __iter__(
        self: "SlidingWindowSampler[M, ONE, Any, Any]",
    ) -> Iterator[NDArray[np.bool_]]: ...
    @overload
    def __iter__(
        self: "SlidingWindowSampler[M, MANY, Any, Any]",
    ) -> Iterator[list[NDArray[np.bool_]]]: ...
    @overload
    def __iter__(
        self: "SlidingWindowSampler[W, ONE, NumpyDTVar, Any]",
    ) -> Iterator[NDArray[NumpyDTVar]]: ...
    @overload
    def __iter__(
        self: "SlidingWindowSampler[W, MANY, NumpyDTVar, Any]",
    ) -> Iterator[list[NDArray[NumpyDTVar]]]: ...

    # endregion overloads
    def __iter__(self):  # pyright: ignore[reportGeneralTypeIssues]
        r"""Iterate through.

        For each k, we return either:

        - mode=points: $(x₀ + k⋅∆t, x₁+k⋅∆t, …, xₘ+k⋅∆t)$
        - mode=slices: $(slice(x₀ + k⋅∆t, x₁+k⋅∆t), …, slice(xₘ₋₁+k⋅∆t, xₘ+k⋅∆t))$
        - mode=masks: $(mask_1, …, mask_m)$
        """
        grid = self.grid[:-1] if self.drop_last else self.grid

        if self.shuffle:
            grid = grid[np.random.permutation(len(grid))]

        # unpack variables (avoid repeated lookups)
        window = self.tmin + self.cumulative_horizons
        stride = self.stride
        make_key = self.make_output

        for k in grid:  # NOTE: k is some range of integers.
            yield make_key(window + k * stride)


# class SlidingSampler: ...
# class SlidingSliceSampler: ...
# class SlidingMaskSampler: ...
# class SlidingWindowSampler: ...
# class SlidingBoundsSampler: ...
# class SlidingIndexSampler: ...


class DiscreteSlidingWindowSampler(BaseSampler):
    """Sample a sliding window from the data source."""


class RandomWindowSampler(BaseSampler):
    """Sample a random window from the data source.

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


def get_index(dataset: Dataset, /) -> Index:
    r"""Return an index object for the dataset.

    We support the following data types:
        - Series, DataFrame.
        - Mapping Types
        - Iterable Types
    """
    match dataset:
        # NOTE: Series and DataFrame satisfy the MapDataset protocol.
        case Series() | DataFrame() as pandas_dataset:  # type: ignore[misc]
            return pandas_dataset.index  # type: ignore[unreachable]
        case MapDataset() as map_dataset:
            return Index(map_dataset.keys())
        case IterableDataset() as iterable_dataset:
            return Index(range(len(iterable_dataset)))
        case _:
            raise TypeError(f"Got unsupported data type {type(dataset)}.")


def get_first(dataset: Dataset[T], /) -> T:
    """Return the first element of the dataset."""
    match dataset:
        case Series() | DataFrame() as pandas_dataset:  # type: ignore[misc]
            return pandas_dataset.iloc[0]  # type: ignore[unreachable]
        case MapDataset() as map_dataset:
            return map_dataset[next(iter(map_dataset.keys()))]
        case IterableDataset() as iterable_dataset:
            return next(iter(iterable_dataset))
        case _:
            raise TypeError(f"Got unsupported data type {type(dataset)}.")


def get_last(dataset: Dataset[T], /) -> T:
    """Return the last element of the dataset."""
    match dataset:
        case Series() | DataFrame() as pandas_dataset:  # type: ignore[misc]
            return pandas_dataset.iloc[-1]  # type: ignore[unreachable]
        case MapDataset() as map_dataset:
            return map_dataset[next(reversed(map_dataset.keys()))]
        case IterableDataset() as iterable_dataset:
            return next(reversed(iterable_dataset))
        case _:
            raise TypeError(f"Got unsupported data type {type(dataset)}.")


@overload
def compute_grid(
    tmin: DTVar, tmax: DTVar, step: TDVar, /, *, offset: Optional[DTVar] = None
) -> NDArray[np.int_]: ...
@overload
def compute_grid(
    tmin: str, tmax: str, step: str, /, *, offset: Optional[str] = None
) -> NDArray[np.int_]: ...
def compute_grid(tmin, tmax, step, /, *, offset=None):
    r"""Compute $\{k∈ℤ ∣ tₘᵢₙ ≤ t₀+k⋅Δt ≤ tₘₐₓ\}$.

    That is, a list of all integers such that $t₀+k⋅Δ$ is in the interval $[tₘᵢₙ, tₘₐₓ]$.
    Special case: if $Δt=0$, returns $[0]$.

    .. math::
        if ∆t > 0
        tₘᵢₙ ≤ t₀+k⋅Δt ⟺ (tₘᵢₙ-t₀)/Δt ≤ k ⟺ k ≥ ⌈(tₘᵢₙ-t₀)/Δt⌉
        t₀+k⋅Δt ≤ tₘₐₓ ⟺ k ≤ ⌊(tₘₐₓ-t₀)/Δt⌋
        if ∆t < 0
        tₘᵢₙ ≤ t₀+k⋅Δt ⟺ (tₘᵢₙ-t₀)/Δt ≥ k ⟺ k ≤ ⌊(tₘᵢₙ-t₀)/Δt⌋
        t₀+k⋅Δt ≤ tₘₐₓ ⟺ k ≥ ⌈(tₘₐₓ-t₀)/Δt⌉

    Note:
        This function is used to compute the strides for the sliding window sampler.
        given a window ∆s<tₘₐₓ-tₘᵢₙ, we want to find all k≥0 such that
        tₘᵢₙ ≤ [tₗ+k⋅Δt, tᵣ+k∆t] ≤ tₘₐₓ. This is equivalent to finding all k such that


    """
    # cast strings to timestamp/timedelta
    tmin = cast(DTVar, Timestamp(tmin) if isinstance(tmin, str) else tmin)
    tmax = cast(DTVar, Timestamp(tmax) if isinstance(tmax, str) else tmax)
    td = cast(TDVar, Timedelta(step) if isinstance(step, str) else step)
    offset = (
        tmin
        if offset is None
        else cast(DTVar, Timestamp(offset) if isinstance(offset, str) else offset)
    )
    # validate inputs
    if tmin > offset or offset > tmax:
        raise ValueError("tₘᵢₙ ≤ t₀ ≤ tₘₐₓ violated!")

    zero_td = tmin - tmin
    if td > zero_td:
        kmin = math.ceil(cast(TDVar, tmin - offset) / td)  # type: ignore[redundant-cast]
        kmax = math.floor(cast(TDVar, tmax - offset) / td)  # type: ignore[redundant-cast]
    elif td < zero_td:
        kmin = math.ceil(cast(TDVar, tmax - offset) / td)
        kmax = math.floor(cast(TDVar, tmin - offset) / td)
    else:
        raise ValueError(f"Δt={td} is not allowed!")
    return np.arange(kmin, kmax + 1)
