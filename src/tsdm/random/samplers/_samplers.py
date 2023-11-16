r"""Samplers for randomly selecting data.

Note:
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
from abc import abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from itertools import chain

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from pandas import DataFrame, Index, Series, Timedelta, Timestamp
from typing_extensions import (
    Any,
    ClassVar,
    Generic,
    Literal,
    Optional,
    Protocol,
    TypeAlias,
    TypeVar,
    cast,
    overload,
    runtime_checkable,
)

from tsdm.types.time import DT, TD, DateTime, TimeDelta as TDLike
from tsdm.types.variables import (
    any_co as T_co,
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


# region helper functions --------------------------------------------------------------
def get_index(dataset: Dataset[T], /) -> Index:
    r"""Return an index object for the dataset.

    We support the following data types:
        - Series, DataFrame.
        - Mapping Types
        - Iterable Types
    """
    match dataset:
        # NOTE: Series and DataFrame satisfy the MapDataset protocol.
        case Series() | DataFrame() as pandas_dataset:
            return pandas_dataset.index
        case MapDataset() as map_dataset:
            return Index(map_dataset.keys())
        case IterableDataset() as iterable_dataset:
            return Index(range(len(iterable_dataset)))
        case _:
            raise TypeError(f"Got unsupported data type {type(dataset)}.")


def get_first(dataset: Dataset[T], /) -> T:
    """Return the first element of the dataset."""
    match dataset:
        case Series() | DataFrame() as pandas_dataset:
            return pandas_dataset.iloc[0]
        case MapDataset() as map_dataset:
            return map_dataset[next(iter(map_dataset.keys()))]
        case IterableDataset() as iterable_dataset:
            return next(iter(iterable_dataset))
        case _:
            raise TypeError(f"Got unsupported data type {type(dataset)}.")


def get_last(dataset: Dataset[T], /) -> T:
    """Return the last element of the dataset."""
    match dataset:
        case Series() | DataFrame() as pandas_dataset:
            return pandas_dataset.iloc[-1]
        case MapDataset() as map_dataset:
            return map_dataset[next(reversed(map_dataset.keys()))]
        case IterableDataset() as iterable_dataset:
            return next(reversed(iterable_dataset))
        case _:
            raise TypeError(f"Got unsupported data type {type(dataset)}.")


def compute_grid(
    tmin: str | DateTime[TD],
    tmax: str | DateTime[TD],
    step: str | TD,
    /,
    *,
    offset: Optional[str | DateTime[TD]] = None,
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
    t_min = cast(Any, Timestamp(tmin) if isinstance(tmin, str) else tmin)
    t_max = cast(Any, Timestamp(tmax) if isinstance(tmax, str) else tmax)
    t_0 = cast(Any, Timestamp(offset) if isinstance(offset, str) else offset)
    delta = Timedelta(step) if isinstance(step, str) else step

    # validate inputs
    if (t_min > t_0) or (t_0 > t_max):
        raise ValueError("tₘᵢₙ ≤ t₀ ≤ tₘₐₓ violated!")

    # NOTE: time-delta types should support divmod / floordiv!
    #  Importantly, floordiv always rounds down, even for negative numbers.
    #  We use this formula for ceil-div: https://stackoverflow.com/a/17511341/9318372
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
class Sampler(Protocol[T_co]):
    r"""Protocol for `Sampler` classes.

    Plug-in replacement for `torch.utils.data.Sampler`.
    In contrast, each Sampler must additionally have a `shuffle` attribute.
    """

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
    """Sample randomly from the data source.

    Note:
        In contrast to torch.utils.data.RandomSampler, this sampler also works for map-style datasets.
        In this case, the sampler will return random values of the mapping.
        For Iterable-style datasets, the sampler will return random values of the iterable.
    """

    data: Dataset[T_co]  # type: ignore[misc]
    shuffle: bool = False

    index: Index = field(init=False)
    size: int = field(init=False)

    def __init__(self, data: Dataset[T_co], /, *, shuffle: bool = False) -> None:
        """Initialize the sampler."""
        super(RandomSampler, self).__init__(shuffle=shuffle)
        self.data = data
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


# class HierarchicalMappingSampler: ...  # subsamplers for MapDataset
# class HierarchicalSequenceSampler: ...  # subsamplers for IterableDataset
@pprint_repr
@dataclass
class HierarchicalSampler(BaseSampler[tuple[K, K2]]):
    r"""Draw samples from a hierarchical data source."""

    data: MapDataset[K, Dataset[K2]]
    r"""The shared index."""
    subsamplers: Mapping[K, Sampler[K2]] = NotImplemented
    r"""The subsamplers to sample from the collection."""

    _: KW_ONLY

    early_stop: bool = False
    r"""Whether to stop sampling when the index is exhausted."""
    shuffle: bool = False
    r"""Whether to sample in random order."""

    def __post_init__(self) -> None:
        if self.subsamplers is NotImplemented:
            self.subsamplers = {
                key: RandomSampler(self.data[key], shuffle=self.shuffle)
                for key in self.data.keys()
            }

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


# TODO: Hierarchical sampler for Sequence

S: TypeAlias = Literal["slices"]  # slice
M: TypeAlias = Literal["masks"]  # bool
B: TypeAlias = Literal["bounds"]  # tuple
W: TypeAlias = Literal["windows"]  #
MODE = TypeVar("MODE", B, M, W, S)
MODES: TypeAlias = B | M | W | S

ONE: TypeAlias = Literal[False]
MANY: TypeAlias = Literal[True]
MULTI = TypeVar("MULTI", ONE, MANY)


# FIXME: Allow ±∞ as bounds for timedelta types? This would allow "growing" windows.
class SlidingWindowSampler(BaseSampler, Generic[DT, MODE, MULTI]):
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
            NOTE: The size is specified as a timedelta, not as the number of data points.
              When sampling discrete data, this may lead to off-by-one errors.
              Consider using `DiscreteSlidingWindowSampler` instead.
            Multiple horizons can be given, in which case the sampler will return a list.
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
            If true, it is guaranteed that each window is completely contained in the data.
            If false, the last window may only partially overlap with the data.
            If multiple horizons are given, these rules apply to the last horizon.

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

    data: NDArray[DT]  # type: ignore[type-var]

    horizons: TDLike | NDArray[TDLike]  # type: ignore[type-var]
    stride: TDLike
    mode: MODE
    multi_horizon: bool
    shuffle: bool
    drop_last: bool

    # dependent variables
    tmin: DT
    tmax: DT
    cumulative_horizons: NDArray[TDLike]
    # grid: Final[NDArray[np.integer]]

    # region __init__ overloads --------------------------------------------------------
    @overload
    def __init__(
        self: "SlidingWindowSampler[DT, S, ONE]",
        data_source: SequentialDataset[DT],
        /,
        *,
        stride: str | TD,
        horizons: str | TD,
        mode: S,
        shuffle: bool = False,
    ) -> None: ...
    @overload
    def __init__(
        self: "SlidingWindowSampler[DT, B, ONE]",
        data_source: SequentialDataset[DT],
        /,
        *,
        stride: str | TD,
        horizons: str | TD,
        mode: B,
        shuffle: bool = False,
    ) -> None: ...
    @overload
    def __init__(
        self: "SlidingWindowSampler[DT, M, ONE]",
        data_source: SequentialDataset[DT],
        /,
        *,
        stride: str | TD,
        horizons: str | TD,
        mode: M,
        shuffle: bool = False,
    ) -> None: ...
    @overload
    def __init__(
        self: "SlidingWindowSampler[DT, W, ONE]",
        data_source: SequentialDataset[DT],
        /,
        *,
        stride: str | TD,
        horizons: str | TD,
        mode: W,
        shuffle: bool = False,
    ) -> None: ...
    @overload
    def __init__(
        self: "SlidingWindowSampler[DT, S, MANY]",
        data_source: SequentialDataset[DT],
        /,
        *,
        stride: str | TD,
        horizons: Sequence[str | TD],
        mode: S,
        shuffle: bool = False,
    ) -> None: ...
    @overload
    def __init__(
        self: "SlidingWindowSampler[DT, B, MANY]",
        data_source: SequentialDataset[DT],
        /,
        *,
        stride: str | TD,
        horizons: Sequence[str | TD],
        mode: B,
        shuffle: bool = False,
    ) -> None: ...
    @overload
    def __init__(
        self: "SlidingWindowSampler[DT, M, MANY]",
        data_source: SequentialDataset[DT],
        /,
        *,
        stride: str | TD,
        horizons: Sequence[str | TD],
        mode: M,
        shuffle: bool = False,
    ) -> None: ...
    @overload
    def __init__(
        self: "SlidingWindowSampler[DT, W, MANY]",
        data_source: SequentialDataset[DT],
        /,
        *,
        stride: str | TD,
        horizons: Sequence[str | TD],
        mode: W,
        shuffle: bool = False,
    ) -> None: ...

    # endregion __init__ overloads -----------------------------------------------------
    def __init__(
        self,
        data_source,
        /,
        *,
        horizons,
        stride,
        mode="masks",
        shuffle=False,
        drop_last=False,
    ):
        super().__init__(shuffle=shuffle)
        self.data = np.asarray(data_source)
        self.mode = mode
        self.drop_last = drop_last
        self.stride = Timedelta(stride) if isinstance(stride, str) else stride
        zero_td = 0 * self.stride  # timedelta of the correct type.
        assert self.stride > zero_td, "stride must be positive."

        # region set horizon(s)
        match horizons:
            case str() as string:
                self.multi_horizon = False
                self.horizons = np.array(Timedelta(string).to_numpy())
            case Iterable() as iterable:
                values = list(iterable)
                self.multi_horizon = True
                self.horizons = (
                    pd.to_timedelta(values).to_numpy()  # Series or TimeDeltaIndex
                    if isinstance(values[0], str)
                    else np.array(values)
                )
            case TDLike() as td:
                self.multi_horizon = False
                self.horizons = np.array([td])
            case _:
                raise TypeError(f"Invalid type {type(horizons)} for {horizons=}")

        self.cumulative_horizons = np.cumsum(
            np.concatenate([
                np.array([zero_td], dtype=self.horizons.dtype),
                self.horizons,
            ])
        )
        # endregion set horizon(s)

        # region set tmin, tmax, offset
        self.tmin = get_first(data_source)
        self.tmax = get_last(data_source)
        # endregion set tmin, tmax, offset

    @property
    def grid(self) -> NDArray[np.integer]:
        r"""Return the grid of indices."""
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

    # region make functions ------------------------------------------------------------
    @staticmethod
    def make_bound(bounds: NDArray[DT]) -> tuple[DT, DT]:
        r"""Return the boundaries of the window."""
        return bounds[0], bounds[-1]

    @staticmethod
    def make_slice(bounds: NDArray[DT]) -> slice:
        r"""Return a tuple of slices."""
        return slice(bounds[0], bounds[-1])

    def make_mask(self, bounds: NDArray[DT]) -> NDArray[np.bool_]:
        r"""Return a tuple of masks."""
        return (bounds[0] <= self.data) & (self.data < bounds[-1])

    def make_window(self, bounds: NDArray[DT]) -> NDArray[DT]:
        r"""Return the actual data points inside the window."""
        return self.data[(bounds[0] <= self.data) & (self.data < bounds[-1])]

    @staticmethod
    def make_bounds(bounds: NDArray[DT]) -> list[tuple[DT, DT]]:
        r"""Return the boundaries of the windows."""
        return [(start, stop) for start, stop in sliding_window_view(bounds, 2)]

    @staticmethod
    def make_slices(bounds: NDArray[DT]) -> list[slice]:
        r"""Return a tuple of slices."""
        return [slice(start, stop) for start, stop in sliding_window_view(bounds, 2)]

    def make_masks(self, bounds: NDArray[DT]) -> list[NDArray[np.bool_]]:
        r"""Return a tuple of masks."""
        return [
            (start <= self.data) & (self.data < stop)
            for start, stop in sliding_window_view(bounds, 2)
        ]

    def make_windows(self, bounds: NDArray[DT]) -> list[NDArray[DT]]:
        r"""Return the actual data points inside the windows."""
        return [
            self.data[(start <= self.data) & (self.data < stop)]
            for start, stop in sliding_window_view(bounds, 2)
        ]

    @property
    def _MAKE_FUNCTIONS(self) -> dict[tuple[MODES, bool], Callable[[NDArray[DT]], Any]]:
        r"""Return the make functions."""
        return {
            ("bounds", False): self.make_bound,
            ("bounds", True): self.make_bounds,
            ("masks", False): self.make_mask,
            ("masks", True): self.make_masks,
            ("slices", False): self.make_slice,
            ("slices", True): self.make_slices,
            ("windows", False): self.make_window,
            ("windows", True): self.make_windows,
        }

    # endregion make functions ---------------------------------------------------------

    # region __iter__ overloads --------------------------------------------------------
    @overload
    def __iter__(self: "SlidingWindowSampler[DT, S, ONE]") -> Iterator[slice]: ...
    @overload
    def __iter__(
        self: "SlidingWindowSampler[DT, B, ONE]",
    ) -> Iterator[tuple[DT, DT]]: ...
    @overload
    def __iter__(
        self: "SlidingWindowSampler[DT, M, ONE]",
    ) -> Iterator[NDArray[np.bool_]]: ...
    @overload
    def __iter__(
        self: "SlidingWindowSampler[DT, W, ONE]",
    ) -> Iterator[NDArray[DT]]: ...
    @overload
    def __iter__(
        self: "SlidingWindowSampler[DT, S, MANY]",
    ) -> Iterator[list[slice]]: ...
    @overload
    def __iter__(
        self: "SlidingWindowSampler[DT, B, MANY]",
    ) -> Iterator[list[tuple[DT, DT]]]: ...
    @overload
    def __iter__(
        self: "SlidingWindowSampler[DT, M, MANY]",
    ) -> Iterator[list[NDArray[np.bool_]]]: ...
    @overload
    def __iter__(
        self: "SlidingWindowSampler[DT, W, MANY]",
    ) -> Iterator[list[NDArray[DT]]]: ...

    # endregion __iter__ overloads -----------------------------------------------------
    def __iter__(self):  # pyright: ignore[reportGeneralTypeIssues]
        r"""Iterate through.

        For each k, we return either:

        - mode=points: $(x₀ + k⋅∆t, x₁+k⋅∆t, …, xₘ+k⋅∆t)$
        - mode=slices: $(slice(x₀ + k⋅∆t, x₁+k⋅∆t), …, slice(xₘ₋₁+k⋅∆t, xₘ+k⋅∆t))$
        - mode=masks: $(mask_1, …, mask_m)$
        """
        # unpack variables (avoids attribute lookup in loop)
        window = self.tmin + self.cumulative_horizons  # type: ignore[operator]
        stride = self.stride
        make_fn = self._MAKE_FUNCTIONS[self.mode, self.multi_horizon]
        grid = self.grid

        if self.shuffle:
            grid = grid[np.random.permutation(len(grid))]

        for k in grid:  # NOTE: k is some range of integers.
            yield make_fn(window + k * stride)


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
