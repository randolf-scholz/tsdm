r"""Samplers for randomly selecting data.

NOTE: In the context of pytorch dataloaders, samplers typically are used to select indices.
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
from collections.abc import Callable, Iterator, Mapping, Sequence
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

from tsdm.types.time import DTVar, NumpyDTVar, NumpyTDVar, TDVar
from tsdm.types.variables import any_co as T_co, key_other_var as K2, key_var as K
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

    def __init__(
        cls, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwds: Any
    ) -> None:
        """When a new class/subclass is created, this method is called."""
        super().__init__(name, bases, namespace, **kwds)

        if "LOGGER" not in namespace:
            cls.LOGGER = logging.getLogger(f"{cls.__module__}.{cls.__name__}")


class BaseSampler(Sampler[T_co], metaclass=BaseSamplerMetaClass):
    r"""Abstract Base Class for all Samplers."""

    LOGGER: ClassVar[logging.Logger]
    r"""Logger for the sampler."""

    shuffle: bool = False
    r"""Whether to randomize sampling."""

    def __init__(self, *, shuffle: bool) -> None:
        r"""Initialize the sampler."""
        self.shuffle = shuffle

    @abstractmethod
    def __len__(self) -> int:
        r"""Return the length of the sampler."""
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[T_co]:
        r"""Return an iterator over the indices of the data source."""
        ...


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


@overload
def compute_grid(
    tmin: DTVar, tmax: DTVar, step: TDVar, /, *, offset: Optional[DTVar] = None
) -> NDArray[np.int_]: ...
@overload
def compute_grid(
    tmin: str, tmax: str, step: str, /, *, offset: Optional[str] = None
) -> NDArray[np.int_]: ...
@overload
def compute_grid(tmin, tmax, step, /, *, offset=None) -> NDArray[np.int_]:
    r"""Compute $\{k∈ℤ ∣ tₘᵢₙ ≤ t₀+k⋅Δt ≤ tₘₐₓ\}$.

    That is, a list of all integers such that $t₀+k⋅Δ$ is in the interval $[tₘᵢₙ, tₘₐₓ]$.
    Special case: if $Δt=0$, returns $[0]$.
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
    if td == tmin - tmin:  # generates zero-variable of the correct type.
        raise ValueError("Δt=0 is not allowed!")
    if tmin > offset or offset > tmax:
        raise ValueError("tₘᵢₙ ≤ t₀ ≤ tₘₐₓ violated!")

    kmax = math.floor(cast(TDVar, tmax - offset) / td)  # type: ignore[redundant-cast]
    kmin = math.ceil(cast(TDVar, tmin - offset) / td)  # type: ignore[redundant-cast]

    return np.arange(kmin, kmax + 1)


@pprint_repr
@dataclass(init=False, slots=True)
class RandomSampler(BaseSampler[K]):
    """Sample randomly from the data source.

    Note:
        - In the input is a DataFrame, we raise an exception since it is not clear what to do.
          In this case, like what is desired is to sample from the inde
    """

    data: Dataset[K, Any]  # map style or iterable style
    shuffle: bool = False

    index: Index = field(init=False)
    size: int = field(init=False)

    @overload
    def __init__(
        self: "RandomSampler[K]",
        data_source: MapDataset[K, Any],
        /,
        shuffle: bool = ...,
    ) -> None: ...
    @overload
    def __init__(
        self: "RandomSampler[int]",
        data_source: IterableDataset,
        /,
        shuffle: bool = ...,
    ) -> None: ...
    def __init__(self, data_source: Dataset, /, shuffle: bool = False) -> None:
        """Initialize the sampler."""
        super().__init__(shuffle=shuffle)
        self.data = data_source

        match data_source:
            case MapDataset() as map_data:  # in this case, K given by the Mapping
                self.index = Index(map_data.keys())
            case IterableDataset() as seq_data:  # can we forcibly bind K to int?
                self.index = Index(range(len(seq_data)))
            case _:
                raise TypeError

        self.size: int = len(self.index)

    def __iter__(self) -> Iterator[K]:
        yield from (
            self.index
            if not self.shuffle
            else self.index[np.random.permutation(self.size)]
        )

    def __len__(self) -> int:
        return self.size


@pprint_repr
@dataclass
class HierarchicalSampler(BaseSampler[tuple[K, K2]]):
    r"""Draw samples from a hierarchical data source."""

    data_source: MapDataset[K, Dataset[K2, Any]]
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

ONE: Final = "single"
MANY: Final = "multi"

MODE = TypeVar(
    "MODE",
    Literal["masks"],
    Literal["slices"],
    Literal["points"],
    Literal["bounds"],
)
MODES: TypeAlias = Literal["bounds", "slices", "masks", "windows"]


# FIXME: Allow ±∞ as bounds for timedelta types? This would allow "growing" windows.
class SlidingWindowSampler(BaseSampler, Generic[MODE, NumpyDTVar, NumpyTDVar]):
    r"""Sampler that generates a single sliding window over an interval.

    Args:
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
        tmin: The minimum value of the interval (optional).
        tmax: The maximum value of the interval (optional).

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

    mode: MODE
    stride: NumpyTDVar
    tmax: NumpyDTVar
    tmin: NumpyDTVar
    grid: Final[NDArray[np.integer]]

    horizons: NumpyTDVar | NDArray[NumpyTDVar]
    initial_windows: NDArray[NumpyDTVar]
    offset: NumpyDTVar
    total_horizon: NumpyTDVar
    zero_td: NumpyTDVar
    multi_horizon: bool
    cumulative_horizons: NDArray[NumpyTDVar]

    shuffle: bool = False

    @overload
    def __init__(
        self: "SlidingWindowSampler[W, ONE]",
        *,
        stride: str | NumpyTDVar,
        horizons: str | TD,
        mode: W,
    ) -> None: ...
    @overload
    def __init__(
        self: "SlidingWindowSampler[S, ONE]",
        *,
        stride: str | NumpyTDVar,
        horizons: str | TD,
        mode: S,
    ) -> None: ...
    @overload
    def __init__(
        self: "SlidingWindowSampler[B, ONE]",
        *,
        stride: str | NumpyTDVar,
        horizons: str | TD,
        mode: B,
    ) -> None: ...
    @overload
    def __init__(
        self: "SlidingWindowSampler[M, ONE]",
        *,
        stride: str | NumpyTDVar,
        horizons: str | TD,
        mode: M,
    ) -> None: ...
    @overload
    def __init__(
        self: "SlidingWindowSampler[W, MANY]",
        *,
        stride: str | NumpyTDVar,
        horizons: Sequence[str | TD],
        mode: W,
    ) -> None: ...
    @overload
    def __init__(
        self: "SlidingWindowSampler[S, MANY]",
        *,
        stride: str | NumpyTDVar,
        horizons: Sequence[str | TD],
        mode: S,
    ) -> None: ...
    @overload
    def __init__(
        self: "SlidingWindowSampler[B, MANY]",
        *,
        stride: str | NumpyTDVar,
        horizons: Sequence[str | TD],
        mode: B,
    ) -> None: ...
    @overload
    def __init__(
        self: "SlidingWindowSampler[M, MANY]",
        *,
        stride: str | NumpyTDVar,
        horizons: Sequence[str | TD],
        mode: M,
    ) -> None: ...
    def __init__(
        self,
        data_source: SequentialDataset[NumpyDTVar],
        /,
        *,
        horizons: str | NumpyTDVar | Sequence[str] | Sequence[NumpyTDVar],
        stride: str | NumpyTDVar,
        mode: MODE = "masks",  # type: ignore[assignment]
        tmin: Optional[str | NumpyDTVar] = None,
        tmax: Optional[str | NumpyDTVar] = None,
        closed: Literal["left", "right", "both", "neither"] = "left",
        shuffle: bool = False,
    ) -> None:
        super().__init__(shuffle=shuffle)
        self.mode = mode
        self.closed = closed

        # NOTE:

        self.stride = Timedelta(stride) if isinstance(stride, str) else stride

        match tmin:
            case None:
                self.tmin = (
                    self.data.iloc[0] if isinstance(self.data, Series) else self.data[0]
                )
            case str() as time_str:
                self.tmin = Timestamp(time_str)
            case _:
                self.tmin = tmin

        match tmax:
            case None:
                self.tmax = (
                    self.data.iloc[-1]
                    if isinstance(self.data, Series)
                    else self.data[-1]
                )
            case str() as time_str:
                self.tmax = Timestamp(time_str)
            case _:
                self.tmax = tmax

        # this gives us the correct zero, depending on the dtype
        self.zero_td = cast(NumpyTDVar, self.tmin - self.tmin)  # type: ignore[redundant-cast]
        assert self.stride > self.zero_td, "stride cannot be zero."

        horizons = Timedelta(horizons) if isinstance(horizons, str) else horizons

        if isinstance(horizons, Sequence):
            self.multi_horizon = True
            if isinstance(horizons[0], str | Timedelta | py_td):
                self.horizons = pd.to_timedelta(horizons)
                concat_horizons = self.horizons.insert(0, self.zero_td)  # type: ignore[union-attr]
            else:
                self.horizons = np.array(horizons)
                concat_horizons = np.concatenate([[self.zero_td], self.horizons])  # type: ignore[arg-type]
        else:
            self.multi_horizon = False
            self.horizons = horizons
            conat_horizons = np.concatenate([self.zero_td, self.horizons])

        self.total_horizon = self.horizons[-1]
        self.cumulative_horizons = np.cumsum(conat_horizons)

        self.initial_windows = self.tmin + self.cumulative_horizons  # type: ignore[assignment, call-overload, operator]

        self.offset = self.tmin + self.total_horizon  # type: ignore[assignment, call-overload, operator]

        # precompute the possible slices
        grid = compute_grid(self.tmin, self.tmax, self.stride, offset=self.offset)
        self.grid = grid[grid >= 0]  # type: ignore[assignment, operator]

    def __len__(self) -> int:
        r"""Return the number of samples."""
        return len(self.grid)

    @staticmethod
    def make_bound(bounds: NDArray[NumpyDTVar]) -> tuple[NumpyDTVar, NumpyTDVar]:
        r"""Return the boundaries of the window."""
        return bounds[0], bounds[-1]

    @staticmethod
    def make_bounds(bounds: NDArray[NumpyDTVar]) -> list[tuple[NumpyDTVar, NumpyTDVar]]:
        r"""Return the boundaries of the windows."""
        return [(start, stop) for start, stop in sliding_window_view(bounds, 2)]

    @staticmethod
    def make_slice(bounds: NDArray[NumpyDTVar]) -> slice:
        r"""Return a tuple of slices."""
        return slice(bounds[0], bounds[-1])

    @staticmethod
    def make_slices(bounds: NDArray[NumpyDTVar]) -> list[slice]:
        r"""Return a tuple of slices."""
        return [slice(start, stop) for start, stop in sliding_window_view(bounds, 2)]

    def make_mask(self, bounds: NDArray[NumpyDTVar]) -> NDArray[np.bool_]:
        r"""Return a tuple of masks."""
        return (bounds[0] <= self.data) & (self.data < bounds[-1])

    def make_masks(self, bounds: NDArray[NumpyDTVar]) -> list[NDArray[np.bool_], ...]:
        r"""Return a tuple of masks."""
        return [
            (start <= self.data) & (self.data < stop)
            for start, stop in sliding_window_view(bounds, 2)
        ]

    def make_window(self, bounds: NDArray[NumpyDTVar]) -> NDArray[NumpyDTVar]:
        r"""Return the actual data points inside the window."""
        return self.data[bounds[0] : bounds[-1]]

    def make_windows(self, bounds: NDArray[NumpyDTVar]) -> list[NDArray[NumpyDTVar]]:
        r"""Return the actual data points inside the windows."""
        return [self.data[start:stop] for start, stop in sliding_window_view(bounds, 2)]

    @property
    def make_key(self) -> Callable[[NDArray[NumpyDTVar]], Any]:
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
    def __iter__(self):  # pyright: ignore[reportGeneralTypeIssues]
        r"""Iterate through.

        For each k, we return either:

        - mode=points: $(x₀ + k⋅∆t, x₁+k⋅∆t, …, xₘ+k⋅∆t)$
        - mode=slices: $(slice(x₀ + k⋅∆t, x₁+k⋅∆t), …, slice(xₘ₋₁+k⋅∆t, xₘ+k⋅∆t))$
        - mode=masks: $(mask_1, …, mask_m)$
        """
        if self.shuffle:
            perm = np.random.permutation(len(self.grid))
            grid = self.grid[perm]
        else:
            grid = self.grid

        # unpack variables (avoid repeated lookups)
        t0 = self.initial_windows
        stride = self.stride
        make_key = self.make_key

        for k in grid:
            yield make_key(t0 + k * stride)


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
