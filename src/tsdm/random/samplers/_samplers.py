r"""Random Samplers."""

__all__ = [
    # ABCs
    "BaseSampler",
    # Classes
    "RandomSampler",
    # "TimeSliceSampler",
    "SequenceSampler",
    "IntervalSampler",
    "HierarchicalSampler",
    "SlidingWindowSampler",
    # Functions
    "compute_grid",
]

import logging
import math
from abc import abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from datetime import timedelta as py_td
from itertools import chain, count
from typing import (
    Any,
    ClassVar,
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
from tsdm.utils.data.datasets import Dataset, IterableDataset, MapDataset
from tsdm.utils.strings import pprint_repr, repr_mapping


@runtime_checkable
class Sampler(Protocol[T_co]):
    r"""Protocol for `Sampler` classes."""

    @property
    @abstractmethod
    def shuffle(self) -> bool:
        """Whether to shuffle the data."""
        ...

    def __iter__(self) -> Iterator[T_co]:
        """Return an iterator over the indices of the data source."""
        ...

    def __len__(self) -> int:
        """The number of samples that can be drawn by __iter__."""
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


def get_index(dataset: Dataset[T_co], /) -> Index:
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


@pprint_repr
@dataclass
class RandomSampler(BaseSampler[T_co]):
    """Sampler randomly from the data source.

    Note:
        If the input is a DataFrame,
    """

    data: Dataset[T_co]

    _: KW_ONLY

    shuffle: bool = False

    def __post_init__(self) -> None:
        self.index: Index = get_index(self.data)
        self.size: int = len(self.index)

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            permutation = np.random.permutation(self.size)
            for key in self.index[permutation]:
                yield self.data[key]
            return

        # shortcut: iterate over index directly
        for key in self.index:
            yield self.data[key]

    def __len__(self) -> int:
        return self.size


class HierarchicalSampler(BaseSampler[tuple[K, T_co]]):
    r"""Samples a single random dataset from a collection of datasets.

    Optionally, we can delegate a subsampler to then sample from the randomly drawn dataset.
    """

    index: Index
    r"""The shared index."""
    subsamplers: Mapping[K, Sampler[T_co]]
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
        data_source: Mapping[K, T],
        /,
        subsamplers: Mapping[K, Sampler[T_co]],
        *,
        shuffle: bool = False,
        early_stop: bool = False,
    ) -> None:
        super().__init__(shuffle=shuffle)
        self.data = data_source
        self.early_stop = early_stop
        self.index = get_index(data_source)
        self.subsamplers = dict(subsamplers)

        self.sizes = Series({key: len(self.subsamplers[key]) for key in self.index})

        if early_stop:  # duplicate keys to match the minimum subsampler size
            self.partition = Series(
                chain(*([key] * min(self.sizes) for key in self.index))
            )
        else:  # duplicate keys to match each sub-sampler's size
            self.partition = Series(
                chain(*([key] * self.sizes[key] for key in self.index))
            )

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
    td = cast(TDVar, Timedelta(timedelta) if isinstance(timedelta, str) else timedelta)

    offset = cast(
        DTVar,
        (
            tmin
            if offset is None
            else Timestamp(offset) if isinstance(offset, str) else offset
        ),
    )

    # generates zero-variable of the correct type.
    zero_dt = tmin - tmin

    if td == zero_dt:
        raise ValueError("Δt=0 is not allowed!")
    if tmin > offset or offset > tmax:
        raise ValueError("tₘᵢₙ ≤ t₀ ≤ tₘₐₓ violated!")

    kmax = math.floor(cast(TDVar, tmax - offset) / td)  # type: ignore[redundant-cast]
    kmin = math.ceil(cast(TDVar, tmin - offset) / td)  # type: ignore[redundant-cast]

    return cast(Sequence[int], np.arange(kmin, kmax + 1))


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

            intervals.extend(
                [(x0 + i * st, x0 + i * st + dt, dt, st) for i in valid_strides]
            )

        # set variables
        self.offset = cast(TDVar, offset)  # type: ignore[redundant-cast]
        self.deltax = deltax
        self.stride = stride
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
        return getattr(self.intervals, key)

    def __getitem__(self, key: int) -> slice:
        r"""Return a slice from the sampler."""
        return self.intervals[key]


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

        self.samples = np.array(
            [
                (
                    (x <= self.data) & (self.data < y)  # type: ignore[operator]
                    if self.return_mask
                    else [x, y]
                )
                for x, y in self._iter_tuples()
            ]
        )

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
    start_values: NDArray[NumpyDTVar]
    offset: NumpyDTVar
    total_horizon: NumpyTDVar
    zero_td: NumpyTDVar
    multi_horizon: bool
    cumulative_horizons: NDArray[NumpyTDVar]

    shuffle: bool = False

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
        super().__init__(shuffle=shuffle)
        self.data = np.asarray(data_source)
        self.mode = mode
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

        # convert horizons to timedelta
        horizons = Timedelta(horizons) if isinstance(horizons, str) else horizons
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

    @property
    def make_key(self) -> Callable[[NDArray], Any]:
        r"""Return the correct yield function."""
        match self.mode, self.multi_horizon:
            case "points", _:
                return self.__make__points__
            case "masks", False:
                return self.__make__mask__
            case "masks", True:
                return self.__make__masks__
            case "slices", False:
                return self.__make__slice__
            case "slices", True:
                return self.__make__slices__
            case _:
                raise ValueError(f"Invalid mode {self.mode=}")

    @overload
    def __iter__(
        self: "SlidingWindowSampler[Literal['slices'], NumpyDTVar, NumpyTDVar]",
    ) -> Iterator[slice] | Iterator[tuple[slice, ...]]: ...
    @overload
    def __iter__(
        self: "SlidingWindowSampler[Literal['masks'], NumpyDTVar, NumpyTDVar]",
    ) -> Iterator[NDArray[np.bool_]] | Iterator[tuple[NDArray[np.bool_], ...]]: ...
    @overload
    def __iter__(
        self: "SlidingWindowSampler[Literal['points'], NumpyDTVar, NumpyTDVar]",
    ) -> Iterator[NDArray[NumpyDTVar]]: ...
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
        t0 = self.start_values
        stride = self.stride
        make_key = self.make_key

        for k in grid:
            vals = t0 + k * stride
            yield make_key(vals)
