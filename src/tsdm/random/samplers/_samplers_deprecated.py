r"""Deprecated samplers."""

__all__ = ["IntervalSampler", "SequenceSampler"]

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from itertools import count

import numpy as np
from numpy._typing import NDArray
from pandas import DataFrame
from typing_extensions import Any, Generic, Optional, cast, deprecated

from tsdm.constants import RNG
from tsdm.random.samplers.base import BaseSampler, compute_grid
from tsdm.types.protocols import Lookup, SupportsLenAndGetItem
from tsdm.types.time import DTVar, TDVar
from tsdm.utils import timedelta, timestamp


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
            stride_left: list[int] = compute_grid(xmin, xmax, st, offset=x0)  # type: ignore[misc]
            stride_right: list[int] = compute_grid(xmin, xmax, st, offset=x0 + dt)  # type: ignore[misc]
            valid_strides: set[int] = set.intersection(
                set(stride_left), set(stride_right)
            )

            if not valid_strides:
                break

            intervals.extend([  # pyright: ignore[reportArgumentType]
                (x0 + i * st, x0 + i * st + dt, dt, st) for i in valid_strides
            ])

        # set variables
        self.offset = cast(TDVar, offset)  # type: ignore[redundant-cast]
        self.deltax = deltax
        self.stride = stride  # pyright: ignore[reportAttributeAccessIssue]
        self.intervals = DataFrame(
            intervals,
            columns=["lower_bound", "upper_bound", "delta", "stride"],
        )

    def __iter__(self) -> Iterator[slice]:
        r"""Return an iterator over the intervals."""
        # assign to avoid attribute lookup in loop
        lower_bounds = self.intervals["lower_bound"]
        upper_bounds = self.intervals["upper_bound"]

        n = len(self)
        index = RNG.permutation(n) if self.shuffle else np.arange(n)

        for k in index:
            yield slice(lower_bounds[k], upper_bounds[k])

    def __len__(self) -> int:
        r"""Length of the sampler."""
        return len(self.intervals)

    def __getattr__(self, key: str) -> Any:
        r"""Forward all other attributes to the interval frame."""
        return getattr(self.intervals, key)

    def __getitem__(self, key: int, /) -> slice:
        r"""Return a slice from the sampler."""
        return self.intervals[key]


@deprecated("Use SlidingWindowSampler instead.")
class SequenceSampler(BaseSampler, Generic[DTVar, TDVar]):
    r"""Samples sequences of fixed length."""

    data: NDArray[DTVar]  # type: ignore[type-var]
    seq_len: TDVar
    r"""The length of the sequences."""
    stride: TDVar
    r"""The stride at which to sample."""
    xmax: DTVar
    r"""The maximum value at which to stop sampling."""
    xmin: DTVar
    r"""The minimum value at which to start sampling."""
    return_mask: bool = False
    r"""Whether to return masks instead of indices."""
    shuffle: bool = False
    r"""Whether to shuffle the data."""

    def __init__(
        self,
        data_source: Iterable[DTVar] | SupportsLenAndGetItem[DTVar],
        /,
        *,
        return_mask: bool = False,
        seq_len: str | TDVar,
        shuffle: bool = False,
        stride: str | TDVar,
        tmin: Optional[DTVar] = None,
        tmax: Optional[DTVar] = None,
    ) -> None:
        super().__init__(shuffle=shuffle)
        self.data = np.asarray(data_source)

        match tmin:
            case None:
                self.xmin = self.data[0]
            case str() as time_str:
                self.xmin = timestamp(time_str)
            case _:
                self.xmin = tmin  # pyright: ignore[reportAttributeAccessIssue]

        match tmax:
            case None:
                self.xmax = self.data[-1]
            case str() as time_str:
                self.xmax = timestamp(time_str)
            case _:
                self.xmax = tmax  # pyright: ignore[reportAttributeAccessIssue]

        total_delta = cast(TDVar, self.xmax - self.xmin)  # type: ignore[redundant-cast]
        self.stride = cast(
            TDVar, timedelta(stride) if isinstance(stride, str) else stride
        )
        self.seq_len = cast(
            TDVar, timedelta(seq_len) if isinstance(seq_len, str) else seq_len
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
        x, y = min(x, y), max(x, y)
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
        n = len(self)
        index = RNG.permutation(n) if self.shuffle else np.arange(n)
        return iter(self.samples[index])

    def __repr__(self) -> str:
        r"""Return a string representation of the object."""
        return f"{self.__class__.__name__}[{self.stride}, {self.seq_len}]"
