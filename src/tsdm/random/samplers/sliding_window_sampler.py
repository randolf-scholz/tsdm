r"""Implementation of a sliding window sampler for continuous time series data."""

__all__ = [
    # types
    "MODE",
    "HORIZON",
    # Classes
    "SlidingWindowSampler",
    "RandomWindowSampler",
    # Functions
    "compute_grid",
]


import datetime as dt
from collections.abc import Callable, Iterable, Iterator
from enum import StrEnum
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    Optional,
    cast,
    overload,
)

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.random import Generator
from numpy.typing import NDArray
from pandas import Interval

from tsdm.constants import RNG
from tsdm.datatools.collections import (
    SequentialDataset,
    get_first_sample,
    get_last_sample,
)
from tsdm.types.abc import Vec
from tsdm.utils import timedelta, timestamp

from .base import BaseSampler

# TODO: consider using numerical_types.scalars.{SpanLikeScalar, TimeLikeScalar}
type TimeLikeScalar[SpanT] = Any
type SpanLikeScalar = Any


# region helper functions --------------------------------------------------------------
def compute_grid[TD: SpanLikeScalar](
    tmin: str | TimeLikeScalar[TD],
    tmax: str | TimeLikeScalar[TD],
    step: str | TD,
    /,
    *,
    offset: Optional[str | TimeLikeScalar[TD]] = None,
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
    t_min = cast("Any", timestamp(tmin) if isinstance(tmin, str) else tmin)
    t_max = cast("Any", timestamp(tmax) if isinstance(tmax, str) else tmax)
    t_0 = cast("Any", timestamp(offset) if isinstance(offset, str) else offset)
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


class MODE(StrEnum):
    r"""Valid modes for the sampler, determining the return format."""

    # fmt: off
    BOUNDS   = "bounds"    # -> tuple[DT, DT]  (equivalent to half open interval [l, u))
    INDEX    = "index"     # -> array[int]
    INTERVAL = "interval"  # -> interval[DT]
    MASK     = "mask"      # -> array[bool]
    POINTS   = "points"    # -> array[DT]
    SLICE    = "slice"     # -> slice[DT, DT]  (equivalent to half open interval [l, u))
    # fmt: on


type B = Literal[MODE.BOUNDS]
type M = Literal[MODE.MASK]
type S = Literal[MODE.SLICE]
type I = Literal[MODE.INTERVAL]
type P = Literal[MODE.POINTS]
type X = Literal[MODE.INDEX]
type UNKNOWN = MODE


class HORIZON(StrEnum):
    r"""Valid horizon types for the sampler."""

    ONE = "one"  # -> single horizon
    MULTI = "multi"  # -> multiple horizons


type ONE = Literal[HORIZON.ONE]
type MULTI = Literal[HORIZON.MULTI]


# FIXME: Allow ±∞ as bounds for timedelta types? This would allow "growing" windows.
class SlidingWindowSampler[
    DType: int | float | dt.date | dt.datetime | dt.timedelta,
    ModeVar: (B, M, S, I, P, X, UNKNOWN),
    MultiVar: (ONE, MULTI),
](BaseSampler):
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

    # NOTE: type checkers seem to break if we do not use 'TypeAlias' here.
    MODE: Final = MODE
    HORIZON: Final = HORIZON

    type Mode = Literal["slice", "mask", "bounds", "interval", "points", "index"]
    r"""Type hint for the mode."""

    data: NDArray

    size: SpanLikeScalar
    stride: SpanLikeScalar
    mode: ModeVar
    multi_horizon: bool
    shuffle: bool
    drop_last: bool
    rng: Generator

    # dependent variables
    tmin: DType
    tmax: DType
    cumulative_horizons: NDArray

    if TYPE_CHECKING:
        # region __new__ overloads -----------------------------------------------------
        # @overload  # single horizon ----------------------------------------------------
        # def __new__[DT: TimeStamp, TD: TimeDelta, _Mode: (B, M, S, I, T, X)](
        #     cls,
        #     data_source: SequentialDataset[DT],
        #     /,
        #     *,
        #     mode: _Mode,
        #     horizons: str | TD,
        #     stride: str | TD,
        #     shuffle: bool = ...,
        #     drop_last: bool = ...,
        #     rng: Generator = ...,
        # ) -> "SlidingWindowSampler[DT, _Mode, ONE]": ...
        # @overload  # multi-horizon -----------------------------------------------------
        # def __new__[DT: TimeStamp, TD: TimeDelta, _Mode: (B, M, S, I, T, X)](
        #     cls,
        #     data_source: SequentialDataset[DT],
        #     /,
        #     *,
        #     mode: _Mode,
        #     horizons: Array[str | TD],
        #     stride: str | TD,
        #     shuffle: bool = ...,
        #     drop_last: bool = ...,
        #     rng: Generator = ...,
        # ) -> "SlidingWindowSampler[DT, _Mode, MULTI]": ...
        # @overload  # single horizon ----------------------------------------------------
        # def __new__[DT: TimeStamp, TD: TimeDelta](
        #     cls,
        #     data_source: SequentialDataset[DT],
        #     /,
        #     *,
        #     mode: str,
        #     horizons: str | TD,
        #     stride: str | TD,
        #     shuffle: bool = ...,
        #     drop_last: bool = ...,
        #     rng: Generator = ...,
        # ) -> "SlidingWindowSampler[DT, U, ONE]": ...
        # @overload  # multi-horizon -----------------------------------------------------
        # def __new__[DT: TimeStamp, TD: TimeDelta](
        #     cls,
        #     data_source: SequentialDataset[DT],
        #     /,
        #     *,
        #     mode: str,
        #     horizons: Array[str | TD],
        #     stride: str | TD,
        #     shuffle: bool = ...,
        #     drop_last: bool = ...,
        #     rng: Generator = ...,
        # ) -> "SlidingWindowSampler[DT, U, MULTI]": ...

        # @overload  # multi-horizon -----------------------------------------------------
        # def __new__[DT: TimeStamp, TD: TimeDelta](
        #     cls,
        #     data_source: SequentialDataset[DT],
        #     /,
        #     *,
        #     mode: Literal["slice", MODE.S],
        #     horizons: Array[str | TD],
        #     stride: str | TD,
        #     shuffle: bool = ...,
        #     drop_last: bool = ...,
        #     rng: Generator = ...,
        # ) -> "SlidingWindowSampler[DT, S, MULTI]": ...
        # @overload
        # def __new__[DT: TimeStamp, TD: TimeDelta](
        #     cls,
        #     data_source: SequentialDataset[DT],
        #     /,
        #     *,
        #     mode: Literal["bound", MODE.B],
        #     horizons: Array[str | TD],
        #     stride: str | TD,
        #     shuffle: bool = ...,
        #     drop_last: bool = ...,
        #     rng: Generator = ...,
        # ) -> "SlidingWindowSampler[DT, B, MULTI]": ...
        # @overload
        # def __new__[DT: TimeStamp, TD: TimeDelta](
        #     cls,
        #     data_source: SequentialDataset[DT],
        #     /,
        #     *,
        #     mode: Literal["mask", MODE.M],
        #     horizons: Array[str | TD],
        #     stride: str | TD,
        #     shuffle: bool = ...,
        #     drop_last: bool = ...,
        #     rng: Generator = ...,
        # ) -> "SlidingWindowSampler[DT, M, MULTI]": ...
        # @overload
        # def __new__[DT: TimeStamp, TD: TimeDelta](
        #     cls,
        #     data_source: SequentialDataset[DT],
        #     /,
        #     *,
        #     mode: Literal["interval", MODE.I],
        #     horizons: Array[str | TD],
        #     stride: str | TD,
        #     shuffle: bool = ...,
        #     drop_last: bool = ...,
        #     rng: Generator = ...,
        # ) -> "SlidingWindowSampler[DT, I, MULTI]": ...
        # @overload
        # def __new__[DT: TimeStamp, TD: TimeDelta](
        #     cls,
        #     data_source: SequentialDataset[DT],
        #     /,
        #     *,
        #     mode: Literal["index", MODE.X],
        #     horizons: Array[str | TD],
        #     stride: str | TD,
        #     shuffle: bool = ...,
        #     drop_last: bool = ...,
        #     rng: Generator = ...,
        # ) -> "SlidingWindowSampler[DT, X, MULTI]": ...
        # @overload
        # def __new__[DT: TimeStamp, TD: TimeDelta](
        #     cls,
        #     data_source: SequentialDataset[DT],
        #     /,
        #     *,
        #     mode: Literal["timestamp", MODE.T],
        #     horizons: Array[str | TD],
        #     stride: str | TD,
        #     shuffle: bool = ...,
        #     drop_last: bool = ...,
        #     rng: Generator = ...,
        # ) -> "SlidingWindowSampler[DT, T, MULTI]": ...
        # @overload  # unknown mode
        # def __new__[DT: TimeStamp, TD: TimeDelta](
        #     cls,
        #     data_source: SequentialDataset[DT],
        #     /,
        #     *,
        #     mode: MODE | Mode | str,
        #     horizons: Array[str | TD],
        #     stride: str | TD,
        #     shuffle: bool = ...,
        #     drop_last: bool = ...,
        #     rng: Generator = ...,
        # ) -> "SlidingWindowSampler[DT, U, MULTI]": ...
        # @overload  # single horizon ----------------------------------------------------
        # def __new__[DT: TimeStamp, TD: TimeDelta](
        #     cls,
        #     data_source: SequentialDataset[DT],
        #     /,
        #     *,
        #     mode: Literal["slice", MODE.S],
        #     horizons: str | TD,
        #     stride: str | TD,
        #     shuffle: bool = ...,
        #     drop_last: bool = ...,
        #     rng: Generator = ...,
        # ) -> "SlidingWindowSampler[DT, S, ONE]": ...
        # @overload
        # def __new__[DT: TimeStamp, TD: TimeDelta](
        #     cls,
        #     data_source: SequentialDataset[DT],
        #     /,
        #     *,
        #     mode: Literal["bounds", MODE.B],
        #     horizons: str | TD,
        #     stride: str | TD,
        #     shuffle: bool = ...,
        #     drop_last: bool = ...,
        #     rng: Generator = ...,
        # ) -> "SlidingWindowSampler[DT, B, ONE]": ...
        # @overload
        # def __new__[DT: TimeStamp, TD: TimeDelta](
        #     cls,
        #     data_source: SequentialDataset[DT],
        #     /,
        #     *,
        #     mode: Literal["mask", MODE.M],
        #     horizons: str | TD,
        #     stride: str | TD,
        #     shuffle: bool = ...,
        #     drop_last: bool = ...,
        #     rng: Generator = ...,
        # ) -> "SlidingWindowSampler[DT, M, ONE]": ...
        # @overload
        # def __new__[DT: TimeStamp, TD: TimeDelta](
        #     cls,
        #     data_source: SequentialDataset[DT],
        #     /,
        #     *,
        #     mode: Literal["index", MODE.X],
        #     horizons: str | TD,
        #     stride: str | TD,
        #     shuffle: bool = ...,
        #     drop_last: bool = ...,
        #     rng: Generator = ...,
        # ) -> "SlidingWindowSampler[DT, X, ONE]": ...
        # @overload
        # def __new__[DT: TimeStamp, TD: TimeDelta](
        #     cls,
        #     data_source: SequentialDataset[DT],
        #     /,
        #     *,
        #     mode: Literal["timestamp", MODE.T],
        #     horizons: str | TD,
        #     stride: str | TD,
        #     shuffle: bool = ...,
        #     drop_last: bool = ...,
        #     rng: Generator = ...,
        # ) -> "SlidingWindowSampler[DT, T, ONE]": ...
        # @overload  # fallback mode=str
        # def __new__[DT: TimeStamp, TD: TimeDelta](
        #     cls,
        #     data_source: SequentialDataset[DT],
        #     /,
        #     *,
        #     mode: MODE | Mode | str,
        #     horizons: str | TD,
        #     stride: str | TD,
        #     shuffle: bool = ...,
        #     drop_last: bool = ...,
        #     rng: Generator = ...,
        # ) -> "SlidingWindowSampler[DT, U, ONE]": ...
        # fmt: on
        # endregion __new__ overloads --------------------------------------------------

        # region __init__ overloads ----------------------------------------------------
        # fmt: off
        # HORIZON=MULTI ----------------------------------------------------------------
        @overload
        def __init__[DT: TimeLikeScalar, TD: SpanLikeScalar](
            self: SlidingWindowSampler[DT, S, MULTI],
            data_source: SequentialDataset[DT],
            /,
            *,
            mode: Literal["slice", MODE.SLICE],
            horizons: Vec[str] | Vec[TD],
            stride: str | TD, shuffle: bool = ..., drop_last: bool = ..., rng: Generator = ...,
        ) -> None: ...
        @overload
        def __init__[DT: TimeLikeScalar, TD: SpanLikeScalar](
            self: SlidingWindowSampler[DT, B, MULTI],
            data_source: SequentialDataset[DT],
            /,
            *,
            mode: Literal["bounds", MODE.BOUNDS],
            horizons: Vec[str] | Vec[TD],
            stride: str | TD, shuffle: bool = ..., drop_last: bool = ..., rng: Generator = ...,
        ) -> None: ...
        @overload
        def __init__[DT: TimeLikeScalar, TD: SpanLikeScalar](
            self: SlidingWindowSampler[DT, M, MULTI],
            data_source: SequentialDataset[DT],
            /,
            *,
            mode: Literal["mask", MODE.MASK],
            horizons: Vec[str] | Vec[TD],
            stride: str | TD, shuffle: bool = ..., drop_last: bool = ..., rng: Generator = ...,
        ) -> None: ...
        @overload
        def __init__[DT: TimeLikeScalar, TD: SpanLikeScalar](
            self: SlidingWindowSampler[DT, I, MULTI],
            data_source: SequentialDataset[DT],
            /,
            *,
            mode: Literal["interval", MODE.INTERVAL],
            horizons: Vec[str] | Vec[TD],
            stride: str | TD, shuffle: bool = ..., drop_last: bool = ..., rng: Generator = ...,
        ) -> None: ...
        @overload
        def __init__[DT: TimeLikeScalar, TD: SpanLikeScalar](
            self: SlidingWindowSampler[DT, X, MULTI],
            data_source: SequentialDataset[DT],
            /,
            *,
            mode: Literal["index", MODE.INDEX],
            horizons: Vec[str] | Vec[TD],
            stride: str | TD, shuffle: bool = ..., drop_last: bool = ..., rng: Generator = ...,
        ) -> None: ...
        @overload
        def __init__[DT: TimeLikeScalar, TD: SpanLikeScalar](
            self: SlidingWindowSampler[DT, P, MULTI],
            data_source: SequentialDataset[DT],
            /,
            *,
            mode: Literal["points", MODE.POINTS],
            horizons: Vec[str] | Vec[TD],
            stride: str | TD, shuffle: bool = ..., drop_last: bool = ..., rng: Generator = ...,
        ) -> None: ...
        @overload  # unknown mode
        def __init__[DT: TimeLikeScalar, TD: SpanLikeScalar](
            self: SlidingWindowSampler[DT, UNKNOWN, MULTI],
            data_source: SequentialDataset[DT],
            /,
            *,
            mode: MODE | Mode | str,
            horizons: Vec[str] | Vec[TD],
            stride: str | TD, shuffle: bool = ..., drop_last: bool = ..., rng: Generator = ...,
        ) -> None: ...
        # HORIZON=ONE ------------------------------------------------------------------
        @overload
        def __init__[DT: TimeLikeScalar, TD: SpanLikeScalar](
            self: SlidingWindowSampler[DT, S, ONE],
            data_source: SequentialDataset[DT],
            /,
            *,
            mode: Literal["slice", MODE.SLICE],
            horizons: str | TD,
            stride: str | TD, shuffle: bool = ..., drop_last: bool = ..., rng: Generator = ...,
        ) -> None: ...
        @overload
        def __init__[DT: TimeLikeScalar, TD: SpanLikeScalar](
            self: SlidingWindowSampler[DT, B, ONE],
            data_source: SequentialDataset[DT],
            /,
            *,
            mode: Literal["bounds", MODE.BOUNDS],
            horizons: str | TD,
            stride: str | TD, shuffle: bool = ..., drop_last: bool = ..., rng: Generator = ...,
        ) -> None: ...
        @overload
        def __init__[DT: TimeLikeScalar, TD: SpanLikeScalar](
            self: SlidingWindowSampler[DT, M, ONE],
            data_source: SequentialDataset[DT],
            /,
            *,
            mode: Literal["mask", MODE.MASK],
            horizons: str | TD,
            stride: str | TD, shuffle: bool = ..., drop_last: bool = ..., rng: Generator = ...,
        ) -> None: ...
        @overload
        def __init__[DT: TimeLikeScalar, TD: SpanLikeScalar](
            self: SlidingWindowSampler[DT, X, ONE],
            data_source: SequentialDataset[DT],
            /,
            *,
            mode: Literal["index", MODE.INDEX],
            horizons: str | TD,
            stride: str | TD, shuffle: bool = ..., drop_last: bool = ..., rng: Generator = ...,
        ) -> None: ...
        @overload
        def __init__[DT: TimeLikeScalar, TD: SpanLikeScalar](
            self: SlidingWindowSampler[DT, P, ONE],
            data_source: SequentialDataset[DT],
            /,
            *,
            mode: Literal["points", MODE.POINTS],
            horizons: str | TD,
            stride: str | TD, shuffle: bool = ..., drop_last: bool = ..., rng: Generator = ...,
        ) -> None: ...
        @overload
        def __init__[DT: TimeLikeScalar, TD: SpanLikeScalar](
            self: SlidingWindowSampler[DT, UNKNOWN, ONE],
            data_source: SequentialDataset[DT],
            /,
            *,
            mode: MODE | Mode | str,
            horizons: str | TD,
            stride: str | TD, shuffle: bool = ..., drop_last: bool = ..., rng: Generator = ...,
        ) -> None: ...
        # fmt: on
        # endregion __init__  overloads ------------------------------------------------

    def __init__[DT: TimeLikeScalar, TD: SpanLikeScalar](
        self,
        data_source: SequentialDataset[DT],
        /,
        *,
        mode: ModeVar | Mode | str,
        horizons: str | TD | Vec[str] | Vec[TD],
        stride: str | TD,
        drop_last: bool = False,
        shuffle: bool = False,
        rng: Generator = RNG,
    ) -> None:
        super().__init__(shuffle=shuffle, rng=rng)

        # region set basic attributes --------------------------------------------------
        self.tmin = cast("DType", get_first_sample(data_source))
        self.tmax = cast("DType", get_last_sample(data_source))
        zero_td = cast("Any", self.tmin - self.tmin)  # type: ignore[operator]
        dt_type: type[DType] = type(self.tmin)
        td_type: type[Any] = type(zero_td)
        self.data = np.array(data_source, dtype=dt_type)
        self.mode = self.MODE(mode)  # type: ignore[assignment]
        self.drop_last = drop_last
        self.stride = timedelta(stride) if isinstance(stride, str) else stride

        if self.stride <= zero_td:
            raise ValueError("stride must be positive.")
        # endregion set basic attributes -----------------------------------------------

        # region set horizon(s) --------------------------------------------------------
        match horizons:
            # cast to pandas.Timedelta and wrap in a numpy array
            case str(unit):
                self.multi_horizon = False
                self.horizons = np.array([timedelta(unit)], dtype=td_type)
            case Iterable() as vals:
                self.multi_horizon = True
                self.horizons = np.array(
                    [(timedelta(td) if isinstance(td, str) else td) for td in vals],
                    dtype=td_type,
                )
            case td_scalar:
                self.multi_horizon = False
                self.horizons = np.array([td_scalar], dtype=td_type)

        zero = np.array([zero_td], dtype=td_type)
        self.cumulative_horizons = np.cumsum(np.concatenate([zero, self.horizons]))
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

    # region __iter__ overloads --------------------------------------------------------
    # fmt: off
    @overload
    def __iter__(self: SlidingWindowSampler[DType, S, MULTI], /) -> Iterator[list["slice[DType, DType]"]]: ...  # noqa: UP037
    @overload
    def __iter__(self: SlidingWindowSampler[DType, B, MULTI], /) -> Iterator[list[tuple[DType, DType]]]: ...
    @overload
    def __iter__(self: SlidingWindowSampler[DType, I, MULTI], /) -> Iterator[list[Interval[DType]]]: ...  # pyright: ignore[reportInvalidTypeArguments]
    @overload
    def __iter__(self: SlidingWindowSampler[DType, M, MULTI], /) -> Iterator[list[NDArray[np.bool_]]]: ...
    @overload
    def __iter__(self: SlidingWindowSampler[DType, X, MULTI], /) -> Iterator[list[NDArray[np.integer]]]: ...
    @overload
    def __iter__(self: SlidingWindowSampler[DType, P, MULTI], /) -> Iterator[list[NDArray]]: ...
    @overload  # fallback mode=str
    def __iter__(self: SlidingWindowSampler[DType, Any, MULTI], /) -> Iterator[list[Any]]: ...
    @overload
    def __iter__(self: SlidingWindowSampler[DType, S, ONE], /) -> Iterator["slice[DType, DType]"]: ...  # noqa: UP037
    @overload
    def __iter__(self: SlidingWindowSampler[DType, B, ONE], /) -> Iterator[tuple[DType, DType]]: ...
    @overload
    def __iter__(self: SlidingWindowSampler[DType, I, ONE], /) -> Iterator[Interval[DType]]: ...  # pyright: ignore[reportInvalidTypeArguments]
    @overload
    def __iter__(self: SlidingWindowSampler[DType, X, ONE], /) -> Iterator[NDArray[np.integer]]: ...
    @overload
    def __iter__(self: SlidingWindowSampler[DType, M, ONE], /) -> Iterator[NDArray[np.bool_]]: ...
    @overload
    def __iter__(self: SlidingWindowSampler[DType, P, ONE], /) -> Iterator[NDArray]: ...
    @overload  # fallback mode=str
    def __iter__(self: SlidingWindowSampler[DType, Any, ONE], /) -> Iterator[Any]: ...
    @overload  # fallback
    def __iter__(self: SlidingWindowSampler[DType, Any, Any], /) -> Iterator[Any]: ...
    # fmt: on
    # endregion __iter__ overloads -----------------------------------------------------
    def __iter__(self, /) -> Iterator[Any]:
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
        data = self.data
        sample_fns: dict[MODE, Callable] = {
            MODE.BOUNDS   : lambda start, stop: (start, stop),
            MODE.MASK     : lambda start, stop: (start <= data) & (data < stop),
            MODE.SLICE    : lambda start, stop: slice(start, stop),  # noqa: PLW0108
            MODE.INTERVAL : lambda start, stop: Interval(start, stop, closed="left"),
            MODE.POINTS   : lambda start, stop: data[(start <= data) & (data < stop)],
            MODE.INDEX    : lambda start, stop: np.where((start <= data) & (data < stop))[0],
        }  # fmt: skip
        sample_fn = sample_fns[self.mode]

        if self.shuffle:
            grid = grid[self.rng.permutation(len(grid))]

        if self.multi_horizon:
            for horizons in (window + k * stride for k in grid):
                yield [
                    sample_fn(start, stop)
                    for start, stop in sliding_window_view(horizons, 2)
                ]
        else:
            for horizons in (window + k * stride for k in grid):
                yield sample_fn(horizons[0], horizons[-1])

    def __len__(self) -> int:
        r"""Return the number of samples."""
        return len(self.grid)


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
