r"""Sliding-window sampling for ordered, continuous time-series coordinates.

The sampler advances one or more half-open horizons along a timestamp-like axis
and can represent each resulting window in several useful formats.
"""

__all__ = [
    # types
    "MODE",
    "HORIZON",
    # Classes
    "SlidingWindowSampler",
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

from tsdm.constants import RNG
from tsdm.datatools import get_first_sample, get_last_sample, timedelta, timestamp
from tsdm.types import SupportsArray
from tsdm.types.abc import Vec
from tsdm.utils.interval import HalfOpenInterval, Interval

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
    r"""Return grid offsets that keep ``offset + k * step`` within the bounds.

    ``offset`` defaults to ``tmin`` and must lie within the closed interval
    ``[tmin, tmax]``. Positive and negative steps are both supported; a zero
    step raises :class:`ValueError`.

    Args:
        tmin: Inclusive lower bound of the coordinate range.
        tmax: Inclusive upper bound of the coordinate range.
        step: Distance between adjacent grid points.
        offset: Grid point assigned to index zero.
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
    r"""Representations available for each sampled window."""

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
    r"""Whether a sample contains one horizon or a sequence of horizons."""

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
    r"""Generate half-open time windows that slide across ordered coordinates.

    A scalar ``horizons`` value produces one window per grid position; a sequence
    produces contiguous horizons and yields one representation for each. Windows
    are closed on the left and open on the right. The ``mode`` chooses whether a
    window is returned as bounds, a slice, an interval, matching positions, their
    indices, or a boolean mask.

    Args:
        data_source: Ordered coordinates that delimit the sampling range.
        mode: Representation to yield for each horizon.
        horizons: Width of one horizon or widths of consecutive horizons.
        stride: Distance by which the complete window advances between samples.
        drop_last: Exclude a final window unless all of its horizons fit in range.
        shuffle: Whether to randomize the order of window positions.
        rng: Generator used when ``shuffle`` is enabled.
    """

    # NOTE: type checkers seem to break if we do not use 'TypeAlias' here.
    MODE: Final = MODE
    HORIZON: Final = HORIZON

    type Mode = Literal["slice", "mask", "bounds", "interval", "points", "index"]
    r"""Accepted string literals for selecting a window representation."""

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
            data_source: Vec[DT] | SupportsArray,
            /,
            *,
            mode: Literal["slice", MODE.SLICE],
            horizons: Vec[str] | Vec[TD],
            stride: str | TD, shuffle: bool = ..., drop_last: bool = ..., rng: Generator = ...,
        ) -> None: ...
        @overload
        def __init__[DT: TimeLikeScalar, TD: SpanLikeScalar](
            self: SlidingWindowSampler[DT, B, MULTI],
            data_source: Vec[DT] | SupportsArray,
            /,
            *,
            mode: Literal["bounds", MODE.BOUNDS],
            horizons: Vec[str] | Vec[TD],
            stride: str | TD, shuffle: bool = ..., drop_last: bool = ..., rng: Generator = ...,
        ) -> None: ...
        @overload
        def __init__[DT: TimeLikeScalar, TD: SpanLikeScalar](
            self: SlidingWindowSampler[DT, M, MULTI],
            data_source: Vec[DT] | SupportsArray,
            /,
            *,
            mode: Literal["mask", MODE.MASK],
            horizons: Vec[str] | Vec[TD],
            stride: str | TD, shuffle: bool = ..., drop_last: bool = ..., rng: Generator = ...,
        ) -> None: ...
        @overload
        def __init__[DT: TimeLikeScalar, TD: SpanLikeScalar](
            self: SlidingWindowSampler[DT, I, MULTI],
            data_source: Vec[DT] | SupportsArray,
            /,
            *,
            mode: Literal["interval", MODE.INTERVAL],
            horizons: Vec[str] | Vec[TD],
            stride: str | TD, shuffle: bool = ..., drop_last: bool = ..., rng: Generator = ...,
        ) -> None: ...
        @overload
        def __init__[DT: TimeLikeScalar, TD: SpanLikeScalar](
            self: SlidingWindowSampler[DT, X, MULTI],
            data_source: Vec[DT] | SupportsArray,
            /,
            *,
            mode: Literal["index", MODE.INDEX],
            horizons: Vec[str] | Vec[TD],
            stride: str | TD, shuffle: bool = ..., drop_last: bool = ..., rng: Generator = ...,
        ) -> None: ...
        @overload
        def __init__[DT: TimeLikeScalar, TD: SpanLikeScalar](
            self: SlidingWindowSampler[DT, P, MULTI],
            data_source: Vec[DT] | SupportsArray,
            /,
            *,
            mode: Literal["points", MODE.POINTS],
            horizons: Vec[str] | Vec[TD],
            stride: str | TD, shuffle: bool = ..., drop_last: bool = ..., rng: Generator = ...,
        ) -> None: ...
        @overload  # unknown mode
        def __init__[DT: TimeLikeScalar, TD: SpanLikeScalar](
            self: SlidingWindowSampler[DT, UNKNOWN, MULTI],
            data_source: Vec[DT] | SupportsArray,
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
            data_source: Vec[DT] | SupportsArray,
            /,
            *,
            mode: Literal["slice", MODE.SLICE],
            horizons: str | TD,
            stride: str | TD, shuffle: bool = ..., drop_last: bool = ..., rng: Generator = ...,
        ) -> None: ...
        @overload
        def __init__[DT: TimeLikeScalar, TD: SpanLikeScalar](
            self: SlidingWindowSampler[DT, B, ONE],
            data_source: Vec[DT] | SupportsArray,
            /,
            *,
            mode: Literal["bounds", MODE.BOUNDS],
            horizons: str | TD,
            stride: str | TD, shuffle: bool = ..., drop_last: bool = ..., rng: Generator = ...,
        ) -> None: ...
        @overload
        def __init__[DT: TimeLikeScalar, TD: SpanLikeScalar](
            self: SlidingWindowSampler[DT, M, ONE],
            data_source: Vec[DT] | SupportsArray,
            /,
            *,
            mode: Literal["mask", MODE.MASK],
            horizons: str | TD,
            stride: str | TD, shuffle: bool = ..., drop_last: bool = ..., rng: Generator = ...,
        ) -> None: ...
        @overload
        def __init__[DT: TimeLikeScalar, TD: SpanLikeScalar](
            self: SlidingWindowSampler[DT, X, ONE],
            data_source: Vec[DT] | SupportsArray,
            /,
            *,
            mode: Literal["index", MODE.INDEX],
            horizons: str | TD,
            stride: str | TD, shuffle: bool = ..., drop_last: bool = ..., rng: Generator = ...,
        ) -> None: ...
        @overload
        def __init__[DT: TimeLikeScalar, TD: SpanLikeScalar](
            self: SlidingWindowSampler[DT, P, ONE],
            data_source: Vec[DT] | SupportsArray,
            /,
            *,
            mode: Literal["points", MODE.POINTS],
            horizons: str | TD,
            stride: str | TD, shuffle: bool = ..., drop_last: bool = ..., rng: Generator = ...,
        ) -> None: ...
        @overload
        def __init__[DT: TimeLikeScalar, TD: SpanLikeScalar](
            self: SlidingWindowSampler[DT, UNKNOWN, ONE],
            data_source: Vec[DT] | SupportsArray,
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
        data_source: Vec[DT] | SupportsArray,
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
        self.tmin = cast("DType", get_first_sample(data_source))  # type: ignore
        self.tmax = cast("DType", get_last_sample(data_source))  # type: ignore
        zero_td = cast("Any", self.tmin - self.tmin)  # type: ignore
        dt_type: type[DType] = type(self.tmin)
        td_type: type[Any] = type(zero_td)
        self.data = np.array(data_source, dtype=dt_type)
        self.mode = self.MODE(mode)  # type: ignore
        self.drop_last = drop_last
        self.stride = timedelta(stride) if isinstance(stride, str) else stride

        if self.stride <= zero_td:
            raise ValueError("stride must be positive.")
        # endregion set basic attributes -----------------------------------------------

        # region set horizon(s) --------------------------------------------------------
        match horizons:
            # cast to timedelta and wrap in a numpy array
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
        r"""Integer offsets of the windows that are eligible for sampling."""
        # NOTE: we use a property so that if drop_last is changed, the grid is recomputed correctly...
        return np.array(
            compute_grid(
                self.tmin,
                self.tmax - self.cumulative_horizons[-1 if self.drop_last else -2],
                self.stride,
            )
        )

    def __len__(self) -> int:
        r"""Return the number of eligible window positions."""
        return len(self.grid)

    # region __iter__ overloads --------------------------------------------------------
    # fmt: off
    @overload
    def __iter__(self: SlidingWindowSampler[DType, S, MULTI], /) -> Iterator[list["slice[DType, DType]"]]: ...  # ruff: ignore[UP037]
    @overload
    def __iter__(self: SlidingWindowSampler[DType, B, MULTI], /) -> Iterator[list[tuple[DType, DType]]]: ...
    @overload
    def __iter__(self: SlidingWindowSampler[DType, I, MULTI], /) -> Iterator[list[HalfOpenInterval[DType]]]: ...
    @overload
    def __iter__(self: SlidingWindowSampler[DType, M, MULTI], /) -> Iterator[list[NDArray[np.bool_]]]: ...
    @overload
    def __iter__(self: SlidingWindowSampler[DType, X, MULTI], /) -> Iterator[list[NDArray[np.integer]]]: ...
    @overload
    def __iter__(self: SlidingWindowSampler[DType, P, MULTI], /) -> Iterator[list[NDArray]]: ...
    @overload  # fallback mode=str
    def __iter__(self: SlidingWindowSampler[DType, Any, MULTI], /) -> Iterator[list[Any]]: ...
    @overload
    def __iter__(self: SlidingWindowSampler[DType, S, ONE], /) -> Iterator["slice[DType, DType]"]: ...  # ruff: ignore[UP037]
    @overload
    def __iter__(self: SlidingWindowSampler[DType, B, ONE], /) -> Iterator[tuple[DType, DType]]: ...
    @overload
    def __iter__(self: SlidingWindowSampler[DType, I, ONE], /) -> Iterator[HalfOpenInterval[DType]]: ...
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
        r"""Yield the requested representation for each eligible time window.

        Multi-horizon samplers yield a list of adjacent window representations;
        single-horizon samplers yield that representation directly.
        """
        # unpack variables (avoids attribute lookup in loop)
        window = self.tmin + self.cumulative_horizons
        stride = self.stride
        grid = self.grid
        data = self.data
        sample_fns: dict[MODE, Callable] = {
            MODE.BOUNDS   : lambda start, stop: (start, stop),
            MODE.MASK     : lambda start, stop: (start <= data) & (data < stop),
            MODE.SLICE    : lambda start, stop: slice(start, stop),  # ruff: ignore[PLW0108]
            MODE.INTERVAL : lambda start, stop: Interval(start, stop, left_closed=True, right_closed=False),
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
