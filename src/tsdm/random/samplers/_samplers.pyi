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
    def __len__(self) -> int: ...
    @staticmethod
    def make_bound(bounds: NDArray[NumpyDTVar]) -> tuple[NumpyDTVar, NumpyTDVar]: ...
    @staticmethod
    def make_bounds(
        bounds: NDArray[NumpyDTVar],
    ) -> list[tuple[NumpyDTVar, NumpyTDVar]]: ...
    @staticmethod
    def make_slice(bounds: NDArray[NumpyDTVar]) -> slice: ...
    @staticmethod
    def make_slices(bounds: NDArray[NumpyDTVar]) -> list[slice]: ...
    def make_mask(self, bounds: NDArray[NumpyDTVar]) -> NDArray[np.bool_]: ...
    def make_masks(
        self, bounds: NDArray[NumpyDTVar]
    ) -> list[NDArray[np.bool_], ...]: ...
    def make_window(self, bounds: NDArray[NumpyDTVar]) -> NDArray[NumpyDTVar]: ...
    def make_windows(
        self, bounds: NDArray[NumpyDTVar]
    ) -> list[NDArray[NumpyDTVar]]: ...
    @property
    def make_key(self) -> Callable[[NDArray[NumpyDTVar]], Any]: ...
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
