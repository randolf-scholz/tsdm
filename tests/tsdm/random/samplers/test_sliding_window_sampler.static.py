r"""Test Sliding Window Sampler."""
# FIXME: https://github.com/python/mypy/pull/16020
# mypy: ignore-errors

import logging
from typing import Literal, assert_type

from tsdm.random.samplers import SlidingWindowSampler

__logger__ = logging.getLogger(__name__)
Y = True
N = False
# type S = Literal["slices"]  # slice
# type M = Literal["masks"]  # bool
# type B = Literal["bounds"]  # tuple
# type W = Literal["windows"]  # windows (list)
# type U = str  # unknown (not statically known)
MODES = SlidingWindowSampler.MODE
type B = Literal[MODES.B]  # -> tuple[DT, DT]
type M = Literal[MODES.M]  # -> array[bool]
type S = Literal[MODES.S]  # -> slice[DT, DT]
type I = Literal[MODES.I]  # -> interval[DT]
type T = Literal[MODES.T]  # -> array[DT]
type X = Literal[MODES.X]  # -> array[int]
type U = str  # fallback
type ONE = Literal["one"]
type MULTI = Literal["multi"]


def static_test() -> None:
    int_list: list[int] = [1, 2, 3]

    assert_type(
        SlidingWindowSampler(int_list, horizons=2, stride=2, mode=MODES.S),
        SlidingWindowSampler[int, S, ONE],
    )
    assert_type(
        SlidingWindowSampler(int_list, horizons=[1, 2], stride=2, mode=MODES.M),
        SlidingWindowSampler[int, M, MULTI],
    )
