r"""Test Sliding Window Sampler."""

import datetime
from collections.abc import Iterator
from typing import Any, Literal, assert_never, assert_type

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from tests.test_utils import pytest_xfail
from tsdm.constants import RNG
from tsdm.datatools.collections import Indexable
from tsdm.random.samplers import SlidingWindowSampler

MODE = SlidingWindowSampler.MODE
type B = Literal[MODE.BOUNDS]  # -> tuple[DT, DT]
type M = Literal[MODE.MASK]  # -> array[bool]
type S = Literal[MODE.SLICE]  # -> slice[DT, DT]
type I = Literal[MODE.INTERVAL]  # -> interval[DT]
type P = Literal[MODE.POINTS]  # -> array[DT]
type X = Literal[MODE.INDEX]  # -> array[int]
HORIZON = SlidingWindowSampler.HORIZON
type ONE = Literal[HORIZON.ONE]
type MULTI = Literal[HORIZON.MULTI]
type Mode = Literal[
    "bound",
    "mask",
    "slice",
    "interval",
    "timestamp",
    "index",
]

Y = True
N = False

DISCRETE_DATA = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# region expected results discrete data ------------------------------------------------
EXPECTED_RESULTS_DISCRETE_BOUNDS = {
    # horizons, stride=1, drop_last=True
    (2, 1, True): [
        (11, 13),
        (12, 14),
        (13, 15),
        (14, 16),
        (15, 17),
        (16, 18),
        (17, 19),
        (18, 20),
    ],
    (3, 1, True): [
        (11, 14),
        (12, 15),
        (13, 16),
        (14, 17),
        (15, 18),
        (16, 19),
        (17, 20),
    ],
    (4, 1, True): [(11, 15), (12, 16), (13, 17), (14, 18), (15, 19), (16, 20)],
    # horizons, stride=2, drop_last=True
    (2, 2, True): [(11, 13), (13, 15), (15, 17), (17, 19)],
    (3, 2, True): [(11, 14), (13, 16), (15, 18), (17, 20)],
    (4, 2, True): [(11, 15), (13, 17), (15, 19)],
    # horizons, stride=1, drop_last=False
    (2, 1, False): [
        (11, 13),
        (12, 14),
        (13, 15),
        (14, 16),
        (15, 17),
        (16, 18),
        (17, 19),
        (18, 20),
        (19, 21),
        (20, 22),
    ],
    (3, 1, False): [
        (11, 14),
        (12, 15),
        (13, 16),
        (14, 17),
        (15, 18),
        (16, 19),
        (17, 20),
        (18, 21),
        (19, 22),
        (20, 23),
    ],
    (4, 1, False): [
        (11, 15),
        (12, 16),
        (13, 17),
        (14, 18),
        (15, 19),
        (16, 20),
        (17, 21),
        (18, 22),
        (19, 23),
        (20, 24),
    ],
    # horizons, stride=2, drop_last=False
    (2, 2, False): [(11, 13), (13, 15), (15, 17), (17, 19), (19, 21)],
    (3, 2, False): [(11, 14), (13, 16), (15, 18), (17, 20), (19, 22)],
    (4, 2, False): [(11, 15), (13, 17), (15, 19), (17, 21), (19, 23)],
}  # fmt: skip

EXPECTED_RESULTS_DISCRETE_SLICES = {
    # horizons, stride=1, drop_last=True
    (2, 1, True): [
        slice(11, 13, None),  # (11, 13)
        slice(12, 14, None),  # (12, 14)
        slice(13, 15, None),  # (13, 15)
        slice(14, 16, None),  # (14, 16)
        slice(15, 17, None),  # (15, 17)
        slice(16, 18, None),  # (16, 18)
        slice(17, 19, None),  # (17, 19)
        slice(18, 20, None),  # (18, 20)
    ],
    (3, 1, True): [
        slice(11, 14, None),  # (11, 14)
        slice(12, 15, None),  # (12, 15)
        slice(13, 16, None),  # (13, 16)
        slice(14, 17, None),  # (14, 17)
        slice(15, 18, None),  # (15, 18)
        slice(16, 19, None),  # (16, 19)
        slice(17, 20, None),  # (17, 20)
    ],
    (4, 1, True): [
        slice(11, 15, None),  # (11, 15)
        slice(12, 16, None),  # (12, 16)
        slice(13, 17, None),  # (13, 17)
        slice(14, 18, None),  # (14, 18)
        slice(15, 19, None),  # (15, 19)
        slice(16, 20, None),  # (16, 20)
    ],
    # horizons, stride=2, drop_last=True
    (2, 2, True): [
        slice(11, 13, None),  # (11, 13)
        slice(13, 15, None),  # (13, 15)
        slice(15, 17, None),  # (15, 17)
        slice(17, 19, None),  # (17, 19)
    ],
    (3, 2, True): [
        slice(11, 14, None),  # (11, 14)
        slice(13, 16, None),  # (13, 16)
        slice(15, 18, None),  # (15, 18)
        slice(17, 20, None),  # (17, 20)
    ],
    (4, 2, True): [
        slice(11, 15, None),  # (11, 15)
        slice(13, 17, None),  # (13, 17)
        slice(15, 19, None),  # (15, 19)
    ],
    # horizons, stride=1, drop_last=False
    (2, 1, False): [
        slice(11, 13, None),
        slice(12, 14, None),
        slice(13, 15, None),
        slice(14, 16, None),
        slice(15, 17, None),
        slice(16, 18, None),
        slice(17, 19, None),
        slice(18, 20, None),
        slice(19, 21, None),
        slice(20, 22, None),
    ],
    (3, 1, False): [
        slice(11, 14, None),
        slice(12, 15, None),
        slice(13, 16, None),
        slice(14, 17, None),
        slice(15, 18, None),
        slice(16, 19, None),
        slice(17, 20, None),
        slice(18, 21, None),
        slice(19, 22, None),
        slice(20, 23, None),
    ],
    (4, 1, False): [
        slice(11, 15, None),
        slice(12, 16, None),
        slice(13, 17, None),
        slice(14, 18, None),
        slice(15, 19, None),
        slice(16, 20, None),
        slice(17, 21, None),
        slice(18, 22, None),
        slice(19, 23, None),
        slice(20, 24, None),
    ],
    # horizons, stride=2, drop_last=False
    (2, 2, False): [
        slice(11, 13, None),
        slice(13, 15, None),
        slice(15, 17, None),
        slice(17, 19, None),
        slice(19, 21, None),
    ],
    (3, 2, False): [
        slice(11, 14, None),
        slice(13, 16, None),
        slice(15, 18, None),
        slice(17, 20, None),
        slice(19, 22, None),
    ],
    (4, 2, False): [
        slice(11, 15, None),
        slice(13, 17, None),
        slice(15, 19, None),
        slice(17, 21, None),
        slice(19, 23, None),
    ],
}
EXPECTED_RESULTS_DISCRETE_MASKS = {
    # horizons, stride=1, drop_last=True
    (2, 1, True): [
        np.array([Y, Y, N, N, N, N, N, N, N, N]),  # (11, 13)
        np.array([N, Y, Y, N, N, N, N, N, N, N]),  # (12, 14)
        np.array([N, N, Y, Y, N, N, N, N, N, N]),  # (13, 15)
        np.array([N, N, N, Y, Y, N, N, N, N, N]),  # (14, 16)
        np.array([N, N, N, N, Y, Y, N, N, N, N]),  # (15, 17)
        np.array([N, N, N, N, N, Y, Y, N, N, N]),  # (16, 18)
        np.array([N, N, N, N, N, N, Y, Y, N, N]),  # (17, 19)
        np.array([N, N, N, N, N, N, N, Y, Y, N]),  # (18, 20)
        # excluded: np.array([F, F, F, F, F, F, F, F, T, T]),
    ],
    (3, 1, True): [
        np.array([Y, Y, Y, N, N, N, N, N, N, N]),  # (11, 14)
        np.array([N, Y, Y, Y, N, N, N, N, N, N]),  # (12, 15)
        np.array([N, N, Y, Y, Y, N, N, N, N, N]),  # (13, 16)
        np.array([N, N, N, Y, Y, Y, N, N, N, N]),  # (14, 17)
        np.array([N, N, N, N, Y, Y, Y, N, N, N]),  # (15, 18)
        np.array([N, N, N, N, N, Y, Y, Y, N, N]),  # (16, 19)
        np.array([N, N, N, N, N, N, Y, Y, Y, N]),  # (17, 20)
        # excluded: np.array([F, F, F, F, F, F, F, T, T, T]),
    ],
    (4, 1, True): [
        np.array([Y, Y, Y, Y, N, N, N, N, N, N]),  # (11, 15)
        np.array([N, Y, Y, Y, Y, N, N, N, N, N]),  # (12, 16)
        np.array([N, N, Y, Y, Y, Y, N, N, N, N]),  # (13, 17)
        np.array([N, N, N, Y, Y, Y, Y, N, N, N]),  # (14, 18)
        np.array([N, N, N, N, Y, Y, Y, Y, N, N]),  # (15, 19)
        np.array([N, N, N, N, N, Y, Y, Y, Y, N]),  # (16, 20)
        # excluded: np.array([F, F, F, F, F, F, T, T, T, T]),
    ],
    # horizons, stride=2, drop_last=True
    (2, 2, True): [
        np.array([Y, Y, N, N, N, N, N, N, N, N]),  # (11, 13)
        np.array([N, N, Y, Y, N, N, N, N, N, N]),  # (13, 15)
        np.array([N, N, N, N, Y, Y, N, N, N, N]),  # (15, 17)
        np.array([N, N, N, N, N, N, Y, Y, N, N]),  # (17, 19)
        # excluded: np.array([F, F, F, F, F, F, F, F, T, T]),
    ],
    (3, 2, True): [
        np.array([Y, Y, Y, N, N, N, N, N, N, N]),  # (11, 14)
        np.array([N, N, Y, Y, Y, N, N, N, N, N]),  # (13, 16)
        np.array([N, N, N, N, Y, Y, Y, N, N, N]),  # (15, 18)
        np.array([N, N, N, N, N, N, Y, Y, Y, N]),  # (17, 20)
    ],
    (4, 2, True): [
        np.array([Y, Y, Y, Y, N, N, N, N, N, N]),  # (11, 15)
        np.array([N, N, Y, Y, Y, Y, N, N, N, N]),  # (13, 17)
        np.array([N, N, N, N, Y, Y, Y, Y, N, N]),  # (15, 19)
        # excluded: np.array([F, F, F, F, F, F, T, T, T, T]),
    ],
    # horizons, stride=1, drop_last=False
    (2, 1, False): [
        np.array([Y, Y, N, N, N, N, N, N, N, N]),
        np.array([N, Y, Y, N, N, N, N, N, N, N]),
        np.array([N, N, Y, Y, N, N, N, N, N, N]),
        np.array([N, N, N, Y, Y, N, N, N, N, N]),
        np.array([N, N, N, N, Y, Y, N, N, N, N]),
        np.array([N, N, N, N, N, Y, Y, N, N, N]),
        np.array([N, N, N, N, N, N, Y, Y, N, N]),
        np.array([N, N, N, N, N, N, N, Y, Y, N]),
        np.array([N, N, N, N, N, N, N, N, Y, Y]),
        np.array([N, N, N, N, N, N, N, N, N, Y]),
    ],
    (3, 1, False): [
        np.array([Y, Y, Y, N, N, N, N, N, N, N]),
        np.array([N, Y, Y, Y, N, N, N, N, N, N]),
        np.array([N, N, Y, Y, Y, N, N, N, N, N]),
        np.array([N, N, N, Y, Y, Y, N, N, N, N]),
        np.array([N, N, N, N, Y, Y, Y, N, N, N]),
        np.array([N, N, N, N, N, Y, Y, Y, N, N]),
        np.array([N, N, N, N, N, N, Y, Y, Y, N]),
        np.array([N, N, N, N, N, N, N, Y, Y, Y]),
        np.array([N, N, N, N, N, N, N, N, Y, Y]),
        np.array([N, N, N, N, N, N, N, N, N, Y]),
    ],
    (4, 1, False): [
        np.array([Y, Y, Y, Y, N, N, N, N, N, N]),
        np.array([N, Y, Y, Y, Y, N, N, N, N, N]),
        np.array([N, N, Y, Y, Y, Y, N, N, N, N]),
        np.array([N, N, N, Y, Y, Y, Y, N, N, N]),
        np.array([N, N, N, N, Y, Y, Y, Y, N, N]),
        np.array([N, N, N, N, N, Y, Y, Y, Y, N]),
        np.array([N, N, N, N, N, N, Y, Y, Y, Y]),
        np.array([N, N, N, N, N, N, N, Y, Y, Y]),
        np.array([N, N, N, N, N, N, N, N, Y, Y]),
        np.array([N, N, N, N, N, N, N, N, N, Y]),
    ],
    # horizons, stride=2, drop_last=False
    (2, 2, False): [
        np.array([Y, Y, N, N, N, N, N, N, N, N]),
        np.array([N, N, Y, Y, N, N, N, N, N, N]),
        np.array([N, N, N, N, Y, Y, N, N, N, N]),
        np.array([N, N, N, N, N, N, Y, Y, N, N]),
        np.array([N, N, N, N, N, N, N, N, Y, Y]),
    ],
    (3, 2, False): [
        np.array([Y, Y, Y, N, N, N, N, N, N, N]),
        np.array([N, N, Y, Y, Y, N, N, N, N, N]),
        np.array([N, N, N, N, Y, Y, Y, N, N, N]),
        np.array([N, N, N, N, N, N, Y, Y, Y, N]),
        np.array([N, N, N, N, N, N, N, N, Y, Y]),
    ],
    (4, 2, False): [
        np.array([Y, Y, Y, Y, N, N, N, N, N, N]),
        np.array([N, N, Y, Y, Y, Y, N, N, N, N]),
        np.array([N, N, N, N, Y, Y, Y, Y, N, N]),
        np.array([N, N, N, N, N, N, Y, Y, Y, Y]),
        np.array([N, N, N, N, N, N, N, N, Y, Y]),
    ],
}
EXPECTED_RESULTS_DISCRETE_WINDOWS = {
    # horizons, stride=1, drop_last=True
    (2, 1, True): [
        np.array([11, 12]),  # (11, 13)
        np.array([12, 13]),  # (12, 14)
        np.array([13, 14]),  # (13, 15)
        np.array([14, 15]),  # (14, 16)
        np.array([15, 16]),  # (15, 17)
        np.array([16, 17]),  # (16, 18)
        np.array([17, 18]),  # (17, 19)
        np.array([18, 19]),  # (18, 20)
        # excluded: np.array([19, 20]),
    ],
    (3, 1, True): [
        np.array([11, 12, 13]),  # (11, 14)
        np.array([12, 13, 14]),  # (12, 15)
        np.array([13, 14, 15]),  # (13, 16)
        np.array([14, 15, 16]),  # (14, 17)
        np.array([15, 16, 17]),  # (15, 18)
        np.array([16, 17, 18]),  # (16, 19)
        np.array([17, 18, 19]),  # (17, 20)
        # excluded: np.array([18, 19, 20]),
    ],
    (4, 1, True): [
        np.array([11, 12, 13, 14]),  # (11, 15)
        np.array([12, 13, 14, 15]),  # (12, 16)
        np.array([13, 14, 15, 16]),  # (13, 17)
        np.array([14, 15, 16, 17]),  # (14, 18)
        np.array([15, 16, 17, 18]),  # (15, 19)
        np.array([16, 17, 18, 19]),  # (16, 20)
        # excluded: np.array([17, 18, 19, 20]),
    ],
    # horizons, stride=2, drop_last=True
    (2, 2, True): [
        np.array([11, 12]),  # (11, 13)
        np.array([13, 14]),  # (13, 15)
        np.array([15, 16]),  # (15, 17)
        np.array([17, 18]),  # (17, 19)
        # excluded: np.array([19, 20]),
    ],
    (3, 2, True): [
        np.array([11, 12, 13]),  # (11, 14)
        np.array([13, 14, 15]),  # (13, 16)
        np.array([15, 16, 17]),  # (15, 18)
        np.array([17, 18, 19]),  # (17, 20)
    ],
    (4, 2, True): [
        np.array([11, 12, 13, 14]),  # (11, 15)
        np.array([13, 14, 15, 16]),  # (13, 17)
        np.array([15, 16, 17, 18]),  # (15, 19)
        # excluded: np.array([17, 18, 19, 20]),
    ],
    # horizons, stride=1, drop_last=False
    (2, 1, False): [
        np.array([11, 12]),
        np.array([12, 13]),
        np.array([13, 14]),
        np.array([14, 15]),
        np.array([15, 16]),
        np.array([16, 17]),
        np.array([17, 18]),
        np.array([18, 19]),
        np.array([19, 20]),
        np.array([20]),
    ],
    (3, 1, False): [
        np.array([11, 12, 13]),
        np.array([12, 13, 14]),
        np.array([13, 14, 15]),
        np.array([14, 15, 16]),
        np.array([15, 16, 17]),
        np.array([16, 17, 18]),
        np.array([17, 18, 19]),
        np.array([18, 19, 20]),
        np.array([19, 20]),
        np.array([20]),
    ],
    (4, 1, False): [
        np.array([11, 12, 13, 14]),
        np.array([12, 13, 14, 15]),
        np.array([13, 14, 15, 16]),
        np.array([14, 15, 16, 17]),
        np.array([15, 16, 17, 18]),
        np.array([16, 17, 18, 19]),
        np.array([17, 18, 19, 20]),
        np.array([18, 19, 20]),
        np.array([19, 20]),
        np.array([20]),
    ],
    # horizons, stride=2, drop_last=False
    (2, 2, False): [
        np.array([11, 12]),
        np.array([13, 14]),
        np.array([15, 16]),
        np.array([17, 18]),
        np.array([19, 20]),
    ],
    (3, 2, False): [
        np.array([11, 12, 13]),
        np.array([13, 14, 15]),
        np.array([15, 16, 17]),
        np.array([17, 18, 19]),
        np.array([19, 20]),
    ],
    (4, 2, False): [
        np.array([11, 12, 13, 14]),
        np.array([13, 14, 15, 16]),
        np.array([15, 16, 17, 18]),
        np.array([17, 18, 19, 20]),
        np.array([19, 20]),
    ],
}
# endregion expected results discrete data ---------------------------------------------
EXPECTED_RESULTS_DISCRETE_DATA: dict[MODE, Any] = {
    MODE.BOUNDS : EXPECTED_RESULTS_DISCRETE_BOUNDS,
    MODE.MASK   : EXPECTED_RESULTS_DISCRETE_MASKS,
    MODE.SLICE  : EXPECTED_RESULTS_DISCRETE_SLICES,
    MODE.POINTS : EXPECTED_RESULTS_DISCRETE_WINDOWS,
}  # fmt: skip
# NOTE that we include duplicates in the continuous data
# We test horizons ∈ {2.5, 3.5, 4.5}, stride ∈ {1.0, 2.5}, drop_last ∈ {True, False}
CONTINUOUS_DATA = [2.5, 3.3, 3.7, 4.0, 5.9, 6.4, 6.4, 6.6, 7.5, 8.9]
# region expected results continuous data ----------------------------------------------
EXPECTED_RESULTS_CONTINUOUS_BOUNDS: dict[tuple, list[tuple[float, float]]] = {
    # horizons, stride=1.0, drop_last=True
    (2.5, 1.0, True): [(2.5, 5.0), (3.5, 6.0), (4.5, 7.0), (5.5, 8.0)],
    (3.5, 1.0, True): [(2.5, 6.0), (3.5, 7.0), (4.5, 8.0)],
    (4.5, 1.0, True): [(2.5, 7.0), (3.5, 8.0)],
    # horizons, stride=2.5, drop_last=True
    (2.5, 2.5, True): [(2.5, 5.0), (5.0, 7.5)],
    (3.5, 2.5, True): [(2.5, 6.0), (5.0, 8.5)],
    (4.5, 2.5, True): [(2.5, 7.0)],
    # horizons, stride=1.0, drop_last=False
    (2.5, 1.0, False): [
        (2.5, 5.0),
        (3.5, 6.0),
        (4.5, 7.0),
        (5.5, 8.0),
        (6.5, 9.0),
        (7.5, 10.0),
        (8.5, 11.0),
    ],
    (3.5, 1.0, False): [
        (2.5, 6.0),
        (3.5, 7.0),
        (4.5, 8.0),
        (5.5, 9.0),
        (6.5, 10.0),
        (7.5, 11.0),
        (8.5, 12.0),
    ],
    (4.5, 1.0, False): [
        (2.5, 7.0),
        (3.5, 8.0),
        (4.5, 9.0),
        (5.5, 10.0),
        (6.5, 11.0),
        (7.5, 12.0),
        (8.5, 13.0),
    ],
    # horizons, stride=2.5, drop_last=False
    (2.5, 2.5, False): [(2.5, 5.0), (5.0, 7.5), (7.5, 10.0)],
    (3.5, 2.5, False): [(2.5, 6.0), (5.0, 8.5), (7.5, 11.0)],
    (4.5, 2.5, False): [(2.5, 7.0), (5.0, 9.5), (7.5, 12.0)],
}  # fmt: skip

EXPECTED_RESULTS_CONTINUOUS_WINDOWS: dict[tuple, list[np.ndarray]] = {
    # horizons, stride=1.0, drop_last=True
    (2.5, 1.0, True): [
        np.array([2.5, 3.3, 3.7, 4.0]),  # (2.5, 5.0)
        np.array([3.7, 4.0, 5.9]),  # (3.5, 6.0)
        np.array([5.9, 6.4, 6.4, 6.6]),  # (4.5, 7.0)
        np.array([5.9, 6.4, 6.4, 6.6, 7.5]),  # (5.5, 8.0)
    ],
    (3.5, 1.0, True): [
        np.array([2.5, 3.3, 3.7, 4.0, 5.9]),  # (2.5, 6.0)
        np.array([3.7, 4.0, 5.9, 6.4, 6.4, 6.6]),  # (3.5, 7.0)
        np.array([5.9, 6.4, 6.4, 6.6, 7.5]),  # (4.5, 8.0)
    ],
    (4.5, 1.0, True): [
        np.array([2.5, 3.3, 3.7, 4.0, 5.9, 6.4, 6.4, 6.6]),  # (2.5, 7.0)
        np.array([3.7, 4.0, 5.9, 6.4, 6.4, 6.6, 7.5]),  # (3.5, 8.0)
    ],
    # horizons, stride=2.5, drop_last=True
    (2.5, 2.5, True): [
        np.array([2.5, 3.3, 3.7, 4.0]),  # (2.5, 5.0)
        np.array([5.9, 6.4, 6.4, 6.6]),  # (5.0, 7.5)
    ],
    (3.5, 2.5, True): [
        np.array([2.5, 3.3, 3.7, 4.0, 5.9]),  # (2.5, 6.0)
        np.array([5.9, 6.4, 6.4, 6.6, 7.5]),  # (5.0, 8.5)
    ],
    (4.5, 2.5, True): [
        np.array([2.5, 3.3, 3.7, 4.0, 5.9, 6.4, 6.4, 6.6]),  # (2.5, 7.0)
    ],
    # horizons, stride=1.0, drop_last=False
    (2.5, 1.0, False): [
        np.array([2.5, 3.3, 3.7, 4.0]),  # (2.5, 5.0)
        np.array([3.7, 4.0, 5.9]),  # (3.5, 6.0)
        np.array([5.9, 6.4, 6.4, 6.6]),  # (4.5, 7.0)
        np.array([5.9, 6.4, 6.4, 6.6, 7.5]),  # (5.5, 8.0)
        np.array([6.6, 7.5, 8.9]),  # (6.5, 9.0)
        np.array([7.5, 8.9]),  # (7.5, 10.0)
        np.array([8.9]),  # (8.5, 11.0)
    ],
    (3.5, 1.0, False): [
        np.array([2.5, 3.3, 3.7, 4.0, 5.9]),  # (2.5, 6.0)
        np.array([3.7, 4.0, 5.9, 6.4, 6.4, 6.6]),  # (3.5, 7.0)
        np.array([5.9, 6.4, 6.4, 6.6, 7.5]),  # (4.5, 8.0)
        np.array([5.9, 6.4, 6.4, 6.6, 7.5, 8.9]),  # (5.5, 9.0)
        np.array([6.6, 7.5, 8.9]),  # (6.5, 10.0)
        np.array([7.5, 8.9]),  # (7.5, 11.0)
        np.array([8.9]),  # (8.5, 12.0)
    ],
    (4.5, 1.0, False): [
        np.array([2.5, 3.3, 3.7, 4.0, 5.9, 6.4, 6.4, 6.6]),  # (2.5, 7.0)
        np.array([3.7, 4.0, 5.9, 6.4, 6.4, 6.6, 7.5]),  # (3.5, 8.0)
        np.array([5.9, 6.4, 6.4, 6.6, 7.5, 8.9]),  # (4.5, 9.0)
        np.array([5.9, 6.4, 6.4, 6.6, 7.5, 8.9]),  # (5.5, 10.0)
        np.array([6.6, 7.5, 8.9]),  # (6.5, 11.0)
        np.array([7.5, 8.9]),  # (7.5, 12.0)
        np.array([8.9]),  # (8.5, 13.0)
    ],
    # horizons, stride=2.5, drop_last=False
    (2.5, 2.5, False): [
        np.array([2.5, 3.3, 3.7, 4.0]),  # (2.5, 5.0)
        np.array([5.9, 6.4, 6.4, 6.6]),  # (5.0, 7.5)
        np.array([7.5, 8.9]),  # (7.5, 10.0)
    ],
    (3.5, 2.5, False): [
        np.array([2.5, 3.3, 3.7, 4.0, 5.9]),  # (2.5, 6.0)
        np.array([5.9, 6.4, 6.4, 6.6, 7.5]),  # (5.0, 8.5)
        np.array([7.5, 8.9]),  # (7.5, 11.0)
    ],
    (4.5, 2.5, False): [
        np.array([2.5, 3.3, 3.7, 4.0, 5.9, 6.4, 6.4, 6.6]),  # (2.5, 7.0)
        np.array([5.9, 6.4, 6.4, 6.6, 7.5, 8.9]),  # (5.0, 9.5)
        np.array([7.5, 8.9]),  # (7.5, 12.0)
    ],
}
EXPECTED_RESULTS_CONTINUOUS_MASKS: dict[tuple, list[np.ndarray]] = {
    # horizons, stride=1.0, drop_last=True
    (2.5, 1.0, True): [
        np.array([Y, Y, Y, Y, N, N, N, N, N, N]),  # (2.5, 5.0)
        np.array([N, N, Y, Y, Y, N, N, N, N, N]),  # (3.5, 6.0)
        np.array([N, N, N, N, Y, Y, Y, Y, N, N]),  # (4.5, 7.0)
        np.array([N, N, N, N, Y, Y, Y, Y, Y, N]),  # (5.5, 8.0)
    ],
    (3.5, 1.0, True): [
        np.array([Y, Y, Y, Y, Y, N, N, N, N, N]),  # (2.5, 6.0)
        np.array([N, N, Y, Y, Y, Y, Y, Y, N, N]),  # (3.5, 7.0)
        np.array([N, N, N, N, Y, Y, Y, Y, Y, N]),  # (4.5, 8.0)
    ],
    (4.5, 1.0, True): [
        np.array([Y, Y, Y, Y, Y, Y, Y, Y, N, N]),  # (2.5, 7.0)
        np.array([N, N, Y, Y, Y, Y, Y, Y, Y, N]),  # (3.5, 8.0)
    ],
    # horizons, stride=2.5, drop_last=True
    (2.5, 2.5, True): [
        np.array([Y, Y, Y, Y, N, N, N, N, N, N]),  # (2.5, 5.0)
        np.array([N, N, N, N, Y, Y, Y, Y, N, N]),  # (5.0, 7.5)
    ],
    (3.5, 2.5, True): [
        np.array([Y, Y, Y, Y, Y, N, N, N, N, N]),  # (2.5, 6.0)
        np.array([N, N, N, N, Y, Y, Y, Y, Y, N]),  # (5.0, 8.5)
    ],
    (4.5, 2.5, True): [
        np.array([Y, Y, Y, Y, Y, Y, Y, Y, N, N]),  # (2.5, 7.0)
    ],
    # horizons, stride=1.0, drop_last=False
    (2.5, 1.0, False): [
        np.array([Y, Y, Y, Y, N, N, N, N, N, N]),  # (2.5, 5.0)
        np.array([N, N, Y, Y, Y, N, N, N, N, N]),  # (3.5, 6.0)
        np.array([N, N, N, N, Y, Y, Y, Y, N, N]),  # (4.5, 7.0)
        np.array([N, N, N, N, Y, Y, Y, Y, Y, N]),  # (5.5, 8.0)
        np.array([N, N, N, N, N, N, N, Y, Y, Y]),  # (6.5, 9.0)
        np.array([N, N, N, N, N, N, N, N, Y, Y]),  # (7.5, 10.0)
        np.array([N, N, N, N, N, N, N, N, N, Y]),  # (8.5, 11.0)
    ],
    (3.5, 1.0, False): [
        np.array([Y, Y, Y, Y, Y, N, N, N, N, N]),  # (2.5, 6.0)
        np.array([N, N, Y, Y, Y, Y, Y, Y, N, N]),  # (3.5, 7.0)
        np.array([N, N, N, N, Y, Y, Y, Y, Y, N]),  # (4.5, 8.0)
        np.array([N, N, N, N, Y, Y, Y, Y, Y, Y]),  # (5.5, 9.0)
        np.array([N, N, N, N, N, N, N, Y, Y, Y]),  # (6.5, 10.0)
        np.array([N, N, N, N, N, N, N, N, Y, Y]),  # (7.5, 11.0)
        np.array([N, N, N, N, N, N, N, N, N, Y]),  # (8.5, 12.0)
    ],
    (4.5, 1.0, False): [
        np.array([Y, Y, Y, Y, Y, Y, Y, Y, N, N]),  # (2.5, 7.0),
        np.array([N, N, Y, Y, Y, Y, Y, Y, Y, N]),  # (3.5, 8.0),
        np.array([N, N, N, N, Y, Y, Y, Y, Y, Y]),  # (4.5, 9.0),
        np.array([N, N, N, N, Y, Y, Y, Y, Y, Y]),  # (5.5, 10.0),
        np.array([N, N, N, N, N, N, N, Y, Y, Y]),  # (6.5, 11.0),
        np.array([N, N, N, N, N, N, N, N, Y, Y]),  # (7.5, 12.0),
        np.array([N, N, N, N, N, N, N, N, N, Y]),  # (8.5, 13.0),
    ],
    # horizons, stride=2.5, drop_last=False
    (2.5, 2.5, False): [
        np.array([Y, Y, Y, Y, N, N, N, N, N, N]),  # (2.5, 5.0)
        np.array([N, N, N, N, Y, Y, Y, Y, N, N]),  # (5.0, 7.5)
        np.array([N, N, N, N, N, N, N, N, Y, Y]),  # (7.5, 10.0)
    ],
    (3.5, 2.5, False): [
        np.array([Y, Y, Y, Y, Y, N, N, N, N, N]),  # (2.5, 6.0)
        np.array([N, N, N, N, Y, Y, Y, Y, Y, N]),  # (5.0, 8.5)
        np.array([N, N, N, N, N, N, N, N, Y, Y]),  # (7.5, 11.0)
    ],
    (4.5, 2.5, False): [
        np.array([Y, Y, Y, Y, Y, Y, Y, Y, N, N]),  # (2.5, 7.0)
        np.array([N, N, N, N, Y, Y, Y, Y, Y, Y]),  # (5.0, 9.5)
        np.array([N, N, N, N, N, N, N, N, Y, Y]),  # (7.5, 12.0)
    ],
}
EXPECTED_RESULTS_CONTINUOUS_SLICES: dict[tuple, list[slice]] = {
    # horizons, stride=1.0, drop_last=True
    (2.5, 1.0, True): [
        slice(2.5, 5.0, None),  # (2.5, 5.0)
        slice(3.5, 6.0, None),  # (3.5, 6.0)
        slice(4.5, 7.0, None),  # (4.5, 7.0)
        slice(5.5, 8.0, None),  # (5.5, 8.0)
    ],
    (3.5, 1.0, True): [
        slice(2.5, 6.0, None),  # (2.5, 6.0)
        slice(3.5, 7.0, None),  # (3.5, 7.0)
        slice(4.5, 8.0, None),  # (4.5, 8.0)
    ],
    (4.5, 1.0, True): [
        slice(2.5, 7.0, None),  # (2.5, 7.0)
        slice(3.5, 8.0, None),  # (3.5, 8.0)
    ],
    # horizons, stride=2.5, drop_last=True
    (2.5, 2.5, True): [
        slice(2.5, 5.0, None),  # (2.5, 5.0)
        slice(5.0, 7.5, None),  # (5.0, 7.5)
    ],
    (3.5, 2.5, True): [
        slice(2.5, 6.0, None),  # (2.5, 6.0)
        slice(5.0, 8.5, None),  # (5.0, 8.5)
    ],
    (4.5, 2.5, True): [
        slice(2.5, 7.0, None),  # (2.5, 7.0)
    ],
    # horizons, stride=1.0, drop_last=False
    (2.5, 1.0, False): [
        slice(2.5, 5.0, None),  # (2.5, 5.0)
        slice(3.5, 6.0, None),  # (3.5, 6.0)
        slice(4.5, 7.0, None),  # (4.5, 7.0)
        slice(5.5, 8.0, None),  # (5.5, 8.0)
        slice(6.5, 9.0, None),  # (6.5, 9.0)
        slice(7.5, 10.0, None),  # (7.5, 10.0)
        slice(8.5, 11.0, None),  # (8.5, 11.0)
    ],
    (3.5, 1.0, False): [
        slice(2.5, 6.0, None),  # (2.5, 6.0)
        slice(3.5, 7.0, None),  # (3.5, 7.0)
        slice(4.5, 8.0, None),  # (4.5, 8.0)
        slice(5.5, 9.0, None),  # (5.5, 9.0)
        slice(6.5, 10.0, None),  # (6.5, 10.0)
        slice(7.5, 11.0, None),  # (7.5, 11.0)
        slice(8.5, 12.0, None),  # (8.5, 12.0)
    ],
    (4.5, 1.0, False): [
        slice(2.5, 7.0, None),  # (2.5, 7.0),
        slice(3.5, 8.0, None),  # (3.5, 8.0),
        slice(4.5, 9.0, None),  # (4.5, 9.0),
        slice(5.5, 10.0, None),  # (5.5, 10.0),
        slice(6.5, 11.0, None),  # (6.5, 11.0),
        slice(7.5, 12.0, None),  # (7.5, 12.0),
        slice(8.5, 13.0, None),  # (8.5, 13.0),
    ],
    # horizons, stride=2.5, drop_last=False
    (2.5, 2.5, False): [
        slice(2.5, 5.0, None),  # (2.5, 5.0)
        slice(5.0, 7.5, None),  # (5.0, 7.5)
        slice(7.5, 10.0, None),  # (7.5, 10.0)
    ],
    (3.5, 2.5, False): [
        slice(2.5, 6.0, None),  # (2.5, 6.0)
        slice(5.0, 8.5, None),  # (5.0, 8.5)
        slice(7.5, 11.0, None),  # (7.5, 11.0)
    ],
    (4.5, 2.5, False): [
        slice(2.5, 7.0, None),  # (2.5, 7.0)
        slice(5.0, 9.5, None),  # (5.0, 9.5)
        slice(7.5, 12.0, None),  # (7.5, 12.0)
    ],
}
# endregion expected results continuous data -------------------------------------------
EXPECTED_RESULTS_CONTINUOUS_DATA: dict[MODE, Any] = {
    MODE.BOUNDS : EXPECTED_RESULTS_CONTINUOUS_BOUNDS,
    MODE.MASK   : EXPECTED_RESULTS_CONTINUOUS_MASKS,
    MODE.SLICE  : EXPECTED_RESULTS_CONTINUOUS_SLICES,
    MODE.POINTS : EXPECTED_RESULTS_CONTINUOUS_WINDOWS,
}  # fmt: skip


# write parametrized unit test with the above data for all modes
@pytest.mark.parametrize("drop_last", [False, True], ids=lambda x: f"drop_last={x}")
@pytest.mark.parametrize("stride", [1, 2], ids=lambda x: f"stride={x}")
@pytest.mark.parametrize("horizons", [2, 3, 4], ids=lambda x: f"horizon={x}")
@pytest.mark.parametrize("mode", ["bounds", "mask", "slice", "points"])
def test_sliding_window_sampler_discrete(
    *, drop_last: bool, stride: int, horizons: int, mode: Mode
) -> None:
    r"""Test the SlidingWindowSampler."""
    sampler = SlidingWindowSampler(
        DISCRETE_DATA,
        stride=stride,
        horizons=horizons,
        mode=mode,
        shuffle=False,
        drop_last=drop_last,
    )
    result = list(sampler)
    sample = result[0]

    # check that static types are correct
    assert_type(sampler, SlidingWindowSampler[int, MODE, ONE])
    assert_type(sampler.mode, MODE)
    assert_type(iter(sampler), Iterator[Any])
    assert_type(result, list[Any])
    assert_type(sample, Any)

    # compare with expected results
    expected = EXPECTED_RESULTS_DISCRETE_DATA[MODE(mode)][horizons, stride, drop_last]

    assert len(sampler) == len(expected), (
        "LENGTH MISMATCH!"
        f"\nsampler:\n{result}\nexpected:\n{expected}\ngrid={sampler.grid}\n"
    )

    for m1, m2 in zip(sampler, expected, strict=True):
        assert np.array_equal(m1, m2), (
            f"SAMPLE MISMATCH!sample:\n{m1}\nexpected:\n{m2}\ngrid={sampler.grid}\n"
        )


# write parametrized unit test with the above data for all modes
@pytest.mark.parametrize("drop_last", [False, True], ids=lambda x: f"drop_last={x}")
@pytest.mark.parametrize("stride", [1.0, 2.5], ids=lambda x: f"stride={x}")
@pytest.mark.parametrize("horizons", [2.5, 3.5, 4.5], ids=lambda x: f"horizon={x}")
@pytest.mark.parametrize("mode", ["bounds", "mask", "slice", "points"])
def test_sliding_window_sampler_continuous(
    *, drop_last: bool, stride: float, horizons: float, mode: Mode
) -> None:
    r"""Test the SlidingWindowSampler."""
    sampler = SlidingWindowSampler(
        CONTINUOUS_DATA,
        stride=stride,
        horizons=horizons,
        mode=mode,
        shuffle=False,
        drop_last=drop_last,
    )
    result = list(sampler)
    sample = result[0]

    # check that static types are correct
    assert_type(sampler, SlidingWindowSampler[float, MODE, ONE])
    assert_type(sampler.mode, MODE)
    assert_type(iter(sampler), Iterator[Any])
    assert_type(result, list[Any])
    assert_type(sample, Any)

    # compare with expected results
    expected = EXPECTED_RESULTS_CONTINUOUS_DATA[MODE(mode)][horizons, stride, drop_last]

    assert len(sampler) == len(expected), (
        "LENGTH MISMATCH!"
        f"\nsampler:\n{result}\nexpected:\n{expected}\ngrid={sampler.grid}\n"
    )

    for m1, m2 in zip(sampler, expected, strict=True):
        assert np.array_equal(m1, m2), (
            f"SAMPLE MISMATCH!sample:\n{m1}\nexpected:\n{m2}\ngrid={sampler.grid}\n"
        )


# dates 2020-01-01 to 2020-01-10
PYTHON_DATES = [
    datetime.datetime(2020, 1, 1) + datetime.timedelta(days=i) for i in range(10)
]
DATETIME_DATA: dict[str, Indexable[Any]] = {
    "list-python"     : PYTHON_DATES,
    "list-pandas"     : [pd.Timestamp(d) for d in PYTHON_DATES],
    "numpy"           : np.array(PYTHON_DATES, dtype="datetime64[ns]"),
    "series-numpy"    : pd.Series(PYTHON_DATES, dtype="datetime64[ns]"),
    "series-pyarrow"  : pd.Series(PYTHON_DATES, dtype="timestamp[ns][pyarrow]"),
    "index-numpy"     : pd.Index(PYTHON_DATES, dtype="datetime64[ns]"),
    "index-pyarrow"   : pd.Index(PYTHON_DATES, dtype="timestamp[ns][pyarrow]"),
}  # fmt: skip


@pytest_xfail(
    "Interval does not support numpy.datetime",
    condition=lambda case, mode: case == "numpy" and mode is MODE.INTERVAL,
    raises=ValueError,
)
@pytest.mark.parametrize("mode", SlidingWindowSampler.MODE)
@pytest.mark.parametrize("case", DATETIME_DATA)
def test_datetime_data(case: str, mode: MODE) -> None:
    r"""Test the SlidingWindowSampler with datetime/timedelta data."""
    data = DATETIME_DATA[case]
    sampler = SlidingWindowSampler(
        data,
        stride="8h",
        horizons="3d",
        mode=mode,
        shuffle=False,
        drop_last=False,
    )
    result = list(sampler)
    sample = result[0]

    # check that static types are correct
    assert_type(sampler, SlidingWindowSampler[Any, MODE, ONE])
    assert_type(sampler.mode, MODE)
    assert_type(iter(sampler), Iterator[Any])
    assert_type(result, list[Any])
    assert_type(sample, Any)

    assert len(result) == 28
    assert len(sampler) == 28

    datetime_type = type(data[0])
    sample = result[0]

    match mode:
        case MODE.BOUNDS:
            assert isinstance(sample, tuple)
            assert isinstance(sample[0], datetime_type), type(sample[0])
        case MODE.SLICE:
            assert isinstance(sample, slice)
            assert isinstance(sample.start, datetime_type)
        case MODE.INTERVAL:
            assert isinstance(sample, pd.Interval)
            assert isinstance(sample.left, datetime_type), type(sample.left)
        case MODE.POINTS:
            assert isinstance(sample, np.ndarray)
            assert isinstance(sample[0], datetime_type), type(sample[0])
        case MODE.MASK:
            assert isinstance(sample, np.ndarray)
            assert np.issubdtype(sample.dtype, np.bool_)
        case MODE.INDEX:
            assert isinstance(sample, np.ndarray)
            assert isinstance(sample[0], np.integer), type(sample[0])
        case _:
            assert_never(mode)


# increasing data with random step size
PYTHON_INTEGERS = [-7, 0, 3, 4, 6, 11, 12, 14, 18, 20, 21]
INTEGER_DATA: dict[str, Indexable[Any]] = {
    "python-int": PYTHON_INTEGERS,
    "numpy-int64": np.array(PYTHON_INTEGERS, dtype=np.int64),
    "numpy-int32": np.array(PYTHON_INTEGERS, dtype=np.int32),
    "series-pyarrow-int32": pd.Series(PYTHON_INTEGERS, dtype="int32[pyarrow]"),
    "index-pyarrow-int32": pd.Index(PYTHON_INTEGERS, dtype="int32[pyarrow]"),
}


@pytest.mark.parametrize("mode", SlidingWindowSampler.MODE)
@pytest.mark.parametrize("example", INTEGER_DATA)
def test_integer_data(example: str, mode: MODE) -> None:
    r"""Test the SlidingWindowSampler with datetime/timedelta data."""
    data = INTEGER_DATA[example]
    sampler = SlidingWindowSampler(
        data,
        stride=2,
        horizons=3,
        mode=mode,
        shuffle=False,
        drop_last=False,
    )
    result = list(sampler)
    sample = result[0]

    # check that static types are correct
    assert_type(sampler, SlidingWindowSampler[Any, MODE, ONE])
    assert_type(sampler.mode, MODE)
    assert_type(iter(sampler), Iterator[Any])
    assert_type(result, list)
    assert_type(sample, Any)

    assert len(result) == 15
    assert len(sampler) == 15

    # FIXME: https://github.com/pandas-dev/pandas/issues/56021
    match mode:
        case MODE.BOUNDS:
            assert isinstance(sample, tuple)
            assert isinstance(sample[0], np.integer), type(sample[0])
        case MODE.SLICE:
            assert isinstance(sample, slice)
            assert isinstance(sample.start, np.integer)
        case MODE.INTERVAL:
            assert isinstance(sample, pd.Interval)
            assert isinstance(sample.left, np.integer), type(sample.left)
        case MODE.POINTS:
            assert isinstance(sample, np.ndarray)
            assert isinstance(sample[0], np.integer), type(sample[0])
        case MODE.MASK:
            assert isinstance(sample, np.ndarray)
            assert np.issubdtype(sample.dtype, np.bool_)
        case MODE.INDEX:
            assert isinstance(sample, np.ndarray)
            assert isinstance(sample[0], np.integer), type(sample[0])
        case _:
            assert_never(mode)


PYTHON_FLOATS = [-2.3, 0.1, 4.2, 5.3, 5.5, 5.6, 6.0, 8.4, 10.7]
FLOAT_DATA: dict[str, Indexable[Any]] = {
    "python-float": PYTHON_FLOATS,
    "numpy-float64": np.array(PYTHON_FLOATS, dtype=np.float64),
    "numpy-float32": np.array(PYTHON_FLOATS, dtype=np.float32),
    "series-pyarrow-float32": pd.Series(PYTHON_FLOATS, dtype="float32[pyarrow]"),
    "index-pyarrow-float32": pd.Index(PYTHON_FLOATS, dtype="float32[pyarrow]"),
}


@pytest.mark.parametrize("mode", SlidingWindowSampler.MODE)
@pytest.mark.parametrize("example", FLOAT_DATA)
def test_float_data(example: str, mode: MODE) -> None:
    r"""Test the SlidingWindowSampler with datetime/timedelta data."""
    data = FLOAT_DATA[example]
    sampler = SlidingWindowSampler(
        data,
        stride=2.1,
        horizons=2.8,
        mode=mode,
        shuffle=False,
        drop_last=False,
    )
    result = list(sampler)
    sample = result[0]

    # check that static types are correct
    assert_type(sampler, SlidingWindowSampler[Any, MODE, ONE])
    assert_type(sampler.mode, MODE)
    assert_type(iter(sampler), Iterator[Any])
    assert_type(result, list[Any])
    assert_type(sample, Any)

    assert len(result) == 7
    assert len(sampler) == 7

    # FIXME: https://github.com/pandas-dev/pandas/issues/56021
    match mode:
        case MODE.BOUNDS:
            assert isinstance(sample, tuple)
            assert isinstance(sample[0], np.floating), type(sample[0])
        case MODE.SLICE:
            assert isinstance(sample, slice)
            assert isinstance(sample.start, np.floating)
        case MODE.INTERVAL:
            assert isinstance(sample, pd.Interval)
            assert isinstance(sample.left, np.floating), type(sample.left)
        case MODE.POINTS:
            assert isinstance(sample, np.ndarray)
            assert isinstance(sample[0], np.floating), type(sample[0])
        case MODE.MASK:
            assert isinstance(sample, np.ndarray)
            assert np.issubdtype(sample.dtype, np.bool_)
        case MODE.INDEX:
            assert isinstance(sample, np.ndarray)
            assert isinstance(sample[0], np.integer), type(sample[0])
        case _:
            assert_never(mode)


# region specific tests ----------------------------------------------------------------
# NOTE: here we have statically known mode, so they should be type checked.
def test_pandas_timestamps() -> None:
    r"""Test the SlidingWindowSampler."""
    timedeltas = pd.Series(pd.to_timedelta(RNG.uniform(size=200), "m"))
    tmin = pd.Timestamp(0)
    time = pd.concat(
        [
            pd.Series([tmin]),
            tmin + timedeltas.cumsum(),
        ]
    ).reset_index(drop=True)
    sampler = SlidingWindowSampler(
        time,
        stride="5m",
        horizons="15m",
        mode="bounds",
        shuffle=False,
        drop_last=False,
    )

    result = list(sampler)
    assert_type(sampler, SlidingWindowSampler[pd.Timestamp, B, ONE])
    assert_type(sampler.mode, B)
    assert_type(result, list[tuple[pd.Timestamp, pd.Timestamp]])

    assert isinstance(result, list)
    assert isinstance(result[0], tuple)
    assert isinstance(result[0][0], pd.Timestamp)
    assert isinstance(result[0][1], pd.Timestamp)


def test_points_single() -> None:
    r"""Test the SlidingWindowSampler."""
    sampler = SlidingWindowSampler(
        DISCRETE_DATA,
        stride=2,
        horizons=3,
        mode="points",
        shuffle=False,
        drop_last=False,
    )
    result = list(sampler)
    sample = result[0]

    # check that static types are correct
    assert_type(sampler, SlidingWindowSampler[int, P, ONE])
    assert_type(iter(sampler), Iterator[NDArray])
    assert_type(result, list[NDArray])
    assert_type(sample, NDArray)

    for m1, m2 in zip(
        result,
        [
            np.array([11, 12, 13]),
            np.array([13, 14, 15]),
            np.array([15, 16, 17]),
            np.array([17, 18, 19]),
            np.array([19, 20]),
        ],
        strict=True,
    ):
        assert np.array_equal(m1, m2)

    # try with drop_last=True
    sampler.drop_last = True
    for m1, m2 in zip(
        sampler,
        [
            np.array([11, 12, 13]),
            np.array([13, 14, 15]),
            np.array([15, 16, 17]),
            np.array([17, 18, 19]),
        ],
        strict=True,
    ):
        assert np.array_equal(m1, m2)


def test_points_multi() -> None:
    r"""Test the SlidingWindowSampler."""
    sampler = SlidingWindowSampler(
        DISCRETE_DATA,
        stride=2,
        horizons=[3, 1],
        mode="points",
        shuffle=False,
        drop_last=False,
    )
    result = list(sampler)
    sample = result[0]

    # check that static types are correct
    assert_type(sampler, SlidingWindowSampler[int, P, MULTI])
    assert_type(sampler.mode, P)
    assert_type(iter(sampler), Iterator[list[NDArray]])
    assert_type(result, list[list[NDArray]])
    assert_type(sample, list[NDArray])
    assert_type(sample[0], NDArray)

    assert all(
        np.array_equal(m1[0], m2[0]) and np.array_equal(m1[1], m2[1])
        for m1, m2 in zip(
            result,
            [
                [np.array([11, 12, 13]), np.array([14])],
                [np.array([13, 14, 15]), np.array([16])],
                [np.array([15, 16, 17]), np.array([18])],
                [np.array([17, 18, 19]), np.array([20])],
            ],
            strict=True,
        )
    )


def test_masks_single() -> None:
    r"""Test the SlidingWindowSampler."""
    sampler = SlidingWindowSampler(
        DISCRETE_DATA,
        stride=2,
        horizons=3,
        mode="mask",
        shuffle=False,
        drop_last=False,
    )
    result = list(sampler)
    sample = result[0]

    # check that static types are correct
    assert_type(sampler, SlidingWindowSampler[int, M, ONE])
    assert_type(sampler.mode, M)
    assert_type(iter(sampler), Iterator[NDArray[np.bool_]])
    assert_type(result, list[NDArray[np.bool_]])
    assert_type(sample, NDArray[np.bool_])

    assert all(
        np.array_equal(m1, m2)
        for m1, m2 in zip(
            result,
            [
                np.array([Y, Y, Y, N, N, N, N, N, N, N]),
                np.array([N, N, Y, Y, Y, N, N, N, N, N]),
                np.array([N, N, N, N, Y, Y, Y, N, N, N]),
                np.array([N, N, N, N, N, N, Y, Y, Y, N]),
                np.array([N, N, N, N, N, N, N, N, Y, Y]),
            ],
            strict=True,
        )
    )

    # try with drop_last=True
    sampler.drop_last = True
    assert all(
        np.array_equal(m1, m2)
        for m1, m2 in zip(
            sampler,
            [
                np.array([Y, Y, Y, N, N, N, N, N, N, N]),
                np.array([N, N, Y, Y, Y, N, N, N, N, N]),
                np.array([N, N, N, N, Y, Y, Y, N, N, N]),
                np.array([N, N, N, N, N, N, Y, Y, Y, N]),
            ],
            strict=True,
        )
    )


def test_masks_multi() -> None:
    r"""Test the SlidingWindowSampler."""
    sampler = SlidingWindowSampler(
        DISCRETE_DATA,
        stride=2,
        horizons=(3, 1),
        mode="mask",
        shuffle=False,
        drop_last=False,
    )
    result = list(sampler)
    sample = result[0]

    # check that static types are correct
    assert_type(sampler, SlidingWindowSampler[int, M, MULTI])
    assert_type(sampler.mode, M)
    assert_type(iter(sampler), Iterator[list[NDArray[np.bool_]]])
    assert_type(result, list[list[NDArray[np.bool_]]])
    assert_type(sample, list[NDArray[np.bool_]])
    assert_type(sample[0], NDArray[np.bool_])

    assert all(
        np.array_equal(m1[0], m2[0]) and np.array_equal(m1[1], m2[1])
        for m1, m2 in zip(
            result,
            [
                [
                    np.array([Y, Y, Y, N, N, N, N, N, N, N]),
                    np.array([N, N, N, Y, N, N, N, N, N, N]),
                ],
                [
                    np.array([N, N, Y, Y, Y, N, N, N, N, N]),
                    np.array([N, N, N, N, N, Y, N, N, N, N]),
                ],
                [
                    np.array([N, N, N, N, Y, Y, Y, N, N, N]),
                    np.array([N, N, N, N, N, N, N, Y, N, N]),
                ],
                [
                    np.array([N, N, N, N, N, N, Y, Y, Y, N]),
                    np.array([N, N, N, N, N, N, N, N, N, Y]),
                ],
            ],
            strict=True,
        )
    )


def test_bounds_single() -> None:
    r"""Test the SlidingWindowSampler."""
    sampler = SlidingWindowSampler(
        DISCRETE_DATA,
        stride=2,
        horizons=3,
        mode="bounds",
        shuffle=False,
        drop_last=False,
    )
    result = list(sampler)
    sample = result[0]

    # check that static types are correct
    assert_type(sampler, SlidingWindowSampler[int, B, ONE])
    assert_type(sampler.mode, B)
    assert_type(iter(sampler), Iterator[tuple[int, int]])
    assert_type(result, list[tuple[int, int]])
    assert_type(sample, tuple[int, int])

    assert result == [
        (11, 14),
        (13, 16),
        (15, 18),
        (17, 20),
        (19, 22),
    ]

    # try with drop_last=True
    sampler.drop_last = True
    assert list(sampler) == [
        (11, 14),
        (13, 16),
        (15, 18),
        (17, 20),
    ]


def test_bounds_multi() -> None:
    r"""Test the SlidingWindowSampler."""
    sampler = SlidingWindowSampler(
        DISCRETE_DATA,
        stride=2,
        horizons=(3, 1),
        mode="bounds",
        shuffle=False,
        drop_last=False,
    )
    result = list(sampler)
    sample = result[0]

    # check that static types are correct
    assert_type(sampler, SlidingWindowSampler[int, B, MULTI])
    assert_type(sampler.mode, B)
    assert_type(iter(sampler), Iterator[list[tuple[int, int]]])
    assert_type(result, list[list[tuple[int, int]]])
    assert_type(sample, list[tuple[int, int]])
    assert_type(sample[0], tuple[int, int])

    assert result == [
        [(11, 14), (14, 15)],
        [(13, 16), (16, 17)],
        [(15, 18), (18, 19)],
        [(17, 20), (20, 21)],
    ]


def test_slices_single() -> None:
    r"""Test the SlidingWindowSampler."""
    sampler = SlidingWindowSampler(
        DISCRETE_DATA,
        stride=2,
        horizons=3,
        mode="slice",
        shuffle=False,
        drop_last=False,
    )
    result = list(sampler)
    sample = result[0]

    # check that static types are correct
    assert_type(sampler, SlidingWindowSampler[int, S, ONE])
    assert_type(sampler.mode, S)
    assert_type(result, list["slice[int, int, int]"])
    assert_type(sample, "slice[int, int, int]")

    assert result == [
        slice(11, 14, None),
        slice(13, 16, None),
        slice(15, 18, None),
        slice(17, 20, None),
        slice(19, 22, None),
    ]

    # try with drop_last=True
    sampler.drop_last = True
    assert list(sampler) == [
        slice(11, 14, None),
        slice(13, 16, None),
        slice(15, 18, None),
        slice(17, 20, None),
    ]


def test_slices_multi() -> None:
    r"""Test the SlidingWindowSampler."""
    sampler = SlidingWindowSampler(
        DISCRETE_DATA,
        stride=2,
        horizons=(3, 1),
        mode="slice",
        shuffle=False,
        drop_last=False,
    )
    result = list(sampler)
    sample = result[0]

    # check that static types are correct
    assert_type(sampler, SlidingWindowSampler[int, S, MULTI])
    assert_type(sampler.mode, S)
    assert_type(result, list[list["slice[int, int, int]"]])
    assert_type(sample, list["slice[int, int, int]"])
    assert_type(sample[0], "slice[int, int, int]")

    assert result == [
        [slice(11, 14, None), slice(14, 15, None)],
        [slice(13, 16, None), slice(16, 17, None)],
        [slice(15, 18, None), slice(18, 19, None)],
        [slice(17, 20, None), slice(20, 21, None)],
    ]


def test_unknown_single() -> None:
    r"""Test the SlidingWindowSampler."""
    sampler = SlidingWindowSampler(
        DISCRETE_DATA,
        stride=2,
        horizons=3,
        mode=str("slice"),  # str() tricks type checker into not using literal
        shuffle=False,
        drop_last=False,
    )
    assert_type(sampler, SlidingWindowSampler[int, MODE, ONE])
    assert_type(sampler.mode, MODE)
    assert_type(iter(sampler), Iterator[Any])


def test_unknown_multi() -> None:
    r"""Test the SlidingWindowSampler."""
    sampler = SlidingWindowSampler(
        DISCRETE_DATA,
        stride=2,
        horizons=(3, 1),
        mode=str("mask"),  # str() tricks type checker into not using literal
        shuffle=False,
        drop_last=False,
    )
    assert_type(sampler, SlidingWindowSampler[int, MODE, MULTI])
    assert_type(sampler.mode, MODE)
    assert_type(iter(sampler), Iterator[list[Any]])


# endregion specific tests -------------------------------------------------------------


def type_slidingsampler_assignable() -> None:
    int_list: list[int] = [1, 2, 3]

    assert_type(
        SlidingWindowSampler(int_list, horizons=2, stride=2, mode=MODE.SLICE),
        SlidingWindowSampler[int, S, ONE],
    )
    assert_type(
        SlidingWindowSampler(int_list, horizons=[1, 2], stride=2, mode=MODE.MASK),
        SlidingWindowSampler[int, M, MULTI],
    )
