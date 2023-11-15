r"""Test Sliding Window Sampler."""

import logging
from typing import assert_type

import numpy as np
import pandas as pd
from pandas import Series
from pytest import mark

from tsdm.random.samplers import SlidingWindowSampler
from tsdm.utils import flatten_dict

__logger__ = logging.getLogger(__name__)

T = True
F = False

DISCRETE_DATA = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# region expected results discrete data ------------------------------------------------
EXPECTED_RESULTS_DISCRETE_BOUNDS = {
    # fmt: off
    # horizons, stride=1, drop_last=True
    (2, 1, True): [(11, 13), (12, 14), (13, 15), (14, 16), (15, 17), (16, 18), (17, 19), (18, 20)],
    (3, 1, True): [(11, 14), (12, 15), (13, 16), (14, 17), (15, 18), (16, 19), (17, 20)],
    (4, 1, True): [(11, 15), (12, 16), (13, 17), (14, 18), (15, 19), (16, 20)],
    # horizons, stride=2, drop_last=True
    (2, 2, True): [(11, 13), (13, 15), (15, 17), (17, 19)],
    (3, 2, True): [(11, 14), (13, 16), (15, 18), (17, 20)],
    (4, 2, True): [(11, 15), (13, 17), (15, 19)],
    # horizons, stride=1, drop_last=False
    (2, 1, False): [(11, 13), (12, 14), (13, 15), (14, 16), (15, 17), (16, 18), (17, 19), (18, 20), (19, 21), (20, 22)],
    (3, 1, False): [(11, 14), (12, 15), (13, 16), (14, 17), (15, 18), (16, 19), (17, 20), (18, 21), (19, 22), (20, 23)],
    (4, 1, False): [(11, 15), (12, 16), (13, 17), (14, 18), (15, 19), (16, 20), (17, 21), (18, 22), (19, 23), (20, 24)],
    # horizons, stride=2, drop_last=False
    (2, 2, False): [(11, 13), (13, 15), (15, 17), (17, 19), (19, 21)],
    (3, 2, False): [(11, 14), (13, 16), (15, 18), (17, 20), (19, 22)],
    (4, 2, False): [(11, 15), (13, 17), (15, 19), (17, 21), (19, 23)],
    # fmt: on
}

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
        np.array([T, T, F, F, F, F, F, F, F, F]),  # (11, 13)
        np.array([F, T, T, F, F, F, F, F, F, F]),  # (12, 14)
        np.array([F, F, T, T, F, F, F, F, F, F]),  # (13, 15)
        np.array([F, F, F, T, T, F, F, F, F, F]),  # (14, 16)
        np.array([F, F, F, F, T, T, F, F, F, F]),  # (15, 17)
        np.array([F, F, F, F, F, T, T, F, F, F]),  # (16, 18)
        np.array([F, F, F, F, F, F, T, T, F, F]),  # (17, 19)
        np.array([F, F, F, F, F, F, F, T, T, F]),  # (18, 20)
        # np.array([F, F, F, F, F, F, F, F, T, T]),
    ],
    (3, 1, True): [
        np.array([T, T, T, F, F, F, F, F, F, F]),  # (11, 14)
        np.array([F, T, T, T, F, F, F, F, F, F]),  # (12, 15)
        np.array([F, F, T, T, T, F, F, F, F, F]),  # (13, 16)
        np.array([F, F, F, T, T, T, F, F, F, F]),  # (14, 17)
        np.array([F, F, F, F, T, T, T, F, F, F]),  # (15, 18)
        np.array([F, F, F, F, F, T, T, T, F, F]),  # (16, 19)
        np.array([F, F, F, F, F, F, T, T, T, F]),  # (17, 20)
        # np.array([F, F, F, F, F, F, F, T, T, T]),
    ],
    (4, 1, True): [
        np.array([T, T, T, T, F, F, F, F, F, F]),  # (11, 15)
        np.array([F, T, T, T, T, F, F, F, F, F]),  # (12, 16)
        np.array([F, F, T, T, T, T, F, F, F, F]),  # (13, 17)
        np.array([F, F, F, T, T, T, T, F, F, F]),  # (14, 18)
        np.array([F, F, F, F, T, T, T, T, F, F]),  # (15, 19)
        np.array([F, F, F, F, F, T, T, T, T, F]),  # (16, 20)
        # np.array([F, F, F, F, F, F, T, T, T, T]),
    ],
    # horizons, stride=2, drop_last=True
    (2, 2, True): [
        np.array([T, T, F, F, F, F, F, F, F, F]),  # (11, 13)
        np.array([F, F, T, T, F, F, F, F, F, F]),  # (13, 15)
        np.array([F, F, F, F, T, T, F, F, F, F]),  # (15, 17)
        np.array([F, F, F, F, F, F, T, T, F, F]),  # (17, 19)
        # np.array([F, F, F, F, F, F, F, F, T, T]),
    ],
    (3, 2, True): [
        np.array([T, T, T, F, F, F, F, F, F, F]),  # (11, 14)
        np.array([F, F, T, T, T, F, F, F, F, F]),  # (13, 16)
        np.array([F, F, F, F, T, T, T, F, F, F]),  # (15, 18)
        np.array([F, F, F, F, F, F, T, T, T, F]),  # (17, 20)
    ],
    (4, 2, True): [
        np.array([T, T, T, T, F, F, F, F, F, F]),  # (11, 15)
        np.array([F, F, T, T, T, T, F, F, F, F]),  # (13, 17)
        np.array([F, F, F, F, T, T, T, T, F, F]),  # (15, 19)
        # np.array([F, F, F, F, F, F, T, T, T, T]),
    ],
    # horizons, stride=1, drop_last=False
    (2, 1, False): [
        np.array([T, T, F, F, F, F, F, F, F, F]),
        np.array([F, T, T, F, F, F, F, F, F, F]),
        np.array([F, F, T, T, F, F, F, F, F, F]),
        np.array([F, F, F, T, T, F, F, F, F, F]),
        np.array([F, F, F, F, T, T, F, F, F, F]),
        np.array([F, F, F, F, F, T, T, F, F, F]),
        np.array([F, F, F, F, F, F, T, T, F, F]),
        np.array([F, F, F, F, F, F, F, T, T, F]),
        np.array([F, F, F, F, F, F, F, F, T, T]),
        np.array([F, F, F, F, F, F, F, F, F, T]),
    ],
    (3, 1, False): [
        np.array([T, T, T, F, F, F, F, F, F, F]),
        np.array([F, T, T, T, F, F, F, F, F, F]),
        np.array([F, F, T, T, T, F, F, F, F, F]),
        np.array([F, F, F, T, T, T, F, F, F, F]),
        np.array([F, F, F, F, T, T, T, F, F, F]),
        np.array([F, F, F, F, F, T, T, T, F, F]),
        np.array([F, F, F, F, F, F, T, T, T, F]),
        np.array([F, F, F, F, F, F, F, T, T, T]),
        np.array([F, F, F, F, F, F, F, F, T, T]),
        np.array([F, F, F, F, F, F, F, F, F, T]),
    ],
    (4, 1, False): [
        np.array([T, T, T, T, F, F, F, F, F, F]),
        np.array([F, T, T, T, T, F, F, F, F, F]),
        np.array([F, F, T, T, T, T, F, F, F, F]),
        np.array([F, F, F, T, T, T, T, F, F, F]),
        np.array([F, F, F, F, T, T, T, T, F, F]),
        np.array([F, F, F, F, F, T, T, T, T, F]),
        np.array([F, F, F, F, F, F, T, T, T, T]),
        np.array([F, F, F, F, F, F, F, T, T, T]),
        np.array([F, F, F, F, F, F, F, F, T, T]),
        np.array([F, F, F, F, F, F, F, F, F, T]),
    ],
    # horizons, stride=2, drop_last=False
    (2, 2, False): [
        np.array([T, T, F, F, F, F, F, F, F, F]),
        np.array([F, F, T, T, F, F, F, F, F, F]),
        np.array([F, F, F, F, T, T, F, F, F, F]),
        np.array([F, F, F, F, F, F, T, T, F, F]),
        np.array([F, F, F, F, F, F, F, F, T, T]),
    ],
    (3, 2, False): [
        np.array([T, T, T, F, F, F, F, F, F, F]),
        np.array([F, F, T, T, T, F, F, F, F, F]),
        np.array([F, F, F, F, T, T, T, F, F, F]),
        np.array([F, F, F, F, F, F, T, T, T, F]),
        np.array([F, F, F, F, F, F, F, F, T, T]),
    ],
    (4, 2, False): [
        np.array([T, T, T, T, F, F, F, F, F, F]),
        np.array([F, F, T, T, T, T, F, F, F, F]),
        np.array([F, F, F, F, T, T, T, T, F, F]),
        np.array([F, F, F, F, F, F, T, T, T, T]),
        np.array([F, F, F, F, F, F, F, F, T, T]),
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
        # np.array([19, 20]),
    ],
    (3, 1, True): [
        np.array([11, 12, 13]),  # (11, 14)
        np.array([12, 13, 14]),  # (12, 15)
        np.array([13, 14, 15]),  # (13, 16)
        np.array([14, 15, 16]),  # (14, 17)
        np.array([15, 16, 17]),  # (15, 18)
        np.array([16, 17, 18]),  # (16, 19)
        np.array([17, 18, 19]),  # (17, 20)
        # np.array([18, 19, 20]),
    ],
    (4, 1, True): [
        np.array([11, 12, 13, 14]),  # (11, 15)
        np.array([12, 13, 14, 15]),  # (12, 16)
        np.array([13, 14, 15, 16]),  # (13, 17)
        np.array([14, 15, 16, 17]),  # (14, 18)
        np.array([15, 16, 17, 18]),  # (15, 19)
        np.array([16, 17, 18, 19]),  # (16, 20)
        # np.array([17, 18, 19, 20]),
    ],
    # horizons, stride=2, drop_last=True
    (2, 2, True): [
        np.array([11, 12]),  # (11, 13)
        np.array([13, 14]),  # (13, 15)
        np.array([15, 16]),  # (15, 17)
        np.array([17, 18]),  # (17, 19)
        # np.array([19, 20]),
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
        # np.array([17, 18, 19, 20]),
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
EXPECTED_RESULTS_DISCRETE_DATA = flatten_dict(
    {
        "bounds": EXPECTED_RESULTS_DISCRETE_BOUNDS,
        "masks": EXPECTED_RESULTS_DISCRETE_MASKS,
        "slices": EXPECTED_RESULTS_DISCRETE_SLICES,
        "windows": EXPECTED_RESULTS_DISCRETE_WINDOWS,
    },
    join_fn=tuple,
    split_fn=lambda x: x,
    recursive=1,
)
# NOTE that we include duplicates in the continuous data
# We test horizons ∈ {2.5, 3.5, 4.5}, stride ∈ {1.0, 2.5}, drop_last ∈ {True, False}
CONTINUOUS_DATA = [2.5, 3.3, 3.7, 4.0, 5.9, 6.4, 6.4, 6.6, 7.5, 8.9]
# region expected results continuous data ----------------------------------------------
EXPECTED_RESULTS_CONTINUOUS_BOUNDS = {
    # fmt: off
    # horizons, stride=1.0, drop_last=True
    (2.5, 1.0, True): [(2.5, 5.0), (3.5, 6.0), (4.5, 7.0), (5.5, 8.0)],
    (3.5, 1.0, True): [(2.5, 6.0), (3.5, 7.0), (4.5, 8.0)],
    (4.5, 1.0, True): [(2.5, 7.0), (3.5, 8.0)],
    # horizons, stride=2.5, drop_last=True
    (2.5, 2.5, True): [(2.5, 5.0), (5.0, 7.5)],
    (3.5, 2.5, True): [(2.5, 6.0), (5.0, 8.5)],
    (4.5, 2.5, True): [(2.5, 7.0)],
    # horizons, stride=1.0, drop_last=False
    (2.5, 1.0, False): [(2.5, 5.0), (3.5, 6.0), (4.5, 7.0), (5.5, 8.0), (6.5, 9.0), (7.5, 10.0), (8.5, 11.0)],
    (3.5, 1.0, False): [(2.5, 6.0), (3.5, 7.0), (4.5, 8.0), (5.5, 9.0), (6.5, 10.0), (7.5, 11.0), (8.5, 12.0)],
    (4.5, 1.0, False): [(2.5, 7.0), (3.5, 8.0), (4.5, 9.0), (5.5, 10.0), (6.5, 11.0), (7.5, 12.0), (8.5, 13.0)],
    # horizons, stride=2.5, drop_last=False
    (2.5, 2.5, False): [(2.5, 5.0), (5.0, 7.5), (7.5, 10.0)],
    (3.5, 2.5, False): [(2.5, 6.0), (5.0, 8.5), (7.5, 11.0)],
    (4.5, 2.5, False): [(2.5, 7.0), (5.0, 9.5), (7.5, 12.0)],
    # fmt: on
}
EXPECTED_RESULTS_CONTINUOUS_WINDOWS = {
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
EXPECTED_RESULTS_CONTINUOUS_MASKS = {
    # horizons, stride=1.0, drop_last=True
    (2.5, 1.0, True): [
        np.array([T, T, T, T, F, F, F, F, F, F]),  # (2.5, 5.0)
        np.array([F, F, T, T, T, F, F, F, F, F]),  # (3.5, 6.0)
        np.array([F, F, F, F, T, T, T, T, F, F]),  # (4.5, 7.0)
        np.array([F, F, F, F, T, T, T, T, T, F]),  # (5.5, 8.0)
    ],
    (3.5, 1.0, True): [
        np.array([T, T, T, T, T, F, F, F, F, F]),  # (2.5, 6.0)
        np.array([F, F, T, T, T, T, T, T, F, F]),  # (3.5, 7.0)
        np.array([F, F, F, F, T, T, T, T, T, F]),  # (4.5, 8.0)
    ],
    (4.5, 1.0, True): [
        np.array([T, T, T, T, T, T, T, T, F, F]),  # (2.5, 7.0)
        np.array([F, F, T, T, T, T, T, T, T, F]),  # (3.5, 8.0)
    ],
    # horizons, stride=2.5, drop_last=True
    (2.5, 2.5, True): [
        np.array([T, T, T, T, F, F, F, F, F, F]),  # (2.5, 5.0)
        np.array([F, F, F, F, T, T, T, T, F, F]),  # (5.0, 7.5)
    ],
    (3.5, 2.5, True): [
        np.array([T, T, T, T, T, F, F, F, F, F]),  # (2.5, 6.0)
        np.array([F, F, F, F, T, T, T, T, T, F]),  # (5.0, 8.5)
    ],
    (4.5, 2.5, True): [
        np.array([T, T, T, T, T, T, T, T, F, F]),  # (2.5, 7.0)
    ],
    # horizons, stride=1.0, drop_last=False
    (2.5, 1.0, False): [
        np.array([T, T, T, T, F, F, F, F, F, F]),  # (2.5, 5.0)
        np.array([F, F, T, T, T, F, F, F, F, F]),  # (3.5, 6.0)
        np.array([F, F, F, F, T, T, T, T, F, F]),  # (4.5, 7.0)
        np.array([F, F, F, F, T, T, T, T, T, F]),  # (5.5, 8.0)
        np.array([F, F, F, F, F, F, F, T, T, T]),  # (6.5, 9.0)
        np.array([F, F, F, F, F, F, F, F, T, T]),  # (7.5, 10.0)
        np.array([F, F, F, F, F, F, F, F, F, T]),  # (8.5, 11.0)
    ],
    (3.5, 1.0, False): [
        np.array([T, T, T, T, T, F, F, F, F, F]),  # (2.5, 6.0)
        np.array([F, F, T, T, T, T, T, T, F, F]),  # (3.5, 7.0)
        np.array([F, F, F, F, T, T, T, T, T, F]),  # (4.5, 8.0)
        np.array([F, F, F, F, T, T, T, T, T, T]),  # (5.5, 9.0)
        np.array([F, F, F, F, F, F, F, T, T, T]),  # (6.5, 10.0)
        np.array([F, F, F, F, F, F, F, F, T, T]),  # (7.5, 11.0)
        np.array([F, F, F, F, F, F, F, F, F, T]),  # (8.5, 12.0)
    ],
    (4.5, 1.0, False): [
        np.array([T, T, T, T, T, T, T, T, F, F]),  # (2.5, 7.0),
        np.array([F, F, T, T, T, T, T, T, T, F]),  # (3.5, 8.0),
        np.array([F, F, F, F, T, T, T, T, T, T]),  # (4.5, 9.0),
        np.array([F, F, F, F, T, T, T, T, T, T]),  # (5.5, 10.0),
        np.array([F, F, F, F, F, F, F, T, T, T]),  # (6.5, 11.0),
        np.array([F, F, F, F, F, F, F, F, T, T]),  # (7.5, 12.0),
        np.array([F, F, F, F, F, F, F, F, F, T]),  # (8.5, 13.0),
    ],
    # horizons, stride=2.5, drop_last=False
    (2.5, 2.5, False): [
        np.array([T, T, T, T, F, F, F, F, F, F]),  # (2.5, 5.0)
        np.array([F, F, F, F, T, T, T, T, F, F]),  # (5.0, 7.5)
        np.array([F, F, F, F, F, F, F, F, T, T]),  # (7.5, 10.0)
    ],
    (3.5, 2.5, False): [
        np.array([T, T, T, T, T, F, F, F, F, F]),  # (2.5, 6.0)
        np.array([F, F, F, F, T, T, T, T, T, F]),  # (5.0, 8.5)
        np.array([F, F, F, F, F, F, F, F, T, T]),  # (7.5, 11.0)
    ],
    (4.5, 2.5, False): [
        np.array([T, T, T, T, T, T, T, T, F, F]),  # (2.5, 7.0)
        np.array([F, F, F, F, T, T, T, T, T, T]),  # (5.0, 9.5)
        np.array([F, F, F, F, F, F, F, F, T, T]),  # (7.5, 12.0)
    ],
}
EXPECTED_RESULTS_CONTINUOUS_SLICES = {
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
EXPECTED_RESULTS_CONTINUOUS_DATA = flatten_dict(
    {
        "bounds": EXPECTED_RESULTS_CONTINUOUS_BOUNDS,
        "masks": EXPECTED_RESULTS_CONTINUOUS_MASKS,
        "slices": EXPECTED_RESULTS_CONTINUOUS_SLICES,
        "windows": EXPECTED_RESULTS_CONTINUOUS_WINDOWS,
    },
    join_fn=tuple,
    split_fn=lambda x: x,
    recursive=1,
)


# write parametrized unit test with the above data for all modes
@mark.parametrize("drop_last", [False, True], ids=lambda x: f"drop_last={x}")
@mark.parametrize("stride", [1, 2], ids=lambda x: f"stride={x}")
@mark.parametrize("horizons", [2, 3, 4], ids=lambda x: f"horizon={x}")
@mark.parametrize("mode", ["bounds", "masks", "slices", "windows"])
def test_sliding_window_sampler_discrete(
    drop_last: bool,
    stride: int,
    horizons: int,
    mode: str,
) -> None:
    """Test the SlidingWindowSampler."""
    sampler = SlidingWindowSampler(
        DISCRETE_DATA,
        stride=stride,
        horizons=horizons,
        mode=mode,
        shuffle=False,
        drop_last=drop_last,
    )
    assert_type(sampler, SlidingWindowSampler)

    expected = EXPECTED_RESULTS_DISCRETE_DATA[mode, horizons, stride, drop_last]

    assert len(sampler) == len(expected), (
        "LENGTH MISMATCH!"
        f"\nsampler:\n{list(sampler)}\nexpected:\n{expected}\ngrid={sampler.grid}\n"
    )

    for m1, m2 in zip(sampler, expected, strict=True):
        assert np.array_equal(
            m1, m2
        ), f"SAMPLE MISMATCH!sample:\n{m1}\nexpected:\n{m2}\ngrid={sampler.grid}\n"


# write parametrized unit test with the above data for all modes
@mark.parametrize("drop_last", [False, True], ids=lambda x: f"drop_last={x}")
@mark.parametrize("stride", [1.0, 2.5], ids=lambda x: f"stride={x}")
@mark.parametrize("horizons", [2.5, 3.5, 4.5], ids=lambda x: f"horizon={x}")
@mark.parametrize("mode", ["bounds", "masks", "slices", "windows"])
def test_sliding_window_sampler_continuous(
    drop_last: bool,
    stride: int,
    horizons: int,
    mode: str,
) -> None:
    """Test the SlidingWindowSampler."""
    sampler = SlidingWindowSampler(
        CONTINUOUS_DATA,
        stride=stride,
        horizons=horizons,
        mode=mode,
        shuffle=False,
        drop_last=drop_last,
    )
    assert_type(sampler, SlidingWindowSampler)

    expected = EXPECTED_RESULTS_CONTINUOUS_DATA[mode, horizons, stride, drop_last]

    assert len(sampler) == len(expected), (
        "LENGTH MISMATCH!"
        f"\nsampler:\n{list(sampler)}\nexpected:\n{expected}\ngrid={sampler.grid}\n"
    )

    for m1, m2 in zip(sampler, expected, strict=True):
        assert np.array_equal(
            m1, m2
        ), f"SAMPLE MISMATCH!sample:\n{m1}\nexpected:\n{m2}\ngrid={sampler.grid}\n"


def test_pandas_timestamps():
    """Test the SlidingWindowSampler."""
    timedeltas = Series(pd.to_timedelta(np.random.rand(200), "m"))
    tmin = pd.Timestamp(0)
    time = pd.concat([
        Series([tmin]),
        tmin + timedeltas.cumsum(),
    ]).reset_index(drop=True)

    sampler = SlidingWindowSampler(
        time,
        stride="5m",
        horizons="15m",
        mode="bounds",
        shuffle=False,
        drop_last=False,
    )
    list(sampler)


def test_single_window() -> None:
    """Test the SlidingWindowSampler."""
    stride = 2
    horizons = 3

    sampler = SlidingWindowSampler(
        DISCRETE_DATA,
        stride=stride,
        horizons=horizons,
        mode="windows",
        shuffle=False,
        drop_last=False,
    )
    assert_type(sampler, SlidingWindowSampler)

    for m1, m2 in zip(
        sampler,
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


def test_single_slice() -> None:
    """Test the SlidingWindowSampler."""
    stride = 2
    horizons = 3

    sampler = SlidingWindowSampler(
        DISCRETE_DATA,
        stride=stride,
        horizons=horizons,
        mode="slices",
        shuffle=False,
        drop_last=False,
    )
    assert_type(sampler, SlidingWindowSampler)

    assert list(sampler) == [
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


def test_mode_windows_multi() -> None:
    """Test the SlidingWindowSampler."""
    stride = 2
    horizons = [3, 1]

    sampler = SlidingWindowSampler(
        DISCRETE_DATA, stride=stride, horizons=horizons, mode="windows", shuffle=False
    )
    assert_type(sampler, SlidingWindowSampler)

    print(list(sampler))

    assert all(
        np.array_equal(m1[0], m2[0]) and np.array_equal(m1[1], m2[1])
        for m1, m2 in zip(
            sampler,
            [
                [np.array([11, 12, 13]), np.array([14])],
                [np.array([13, 14, 15]), np.array([16])],
                [np.array([15, 16, 17]), np.array([18])],
                [np.array([17, 18, 19]), np.array([20])],
            ],
            strict=True,
        )
    )


def test_single_mask() -> None:
    """Test the SlidingWindowSampler."""
    stride = 2
    horizons = 3

    sampler = SlidingWindowSampler(
        DISCRETE_DATA,
        stride=stride,
        horizons=horizons,
        mode="masks",
        shuffle=False,
        drop_last=False,
    )
    assert_type(sampler, SlidingWindowSampler)

    assert all(
        np.array_equal(m1, m2)
        for m1, m2 in zip(
            sampler,
            [
                np.array([T, T, T, F, F, F, F, F, F, F]),
                np.array([F, F, T, T, T, F, F, F, F, F]),
                np.array([F, F, F, F, T, T, T, F, F, F]),
                np.array([F, F, F, F, F, F, T, T, T, F]),
                np.array([F, F, F, F, F, F, F, F, T, T]),
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
                np.array([T, T, T, F, F, F, F, F, F, F]),
                np.array([F, F, T, T, T, F, F, F, F, F]),
                np.array([F, F, F, F, T, T, T, F, F, F]),
                np.array([F, F, F, F, F, F, T, T, T, F]),
            ],
            strict=True,
        )
    )


def test_single_bounds() -> None:
    """Test the SlidingWindowSampler."""
    stride = 2
    horizons = 3

    sampler = SlidingWindowSampler(
        DISCRETE_DATA,
        stride=stride,
        horizons=horizons,
        mode="bounds",
        shuffle=False,
        drop_last=False,
    )
    assert_type(sampler, SlidingWindowSampler)

    assert list(sampler) == [
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


def test_mode_slices_multi() -> None:
    """Test the SlidingWindowSampler."""
    sampler = SlidingWindowSampler(
        DISCRETE_DATA,
        horizons=(3, 1),
        stride=2,
        mode="slices",
        shuffle=False,
        drop_last=False,
    )
    assert_type(sampler, SlidingWindowSampler)

    assert list(sampler) == [
        [slice(11, 14, None), slice(14, 15, None)],
        [slice(13, 16, None), slice(16, 17, None)],
        [slice(15, 18, None), slice(18, 19, None)],
        [slice(17, 20, None), slice(20, 21, None)],
    ]
