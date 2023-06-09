#!/usr/bin/env python
r"""Test Sliding Window Sampler."""

import logging

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from tsdm.random.samplers import SlidingWindowSampler

logging.basicConfig(level=logging.INFO)
__logger__ = logging.getLogger(__name__)


def test_SlidingWindowSampler():
    """Test the SlidingWindowSampler."""
    LOGGER = __logger__.getChild(SlidingWindowSampler.__name__)
    LOGGER.info("Testing.")

    tds = Series(pd.to_timedelta(np.random.rand(200), "m"))
    tmin = pd.Timestamp(0)
    tmax = tmin + pd.Timedelta(2, "h")
    T = pd.concat([Series([tmin]), tmin + tds.cumsum(), Series([tmax])])
    T = T.reset_index(drop=True)

    stride = "5m"
    # mode = "points"
    horizons = "15m"
    shuffle = False

    sampler = SlidingWindowSampler(
        T, stride=stride, horizons=horizons, mode="points", shuffle=shuffle
    )
    indices = list(sampler)
    X = DataFrame(np.random.randn(len(T), 2), columns=["ch1", "ch2"], index=T)
    assert len(indices) >= 0 and len(X) > 0  # TODO: implement test
    # samples = X.loc[indices]


def _main() -> None:
    test_SlidingWindowSampler()


if __name__ == "__main__":
    _main()
