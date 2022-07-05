#!/usr/bin/env python
r"""Test PhysioNet 2012."""

# import logging
#
# import numpy as np
# import pandas as pd
# from pandas import DataFrame, Index, Series
# from pytest import mark
#
# from tsdm.random.samplers import SlidingWindowSampler
#
#
# def test_SlidingWindowSampler():
#     r"""Test PhysioNet 2012."""
#     tds = Series(pd.to_timedelta(np.random.rand(200), "m"))
#     tmin = pd.Timestamp(0)
#     tmax = tmin + pd.Timedelta(2, "h")
#     T = pd.concat([Series([tmin]), tmin + tds.cumsum(), Series([tmax])])
#     T = T.reset_index(drop=True)
#
#     stride = "5m"
#     mode = "points"
#     horizons = "15m"
#     shuffle = False
#
#     sampler = SlidingWindowSampler(
#         T, stride=stride, horizons=horizons, mode=mode, shuffle=shuffle
#     )
#     indices = list(sampler)
#
#     X = DataFrame(np.random.randn(len(T), 2), columns=["ch1", "ch2"], index=T)
#     # samples = X.loc[indices]
#
#
# def __main__():
#     logging.basicConfig(level=logging.INFO)
#     test_SlidingWindowSampler()
#
#
# if __name__ == "__main__":
#     __main__()
