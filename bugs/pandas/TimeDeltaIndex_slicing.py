#!/usr/bin/env python
# https://github.com/pandas-dev/pandas/issues/33603

from datetime import timedelta

import numpy as np
import pandas as pd

# create 24h range with 1 min spacing
td = pd.timedelta_range(start="0h", end="24h", freq="1min").to_series()

# last timedelta included in slice
print(td[: timedelta(hours=1)].iloc[-1])  # 01:00 ✔
print(td[: np.timedelta64(1, "h")].iloc[-1])  # 01:00 ✔
print(td[: pd.Timedelta(1, "h")].iloc[-1])  # 01:00 ✔
print(td[: pd.Timedelta("1h")].iloc[-1])  # 01:00 ✔
print(td[:"1h"].iloc[-1])  # 01:59 ✘

# with loc
print(td.loc[: timedelta(hours=1)].iloc[-1])  # 01:00 ✔
print(td.loc[: np.timedelta64(1, "h")].iloc[-1])  # 01:00 ✔
print(td.loc[: pd.Timedelta(1, "h")].iloc[-1])  # 01:00 ✔
print(td.loc[: pd.Timedelta("1h")].iloc[-1])  # 01:00 ✔
print(td.loc[:"1h"].iloc[-1])  # 01:59 ✘
