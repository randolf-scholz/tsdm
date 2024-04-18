#!/usr/bin/env python

import operator
from functools import reduce

import numpy as np
import pandas as pd

data = np.random.randn(10_000, 10) > 0.5
df_numpy = pd.DataFrame(data, dtype=bool)
df_arrow = df_numpy.astype("bool[pyarrow]")


def test():
    for k in range(1000):
        df_arrow.all(axis="columns")


def test2():
    for k in range(1000):
        df_arrow.all(axis="index")


# %timeit df_numpy.all(axis="index")     # 216 µs ± 1.58 µs
# %timeit df_numpy.all(axis="columns")   # 274 µs ± 2.86 µs
# %timeit df_arrow.all(axis="index")     # 442 µs ± 3.07 µs
# %timeit df_arrow.all(axis="columns")   # 2.29 ms ± 6.59 µs
# %timeit reduce(operator.__and__, (s for _, s in df_arrow.items()))  # 362 µs ± 1.16 µs
