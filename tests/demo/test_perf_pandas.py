"""Test pandas boolean array performance."""

import operator
from functools import reduce

import numpy as np
import pandas as pd

rng = np.random.default_rng()
data = rng.normal(size=(10_000, 10)) > 0.5
df_numpy = pd.DataFrame(data, dtype=bool)
df_arrow = df_numpy.astype("bool[pyarrow]")


def test_numpy_columns():
    for _ in range(1000):
        df_numpy.all(axis="columns")


def test_numpy_rows():
    for _ in range(1000):
        df_numpy.all(axis="index")


def test_arrow_columns():
    for _ in range(1000):
        df_arrow.all(axis="columns")


def test_arrow_rows():
    for _ in range(1000):
        df_arrow.all(axis="index")


def test_arrow_rows_manual():
    for _ in range(1000):
        reduce(operator.__and__, (s for _, s in df_arrow.items()))
