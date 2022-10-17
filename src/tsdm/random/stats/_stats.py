r"""Utility functions to get statistics from dataset."""

__all__ = [
    # Functions
    "data_overview",
    "sparsity",
]

from collections.abc import Hashable
from typing import Optional

import pandas
from pandas import DataFrame, Series


def sparsity(df: DataFrame) -> tuple[float, float]:
    r"""Quantify sparsity in the data."""
    mask = pandas.isna(df)
    col_wise = mask.mean(axis=0)
    total = mask.mean()
    return col_wise, total


# def linearness():
#     r"""Quantify linear signals in the data using regularized least-squares."""
#
#
# def periodicity():
#     r"""Quantify periodic signals in the data using (Non-Uniform) FFT in O(N log N)."""
#
#
# def summary_stats():
#     r"""Summary statistics: column-wise mean/median/std/histogram. Cross-channel correlation."""


def data_overview(
    df: DataFrame, index_col: Optional[Hashable] = None, digits: int = 2
) -> DataFrame:
    r"""Get a summary of the data."""
    overview = DataFrame(index=df.columns)
    mask = df.isna()
    numerical_cols = df.select_dtypes(include="number").columns
    # float_cols = df.select_dtypes(include="float").columns
    # other_cols = df.select_dtypes(exclude="number").columns

    overview["datapoints"] = (~mask).sum()
    overview["missing"] = (mask.mean() * 100).round(2)

    overview.loc[numerical_cols, "min"] = df[numerical_cols].min()
    overview.loc[numerical_cols, "mean"] = df[numerical_cols].mean()
    overview.loc[numerical_cols, "std"] = df[numerical_cols].std()
    overview.loc[numerical_cols, "max"] = df[numerical_cols].max()

    # fmt: off
    column_dtypes = {
        "datapoints" : "Int64",
        "missing"    : "Float64",
        "min"        : "Float64",
        "mean"       : "Float64",
        "std"        : "Float64",
        "max"        : "Float64",
    }
    # fmt: on

    overview = overview.astype(column_dtypes)
    for col, dtype in column_dtypes.items():
        if pandas.api.types.is_float_dtype(dtype):
            overview[col] = overview[col].round(digits)

    if index_col is not None:
        freq = {}
        for col in df:
            mask = pandas.notna(df[col].squeeze())
            time = df.get_level_values(index_col)[mask]
            freq[col] = Series(time).diff().mean()
        overview["freq"] = Series(freq)
    return overview
