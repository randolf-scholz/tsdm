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


def data_overview(df: DataFrame, index_col: Optional[Hashable] = None) -> DataFrame:
    r"""Get a summary of the data."""
    overview = DataFrame(index=df.columns)
    mask = df.isna()
    numerical_cols = df.select_dtypes(include="number").columns
    float_cols = df.select_dtypes(include="float").columns
    # other_cols = df.select_dtypes(exclude="number").columns

    overview["# datapoints"] = (~mask).sum()
    overview["% missing"] = (mask.mean() * 100).round(2)

    overview.loc[numerical_cols, "min"] = df[numerical_cols].min().round(2)
    overview.loc[numerical_cols, "mean"] = df[numerical_cols].mean().round(2)
    overview.loc[numerical_cols, "std"] = df[numerical_cols].std().round(2)
    overview.loc[numerical_cols, "max"] = df[numerical_cols].max().round(2)

    overview.loc[float_cols, "min"] = overview.loc[float_cols, "min"].round(2)
    overview.loc[float_cols, "mean"] = overview.loc[float_cols, "mean"].round(2)
    overview.loc[float_cols, "std"] = overview.loc[float_cols, "std"].round(2)
    overview.loc[float_cols, "max"] = overview.loc[float_cols, "max"].round(2)

    if index_col is not None:
        freq = {}
        for col in df:
            mask = pandas.notna(df[col].squeeze())
            time = df.get_level_values(index_col)[mask]
            freq[col] = Series(time).diff().mean()
        overview["freq"] = Series(freq)
    return overview
