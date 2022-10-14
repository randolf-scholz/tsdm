r"""Utility functions to get statistics from dataset."""

__all__ = [
    # Functions
    "data_overview",
    "sparsity",
]

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


def data_overview(df: DataFrame) -> DataFrame:
    r"""Get a summary of the data.

    Parameters
    ----------
    df: DataFrame

    Returns
    -------
    DataFrame
    """
    overview = DataFrame(index=df.columns)
    mask = df.isna()
    numerical_cols = df.select_dtypes(include="number").columns
    # other_cols = df.select_dtypes(exclude="number").columns
    float_cols = df.select_dtypes(include="float").columns

    overview["# datapoints"] = (~mask).sum()
    overview["% missing"] = (mask.mean() * 100).round(2)
    overview["min", numerical_cols] = df[numerical_cols].min().round(2)
    overview["mean", numerical_cols] = df[numerical_cols].mean().round(2)
    overview["std", numerical_cols] = df[numerical_cols].std().round(2)
    overview["max", numerical_cols] = df[numerical_cols].max().round(2)

    overview["min", float_cols] = overview["min", float_cols].round(2)
    overview["mean", float_cols] = overview["mean", float_cols].round(2)
    overview["std", float_cols] = overview["std", float_cols].round(2)
    overview["max", float_cols] = overview["max", float_cols].round(2)

    freq = {}
    for col in df:
        mask = pandas.notna(df[col].squeeze())
        time = df.index[mask]
        freq[col] = Series(time).diff().mean()
    overview["freq"] = Series(freq)
    return overview
