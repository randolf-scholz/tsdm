r"""Module Docstring."""
from __future__ import annotations

import logging
from typing import Final

import pandas
from pandas import DataFrame

LOGGER = logging.getLogger(__name__)
__all__: Final[list[str]] = []


def sparsity(df: DataFrame):
    r"""Quantify sparsity in the data."""
    mask = pandas.isna(df)
    col_wise = mask.mean(axis=0)
    total = mask.mean()
    return col_wise, total


def linearness(df: DataFrame):
    r"""Quantify linear signals in the data using regularized least-squares."""


def periodicity(df: DataFrame):
    r"""Quantify periodic signals in the data using (Non-Uniform) FFT in O(N log N)."""


def summary_stats(df: DataFrame):
    r"""Summary statistics: column-wise mean/median/std/histogram. Cross-channel correlation."""
