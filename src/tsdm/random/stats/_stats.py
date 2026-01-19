r"""Utility functions to get statistics from dataset."""

__all__ = [
    # Functions
    "data_overview",
]

from typing import Optional

import pandas as pd
from pandas import DataFrame, Series


def data_overview(
    df: DataFrame, /, *, index_col: Optional[int | str] = None, digits: int = 2
) -> DataFrame:
    r"""Get a summary of the data."""
    overview = DataFrame(index=df.columns)
    null_values = df.isna()
    numerical_cols = df.select_dtypes(include="number").columns

    overview["datapoints"] = (~null_values).sum()
    overview["num_unique"] = df.nunique()
    overview["missing"] = (null_values.mean() * 100).round(2)

    overview.loc[numerical_cols, "min"] = df[numerical_cols].min()
    overview.loc[numerical_cols, "mean"] = df[numerical_cols].mean()
    overview.loc[numerical_cols, "std"] = df[numerical_cols].std()
    overview.loc[numerical_cols, "max"] = df[numerical_cols].max()

    column_dtypes = {
        "datapoints" : "int64[pyarrow]",
        "num_unique" : "int64[pyarrow]",
        "missing"    : "float64[pyarrow]",
        "min"        : "float64[pyarrow]",
        "mean"       : "float64[pyarrow]",
        "std"        : "float64[pyarrow]",
        "max"        : "float64[pyarrow]",
    }  # fmt: skip

    overview = overview.astype(column_dtypes)
    for col, dtype in column_dtypes.items():
        if pd.api.types.is_float_dtype(dtype):
            overview[col] = overview[col].round(digits)

    if index_col is not None:
        freq = {}
        for col in df.columns:
            mask = pd.notna(df[col].squeeze())
            time = df.index.get_level_values(index_col)[mask]
            freq[col] = Series(time).diff().mean()
        overview["freq"] = Series(freq)
    return overview
