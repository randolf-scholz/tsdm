import pandas
import pandas as pd
import numpy as np
from pandas import DataFrame


def make_dense_triplets(df: DataFrame) -> DataFrame:
    """
    Dataframe with index time stamps and measurement columns. Column have to correspond to variable names
    in metadata. For details see format_specification.md

    References: unpivot function at https://pandas.pydata.org/docs/user_guide/reshaping.html

    Parameters
    ----------
    df: DataFrame

    Returns
    -------
    data: DataFrame
    """
    result = df.melt(ignore_index=False)
    variables = result.columns[0]
    result[variables] = result[variables].astype(pandas.StringDtype()).rename("variable")
    result.index.rename("time", inplace=True)
    result.sort_values(by=["time", "variable"], inplace=True)
    return result


def make_sparse_triplets(df: DataFrame) -> (DataFrame, DataFrame):
    """
    Dataframe with index time stamps and measurement columns. Column have to correspond to variable names
    in metadata. For details see format_specification.md

    References: unpivot function at https://pandas.pydata.org/docs/user_guide/reshaping.html

    Parameters
    ----------
    df: DataFrame

    Returns
    -------
    data: DataFrame
    """
    triplets = make_dense_triplets(df)
    result = pandas.get_dummies(triplets, columns=["variable"], sparse=True, prefix="", prefix_sep="")
    return result


def make_masked_format(df: DataFrame) -> (DataFrame, DataFrame, DataFrame):
    from numba import njit
    """
    Converts DataFrame into masked format.

    Parameters
    ----------
    df: DataFrame
    sparse: bool
        Whether to encode the tensors sparsely
    Returns
    -------
    x: DataFrame
        The original dataframe
    m: DataFrame[uint8]
        The mask corresponding to the observed values
    d: DataFrame[TimeDelta64]
        the time deltas since the last observation

    """

    m = df.isna().astype(np.uint8)
    s = pandas.Series(df.index).diff()
    s[0] = 0 * s[1]
    s = pandas.Index(s)

    @njit
    def compute_deltas(a, b):
        c = np.outer(s, np.zeros(m.shape[-1]))
        for i in range(1, len(s)):
            c[i] = a[i] + c[i-1] * b[i]
            # note: a[i] + b[i] * c[i-1] does not work - not implemented
        return c

    d = pandas.DataFrame(compute_deltas(s, m), index=m.index, columns=m.columns, dtype=s.dtype)

    return df, m, d
