import numpy as np
import pandas
from numba import njit
from pandas import DataFrame


def make_dense_triplets(df: DataFrame) -> DataFrame:
    """
    Converts DataFrame to Triplet Format (num_values x 3) with index='time', columns=['variable', 'value']

    References:
        - pandas 'melt' function https://pandas.pydata.org/docs/reference/api/pandas.melt.html
        - Set-Functions For TIme Series Paper

    Parameters
    ----------
    df: DataFrame

    Returns
    -------
    data: DataFrame
    """
    result = df.melt(ignore_index=False)
    variable = result.columns[0]
    result[variable] = result[variable].astype(pandas.StringDtype())
    result.rename(columns={variable: 'variable'}, inplace=True)
    result.index.rename("time", inplace=True)
    result.sort_values(by=["time", "variable"], inplace=True)
    return result


def make_sparse_triplets(df: DataFrame) -> (DataFrame, DataFrame):
    """
    Converts DataFrame to sparse triplet Format (num_measurements x (1+num_variables)) with index='time',
    columns=['value', *variables], that is the (categorical) variable 'variable' gets stores in one-hot-encoded form

    References:
        - pandas 'get_dummies' function https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html

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


def make_masked_format(df: DataFrame) -> tuple[DataFrame, DataFrame, DataFrame]:
    """
    Converts DataFrame into masked format.
    References: GRU-D paper

    Parameters
    ----------
    df: DataFrame

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
    s = pandas.Series(df.index).diff()  # note: not the same s as in the GRU-D paper, but s(t) - s(t-1)
    s[0] = 0 * s[1]
    s = pandas.Index(s)

    @njit
    def compute_deltas(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # using numba jit compiled for speed - pandas was too slow!
        c = np.outer(a, np.zeros(b.shape[-1]))
        for i in range(1, len(a)):
            c[i] = a[i] + c[i - 1] * b[i]
            # note: a[i] + b[i] * c[i-1] does not work - not implemented
        return c

    d = pandas.DataFrame(compute_deltas(s.values, m.values), index=m.index, columns=m.columns, dtype=s.dtype)

    return df, m, d
