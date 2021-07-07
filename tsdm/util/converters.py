r"""Converts for different data formats."""

from typing import Union

import numpy as np
import pandas as pd
from numba import njit
from pandas import CategoricalDtype, DataFrame, Series


def infer_categories(s: Series) -> set:
    r"""Return categories."""
    # assert is_categorical_dtype(s) or is_string_dtype(s) or is_object_dtype(s), \
    #     F"Series {s=}  with {s.dtype=} does not look like categorical data"
    categories = s.astype(CategoricalDtype()).categories

    return set(categories)


def triplet2dense(
    df: DataFrame, cat_features: Union[dict[str, set], set] = None
) -> DataFrame:
    r"""Convert a DataFrame in triplet format to dense format. Inverse operation of dense2triplet.

    Parameters
    ----------
    df: DataFrame
    cat_features:
        Either a set of keys denoting the columns containing categorical features.
        In this case the categories will be inferred from data.
        Or a dictionary of sets such that a key:value pair corresponds to a column and
        all possible categories in that column. Use empty set to infer categories from data.

    Returns
    -------
    DataFrame
    """
    raise NotImplementedError


def make_dense_triplets(df: DataFrame) -> DataFrame:
    r"""Convert DataFrame to dense triplet format.

    Given that `df` has $d$ columns
    with $n$ rows containing $N ≤ n⋅d$ observations (non-NaN entries),
    the result is a $(N×3)$ array $(t_i, v_i, x_i)_{i=1:N}$

    - $t_i$ timestamp (index)
    - $v_i$ indicator variable
    - $x_i$ observed value

    References
    ----------
    - :func:`pandas.melt`
    - `Set-Functions For Time Series <http://proceedings.mlr.press/v119/horn20a.html>`_

    Parameters
    ----------
    df: DataFrame

    Returns
    -------
    DataFrame


        ========  ================================================
        column    data type
        ========  ================================================
        index     same as input
        variable  :class:`pandas.StringDtype`
        value     same as input
        ========  ================================================

    See Also
    --------
    make_sparse_triplets
    make_masked_format
    """
    result = df.melt(ignore_index=False)
    observed = result["value"].notna()
    result = result[observed]
    variable = result.columns[0]
    result[variable] = result[variable].astype(pd.StringDtype())
    result.rename(columns={variable: "variable"}, inplace=True)
    result.index.rename("time", inplace=True)
    result.sort_values(by=["time", "variable"], inplace=True)
    return result


def make_sparse_triplets(df: DataFrame) -> DataFrame:
    r"""Convert DataFrame to sparse triplet format.

    Given that `df` has $d$ columns with $n$ rows containing $N ≤ n⋅d$ observations
    (non-NaN entries), the result is a $(N×(d+1))$ array $(t_i, v_i, x_i)_{i=1:N}$

    - $t_i$ timestamp (index)
    - $v_i$ one-hot encoded indicator variable
    - $x_i$ observed value

    Parameters
    ----------
    df: DataFrame

    Returns
    -------
    result: DataFrame


        ======  ================================================
        column  data type
        ======  ================================================
        index   same as input
        value   same as input
        \*      :class:`pandas.SparseDtype` ``Sparse[uint8, 0]``
        ======  ================================================

    References
    ----------
    - :func:`pandas.melt`
    - :func:`pandas.get_dummies`
    - `Set-Functions For Time Series <http://proceedings.mlr.press/v119/horn20a.html>`_

    See Also
    --------
    make_dense_triplets
    make_masked_format
    """
    triplets = make_dense_triplets(df)
    result = pd.get_dummies(
        triplets, columns=["variable"], sparse=True, prefix="", prefix_sep=""
    )
    return result


def make_masked_format(df: DataFrame) -> tuple[DataFrame, DataFrame, DataFrame]:
    r"""Convert DataFrame into masked format, returning 3 DataFrames with the same shape.

    Parameters
    ----------
    df: :class:`pandas.DataFrame`

    Returns
    -------
    x: :class:`pandas.DataFrame` ``[dtype]``
        The original dataframe
    m: :class:`pandas.DataFrame` ``[uint8]``
        mask $m_t = \begin{cases}1:& x_t = \text{NaN} \\ 0:& \text{else} \end{cases}$
    d: :class:`pandas.DataFrame` ``[TimeDelta64]``
        time delta  $δ_t = (1-m_{t-1})⊙δ_{t-1} + Δt$, with $δ_0=0$

    References
    ----------
    - `Recurrent Neural Networks for Multivariate Time Series with Missing Values
      <https://www.nature.com/articles/s41598-018-24271-9>`_

    See Also
    --------
    make_dense_triplets
    make_sparse_triplets
    """
    m = df.notna().astype(np.uint8)
    # note: s here is not the same s as in the GRU-D paper, but s(t) - s(t-1)
    s = pd.Series(df.index).diff()
    s[0] = 0 * s[1]
    s = pd.Index(s)

    @njit
    def get_deltas(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # using numba jit compiled for speed - pandas was too slow!
        c = np.outer(a, np.zeros(b.shape[-1]))
        for i in range(1, len(a)):
            c[i] = a[i] + c[i - 1] * (1 - b[i - 1])
            # note: a[i] + b[i] * c[i-1] does not work - not implemented
        return c

    d = DataFrame(
        get_deltas(s.values, m.values), index=m.index, columns=m.columns, dtype=s.dtype
    )

    return df, m, d


def time2int(ds: Series) -> Series:
    r"""Convert `Series` encoded as `datetime64` or `timedelta64` to `integer`.

    Parameters
    ----------
    ds: Series

    Returns
    -------
    Series
    """
    if np.issubdtype(ds.dtype, np.integer):
        return ds
    if np.issubdtype(ds.dtype, np.datetime64):
        ds = ds.view("datetime64[ns]")
        timedeltas = ds - ds[0]
    elif np.issubdtype(ds.dtype, np.timedelta64):
        timedeltas = ds.view("timedelta64[ns]")
    else:
        raise ValueError(f"{ds.dtype=} not supported")

    common_interval = np.gcd.reduce(timedeltas.view(int)).view("timedelta64[ns]")

    return timedeltas // common_interval


def time2float(ds: Series) -> Series:
    r"""Convert `Series` encoded as `datetime64` or `timedelta64` to `floating`.

    Parameters
    ----------
    ds: Series

    Returns
    -------
    Series
    """
    if np.issubdtype(ds.dtype, np.integer):
        return ds
    if np.issubdtype(ds.dtype, np.datetime64):
        ds = ds.view("datetime64[ns]")
        timedeltas = ds - ds[0]
    elif np.issubdtype(ds.dtype, np.timedelta64):
        timedeltas = ds.view("timedelta64[ns]")
    else:
        raise ValueError(f"{ds.dtype=} not supported")

    common_interval = np.gcd.reduce(timedeltas.view(int)).view("timedelta64[ns]")

    return (timedeltas / common_interval).astype(float)
