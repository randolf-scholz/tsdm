"""
converters
----------
"""

import numpy as np
import pandas as pd
from numba import njit
from pandas import DataFrame


def make_dense_triplets(df: DataFrame) -> DataFrame:
    r"""
    Converts DataFrame to dense triplet format. Given that `df` has $d$ columns
    with $n$ rows containing $N\le n\cdot d$ observations (non-NaN entries),
    the result is a $(N \times 3)$ array $(t_i, v_i, x_i)_{i=1:N}$

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
    result: DataFrame


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
    observed = result['value'].notna()
    result = result[observed]
    variable = result.columns[0]
    result[variable] = result[variable].astype(pd.StringDtype())
    result.rename(columns={variable: 'variable'}, inplace=True)
    result.index.rename("time", inplace=True)
    result.sort_values(by=["time", "variable"], inplace=True)
    return result


def make_sparse_triplets(df: DataFrame) -> DataFrame:
    r"""
    Converts DataFrame to sparse triplet format. Given that `df` has $d$ columns
    with $n$ rows containing $N\le n\cdot d$ observations (non-NaN entries),
    the result is a $(N \times (d+1))$ array $(t_i, v_i, x_i)_{i=1:N}$

    - $t_i$ timestamp (index)
    - $v_i$ one-hot encoded indicator variable
    - $x_i$ observed value

    References
    ----------
    - :func:`pandas.melt`
    - :func:`pandas.get_dummies`
    - `Set-Functions For Time Series <http://proceedings.mlr.press/v119/horn20a.html>`_

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

    See Also
    --------
    make_dense_triplets
    make_masked_format
    """
    triplets = make_dense_triplets(df)
    result = pd.get_dummies(triplets, columns=["variable"], sparse=True, prefix="", prefix_sep="")
    return result


def make_masked_format(df: DataFrame) -> tuple[DataFrame, DataFrame, DataFrame]:
    r"""
    Converts DataFrame into masked format, returning 3 DataFrames with the same shape.

    References:
    - `Recurrent Neural Networks for Multivariate Time Series with Missing Values
    <https://www.nature.com/articles/s41598-018-24271-9>`_

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
        time delta  $\delta_t = (1-m_{t-1})\odot \delta_{t-1} + \Delta t$, with $\delta_0=0$

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
            c[i] = a[i] + c[i-1] * (1-b[i-1])
            # note: a[i] + b[i] * c[i-1] does not work - not implemented
        return c

    d = DataFrame(get_deltas(s.values, m.values), index=m.index, columns=m.columns, dtype=s.dtype)

    return df, m, d
