"""
converters
----------
"""

import numpy as np
from numpy import ndarray
import pandas as pd
from numba import njit
from pandas import DataFrame, Series, CategoricalDtype
from typing import Union
from numpy.typing import ArrayLike
from collections.abc import Iterable
from pandas.api.types import is_categorical_dtype, is_string_dtype, is_object_dtype




def infer_categories(s: Series) -> set:
    # assert is_categorical_dtype(s) or is_string_dtype(s) or is_object_dtype(s), \
    #     F"Series {s=}  with {s.dtype=} does not look like categorical data"

    categories = s.astype(CategoricalDtype()).categories

    return set(categories)


def encode_categorical_data(data: ArrayLike) -> ArrayLike:
    pass


def encode_data(data: ArrayLike) -> ndarray:
    # compatible with numpy and torch
    # make tensor if not already
    data = np.asanyarray(data)

    if np.issubdtype(data.dtype, np.floating):
        return data
    else:
        pass







def encode_metadata(metadata: dict[str, ArrayLike]) -> ArrayLike:



    pass

def encode_categories(df: DataFrame):
    pass


def triplet2dense(df: DataFrame,
                  cat_features: Union[dict[str, set], set] = None
                  ) -> DataFrame:
    r"""
    Converts a DataFrame in triplet format to dense format.
    Inverse operation of dense2triplet.

    Parameters
    ----------
    df:
    cat_features:
        Either a set of keys denoting the columns containing categorical features. In this case the categories will be
        infered from data.
        Or a dictionary of sets such that a key:value pair corresponds to a column and all possible categories in that
        column. Use empty set to infer categories from data.

    Returns
    -------

    """
    pass







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


def time2int(ds: Series) -> Series:
    """Convert :class:`~pandas.Series` encoded as
    :class:`~numpy.datetime64` or :class:`~numpy.timedelta64` to :class:`~numpy.integer`

    Parameters
    ----------
    ds: Series

    Returns
    -------
    Series
    """

    if np.issubdtype(ds.dtype, np.integer):
        return ds
    elif np.issubdtype(ds.dtype, np.datetime64):
        ds = ds.astype('datetime64[ns]')
        timedeltas = ds - ds[0]
    elif np.issubdtype(ds.dtype, np.timedelta64):
        timedeltas = ds.astype('timedelta64[ns]')
    else:
        raise ValueError(F"{ds.dtype=} not supported")

    common_interval = np.gcd.reduce( timedeltas.astype(int) ).astype('timedelta64[ns]')

    return timedeltas // common_interval


def time2float(ds: Series) -> Series:
    """Convert :class:`~pandas.Series` encoded as
    :class:`~numpy.datetime64` or :class:`~numpy.timedelta64` to :class:`~numpy.floating`

    Parameters
    ----------
    ds: Series

    Returns
    -------
    Series
    """
    if np.issubdtype(ds.dtype, np.integer):
        return ds
    elif np.issubdtype(ds.dtype, np.datetime64):
        ds = ds.astype('datetime64[ns]')
        timedeltas = ds - ds[0]
    elif np.issubdtype(ds.dtype, np.timedelta64):
        timedeltas = ds.astype('timedelta64[ns]')
    else:
        raise ValueError(F"{ds.dtype=} not supported")

    common_interval = np.gcd.reduce( timedeltas.astype(int) ).astype('timedelta64[ns]')

    return (timedeltas / common_interval).astype(float)
