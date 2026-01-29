r"""Test other protocols."""

from array import array as python_array

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
import torch

from tsdm.types.numerical.mixins import SupportsArray, SupportsArrayUfunc, SupportsDtype

RNG = np.random.default_rng()

BOOLS = [True, False, True, False]
STRINGS = ["a", "b", "c", "d"]
INTEGERS = [1, 2, 3, 4]
MATRIX = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
DICT_FLOAT = {
    "x": [1.1, 2.2, 3.3, 4.4],
    "y": [5.5, 6.6, 7.7, 8.8],
    "z": [9.9, 0.0, 1.1, 2.2],
}
DICT_MIXED = {
    "label": ["a", "b", "c", "d"],
    "x": [1.1, 2.2, 3.3, 4.4],
    "y": [5.5, 6.6, 7.7, 8.8],
}
DATETIMES = pd.date_range("2021-01-01", periods=4)

# tensorial (single dtype)
NP_ARRAY_1D = np.array(INTEGERS)
NP_ARRAY_2D = np.array(MATRIX)
PA_ARRAY_INT = pa.array(INTEGERS)
PA_ARRAY_STR = pa.array(STRINGS)
PD_INDEX_STR = pd.Index(STRINGS)
PD_INDEX_INT = pd.Index(INTEGERS)
PD_MULTIINDEX = pd.MultiIndex.from_tuples(zip(STRINGS, INTEGERS, strict=True))
PD_SERIES_INT = pd.Series(INTEGERS, index=DATETIMES)
PD_SERIES_STR = pd.Series(STRINGS, index=DATETIMES)
PD_ARRAY_INT = pd.Series(INTEGERS).array
PD_ARRAY_STR = pd.Series(STRINGS).array
PD_ARRAY_PD_INT = pd.Series(INTEGERS, dtype="Int64").array
PD_ARRAY_PD_STR = pd.Series(STRINGS, dtype="string").array
PD_ARRAY_PA_INT = pd.Series(INTEGERS, dtype="int64[pyarrow]").array
PD_ARRAY_PA_STR = pd.Series(INTEGERS, dtype="string[pyarrow]").array
PL_SERIES_INT = pl.Series(INTEGERS)
PL_SERIES_STR = pl.Series(STRINGS)
PT_TENSOR_1D = torch.tensor(INTEGERS)
PT_TENSOR_2D = torch.tensor(MATRIX)
PY_ARRAY = memoryview(python_array("i", [1, 2, 3]))

# tabular (mixed dtype)
PA_TABLE_FLOAT = pa.table(DICT_FLOAT)
PA_TABLE_MIXED = pa.table(DICT_MIXED)
PD_TABLE_FLOAT = pd.DataFrame(DICT_FLOAT, index=DATETIMES)
PD_TABLE_MIXED = pd.DataFrame(DICT_MIXED, index=DATETIMES)
PL_TABLE_FLOAT = pl.DataFrame(DICT_FLOAT)
PL_TABLE_MIXED = pl.DataFrame(DICT_MIXED)


SUPPORTS_ARRAY: dict[str, SupportsArray] = {
    "numpy_ndarray_1d"    : NP_ARRAY_1D,
    "numpy_ndarray_2d"    : NP_ARRAY_2D,
    "pandas_array_int"    : PD_ARRAY_INT,
    "pandas_array_str"    : PD_ARRAY_STR,
    "pandas_index_int"    : PD_INDEX_INT,
    "pandas_index_str"    : PD_INDEX_STR,
    "pandas_multiindex"   : PD_MULTIINDEX,
    "pandas_series_int"   : PD_SERIES_INT,
    "pandas_series_str"   : PD_SERIES_STR,
    "pandas_table_float"  : PD_TABLE_FLOAT,
    "pandas_table_mixed"  : PD_TABLE_MIXED,
    "polars_series_int"   : PL_SERIES_INT,
    "polars_series_str"   : PL_SERIES_STR,
    "polars_table_float"  : PL_TABLE_FLOAT,
    "polars_table_mixed"  : PL_TABLE_MIXED,
    "pyarrow_array_int"   : PA_ARRAY_INT,
    "pyarrow_array_str"   : PA_ARRAY_STR,
    "pyarrow_table_float" : PA_TABLE_FLOAT,
    "pyarrow_table_mixed" : PA_TABLE_MIXED,
    "torch_tensor_1d"     : PT_TENSOR_1D,
    "torch_tensor_2d"     : PT_TENSOR_2D,
}  # fmt: skip
r"""Collection of all test arrays."""


SUPPORTS_DTYPE: dict[str, SupportsDtype] = {
    "numpy_ndarray_1d"    : NP_ARRAY_1D,
    "numpy_ndarray_2d"    : NP_ARRAY_2D,
    "pandas_array_int"    : PD_ARRAY_INT,
    "pandas_array_str"    : PD_ARRAY_STR,
    "pandas_index_int"    : PD_INDEX_INT,
    "pandas_index_str"    : PD_INDEX_STR,
    "pandas_series_int"   : PD_SERIES_INT,
    "pandas_series_str"   : PD_SERIES_STR,
    "polars_series_int"   : PL_SERIES_INT,
    "polars_series_str"   : PL_SERIES_STR,
    "torch_tensor_1d"     : PT_TENSOR_1D,
    "torch_tensor_2d"     : PT_TENSOR_2D,
}  # fmt: skip

SUPPORTS_ARRAYS_UFUNC: dict[str, SupportsArrayUfunc] = {
    "numpy_ndarray_1d"    : NP_ARRAY_1D,
    "numpy_ndarray_2d"    : NP_ARRAY_2D,
    "pandas_array_int"    : PD_ARRAY_INT,
    "pandas_index_int"    : PD_INDEX_INT,
    "pandas_series_int"   : PD_SERIES_INT,
    "pandas_table_float"  : PD_TABLE_FLOAT,
    "polars_series_int"   : PL_SERIES_INT,
}  # fmt: skip


@pytest.mark.parametrize("case", SUPPORTS_ARRAY)
def test_supports_array(case: str) -> None:
    r"""Test the SupportsArray protocol."""
    obj = SUPPORTS_ARRAY[case]
    assert isinstance(obj, SupportsArray)
    assert isinstance(obj.__array__(), np.ndarray)


@pytest.mark.parametrize("case", SUPPORTS_DTYPE)
def test_supports_dtype(case: str) -> None:
    r"""Test the SupportsDtype protocol."""
    obj = SUPPORTS_DTYPE[case]
    assert isinstance(obj, SupportsDtype)
    assert isinstance(obj.dtype, object)
