r"""Dtype data for numpy/pandas/torch.

Numerical Type Hierarchy:
    - object
    - datetime, timedelta, datetimeTZ
    - interval
    - period
    - string
        - unicode
        - ascii
        - bytes
    - numerical
        - complex
        - float
        - int
        - uint
        - bool
    - empty (contains only NA)
"""

__all__ = [
    # TypeVars and TypeAliases
    # DTYPES
    "NUMPY_DTYPES",
    "TORCH_DTYPES",
    "PYTHON_DTYPES",
    "POLARS_DTYPES",
    "PYARROW_DTYPES",
    # CONVERSIONS
    "PYARROW_TO_POLARS",
    # PANDAS DTYPES
    "PANDAS_ARROW_DATE_TYPES",
    "PANDAS_ARROW_DURATION_TYPES",
    "PANDAS_ARROW_TIMESTAMP_TYPES",
    "PANDAS_NULLABLE_DTYPES",
    # TYPESTRINGS
    "NUMPY_TYPESTRINGS",
    "NUMPY_TYPECODES",
    "TORCH_TYPESTRINGS",
    "PANDAS_TYPESTRINGS",
    "PYTHON_TYPESTRINGS",
    # NUMPY TYPECODES
    "NUMPY_BOOL_TYPECODES",
    "NUMPY_COMPLEX_TYPECODES",
    "NUMPY_FLOAT_TYPECODES",
    "NUMPY_INT_TYPECODES",
    "NUMPY_OBJECT_TYPECODES",
    "NUMPY_STRING_TYPECODES",
    "NUMPY_TIME_TYPECODES",
    "NUMPY_UINT_TYPECODES",
    # NUMPY TYPESTRINGS
    "NUMPY_BOOL_TYPESTRINGS",
    "NUMPY_COMPLEX_TYPESTRINGS",
    "NUMPY_FLOAT_TYPESTRINGS",
    "NUMPY_INT_TYPESTRINGS",
    "NUMPY_OBJECT_TYPESTRINGS",
    "NUMPY_STRING_TYPESTRINGS",
    "NUMPY_TIME_TYPESTRINGS",
    "NUMPY_UINT_TYPESTRINGS",
    # TORCH TYPESTRINGS
    "TORCH_BOOL_TYPESTRINGS",
    "TORCH_COMPLEX_TYPESTRINGS",
    "TORCH_FLOAT_TYPESTRINGS",
    "TORCH_INT_TYPESTRINGS",
    "TORCH_UINT_TYPESTRINGS",
    # Constants
    "TYPESTRINGS",
    # Functions
    "map_pandas_arrowtime_numpy",
]

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import torch
from pandas import ArrowDtype
from pandas.api.extensions import ExtensionDtype
from typing_extensions import Final

# region numpy typecodes ---------------------------------------------------------------
NUMPY_DTYPES: Final[dict[str, type[np.generic]]] = {
    "int8"        : np.int8,
    "int16"       : np.int16,
    "int32"       : np.int32,
    "int64"       : np.int64,
    "float_"      : np.float64,
    "float16"     : np.float16,
    "float32"     : np.float32,
    "float64"     : np.float64,
    "complex64"   : np.complex64,
    "complex128"  : np.complex128,
    "uint8"       : np.uint8,
    "uint16"      : np.uint16,
    "uint32"      : np.uint32,
    "uint64"      : np.uint64,
    "timedelta64" : np.timedelta64,
    "datetime64"  : np.datetime64,
    "bool"        : np.bool_,
    "bytes"       : np.bytes_,
    "str"         : np.str_,
    "unicode"     : np.str_,
    "void"        : np.void,
    "object"      : np.object_,
}  # fmt: skip
r"""Dictionary of all `numpy` data types."""

NUMPY_INT_TYPECODES: Final[dict[type[np.signedinteger], str]] = {
    np.int8     : "b",
    np.int16    : "h",
    np.int32    : "i",
    np.int64    : "l",
    np.byte     : "b",
    np.short    : "h",
    np.intc     : "i",
    np.int_     : "l",
    np.intp     : "l",
    np.longlong : "q",
}  # fmt: skip
r"""Dictionary of all signed `numpy` integer data type typecodes."""

NUMPY_UINT_TYPECODES: Final[dict[type[np.unsignedinteger], str]] = {
    np.uint8     : "B",
    np.uint16    : "H",
    np.uint32    : "I",
    np.uint64    : "L",
    np.ubyte     : "B",
    np.ushort    : "H",
    np.uintc     : "I",
    np.uint      : "L",
    np.uintp     : "L",
    np.ulonglong : "Q",
}  # fmt: skip
r"""Dictionary of all unsigned `numpy` integer data type typecodes."""

NUMPY_FLOAT_TYPECODES: Final[dict[type[np.floating], str]] = {
    np.float16    : "e",
    np.float32    : "f",
    np.float64    : "d",
    # np.float128   : "g",
    np.half       : "e",
    np.single     : "f",
    np.double     : "d",
    np.longdouble : "g",
}  # fmt: skip
r"""Dictionary of all `numpy` float data type typecodes."""

NUMPY_COMPLEX_TYPECODES: Final[dict[type[np.complexfloating], str]] = {
    np.complex64   : "F",
    np.complex128  : "D",
    # np.complex256  : "G",
    np.csingle     : "F",
    np.cdouble     : "D",
    np.clongdouble : "G",
}  # fmt: skip
r"""Dictionary of all `numpy` complex data types."""


NUMPY_TIME_TYPECODES: Final[dict[type[np.generic], str]] = {
    np.timedelta64 : "M",  # timedelta64
    np.datetime64  : "m",  # datetime64
}  # fmt: skip
r"""Dictionary of all `numpy` time data type typecodes."""

NUMPY_BOOL_TYPECODES: Final[dict[type[np.generic], str]] = {
    np.bool_ : "?",  # bool
}  # fmt: skip
r"""Dictionary of all `numpy` bool data type typecodes."""

NUMPY_STRING_TYPECODES: Final[dict[type[np.flexible], str]] = {
    np.bytes_ : "S",  # bytes
    np.str_   : "U",  # unicode
    np.void   : "V",  # "void"
}  # fmt: skip
r"""Dictionary of all `numpy` string data type typecodes."""

NUMPY_OBJECT_TYPECODES: Final[dict[type[np.generic], str]] = {
    np.object_ : "O",
}  # fmt: skip
r"""Dictionary of all `numpy` generic data type typecodes."""

NUMPY_TYPECODES: Final[dict[type[np.generic], str]] = (
    NUMPY_INT_TYPECODES
    | NUMPY_UINT_TYPECODES
    | NUMPY_FLOAT_TYPECODES
    | NUMPY_COMPLEX_TYPECODES
    | NUMPY_TIME_TYPECODES
    | NUMPY_STRING_TYPECODES
    | NUMPY_OBJECT_TYPECODES
)
r"""Dictionary of all `numpy` data type typecodes."""
# endregion numpy typecodes ------------------------------------------------------------


# region numpy dtypes ------------------------------------------------------------------
NUMPY_INT_TYPESTRINGS: Final[dict[type[np.signedinteger], str]] = {
    np.int8     : "int8",
    np.int16    : "int16",
    np.int32    : "int32",
    np.int64    : "int64",
    np.byte     : "int8",
    np.short    : "int16",
    np.intc     : "int32",
    np.int_     : "int64",
    np.intp     : "int64",
    np.longlong : "q",
}  # fmt: skip
r"""Dictionary of all signed `numpy` integer data type typestrings."""

NUMPY_UINT_TYPESTRINGS: Final[dict[type[np.unsignedinteger], str]] = {
    np.uint8     : "uint8",
    np.uint16    : "uint16",
    np.uint32    : "uint32",
    np.uint64    : "uint64",
    np.ubyte     : "uint8",
    np.ushort    : "uint16",
    np.uintc     : "uint32",
    np.uint      : "uint64",
    np.uintp     : "uint64",
    np.ulonglong : "Q",
}  # fmt: skip
r"""Dictionary of all unsigned `numpy` integer data type typestrings."""

NUMPY_FLOAT_TYPESTRINGS: Final[dict[type[np.floating], str]] = {
    np.float16    : "float16",
    np.float32    : "float32",
    np.float64    : "float64",
    # np.float128   : "float128",
    np.half       : "float16",
    np.single     : "float32",
    np.double     : "float64",
    np.longdouble : "float128",
}  # fmt: skip
r"""Dictionary of all `numpy` float data type typestrings."""

NUMPY_COMPLEX_TYPESTRINGS: Final[dict[type[np.complexfloating], str]] = {
    np.complex64   : "complex64",
    np.complex128  : "complex128",
    # np.complex256  : "complex256",
    np.csingle     : "complex64",
    np.cdouble     : "complex128",
    np.clongdouble : "complex256",
}  # fmt: skip
r"""Dictionary of all `numpy` complex data typestrings."""


NUMPY_TIME_TYPESTRINGS: Final[dict[type[np.generic], str]] = {
    np.timedelta64 : "timedelta64",  # timedelta64
    np.datetime64  : "datetime64",  # datetime64
}  # fmt: skip
r"""Dictionary of all `numpy` time data type typestrings."""

NUMPY_BOOL_TYPESTRINGS: Final[dict[type[np.generic], str]] = {
    np.bool_ : "bool",  # bool
}  # fmt: skip
r"""Dictionary of all `numpy` bool data type typestrings."""

NUMPY_STRING_TYPESTRINGS: Final[dict[type[np.flexible], str]] = {
    np.bytes_ : "bytes",  # str
    np.str_   : "str",  # str
    np.void   : "void",  # "void"
}  # fmt: skip
r"""Dictionary of all `numpy` string data type typestrings."""

NUMPY_OBJECT_TYPESTRINGS: Final[dict[type[np.generic], str]] = {
    np.object_: "object",
}  # fmt: skip
r"""Dictionary of all `numpy` generic data type typestrings."""

NUMPY_TYPESTRINGS: Final[dict[type[np.generic], str]] = (
    NUMPY_INT_TYPESTRINGS
    | NUMPY_UINT_TYPESTRINGS
    | NUMPY_FLOAT_TYPESTRINGS
    | NUMPY_COMPLEX_TYPESTRINGS
    | NUMPY_TIME_TYPESTRINGS
    | NUMPY_STRING_TYPESTRINGS
    | NUMPY_OBJECT_TYPESTRINGS
)
r"""Dictionary of all `numpy` data type typestrings."""
# endregion numpy typestrings ----------------------------------------------------------


# region pandas dtypes -----------------------------------------------------------------
PANDAS_TYPESTRINGS: Final[dict[type[ExtensionDtype], str]] = {
    pd.BooleanDtype     : "boolean",
    pd.CategoricalDtype : "category",
    pd.DatetimeTZDtype  : "datetime64[ns, tz]",  # datetime64[ns, <tz>]
    pd.Float32Dtype     : "Float32",
    pd.Float64Dtype     : "Float64",
    pd.Int16Dtype       : "Int16",
    pd.Int32Dtype       : "Int32",
    pd.Int64Dtype       : "Int64",
    pd.Int8Dtype        : "Int8",
    pd.IntervalDtype    : "interval",  # e.g. to denote ranges of variables
    pd.PeriodDtype      : "period",  # period[<freq>]
    pd.SparseDtype      : "Sparse",
    pd.StringDtype      : "string",
    pd.UInt16Dtype      : "UInt16",
    pd.UInt32Dtype      : "UInt32",
    pd.UInt64Dtype      : "UInt64",
    pd.UInt8Dtype       : "UInt8",
}  # fmt: skip
r"""Dictionary of all `pandas` data type typestrings."""

PANDAS_NULLABLE_DTYPES: Final[dict[str, type[ExtensionDtype]]] = {
    "datetime64[ns, tz]": pd.DatetimeTZDtype,  # datetime64[ns, <tz>]
    "boolean"  : pd.BooleanDtype,
    "category" : pd.CategoricalDtype,
    "Float32"  : pd.Float32Dtype,
    "Float64"  : pd.Float64Dtype,
    "Int16"    : pd.Int16Dtype,
    "Int32"    : pd.Int32Dtype,
    "Int64"    : pd.Int64Dtype,
    "Int8"     : pd.Int8Dtype,
    "interval" : pd.IntervalDtype,  # e.g. to denote ranges of variables
    "period"   : pd.PeriodDtype,  # period[<freq>]
    "Sparse"   : pd.SparseDtype,
    "string"   : pd.StringDtype,
    "UInt16"   : pd.UInt16Dtype,
    "UInt32"   : pd.UInt32Dtype,
    "UInt64"   : pd.UInt64Dtype,
    "UInt8"    : pd.UInt8Dtype,
}  # fmt: skip
r"""Dictionary of all `pandas` data types."""

PANDAS_ARROW_DURATION_TYPES: set[ArrowDtype] = {
    ArrowDtype(pa.duration(unit)) for unit in ["s", "ms", "us", "ns"]
}
r"""Set of all `pandas` arrow duration types."""

PANDAS_ARROW_TIMESTAMP_TYPES: set[ArrowDtype] = {
    ArrowDtype(pa.timestamp(unit)) for unit in ["s", "ms", "us", "ns"]
}
r"""Set of all `pandas` arrow timestamp types."""

PANDAS_ARROW_DATE_TYPES: set[ArrowDtype] = {
    ArrowDtype(pa.date32()),
    ArrowDtype(pa.date64()),
}
r"""Set of all `pandas` arrow date types."""


def map_pandas_arrowtime_numpy(df: pd.DataFrame) -> pd.DataFrame:
    r"""Converts pyarrow date/timestamp/duration types to numpy equivalents.

    Rationale: pyarrow types are currently bugged and do not support all operations.
    """
    for col, dtype in df.dtypes.items():
        if dtype in PANDAS_ARROW_DURATION_TYPES:
            df[col] = df[col].astype("timedelta64[ms]")
        elif dtype in PANDAS_ARROW_TIMESTAMP_TYPES:
            df[col] = df[col].astype("datetime64[ms]")
        elif dtype in PANDAS_ARROW_DATE_TYPES:
            df[col] = df[col].astype("datetime64[s]")
    return df


# endregion pandas dtypes --------------------------------------------------------------


# region polars dtypes -----------------------------------------------------------------
PYARROW_DTYPES: Final[dict[str, pa.DataType]] = {
    # numeric
    "null"    : pa.null(),
    "bool"    : pa.bool_(),
    "int8"    : pa.int8(),
    "int16"   : pa.int16(),
    "int32"   : pa.int32(),
    "int64"   : pa.int64(),
    "uint8"   : pa.uint8(),
    "uint16"  : pa.uint16(),
    "uint32"  : pa.uint32(),
    "uint64"  : pa.uint64(),
    "float16" : pa.float16(),
    "float32" : pa.float32(),
    "float64" : pa.float64(),
    # temporal
    "date32"        : pa.date32(),
    "date64"        : pa.date64(),
    "time32"        : pa.time32("s"),
    "time64"        : pa.time64("ns"),
    "timestamp[ns]" : pa.timestamp("ns"),
    "timestamp[us]" : pa.timestamp("us"),
    "timestamp[ms]" : pa.timestamp("ms"),
    "timestamp[s]"  : pa.timestamp("s"),
    "duration[ns]"  : pa.duration("ns"),
    "duration[us]"  : pa.duration("us"),
    "duration[ms]"  : pa.duration("ms"),
    "duration[s]"   : pa.duration("s"),
    # string/binary
    "binary"       : pa.binary(),
    "large_string" : pa.large_string(),
    "string"       : pa.string(),
}  # fmt: skip
r"""Dictionary of all `pyarrow` data types."""
# endregion polars dtypes --------------------------------------------------------------


# region polars dtypes -----------------------------------------------------------------
POLARS_DTYPES: Final[dict[str, pl.PolarsDataType]] = {
    # numeric
    "Float32"    : pl.Float32(),
    "Float64"    : pl.Float64(),
    "Int8"       : pl.Int8(),
    "Int16"      : pl.Int16(),
    "Int32"      : pl.Int32(),
    "Int64"      : pl.Int64(),
    "UInt8"      : pl.UInt8(),
    "UInt16"     : pl.UInt16(),
    "UInt32"     : pl.UInt32(),
    "UInt64"     : pl.UInt64(),
    # temporal
    "Date"       : pl.Date(),
    "Datetime"   : pl.Datetime(),
    "Duration"   : pl.Duration(),
    "Time"       : pl.Time(),
    # other
    "Binary"     : pl.Binary(),
    "Boolean"    : pl.Boolean(),
    "Categorical": pl.Categorical(),
    "Null"       : pl.Null(),
    "Object"     : pl.Object(),
    "Utf8"       : pl.Utf8(),
}  # fmt: skip
r"""Dictionary of all elementary `polars` data types."""
# endregion polars dtypes --------------------------------------------------------------


# region torch dtypes ------------------------------------------------------------------
TORCH_INT_TYPESTRINGS: Final[dict[torch.dtype, str]] = {
    torch.int8   : "int8",
    torch.int16  : "int16",
    torch.int32  : "int32",
    torch.int64  : "int64",
    torch.qint8  : "qint8",
    torch.qint32 : "qint32",
}  # fmt: skip
r"""Dictionary of all `torch` signed integer data type typestrings."""

TORCH_UINT_TYPESTRINGS: Final[dict[torch.dtype, str]] = {
    torch.uint8    : "uint8",
    # torch.uint16   : "uint16",
    # torch.uint32   : "uint32",
    # torch.uint64   : "uint64",
    torch.quint8   : "quint8",
    torch.quint4x2 : "quint4x2",
}  # fmt: skip
r"""Dictionary of all `torch` unsigned integer data type typestrings."""

TORCH_FLOAT_TYPESTRINGS: Final[dict[torch.dtype, str]] = {
    torch.bfloat16 : "bfloat16",
    torch.float16  : "float16",
    torch.float32  : "float32",
    torch.float64  : "float64",
}  # fmt: skip
r"""Dictionary of all `torch` float data type typestrings."""

TORCH_COMPLEX_TYPESTRINGS: Final[dict[torch.dtype, str]] = {
    torch.complex32  : "complex32",
    torch.complex64  : "complex64",
    torch.complex128 : "complex128",
}  # fmt: skip
r"""Dictionary of all `torch` complex data type typestrings."""

TORCH_BOOL_TYPESTRINGS: Final[dict[torch.dtype, str]] = {
    torch.bool : "bool",
}  # fmt: skip
r"""Dictionary of all `torch` bool data type typestrings."""

TORCH_TYPESTRINGS: Final[dict[torch.dtype, str]] = (
    TORCH_INT_TYPESTRINGS
    | TORCH_UINT_TYPESTRINGS
    | TORCH_FLOAT_TYPESTRINGS
    | TORCH_COMPLEX_TYPESTRINGS
    | TORCH_BOOL_TYPESTRINGS
)
r"""Dictionary of all `torch` data type typestrings."""

TORCH_DTYPES: Final[dict[str, torch.dtype]] = {
    "int8"       : torch.int8,
    "int16"      : torch.int16,
    "int32"      : torch.int32,
    "int64"      : torch.int64,
    "qint8"      : torch.qint8,
    "qint32"     : torch.qint32,
    "uint8"      : torch.uint8,
    "quint8"     : torch.quint8,
    "quint4x2"   : torch.quint4x2,
    "bfloat16"   : torch.bfloat16,
    "float16"    : torch.float16,
    "float32"    : torch.float32,
    "float64"    : torch.float64,
    "complex32"  : torch.complex32,
    "complex64"  : torch.complex64,
    "complex128" : torch.complex128,
    "bool"       : torch.bool,
}  # fmt: skip
r"""Dictionary of all `torch` data types."""
# endregion torch dtypes ---------------------------------------------------------------


# region python dtypes -----------------------------------------------------------------
PYTHON_DTYPES: Final[dict[str, type]] = {
    "bool"      : bool,
    "int"       : int,
    "float"     : float,
    "complex"   : complex,
    "str"       : str,
    "bytes"     : bytes,
    "datetime"  : datetime,
    "timedelta" : timedelta,
    "object"    : object,
}  # fmt: skip
r"""Dictionary of all `python` data types."""

PYTHON_TYPESTRINGS: Final[dict[type, str]] = {
    bool      : "bool",
    int       : "int",
    float     : "float",
    complex   : "complex",
    str       : "str",
    bytes     : "bytes",
    datetime  : "datetime",
    timedelta : "timedelta",
    object    : "object",
}  # fmt: skip
r"""Dictionary of all `python` data types."""
# endregion python dtypes --------------------------------------------------------------


# region dtype conversion --------------------------------------------------------------
PYARROW_TO_POLARS: Final[dict[pa.DataType, pl.PolarsDataType]] = {
    pa.null()          : pl.Null(),
    pa.bool_()         : pl.Boolean(),
    pa.int8()          : pl.Int8(),
    pa.int16()         : pl.Int16(),
    pa.int32()         : pl.Int32(),
    pa.int64()         : pl.Int64(),
    pa.uint8()         : pl.UInt8(),
    pa.uint16()        : pl.UInt16(),
    pa.uint32()        : pl.UInt32(),
    pa.uint64()        : pl.UInt64(),
    pa.float16()       : pl.Float32(),
    pa.float32()       : pl.Float32(),
    pa.float64()       : pl.Float64(),
    pa.date32()        : pl.Date(),
    pa.date64()        : pl.Date(),
    pa.time32("s")     : pl.Time(),
    pa.time64("ns")    : pl.Time(),
    pa.timestamp("ns") : pl.Datetime(),
    pa.timestamp("us") : pl.Datetime(),
    pa.timestamp("ms") : pl.Datetime(),
    pa.timestamp("s")  : pl.Datetime(),
    pa.duration("ns")  : pl.Duration(),
    pa.duration("us")  : pl.Duration(),
    pa.duration("ms")  : pl.Duration(),
    pa.duration("s")   : pl.Duration(),
    pa.binary()        : pl.Binary(),
    pa.large_string()  : pl.Utf8(),
    pa.string()        : pl.Utf8(),
    pa.dictionary(pa.int32(), pa.string())   : pl.Categorical(),
}  # fmt: skip
r"""Dictionary of converting pyarrow to polars."""
# endregion dtype conversion -----------------------------------------------------------


TYPESTRINGS: Final[dict[type | torch.dtype, str]] = (
    PANDAS_TYPESTRINGS | NUMPY_TYPESTRINGS | TORCH_TYPESTRINGS
)
r"""Dictionary of all type strings."""
