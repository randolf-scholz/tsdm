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
    "PANDAS_NULLABLE_DTYPES",
    "PYTHON_DTYPES",
    "POLARS_DTYPES",
    "PYARROW_DTYPES",
    # CONVERSIONS
    "PYARROW_TO_POLARS",
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
]

from datetime import datetime, timedelta
from typing import Final

import numpy as np
import pandas
import polars
import pyarrow
import torch
from pandas.api.extensions import ExtensionDtype

# region numpy typecodes -----------------------------------------------------------------------------------------------
NUMPY_DTYPES: Final[dict[str, type[np.generic]]] = {
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "float_": np.float_,
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    # "float128": np.float128,
    "complex64": np.complex64,
    "complex128": np.complex128,
    # "complex256": np.complex256,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "timedelta64": np.timedelta64,
    "datetime64": np.datetime64,
    "bool": np.bool_,
    "bytes": np.bytes_,
    "str": np.str_,
    "unicode": np.unicode_,
    "void": np.void,
    "object": np.object_,
}
r"""Dictionary of all `numpy` data types."""

NUMPY_INT_TYPECODES: Final[dict[type[np.signedinteger], str]] = {
    np.int8: "b",
    np.int16: "h",
    np.int32: "i",
    np.int64: "l",
    np.byte: "b",
    np.short: "h",
    np.intc: "i",
    np.int_: "l",
    np.intp: "l",
    np.longlong: "q",
}
r"""Dictionary of all signed `numpy` integer data type typecodes."""

NUMPY_UINT_TYPECODES: Final[dict[type[np.unsignedinteger], str]] = {
    np.uint8: "B",
    np.uint16: "H",
    np.uint32: "I",
    np.uint64: "L",
    np.ubyte: "B",
    np.ushort: "H",
    np.uintc: "I",
    np.uint: "L",
    np.uintp: "L",
    np.ulonglong: "Q",
}
r"""Dictionary of all unsigned `numpy` integer data type typecodes."""

NUMPY_FLOAT_TYPECODES: Final[dict[type[np.floating], str]] = {
    np.float_: "d",
    np.float16: "e",
    np.float32: "f",
    np.float64: "d",
    # np.float128: "g",
    np.half: "e",
    np.single: "f",
    np.double: "d",
    np.longdouble: "g",
    np.longfloat: "g",
}
r"""Dictionary of all `numpy` float data type typecodes."""

NUMPY_COMPLEX_TYPECODES: Final[dict[type[np.complexfloating], str]] = {
    np.complex64: "F",
    np.complex128: "D",
    # np.complex256: "G",
    np.csingle: "F",
    np.singlecomplex: "F",
    np.cdouble: "D",
    np.cfloat: "D",
    np.complex_: "D",
    np.clongdouble: "G",
    np.clongfloat: "G",
    np.longcomplex: "G",
}
r"""Dictionary of all `numpy` complex data types."""


NUMPY_TIME_TYPECODES: Final[dict[type[np.generic], str]] = {
    np.timedelta64: "M",  # timedelta64
    np.datetime64: "m",  # datetime64
}
r"""Dictionary of all `numpy` time data type typecodes."""

NUMPY_BOOL_TYPECODES: Final[dict[type[np.generic], str]] = {
    np.bool_: "?",  # bool
}
r"""Dictionary of all `numpy` bool data type typecodes."""

NUMPY_STRING_TYPECODES: Final[dict[type[np.flexible], str]] = {
    np.bytes_: "S",  # str
    np.string_: "S",  # bytes
    np.str_: "U",  # str
    np.unicode_: "U",  # unicode
    np.void: "V",  # "void"
}
r"""Dictionary of all `numpy` string data type typecodes."""

NUMPY_OBJECT_TYPECODES: Final[dict[type[np.generic], str]] = {
    np.object_: "O",
}
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
# endregion numpy typecodes --------------------------------------------------------------------------------------------


# region numpy dtypes -------------------------------------------------------------------------------------------------
NUMPY_INT_TYPESTRINGS: Final[dict[type[np.signedinteger], str]] = {
    np.int8: "int8",
    np.int16: "int16",
    np.int32: "int32",
    np.int64: "int64",
    np.byte: "int8",
    np.short: "int16",
    np.intc: "int32",
    np.int_: "int64",
    np.intp: "int64",
    np.longlong: "q",
}
r"""Dictionary of all signed `numpy` integer data type typestrings."""

NUMPY_UINT_TYPESTRINGS: Final[dict[type[np.unsignedinteger], str]] = {
    np.uint8: "uint8",
    np.uint16: "uint16",
    np.uint32: "uint32",
    np.uint64: "uint64",
    np.ubyte: "uint8",
    np.ushort: "uint16",
    np.uintc: "uint32",
    np.uint: "uint64",
    np.uintp: "uint64",
    np.ulonglong: "Q",
}
r"""Dictionary of all unsigned `numpy` integer data type typestrings."""

NUMPY_FLOAT_TYPESTRINGS: Final[dict[type[np.floating], str]] = {
    np.float_: "float64",
    np.float16: "float16",
    np.float32: "float32",
    np.float64: "float64",
    # np.float128: "float128",
    np.half: "float16",
    np.single: "float32",
    np.double: "float64",
    np.longdouble: "float128",
    np.longfloat: "float128",
}
r"""Dictionary of all `numpy` float data type typestrings."""

NUMPY_COMPLEX_TYPESTRINGS: Final[dict[type[np.complexfloating], str]] = {
    np.complex64: "complex64",
    np.complex128: "complex128",
    # np.complex256: "complex256",
    np.csingle: "complex64",
    np.singlecomplex: "complex64",
    np.cdouble: "complex128",
    np.cfloat: "complex128",
    np.complex_: "complex128",
    np.clongdouble: "complex256",
    np.clongfloat: "complex256",
    np.longcomplex: "complex256",
}
r"""Dictionary of all `numpy` complex data typestrings."""


NUMPY_TIME_TYPESTRINGS: Final[dict[type[np.generic], str]] = {
    np.timedelta64: "timedelta64",  # timedelta64
    np.datetime64: "datetime64",  # datetime64
}
r"""Dictionary of all `numpy` time data type typestrings."""

NUMPY_BOOL_TYPESTRINGS: Final[dict[type[np.generic], str]] = {
    np.bool_: "bool",  # bool
}
r"""Dictionary of all `numpy` bool data type typestrings."""

NUMPY_STRING_TYPESTRINGS: Final[dict[type[np.flexible], str]] = {
    np.bytes_: "bytes",  # str
    np.string_: "str",  # bytes
    np.str_: "str",  # str
    np.unicode_: "unicode",  # unicode
    np.void: "void",  # "void"
}
r"""Dictionary of all `numpy` string data type typestrings."""

NUMPY_OBJECT_TYPESTRINGS: Final[dict[type[np.generic], str]] = {
    np.object_: "object",
}
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
# endregion numpy typestrings -----------------------------------------------------------------------------------------


# region pandas dtypes ------------------------------------------------------------------------------------------------
PANDAS_TYPESTRINGS: Final[dict[type[ExtensionDtype], str]] = {
    pandas.BooleanDtype: "boolean",
    pandas.CategoricalDtype: "category",
    pandas.DatetimeTZDtype: "datetime64[ns, tz]",  # datetime64[ns, <tz>]
    pandas.Float32Dtype: "Float32",
    pandas.Float64Dtype: "Float64",
    pandas.Int16Dtype: "Int16",
    pandas.Int32Dtype: "Int32",
    pandas.Int64Dtype: "Int64",
    pandas.Int8Dtype: "Int8",
    pandas.IntervalDtype: "interval",  # e.g. to denote ranges of variables
    pandas.PeriodDtype: "period",  # period[<freq>]
    pandas.SparseDtype: "Sparse",
    pandas.StringDtype: "string",
    pandas.UInt16Dtype: "UInt16",
    pandas.UInt32Dtype: "UInt32",
    pandas.UInt64Dtype: "UInt64",
    pandas.UInt8Dtype: "UInt8",
}
r"""Dictionary of all `pandas` data type typestrings."""

PANDAS_NULLABLE_DTYPES: Final[dict[str, type[ExtensionDtype]]] = {
    "boolean": pandas.BooleanDtype,
    "category": pandas.CategoricalDtype,
    "datetime64[ns, tz]": pandas.DatetimeTZDtype,  # datetime64[ns, <tz>]
    "Float32": pandas.Float32Dtype,
    "Float64": pandas.Float64Dtype,
    "Int16": pandas.Int16Dtype,
    "Int32": pandas.Int32Dtype,
    "Int64": pandas.Int64Dtype,
    "Int8": pandas.Int8Dtype,
    "interval": pandas.IntervalDtype,  # e.g. to denote ranges of variables
    "period": pandas.PeriodDtype,  # period[<freq>]
    "Sparse": pandas.SparseDtype,
    "string": pandas.StringDtype,
    "UInt16": pandas.UInt16Dtype,
    "UInt32": pandas.UInt32Dtype,
    "UInt64": pandas.UInt64Dtype,
    "UInt8": pandas.UInt8Dtype,
}
r"""Dictionary of all `pandas` data types."""
# endregion pandas dtypes ---------------------------------------------------------------------------------------------


# region polars dtypes ------------------------------------------------------------------------------------------------
PYARROW_DTYPES: Final[dict[str, pyarrow.DataType]] = {
    # fmt: off
    # numeric
    "null"    : pyarrow.null(),
    "bool"    : pyarrow.bool_(),
    "int8"    : pyarrow.int8(),
    "int16"   : pyarrow.int16(),
    "int32"   : pyarrow.int32(),
    "int64"   : pyarrow.int64(),
    "uint8"   : pyarrow.uint8(),
    "uint16"  : pyarrow.uint16(),
    "uint32"  : pyarrow.uint32(),
    "uint64"  : pyarrow.uint64(),
    "float16" : pyarrow.float16(),
    "float32" : pyarrow.float32(),
    "float64" : pyarrow.float64(),
    # temporal
    "date32"  : pyarrow.date32(),
    "date64"  : pyarrow.date64(),
    "time32"  : pyarrow.time32("s"),
    "time64"  : pyarrow.time64("ns"),
    "timestamp[ns]" : pyarrow.timestamp("ns"),
    "timestamp[us]" : pyarrow.timestamp("us"),
    "timestamp[ms]" : pyarrow.timestamp("ms"),
    "timestamp[s]"  : pyarrow.timestamp("s"),
    "duration[ns]"  : pyarrow.duration("ns"),
    "duration[us]"  : pyarrow.duration("us"),
    "duration[ms]"  : pyarrow.duration("ms"),
    "duration[s]"   : pyarrow.duration("s"),
    # string/binary
    "binary"  : pyarrow.binary(),
    "large_string"  : pyarrow.large_string(),
    "string"  : pyarrow.string(),
}
r"""Dictionary of all `pyarrow` data types."""
# endregion polars dtypes ---------------------------------------------------------------------------------------------


# region polars dtypes ------------------------------------------------------------------------------------------------
POLARS_DTYPES: Final[dict[str, polars.DataType]] = {
    # fmt: off
    # numeric
    "Float32"    : polars.Float32(),
    "Float64"    : polars.Float64(),
    "Int8"       : polars.Int8(),
    "Int16"      : polars.Int16(),
    "Int32"      : polars.Int32(),
    "Int64"      : polars.Int64(),
    "UInt8"      : polars.UInt8(),
    "UInt16"     : polars.UInt16(),
    "UInt32"     : polars.UInt32(),
    "UInt64"     : polars.UInt64(),
    # temporal
    "Date"       : polars.Date(),
    "Datetime"   : polars.Datetime(),
    "Duration"   : polars.Duration(),
    "Time"       : polars.Time(),
    # other
    "Binary"     : polars.Binary(),
    "Boolean"    : polars.Boolean(),
    "Categorical": polars.Categorical(),
    "Null"       : polars.Null(),
    "Object"     : polars.Object(),
    "Utf8"       : polars.Utf8(),
    # fmt: on
}
r"""Dictionary of all elementary `polars` data types."""
# endregion polars dtypes ---------------------------------------------------------------------------------------------


# region torch dtypes --------------------------------------------------------------------------------------------------
TORCH_INT_TYPESTRINGS: Final[dict[torch.dtype, str]] = {
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.qint8: "qint8",
    torch.qint32: "qint32",
}
r"""Dictionary of all `torch` signed integer data type typestrings."""

TORCH_UINT_TYPESTRINGS: Final[dict[torch.dtype, str]] = {
    torch.uint8: "uint8",
    # torch.uint16: "uint16",
    # torch.uint32: "uint32",
    # torch.uint64: "uint64",
    torch.quint8: "quint8",
    torch.quint4x2: "quint4x2",
}
r"""Dictionary of all `torch` unsigned integer data type typestrings."""

TORCH_FLOAT_TYPESTRINGS: Final[dict[torch.dtype, str]] = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
}
r"""Dictionary of all `torch` float data type typestrings."""

TORCH_COMPLEX_TYPESTRINGS: Final[dict[torch.dtype, str]] = {
    torch.complex32: "complex32",
    torch.complex64: "complex64",
    torch.complex128: "complex128",
}
r"""Dictionary of all `torch` complex data type typestrings."""

TORCH_BOOL_TYPESTRINGS: Final[dict[torch.dtype, str]] = {
    torch.bool: "bool",
}
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
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "qint8": torch.qint8,
    "qint32": torch.qint32,
    "uint8": torch.uint8,
    "quint8": torch.quint8,
    "quint4x2": torch.quint4x2,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "complex32": torch.complex32,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
    "bool": torch.bool,
}
r"""Dictionary of all `torch` data types."""
# endregion torch dtypes -----------------------------------------------------------------------------------------------


# region python dtypes -------------------------------------------------------------------------------------------------
PYTHON_DTYPES: Final[dict[str, type]] = {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "str": str,
    "bytes": bytes,
    "datetime": datetime,
    "timedelta": timedelta,
    "object": object,
}
r"""Dictionary of all `python` data types."""

PYTHON_TYPESTRINGS: Final[dict[type, str]] = {
    bool: "bool",
    int: "int",
    float: "float",
    complex: "complex",
    str: "str",
    bytes: "bytes",
    datetime: "datetime",
    timedelta: "timedelta",
    object: "object",
}
r"""Dictionary of all `python` data types."""
# endregion python dtypes ----------------------------------------------------------------------------------------------


# region dtype conversion ----------------------------------------------------------------------------------------------
PYARROW_TO_POLARS: Final[dict[pyarrow.DataType, polars.DataType]] = {
    # fmt: off
    pyarrow.null()          : polars.Null(),
    pyarrow.bool_()         : polars.Boolean(),
    pyarrow.int8()          : polars.Int8(),
    pyarrow.int16()         : polars.Int16(),
    pyarrow.int32()         : polars.Int32(),
    pyarrow.int64()         : polars.Int64(),
    pyarrow.uint8()         : polars.UInt8(),
    pyarrow.uint16()        : polars.UInt16(),
    pyarrow.uint32()        : polars.UInt32(),
    pyarrow.uint64()        : polars.UInt64(),
    pyarrow.float16()       : polars.Float32(),
    pyarrow.float32()       : polars.Float32(),
    pyarrow.float64()       : polars.Float64(),
    pyarrow.date32()        : polars.Date(),
    pyarrow.date64()        : polars.Date(),
    pyarrow.time32("s")     : polars.Time(),
    pyarrow.time64("ns")    : polars.Time(),
    pyarrow.timestamp("ns") : polars.Datetime(),
    pyarrow.timestamp("us") : polars.Datetime(),
    pyarrow.timestamp("ms") : polars.Datetime(),
    pyarrow.timestamp("s")  : polars.Datetime(),
    pyarrow.duration("ns")  : polars.Duration(),
    pyarrow.duration("us")  : polars.Duration(),
    pyarrow.duration("ms")  : polars.Duration(),
    pyarrow.duration("s")   : polars.Duration(),
    pyarrow.binary()        : polars.Binary(),
    pyarrow.large_string()  : polars.Utf8(),
    pyarrow.string()        : polars.Utf8(),
    pyarrow.dictionary(pyarrow.int32(), pyarrow.string())   : polars.Categorical(),
    # fmt: on
}
r"""Dictionary of converting pyarrow to polars."""
# endregion dtype conversion -------------------------------------------------------------------------------------------


TYPESTRINGS: Final[dict[type[np.generic] | torch.dtype | type[ExtensionDtype], str]] = (
    NUMPY_TYPESTRINGS | TORCH_TYPESTRINGS | PANDAS_TYPESTRINGS  # type: ignore[operator]
)
r"""Dictionary of all type strings."""
