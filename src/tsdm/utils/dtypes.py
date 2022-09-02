r"""TODO: Module Docstring.

TODO: Module description.
"""

__all__ = [
    # Constants
    "BOOLS",
    "CATEGORIES",
    "PRECISION",
    "EMOJIS",
    "STRINGS",
]

from collections import namedtuple
from typing import Final

import numpy as np
import pandas
import torch
from pandas.api.extensions import ExtensionDtype

PRECISION: Final[dict] = {
    16: 2**-11,
    32: 2**-24,
    64: 2**-53,
    torch.float16: 2**-11,
    torch.float32: 2**-24,
    torch.float64: 2**-53,
    np.float16: 2**-11,
    np.float32: 2**-24,
    np.float64: 2**-53,
}
r"""Maps precision to the corresponding precision factor."""

NUMPY_INT_DTYPES: Final[dict[type[np.signedinteger], str]] = {
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
r"""Dictionary of all signed `numpy` integer data types."""

NUMPY_UINT_DTYPES: Final[dict[type[np.unsignedinteger], str]] = {
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
r"""Dictionary of all unsigned `numpy` integer data types."""

NUMPY_FLOAT_DTYPES: Final[dict[type[np.floating], str]] = {
    np.float_: "d",
    np.float16: "e",
    np.float32: "f",
    np.float64: "d",
    np.float128: "g",
    np.half: "e",
    np.single: "f",
    np.double: "d",
    np.longdouble: "g",
    np.longfloat: "g",
}
r"""Dictionary of all `numpy` float data types."""

NUMPY_COMPLEX_DTYPES: Final[dict[type[np.complexfloating], str]] = {
    np.complex64: "F",
    np.complex128: "D",
    np.complex256: "G",
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


NUMPY_TIME_DTYPES: Final[dict[type[np.generic], str]] = {
    np.timedelta64: "M",
    np.datetime64: "m",
}
r"""Dictionary of all `numpy` time data types."""

NUMPY_BOOL_DTYPES: Final[dict[type[np.generic], str]] = {
    np.bool_: "?",
}
r"""Dictionary of all `numpy` bool data types."""

NUMPY_STRING_DTYPES: Final[dict[type[np.flexible], str]] = {
    np.bytes_: "S",
    np.string_: "S",
    np.str_: "U",
    np.unicode_: "U",
    np.void: "V",
}
r"""Dictionary of all `numpy` string types."""

NUMPY_OTHER_DTYPES: Final[dict[type[np.generic], str]] = {
    np.object_: "O",
}
r"""Dictionary of all `numpy` generic data types."""

PANDAS_DTYPES: Final[dict[type[ExtensionDtype], str]] = {
    pandas.BooleanDtype: "boolean",
    pandas.CategoricalDtype: "category",
    pandas.DatetimeTZDtype: "datetime64",  # datetime64[ns, <tz>]
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
r"""Dictionary of all `pandas` data types."""


NULLABLE_DTYPES: Final[
    dict[type[np.generic] | type[ExtensionDtype], str]
] = PANDAS_DTYPES | {
    np.timedelta64: "M",
    np.datetime64: "m",
}

BOOLS: Final[list[bool]] = [True, False]
r"""List of example bool objects."""

EMOJIS: Final[list[str]] = list(
    "😀😁😂😃😄😅😆😇😈😉😊😋😌😍😎😏"
    "😐😑😒😓😔😕😖😗😘😙😚😛😜😝😞😟"
    "😠😡😢😣😤😥😦😧😨😩😪😫😬😭😮😯"
    "😰😱😲😳😴😵😶😷😸😹😺😻😼😽😾😿"
    "🙀🙁🙂🙃🙄🙅🙆🙇🙈🙉🙊🙋🙌🙍🙎🙏"
)
r"""List of example unicode objects."""


STRINGS: Final[list[str]] = [
    "Alfa",
    "Bravo",
    "Charlie",
    "Delta",
    "Echo",
    "Foxtrot",
    "Golf",
    "Hotel",
    "India",
    "Juliett",
    "Kilo",
    "Lima",
    "Mike",
    "November",
    "Oscar",
    "Papa",
    "Quebec",
    "Romeo",
    "Sierra",
    "Tango",
    "Uniform",
    "Victor",
    "Whiskey",
    "X-ray",
    "Yankee",
    "Zulu",
]
r"""List of example string objects."""


label = namedtuple("label", ["object", "color"])

CATEGORIES: Final[list[label]] = [
    label(object="bear", color="brown"),
    label(object="bear", color="black"),
    label(object="bear", color="white"),
    label(object="beet", color="red"),
    label(object="beet", color="yellow"),
    label(object="beet", color="orange"),
    label(object="beet", color="white"),
    label(object="beet", color="violet"),
]
r"""List of example categorical objects."""
