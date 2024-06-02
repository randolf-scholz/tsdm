r"""Constants used throughout the package."""

__all__ = [
    # Constants
    "ATOL",
    "BOOLEAN_PAIRS",
    "BUILTIN_CONSTANTS",
    "BUILTIN_TYPES",
    "EMPTY_MAP",
    "EPS",
    "EXAMPLE_BOOLS",
    "EXAMPLE_CATEGORIES",
    "EXAMPLE_EMOJIS",
    "EXAMPLE_STRINGS",
    "KEYWORD_ONLY",
    "NA_STRINGS",
    "NA_VALUES",
    "NULL_VALUES",
    "POSITIONAL_ONLY",
    "POSITIONAL_OR_KEYWORD",
    "PRECISION",
    "RNG",
    "ROOT_3",
    "RTOL",
    "TIME_UNITS",
    "VAR_KEYWORD",
    "VAR_POSITIONAL",
]

from collections.abc import Mapping
from inspect import Parameter
from types import EllipsisType, MappingProxyType, NoneType, NotImplementedType

import numpy as np
import pandas as pd
import torch
from numpy.random import Generator
from typing_extensions import Any, Final, Never

# NOTE: Use frozenmap() if PEP 603 is accepted.
EMPTY_MAP: Final[Mapping[Any, Never]] = MappingProxyType({})
r"""Constant: Immutable Empty `Mapping`, use as default in function signatures."""
RNG: Final[Generator] = np.random.default_rng()
r"""Default random number generator."""
ROOT_3: Final[float] = float(np.sqrt(3))
r"""Square root of 3."""
ATOL: Final[float] = 1e-6
r"""CONST: Default absolute precision."""
RTOL: Final[float] = 1e-6
r"""CONST: Default relative precision."""
EPS: Final[dict[torch.dtype, float]] = {
    torch.bfloat16   : 1e-2,
    torch.complex128 : 1e-15,
    torch.complex32  : 1e-3,
    torch.complex64  : 1e-6,
    torch.float16    : 1e-3,
    torch.float32    : 1e-6,
    torch.float64    : 1e-15,
}  # fmt: skip
r"""CONST: Default epsilon for each dtype."""

TIME_UNITS: Final[dict[str, np.timedelta64]] = {
    "Y": np.timedelta64(1, "Y"),
    "M": np.timedelta64(1, "M"),
    "W": np.timedelta64(1, "W"),
    "D": np.timedelta64(1, "D"),
    "h": np.timedelta64(1, "h"),
    "m": np.timedelta64(1, "m"),
    "s": np.timedelta64(1, "s"),
    "us": np.timedelta64(1, "us"),
    "ns": np.timedelta64(1, "ns"),
    "ps": np.timedelta64(1, "ps"),
    "fs": np.timedelta64(1, "fs"),
    "as": np.timedelta64(1, "as"),
}
r"""Time units for `numpy.timedelta64`."""

BUILTIN_CONSTANTS: Final[frozenset[object]] = frozenset({
    None,
    True,
    False,
    Ellipsis,
    NotImplemented,
})
r"""Builtin constants https://docs.python.org/3/library/constants.html."""

BUILTIN_TYPES: Final[frozenset[type]] = frozenset({
    NoneType,
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    list,
    tuple,
    set,
    frozenset,
    dict,
    type,
    slice,
    range,
    object,
    EllipsisType,
    NotImplementedType,
})
r"""Builtin types https://docs.python.org/3/library/stdtypes.html."""

# region Parameter constants------------------------------------------------------------
KEYWORD_ONLY = Parameter.KEYWORD_ONLY
POSITIONAL_ONLY = Parameter.POSITIONAL_ONLY
POSITIONAL_OR_KEYWORD = Parameter.POSITIONAL_OR_KEYWORD
VAR_KEYWORD = Parameter.VAR_KEYWORD
VAR_POSITIONAL = Parameter.VAR_POSITIONAL
# endregion Parameter constants---------------------------------------------------------


NA_STRINGS: Final[frozenset[str]] = frozenset({
    "", "-",
    "n/a", "N/A",
    "<na>", "<NA>",
    "nan", "NaN", "NAN",
    "NaT",
    "none", "None", "NONE",
})  # fmt: skip
r"""String that correspond to NA values."""

NULL_VALUES: Final[frozenset[str]] = frozenset({
    "", "-", "--", "?", "??",
    "1.#IND", "+1.#IND", "-1.#IND", "1.#QNAN", "+1.#QNAN", "-1.#QNAN",
    "#N/A N/A",
    "NaT",
    "N.A.",    "N.a.",    "n.a.",    "#N.A.",    "#N.a.",    "#n.a.",    "<N.A.>",    "<N.a.>",    "<n.a.>",
                                     "#NA",      "#Na",      "#na",      "<NA>",      "<Na>",      "<na>",
    "N/A",     "N/a",     "n/a",     "#N/A",     "#N/a",     "#n/a",     "<N/A>",     "<N/a>",     "<n/a>",
    "NAN",     "NaN",     "nan",     "#NAN",     "#NaN",     "#nan",     "<NAN>",     "<NaN>",     "<nan>"
    "+NAN",    "+NaN",    "+nan",    "#+NAN",    "#+NaN",    "#+nan",    "<+NAN>",    "<+NaN>",    "<+nan>"
    "-NAN",    "-NaN",    "-nan",    "#-NAN",    "#-NaN",    "#-nan",    "<-NAN>",    "<-NaN>",    "<-nan>"
    "-N/A",    "-N/a",    "-n/a",    "#-N/A",    "#-N/a",    "#-n/a",    "<-N/A>",    "<-N/a>",    "<-n/a>",
    "+N/A",    "+N/a",    "+n/a",    "#+N/A",    "#+N/a",    "#+n/a",    "<+N/A>",    "<+N/a>",    "<+n/a>",
    "NONE",    "None",    "none",    "#NONE",    "#None",    "#none",    "<NONE>",    "<None>",    "<none>",
    "NULL",    "Null",    "null",    "#NULL",    "#Null",    "#null",    "<NULL>",    "<Null>",    "<null>",
    "MISS",    "Miss",    "miss",    "#MISS",    "#Miss",    "#miss",    "<MISS>",    "<Miss>",    "<miss>",
    "UNKNOWN", "Unknown", "unknown", "#UNKNOWN", "#Unknown", "#unknown", "<UNKNOWN>", "<Unknown>", "<unknown>",
    "MISSING", "Missing", "missing", "#MISSING", "#Missing", "#missing", "<MISSING>", "<Missing>", "<missing>",
    "NOT APPLICABLE", "not applicable",
    "NOT AVAILABLE",  "not available",
    "NO ANSWER",      "no answer",
})  # fmt: skip
r"""A list of common null value string represenations."""

NA_VALUES: Final[frozenset[object]] = frozenset({
    None,
    float("nan"),
    np.nan,
    pd.NA,
    pd.NaT,
    np.datetime64("NaT"),
})
r"""Values that correspond to NaN."""

BOOLEAN_PAIRS: Final[list[dict[str | int | float, bool]]] = [
    {"f"     : False, "t"    : True},
    {"false" : False, "true" : True},
    {"n"     : False, "y"    : True},
    {"no"    : False, "yes"  : True},
    {"-"     : False, "+"    : True},
    {0       : False, 1      : True},
    {-1      : False, +1     : True},
    {0.0     : False, 1.0    : True},
    {-1.0    : False, +1.0   : True},
]  # fmt: skip
r"""Matched pairs of values that correspond to booleans."""

PRECISION: Final[dict[int | type | torch.dtype, float]] = {
    16            : 2**-11,
    32            : 2**-24,
    64            : 2**-53,
    torch.float16 : 2**-11,
    torch.float32 : 2**-24,
    torch.float64 : 2**-53,
    np.float16    : 2**-11,
    np.float32    : 2**-24,
    np.float64    : 2**-53,
}  # fmt: skip
r"""Maps precision to the corresponding precision factor."""


# region example collections------------------------------------------------------------

EXAMPLE_BOOLS: Final[list[bool]] = [True, False]
r"""List of example bool objects."""

EXAMPLE_EMOJIS: Final[list[str]] = [
    "😀", "😁", "😂", "😃", "😄", "😅", "😆", "😇", "😈", "😉", "😊", "😋", "😌", "😍", "😎", "😏"
    "😐", "😑", "😒", "😓", "😔", "😕", "😖", "😗", "😘", "😙", "😚", "😛", "😜", "😝", "😞", "😟"
    "😠", "😡", "😢", "😣", "😤", "😥", "😦", "😧", "😨", "😩", "😪", "😫", "😬", "😭", "😮", "😯"
    "😰", "😱", "😲", "😳", "😴", "😵", "😶", "😷", "😸", "😹", "😺", "😻", "😼", "😽", "😾", "😿"
    "🙀", "🙁", "🙂", "🙃", "🙄", "🙅", "🙆", "🙇", "🙈", "🙉", "🙊", "🙋", "🙌", "🙍", "🙎", "🙏"
]  # fmt: skip
r"""List of example unicode objects."""

EXAMPLE_STRINGS: Final[list[str]] = [
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

EXAMPLE_CATEGORIES: Final[list[tuple[str, str]]] = [
    ("bear", "brown"),
    ("bear", "black"),
    ("bear", "white"),
    ("beet", "red"),
    ("beet", "yellow"),
    ("beet", "orange"),
    ("beet", "white"),
    ("beet", "violet"),
]
r"""List of example categorical objects."""

# endregion example collections---------------------------------------------------------
