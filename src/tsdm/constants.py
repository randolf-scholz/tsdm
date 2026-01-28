r"""Constants used throughout the package."""

__all__ = [
    # ENUMS
    "FLOAT",
    # Constants
    "ATOL",
    "BOOLEAN_PAIRS",
    "BUILTIN_CONSTANTS",
    "BUILTIN_TYPES",
    "EMPTY_FN",
    "EMPTY_MAP",
    "EMPTY_SET",
    "EXAMPLE_BOOLS",
    "EXAMPLE_CATEGORIES",
    "EXAMPLE_EMOJIS",
    "EXAMPLE_STRINGS",
    "IDENTITY",
    "KEYWORD_ONLY",
    "NA_STRINGS",
    "NA_VALUES",
    "UNDEFINED",
    "NULL_VALUES",
    "POSITIONAL_ONLY",
    "POSITIONAL_OR_KEYWORD",
    "RNG",
    "RTOL",
    "VAR_KEYWORD",
    "VAR_POSITIONAL",
]

import math
from collections.abc import Callable, Hashable, Mapping, Set as AbstractSet
from enum import Enum
from inspect import Parameter
from types import EllipsisType, MappingProxyType, NoneType, NotImplementedType
from typing import Any, Final, Never

import numpy as np
import pandas as pd
from numpy.random import Generator


class FLOAT(float, Enum):
    r"""Enum: Common floating point values."""

    ZERO = 0.0
    ONE = 1.0
    INF = math.inf
    NAN = math.nan

    E = math.e
    PI = math.pi
    ROOT_2 = math.sqrt(2)
    ROOT_2PI = math.sqrt(2 * math.pi)
    ROOT_3 = math.sqrt(3)


NA_VALUES: Final[frozenset[Hashable]] = frozenset({
    None,
    float("nan"),
    np.nan,
    pd.NA,
    pd.NaT,
    np.datetime64("NaT"),
})
r"""Values that correspond to NaN."""

# region precision constants -----------------------------------------------------------
RNG: Final[Generator] = np.random.default_rng()
r"""Default random number generator."""
ATOL: Final[float] = 1e-6
r"""CONST: Default absolute precision."""
RTOL: Final[float] = 1e-6
r"""CONST: Default relative precision."""
# endregion precision constants --------------------------------------------------------


# region callable constants ------------------------------------------------------------
IDENTITY: Final[Callable] = lambda _: _  # noqa: E731
r"""Constant: Identity function, use as default in function signatures."""
EMPTY_FN: Final[Callable[..., None]] = lambda *_, **__: None  # noqa: E731
r"""Constant: Empty function, use as default in function signatures."""
# region collection constants ----------------------------------------------------------

EMPTY_MAP: Final[Mapping[Any, Never]] = MappingProxyType({})  # FIXME: PEP 603
r"""Constant: Immutable empty `Mapping`, use as default in function signatures."""
EMPTY_SET: Final[AbstractSet[Any]] = frozenset()
r"""Constant: Immutable empty `Set`, use as default in function signatures."""
# endregion collection constants -------------------------------------------------------


UNDEFINED: Final[Any] = object()
r"""CONST: Default value for optional arguments."""


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
