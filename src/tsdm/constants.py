r"""Constants used throughout the package."""

__all__ = [
    # Constant Functions
    "CONST_TRUE_FN",
    "CONST_FALSE_FN",
    "CONST_NONE_FN",
    "CONSTANT_FUNCTIONS",
    # Constants
    "ATOL",
    "RTOL",
    "EPS",
    "BUILTIN_CONSTANTS",
    "BUILTIN_TYPES",
    "EMPTY_PATH",
    "EMPTY_MAP",
    "NULL_VALUES",
    "PRECISION",
    "BOOLEAN_PAIRS",
    "NA_VALUES",
    "NA_STRINGS",
    "EXAMPLE_BOOLS",
    "EXAMPLE_EMOJIS",
    "EXAMPLE_STRINGS",
    "EXAMPLE_CATEGORIES",
    "TIME_UNITS",
]

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas
import torch
from typing_extensions import Any, Final, Literal

ATOL: Final[float] = 1e-6
"""CONST: Default absolute precision."""
RTOL: Final[float] = 1e-6
"""CONST: Default relative precision."""
EPS: Final[dict[torch.dtype, float]] = {
    torch.float16: 1e-3,
    torch.float32: 1e-6,
    torch.float64: 1e-15,
    # complex floats
    torch.complex32: 1e-3,
    torch.complex64: 1e-6,
    torch.complex128: 1e-15,
    # other floats
    torch.bfloat16: 1e-2,
}
"""CONST: Default epsilon for each dtype."""

EMPTY_PATH: Final[Path] = Path()
r"""Constant: Blank path."""

TIME_UNITS: dict[str, np.timedelta64] = {
    u: np.timedelta64(1, u)
    for u in ("Y", "M", "W", "D", "h", "m", "s", "us", "ns", "ps", "fs", "as")
}


# NOTE: Use frozenmap() if PEP 603 is accepted.
EMPTY_MAP: Final[Mapping] = MappingProxyType({})
r"""Constant: Immutable Empty `Mapping`."""


def CONST_TRUE_FN(*_: Any, **__: Any) -> Literal[True]:
    """Constant True Function."""
    return True


def CONST_FALSE_FN(*_: Any, **__: Any) -> Literal[False]:
    """Constant False Function."""
    return False


def CONST_NONE_FN(*_: Any, **__: Any) -> Literal[None]:
    """Constant None Function."""
    return None


BUILTIN_CONSTANTS = [None, True, False, Ellipsis, NotImplemented]
"""Builtin constants https://docs.python.org/3/library/constants.html."""

CONSTANT_FUNCTIONS = [CONST_TRUE_FN, CONST_FALSE_FN, CONST_NONE_FN]

BUILTIN_TYPES: Final[list[type]] = [
    type(None),
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
    type(Ellipsis),
    type(NotImplemented),
]
"""Builtin types https://docs.python.org/3/library/stdtypes.html."""


NA_STRINGS: Final[set[str]] = {
    r"",
    r"-",
    r"n/a",
    r"N/A",
    r"<na>",
    r"<NA>",
    r"nan",
    r"NaN",
    r"NAN",
    r"NaT",
    r"none",
    r"None",
    r"NONE",
}
r"""String that correspond to NA values."""

NA_VALUES: Final[set] = {
    None,
    float("nan"),
    np.nan,
    pandas.NA,
    pandas.NaT,
    np.datetime64("NaT"),
}
r"""Values that correspond to NaN."""

# fmt: off
NULL_VALUES: Final[list[str]] = [
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
]
r"""A list of common null value string represenations."""
# fmt: on

BOOLEAN_PAIRS: Final[list[dict[str | int | float, bool]]] = [
    # fmt: off
    {"f"     : False, "t"    : True},
    {"false" : False, "true" : True},
    {"n"     : False, "y"    : True},
    {"no"    : False, "yes"  : True},
    {"-"     : False, "+"    : True},
    {0       : False, 1      : True},
    {-1      : False, +1     : True},
    {0.0     : False, 1.0    : True},
    {-1.0    : False, +1.0   : True},
    # fmt: on
]
r"""Matched pairs of values that correspond to booleans."""

PRECISION: Final[dict] = {
    # fmt: off
    16            : 2**-11,
    32            : 2**-24,
    64            : 2**-53,
    torch.float16 : 2**-11,
    torch.float32 : 2**-24,
    torch.float64 : 2**-53,
    np.float16    : 2**-11,
    np.float32    : 2**-24,
    np.float64    : 2**-53,
    # fmt: on
}
r"""Maps precision to the corresponding precision factor."""


# region example collections------------------------------------------------------------

EXAMPLE_BOOLS: Final[list[bool]] = [True, False]
r"""List of example bool objects."""

EXAMPLE_EMOJIS: Final[list[str]] = list(
    "😀😁😂😃😄😅😆😇😈😉😊😋😌😍😎😏"
    "😐😑😒😓😔😕😖😗😘😙😚😛😜😝😞😟"
    "😠😡😢😣😤😥😦😧😨😩😪😫😬😭😮😯"
    "😰😱😲😳😴😵😶😷😸😹😺😻😼😽😾😿"
    "🙀🙁🙂🙃🙄🙅🙆🙇🙈🙉🙊🙋🙌🙍🙎🙏"
)
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
