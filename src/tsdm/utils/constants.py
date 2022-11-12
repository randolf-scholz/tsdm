r"""TODO: Add module summary line.

TODO: Add module description.
"""

__all__ = [
    "EMPTY_PATH",
    "NULL_VALUES",
    "PRECISION",
    "BOOLEAN_PAIRS",
    "NA_VALUES",
    "NA_STRINGS",
    "EXAMPLE_BOOLS",
    "EXAMPLE_EMOJIS",
    "EXAMPLE_STRINGS",
    "EXAMPLE_CATEGORIES",
]

from pathlib import Path
from typing import Final

import numpy as np
import pandas
import torch

EMPTY_PATH: Final[Path] = Path()
r"""Constant: Blank path."""

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
    {"f": False, "t": True},
    {"false": False, "true": True},
    {"n": False, "y": True},
    {"no": False, "yes": True},
    {"-": False, "+": True},
    {0: False, 1: True},
    {-1: False, +1: True},
    {0.0: False, 1.0: True},
    {-1.0: False, +1.0: True},
]
r"""Matched pairs of values that correspond to booleans."""

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


# region example collections--------------------------------------------------------------------------------------------

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

# endregion example collections-----------------------------------------------------------------------------------------
