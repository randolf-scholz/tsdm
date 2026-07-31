from typing import Final

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
    "NAN",     "NaN",     "nan",     "#NAN",     "#NaN",     "#nan",     "<NAN>",     "<NaN>",     "<nan>",
    "+NAN",    "+NaN",    "+nan",    "#+NAN",    "#+NaN",    "#+nan",    "<+NAN>",    "<+NaN>",    "<+nan>",
    "-NAN",    "-NaN",    "-nan",    "#-NAN",    "#-NaN",    "#-nan",    "<-NAN>",    "<-NaN>",    "<-nan>",
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
