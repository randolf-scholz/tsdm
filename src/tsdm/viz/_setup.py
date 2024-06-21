r"""Plotting configuration."""

__all__ = [
    # Constants
    "MATPLOTLIB_CONFIG",
    "PGF_PREAMBLE",
    "LATEX_PREAMBLE",
    # Functions
    "enable_latex_plotting",
]

import shutil
import warnings

import matplotlib as mpl

PGF_PREAMBLE = r"""
\usepackage{fontspec}
\usepackage[T1]{fontenc}
\usepackage[utf8x]{inputenc}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{unicode-math}
"""

LATEX_PREAMBLE = r"""
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
"""

MATPLOTLIB_CONFIG = {
    # "mathtext.fontset": "stix",
    # "font.family": "STIXGeneral",
    # "svg.fonttype": "none",
    "text.usetex": True,
    "pgf.texsystem": r"lualatex",
    "pgf.preamble": PGF_PREAMBLE,
    "text.latex.preamble": LATEX_PREAMBLE,
}


def enable_latex_plotting() -> None:
    r"""Enable matplotlib to use LaTeX for rendering."""
    try:
        if shutil.which("lualatex") is not None:
            mpl.rcParams.update(MATPLOTLIB_CONFIG)
            mpl.use("pgf")
        else:
            warnings.warn(
                "lualatex not found. Using default matplotlib backend.", stacklevel=1
            )
    except ValueError:
        warnings.warn(
            "matplotlib: pgf backend not available / no LaTeX rendering!", stacklevel=1
        )
