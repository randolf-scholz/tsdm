"""Plotting configuration."""

__all__ = ["MATPLOTLIB_CONFIG", "enable_latex_plotting"]

import shutil
import warnings

import matplotlib

MATPLOTLIB_CONFIG = {
    # "mathtext.fontset": "stix",
    # "font.family": "STIXGeneral",
    # "svg.fonttype": "none",
    "text.usetex": True,
    "pgf.texsystem": r"lualatex",
    "pgf.preamble": "\n".join([
        r"\usepackage{fontspec}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8x]{inputenc}",
        r"\usepackage{amsmath}",
        r"\usepackage{amsfonts}",
        r"\usepackage{amssymb}",
        r"\usepackage{unicode-math}",
    ]),
    "text.latex.preamble": "\n".join([
        r"\usepackage{amsmath}",
        r"\usepackage{amsfonts}",
        r"\usepackage{amssymb}",
    ]),
}


def enable_latex_plotting() -> None:
    r"""Enable matplotlib to use LaTeX for rendering."""
    try:
        if shutil.which("lualatex") is not None:
            matplotlib.rcParams.update(MATPLOTLIB_CONFIG)
            matplotlib.use("pgf")
        else:
            warnings.warn(
                "lualatex not found. Using default matplotlib backend.", stacklevel=1
            )
    except ValueError:
        warnings.warn(
            "matplotlib: pgf backend not available / no LaTeX rendering!", stacklevel=1
        )
