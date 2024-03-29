r"""Plotting Functionality."""

__all__ = [
    # Constants
    "MATPLOTLIB_CONFIG",
    # Functions
    "center_axes",
    "kernel_heatmap",
    "plot_spectrum",
    "rasterize",
    "set_latex_plotting",
    "shared_grid_plot",
    "visualize_distribution",
]

from tsdm.viz._image import kernel_heatmap, rasterize
from tsdm.viz._plotting import (
    center_axes,
    plot_spectrum,
    shared_grid_plot,
    visualize_distribution,
)

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


def set_latex_plotting() -> None:
    r"""Set matplotlib to use LaTeX for rendering."""
    # pylint: disable=import-outside-toplevel
    import shutil
    import warnings

    import matplotlib

    # pylint: enable=import-outside-toplevel

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
    finally:
        del matplotlib, shutil, warnings
