r"""Plotting Functionality."""

__all__ = [
    # Functions
    "shared_grid_plot",
    "visualize_distribution",
    "plot_spectrum",
    "kernel_heatmap",
    "rasterize",
    "center_axes",
]

import shutil
import warnings

import matplotlib

MATPLOTLIB_CONFIG = {
    # "mathtext.fontset": "stix",
    # "font.family": "STIXGeneral",
    # "svg.fonttype": "none",
    "text.usetex": True,
    "pgf.texsystem": r"lualatex",
    "pgf.preamble": "\n".join(
        [
            r"\usepackage{fontspec}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage[utf8x]{inputenc}",
            r"\usepackage{amsmath}",
            r"\usepackage{amsfonts}",
            r"\usepackage{amssymb}",
            r"\usepackage{unicode-math}",
        ]
    ),
    "text.latex.preamble": "\n".join(
        [
            r"\usepackage{amsmath}",
            r"\usepackage{amsfonts}",
            r"\usepackage{amssymb}",
        ]
    ),
}

try:
    matplotlib.use("pgf")
except ValueError:
    warnings.warn("matplotlib: pgf backend not available / no LaTeX rendering!")
else:
    matplotlib.rcParams.update(MATPLOTLIB_CONFIG)

# pylint: disable=wrong-import-position
from tsdm.viz._image import kernel_heatmap  # noqa: E402
from tsdm.viz._plotting import (  # noqa: E402
    center_axes,
    plot_spectrum,
    rasterize,
    shared_grid_plot,
    visualize_distribution,
)

# pylint: enable=wrong-import-position

del matplotlib, shutil, warnings
