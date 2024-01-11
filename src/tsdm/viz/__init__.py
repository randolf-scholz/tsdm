r"""Plotting Functionality."""

__all__ = [
    # Constants
    "MATPLOTLIB_CONFIG",
    # Functions
    "center_axes",
    "kernel_heatmap",
    "plot_spectrum",
    "rasterize",
    "enable_latex_plotting",
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
from tsdm.viz._setup import MATPLOTLIB_CONFIG, enable_latex_plotting
