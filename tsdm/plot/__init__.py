r"""Plotting Functionality.

TODO: Module summary.
"""

from __future__ import annotations

__all__ = [
    # Sub-Modules
    "__setup__",
    # Functions
    "shared_grid_plot",
    "visualize_distribution",
    "plot_spectrum",
    "kernel_heatmap",
]

import logging

from tsdm.plot import __setup__
from tsdm.plot.image import kernel_heatmap
from tsdm.plot.plotting import plot_spectrum, shared_grid_plot, visualize_distribution

__logger__ = logging.getLogger(__name__)
