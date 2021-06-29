r"""
tsdm.plot
=========
Visualization functions
"""

import matplotlib

from .plotting import shared_grid_plot, visualize_distribution

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{{amsmath}}"
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['svg.fonttype'] = 'none'

__all__ = ['shared_grid_plot', 'visualize_distribution']