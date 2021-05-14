"""
tsdm.util
=========

Provides utility functions

"""

import matplotlib
from .util import ACTIVATIONS, deep_dict_update, deep_kval_update, scaled_norm, visualize_distribution, relative_error

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{{amsmath}}"
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

__all__ = ['ACTIVATIONS',
           'deep_dict_update', 'deep_kval_update',
           'relative_error', 'scaled_norm',
           'visualize_distribution']
