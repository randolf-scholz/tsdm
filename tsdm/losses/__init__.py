r"""
tsdm.losses
===========

Implementation of loss functions
"""

from .functional import nd, nrmse
from .module import ND, NRMSE

__all__ = ['nd', 'nrmse', 'ND', 'NRMSE']
