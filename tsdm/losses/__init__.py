r"""Implementation of loss functions.

tsdm.losses
===========
"""

from .functional import nd, nrmse
from .module import ND, NRMSE

__all__ = ["nd", "nrmse", "ND", "NRMSE"]
