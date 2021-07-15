r"""Implementation of loss functions.

tsdm.losses
===========
"""

from tsdm.losses.functional import nd, nrmse
from tsdm.losses.module import ND, NRMSE

__all__ = ["nd", "nrmse",] + [
    "ND",
    "NRMSE",
]
