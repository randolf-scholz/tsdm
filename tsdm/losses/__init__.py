r"""Implementation of loss functions.

contains object oriented loss functions.
See `tsdm.losses.functional` for functional implementations.
"""

from tsdm.losses import functional
from tsdm.losses._module import ND, NRMSE


__all__ = ["functional", "LOSSES", "ND", "NRMSE"]

LOSSES = {
    "ND": ND,
    "NRMSE": NRMSE,
}
