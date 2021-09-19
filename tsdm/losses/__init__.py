r"""Implementation of loss functions.

contains object oriented loss functions.
See `tsdm.losses.functional` for functional implementations.
"""

import logging
from typing import Final, Type

from torch.nn import Module

from tsdm.losses import functional
from tsdm.losses.modular import ND, NRMSE, Q_Quantile, Q_Quantile_Loss

LOGGER = logging.getLogger(__name__)
__all__: Final[list[str]] = [
    "Loss",
    "LOSSES",
    "functional",
    "ND",
    "NRMSE",
    "Q_Quantile",
    "Q_Quantile_Loss",
]


Loss = Type[Module]
r"""Type hint for losses."""

LOSSES: Final[dict[str, Loss]] = {
    "ND": ND,
    "NRMSE": NRMSE,
    "Q_Quantile": Q_Quantile,
    "Q_Quantile_Loss": Q_Quantile_Loss,
}
r"""Dictionary containing all available losses."""
