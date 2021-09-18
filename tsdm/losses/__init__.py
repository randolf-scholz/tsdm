r"""Implementation of loss functions.

contains object oriented loss functions.
See `tsdm.losses.functional` for functional implementations.
"""

import logging
from typing import Final, Type

from torch import nn

from tsdm.losses import functional
from tsdm.losses.modular import ND, NRMSE, Q_Quantile, Q_Quantile_Loss

logger = logging.getLogger(__name__)
__all__: Final[list[str]] = [
    "functional",
    "LOSSES",
    "ND",
    "NRMSE",
    "Q_Quantile",
    "Q_Quantile_Loss",
]

LOSSES: Final[dict[str, Type[nn.Module]]] = {
    "ND": ND,
    "NRMSE": NRMSE,
    "Q_Quantile": Q_Quantile,
    "Q_Quantile_Loss": Q_Quantile_Loss,
}
