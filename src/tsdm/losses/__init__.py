r"""Implementation of loss functions.

Notes
-----
Contains losses in both modular and functional form.
  - See :mod:`tsdm.losses.functional` for functional implementations.
  - See :mod:`tsdm.losses.modular` for modular implementations.
"""

__all__ = [
    # Sub-Modules
    "functional",
    "modular",
    # Types
    "Loss",
    "FunctionalLoss",
    "ModularLoss",
    # Constants
    "LOSSES",
    "FunctionalLosses",
    "ModularLosses",
    # Classes
    "ND",
    "NRMSE",
    "Q_Quantile",
    "Q_Quantile_Loss",
    # Functions
    "nd",
    "nrmse",
    "rmse",
    "q_quantile",
    "q_quantile_loss",
]


import logging
from typing import Final, Union

from tsdm.losses import functional, modular
from tsdm.losses.functional import (
    FunctionalLoss,
    FunctionalLosses,
    nd,
    nrmse,
    q_quantile,
    q_quantile_loss,
    rmse,
)
from tsdm.losses.modular import (
    ND,
    NRMSE,
    ModularLoss,
    ModularLosses,
    Q_Quantile,
    Q_Quantile_Loss,
)
from tsdm.util.types import LookupTable

__logger__ = logging.getLogger(__name__)

Loss = Union[FunctionalLoss, ModularLoss]
r"""Type hint for losses."""

LOSSES: Final[LookupTable[Union[FunctionalLoss, type[ModularLoss]]]] = {
    **FunctionalLosses,
    **ModularLosses,
}
r"""Dictionary of all available losses."""
