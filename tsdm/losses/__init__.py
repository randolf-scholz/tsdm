r"""Implementation of loss functions.

Notes
-----
Contains losses in both modular and functional form.
  - See :mod:`tsdm.losses.functional` for functional implementations.
  - See :mod:`tsdm.losses.modular` for modular implementations.
"""

from __future__ import annotations

__all__ = [
    # Sub-Modules
    "functional",
    "modular",
    # Types
    "Loss",
    "LossType",
    "FunctionalLoss",
    "FunctionalLossType",
    "ModularLoss",
    "ModularLossType",
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
    "q_quantile",
    "q_quantile_loss",
]


import logging
from typing import Final, Union

from tsdm.losses import functional, modular
from tsdm.losses.functional import (
    FunctionalLoss,
    FunctionalLosses,
    FunctionalLossType,
    nd,
    nrmse,
    q_quantile,
    q_quantile_loss,
)
from tsdm.losses.modular import (
    ND,
    NRMSE,
    ModularLoss,
    ModularLosses,
    ModularLossType,
    Q_Quantile,
    Q_Quantile_Loss,
)

LOGGER = logging.getLogger(__name__)

Loss = Union[FunctionalLoss, ModularLoss]
r"""Type hint for losses."""

LossType = Union[FunctionalLossType, ModularLossType]
r"""Type hint for losses."""

LOSSES: Final[dict[str, LossType]] = {**FunctionalLosses, **ModularLosses}
r"""Dictionary of all available losses."""
