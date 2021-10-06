r"""Implementations of loss functions.

Notes
-----
Contains losses in modular form.
  - See :mod:`tsdm.losses.functional` for functional implementations.
"""

from __future__ import annotations

__all__ = [
    # Constants
    "ModularLoss",
    "ModularLosses",
    "ModularLossType",
    # Classes
    "ND",
    "NRMSE",
    "Q_Quantile",
    "Q_Quantile_Loss",
]


import logging
from typing import Final

from torch import nn

from tsdm.losses.modular._modular import ND, NRMSE, Q_Quantile, Q_Quantile_Loss

LOGGER = logging.getLogger(__name__)


ModularLoss = nn.Module
r"""Type hint for modular losses."""
ModularLossType = type[nn.Module]
r"""Type hint for modular losses."""
TORCH_LOSSES: Final[dict[str, ModularLossType]] = {
    "L1": nn.L1Loss,
    "CosineEmbedding": nn.CosineEmbeddingLoss,
    "CrossEntropy": nn.CrossEntropyLoss,
    "CTC": nn.CTCLoss,
    "NLL": nn.NLLLoss,
    "PoissonNLL": nn.PoissonNLLLoss,
    "GaussianNLL": nn.GaussianNLLLoss,
    "KLDiv": nn.KLDivLoss,
    "BCE": nn.BCELoss,
    "BCEWithLogits": nn.BCEWithLogitsLoss,
    "MarginRanking": nn.MarginRankingLoss,
    "MSE": nn.MSELoss,
    "HingeEmbedding": nn.HingeEmbeddingLoss,
    "Huber": nn.HuberLoss,
    "SmoothL1": nn.SmoothL1Loss,
    "SoftMargin": nn.SoftMarginLoss,
    "MultiMargin": nn.MultiMarginLoss,
    "MultiLabelMargin": nn.MultiLabelMarginLoss,
    "MultiLabelSoftMargin": nn.MultiLabelSoftMarginLoss,
    "TripletMargin": nn.TripletMarginLoss,
    "TripletMarginWithDistance": nn.TripletMarginWithDistanceLoss,
}
r"""Dictionary of all available losses in torch."""

TORCH_ALIASES: Final[dict[str, ModularLossType]] = {
    "MAE": nn.L1Loss,
    "L2": nn.MSELoss,
    "XENT": nn.CrossEntropyLoss,
    "KL": nn.KLDivLoss,
}
r"""Dictionary containing additional aliases for losses in torch."""

ModularLosses: Final[dict[str, ModularLossType]] = {
    "ND": ND,
    "NRMSE": NRMSE,
    "Q_Quantile": Q_Quantile,
    "Q_Quantile_Loss": Q_Quantile_Loss,
} | (TORCH_LOSSES | TORCH_ALIASES)
r"""Dictionary of all available modular losses."""
