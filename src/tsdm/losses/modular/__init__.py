r"""Implementations of loss functions.

Notes
-----
Contains losses in modular form.
  - See :mod:`tsdm.losses.functional` for functional implementations.
"""

__all__ = [
    # Types
    "ModularLoss",
    # Constants,
    "ModularLosses",
    # Classes
    "ND",
    "NRMSE",
    "Q_Quantile",
    "Q_Quantile_Loss",
    "WRMSE",
    "RMSE",
]


import logging
from typing import Final

from torch import nn

from tsdm.losses.modular._modular import (
    ND,
    NRMSE,
    RMSE,
    WRMSE,
    Q_Quantile,
    Q_Quantile_Loss,
)
from tsdm.util.types import LookupTable

__logger__ = logging.getLogger(__name__)


ModularLoss = nn.Module
r"""Type hint for modular losses."""

TORCH_LOSSES: Final[LookupTable[type[nn.Module]]] = {
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
r"""Dictionary of all available modular losses in torch."""

TORCH_ALIASES: Final[LookupTable[type[nn.Module]]] = {
    "MAE": nn.L1Loss,
    "L2": nn.MSELoss,
    "XENT": nn.CrossEntropyLoss,
    "KL": nn.KLDivLoss,
}
r"""Dictionary containing additional aliases for modular losses in torch."""

ModularLosses: Final[LookupTable[type[nn.Module]]] = {
    "ND": ND,
    "NRMSE": NRMSE,
    "Q_Quantile": Q_Quantile,
    "Q_Quantile_Loss": Q_Quantile_Loss,
    "RMSE": RMSE,
} | (TORCH_LOSSES | TORCH_ALIASES)
r"""Dictionary of all available modular losses."""
