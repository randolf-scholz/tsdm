r"""Implementation of loss functions.

contains object oriented loss functions.
See `tsdm.losses.functional` for functional implementations.
"""

import logging
from typing import Final, Type

from torch import nn

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


Loss = Type[nn.Module]
r"""Type hint for losses."""

TORCH_LOSSES: Final[dict[str, Loss]] = {
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
r"""Dictionary containing all available losses in torch."""

TORCH_ALIASES: Final[dict[str, Loss]] = {
    "MAE": nn.L1Loss,
    "L2": nn.MSELoss,
    "XENT": nn.CrossEntropyLoss,
    "KL": nn.KLDivLoss,
}
r"""Dictionary containing additional aliases for losses in torch."""

LOSSES: Final[dict[str, Loss]] = (
    {
        "ND": ND,
        "NRMSE": NRMSE,
        "Q_Quantile": Q_Quantile,
        "Q_Quantile_Loss": Q_Quantile_Loss,
    }
    | TORCH_LOSSES
    | TORCH_ALIASES
)
r"""Dictionary containing all available losses."""
