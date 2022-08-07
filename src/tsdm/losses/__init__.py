r"""Implementation of loss functions.

Notes
-----
Contains losses in both modular and functional form.
  - See `tsdm.losses.functional` for functional implementations.
  - See `tsdm.losses` for modular implementations.
"""

__all__ = [
    # Sub-Modules
    "functional",
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
    "WRMSE",
    "RMSE",
    # Functions
    "nd",
    "nrmse",
    "rmse",
    "q_quantile",
    "q_quantile_loss",
]


from typing import Final, Union

from torch import nn

from tsdm.losses._modular import ND, NRMSE, RMSE, WRMSE, Q_Quantile, Q_Quantile_Loss
from tsdm.losses.functional import (
    FunctionalLoss,
    FunctionalLosses,
    nd,
    nrmse,
    q_quantile,
    q_quantile_loss,
    rmse,
)

ModularLoss = nn.Module
r"""Type hint for modular losses."""

TORCH_LOSSES: Final[dict[str, type[nn.Module]]] = {
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

TORCH_ALIASES: Final[dict[str, type[nn.Module]]] = {
    "MAE": nn.L1Loss,
    "L2": nn.MSELoss,
    "XENT": nn.CrossEntropyLoss,
    "KL": nn.KLDivLoss,
}
r"""Dictionary containing additional aliases for modular losses in torch."""

ModularLosses: Final[dict[str, type[nn.Module]]] = {
    "ND": ND,
    "NRMSE": NRMSE,
    "Q_Quantile": Q_Quantile,
    "Q_Quantile_Loss": Q_Quantile_Loss,
    "RMSE": RMSE,
} | (TORCH_LOSSES | TORCH_ALIASES)
r"""Dictionary of all available modular losses."""


Loss = Union[FunctionalLoss, ModularLoss]
r"""Type hint for losses."""

LOSSES: Final[dict[str, Union[FunctionalLoss, type[ModularLoss]]]] = {
    **FunctionalLosses,
    **ModularLosses,
}
r"""Dictionary of all available losses."""
