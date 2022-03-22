r"""Implementations of loss functions.

Notes
-----
Contains losses in functional form.
  - See :mod:`tsdm.losses.modular` for modular implementations.
"""

__all__ = [
    # Types
    "FunctionalLoss",
    # Constants
    "FunctionalLosses",
    # Functions
    "nd",
    "nrmse",
    "rmse",
    "q_quantile",
    "q_quantile_loss",
]

import logging
from collections.abc import Callable
from typing import Final

from torch import Tensor
from torch.nn import functional as F

from tsdm.losses.functional._functional import (
    nd,
    nrmse,
    q_quantile,
    q_quantile_loss,
    rmse,
)

__logger__ = logging.getLogger(__name__)

# TODO: use better definition [Tensor, Tensor, ...] -> Tensor once supported
FunctionalLoss = Callable[..., Tensor]
r"""Type hint for functional losses."""

TORCH_LOSSES: Final[dict[str, FunctionalLoss]] = {
    "binary_cross_entropy": F.binary_cross_entropy,
    # Function that measures the Binary Cross Entropy between the target and the output.
    "binary_cross_entropy_with_logits": F.binary_cross_entropy_with_logits,
    # Function that measures Binary Cross Entropy between target and output logits.
    "poisson_nll": F.poisson_nll_loss,
    # Poisson negative log likelihood loss.
    "cosine_embedding": F.cosine_embedding_loss,
    # See CosineEmbeddingLoss for details.
    "cross_entropy": F.cross_entropy,
    # This criterion combines log_softmax and nll_loss in a single function.
    "ctc_loss": F.ctc_loss,
    # The Connectionist Temporal Classification loss.
    "gaussian_nll": F.gaussian_nll_loss,
    # Gaussian negative log likelihood loss.
    "hinge_embedding": F.hinge_embedding_loss,
    # See HingeEmbeddingLoss for details.
    "kl_div": F.kl_div,
    # The Kullback-Leibler divergence Loss
    "l1": F.l1_loss,
    # Function that takes the mean element-wise absolute value difference.
    "mse": F.mse_loss,
    # Measures the element-wise mean squared error.
    "margin_ranking": F.margin_ranking_loss,
    # See MarginRankingLoss for details.
    "multilabel_margin": F.multilabel_margin_loss,
    # See MultiLabelMarginLoss for details.
    "multilabel_soft_margin": F.multilabel_soft_margin_loss,
    # See MultiLabelSoftMarginLoss for details.
    "multi_margin": F.multi_margin_loss,
    # multi_margin_loss(input, target, p=1, margin=1, weight=None, size_average=None,
    "nll": F.nll_loss,
    # The negative log likelihood loss.
    "huber": F.huber_loss,
    # Function that uses a squared term if the absolute element-wise error falls below
    # delta and a delta-scaled L1 term otherwise.
    "smooth_l1": F.smooth_l1_loss,
    # Function that uses a squared term if the absolute element-wise error falls below
    # beta and an L1 term otherwise.
    "soft_margin": F.soft_margin_loss,
    # See SoftMarginLoss for details.
    "triplet_margin": F.triplet_margin_loss,
    # See TripletMarginLoss for details
    "triplet_margin_with_distance": F.triplet_margin_with_distance_loss,
    # See TripletMarginWithDistanceLoss for details.
}
r"""Dictionary of all available losses in torch."""

TORCH_ALIASES: Final[dict[str, FunctionalLoss]] = {
    "mae": F.l1_loss,
    "l2": F.mse_loss,
    "xent": F.cross_entropy,
    "kl": F.kl_div,
}
r"""Dictionary containing additional aliases for losses in torch."""

FunctionalLosses: Final[dict[str, FunctionalLoss]] = {
    "nd": nd,
    "nrmse": nrmse,
    "q_quantile": q_quantile,
    "q_quantile_loss": q_quantile_loss,
} | (TORCH_LOSSES | TORCH_ALIASES)
r"""Dictionary of all available functional losses."""
