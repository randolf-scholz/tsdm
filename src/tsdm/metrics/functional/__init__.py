r"""Implementations of loss functions.

Contains losses in functional form.
"""

__all__ = [
    # Types
    "Metric",
    # Constants
    "FUNCTIONAL_LOSSES",
    "TORCH_ALIASES",
    "TORCH_FUNCTIONAL_LOSSES",
    "TORCH_SPECIAL_LOSSES",
    # Functions
    "nd",
    "nrmse",
    "q_quantile",
    "q_quantile_loss",
    "rmse",
]

from torch import nn

from tsdm.metrics.functional._functional import (
    Metric,
    nd,
    nrmse,
    q_quantile,
    q_quantile_loss,
    rmse,
)

TORCH_FUNCTIONAL_LOSSES: dict[str, Metric] = {
    "binary_cross_entropy": nn.functional.binary_cross_entropy,
    # Function that measures the Binary Cross Entropy between the target and the output.
    "binary_cross_entropy_with_logits": nn.functional.binary_cross_entropy_with_logits,
    # Function that measures Binary Cross Entropy between target and output logits.
    "poisson_nll": nn.functional.poisson_nll_loss,
    # Poisson negative log likelihood loss.
    "cross_entropy": nn.functional.cross_entropy,
    # This criterion combines log_softmax and nll_loss in a single function.
    "hinge_embedding": nn.functional.hinge_embedding_loss,
    # See HingeEmbeddingLoss for details.
    "kl_div": nn.functional.kl_div,
    # The Kullback-Leibler divergence Loss
    "l1": nn.functional.l1_loss,
    # Function that takes the mean element-wise absolute value difference.
    "mse": nn.functional.mse_loss,
    # Measures the element-wise mean squared error.
    "multilabel_margin": nn.functional.multilabel_margin_loss,
    # See MultiLabelMarginLoss for details.
    "multilabel_soft_margin": nn.functional.multilabel_soft_margin_loss,
    # See MultiLabelSoftMarginLoss for details.
    "multi_margin": nn.functional.multi_margin_loss,
    # multi_margin_loss(input, target, p=1, margin=1, weight=None, size_average=None,
    "nll": nn.functional.nll_loss,
    # The negative log likelihood loss.
    "huber": nn.functional.huber_loss,
    # Function that uses a squared term if the absolute element-wise error falls below
    # delta and a delta-scaled L1 term otherwise.
    "smooth_l1": nn.functional.smooth_l1_loss,
    # Function that uses a squared term if the absolute element-wise error falls below
    # beta and an L1 term otherwise.
    "soft_margin": nn.functional.soft_margin_loss,
    # See SoftMarginLoss for details.
}
r"""Dictionary of all available losses in torch."""

TORCH_SPECIAL_LOSSES = {
    "cosine_embedding": nn.functional.cosine_embedding_loss,
    # See CosineEmbeddingLoss for details.
    "ctc_loss": nn.functional.ctc_loss,
    # The Connectionist Temporal Classification loss.
    "gaussian_nll": nn.functional.gaussian_nll_loss,
    # Gaussian negative log likelihood loss.
    "margin_ranking": nn.functional.margin_ranking_loss,
    # See MarginRankingLoss for details.
    "triplet_margin": nn.functional.triplet_margin_loss,
    # See TripletMarginLoss for details
    "triplet_margin_with_distance": nn.functional.triplet_margin_with_distance_loss,
    # See TripletMarginWithDistanceLoss for details.
}
"""Special losses that do not represent usual loss functions."""

TORCH_ALIASES: dict[str, Metric] = {
    "mae": nn.functional.l1_loss,
    "l2": nn.functional.mse_loss,
    "xent": nn.functional.cross_entropy,
    "kl": nn.functional.kl_div,
}
r"""Dictionary containing additional aliases for losses in torch."""

FUNCTIONAL_LOSSES: dict[str, Metric] = {
    "nd": nd,
    "nrmse": nrmse,
    "q_quantile": q_quantile,
    "q_quantile_loss": q_quantile_loss,
} | (TORCH_FUNCTIONAL_LOSSES | TORCH_ALIASES)
r"""Dictionary of all available functional losses."""

del nn
