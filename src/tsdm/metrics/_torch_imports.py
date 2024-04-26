"""Metrics imported from torch."""

__all__ = [
    # Constants
    "TORCH_ALIASES",
    "TORCH_ALIASES_FUNCTIONAL",
    "TORCH_LOSSES",
    "TORCH_LOSSES_FUNCTIONAL",
    "TORCH_SPECIAL_LOSSES",
    "TORCH_SPECIAL_LOSSES_FUNCTIONAL",
]

from collections.abc import Callable

from torch import Tensor, nn

from tsdm.metrics.base import Metric, NN_Metric

TORCH_LOSSES_FUNCTIONAL: dict[str, Metric] = {
    "binary_cross_entropy"             : nn.functional.binary_cross_entropy,
    "binary_cross_entropy_with_logits" : nn.functional.binary_cross_entropy_with_logits,
    "cross_entropy"                    : nn.functional.cross_entropy,
    "hinge_embedding"                  : nn.functional.hinge_embedding_loss,
    "huber"                            : nn.functional.huber_loss,
    "kl_div"                           : nn.functional.kl_div,
    "l1"                               : nn.functional.l1_loss,
    "mse"                              : nn.functional.mse_loss,
    "multi_margin"                     : nn.functional.multi_margin_loss,
    "multilabel_margin"                : nn.functional.multilabel_margin_loss,
    "multilabel_soft_margin"           : nn.functional.multilabel_soft_margin_loss,
    "nll"                              : nn.functional.nll_loss,
    "poisson_nll"                      : nn.functional.poisson_nll_loss,
    "smooth_l1"                        : nn.functional.smooth_l1_loss,
    "soft_margin"                      : nn.functional.soft_margin_loss,
}  # fmt: skip
r"""Dictionary of all available losses in torch."""


TORCH_SPECIAL_LOSSES_FUNCTIONAL: dict[str, Callable[..., Tensor]] = {
    "cosine_embedding"             : nn.functional.cosine_embedding_loss,
    "ctc_loss"                     : nn.functional.ctc_loss,
    "gaussian_nll"                 : nn.functional.gaussian_nll_loss,
    "margin_ranking"               : nn.functional.margin_ranking_loss,
    "triplet_margin"               : nn.functional.triplet_margin_loss,
    "triplet_margin_with_distance" : nn.functional.triplet_margin_with_distance_loss,
}  # fmt: skip
r"""Special losses that do not represent usual loss functions."""

TORCH_ALIASES_FUNCTIONAL: dict[str, Metric] = {
    "mae"  : nn.functional.l1_loss,
    "l2"   : nn.functional.mse_loss,
    "xent" : nn.functional.cross_entropy,
    "kl"   : nn.functional.kl_div,
}  # fmt: skip
r"""Dictionary containing additional aliases for losses in torch."""

TORCH_ALIASES: dict[str, type[NN_Metric]] = {
    "KL"   : nn.KLDivLoss,
    "L2"   : nn.MSELoss,
    "MAE"  : nn.L1Loss,
    "XENT" : nn.CrossEntropyLoss,
}  # fmt: skip
r"""Dictionary containing additional aliases for modular losses in torch."""


TORCH_LOSSES: dict[str, type[NN_Metric]] = {
    "BCE"                  : nn.BCELoss,
    "BCEWithLogits"        : nn.BCEWithLogitsLoss,
    "CrossEntropy"         : nn.CrossEntropyLoss,
    "HingeEmbedding"       : nn.HingeEmbeddingLoss,
    "Huber"                : nn.HuberLoss,
    "KLDiv"                : nn.KLDivLoss,
    "L1"                   : nn.L1Loss,
    "MSE"                  : nn.MSELoss,
    "MultiLabelMargin"     : nn.MultiLabelMarginLoss,
    "MultiLabelSoftMargin" : nn.MultiLabelSoftMarginLoss,
    "MultiMargin"          : nn.MultiMarginLoss,
    "NLL"                  : nn.NLLLoss,
    "PoissonNLL"           : nn.PoissonNLLLoss,
    "SmoothL1"             : nn.SmoothL1Loss,
    "SoftMargin"           : nn.SoftMarginLoss,
}  # fmt: skip
r"""Dictionary of all available modular losses in torch."""


TORCH_SPECIAL_LOSSES: dict[str, type[nn.Module]] = {
    "CTC"                       : nn.CTCLoss,
    "CosineEmbedding"           : nn.CosineEmbeddingLoss,
    "GaussianNLL"               : nn.GaussianNLLLoss,
    "MarginRanking"             : nn.MarginRankingLoss,
    "TripletMargin"             : nn.TripletMarginLoss,
    "TripletMarginWithDistance" : nn.TripletMarginWithDistanceLoss,
}  # fmt: skip
r"""Special losses that do not represent usual loss functions."""
