r"""Implementation of loss functions.

Theory
------
We define the following

1. A metric is a  function

    .. math:: 𝔪： 𝓟_0(𝓨×𝓨) ⟶ ℝ_{≥0}

    such that $𝔪(\{ (y_i, \hat{y}_i) ∣ i=1:n \}) = 0$ if and only if $y_i=\hat{y}_i∀i$

2. A metric is called decomposable, if it can be written as a function

    .. math
        𝔪 = Ψ∘(ℓ×𝗂𝖽)
        ℓ： 𝓨×𝓨 ⟶ ℝ_{≥0}
        Ψ： 𝓟_0(ℝ_{≥0}) ⟶ ℝ_{≥0}

    I.e. the function $ℓ$ is applied element-wise to all pairs $(y, \hat{y}$ and the function $Ψ$
    "accumulates" the results. Oftentimes, $Ψ$ is just the sum/mean/expectation value, although
    other accumulations such as the median value are also possible.

3. A metric is called instance-wise, if it can be written in the form

    .. math::
        𝔪： 𝓟_0(𝓨×𝓨) ⟶ ℝ_{≥ 0}, 𝔪(\{(y_i, \hat{y}_i) ∣  i=1:n \})
        = ∑_{i=1}^n ω(i, n)ℓ(y_i, \hat{y}_i)

4. A metric is called a loss-function, if and only if

   - It is differentiable almost everywhere.
   - It is non-constant, at least on some open set.

Note that in the context of time-series, we allow the accumulator to depend on the time variable.

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
    "BaseLoss",
    "FunctionalLoss",
    # Constants
    "LOSSES",
    "FUNCTIONAL_LOSSES",
    "MODULAR_LOSSES",
    # Base Classes
    "BaseLoss",
    "WeightedLoss",
    # Classes
    "MAE",
    "MSE",
    "RMSE",
    "WMAE",
    "WMSE",
    "WRMSE",
    # Functions
    "nd",
    "nrmse",
    "rmse",
    "q_quantile",
    "q_quantile_loss",
]


from abc import ABCMeta

from torch import nn

from tsdm.metrics._modular import (
    MAE,
    MSE,
    RMSE,
    WMAE,
    WMSE,
    WRMSE,
    BaseLoss,
    Loss,
    WeightedLoss,
)
from tsdm.metrics._timeseries import (
    ND,
    NRMSE,
    Q_Quantile,
    Q_Quantile_Loss,
    TimeSeriesMSE,
    TimeSeriesWMSE,
)
from tsdm.metrics.functional import (
    FUNCTIONAL_LOSSES,
    FunctionalLoss,
    nd,
    nrmse,
    q_quantile,
    q_quantile_loss,
    rmse,
)

TORCH_ALIASES: dict[str, type[nn.Module]] = {
    "MAE": nn.L1Loss,
    "L2": nn.MSELoss,
    "XENT": nn.CrossEntropyLoss,
    "KL": nn.KLDivLoss,
}
r"""Dictionary containing additional aliases for modular losses in torch."""


TORCH_LOSSES: dict[str, type[nn.Module]] = {
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
} | TORCH_ALIASES
r"""Dictionary of all available modular losses in torch."""


MODULAR_LOSSES: dict[str, type[Loss]] = {
    "ND": ND,
    "NRMSE": NRMSE,
    "Q_Quantile": Q_Quantile,
    "Q_Quantile_Loss": Q_Quantile_Loss,
    "MAE": MAE,
    "MSE": MSE,
    "RMSE": RMSE,
    "WMAE": WMAE,
    "WMSE": WMSE,
    "WRMSE": WRMSE,
    "TimeSeriesMSE": TimeSeriesMSE,
    "TimeSeriesWMSE": TimeSeriesWMSE,
}
r"""Dictionary of all available modular losses."""


LOSSES: dict[str, FunctionalLoss | type[Loss]] = {
    **FUNCTIONAL_LOSSES,
    **MODULAR_LOSSES,
}
r"""Dictionary of all available losses."""

del nn, ABCMeta
