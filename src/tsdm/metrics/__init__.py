r"""Implementation of loss functions.

Contains losses in both modular and functional form.

Theory
------
We define the following

1. A metric is a  function

    .. math:: 𝔪： ⋃_{n∈ℕ}(𝓨×𝓨)^n ⟶ ℝ_{≥0}
        \qq{s.t.} \text{$𝔪(Y，Ŷ) = 0$ if and only if $y_n=ŷ_n∀n=1:N$}

    I.e., a function that takes a finite number of pairs $(y_n, ŷ_n)_{n=1:N}$
    and returns a non-negative scalar. We denote $Y≔(y_n)_n$ and $Ŷ≔(ŷ_n)_n$
    and write $𝔪(Y，Ŷ)$ for the metric value.

2. A metric is called **decomposable**, if and only if it can be written as a composition
   of an **aggregation function** $Ψ$ and an **intance-wise loss function** $ℓ$:

    .. math:: 𝔪 = Ψ∘(ℓ×𝗂𝖽) \qq{with} ℓ：𝓨×𝓨 ⟶ ℝ_{≥0} \qq{and} Ψ：⋃_{n∈ℕ}ℝ^n ⟶ ℝ_{≥0}

    I.e. the function $ℓ$ is applied element-wise to all pairs $(y, ŷ)$ and the function $Ψ$
    accumulates the results. Typical choices of $ψ$ are:

    - sum: $Ψ(r) = ∑_n r_n$
    - mean: $Ψ(r) = 𝐄_n r_n ≔ \frac{1}{N} ∑_{n=1}^N r_N$
    - median: $Ψ(r) = 𝐌_n r_n ≔ \Median((r_n)_{n=1:N})$

3. A metric is called **instance-wise** if it can be written in the form

    .. math:: 𝔪： ⋃_{n∈ℕ}(𝓨×𝓨)^n ⟶ ℝ_{≥0}, 𝔪(Y，Ŷ) = ∑_{n=1}^N ω(n,N) ℓ(y_n，ŷ_n)

    with a weight function $ω：ℕ×ℕ ⟶ ℝ_{≥0}$ and an instance-wise loss function $ℓ$.

4. A metric is called a loss-function, if and only if

   - It is differentiable almost everywhere.
   - It is non-constant, at least on some open set.

Note that in the context of time-series, we allow the accumulator to depend on the time variable.

See Also:
    - `tsdm.losses.functional` for functional implementations.
    - `tsdm.losses` for modular implementations.
"""

__all__ = [
    # Sub-Modules
    "functional",
    # Protocols
    "Metric",
    "NN_Metric",
    # Constants
    "LOSSES",
    "FUNCTIONAL_LOSSES",
    "MODULAR_LOSSES",
    "TORCH_LOSSES",
    "TORCH_ALIASES",
    "TORCH_SPECIAL_LOSSES",
    # Base Classes
    "BaseMetric",
    "WeightedMetric",
    # Classes
    "MAE",
    "MSE",
    "RMSE",
    "WMAE",
    "WMSE",
    "WRMSE",
    "ND",
    "NRMSE",
    "Q_Quantile",
    "Q_Quantile_Loss",
    "TimeSeriesMSE",
    "TimeSeriesWMSE",
    # Functions
    "nd",
    "nrmse",
    "rmse",
    "q_quantile",
    "q_quantile_loss",
]


from torch import nn

from tsdm.metrics._modular import (
    MAE,
    MSE,
    RMSE,
    WMAE,
    WMSE,
    WRMSE,
    BaseMetric,
    NN_Metric,
    WeightedMetric,
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
    Metric,
    nd,
    nrmse,
    q_quantile,
    q_quantile_loss,
    rmse,
)

TORCH_ALIASES: dict[str, type[NN_Metric]] = {
    "MAE": nn.L1Loss,
    "L2": nn.MSELoss,
    "XENT": nn.CrossEntropyLoss,
    "KL": nn.KLDivLoss,
}
r"""Dictionary containing additional aliases for modular losses in torch."""


TORCH_LOSSES: dict[str, type[NN_Metric]] = {
    "L1": nn.L1Loss,
    "CrossEntropy": nn.CrossEntropyLoss,
    "NLL": nn.NLLLoss,
    "PoissonNLL": nn.PoissonNLLLoss,
    "KLDiv": nn.KLDivLoss,
    "BCE": nn.BCELoss,
    "BCEWithLogits": nn.BCEWithLogitsLoss,
    "MSE": nn.MSELoss,
    "HingeEmbedding": nn.HingeEmbeddingLoss,
    "Huber": nn.HuberLoss,
    "SmoothL1": nn.SmoothL1Loss,
    "SoftMargin": nn.SoftMarginLoss,
    "MultiMargin": nn.MultiMarginLoss,
    "MultiLabelMargin": nn.MultiLabelMarginLoss,
    "MultiLabelSoftMargin": nn.MultiLabelSoftMarginLoss,
}
r"""Dictionary of all available modular losses in torch."""


TORCH_SPECIAL_LOSSES: dict[str, type[nn.Module]] = {
    "CosineEmbedding": nn.CosineEmbeddingLoss,
    "CTC": nn.CTCLoss,
    "GaussianNLL": nn.GaussianNLLLoss,
    "MarginRanking": nn.MarginRankingLoss,
    "TripletMargin": nn.TripletMarginLoss,
    "TripletMarginWithDistance": nn.TripletMarginWithDistanceLoss,
}
"""Special losses that do not represent usual loss functions."""


MODULAR_LOSSES: dict[str, type[Metric]] = {
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


LOSSES: dict[str, Metric | type[Metric]] = {
    **FUNCTIONAL_LOSSES,
    **MODULAR_LOSSES,
}
r"""Dictionary of all available losses."""

del nn
