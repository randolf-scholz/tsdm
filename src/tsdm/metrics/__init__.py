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
    "modular",
    "timeseries",
    # Constants
    "LOSSES",
    "FUNCTIONAL_LOSSES",
    "MODULAR_LOSSES",
    "TORCH_ALIASES",
    "TORCH_ALIASES_FUNCTIONAL",
    "TORCH_LOSSES",
    "TORCH_LOSSES_FUNCTIONAL",
    "TORCH_SPECIAL_LOSSES",
    "TORCH_SPECIAL_LOSSES_FUNCTIONAL",
    # ABCs & Protocols
    "Metric",
    "BaseMetric",
    "WeightedMetric",
    "NN_Metric",
    "TimeSeriesLoss",
    "TimeSeriesBaseLoss",
    # Classes
    "MAE",
    "MSE",
    "ND",
    "NRMSE",
    "Q_Quantile",
    "Q_Quantile_Loss",
    "RMSE",
    "TimeSeriesMSE",
    "TimeSeriesWMSE",
    "WMAE",
    "WMSE",
    "WRMSE",
    # Functions
    "nd",
    "nrmse",
    "q_quantile",
    "q_quantile_loss",
    "rmse",
]


from tsdm.metrics import functional, modular, timeseries
from tsdm.metrics._torch_imports import (
    TORCH_ALIASES,
    TORCH_ALIASES_FUNCTIONAL,
    TORCH_LOSSES,
    TORCH_LOSSES_FUNCTIONAL,
    TORCH_SPECIAL_LOSSES,
    TORCH_SPECIAL_LOSSES_FUNCTIONAL,
)
from tsdm.metrics.base import BaseMetric, Metric, NN_Metric, WeightedMetric
from tsdm.metrics.functional import nd, nrmse, q_quantile, q_quantile_loss, rmse
from tsdm.metrics.modular import MAE, MSE, RMSE, WMAE, WMSE, WRMSE
from tsdm.metrics.timeseries import (
    ND,
    NRMSE,
    Q_Quantile,
    Q_Quantile_Loss,
    TimeSeriesBaseLoss,
    TimeSeriesLoss,
    TimeSeriesMSE,
    TimeSeriesWMSE,
)

FUNCTIONAL_LOSSES: dict[str, Metric] = {
    "nd"              : nd,
    "rmse"            : rmse,
    "nrmse"           : nrmse,
    "q_quantile"      : q_quantile,
    "q_quantile_loss" : q_quantile_loss,
}  # fmt: skip
r"""Dictionary of all available functional losses."""

MODULAR_LOSSES: dict[str, type[Metric]] = {
    "MAE"             : MAE,
    "MSE"             : MSE,
    "ND"              : ND,
    "NRMSE"           : NRMSE,
    "Q_Quantile"      : Q_Quantile,
    "Q_Quantile_Loss" : Q_Quantile_Loss,
    "RMSE"            : RMSE,
    "TimeSeriesMSE"   : TimeSeriesMSE,
    "TimeSeriesWMSE"  : TimeSeriesWMSE,
    "WMAE"            : WMAE,
    "WMSE"            : WMSE,
    "WRMSE"           : WRMSE,
}  # fmt: skip
r"""Dictionary of all available modular losses."""


LOSSES: dict[str, Metric | type[Metric]] = {
    **FUNCTIONAL_LOSSES,
    **MODULAR_LOSSES,
}  # fmt: skip
r"""Dictionary of all available losses."""
