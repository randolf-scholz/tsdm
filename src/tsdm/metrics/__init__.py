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
    "samplewise",
    "sequential",
    # Constants
    "LOSSES",
    "FUNCTIONAL_LOSSES",
    "MODULAR_LOSSES",
    "TIMESERIES_LOSSES",
    # torch imports
    "TORCH_ALIASES",
    "TORCH_ALIASES_FUNCTIONAL",
    "TORCH_LOSSES",
    "TORCH_LOSSES_FUNCTIONAL",
    "TORCH_SPECIAL_LOSSES",
    "TORCH_SPECIAL_LOSSES_FUNCTIONAL",
]


from . import base, samplewise, sequential
from ._torch_imports import (
    TORCH_ALIASES,
    TORCH_ALIASES_FUNCTIONAL,
    TORCH_LOSSES,
    TORCH_LOSSES_FUNCTIONAL,
    TORCH_SPECIAL_LOSSES,
    TORCH_SPECIAL_LOSSES_FUNCTIONAL,
)
from .base import *  # ruff: ignore[F403]
from .samplewise import *  # ruff: ignore[F403]
from .sequential import *  # ruff: ignore[F403]

__all__ += base.__all__
__all__ += samplewise.__all__
__all__ += sequential.__all__

FUNCTIONAL_LOSSES: dict[str, base.Metric] = {
    "nd"              : sequential.nd,
    "rmse"            : samplewise.rmse,
    "nrmse"           : sequential.nrmse,
    "q_quantile"      : sequential.q_quantile,
    "q_quantile_loss" : sequential.q_quantile_loss,
}  # fmt: skip
r"""Dictionary of all available functional losses."""

MODULAR_LOSSES: dict[str, type[base.BaseMetric]] = {
    "LP"              : samplewise.LP,
    "MAE"             : samplewise.MAE,
    "MSE"             : samplewise.MSE,
    "RMSE"            : samplewise.RMSE,
    # timeseries
    "ND"              : sequential.ND,
    "NRMSE"           : sequential.NRMSE,
    "Q_Quantile"      : sequential.Q_Quantile,
    "Q_Quantile_Loss" : sequential.Q_Quantile_Loss,
    "TimeSeriesMSE"   : sequential.TimeSeriesMSE,
}  # fmt: skip
r"""Dictionary of all available modular losses."""

TIMESERIES_LOSSES: dict[str, type[base.SequentialBaseMetric]] = {
    "ND"              : sequential.ND,
    "NRMSE"           : sequential.NRMSE,
    "Q_Quantile"      : sequential.Q_Quantile,
    "Q_Quantile_Loss" : sequential.Q_Quantile_Loss,
    "TimeSeriesMSE"   : sequential.TimeSeriesMSE,
}  # fmt: skip
r"""Dictionary of all available time-series losses."""

LOSSES: dict[str, base.Metric | type[base.Metric]] = {
    **FUNCTIONAL_LOSSES,
    **MODULAR_LOSSES,
    **TIMESERIES_LOSSES,
}  # fmt: skip
r"""Dictionary of all available losses."""
