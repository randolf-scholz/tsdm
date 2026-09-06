r"""Implementation of loss functions.

Contains losses in both modular and functional form.

Theory
------
We define the following

1. A metric is a function

    .. math:: 𝔪： ⋃_{n∈ℕ}(𝓨×𝓨)ⁿ ⟶ ℝ_{≥0}
        \qq{s.t.} \text{$𝔪(ŷ，y) = 0$ if and only if $yₙ=ŷₙ ∀n=1:N$}

    I.e., a function that takes a finite number of pairs $(ŷₙ，yₙ)_{n=1:N}$
    and returns a non-negative scalar. We denote $Y≔(yₙ)ₙ$ and $Ŷ≔(ŷₙ)ₙ$
    and write $𝔪(ŷ，y)$ for the metric value.

2. A metric is called **decomposable**, if and only if it can be written as a composition
   of an **aggregation function** $Ψ$ and an **intance-wise loss function** $ℓ$:

    .. math:: 𝔪 = Ψ∘(ℓ×𝗂𝖽) \qq{with} ℓ：𝓨×𝓨 ⟶ ℝ_{≥0} \qq{and} Ψ：⋃_{n∈ℕ}ℝⁿ ⟶ ℝ_{≥0}

    I.e. the function $ℓ$ is applied element-wise to all pairs $(y, ŷ)$ and the function $Ψ$
    accumulates the results. Typical choices of $ψ$ are:

    - sum: $Ψ(r) = ∑ₙ rₙ$
    - mean: $Ψ(r) = \frac{1}{N} ∑ₙ rₙ$
    - median: $Ψ(r) = 𝐌ₙ rₙ ≔ \Median((rₙ)_{n=1:N})$

3. A metric is called **instance-wise** if it can be written in the form

    .. math:: 𝔪： ⋃_{n∈ℕ}(𝓨×𝓨)ⁿ ⟶ ℝ_{≥0}, 𝔪(ŷ，y) = ∑ₙ ω(n,N) ℓ(ŷₙ，yₙ)

    with a weight function $ω：ℕ×ℕ ⟶ ℝ_{≥0}$ and an instance-wise loss function $ℓ$.

4. A metric is called a loss-function, if and only if

   - It is differentiable almost everywhere.
   - It is non-constant, at least on some open set.

Note that in the context of time-series, we allow the accumulator to depend on the time variable.
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
    "rmse"            : samplewise.rmse_loss,
    "nrmse"           : sequential.nrmse,
    "q_quantile"      : samplewise.q_quantile,
    "q_quantile_loss" : sequential.q_quantile_loss,
}  # fmt: skip
r"""Dictionary of all available functional losses."""

MODULAR_LOSSES: dict[str, type[base.BaseMetric]] = {
    "LP_Loss"   : samplewise.LP_Loss,
    "MAE_loss"  : samplewise.MAE_loss,
    "MSE_Loss"  : samplewise.MSE_Loss,
    "RMSE_Loss" : samplewise.RMSE_Loss,
}  # fmt: skip
r"""Dictionary of all available modular losses."""

TIMESERIES_LOSSES: dict[str, type[base.SequentialBaseMetric]] = {
    "ND"              : sequential.ND,
    "NRMSE"           : sequential.NRMSE,
    "Q_Quantile_Loss" : sequential.Q_Quantile_Loss,
    "SequentialMSE"   : sequential.SequentialMSE,
}  # fmt: skip
r"""Dictionary of all available time-series losses."""

LOSSES: dict[str, base.Metric | type[base.Metric]] = {
    **FUNCTIONAL_LOSSES,
    **MODULAR_LOSSES,
    **TIMESERIES_LOSSES,
}  # fmt: skip
r"""Dictionary of all available losses."""
