r"""Penalties and losses for vector and sequence data.

A **penalty** measures the discrepancy between one prediction $ŷ$ and its
target $y$. It reduces the feature or sequence dimensions to a nonnegative value:

$$ ℓ：𝓨×𝓨 ⟶ ℝ_{≥0} $$

Applied to a batch of $N$ prediction-target pairs, a penalty produces one value
$rₙ=ℓ(ŷₙ,yₙ)$ per sample, retaining the batch dimensions. Write $Ŷ=(ŷₙ)ₙ₌₁ᴺ$
and $Y=(yₙ)ₙ₌₁ᴺ$ for the batch. A **separable loss** combines these penalties
using an outer reduction $Ψ$:

$$ 𝓛(Ŷ,Y) = Ψ\bigl((ℓ(ŷₙ,yₙ))ₙ₌₁ᴺ\bigr) $$

`Reduction` provides three choices:

- `SUM`: Sum the per-sample penalties.
- `MEAN`: Average the per-sample penalties.
- `NONE`: Retain the per-sample penalties without an outer reduction.

Optional nonnegative sample weights $wₙ$ act in this outer reduction. The weighted sum and
mean are

$$\begin{aligned}
    𝓛_{\mathrm{sum}}(Ŷ,Y) &≔ ∑ₙ₌₁ᴺ wₙℓ(ŷₙ,yₙ) \\
    𝓛_{\mathrm{mean}}(Ŷ,Y) &≔ \frac{∑ₙ₌₁ᴺ wₙℓ(ŷₙ,yₙ)}{∑ₙ₌₁ᴺ wₙ}
\end{aligned}$$

The weighted mean requires positive total weight. With `NONE`, sample weights
multiply the individual penalties. Feature, channel, and time weights instead
act within each penalty, before the outer reduction.

The `pointwise` module reduces selected feature axes independently for each
sample. The `sequence` module also accounts for time axes, masks, and sequence
normalization, allowing batches of sequences with different observed lengths.
The custom penalties use explicit boolean masks to exclude missing observations;
NaN padding alone does not identify valid entries.

Losses are available as functions and as PyTorch modules. The `penalties`
subpackage provides the underlying norms and per-sequence errors, while `base`
defines the shared protocols, base classes, and reductions. The registries below
collect the available custom losses and penalties, along with imported PyTorch
losses and their aliases.
"""

__all__ = [
    # Sub-Modules
    "base",
    "penalties",
    "pointwise",
    "sequence",
    # Constants
    "FUNCTIONAL_VECTOR_LOSSES",
    "FUNCTIONAL_SEQUENCE_LOSSES",
    "SEQUENCE_LOSSES",
    "SEQUENCE_PENALTIES",
    "VECTOR_LOSSES",
    "VECTOR_PENALTIES",
    # torch imports
    "TORCH_ALIASES",
    "TORCH_ALIASES_FUNCTIONAL",
    "TORCH_LOSSES",
    "TORCH_LOSSES_FUNCTIONAL",
    "TORCH_SPECIAL_LOSSES",
    "TORCH_SPECIAL_LOSSES_FUNCTIONAL",
]


from . import base, penalties, pointwise, sequence
from ._torch_imports import (
    TORCH_ALIASES,
    TORCH_ALIASES_FUNCTIONAL,
    TORCH_LOSSES,
    TORCH_LOSSES_FUNCTIONAL,
    TORCH_SPECIAL_LOSSES,
    TORCH_SPECIAL_LOSSES_FUNCTIONAL,
)
from .base import *  # ruff: ignore[F403]

__all__ += base.__all__


VECTOR_PENALTIES: dict[str, base.Penalty] = {
    "lp_error" : pointwise.lp_error,
}  # fmt: skip
r"""Dictionary of all available vector penalties."""

FUNCTIONAL_VECTOR_LOSSES: dict[str, base.Loss] = {
    # "lp_loss"         : pointwise.lp_loss,  # needs extra args
    "l2_loss"         : pointwise.l2_loss,
    "mae_loss"        : pointwise.mae_loss,
    "mse_loss"        : pointwise.mse_loss,
    "rmse_loss"       : pointwise.rmse_loss,
}  # fmt: skip
r"""Dictionary of all available functional losses."""


VECTOR_LOSSES: dict[str, type[base.BaseLoss]] = {
    "L2_Loss"   : pointwise.L2_Loss,
    "LP_Loss"   : pointwise.LP_Loss,
    "MAE_Loss"  : pointwise.MAE_Loss,
    "MSE_Loss"  : pointwise.MSE_Loss,
    "RMSE_Loss" : pointwise.RMSE_Loss,
}  # fmt: skip
r"""Dictionary of all available modular losses."""


SEQUENCE_PENALTIES: dict[str, base.SequencePenalty] = {
    "l2_error" : sequence.l2_error,
    "lp_error" : sequence.lp_error,
    "mae"      : sequence.mae,
    "mse"      : sequence.mse,
    "nd"       : sequence.nd,
    "nrmse"    : sequence.nrmse,
    "rmse"     : sequence.rmse,
}  # fmt: skip
r"""Dictionary of all available sequence penalties."""

FUNCTIONAL_SEQUENCE_LOSSES: dict[str, base.SequenceLoss] = {
    # "lp_loss"       : sequence.lp_loss,  # needs extra args
    "l2_loss"       : sequence.l2_loss,
    "mae_loss"      : sequence.mae_loss,
    "mse_loss"      : sequence.mse_loss,
    "nd_loss"       : sequence.nd_loss,
    "nrmse_loss"    : sequence.nrmse_loss,
    "rmse_loss"     : sequence.rmse_loss,
}  # fmt: skip
r"""Dictionary of all available sequence losses."""


SEQUENCE_LOSSES: dict[str, type[base.SequenceLoss]] = {
    "SeparableLoss" : sequence.SeparableLoss,
    "L2_Loss"       : sequence.L2_Loss,
    "LP_Loss"       : sequence.LP_Loss,
    "MAE_Loss"      : sequence.MAE_Loss,
    "MSE_Loss"      : sequence.MSE_Loss,
    "ND_Loss"       : sequence.ND_Loss,
    "NRMSE_Loss"    : sequence.NRMSE_Loss,
    "RMSE_Loss"     : sequence.RMSE_Loss,
}  # fmt: skip
r"""Dictionary of all available time-series losses."""
