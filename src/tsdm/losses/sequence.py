r"""Loss functions for sequential data.

We have some data sets where the number of samples can vary drastically between channels.
In that case, since we want the model still to be able to predict different channels,
one should normalize the per-channel loss.

a general Lₚ loss could look like:

Nₙₖ = ∑ₜmₙₜₖ
Iₙₖ = [Nₙₖ > 0]
Sₖ = ∑ₙIₙₖ

ℓₚ = ( ∑ₙ (1/K) ∑ₖ ( [Nₙₖ>0]  / )    ∑ₜ mₙₜₖ|xₙₜₖ|ᵖ )^{1/p}

Iₙₖ = Anyₜ(mₙₜₖ) = 1-∏ₜ(1-mₙₜₖ)  # was any value observed in channel k for batch element n
Kₙ = ∑ₖ Iₙₖ  # number of channels with at least one observation for batch element n.
Sₖ = ∑ₙ Iₙₖ  # number of batch elements with at least one obervation in channel k.

Wₙ = ∑ₖIₙₖwₖ  # per sample total channel weight.




Since batch elements can be of different lengths, we need to be careful about the
computation of the loss. We add a mask argument.

Typically, the mask should agree with the arguments provided by `targets` and `predictions`
should provide arguments for these positions.
"""

__all__ = [
    # Classes
    "ND",
    "NRMSE",
    "Quantile_Loss",
    "SequentialMSE",
    # losses
    "lp_loss",
    "nd",
    "nrmse",
    "quantile_loss",
    # functions
]

import torch
from torch import Tensor

from .base import BaseSequenceLoss, Dim
from .penalties import quantile_error
from .penalties.sequence import Normalization, lp_norm
from .reductions import Reduction


def lp_loss(
    *,
    predictions: Tensor,  # Float[..., $N, *D]
    targets: Tensor,  # Float[..., $N, *D]
    time_dim: int,
    channel_dim: int | tuple[int, ...],  # *D
    # optional args
    p: float = 2.0,
    mask: Tensor | None = None,  # Bool[..., $N, *D]
    channel_weight: Tensor | None = None,  # Float[*D]
    time_weight: Tensor | None = None,  # Float[..., $N]
    normalization: Normalization | Tensor | None = Normalization.CHANNEL_PREVALENCE,
    reduction: Reduction = Reduction.MEAN,
    relative: bool = False,
) -> Tensor:  # Float[()]
    r"""Compute the reduced time-normalized $p$-norm of prediction residuals.

    .. math:: ℓₚ(x̂, x) ≔ 𝔼[‖x̂ - x‖ₚ]

    The arguments other than ``predictions``, ``targets``, and ``reduction``
    are forwarded to :func:`lp_norm`. ``reduction`` aggregates over the batch
    dimensions of the resulting norms.

    Args:
        predictions: Predicted time series.
        targets: Target time series.
        p: Order of the norm.
        mask: Boolean mask indicating valid values.
        channel_dim: Tensor axes of the channel dimensions.
        channel_weight: Optional weights applied within each channel norm.
        time_dim: Tensor axis of the time dimension.
        time_weight: Optional weights applied to time steps.
        normalization: Time-series normalization applied within each norm.
        reduction: Aggregation applied to the per-sequence losses.
        relative: If ``True``, divide each residual norm by the corresponding
            target norm, computing $‖x̂ - x‖ₚ / ‖x‖ₚ$.
    """
    norms = lp_norm(
        predictions - targets,
        p=p,
        mask=mask,
        channel_dim=channel_dim,
        channel_weight=channel_weight,
        time_dim=time_dim,
        time_weight=time_weight,
        normalization=normalization,
    )
    if relative:
        norms = norms / lp_norm(
            targets,
            p=p,
            mask=mask,
            channel_dim=channel_dim,
            channel_weight=channel_weight,
            time_dim=time_dim,
            time_weight=time_weight,
            normalization=normalization,
        )
    match reduction:
        case Reduction.SUM:
            return norms.sum()
        case Reduction.MEAN:
            return norms.mean()
        case Reduction.NONE:
            return norms
        case _:
            raise ValueError(f"Invalid reduction: {reduction}")


def nd(
    *,
    predictions: Tensor,  # Float[..., $N, *D]
    targets: Tensor,  # Float[..., $N, *D]
    time_dim: int = -2,
    channel_dim: Dim = -1,
    # optional args
    mask: Tensor | None = None,  # Bool[..., $N, *D]
    eps: float = 2**-24,
) -> Tensor:  # Float[...]
    r"""Compute the normalized deviation score.

    .. math:: 𝖭𝖣(x̂，x) ≔ \frac{∑ₜₖ|x̂ₜₖ - xₜₖ|}{∑ₜₖ|xₜₖ|}

    or, more generally:

    .. math:: 𝖭𝖣(x̂，x) ≔ \frac{‖x̂ - x‖_{1⁎}}{‖x‖_{1⁎}}

    where $‖x‖_{p⁎} ≔ \sqrt[p]{(1/T)∑ₜₖ|xₜₖ|ᵖ}$ is a scaled $p$-norm of the tensor.

    References:
        - | Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction
          | Hsiang-Fu Yu, Nikhil Rao, Inderjit S. Dhillon
          | Advances in Neural Information Processing Systems 29 (NIPS 2016)
          | https://papers.nips.cc/paper/2016/hash/85422afb467e9456013a2a51d4dff702-Abstract.html
        - | N-BEATS: Neural basis expansion analysis for interpretable time series forecasting
          | https://openreview.net/forum?id=r1ecqn4YwB
    """
    x_pred = predictions
    x_true = targets
    res = torch.sum((x_pred - x_true).abs(), dim=(-2, -1))
    mag = torch.sum(x_true.abs(), dim=(-2, -1))
    mag = torch.maximum(mag, torch.full_like(mag, eps))
    return torch.mean(res / mag)  # get rid of any batch dimensions


def nrmse(
    *,
    predictions: Tensor,  # Float[..., $N, *D]
    targets: Tensor,  # Float[..., $N, *D]
    time_dim: int = -2,
    channel_dim: Dim = -1,
    # optional args
    mask: Tensor | None = None,  # Bool[..., $N, *D]
    eps: float = 2**-24,
) -> Tensor:  # Float[...]
    r"""Compute the normalized root mean square error as defined by [TRMF]_.

    .. math:: 𝖭𝖱𝖬𝖲𝖤(x̂，x) ≔ \frac{\sqrt{\frac{1}{T}∑ₜₖ|x̂ₜₖ - xₜₖ|²}}{\frac{1}{T}∑ₜₖ|xₜₖ|}

    or, more generally:

    .. math:: 𝖭𝖱𝖬𝖲𝖤(x̂，x) ≔ \frac{ ‖x̂-x‖_{2⁎} }{ ‖x‖_{1⁎} }

    where $‖x‖_{p⁎} ≔ \sqrt[p]{ (1/T)∑ₜₖ|xₜₖ|ᵖ }$ is a scaled $p$-norm of the tensor.

    References:
        .. [TRMF]
        | Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction
        | Hsiang-Fu Yu, Nikhil Rao, Inderjit S. Dhillon
        | Advances in Neural Information Processing Systems 29 (NIPS 2016)
        | https://papers.nips.cc/paper/2016/hash/85422afb467e9456013a2a51d4dff702-Abstract.html
    """
    x_pred = predictions
    x_true = targets
    res = torch.sqrt(torch.sum(torch.abs(x_pred - x_true) ** 2, dim=(-2, -1)))
    mag = torch.sum(x_true.abs(), dim=(-2, -1))
    mag = torch.maximum(mag, torch.full_like(x_true, eps))

    return torch.mean(res / mag)  # get rid of any batch dimensions


def quantile_loss(
    *,
    predictions: Tensor,  # Float[..., $N, 1]
    targets: Tensor,  # Float[..., $N, 1]
    q: float = 0.5,
    mask: Tensor | None = None,
    relative: bool = True,
) -> Tensor:  # Float[()]
    r"""Return the q-quantile loss.

    Given a collection of of univariate time series $x̂ₜ⁽ⁱ⁾$:

    .. math:: 𝖰𝖫_q(x̂，x) ≔ 2\frac{ ∑ᵢₜ ρ_q(x̂ₜ⁽ⁱ⁾，xₜ⁽ⁱ⁾) }{ ∑ᵢₜ|xₜ⁽ⁱ⁾| }

    In order to prevent accidental application on multivariate time series,
    this method checks for the presence of a trailing singleton dimension.

    If a mask is given, instead compute:

    .. math:: 𝖰𝖫_q(x̂，x) ≔
        2\frac{ ∑ᵢₜ ⟦ mₜ⁽ⁱ⁾ \? ρ_q(x̂ₜ⁽ⁱ⁾，xₜ⁽ⁱ⁾) : 0 ⟧ }{ ∑ᵢₜ⟦ mₜ⁽ⁱ⁾ \? |xₜ⁽ⁱ⁾| : 0 ⟧ }

    Args:
        predictions: Predicted values of shape ``[..., N, 1]``.
        targets: Target values of shape ``[..., N, 1]``.
        q: Quantile level in the closed interval ``[0, 1]``.
        mask: Optional Boolean mask of observations to include. It must be
            broadcastable to ``predictions`` and may include the trailing
            singleton channel dimension.
        relative: Whether to normalize the summed quantile errors by the sum
            of absolute target values.

    References:
        - | Deep State Space Models for Time Series Forecasting
          | Syama Sundar Rangapuram et al.
          | Advances in Neural Information Processing Systems 31 (NeurIPS 2018)
          | https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html
    """
    if targets.shape[-1] != 1 or predictions.shape[-1] != 1:
        raise ValueError("This loss is only defined for univariate time series.")

    predictions = predictions.squeeze(-1)
    targets = targets.squeeze(-1)

    if mask is not None:
        mask = mask.squeeze(-1)
        predictions = torch.where(mask, predictions, 0.0)
        targets = torch.where(mask, targets, 0.0)

    q_err = 2 * quantile_error(predictions - targets, q=q)
    return q_err.sum() / targets.abs().sum() if relative else q_err.sum()


class ND(BaseSequenceLoss):
    r"""Compute the normalized deviation score.

    .. math:: 𝖭𝖣(x̂，x) ≔ \frac{∑ₜₖ|x̂ₜₖ - xₜₖ|}{∑ₜₖ|xₜₖ|}

    TODO: How to distinguish batch univariate vs single multivariate?
    => Batch makes little sense since all could have different length!

    References:
        - | Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction
          | https://papers.nips.cc/paper/2016/hash/85422afb467e9456013a2a51d4dff702-Abstract.html
        - | N-BEATS: Neural basis expansion analysis for interpretable time series forecasting
          | https://openreview.net/forum?id=r1ecqn4YwB
    """

    # Float[..., $N], Float[..., $N] -> Float[()]
    def forward(
        self, *, predictions: Tensor, targets: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        return nd(
            predictions=predictions,
            targets=targets,
            mask=mask,
            time_dim=self.time_dim,
            channel_dim=self.channel_dim,
        )


class NRMSE(BaseSequenceLoss):
    r"""Compute the normalized root mean squared error.

    .. math:: 𝖭𝖱𝖬𝖲𝖤(x̂，x) ≔ \frac{\sqrt{\frac{1}{T}∑ₜₖ|x̂ₜₖ - xₜₖ|²}}{∑ₜₖ|xₜₖ|}

    References:
        - | Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction
          | https://papers.nips.cc/paper/2016/hash/85422afb467e9456013a2a51d4dff702-Abstract.html
    """

    def forward(
        self, *, predictions: Tensor, targets: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        r"""Compute the loss value."""
        return nrmse(
            predictions=predictions,
            targets=targets,
            mask=mask,
            time_dim=self.time_dim,
            channel_dim=self.channel_dim,
        )


class Quantile_Loss(BaseSequenceLoss):
    r"""The q-quantile loss.

    .. math:: 𝖰𝖫_q(x̂，x) ≔ 2\frac{ ∑ₜₖρ_q(x̂ₜₖ，xₜₖ) }{ ∑ₜₖ|xₜₖ| }

    References:
        - | Deep State Space Models for Time Series Forecasting
          | Syama Sundar Rangapuram et al.
          | Advances in Neural Information Processing Systems 31 (NeurIPS 2018)
          | https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html
    """

    def forward(
        self, *, predictions: Tensor, targets: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        r"""Compute the loss value."""
        return quantile_loss(
            predictions=predictions,
            targets=targets,
            mask=mask,
        )


class SequentialMSE(BaseSequenceLoss):
    r"""Time-Series Mean Square Error.

    Given two random sequences $x,x̂∈( ℝ ∪ \{𝙽𝙰\})^{T×K}$, the time-series
    mean square error is defined as:

    .. math:: 𝖳𝖲-𝖬𝖲𝖤(x̂，x) ≔ ∑ₜₖ \frac{[mₜₖ \? (x̂ₜₖ - xₜₖ)² : 0]}{∑ₛ mₛₖ}

    Or, more precisely, to avoid division by zero, we use the following

    .. math:: ∑ₜₖ \Bigr[ ∑ₛmₛₖ > 0 \? \frac{[mₜₖ \? (x̂ₜₖ - xₜₖ)² : 0]}{∑ₛmₛₖ} : 0\Bigl]

    By default, each channel is normalized by the number of observations in that channel.
    Other normalization schemes are possible, e.g. by the number of observations in the
    entire time series, or by the number of observations in each time step:

    With time-normalization:

    .. math:: ∑ₜₖ \frac{[mₜₖ \? |x̂ₜₖ - xₜₖ|² : 0]}{∑ₛ mₛₖ}

    with channel-normalization:

    .. math:: ∑ₜₖ \frac{[mₜₖ \? |x̂ₜₖ - xₜₖ|² : 0]}{∑ⱼ mₜⱼ}

    with both:

    .. math:: ∑ₜₖ \frac{[mₜₖ \? |x̂ₜₖ - xₜₖ|² : 0]}{∑ₛⱼ mₛⱼ}

    Moreover, we can consider adding a discount factor with respect to the time,
    i.e. a simple geometric dsitribution, which amounts to adding a term of the form
    $γ^{∑ₖ ∆tₖ}$ to the denominator, where $γ$ is the discount factor and $∆t_k$
    is the time difference between the $k$-th and $(k+1)$-th time point.

    Possible batch-dimensions are averaged over.
    """

    # Float[..., $N, *D], Float[..., $N, *D] -> Float[()]
    def forward(
        self, *, predictions: Tensor, targets: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        w = self.channel_weights
        m = ~targets.isnan()  # 1 if not nan, 0 if nan
        r = predictions - targets
        r = torch.where(m, r, 0.0)
        # must come after where, else we get NaN gradients!
        r = r**2 if w is None else w * r**2

        # compute normalization constant
        match self.normalize_time, self.normalize_channels:
            case True, True:
                c = torch.sum(
                    m if w is None else w * m, dim=self.combined_dim, keepdim=True
                )
                s = torch.sum(r / c, dim=self.combined_dim, keepdim=True)
                r = torch.where(c > 0, s, 0.0)

            case True, False:
                c = torch.sum(m, dim=self.time_dim, keepdim=True)
                s = torch.sum(r / c, dim=self.time_dim, keepdim=True)
                r = torch.where(c > 0, s, 0.0)
                r = torch.sum(r, dim=self.channel_dim, keepdim=True)

            case False, True:
                c = torch.sum(
                    m if w is None else w * m, dim=self.channel_dim, keepdim=True
                )
                s = torch.sum(r / c, dim=self.channel_dim, keepdim=True)
                r = torch.where(c > 0, s, 0.0)
                r = torch.sum(r, dim=self.time_dim, keepdim=True)

            case False, False:
                r = torch.sum(r, dim=self.combined_dim, keepdim=True)

            case _:
                raise RuntimeError("unreachable")

        # aggregate over batch-dimensions
        r = torch.mean(r)
        return r
