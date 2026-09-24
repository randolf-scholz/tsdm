__all__ = ["nd", "nrmse", "ND", "NRMSE"]

import torch
from torch import Tensor

from .base import BaseSequenceLoss, Dim
from .sequence import _normalize_dims


def nd(
    *,
    predictions: Tensor,  # Float[..., $N, *D]
    targets: Tensor,  # Float[..., $N, *D]
    time_dim: int = -2,
    channel_dim: int | tuple[int, ...] = -1,
    # optional args
    mask: Tensor | None = None,  # Bool[..., $N, *D]
) -> Tensor:  # Float[()]
    r"""Compute the normalized deviation score as defined by [TRMF]_.

    For a collection of dense, univariate time series $x̂ₜ⁽ⁱ⁾$, [TRMF]_ defines:

    .. math:: 𝖭𝖣(x̂，x) ≔ \frac{∑ᵢₜ|x̂ₜ⁽ⁱ⁾ - xₜ⁽ⁱ⁾|}{∑ᵢₜ|xₜ⁽ⁱ⁾|}

    which we generalize towards sparse, multivariate time series as:

    .. math::  𝖭𝖣(x̂，x) ≔
        \frac{∑ᵢₜₖ ⟦ mᵢₜₖ \? |x̂ₜₖ⁽ⁱ⁾ - xₜₖ⁽ⁱ⁾| : 0 ⟧ }{∑ᵢₜₖ ⟦mᵢₜₖ \? |xₜₖ⁽ⁱ⁾| : 0 ⟧}

    which is equivalent to `lp_loss` with `p=1`, `relative=True` and `normalization=observation_count`.

    .. math:: 𝖭𝖣(x̂，x) ≔ \frac{‖x̂ - x‖_{1⁎}}{‖x‖_{1⁎}}

    where $‖x‖_{p⁎} ≔ \sqrt[p]{(1/T)∑ₜₖ|xₜₖ|ᵖ}$ is a scaled $p$-norm of the tensor.

    References:
        .. [TRMF]
        | Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction
        | Hsiang-Fu Yu, Nikhil Rao, Inderjit S. Dhillon
        | Advances in Neural Information Processing Systems 29 (NIPS 2016)
        | https://papers.nips.cc/paper/2016/hash/85422afb467e9456013a2a51d4dff702-Abstract.html
        .. [NBEATS]
        | N-BEATS: Neural basis expansion analysis for interpretable time series forecasting
        | https://openreview.net/forum?id=r1ecqn4YwB
    """
    time_dim, channel_dims = _normalize_dims(
        ndim=targets.ndim,
        time_dim=time_dim,
        channel_dim=channel_dim,
    )

    residuals = predictions - targets

    if mask is not None:
        assert mask.shape == residuals.shape
        residuals = torch.where(mask, residuals, 0.0)
        targets = torch.where(mask, targets, 0.0)

    return residuals.abs().sum().div(targets.abs().sum())


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

    For a collection of univariate time series $x̂ₜ⁽ⁱ⁾$, [TRMF]_ defines:

    .. math:: 𝖭𝖱𝖬𝖲𝖤(x̂，x) ≔ \frac{\sqrt{\frac{1}{T}∑ᵢₜ|x̂ₜ⁽ⁱ⁾ - xₜ⁽ⁱ⁾|²}}{\frac{1}{T}∑ᵢₜ|xₜ⁽ⁱ⁾|}

    which we generalize towards sparse, multivariate time series as:

    which is equivalent to `lp_loss` with `p=2` and `relative=1`.

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
