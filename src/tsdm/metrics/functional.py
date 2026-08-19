r"""Implementations of loss functions.

Notes:
    Contains losses in functional form.
    See `tsdm.metrics.modular` for modular implementations.
"""

__all__ = [
    # Functions
    "nd",
    "nrmse",
    "rmse",
    "q_quantile",
    "q_quantile_loss",
]


import torch
from torch import Tensor


@torch.compile(fullgraph=True)
def nd(predictions: Tensor, targets: Tensor, *, eps: float = 2**-24) -> Tensor:
    r"""Compute the normalized deviation score.

    .. math:: 𝖭𝖣(x，x̂) ≔ \frac{∑_{tk} |x̂_{tk} - x_{tk}|}{∑_{tk} |x_{tk}|}

    TODO: How to distinguish batch univariate vs single multivariate?
    => Batch makes little sense since all could have different length!

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
    mag = torch.maximum(mag, torch.full_like(x_true, eps))
    return torch.mean(res / mag)  # get rid of any batch dimensions


@torch.compile(fullgraph=True)
def nrmse(predictions: Tensor, targets: Tensor, *, eps: float = 2**-24) -> Tensor:
    r"""Compute the normalized deviation score.

    .. math:: 𝖭𝖱𝖬𝖲𝖤(x，x̂) ≔ \frac{\sqrt{\frac{1}{T}∑_{tk}|x̂_{tk} - x_{tk}|^2}}{∑_{tk}|x_{tk}|}

    References:
        - | Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction
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


@torch.compile(fullgraph=True)
def q_quantile(predictions: Tensor, targets: Tensor, *, q: float = 0.5) -> Tensor:
    r"""Return the q-quantile.

    .. math:: 𝖯_q(x，x̂) ≔ \begin{cases}\hfill q⋅|x-x̂|:& x≥x̂ \\ (1-q)⋅|x-x̂|:& x≤x̂ \end{cases}

    References:
        - | Deep State Space Models for Time Series Forecasting
          | Syama Sundar Rangapuram, Matthias W. Seeger, Jan Gasthaus, Lorenzo Stella, Yuyang Wang,
            Tim Januschowski
          | Advances in Neural Information Processing Systems 31 (NeurIPS 2018)
          | https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html
    """
    residual = targets - predictions
    return torch.max((q - 1) * residual, q * residual)  # simplified formula


@torch.compile(fullgraph=True)
def q_quantile_loss(predictions: Tensor, targets: Tensor, *, q: float = 0.5) -> Tensor:
    r"""Return the q-quantile loss.

    .. math:: 𝖰𝖫_q(x，x̂) ≔ 2\frac{∑_{tk}𝖯_q(x_{tk}，x̂_{tk})}{∑_{tk}|x_{tk}|}

    References:
        - | Deep State Space Models for Time Series Forecasting
          | Syama Sundar Rangapuram, Matthias W. Seeger, Jan Gasthaus, Lorenzo Stella, Yuyang Wang,
            Tim Januschowski
          | Advances in Neural Information Processing Systems 31 (NeurIPS 2018)
          | https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html
    """
    return (
        2 * torch.sum(q_quantile(predictions, targets, q=q)) / torch.sum(targets.abs())
    )


@torch.compile(fullgraph=True)
def rmse(predictions: Tensor, targets: Tensor) -> Tensor:
    r"""Compute the RMSE.

    .. math:: 𝗋𝗆𝗌𝖾(x，x̂) ≔ \sqrt{𝔼[‖x - x̂‖^2]}
    """
    return torch.sqrt(torch.mean((predictions - targets) ** 2))
