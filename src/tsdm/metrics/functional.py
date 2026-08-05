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
def nd(x: Tensor, xhat: Tensor, /, *, eps: float = 2**-24) -> Tensor:
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
    res = torch.sum(torch.abs(xhat - x), dim=(-2, -1))
    mag = torch.sum(torch.abs(x), dim=(-2, -1))
    mag = torch.maximum(mag, torch.tensor(eps, dtype=x.dtype, device=x.device))
    return torch.mean(res / mag)  # get rid of any batch dimensions


@torch.compile(fullgraph=True)
def nrmse(x: Tensor, xhat: Tensor, /, *, eps: float = 2**-24) -> Tensor:
    r"""Compute the normalized deviation score.

    .. math:: 𝖭𝖱𝖬𝖲𝖤(x，x̂) ≔ \frac{\sqrt{\frac{1}{T}∑_{tk}|x̂_{tk} - x_{tk}|^2}}{∑_{tk}|x_{tk}|}

    References:
        - | Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction
          | Hsiang-Fu Yu, Nikhil Rao, Inderjit S. Dhillon
          | Advances in Neural Information Processing Systems 29 (NIPS 2016)
          | https://papers.nips.cc/paper/2016/hash/85422afb467e9456013a2a51d4dff702-Abstract.html
    """
    res = torch.sqrt(torch.sum(torch.abs(xhat - x) ** 2, dim=(-2, -1)))
    mag = torch.sum(torch.abs(x), dim=(-2, -1))
    mag = torch.maximum(mag, torch.tensor(eps, dtype=x.dtype, device=x.device))
    return torch.mean(res / mag)  # get rid of any batch dimensions


@torch.compile(fullgraph=True)
def q_quantile(x: Tensor, xhat: Tensor, /, *, q: float = 0.5) -> Tensor:
    r"""Return the q-quantile.

    .. math:: 𝖯_q(x，x̂) ≔ \begin{cases}\hfill q⋅|x-x̂|:& x≥x̂ \\ (1-q)⋅|x-x̂|:& x≤x̂ \end{cases}

    References:
        - | Deep State Space Models for Time Series Forecasting
          | Syama Sundar Rangapuram, Matthias W. Seeger, Jan Gasthaus, Lorenzo Stella, Yuyang Wang,
            Tim Januschowski
          | Advances in Neural Information Processing Systems 31 (NeurIPS 2018)
          | https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html
    """
    residual = x - xhat
    return torch.max((q - 1) * residual, q * residual)  # simplified formula


@torch.compile(fullgraph=True)
def q_quantile_loss(x: Tensor, xhat: Tensor, /, *, q: float = 0.5) -> Tensor:
    r"""Return the q-quantile loss.

    .. math:: 𝖰𝖫_q(x，x̂) ≔ 2\frac{∑_{tk}𝖯_q(x_{tk}，x̂_{tk})}{∑_{tk}|x_{tk}|}

    References:
        - | Deep State Space Models for Time Series Forecasting
          | Syama Sundar Rangapuram, Matthias W. Seeger, Jan Gasthaus, Lorenzo Stella, Yuyang Wang,
            Tim Januschowski
          | Advances in Neural Information Processing Systems 31 (NeurIPS 2018)
          | https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html
    """
    return 2 * torch.sum(q_quantile(x, xhat, q=q)) / torch.sum(torch.abs(x))


@torch.compile(fullgraph=True)
def rmse(x: Tensor, xhat: Tensor, /) -> Tensor:
    r"""Compute the RMSE.

    .. math:: 𝗋𝗆𝗌𝖾(x，x̂) ≔ \sqrt{𝔼[‖x - x̂‖^2]}
    """
    return torch.sqrt(torch.mean((x - xhat) ** 2))
