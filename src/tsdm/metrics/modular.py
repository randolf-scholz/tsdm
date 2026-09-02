r"""Implementations of loss functions.

Note:
    Contains losses in modular form.
    See `tsdm.metrics.functional` for functional implementations.
"""

__all__ = ["RMSE", "MSE", "MAE", "LP"]

from typing import Final

import torch
from torch import Tensor

from tsdm.types.aliases import Axis

from .base import BaseMetric


class MAE(BaseMetric):
    r"""Mean Absolute Error.

    Given two random vectors $x,x̂∈ℝ^K$, the mean absolute error is defined as:

    .. math:: 𝖬𝖠𝖤(x，x̂) ≔ 𝔼[‖x - x̂‖]

    Given $N$ random samples $x₁, …, x_N ∼ x$ and $x̂₁, …, x̂_N ∼ x̂$, it can be estimated as:

    .. math:: 𝖬𝖠𝖤(x，x̂) ∼ \frac{1}{N}∑_{n=1}^N ‖x̂ₙ - xₙ‖

    If weights are provided, then the norm $‖z‖² ≔ ∑ₖ wₖ |z_k|²$ is used.
    """

    @torch.compile(fullgraph=True)
    def forward(self, *, predictions: Tensor, targets: Tensor) -> Tensor:
        r""".. signature:: ``[(..., 𝐦), (..., 𝐦)] → ...``."""
        w = self.weight
        m = ~targets.isnan()
        r = predictions - targets
        r = torch.where(m, r, 0.0)
        r = r.abs() if w is None else w * r.abs()
        r = torch.sum(r, dim=self.axis)

        if self.normalize:
            c = torch.sum(m if w is None else w * m, dim=self.axis)
        else:
            c = torch.tensor(1.0, device=targets.device, dtype=targets.dtype)

        r = torch.where(c > 0, r / c, 0.0)

        # aggregate over batch dimensions
        r = torch.mean(r)
        return r


class MSE(BaseMetric):
    r"""Mean Square Error.

    Given two random vectors $x,x̂∈ℝ^K$, the mean square error is defined as:

    .. math:: 𝖬𝖲𝖤(x，x̂) ≔ 𝔼[‖x̂-x‖^2] ∼ \frac{1}{N}∑_{n=1}^N ‖x̂_n - x_n‖^2

    Given $N$ random samples $x_1, …, x_N ∼ x$ and $x̂_1, …, x̂_N ∼ x̂$, it can be estimated as:

    .. math:: 𝖬𝖲𝖤(x，x̂) ∼ \frac{1}{N}∑_{n=1}^N ‖x̂_n - x_n‖^2

    If the `normalize` option is set to True, then the normalized ℓ²-norm is used instead:

    .. math:: ‖z‖^2_{2^*} ≔ \frac{1}{K}∑_{k=1}^K z_k^2

    If nan_policy is set to 'omit', then NaN targets are ignored, not counting them as observations.
    In this case, the loss is computed as if the NaN channels would not exist.

    .. math:: ‖z‖^2_{2^*} ≔ \frac{1}{∑_k m_k} ∑_{k=1}^K [m_k \? z_k^2 : 0]

    Since it could happen that all channels are NaN, the loss is set to zero in this case.

    So, in total, there are 4 variants of the MSE loss:

    Note that this is equivalent to a weighted MSE loss with weights equal to 1.0.

    1. MSE with normalization and NaNs ignored

       .. math:: \frac{1}{N}∑_{n=1}^N \frac{1}{∑_k m_k}∑_{k=1}^K [m_k \? (x̂_{n,k} - x_{n,k})^2 : 0]

    2. MSE with normalization and NaNs counted

       .. math:: \frac{1}{N}∑_{n=1}^N \frac{1}{K}∑_{k=1}^K (x̂_{n,k} - x_{n,k})^2

    3. MSE without normalization and NaNs ignored

       .. math:: \frac{1}{N}∑_{n=1}^N ∑_{k=1}^K [m_i \? (x̂_{n,k} - x_{n,k})^2 : 0]

    4. MSE without normalization and NaNs counted

       .. math:: \frac{1}{N}∑_{n=1}^N ∑_{k=1}^K (x̂_{n,k} - x_{n,k})^2
    """

    @torch.compile(fullgraph=True)
    def forward(self, *, predictions: Tensor, targets: Tensor) -> Tensor:
        r""".. signature:: ``[(..., 𝐦), (..., 𝐦)] → ...``."""
        w = self.weight
        m = ~targets.isnan()
        r = predictions - targets
        r = torch.where(m, r, 0.0)
        r = r**2 if w is None else w * r**2
        r = torch.sum(r, dim=self.axis)  # shape=(..., )

        if self.normalize:
            c = torch.sum(m if w is None else w * m, dim=self.axis)
        else:
            c = torch.tensor(1.0, device=targets.device, dtype=targets.dtype)

        r = torch.where(c > 0, r / c, 0.0)

        # aggregate over batch dimensions
        r = torch.mean(r)
        return r


class RMSE(BaseMetric):
    r"""Root Mean Square Error.

    Given two random vectors $x,x̂∈ℝ^K$, the root-mean-square error is defined as:

    .. math:: 𝖱𝖬𝖲𝖤(x，x̂) ≔ \sqrt{𝔼[‖x - x̂‖^2]}

    Given $N$ random samples $x_1, …, x_N ∼ x$ and $x̂_1, …, x̂_N ∼ x̂$, it can be estimated as:

    .. math:: 𝖱𝖬𝖲𝖤(x，x̂) ∼ \sqrt{\frac{1}{N}∑_{n=1}^N ‖x̂_n - x_n‖^2}
    """

    @torch.compile(fullgraph=True)
    def forward(self, *, predictions: Tensor, targets: Tensor) -> Tensor:
        r""".. signature:: ``[(..., 𝐦), (..., 𝐦)] → ...``."""
        w = self.weight
        m = ~targets.isnan()
        r = predictions - targets
        r = torch.where(m, r, 0.0)
        r = r**2 if w is None else w * r**2
        r = torch.sum(r, dim=self.axis)

        if self.normalize:
            c = torch.sum(m if w is None else w * m, dim=self.axis)
        else:
            c = torch.tensor(1.0, device=targets.device, dtype=targets.dtype)

        r = torch.where(c > 0, r / c, 0.0)

        # aggregate over batch dimensions
        r = torch.mean(r)
        return torch.sqrt(r)


class LP(BaseMetric):
    r"""$Lᵖ$ Loss.

    Given two random vectors $x,x̂∈ℝᴷ$, the $Lᵖ$-loss is defined as:

    .. math:: 𝖱𝖬𝖲𝖤(x，x̂) ≔ \sqrt[p]{𝔼[‖x - x̂‖ᵖ]}

    Given $N$ random samples $x_1, …, x_N ∼ x$ and $x̂_1, …, x̂_N ∼ x̂$, it can be estimated as:

    .. math:: 𝖱𝖬𝖲𝖤(x，x̂) ∼ \sqrt[p]{\frac{1}{N}∑_{n=1}^N ‖x̂ₙ - xₙ‖ᵖ}

    Special cases:
        - $p=1$: :class:`MAE`
        - $p=2$: :class:`RMSE`
    """

    p: Final[float]
    r"""The $p$-norm to use."""

    def __init__(
        self,
        p: float = 2.0,
        *,
        weight: Tensor | None = None,
        normalize: bool = False,
        axis: Axis = None,
        learnable: bool = False,
    ) -> None:
        super().__init__(
            normalize=normalize,
            axis=axis,
            weight=weight,
            learnable=learnable,
        )
        self.p = p

    @torch.compile(fullgraph=True)
    def forward(self, *, predictions: Tensor, targets: Tensor) -> Tensor:
        r""".. signature:: ``[(..., 𝐦), (..., 𝐦)] → ...``."""
        w = self.weight
        p = self.p
        m = ~targets.isnan()
        r = predictions - targets
        r = torch.where(m, r, 0.0)
        r = r**p if w is None else w * r**p
        r = torch.sum(r, dim=self.axis)

        if self.normalize:
            c = torch.sum(m if w is None else w * m, dim=self.axis)
        else:
            c = torch.tensor(1.0, device=targets.device, dtype=targets.dtype)

        r = torch.where(c > 0, r / c, 0.0)

        # aggregate over batch dimensions
        r = torch.mean(r)
        return torch.pow(r, 1 / p)
