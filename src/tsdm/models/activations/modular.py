r"""Implementations of activation functions as `nn.Module`."""

__all__ = [
    # Classes
    "HardBend",
]


import torch
from torch import Tensor, nn


class HardBend(nn.Module):
    r"""Optimized implementation of 2-parameter HardBend that precomputes the threshold and slope.

    .. math:: ϕ(x, a, t) = odesolve(u'(t) = a * tanh(u), t, u(0) = x) = sinh⁻¹(eᵃᵗ sinh(ax))

    The hard bend activation function is defined as:

    .. math:: ϕ(x, a, t) =
        \begin{cases}
            x + t    &  x  >  \frac{t}{e^{at} - 1} \\
            e^{at} x & |x| ≤  \frac{t}{e^{at} - 1} \\
            x - t    &  x  < -\frac{t}{e^{at} - 1}
        \end{cases}
    """

    a: Tensor
    t: Tensor
    threshold: Tensor
    slope: Tensor

    def __init__(self, a: float = 1.0, t: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("a", torch.tensor(a))
        self.register_buffer("t", torch.tensor(t))
        self.register_buffer("threshold", self.t / (torch.exp(self.a * self.t) - 1))
        self.register_buffer("slope", torch.exp(self.a * self.t))

    def forward(self, x: Tensor) -> Tensor:
        mask = x.abs() <= self.threshold
        return torch.where(mask, self.slope * x, x + torch.sign(x) * self.t)
