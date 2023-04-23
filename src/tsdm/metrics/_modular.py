r"""Implementations of loss functions.

Notes
-----
Contains losses in modular form.

- See `tsdm.losses.functional` for functional implementations.
"""

__all__ = [
    # Base Classes
    "Loss",
    "BaseLoss",
    "WeightedLoss",
    # Classes
    "WRMSE",
    "RMSE",
    "MSE",
    "WMSE",
    "MAE",
    "WMAE",
    # "TimeSeriesMAE",
    # "TimeSeriesWMAE",
    # "TimeSeriesRMSE",
    # "TimeSeriesWRMSE",
]


from abc import ABCMeta, abstractmethod
from typing import Final, Optional, Protocol, runtime_checkable

import torch
from torch import Tensor, jit, nn

from tsdm.utils.decorators import autojit


@runtime_checkable
class Loss(Protocol):
    r"""Protocol for a loss function."""

    def __call__(self, targets: Tensor, predictions: Tensor, /) -> Tensor:
        r"""Compute the loss."""


class BaseLoss(nn.Module, metaclass=ABCMeta):
    r"""Base class for a loss function."""

    # Constants
    axes: Final[tuple[int, ...]]
    r"""CONST: The axes over which the loss is computed."""
    normalize: Final[bool]
    r"""CONST: Whether to normalize the weights."""
    rank: Final[int]
    r"""CONST: The number of dimensions over which the loss is computed"""

    def __init__(
        self,
        axes: int | tuple[int, ...] = -1,
        /,
        *,
        normalize: bool = False,
    ):
        super().__init__()
        self.normalize = normalize
        self.axes = (axes,) if isinstance(axes, int) else tuple(axes)
        self.rank = len(self.axes)

    @abstractmethod
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r"""Compute the loss."""
        raise NotImplementedError


class WeightedLoss(nn.Module, metaclass=ABCMeta):
    r"""Base class for a weighted loss function."""

    # Parameters
    weight: Tensor
    r"""PARAM: The weight-vector."""

    # Constants
    axes: Final[tuple[int, ...]]
    r"""CONST: The axes over which the loss is computed."""
    learnable: Final[bool]
    r"""CONST: Whether the weights are learnable."""
    normalize: Final[bool]
    r"""CONST: Whether to normalize the weights."""
    rank: Final[int]
    r"""CONST: The number of dimensions of the weight tensor."""
    shape: Final[tuple[int, ...]]
    r"""CONST: The shape of the weight tensor."""

    def __init__(
        self,
        weight: Tensor,
        /,
        *,
        learnable: bool = False,
        normalize: bool = False,
        axes: Optional[tuple[int, ...]] = None,
    ):
        super().__init__()
        self.learnable = learnable
        self.normalize = normalize

        w = torch.as_tensor(weight, dtype=torch.float32)
        assert torch.all(w >= 0) and torch.any(w > 0)
        w = w / torch.sum(w) if self.normalize else w

        self.weight = nn.Parameter(w, requires_grad=self.learnable)
        self.rank = len(self.weight.shape)
        self.shape = tuple(self.weight.shape)
        self.axes = tuple(range(-self.rank, 0)) if axes is None else axes
        assert len(self.axes) == self.rank

    @abstractmethod
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r"""Compute the loss."""
        raise NotImplementedError


@autojit
class MAE(BaseLoss):
    r"""Mean Absolute Error.

    Given two random vectors $x,x̂∈ℝ^K$, the mean absolute error is defined as:

    .. math:: 𝖬𝖠𝖤(x，x̂) ≔ 𝔼[‖x - x̂‖]

    Given $N$ random samples $x_1, …, x_N ∼ x$ and $x̂_1, …, x̂_N ∼ x̂$, it can be estimated as:

    .. math:: 𝖬𝖠𝖤(x，x̂) ∼ \frac{1}{N}∑_{n=1}^N ‖x̂_n - x_n‖
    """

    @jit.export
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r""".. Signature:: ``[(..., 𝐦), (..., 𝐦)] → ...``."""
        r = predictions - targets

        m = ~torch.isnan(targets)
        r = torch.where(m, r, 0.0)
        r = torch.abs(r)
        r = torch.sum(r, dim=self.axes)

        if self.normalize:
            c = torch.sum(m, dim=self.axes)
        else:
            c = torch.tensor(1.0, device=targets.device, dtype=targets.dtype)

        r = torch.where(c > 0, r / c, 0.0)

        # aggregate over batch dimensions
        r = torch.mean(r)
        return r


@autojit
class WMAE(WeightedLoss):
    r"""Weighted Mean Absolute Error.

    Given two random vectors $x,x̂∈ℝ^K$, the weighted mean absolute error is defined as:

    .. math:: 𝗐𝖬𝖠𝖤(x，x̂) ≔ \sqrt{𝔼[‖x - x̂‖_w]}

    Given $N$ random samples $x_1, …, x_N ∼ x$ and $x̂_1, …, x̂_N ∼ x̂$, it can be estimated as:

    .. math:: 𝗐𝖬𝖠𝖤(x，x̂) ≔ \sqrt{\frac{1}{N}∑_{n=1}^N ‖x̂_n - x_n‖_w}
    """

    @jit.export
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r""".. Signature:: ``[(..., 𝐦), (..., 𝐦)] → ...``."""
        r = predictions - targets

        m = ~torch.isnan(targets)
        r = torch.where(m, r, 0.0)
        r = self.weight * torch.abs(r)
        r = torch.sum(r, dim=self.axes)

        if self.normalize:
            c = torch.sum(m * self.weight, dim=self.axes)
        else:
            c = torch.tensor(1.0, device=targets.device, dtype=targets.dtype)

        r = torch.where(c > 0, r / c, 0.0)

        # aggregate over batch dimensions
        r = torch.mean(r)
        return r


@autojit
class MSE(BaseLoss):
    r"""Mean Square Error.

    Given two random vectors $x,x̂∈ℝ^K$, the mean square error is defined as:

    .. math:: 𝖬𝖲𝖤(x，x̂) ≔ 𝔼[‖x̂-x‖^2] ∼ \frac{1}{N}∑_{n=1}^N ‖x̂_n - x_n‖^2

    Given $N$ random samples $x_1, …, x_N ∼ x$ and $x̂_1, …, x̂_N ∼ x̂$, it can be estimated as:

    .. math:: 𝖬𝖲𝖤(x，x̂) ∼ \frac{1}{N}∑_{n=1}^N ‖x̂_n - x_n‖^2

    If the normalize option is set to True, then the normalized ℓ²-norm is used instead:

    .. math:: ‖z‖^2_{2^*} ≔ \frac{1}{K}∑_{k=1}^K z_k^2

    If nan_policy is set to 'omit', then NaN targets are ignored, not counting them as observations.
    In this case, the loss is computed as-if the NaN channels would not exist.

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

    @jit.export
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r""".. Signature:: ``[(..., 𝐦), (..., 𝐦)] → ...``."""
        r = predictions - targets

        m = ~torch.isnan(targets)
        r = torch.where(m, r, 0.0)
        r = r**2
        r = torch.sum(r, dim=self.axes)  # shape: (..., )

        if self.normalize:
            c = torch.sum(m, dim=self.axes)
        else:
            c = torch.tensor(1.0, device=targets.device, dtype=targets.dtype)

        r = torch.where(c > 0, r / c, 0.0)

        # aggregate over batch dimensions
        r = torch.mean(r)
        return r


@autojit
class WMSE(WeightedLoss):
    r"""Weighted Mean Square Error.

    Given two random vectors $x,x̂∈ℝ^K$, the weighted mean square error is defined as:

    .. math:: 𝗐𝖬𝖲𝖤(x，x̂) ≔ 𝔼[‖x - x̂‖_w^2]

    Given $N$ random samples $x_1, …, x_N ∼ x$ and $x̂_1, …, x̂_N ∼ x̂$, it can be estimated as:

    .. math:: 𝗐𝖬𝖲𝖤(x，x̂) ∼ \frac{1}{N}∑_{n=1}^N ‖x̂_n - x_n‖_w^2

    If the normalize option is set to True, then the weighted normalized weighted ℓ²-norm instead:

    .. math:: ‖z‖^2_{w^*} ≔ \frac{1}{∑_k m_k} ∑_{k=1}^K w_k z_k^2

    If nan_policy is set to 'omit', then NaN targets are ignored, not counting them as observations.
    In this case, the loss is computed as-if the NaN channels would not exist.
    Crucially, the existing weights are re-weighted:

    .. math:: ‖z‖^2_{w^*} ≔ \frac{1}{∑_k m_k w_k} ∑_{k=1}^K [m_k \? w_k z_k^2 : 0]

    Since it could happen that all channels are NaN, the loss is set to zero in this case.

    So, in total, there are 4 variants of the weighted MSE loss:

    1. wMSE with normalization and NaNs ignored

       .. math:: \frac{1}{N}∑_{n=1}^N \frac{1}{∑_k m_k w_k} ∑_{k=1}^K [m_k \? w_k(x̂_{nk} - x_{nk})^2 : 0]

    2. wMSE with normalization and NaNs counted

       .. math:: \frac{1}{N}∑_{n=1}^N \frac{1}{∑_k m_k}∑_{k=1}^K w_k(x̂_{nk} - x_{nk})^2

    3. wMSE without normalization and NaNs ignored

       .. math:: \frac{1}{N}∑_{n=1}^N ∑_{k=1}^K [m_k \? w_k(x̂_{nk} - x_{nk})^2 : 0]

    4. wMSE without normalization and NaNs counted

       .. math:: \frac{1}{N}∑_{n=1}^N ∑_{k=1}^K w_k(x̂_{nk} - x_{nk})^2
    """

    @jit.export
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r""".. Signature:: ``[(..., 𝐦), (..., 𝐦)] → ...``."""
        r = predictions - targets

        m = ~torch.isnan(targets)
        r = torch.where(m, r, 0.0)
        r = self.weight * r**2
        r = torch.sum(r, dim=self.axes)

        if self.normalize:
            c = torch.sum(m * self.weight, dim=self.axes)
        else:
            c = torch.tensor(1.0, device=targets.device, dtype=targets.dtype)

        r = torch.where(c > 0, r / c, 0.0)

        # aggregate over batch dimensions
        r = torch.mean(r)
        return r


@autojit
class RMSE(BaseLoss):
    r"""Root Mean Square Error.

    Given two random vectors $x,x̂∈ℝ^K$, the root-mean-square error is defined as:

    .. math:: 𝖱𝖬𝖲𝖤(x，x̂) ≔ \sqrt{𝔼[‖x - x̂‖^2]}

    Given $N$ random samples $x_1, …, x_N ∼ x$ and $x̂_1, …, x̂_N ∼ x̂$, it can be estimated as:

    .. math:: 𝖱𝖬𝖲𝖤(x，x̂) ∼ \sqrt{\frac{1}{N}∑_{n=1}^N ‖x̂_n - x_n‖^2}
    """

    @jit.export
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r""".. Signature:: ``[(..., 𝐦), (..., 𝐦)] → ...``."""
        r = predictions - targets

        m = ~torch.isnan(targets)
        r = torch.where(m, r, 0.0)
        r = r**2
        r = torch.sum(r, dim=self.axes)

        if self.normalize:
            c = torch.sum(m, dim=self.axes)
        else:
            c = torch.tensor(1.0, device=targets.device, dtype=targets.dtype)

        r = torch.where(c > 0, r / c, 0.0)

        # aggregate over batch dimensions
        r = torch.mean(r)
        return torch.sqrt(r)


@autojit
class WRMSE(WeightedLoss):
    r"""Weighted Root Mean Square Error.

    Given two random vectors $x,x̂∈ℝ^K$, the root-mean-square error is defined as:

    .. math:: 𝗐𝖱𝖬𝖲𝖤(x，x̂) ≔ \sqrt{𝔼[‖x - x̂‖_w^2]}

    Given $N$ random samples $x_1, …, x_n ∼ x$ and $x̂_1, …, x̂_n ∼ x̂$, it can be estimated as:

    .. math:: 𝗐𝖱𝖬𝖲𝖤(x，x̂) ∼ \sqrt{\frac{1}{N}∑_{n=1}^N ‖x̂_n - x_n‖_w^2}
    """

    @jit.export
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r""".. Signature:: ``[(..., 𝐦), (..., 𝐦)] → ...``."""
        r = predictions - targets

        m = ~torch.isnan(targets)
        r = torch.where(m, r, 0.0)
        r = self.weight * r**2
        r = torch.sum(r, dim=self.axes)

        if self.normalize:
            c = torch.sum(m * self.weight, dim=self.axes)
        else:
            c = torch.tensor(1.0, device=targets.device, dtype=targets.dtype)

        r = torch.where(c > 0, r / c, 0.0)

        # aggregate over batch dimensions
        r = torch.mean(r)
        return torch.sqrt(r)


class LP(BaseLoss):
    r"""$L^p$ Loss.

    Given two random vectors $x,x̂∈ℝ^K$, the $L^p$-loss is defined as:

    .. math:: 𝖱𝖬𝖲𝖤(x，x̂) ≔ \sqrt[p]{𝔼[‖x - x̂‖^p]}

    Given $N$ random samples $x_1, …, x_N ∼ x$ and $x̂_1, …, x̂_N ∼ x̂$, it can be estimated as:

    .. math:: 𝖱𝖬𝖲𝖤(x，x̂) ∼ \sqrt[p]{\frac{1}{N}∑_{n=1}^N ‖x̂_n - x_n‖^p}

    Special cases:
        - $p=1$: :class:`.MAE`
        - $p=2$: :class:`.RMSE`
        - $p=∞$: :class:`.MXE`
    """

    p: Final[float]
    """The $p$-norm to use."""

    def __init__(
        self,
        p: float = 2.0,
        learnable: bool = False,
        normalize: bool = False,
        axes: Optional[tuple[int, ...]] = None,
    ):
        super().__init__(normalize=normalize, learnable=learnable, axes=axes)
        self.p = p

    @jit.export
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r""".. Signature:: ``[(..., 𝐦), (..., 𝐦)] → ...``."""
        r = predictions - targets

        m = ~torch.isnan(targets)
        r = torch.where(m, r, 0.0)
        r = r**self.p
        r = torch.sum(r, dim=self.axes)

        if self.normalize:
            c = torch.sum(m, dim=self.axes)
        else:
            c = torch.tensor(1.0, device=targets.device, dtype=targets.dtype)

        r = torch.where(c > 0, r / c, 0.0)

        # aggregate over batch dimensions
        r = torch.mean(r)
        return torch.pow(r, 1 / self.p)


class WLP(WeightedLoss):
    r"""Weighted $L^p$ Loss.

    Given two random vectors $x,x̂∈ℝ^K$, the weighted $L^p$-loss is defined as:

    .. math:: 𝖱𝖬𝖲𝖤(x，x̂) ≔ \sqrt[p]{𝔼[‖x - x̂‖_w^p]}

    Given $N$ random samples $x_1, …, x_N ∼ x$ and $x̂_1, …, x̂_N ∼ x̂$, it can be estimated as:

    .. math:: 𝖱𝖬𝖲𝖤(x，x̂) ∼ \sqrt[p]{\frac{1}{N}∑_{n=1}^N ‖x̂_n - x_n‖_w^p}

    Special cases:
        - $p=1$: :class:`.WMAE`
        - $p=2$: :class:`.WRMSE`
        - $p=∞$: :class:`.WMXE`
    """

    p: Final[float]
    """The $p$-norm to use."""

    def __init__(
        self,
        weight: Tensor,
        *,
        p: float = 2.0,
        learnable: bool = False,
        normalize: bool = False,
        axes: Optional[tuple[int, ...]] = None,
    ):
        super().__init__(weight, normalize=normalize, learnable=learnable, axes=axes)
        self.p = p

    @jit.export
    def forward(self, targets: Tensor, predictions: Tensor) -> Tensor:
        r""".. Signature:: ``[(..., 𝐦), (..., 𝐦)] → ...``."""
        r = predictions - targets

        m = ~torch.isnan(targets)
        r = torch.where(m, r, 0.0)
        r = self.weight * r**self.p
        r = torch.sum(r, dim=self.axes)

        if self.normalize:
            c = torch.sum(m * self.weight, dim=self.axes)
        else:
            c = torch.tensor(1.0, device=targets.device, dtype=targets.dtype)

        r = torch.where(c > 0, r / c, 0.0)

        # aggregate over batch dimensions
        r = torch.mean(r)
        return torch.pow(r, 1 / self.p)
